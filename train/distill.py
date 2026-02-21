import os
import argparse
import io
import random
import numpy as np
import torch
import librosa
import matplotlib.pyplot as plt
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_dataset, Audio
from transformers import (
    GenerationConfig, 
    Trainer, 
    TrainingArguments, 
    AutoProcessor,
    TrainerCallback,
    EarlyStoppingCallback,
    set_seed
)
from accelerate import init_empty_weights

# Assurez-vous que vos imports locaux fonctionnent
from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
from reduce_model import create_smaller_architecture


def patch_outer_forward(model):
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return

    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: model has no `.thinker.forward`.")

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        input_features=None,
        feature_attention_mask=None,
        labels=None,
        **kwargs,
    ):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True

def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]

# --------------------------------------------------------------------------------
# FONCTION DE BRUITAGE DYNAMIQUE
# --------------------------------------------------------------------------------
def mix_audio_with_noise(signal, noise, snr_db):
    """Ajoute un bruit à un signal avec un SNR donné."""
    # Ajuster la taille du bruit pour correspondre au signal
    if len(noise) < len(signal):
        noise = np.tile(noise, int(np.ceil(len(signal) / len(noise))))
    noise = noise[:len(signal)]
    
    p_signal = np.mean(signal**2)
    p_noise = np.mean(noise**2)
    
    if p_noise == 0 or p_signal == 0: 
        return signal
        
    snr_linear = 10 ** (snr_db / 10)
    noise_factor = np.sqrt(p_signal / (p_noise * snr_linear))
    
    mixed = signal + noise_factor * noise
    return mixed.astype(np.float32)

# --------------------------------------------------------------------------------
# DATASET PYTORCH CUSTOM (Gère le bruit et Librosa)
# --------------------------------------------------------------------------------
class ASRDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, processor, noise_dataset=None, mode="train"):
        self.hf_dataset = hf_dataset
        self.processor = processor
        self.noise_dataset = noise_dataset
        self.mode = mode # "train", "val_clean", "val_noisy"

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, i):
        item = self.hf_dataset[i]
        audio_bytes = item["audio"]["bytes"]
        transcript = item["transcript"]

        # Chargement via librosa (decode=False du dataset HF)
        wav, _ = librosa.load(io.BytesIO(audio_bytes), sr=16000)

        # Ajout du bruit
        # - Validation bruitée : 100% de chance d'avoir du bruit
        # - Entraînement : 50% de chance d'avoir du bruit
        apply_noise = (self.mode == "val_noisy") or (self.mode == "train" and random.random() < 0.5)

        if apply_noise and self.noise_dataset is not None:
            noise_idx = random.randint(0, len(self.noise_dataset) - 1)
            noise_bytes = self.noise_dataset[noise_idx]["audio"]["bytes"]
            noise_wav, _ = librosa.load(io.BytesIO(noise_bytes), sr=16000)
            snr = random.uniform(0, 20)
            wav = mix_audio_with_noise(wav, noise_wav, snr)

        # Préparation du prompt texte
        prompt = ""
        prefix_msgs = build_prefix_messages(prompt, None)
        prefix_text = self.processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]

        return {
            "wav": wav,
            "target": transcript.upper(),
            "prefix_text": prefix_text,
        }

# --------------------------------------------------------------------------------
# DATA COLLATOR
# --------------------------------------------------------------------------------
@dataclass
class DataCollatorForSmallModel:
    processor: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audios = [f["wav"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]

        full_inputs = self.processor(
            text=full_texts, audio=audios, return_tensors="pt", padding=True, truncation=False, sampling_rate=16000
        )
        prefix_inputs = self.processor(
            text=prefix_texts, audio=audios, return_tensors="pt", padding=True, truncation=False, sampling_rate=16000
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :int(pl)] = -100

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        full_inputs["labels"] = labels
        return full_inputs

class CastFloatInputsTrainer(Trainer):
    def _prepare_inputs(self, inputs):
        inputs = super()._prepare_inputs(inputs)
        model_dtype = getattr(self.model, "dtype", None)
        if model_dtype is not None:
            for k, v in list(inputs.items()):
                if torch.is_tensor(v) and v.is_floating_point():
                    inputs[k] = v.to(dtype=model_dtype)
        return inputs

# --------------------------------------------------------------------------------
# CALLBACKS CUSTOM (Sauvegarde & Graphique)
# --------------------------------------------------------------------------------
class SaveEvery10EpochsCallback(TrainerCallback):
    """Sauvegarde le modèle complet toutes les 10 époques dans un dossier distinct."""
    def on_epoch_end(self, args, state, control, **kwargs):
        epoch = round(state.epoch)
        if epoch > 0 and epoch % 10 == 0:
            output_dir = os.path.join(args.output_dir, f"model_epoch_{epoch}")
            kwargs['model'].save_pretrained(output_dir)
            kwargs['tokenizer'].save_pretrained(output_dir)
            print(f"\n[Info] Modèle sauvegardé à l'époque {epoch} dans {output_dir}")

class PlotLossesCallback(TrainerCallback):
    """Génère le graphique Matplotlib à la fin de l'entraînement."""
    def on_train_end(self, args, state, control, **kwargs):
        history = state.log_history
        
        train_loss = [(x['epoch'], x['loss']) for x in history if 'loss' in x]
        val_clean_loss = [(x['epoch'], x['eval_clean_loss']) for x in history if 'eval_clean_loss' in x]
        val_noisy_loss = [(x['epoch'], x['eval_noisy_loss']) for x in history if 'eval_noisy_loss' in x]
        
        plt.figure(figsize=(10, 6))
        if train_loss: plt.plot(*zip(*train_loss), label='Train Loss', color='blue')
        if val_clean_loss: plt.plot(*zip(*val_clean_loss), label='Val Clean Loss', color='green')
        if val_noisy_loss: plt.plot(*zip(*val_noisy_loss), label='Val Noisy Loss', color='red')
            
        plt.xlabel('Époques')
        plt.ylabel('Loss')
        plt.title('Progression de l\'entraînement')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_plot.png')
        plt.close()
        print("\n[Info] Graphique sauvegardé sous 'training_plot.png'")


# --------------------------------------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------------------------------------
def main():
    # 4. Fixer toutes les seeds pour reproductibilité
    set_seed(42)
    random.seed(42)
    np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output_dir", type=str, default="./qwen3-asr-distilled")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=80) # Nombre d'époques maximum : 80
    args = parser.parse_args()

    print("Loading teacher processor and configuration...")
    processor = AutoProcessor.from_pretrained(args.teacher_model, fix_mistral_regex=True)
    small_config = create_smaller_architecture(args.teacher_model)

    print("Creating small model architecture (from scratch)...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    model = Qwen3ASRForConditionalGeneration(small_config)
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    # 1. & 2. Chargement et Split des datasets
    print("Loading datasets...")
    clean_ds = load_dataset("AdoCleanCode/libri_clean_train.100", split="train")
    noise_ds = load_dataset("nguyenvulebinh/wham", split="train")

    # Désactiver le décodage auto pour utiliser Librosa dans le Dataset custom
    clean_ds = clean_ds.cast_column("audio", Audio(decode=False))
    noise_ds = noise_ds.cast_column("audio", Audio(decode=False))

    print("Splitting datasets by speaker (90/5/5)...")
    spk_to_idx = defaultdict(list)
    for i, spk in enumerate(clean_ds["speaker_id"]):
        spk_to_idx[spk].append(i)

    train_idxs, val_clean_idxs, val_noisy_idxs = [], [], []
    for spk, idxs in spk_to_idx.items():
        np.random.shuffle(idxs)
        n = len(idxs)
        n_val = int(np.round(n * 0.05))
        if n_val == 0 and n > 1: n_val = 1 # Force min 1 occurence en validation si possible
        
        val_clean_idxs.extend(idxs[:n_val])
        val_noisy_idxs.extend(idxs[n_val:2*n_val])
        train_idxs.extend(idxs[2*n_val:])

    # 3. Création des datasets PyTorch customisés
    train_dataset = ASRDataset(clean_ds.select(train_idxs), processor, noise_dataset=noise_ds, mode="train")
    val_clean_dataset = ASRDataset(clean_ds.select(val_clean_idxs), processor, noise_dataset=None, mode="val_clean")
    val_noisy_dataset = ASRDataset(clean_ds.select(val_noisy_idxs), processor, noise_dataset=noise_ds, mode="val_noisy")

    collator = DataCollatorForSmallModel(processor=processor)

    # Paramètres d'entraînement
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        eval_strategy="steps",
        save_strategy="steps", 
        eval_steps=500,
        save_steps=500,
        save_total_limit=3,     # Limite de poids standards pour économiser le disque
        bf16=use_bf16,
        fp16=not use_bf16,
        remove_unused_columns=False, # Important pour nos dictionnaires custom
        report_to="none",
        metric_for_best_model="eval_noisy_loss", # 5. Early stopping sur bruit
        greater_is_better=False,
        load_best_model_at_end=True,
        gradient_checkpointing=True,
        dataloader_prefetch_factor=2,
        dataloader_num_workers=2,
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset={"clean": val_clean_dataset, "noisy": val_noisy_dataset},
        data_collator=collator,
        tokenizer=processor.tokenizer,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=4),
            PlotLossesCallback()
        ]
    )

    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()