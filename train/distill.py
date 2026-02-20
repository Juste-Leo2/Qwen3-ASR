import os
import argparse
import io
import torch
import librosa
from dataclasses import dataclass
from typing import Any, Dict, List
from datasets import load_dataset, Audio

from qwen_asr import Qwen3ASRModel
from qwen_asr.core.transformers_backend import Qwen3ASRForConditionalGeneration
from transformers import GenerationConfig, Trainer, TrainingArguments, AutoProcessor
from accelerate import init_empty_weights

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

def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]

def make_preprocess_fn(processor):
    def _preprocess(ex: Dict[str, Any]) -> Dict[str, Any]:
        prompt = ""
        dummy_audio = None
        prefix_msgs = build_prefix_messages(prompt, dummy_audio)
        prefix_text = processor.apply_chat_template(
            [prefix_msgs], add_generation_prompt=True, tokenize=False
        )[0]
        return {
            "prompt": prompt,
            "audio_bytes": ex["audio"]["bytes"], # raw bytes from dataset
            "target": ex["text"].upper(),
            "prefix_text": prefix_text,
        }
    return _preprocess

@dataclass
class DataCollatorForSmallModel:
    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        audio_bytes_list = [f["audio_bytes"] for f in features]
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]

        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [pfx + tgt + eos for pfx, tgt in zip(prefix_texts, targets)]
        
        # Load audio on the fly with librosa
        audios = []
        for ab in audio_bytes_list:
            wav, _ = librosa.load(io.BytesIO(ab), sr=self.sampling_rate)
            audios.append(wav)

        full_inputs = self.processor(
            text=full_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        prefix_inputs = self.processor(
            text=prefix_texts,
            audio=audios,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )

        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="Qwen/Qwen3-ASR-0.6B")
    parser.add_argument("--output_dir", type=str, default="./qwen3-asr-distilled")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_acc", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4) # learning rate slightly higher for training from scratch
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()

    print("Loading teacher processor and configuration...")
    processor = AutoProcessor.from_pretrained(args.teacher_model, fix_mistral_regex=True)
    small_config = create_smaller_architecture(args.teacher_model)

    print("Creating small model architecture (from scratch)...")
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16

    # Instantiate model from scratch
    model = Qwen3ASRForConditionalGeneration(small_config)
    patch_outer_forward(model)
    
    # We are not copying weights right now due to deep dimension changes, so this is a 
    # train from scratch initialization of the student architecture
    
    model.generation_config = GenerationConfig.from_model_config(model.config)

    print("Loading dummy dataset...")
    # Load dataset, disable automatic audio decoding to keep bytes
    raw_ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    raw_ds = raw_ds.cast_column("audio", Audio(decode=False))

    ds = raw_ds.map(make_preprocess_fn(processor), num_proc=1, remove_columns=raw_ds.column_names)

    collator = DataCollatorForSmallModel(processor=processor, sampling_rate=16000)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=1,
        save_strategy="epoch",
        bf16=False,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = CastFloatInputsTrainer(
        model=model,
        args=training_args,
        train_dataset=ds,  # using the dummy validation split as dummy train data
        data_collator=collator,
        tokenizer=processor.tokenizer,
    )

    print("Starting training...")
    trainer.train()

if __name__ == "__main__":
    main()
