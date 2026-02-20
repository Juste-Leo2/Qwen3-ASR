import io
import torch
from datasets import load_dataset, Audio
from qwen_asr import Qwen3ASRModel
import evaluate
import numpy as np
import librosa

def main():
    print("Loading datasets...")
    # 1. On charge sans l'argument decode
    dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    
    # 2. On empêche le décodage automatique (pour garder les bytes raw et éviter torchcodec)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    print("Loading model...")
    # Use bfloat16 if device supports it, otherwise float16
    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    
    model = Qwen3ASRModel.from_pretrained(
        "Qwen/Qwen3-ASR-0.6B",
        dtype=dtype,
        device_map="cuda:0" if torch.cuda.is_available() else "cpu",
        max_inference_batch_size=8,
        max_new_tokens=256,
    )
    
    print("Running inference...")
    
    # We will collect references and predictions
    references = []
    predictions = []
    
    # Process in batches for speed if needed, or just iterate (it's a small dummy dataset)
    for i, item in enumerate(dataset):
        # On décode avec librosa à la volée
        audio_bytes = item["audio"]["bytes"]
        audio_array, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        
        target_text = item["text"].upper()
        
        # transcribe takes audio as file path, url, base64, or (np.ndarray, sr) tuple
        results = model.transcribe(
            audio=(audio_array, sr),
            language="English"
        )
        
        pred_text = results[0].text.upper()
        
        references.append(target_text)
        predictions.append(pred_text)
        
        if (i + 1) % 10 == 0:
            print(f"Processed {i+1} / {len(dataset)} examples.")
            
    print("Inference done. Calculating WER...")
    try:
        wer_metric = evaluate.load("wer")
        wer = wer_metric.compute(predictions=predictions, references=references)
        print(f"Overall WER: {wer:.4f}")
    except Exception as e:
        print(f"Could not calculate WER via evaluate: {e}")
        # manual fallback
        import jiwer
        wer = jiwer.wer(references, predictions)
        print(f"Overall WER (jiwer): {wer:.4f}")
        
    print("\nSample predictions:")
    for i in range(min(5, len(references))):
        print(f"Ref: {references[i]}")
        print(f"Prd: {predictions[i]}")
        print("-" * 30)

if __name__ == "__main__":
    main()