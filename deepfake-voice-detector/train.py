"""
Deepfake Voice Detector - Wav2Vec2 Fine-Tuning
This script fine-tunes a Hugging Face Wav2Vec2 model for anti-spoofing.
It uses:
- dataset/fake/ -> Class 0 (Synthetic)
- dataset/real/ -> Class 1 (Human)
- dataset/live_mic_real/ -> Class 1 (Live Human Microphone, for domain adaptation)
"""
import os
os.environ["WANDB_DISABLED"] = "true"

import glob
import torch
import numpy as np
import librosa
from pathlib import Path
from datasets import Dataset, DatasetDict
from transformers import (
    AutoFeatureExtractor,
    Wav2Vec2ForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import traceback
import warnings
warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
FAKE_DIR = DATASET_DIR / "fake"
REAL_DIR = DATASET_DIR / "real"
LIVE_MIC_DIR = DATASET_DIR / "live_mic_real"
LIVE_MIC_FAKE_DIR = DATASET_DIR / "live_mic_fake"
MODEL_DIR = BASE_DIR / "models"

MODEL_ID = "facebook/wav2vec2-base"
SR = 16000

# Ensure directories exist
FAKE_DIR.mkdir(parents=True, exist_ok=True)
REAL_DIR.mkdir(parents=True, exist_ok=True)
LIVE_MIC_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_audio(filepath: str, max_duration_s=2.0):
    """Load audio and resample to 16kHz."""
    y, _ = librosa.load(filepath, sr=SR, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)
    
    # Cap duration to prevent OOM
    max_samples = int(max_duration_s * SR)
    if len(y) > max_samples:
        y = y[:max_samples]
        
    return y


def build_dataset():
    """Build a list of audio arrays and labels."""
    audio_data = []
    labels = []
    
    # 0 = Fake
    for f in FAKE_DIR.glob("*"):
        if f.is_file() and f.suffix in ['.wav', '.mp3', '.ogg', '.m4a']:
            try:
                y = load_audio(str(f))
                if len(y) > SR * 0.5:
                    audio_data.append(y)
                    labels.append(0)
            except Exception as e:
                print(f"Failed {f}: {e}")
                
    # 1 = Real
    for f in REAL_DIR.glob("*"):
        if f.is_file() and f.suffix in ['.wav', '.mp3', '.ogg', '.m4a']:
            try:
                y = load_audio(str(f))
                if len(y) > SR * 0.5:
                    audio_data.append(y)
                    labels.append(1)
            except Exception as e:
                print(f"Failed {f}: {e}")
                
    # 1 = Real (Live Mic)
    if LIVE_MIC_DIR.exists():
        for f in LIVE_MIC_DIR.glob("*.wav"):
            try:
                y = load_audio(str(f))
                if len(y) > SR * 0.5:
                    audio_data.append(y)
                    labels.append(1)
            except Exception as e:
                print(f"Failed {f}: {e}")

    # 0 = Fake (Live Mic)
    if LIVE_MIC_FAKE_DIR.exists():
        for f in LIVE_MIC_FAKE_DIR.glob("*.wav"):
            try:
                y = load_audio(str(f))
                if len(y) > SR * 0.5:
                    audio_data.append(y)
                    labels.append(0)
            except Exception as e:
                print(f"Failed {f}: {e}")

    if not audio_data:
        raise ValueError("No valid audio files found in dataset directories.")

    return Dataset.from_dict({"audio": audio_data, "label": labels})


def main():
    print(f"Loading dataset...")
    ds = build_dataset()
    
    # Split into train/test
    ds = ds.train_test_split(test_size=0.2, seed=42)
    print(f"Train samples: {len(ds['train'])}, Test samples: {len(ds['test'])}")
    
    print(f"Loading Feature Extractor ({MODEL_ID})...")
    feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    
    def preprocess_function(examples):
        # inputs are list of numpy arrays
        audio_arrays = examples["audio"]
        inputs = feature_extractor(
            audio_arrays,
            sampling_rate=SR,
            max_length=int(SR * 2.0), 
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = examples["label"]
        return inputs

    print("Preprocessing dataset...")
    encoded_dataset = ds.map(preprocess_function, remove_columns=["audio"], batched=True, batch_size=4)
    
    print(f"Loading Wav2Vec2 Model ({MODEL_ID})...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID,
        num_labels=2,
        ignore_mismatched_sizes=True
    )
    
    # Freeze the CNN feature encoder to save memory and prevent PyTorch CPU segfaults
    model.freeze_feature_encoder()


    
    os.environ["WANDB_DISABLED"] = "true"

    print("Initializing TrainingArguments...")
    training_args = TrainingArguments(
        output_dir="./outputs",
        eval_strategy="no",
        save_strategy="no",
        learning_rate=3e-5,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=3,
        warmup_steps=10,
        logging_steps=5,
        disable_tqdm=True,
        load_best_model_at_end=False,
        push_to_hub=False,
        report_to="none",
    )
    
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
    )
    
    print("Starting fine-tuning...")
    trainer.train()
    
    print("Saving fine-tuned model...")
    save_path = str(MODEL_DIR / "wav2vec2-deepfake-finetuned")
    trainer.save_model(save_path)
    feature_extractor.save_pretrained(save_path)
    
    print(f"Done! Model saved to {save_path}.")
    print(f"Update backend/voice_detector.py to use: self.model_name = r'{save_path}'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
