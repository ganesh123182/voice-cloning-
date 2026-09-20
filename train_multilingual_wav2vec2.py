"""
TrustVoice Multilingual Wav2Vec2 Fine-Tuning Script
Fine-tunes Wav2Vec2 on the newly built multilingual voice deepfake dataset (English, Hindi, Hinglish).
Evaluates across 10 epochs, saves the best checkpoint, and benchmarks on unseen test audio.
"""
import os
import sys
import time
import json
import torch
import random
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

# Fixed seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Threading optimization for CPU
torch.set_num_threads(4)
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# Paths
BASE_DIR = Path("E:/VoiceDeepfakeAI")
DATASET_DIR = BASE_DIR / "voice_deepfake_dataset"
METADATA_DIR = DATASET_DIR / "metadata"
CHECKPOINT_IN = BASE_DIR / "checkpoints" / "wav2vec2-deepfake-v2"
CHECKPOINT_OUT = BASE_DIR / "checkpoints" / "wav2vec2-multilingual-v3"
REPORT_DIR = BASE_DIR / "reports"

SR = 16000
MAX_DURATION_SEC = 3.0
MAX_SAMPLES = int(SR * MAX_DURATION_SEC)
BATCH_SIZE = 8
LEARNING_RATE = 1e-5
NUM_EPOCHS = 10

class MultilingualAudioDataset(Dataset):
    def __init__(self, csv_path: Path, extractor):
        self.df = pd.read_csv(csv_path)
        self.extractor = extractor
        self.label_map = {"fake": 0, "real": 1}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row["file_path"]
        
        try:
            y, _ = librosa.load(file_path, sr=SR, mono=True)
        except Exception:
            y = np.zeros(MAX_SAMPLES, dtype=np.float32)
            
        # Trim silence
        try:
            y, _ = librosa.effects.trim(y, top_db=20)
        except Exception:
            pass
            
        # Pad or crop
        if len(y) < MAX_SAMPLES:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)), mode='constant')
        else:
            y = y[:MAX_SAMPLES]
            
        inputs = self.extractor(
            y, 
            sampling_rate=SR, 
            max_length=MAX_SAMPLES, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        
        item = {k: v.squeeze(0) for k, v in inputs.items()}
        item["labels"] = torch.tensor(self.label_map[row["label"]], dtype=torch.long)
        item["language"] = row["language"]
        return item

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    all_preds = []
    all_labels = []
    all_langs = []
    
    criterion = CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            total_loss += loss.item() * len(labels)
            preds = torch.argmax(logits, dim=-1)
            
            correct += (preds == labels).sum().item()
            total += len(labels)
            
            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_langs.extend(batch["language"])
            
    avg_loss = total_loss / total if total > 0 else 0.0
    accuracy = (correct / total) * 100.0 if total > 0 else 0.0
    
    # Calculate Precision, Recall, F1 for class 1 (REAL) and class 0 (FAKE)
    from collections import Counter
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    # Per language accuracy
    lang_acc = {}
    for lang in set(all_langs):
        l_correct = sum(1 for p, l, g in zip(all_preds, all_labels, all_langs) if g == lang and p == l)
        l_total = sum(1 for g in all_langs if g == lang)
        lang_acc[lang] = (l_correct / l_total * 100) if l_total > 0 else 0.0
        
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "lang_acc": lang_acc
    }

def main():
    print("=" * 78, flush=True)
    print("      TRUSTVOICE: WAV2VEC2 MULTILINGUAL VOICE DEEPFAKE FINE-TUNING     ", flush=True)
    print("=" * 78, flush=True)
    
    device = torch.device("cpu")
    print(f" Compute Device       : {device} (4 Threads)", flush=True)
    print(f" Checkpoint Source    : {CHECKPOINT_IN}", flush=True)
    print(f" Target Checkpoint    : {CHECKPOINT_OUT}", flush=True)
    print(f" Number of Epochs     : {NUM_EPOCHS}", flush=True)
    print(f" Batch Size           : {BATCH_SIZE}", flush=True)
    print(f" Learning Rate        : {LEARNING_RATE}", flush=True)
    
    CHECKPOINT_OUT.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load feature extractor and model
    print("\n[1/4] Loading Pre-Trained Deepfake Model...", flush=True)
    extractor = AutoFeatureExtractor.from_pretrained(str(CHECKPOINT_IN))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(CHECKPOINT_IN), num_labels=2)
    model.to(device)
    
    # Freeze the 7-layer CNN feature encoder for fast, stable CPU training
    model.freeze_feature_encoder()
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f" Total Parameters     : {total_params:,}", flush=True)
    print(f" Trainable Parameters : {trainable_params:,} (CNN Feature Encoder Frozen)", flush=True)
    
    # Create Datasets and Dataloaders
    print("\n[2/4] Loading Dataset Splits...", flush=True)
    train_dataset = MultilingualAudioDataset(METADATA_DIR / "train.csv", extractor)
    val_dataset = MultilingualAudioDataset(METADATA_DIR / "validation.csv", extractor)
    test_dataset = MultilingualAudioDataset(METADATA_DIR / "test.csv", extractor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f" Train Samples : {len(train_dataset)} ({len(train_loader)} batches)", flush=True)
    print(f" Val Samples   : {len(val_dataset)} ({len(val_loader)} batches)", flush=True)
    print(f" Test Samples  : {len(test_dataset)} ({len(test_loader)} batches)", flush=True)
    
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=0.01)
    
    best_val_f1 = -1.0
    best_val_acc = 0.0
    training_history = []
    
    print("\n[3/4] Starting 10-Epoch Training Loop...", flush=True)
    print("-" * 78, flush=True)
    print(f" {'EPOCH':<7} | {'TRAIN LOSS':<12} | {'VAL LOSS':<10} | {'VAL ACC':<10} | {'VAL F1':<10} | {'TIME'}", flush=True)
    print("-" * 78, flush=True)
    
    total_start = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0
        train_total = 0
        
        for batch_idx, batch in enumerate(train_loader, 1):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_values=input_values, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * len(labels)
            train_total += len(labels)
            
            if batch_idx % 25 == 0 or batch_idx == len(train_loader):
                elapsed_b = time.time() - epoch_start
                print(f"  [Epoch {epoch:02d}] Step {batch_idx:02d}/{len(train_loader)} | Loss: {loss.item():.4f} ({elapsed_b:.1f}s)", flush=True)
                
        avg_train_loss = train_loss / train_total
        
        # Validation Evaluation
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["f1"]
        val_loss = val_metrics["loss"]
        
        is_best = val_f1 > best_val_f1
        best_marker = "[BEST]" if is_best else ""
        
        print(f" {epoch:02d}/{NUM_EPOCHS:02d}   | {avg_train_loss:<12.4f} | {val_loss:<10.4f} | {val_acc:<9.1f}% | {val_f1:<9.1f}% | {epoch_time:.1f}s {best_marker}", flush=True)
        
        history_entry = {
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_f1": val_f1,
            "epoch_seconds": epoch_time
        }
        training_history.append(history_entry)
        
        # Save Best Checkpoint
        if is_best:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            model.save_pretrained(str(CHECKPOINT_OUT))
            extractor.save_pretrained(str(CHECKPOINT_OUT))
            
    total_train_time = time.time() - total_start
    print("-" * 78, flush=True)
    print(f" Training Finished in {total_train_time/60:.2f} minutes.", flush=True)
    print(f" Best Validation F1: {best_val_f1:.2f}% | Best Validation Acc: {best_val_acc:.2f}%", flush=True)
    
    # ── Final Unseen Test Benchmark ──
    print("\n[4/4] Evaluating Best Checkpoint on Unseen Test Split (140 files)...", flush=True)
    best_model = Wav2Vec2ForSequenceClassification.from_pretrained(str(CHECKPOINT_OUT))
    best_model.to(device)
    
    test_metrics = evaluate(best_model, test_loader, device)
    
    print("=" * 78, flush=True)
    print("                    FINAL UNSEEN TEST EVALUATION REPORT                ", flush=True)
    print("=" * 78, flush=True)
    print(f" Test Accuracy   : {test_metrics['accuracy']:.2f}%", flush=True)
    print(f" Test F1-Score   : {test_metrics['f1']:.2f}%", flush=True)
    print(f" Test Precision  : {test_metrics['precision']:.2f}%", flush=True)
    print(f" Test Recall     : {test_metrics['recall']:.2f}%", flush=True)
    print(f" Test Loss       : {test_metrics['loss']:.4f}", flush=True)
    print("\n Per-Language Accuracy:", flush=True)
    for lang, acc in test_metrics["lang_acc"].items():
        print(f"   - {lang.capitalize():<10}: {acc:.1f}%", flush=True)
        
    print("\n Confusion Matrix (Unseen Test Audio):", flush=True)
    print(f"   True Real -> Pred Real: {test_metrics['tp']:<3} | Pred Fake: {test_metrics['fn']:<3}", flush=True)
    print(f"   True Fake -> Pred Fake: {test_metrics['tn']:<3} | Pred Real: {test_metrics['fp']:<3}", flush=True)
    print("=" * 78, flush=True)
    
    # Save Report JSON and TXT
    final_report = {
        "model_id": str(CHECKPOINT_OUT),
        "total_epochs": NUM_EPOCHS,
        "training_time_minutes": round(total_train_time / 60, 2),
        "best_val_accuracy": round(best_val_acc, 2),
        "best_val_f1": round(best_val_f1, 2),
        "test_metrics": {
            "accuracy": round(test_metrics["accuracy"], 2),
            "f1": round(test_metrics["f1"], 2),
            "precision": round(test_metrics["precision"], 2),
            "recall": round(test_metrics["recall"], 2),
            "loss": round(test_metrics["loss"], 4),
            "per_language_accuracy": {k: round(v, 2) for k, v in test_metrics["lang_acc"].items()}
        },
        "history": training_history
    }
    
    with open(REPORT_DIR / "training_report_v3.json", "w") as f:
        json.dump(final_report, f, indent=4)
        
    with open("training_report_v3.txt", "w", encoding="utf-8") as f:
        f.write(f"TrustVoice Multilingual Model Fine-Tuning Report\n")
        f.write(f"Checkpoint: {CHECKPOINT_OUT}\n")
        f.write(f"Test Accuracy: {test_metrics['accuracy']:.2f}%\n")
        f.write(f"Test F1-Score: {test_metrics['f1']:.2f}%\n")
        f.write(f"Per-Language Accuracy:\n")
        for lang, acc in test_metrics["lang_acc"].items():
            f.write(f"  - {lang.capitalize()}: {acc:.2f}%\n")
            
    print(f"\n[OK] Model successfully saved to: {CHECKPOINT_OUT}", flush=True)
    print(f"[OK] Training report saved to: {REPORT_DIR / 'training_report_v3.json'}", flush=True)

if __name__ == "__main__":
    main()
