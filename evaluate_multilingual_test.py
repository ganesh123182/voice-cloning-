"""
TrustVoice Multilingual Wav2Vec2 Unseen Test Evaluation Script
Benchmarks the trained checkpoint on the 140 unseen test audio files
across English, Hindi, and Hinglish.
"""
import sys
import json
import time
import torch
import librosa
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
torch.set_num_threads(4)

BASE_DIR = Path("E:/VoiceDeepfakeAI")
DATASET_DIR = BASE_DIR / "voice_deepfake_dataset"
METADATA_DIR = DATASET_DIR / "metadata"
CHECKPOINT = BASE_DIR / "checkpoints" / "wav2vec2-multilingual-v3"
REPORT_DIR = BASE_DIR / "reports"

SR = 16000
MAX_DURATION_SEC = 3.0
MAX_SAMPLES = int(SR * MAX_DURATION_SEC)
BATCH_SIZE = 8

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
            
        try:
            y, _ = librosa.effects.trim(y, top_db=20)
        except Exception:
            pass
            
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
        item["file_path"] = file_path
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
    
    tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
    
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
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
    print("=" * 78)
    print("      TRUSTVOICE: BENCHMARKING MULTILINGUAL MODEL ON UNSEEN TEST AUDIO  ")
    print("=" * 78)
    
    device = torch.device("cpu")
    print(f" Loading checkpoint from: {CHECKPOINT}")
    extractor = AutoFeatureExtractor.from_pretrained(str(CHECKPOINT))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(CHECKPOINT), num_labels=2)
    model.to(device)
    
    test_csv = METADATA_DIR / "test.csv"
    test_dataset = MultilingualAudioDataset(test_csv, extractor)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f" Evaluating on {len(test_dataset)} unseen test audio files...")
    start_time = time.time()
    metrics = evaluate(model, test_loader, device)
    elapsed = time.time() - start_time
    
    print("=" * 78)
    print("                    FINAL UNSEEN TEST EVALUATION REPORT                ")
    print("=" * 78)
    print(f" Test Accuracy   : {metrics['accuracy']:.2f}%")
    print(f" Test F1-Score   : {metrics['f1']:.2f}%")
    print(f" Test Precision  : {metrics['precision']:.2f}%")
    print(f" Test Recall     : {metrics['recall']:.2f}%")
    print(f" Test Loss       : {metrics['loss']:.4f}")
    print(f" Evaluation Time : {elapsed:.2f} seconds")
    print("\n Per-Language Accuracy:")
    for lang, acc in metrics["lang_acc"].items():
        print(f"   - {lang.capitalize():<10}: {acc:.2f}%")
        
    print("\n Confusion Matrix (Unseen Test Audio):")
    print(f"   True Real -> Pred Real: {metrics['tp']:<3} | Pred Fake: {metrics['fn']:<3}")
    print(f"   True Fake -> Pred Fake: {metrics['tn']:<3} | Pred Real: {metrics['fp']:<3}")
    print("=" * 78)
    
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_data = {
        "checkpoint": str(CHECKPOINT),
        "test_samples": len(test_dataset),
        "test_accuracy": round(metrics["accuracy"], 2),
        "test_f1": round(metrics["f1"], 2),
        "test_precision": round(metrics["precision"], 2),
        "test_recall": round(metrics["recall"], 2),
        "test_loss": round(metrics["loss"], 4),
        "per_language_accuracy": {k: round(v, 2) for k, v in metrics["lang_acc"].items()},
        "confusion_matrix": {
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            "tn": metrics["tn"]
        }
    }
    
    with open(REPORT_DIR / "training_report_v3.json", "w") as f:
        json.dump(report_data, f, indent=4)
        
    with open("training_report_v3.txt", "w", encoding="utf-8") as f:
        f.write("TrustVoice Multilingual Model Fine-Tuning Benchmark Report\n")
        f.write(f"Checkpoint: {CHECKPOINT}\n")
        f.write(f"Test Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Test F1-Score: {metrics['f1']:.2f}%\n")
        f.write("Per-Language Accuracy:\n")
        for lang, acc in metrics["lang_acc"].items():
            f.write(f"  - {lang.capitalize()}: {acc:.2f}%\n")
            
    print(f"\n[PASS] Report saved to: {REPORT_DIR / 'training_report_v3.json'}")
    print(f"[PASS] Summary written to: training_report_v3.txt")

if __name__ == "__main__":
    main()
