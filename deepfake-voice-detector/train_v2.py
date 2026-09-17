"""
Wav2Vec2 Fine-Tuning Script — VoiceDeepfakeAI
Uses dataset from E:/VoiceDeepfakeAI/datasets/manifest.json
Trains Wav2Vec2ForSequenceClassification for REAL vs FAKE
10 epochs with early stopping, comprehensive metrics
"""
import os
import json
import gc
import time
import random
import numpy as np
import torch
import librosa
from pathlib import Path
from collections import Counter
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor

# All caches to E: drive
os.environ["HF_HOME"] = "E:/VoiceDeepfakeAI/hf_cache"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TMPDIR"] = "E:/VoiceDeepfakeAI/tmp"
os.environ["TEMP"] = "E:/VoiceDeepfakeAI/tmp"

torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
MAX_DURATION = 4.0  # 4 seconds for better context
MODEL_ID = "facebook/wav2vec2-base"
CHECKPOINT_DIR = Path("E:/VoiceDeepfakeAI/checkpoints/wav2vec2-deepfake-v2")
REPORTS_DIR = Path("E:/VoiceDeepfakeAI/reports")
MANIFEST_PATH = Path("E:/VoiceDeepfakeAI/datasets/manifest.json")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Dataset Split
# ============================================================
def create_splits(manifest_path):
    """Create randomized train/val/test splits for local data"""
    with open(manifest_path) as f:
        data = json.load(f)

    reals = [item for item in data if item["label_id"] == 1]
    fakes = [item for item in data if item["label_id"] == 0]
    
    random.seed(42)
    random.shuffle(reals)
    random.shuffle(fakes)
    
    n_r = len(reals)
    n_f = len(fakes)
    
    train = reals[:int(0.7*n_r)] + fakes[:int(0.7*n_f)]
    val = reals[int(0.7*n_r):int(0.85*n_r)] + fakes[int(0.7*n_f):int(0.85*n_f)]
    test = reals[int(0.85*n_r):] + fakes[int(0.85*n_f):]
    
    random.shuffle(train)
    random.shuffle(val)
    random.shuffle(test)

    # Save splits
    split_dir = Path("E:/VoiceDeepfakeAI/datasets")
    for name, split_data in [("train", train), ("val", val), ("test", test)]:
        with open(split_dir / f"{name}_v2.json", "w") as f:
            json.dump(split_data, f, indent=2)

    fake_train = sum(1 for x in train if x["label_id"] == 0)
    real_train = sum(1 for x in train if x["label_id"] == 1)
    fake_val = sum(1 for x in val if x["label_id"] == 0)
    real_val = sum(1 for x in val if x["label_id"] == 1)
    fake_test = sum(1 for x in test if x["label_id"] == 0)
    real_test = sum(1 for x in test if x["label_id"] == 1)

    print(f"Train: {len(train)} ({fake_train} fake, {real_train} real)")
    print(f"Val:   {len(val)} ({fake_val} fake, {real_val} real)")
    print(f"Test:  {len(test)} ({fake_test} fake, {real_test} real)")

    return train, val, test


# ============================================================
# Dataset Class
# ============================================================
class AudioDataset(Dataset):
    def __init__(self, items, extractor, augment=False):
        self.items = items
        self.extractor = extractor
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            y, _ = librosa.load(item["file"], sr=SR, mono=True)
        except Exception:
            y = np.zeros(int(SR * 1.0), dtype=np.float32)

        y, _ = librosa.effects.trim(y, top_db=20)

        # Augmentation
        if self.augment:
            if random.random() < 0.3:
                y = y + np.random.randn(len(y)).astype(np.float32) * 0.005
            if random.random() < 0.3:
                gain = random.uniform(0.7, 1.3)
                y = y * gain

        # Pad/truncate
        max_len = int(SR * MAX_DURATION)
        if len(y) < int(SR * 0.5):
            y = np.pad(y, (0, int(SR * 0.5) - len(y)), mode='constant')
        if len(y) > max_len:
            if self.augment:
                start = random.randint(0, len(y) - max_len)
                y = y[start:start + max_len]
            else:
                y = y[:max_len]

        inputs = self.extractor(
            y, sampling_rate=SR,
            max_length=max_len, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(item["label_id"], dtype=torch.long)
        return inputs


# ============================================================
# Training
# ============================================================
def train_model(train_data, val_data):
    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}

    print("Loading base model...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir="E:/VoiceDeepfakeAI/hf_cache")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, cache_dir="E:/VoiceDeepfakeAI/hf_cache",
        id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True
    )

    # Freeze CNN + lower transformer layers
    model.freeze_feature_encoder()
    if hasattr(model, "wav2vec2"):
        for name, param in model.wav2vec2.named_parameters():
            if "encoder.layers" in name:
                layer_num = int(name.split("encoder.layers.")[1].split(".")[0])
                param.requires_grad = (layer_num >= 6)
            elif "feature_extractor" in name or "feature_projection" in name:
                param.requires_grad = False

    model.to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total_params:,} ({100*trainable/total_params:.1f}%)")

    # Datasets
    train_ds = AudioDataset(train_data, extractor, augment=True)
    val_ds = AudioDataset(val_data, extractor, augment=False)

    # Class weights for imbalance
    labels = [item["label_id"] for item in train_data]
    counts = Counter(labels)
    total = len(labels)
    weights = torch.tensor([total / (2 * counts[0]), total / (2 * counts[1])], dtype=torch.float32).to(DEVICE)
    print(f"Class weights: FAKE={weights[0]:.2f}, REAL={weights[1]:.2f}")
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=2, num_workers=0)

    # Optimizer
    head_params = list(model.classifier.parameters()) + list(model.projector.parameters())
    body_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and "classifier" not in n and "projector" not in n]

    optimizer = AdamW([
        {"params": body_params, "lr": 5e-6},
        {"params": head_params, "lr": 2e-5}
    ], weight_decay=0.01)

    epochs = 10
    accumulation_steps = 4
    patience = 3
    best_val_loss = float("inf")
    no_improve = 0
    history = []

    print(f"\nStarting training: {epochs} epochs, {len(train_loader)} steps/epoch")
    print(f"Device: {DEVICE}")

    for epoch in range(epochs):
        epoch_start = time.time()
        model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels_batch = batch["labels"].to(DEVICE)

            outputs = model(**inputs)
            loss = loss_fn(outputs.logits, labels_batch) / accumulation_steps
            loss.backward()

            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels_batch.cpu().tolist())

            if (step + 1) % 20 == 0:
                print(f"  Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item()*accumulation_steps:.4f}")

            # Free memory
            del outputs, loss
            if step % 50 == 0:
                gc.collect()

        avg_train_loss = total_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)

        # Validation
        model.eval()
        val_loss = 0
        val_preds = []
        val_labels = []
        val_scores = []

        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
                labels_batch = batch["labels"].to(DEVICE)
                outputs = model(**inputs)
                val_loss += loss_fn(outputs.logits, labels_batch).item()
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                val_preds.extend(torch.argmax(probs, dim=-1).cpu().tolist())
                val_labels.extend(labels_batch.cpu().tolist())
                val_scores.extend(probs[:, 0].cpu().tolist())  # fake probability

        avg_val_loss = val_loss / len(val_loader)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds, pos_label=0, zero_division=0)
        val_prec = precision_score(val_labels, val_preds, pos_label=0, zero_division=0)
        val_rec = recall_score(val_labels, val_preds, pos_label=0, zero_division=0)
        try:
            true_binary = [1 if l == 0 else 0 for l in val_labels]
            val_auc = roc_auc_score(true_binary, val_scores)
        except:
            val_auc = 0.5

        epoch_time = time.time() - epoch_start

        epoch_info = {
            "epoch": epoch + 1, "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4), "train_acc": round(train_acc, 4),
            "val_acc": round(val_acc, 4), "val_f1": round(val_f1, 4),
            "val_precision": round(val_prec, 4), "val_recall": round(val_rec, 4),
            "val_auc": round(val_auc, 4), "epoch_time_sec": round(epoch_time, 1)
        }
        history.append(epoch_info)

        print(f"\n--- Epoch {epoch+1}/{epochs} ({epoch_time:.0f}s) ---")
        print(f"  Train: loss={avg_train_loss:.4f} acc={train_acc:.3f}")
        print(f"  Val:   loss={avg_val_loss:.4f} acc={val_acc:.3f} F1={val_f1:.3f} AUC={val_auc:.3f}")
        print(f"  Val:   prec={val_prec:.3f} rec={val_rec:.3f}")

        # Save best
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improve = 0
            print(f"  >>> New best! Saving checkpoint...")
            model.save_pretrained(CHECKPOINT_DIR)
            extractor.save_pretrained(CHECKPOINT_DIR)
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{patience})")

        if no_improve >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

        gc.collect()

    with open(REPORTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best checkpoint: {CHECKPOINT_DIR}")
    return history


# ============================================================
# Evaluation
# ============================================================
def evaluate_model(test_data):
    print("\n" + "=" * 60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 60)

    extractor = AutoFeatureExtractor.from_pretrained(str(CHECKPOINT_DIR))
    model = Wav2Vec2ForSequenceClassification.from_pretrained(str(CHECKPOINT_DIR))
    model.to(DEVICE)
    model.eval()

    test_ds = AudioDataset(test_data, extractor, augment=False)
    test_loader = DataLoader(test_ds, batch_size=2, num_workers=0)

    all_preds = []
    all_labels = []
    all_scores = []
    all_items = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels_batch = batch["labels"]
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits.float(), dim=-1)
            preds = torch.argmax(probs, dim=-1).cpu().tolist()
            fake_scores = probs[:, 0].cpu().tolist()

            all_preds.extend(preds)
            all_labels.extend(labels_batch.tolist())
            all_scores.extend(fake_scores)

            start_idx = batch_idx * 2
            for j in range(len(preds)):
                if start_idx + j < len(test_data):
                    all_items.append(test_data[start_idx + j])

    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, pos_label=0, zero_division=0)
    rec = recall_score(all_labels, all_preds, pos_label=0, zero_division=0)
    f1 = f1_score(all_labels, all_preds, pos_label=0, zero_division=0)
    true_binary = [1 if l == 0 else 0 for l in all_labels]
    try:
        auc = roc_auc_score(true_binary, all_scores)
    except:
        auc = 0.5

    cm = confusion_matrix(all_labels, all_preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    far = fp / (fp + tn) if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0
    eer = (far + frr) / 2

    # Per-language breakdown
    lang_results = {}
    for i, item in enumerate(all_items):
        lang = item.get("language", "unknown")
        label = item.get("label", "unknown")
        key = f"{lang}_{label}"
        if key not in lang_results:
            lang_results[key] = {"correct": 0, "total": 0}
        lang_results[key]["total"] += 1
        if all_preds[i] == all_labels[i]:
            lang_results[key]["correct"] += 1

    print(f"\nAccuracy:  {acc*100:.1f}%")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1:        {f1:.3f}")
    print(f"ROC-AUC:   {auc:.3f}")
    print(f"FAR:       {far*100:.1f}%")
    print(f"FRR:       {frr*100:.1f}%")
    print(f"EER:       {eer*100:.1f}%")
    print(f"\nConfusion Matrix:")
    print(f"              Pred FAKE  Pred REAL")
    print(f"  Actual FAKE    {cm[0][0]:5d}      {cm[0][1]:5d}")
    print(f"  Actual REAL    {cm[1][0]:5d}      {cm[1][1]:5d}")

    print(f"\nPer-language breakdown:")
    for key, val in sorted(lang_results.items()):
        pct = val["correct"]/val["total"]*100 if val["total"] > 0 else 0
        print(f"  {key}: {val['correct']}/{val['total']} ({pct:.0f}%)")

    report = {
        "accuracy": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "roc_auc": round(auc, 4),
        "far": round(far, 4), "frr": round(frr, 4), "eer": round(eer, 4),
        "confusion_matrix": cm.tolist(),
        "per_language": lang_results,
        "total_test_samples": len(all_labels)
    }
    with open(REPORTS_DIR / "final_evaluation.json", "w") as f:
        json.dump(report, f, indent=2)

    predictions = []
    for i in range(len(all_preds)):
        pred_label = "FAKE" if all_preds[i] == 0 else "REAL"
        true_label = "FAKE" if all_labels[i] == 0 else "REAL"
        predictions.append({
            "file": all_items[i]["file"] if i < len(all_items) else "unknown",
            "true": true_label, "pred": pred_label,
            "fake_score": round(all_scores[i], 4)
        })
    with open(REPORTS_DIR / "test_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    return report


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("VoiceDeepfakeAI — Wav2Vec2 Training Pipeline v2")
    print("=" * 60)

    if not MANIFEST_PATH.exists():
        print(f"ERROR: Manifest not found at {MANIFEST_PATH}")
        print("Run download_dataset.py first!")
        exit(1)

    print("\nPhase 3: Creating train/val/test splits...")
    train_data, val_data, test_data = create_splits(MANIFEST_PATH)

    print("\nPhase 4: Training Wav2Vec2...")
    history = train_model(train_data, val_data)

    print("\nPhase 5-6: Evaluating on test set...")
    report = evaluate_model(test_data)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
