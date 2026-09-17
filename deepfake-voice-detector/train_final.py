import os
import json
import torch
import numpy as np
import librosa
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

# CPU Optimization for Windows
torch.set_num_threads(4)

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HOME"] = "E:/VoiceDeepfakeAI/cache"

CHECKPOINT_DIR = Path("E:/VoiceDeepfakeAI/checkpoints")
MODEL_ID = "facebook/wav2vec2-base" # We start from the base and build our own head
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SR = 16000
MAX_DURATION = 2.0

class LazyAudioDataset(Dataset):
    def __init__(self, json_path, extractor):
        with open(json_path, "r") as f:
            self.data = json.load(f)
        self.extractor = extractor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        y, _ = librosa.load(item["path"], sr=SR, mono=True)
        y, _ = librosa.effects.trim(y, top_db=20)
        
        # Add slight noise if it's training data for augmentation
        if "train.json" in str(self.data):
            noise = np.random.randn(len(y))
            y = y + 0.005 * noise

        inputs = self.extractor(
            y, 
            sampling_rate=SR, 
            max_length=int(SR * MAX_DURATION), 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        # Squeeze batch dimension added by extractor
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(item["label"], dtype=torch.long)
        return inputs

def main():
    print("="*50)
    print("STAGE 2: MEMORY-EFFICIENT WAV2VEC2 FINE-TUNING")
    print("="*50)
    print(f"Device: {DEVICE}")

    # Explicit Label Mapping
    id2label = {0: "FAKE", 1: "REAL"}
    label2id = {"FAKE": 0, "REAL": 1}

    print("Loading Feature Extractor and Model from Base...")
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir="E:/VoiceDeepfakeAI/cache")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, 
        num_labels=2, 
        cache_dir="E:/VoiceDeepfakeAI/cache",
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    print("Loading JSON datasets from E: drive...")
    train_ds = LazyAudioDataset("E:/VoiceDeepfakeAI/dataset/train.json", extractor)
    val_ds = LazyAudioDataset("E:/VoiceDeepfakeAI/dataset/val.json", extractor)
    
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    # Tiny batch size to prevent crashes
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    
    print("Freezing feature encoder (CNNs)...")
    model.freeze_feature_encoder()
    
    print("Freezing lower Transformer layers to save memory and focus on acoustic artifacts...")
    if hasattr(model, "wav2vec2"):
        for name, param in model.wav2vec2.named_parameters():
            if "encoder.layers" in name:
                layer_num = int(name.split("encoder.layers.")[1].split(".")[0])
                if layer_num < 6:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            elif "feature_extractor" in name or "feature_projection" in name:
                param.requires_grad = False
    
    model.to(DEVICE)
    
    # Use different learning rates for the body and head
    head_params = list(model.classifier.parameters()) + list(model.projector.parameters())
    body_params = [p for n, p in model.named_parameters() if p.requires_grad and "classifier" not in n and "projector" not in n]
    
    optimizer = AdamW([
        {'params': body_params, 'lr': 5e-6},
        {'params': head_params, 'lr': 2e-5}
    ])
    
    epochs = 7 # Kept short for prototype demonstration
    accumulation_steps = 4
    best_val_loss = float('inf')
    
    print("Starting training loop...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            
            if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * accumulation_steps
            print(f"Epoch {epoch+1}/{epochs} | Step {step+1}/{len(train_loader)} | Loss: {loss.item() * accumulation_steps:.4f}")
            
        avg_train_loss = total_loss / len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(DEVICE)
                outputs = model(**inputs, labels=labels)
                val_loss += outputs.loss.item()
                
                preds = torch.argmax(outputs.logits, dim=-1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total if total > 0 else 0
        
        print(f"--- Epoch {epoch+1} Summary ---")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc: {val_acc*100:.1f}%")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("New best validation loss! Saving checkpoint...")
            model.save_pretrained(CHECKPOINT_DIR / "wav2vec2-deepfake-finetuned")
            extractor.save_pretrained(CHECKPOINT_DIR / "wav2vec2-deepfake-finetuned")

    print("\nTraining Complete! Best model saved to E:/VoiceDeepfakeAI/checkpoints/wav2vec2-deepfake-finetuned")

if __name__ == "__main__":
    main()
