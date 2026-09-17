import os
import torch
import numpy as np
from pathlib import Path
from transformers import Wav2Vec2ForSequenceClassification
from sklearn.metrics import accuracy_score
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

os.environ["WANDB_DISABLED"] = "true"
os.environ["HF_HOME"] = "E:/VoiceDeepfakeAI/cache"

CHECKPOINT_DIR = Path("E:/VoiceDeepfakeAI/checkpoints")
MODEL_ID = "facebook/wav2vec2-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PreprocessedDataset(Dataset):
    def __init__(self, data_dict):
        self.inputs = data_dict["inputs"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.inputs[idx]
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

def main():
    print("Loading model weights (FIRST to prevent PyTorch RNG/Context segfaults)...")
    model = Wav2Vec2ForSequenceClassification.from_pretrained(
        MODEL_ID, num_labels=2, cache_dir="E:/VoiceDeepfakeAI/cache"
    )
    
    print(f"Device: {DEVICE}")
    
    print("Loading preprocessed datasets...")
    train_data = torch.load("E:/VoiceDeepfakeAI/dataset/train_tensors.pt", weights_only=False)
    val_data = torch.load("E:/VoiceDeepfakeAI/dataset/val_tensors.pt", weights_only=False)
    
    train_ds = PreprocessedDataset(train_data)
    val_ds = PreprocessedDataset(val_data)
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2)
    
    print("Freezing feature encoder...")
    model.freeze_feature_encoder()
    
    print("Freezing entire Wav2Vec2 base (Linear Probing)...")
    if hasattr(model, "wav2vec2"):
        for param in model.wav2vec2.parameters():
            param.requires_grad = False
            
    # Only the classification head and projector are trainable.
    model.to(DEVICE)
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    
    print("Starting training loop...")
    epochs = 5
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
            labels = batch["labels"].to(DEVICE)
            
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
                
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(DEVICE) for k, v in batch.items() if k != "labels"}
                labels = batch["labels"].to(DEVICE)
                logits = model(**inputs).logits
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        acc = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} Summary | Avg Loss: {total_loss/len(train_loader):.4f} | Val Acc: {acc*100:.1f}%")
        
        if acc >= best_acc:
            best_acc = acc
            final_model_path = str(CHECKPOINT_DIR / "wav2vec2-deepfake-finetuned")
            model.save_pretrained(final_model_path)
            print(f"--> Saved best model to {final_model_path}")
            
            # Save extractor too
            from transformers import AutoFeatureExtractor
            AutoFeatureExtractor.from_pretrained(MODEL_ID, cache_dir="E:/VoiceDeepfakeAI/cache").save_pretrained(final_model_path)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
