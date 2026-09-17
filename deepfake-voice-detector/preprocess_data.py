import os
import torch
import numpy as np
from pathlib import Path
import librosa
from transformers import AutoFeatureExtractor

os.environ["HF_HOME"] = "E:/VoiceDeepfakeAI/cache"

BASE_DIR = Path("E:/VoiceDeepfakeAI/dataset")
SR = 16000

def load_audio(path, duration=2.0):
    try:
        y, sr = librosa.load(path, sr=SR)
        y, _ = librosa.effects.trim(y, top_db=20)
        return y
    except:
        return None

def process_split(split, extractor):
    audio_data = []
    labels = []
    split_dir = BASE_DIR / split
    
    for f in (split_dir / "fake").glob("*.wav"):
        y = load_audio(str(f))
        if y is not None:
            audio_data.append(y)
            labels.append(0)
            
    for f in (split_dir / "real").glob("*.wav"):
        y = load_audio(str(f))
        if y is not None:
            audio_data.append(y)
            labels.append(1)
            
    # Process all at once or in a loop
    all_inputs = []
    for audio in audio_data:
        inputs = extractor(
            audio, 
            sampling_rate=SR, 
            padding="max_length", 
            max_length=int(SR * 2.0), 
            truncation=True,
            return_tensors="pt"
        )
        all_inputs.append({key: val[0] for key, val in inputs.items()})
        
    return all_inputs, labels

if __name__ == "__main__":
    print("Loading Extractor...")
    extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base", cache_dir="E:/VoiceDeepfakeAI/cache")
    
    print("Processing Train...")
    train_inputs, train_labels = process_split("train", extractor)
    torch.save({"inputs": train_inputs, "labels": train_labels}, "E:/VoiceDeepfakeAI/dataset/train_tensors.pt")
    
    print("Processing Val...")
    val_inputs, val_labels = process_split("val", extractor)
    torch.save({"inputs": val_inputs, "labels": val_labels}, "E:/VoiceDeepfakeAI/dataset/val_tensors.pt")
    
    print("Done preprocessing!")
