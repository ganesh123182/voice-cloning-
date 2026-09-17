import os
import json
import librosa
from pathlib import Path

# Paths
SRC_DIR = Path(r"C:\Users\Ganesh\OneDrive\Documents\voice cloning\deepfake-voice-detector\dataset")
E_DRIVE = Path("E:/VoiceDeepfakeAI")
DEST_DIR = E_DRIVE / "dataset"

# Create directories
for d in ["dataset", "checkpoints", "cache", "logs", "results"]:
    (E_DRIVE / d).mkdir(parents=True, exist_ok=True)

SR = 16000

def get_duration(filepath):
    try:
        # Quick duration check without loading full array
        return librosa.get_duration(path=filepath)
    except:
        return 0.0

def main():
    splits = {"train": [], "val": [], "test": []}
              
    def add_item(filepath, label, split_name):
        duration = get_duration(filepath)
        if duration < 0.5:
            return # Skip too short files
            
        splits[split_name].append({
            "path": str(filepath.resolve()),
            "label": label,
            "duration": duration
        })

    # ==========================================
    # EXPLICIT SPEAKER-DISJOINT SPLITTING RULES
    # ==========================================
    
    # ── FAKE (Label 0) ──
    # Train: FineVoice*, fake_james* (Unseen to test)
    # Val: live_mic_fake chunks (early ones)
    # Test: fake_simone_testing, fake_finevoice, live_mic_fake chunks (later ones)
    
    print("Processing FAKE samples...")
    fake_dir = SRC_DIR / "fake"
    if fake_dir.exists():
        for f in fake_dir.glob("*"):
            if f.is_file():
                if "simone" in f.name.lower() or f.name.lower() == "fake_finevoice.mp3":
                    add_item(f, 0, "test")
                else:
                    add_item(f, 0, "train")
                
    live_mic_fake = list((SRC_DIR / "live_mic_fake").glob("*.wav"))
    
    collected_fake = list((E_DRIVE / "collected_audio" / "fake").glob("*.wav"))
    live_mic_fake.extend(collected_fake)

    live_mic_fake.sort() # sort by timestamp
    for i, f in enumerate(live_mic_fake):
        if i < len(live_mic_fake) // 3:
            add_item(f, 0, "train")
        elif i < (len(live_mic_fake) * 2) // 3:
            add_item(f, 0, "val")
        else:
            add_item(f, 0, "test")

    # ── REAL (Label 1) ──
    # Train: dhurandhar, WhatsApp (early)
    # Val: WhatsApp (mid), live_mic_real (early)
    # Test: real_whatsapp.mp4, WhatsApp (late), live_mic_real (late)
    
    print("Processing REAL samples...")
    real_dir = SRC_DIR / "real"
    if real_dir.exists():
        for f in real_dir.glob("*"):
            if f.is_file():
                if "real_whatsapp" in f.name.lower() or "dhurandhar_real" in f.name.lower():
                    add_item(f, 1, "test")
                else:
                    add_item(f, 1, "train")

    live_mic_real = list((SRC_DIR / "live_mic_real").glob("*.wav"))
    collected_real = list((E_DRIVE / "collected_audio" / "real").glob("*.wav"))
    live_mic_real.extend(collected_real)

    live_mic_real.sort()
    for i, f in enumerate(live_mic_real):
        if i < len(live_mic_real) // 3:
            add_item(f, 1, "train")
        elif i < (len(live_mic_real) * 2) // 3:
            add_item(f, 1, "val")
        else:
            add_item(f, 1, "test")
            
    # Save JSON files
    for split_name, data in splits.items():
        out_path = DEST_DIR / f"{split_name}.json"
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Saved {len(data)} samples to {out_path}")
        
        # Verify stats
        fake_count = sum(1 for item in data if item["label"] == 0)
        real_count = sum(1 for item in data if item["label"] == 1)
        print(f"  {split_name.upper()}: {fake_count} FAKE, {real_count} REAL")

if __name__ == "__main__":
    main()
