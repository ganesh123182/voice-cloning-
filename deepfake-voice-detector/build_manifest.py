import os
import json
import librosa
from pathlib import Path

BASE = Path("E:/VoiceDeepfakeAI/datasets")
BASE.mkdir(parents=True, exist_ok=True)
MANIFEST = []

print("Scanning local audio for manifest...")
for label_dir, label_str, label_id in [("real", "REAL", 1), ("fake", "FAKE", 0)]:
    src = Path(f"E:/VoiceDeepfakeAI/collected_audio/{label_dir}")
    if src.exists():
        for f in src.glob("*.wav"):
            try:
                dur = librosa.get_duration(path=str(f))
                if dur < 0.5:
                    continue
                MANIFEST.append({
                    "file": str(f), 
                    "label": label_str, 
                    "label_id": label_id,
                    "language": "en", 
                    "source": "gemini_collected" if label_dir == "fake" else "local_real",
                    "duration": round(dur, 2), 
                    "speaker": "local"
                })
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue

manifest_path = BASE / "manifest.json"
with open(manifest_path, "w") as f:
    json.dump(MANIFEST, f, indent=2)

total_real = sum(1 for m in MANIFEST if m["label"] == "REAL")
total_fake = sum(1 for m in MANIFEST if m["label"] == "FAKE")

print(f"Manifest created at {manifest_path}")
print(f"Total valid samples: {len(MANIFEST)} ({total_real} real, {total_fake} fake)")
