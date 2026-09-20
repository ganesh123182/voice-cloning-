import sys
sys.path.insert(0, '.')
import glob, os
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from backend.audio_utils import extract_robust_vocoder_features, condition_speakerphone_audio, simulate_speakerphone_rir_and_codec

# Collect real and fake files
ds_dir = "E:/VoiceDeepfakeAI/voice_deepfake_dataset"
train_df = pd.read_csv(os.path.join(ds_dir, "metadata", "train.csv"))
test_df = pd.read_csv(os.path.join(ds_dir, "metadata", "test.csv"))
combined = pd.concat([train_df, test_df], ignore_index=True)

real_files = [p for p in combined[combined['label'] == 'real']['file_path'].tolist() if os.path.exists(p)][:40]
for p in glob.glob('backend/storage/enrolled_voices/*.wav'):
    if os.path.getsize(p) > 20000:
        real_files.append(p)

fake_files = [p for p in combined[combined['label'] == 'fake']['file_path'].tolist() if os.path.exists(p)][:35]
for f in glob.glob('backend/models/tts_cache/*.mp3'):
    fake_files.append(f)
for f in ['google_tts_test.mp3', 'edge_tts_test.mp3', 'test.wav']:
    if os.path.exists(f) and f not in fake_files:
        fake_files.append(f)

print(f"Dataset: Real={len(real_files)}, Fake={len(fake_files)}")

X_ac = []
Y_ac = []

for p in fake_files:
    try:
        y, sr = librosa.load(p, sr=16000)
        if len(y) < sr * 0.5: continue
        # Clean
        X_ac.append(extract_robust_vocoder_features(y, sr))
        Y_ac.append(0) # Fake
        # Speakerphone + noise
        y_sim = simulate_speakerphone_rir_and_codec(y, sr, reverb_intensity=0.30)
        y_cond = condition_speakerphone_audio(y_sim, sr)
        X_ac.append(extract_robust_vocoder_features(y_cond, sr))
        Y_ac.append(0)
    except Exception:
        pass

for p in real_files:
    try:
        y, sr = librosa.load(p, sr=16000)
        if len(y) < sr * 0.5: continue
        # Clean
        X_ac.append(extract_robust_vocoder_features(y, sr))
        Y_ac.append(1) # Real
        # Speakerphone + noise
        y_sim = simulate_speakerphone_rir_and_codec(y, sr, reverb_intensity=0.30)
        y_cond = condition_speakerphone_audio(y_sim, sr)
        X_ac.append(extract_robust_vocoder_features(y_cond, sr))
        Y_ac.append(1)
    except Exception:
        pass

X_ac = np.array(X_ac)
Y_ac = np.array(Y_ac)
print(f"Extracted acoustic features: {X_ac.shape} (Fake={np.sum(Y_ac==0)}, Real={np.sum(Y_ac==1)})")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, clf in [
    ("LogisticRegression", make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
    ("RandomForest", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)),
    ("GradientBoosting", GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
]:
    scores = cross_val_score(clf, X_ac, Y_ac, cv=cv, scoring='accuracy')
    print(f"Model: {name:20s} -> CV Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")
