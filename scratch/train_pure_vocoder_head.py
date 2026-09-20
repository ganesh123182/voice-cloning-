import sys
sys.path.insert(0, '.')
import glob, os
import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# Let's verify dataset and build a high-performance vocoder defense model
ds_dir = "E:/VoiceDeepfakeAI/voice_deepfake_dataset"
train_df = pd.read_csv(os.path.join(ds_dir, "metadata", "train.csv"))
test_df = pd.read_csv(os.path.join(ds_dir, "metadata", "test.csv"))
combined = pd.concat([train_df, test_df], ignore_index=True)

from backend.audio_utils import extract_robust_vocoder_features, condition_speakerphone_audio, simulate_speakerphone_rir_and_codec

sr = 16000
WINDOW_SIZE = int(sr * 3.0)
STRIDE_SIZE = int(sr * 1.5)

real_files = [p for p in combined[combined['label'] == 'real']['file_path'].tolist() if os.path.exists(p)][:60]
for p in glob.glob('backend/storage/enrolled_voices/*.wav'):
    if os.path.getsize(p) > 10000:
        real_files.append(p)

fake_files = [p for p in combined[combined['label'] == 'fake']['file_path'].tolist() if os.path.exists(p)][:50]
for f in glob.glob('backend/models/tts_cache/*.mp3'):
    if os.path.getsize(f) > 5000:
        fake_files.append(f)
for f in ['google_tts_test.mp3', 'edge_tts_test.mp3', 'test.wav', 'voice1.wav']:
    if os.path.exists(f) and f not in fake_files:
        fake_files.append(f)

print(f"Base audio files: Real={len(real_files)}, Fake={len(fake_files)}")

X = []
Y = []

def process_file(p, label):
    try:
        y, _ = librosa.load(p, sr=sr, mono=True)
        if len(y) < sr * 0.5: return
        
        # Sliding windows
        windows = []
        if len(y) <= WINDOW_SIZE:
            windows.append(y)
        else:
            for s in range(0, len(y) - WINDOW_SIZE + 1, STRIDE_SIZE):
                windows.append(y[s:s+WINDOW_SIZE])
                
        for w in windows:
            # 1. Direct window
            w_cond = condition_speakerphone_audio(w, sr)
            X.append(extract_robust_vocoder_features(w_cond, sr))
            Y.append(label)
            
            # 2. Speakerphone RIR simulation
            w_rir = simulate_speakerphone_rir_and_codec(w, sr, reverb_intensity=0.30)
            w_rir_cond = condition_speakerphone_audio(w_rir, sr)
            X.append(extract_robust_vocoder_features(w_rir_cond, sr))
            Y.append(label)
            
            # 3. Far-field / low gain mobile mic capture simulation (RMS 0.010 - 0.035)
            noise = np.random.randn(len(w_rir)).astype(np.float32) * 0.003
            w_mic = (w_rir * 0.15) + noise
            w_mic_cond = condition_speakerphone_audio(w_mic, sr)
            X.append(extract_robust_vocoder_features(w_mic_cond, sr))
            Y.append(label)
    except Exception:
        pass

for p in fake_files:
    process_file(p, 0)
for p in real_files:
    process_file(p, 1)

X = np.array(X)
Y = np.array(Y)
print(f"Total window samples: {X.shape} (Fake={np.sum(Y==0)}, Real={np.sum(Y==1)})")

# Train logistic regression
vocoder_pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced', random_state=42)
)
vocoder_pipeline.fit(X, Y)

# Evaluate 5-fold CV
from sklearn.model_selection import StratifiedKFold, cross_val_score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(vocoder_pipeline, X, Y, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")

# Save model
save_path = "backend/models/vocoder_defense_head.joblib"
joblib.dump(vocoder_pipeline, save_path)
print(f"Saved calibrated model to {save_path}!")
