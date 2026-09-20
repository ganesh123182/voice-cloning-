import sys
sys.path.insert(0, '.')
import glob, os
import librosa
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from backend.audio_utils import extract_robust_vocoder_features, condition_speakerphone_audio, simulate_speakerphone_rir_and_codec

# 1. Train on real and fake files using sliding windows of 3.0s!
ds_dir = "E:/VoiceDeepfakeAI/voice_deepfake_dataset"
train_df = pd.read_csv(os.path.join(ds_dir, "metadata", "train.csv"))
test_df = pd.read_csv(os.path.join(ds_dir, "metadata", "test.csv"))
combined = pd.concat([train_df, test_df], ignore_index=True)

real_files = [p for p in combined[combined['label'] == 'real']['file_path'].tolist() if os.path.exists(p)][:50]
for p in glob.glob('backend/storage/enrolled_voices/*.wav'):
    if os.path.getsize(p) > 20000:
        real_files.append(p)

fake_files = [p for p in combined[combined['label'] == 'fake']['file_path'].tolist() if os.path.exists(p)][:40]
for f in glob.glob('backend/models/tts_cache/*.mp3'):
    fake_files.append(f)
for f in ['google_tts_test.mp3', 'edge_tts_test.mp3', 'test.wav']:
    if os.path.exists(f) and f not in fake_files:
        fake_files.append(f)

sr = 16000
WINDOW_SIZE = int(sr * 3.0)
STRIDE_SIZE = int(sr * 1.5)

X = []
Y = [] # 0 = Fake, 1 = Real

def extract_windows_and_augment(file_path, label):
    try:
        y, _ = librosa.load(file_path, sr=sr, mono=True)
        if len(y) < sr * 0.5: return
        
        # Slices of 3s
        chunks = []
        if len(y) <= WINDOW_SIZE:
            chunks.append(y)
        else:
            for s in range(0, len(y) - WINDOW_SIZE + 1, STRIDE_SIZE):
                chunks.append(y[s:s+WINDOW_SIZE])
                
        for ch in chunks:
            # 1. Clean
            X.append(extract_robust_vocoder_features(ch, sr))
            Y.append(label)
            
            # 2. Speakerphone RIR + Cond
            ch_sim = simulate_speakerphone_rir_and_codec(ch, sr, reverb_intensity=0.30)
            ch_cond = condition_speakerphone_audio(ch_sim, sr)
            X.append(extract_robust_vocoder_features(ch_cond, sr))
            Y.append(label)
            
            # 3. Low amplitude mic capture (RMS ~0.015)
            noise = np.random.randn(len(ch_sim)).astype(np.float32) * 0.003
            ch_low = (ch_sim * 0.15) + noise
            ch_low_cond = condition_speakerphone_audio(ch_low, sr)
            X.append(extract_robust_vocoder_features(ch_low_cond, sr))
            Y.append(label)
    except Exception as e:
        pass

print("Extracting multi-window augmented features...")
for p in fake_files:
    extract_windows_and_augment(p, 0)
for p in real_files:
    extract_windows_and_augment(p, 1)

X = np.array(X)
Y = np.array(Y)
print(f"Total training windows: {X.shape[0]} (Fake={np.sum(Y==0)}, Real={np.sum(Y==1)})")

# Train logistic regression on acoustic features
model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000, class_weight='balanced', random_state=42))
model.fit(X, Y)

# Test on google_tts_test.mp3 in 3-second windows
print("\n--- Testing on google_tts_test.mp3 windows ---")
y_gem, _ = librosa.load('google_tts_test.mp3', sr=sr)
for i in range(0, len(y_gem) - WINDOW_SIZE + 1, STRIDE_SIZE):
    w = y_gem[i:i+WINDOW_SIZE]
    w_cond = condition_speakerphone_audio(w, sr)
    feat = extract_robust_vocoder_features(w_cond, sr).reshape(1, -1)
    prob_fake = model.predict_proba(feat)[0][0] * 100.0
    print(f"Window {i//STRIDE_SIZE + 1} ({i/sr:.1f}s - {(i+WINDOW_SIZE)/sr:.1f}s): Clean Fake Prob = {prob_fake:.1f}%")

print("\n--- Testing on google_tts_test.mp3 Mobile Phone Capture Simulation (RMS ~0.015) ---")
y_sim = simulate_speakerphone_rir_and_codec(y_gem, sr, reverb_intensity=0.25)
noise = np.random.randn(len(y_sim)).astype(np.float32) * 0.003
y_room = y_sim * 0.15 + noise
for i in range(0, len(y_room) - WINDOW_SIZE + 1, STRIDE_SIZE):
    w = y_room[i:i+WINDOW_SIZE]
    w_cond = condition_speakerphone_audio(w, sr)
    feat = extract_robust_vocoder_features(w_cond, sr).reshape(1, -1)
    prob_fake = model.predict_proba(feat)[0][0] * 100.0
    print(f"Window {i//STRIDE_SIZE + 1} ({i/sr:.1f}s - {(i+WINDOW_SIZE)/sr:.1f}s): Mic Sim Fake Prob = {prob_fake:.1f}%")

print("\n--- Testing on Real Human Samples ---")
for rf in glob.glob('backend/storage/enrolled_voices/*.wav')[:3]:
    y_r, _ = librosa.load(rf, sr=sr)
    for i in range(0, min(len(y_r) - WINDOW_SIZE + 1, STRIDE_SIZE * 2), STRIDE_SIZE):
        w = y_r[i:i+WINDOW_SIZE]
        w_cond = condition_speakerphone_audio(w, sr)
        feat = extract_robust_vocoder_features(w_cond, sr).reshape(1, -1)
        prob_fake = model.predict_proba(feat)[0][0] * 100.0
        print(f"Real Human ({os.path.basename(rf)}): Fake Prob = {prob_fake:.1f}%")
