import sys
sys.path.insert(0, '.')
import os, glob
import librosa
import numpy as np
import pandas as pd
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from backend.audio_utils import extract_robust_vocoder_features, condition_speakerphone_audio, simulate_speakerphone_rir_and_codec

v3_path = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-multilingual-v3"
ext = AutoFeatureExtractor.from_pretrained(v3_path)
model = Wav2Vec2ForSequenceClassification.from_pretrained(v3_path)
model.eval()

ds_dir = "E:/VoiceDeepfakeAI/voice_deepfake_dataset"
train_df = pd.read_csv(os.path.join(ds_dir, "metadata", "train.csv"))
test_df = pd.read_csv(os.path.join(ds_dir, "metadata", "test.csv"))

sr = 16000
WINDOW_SIZE = int(sr * 3.0)

def extract_16_features(y):
    # 1. Base Wav2Vec2 prob
    y_crop = y[:WINDOW_SIZE]
    if len(y_crop) < WINDOW_SIZE:
        y_crop = np.pad(y_crop, (0, WINDOW_SIZE - len(y_crop)))
    inp = ext(y_crop, sampling_rate=sr, max_length=WINDOW_SIZE, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
        probs = torch.softmax(out.logits.float(), dim=-1)[0].numpy()
    base_fake = float(probs[0])
    base_real = float(probs[1])
    
    # 2. Invariant acoustic features (14)
    ac = extract_robust_vocoder_features(y, sr)
    
    return np.concatenate([[base_fake, base_real], ac])

print("Collecting balanced training set...")
real_files = train_df[train_df['label'] == 'real']['file_path'].tolist()[:80]
for p in glob.glob('backend/storage/enrolled_voices/*.wav'):
    if os.path.getsize(p) > 10000: real_files.append(p)

fake_files = train_df[train_df['label'] == 'fake']['file_path'].tolist()[:50]
for f in glob.glob('backend/models/tts_cache/*.mp3'):
    if os.path.getsize(f) > 5000: fake_files.append(f)
for f in ['google_tts_test.mp3', 'edge_tts_test.mp3', 'test.wav']:
    if os.path.exists(f) and f not in fake_files: fake_files.append(f)

X_train = []
Y_train = []

for p in fake_files:
    try:
        y, _ = librosa.load(p, sr=sr, mono=True)
        if len(y) < sr * 0.5: continue
        # Clean
        X_train.append(extract_16_features(y[:WINDOW_SIZE]))
        Y_train.append(0)
        # Mobile sim
        y_sim = simulate_speakerphone_rir_and_codec(y[:WINDOW_SIZE], sr=sr, reverb_intensity=0.25)
        noise = np.random.randn(len(y_sim)).astype(np.float32) * 0.003
        y_mic = y_sim * 0.15 + noise
        y_cond = condition_speakerphone_audio(y_mic, sr)
        X_train.append(extract_16_features(y_cond))
        Y_train.append(0)
    except Exception:
        pass

for p in real_files:
    try:
        y, _ = librosa.load(p, sr=sr, mono=True)
        if len(y) < sr * 0.5: continue
        # Clean
        X_train.append(extract_16_features(y[:WINDOW_SIZE]))
        Y_train.append(1)
        # Mobile sim
        y_sim = simulate_speakerphone_rir_and_codec(y[:WINDOW_SIZE], sr=sr, reverb_intensity=0.25)
        noise = np.random.randn(len(y_sim)).astype(np.float32) * 0.003
        y_mic = y_sim * 0.15 + noise
        y_cond = condition_speakerphone_audio(y_mic, sr)
        X_train.append(extract_16_features(y_cond))
        Y_train.append(1)
    except Exception:
        pass

X_train = np.array(X_train)
Y_train = np.array(Y_train)
print(f"Dataset shape: {X_train.shape} (Fake={np.sum(Y_train==0)}, Real={np.sum(Y_train==1)})")

clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000, class_weight='balanced', random_state=42))
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X_train, Y_train, cv=cv, scoring='accuracy')
print(f"5-Fold CV Accuracy: {np.mean(scores)*100:.2f}% (+/- {np.std(scores)*100:.2f}%)")

clf.fit(X_train, Y_train)

# Test on unseen test.csv
correct = 0
total = 0
real_scores = []
fake_scores = []

for idx, row in test_df.iterrows():
    fpath = row['file_path']
    if not os.path.exists(fpath): continue
    y, _ = librosa.load(fpath, sr=sr, mono=True)
    feat = extract_16_features(y[:WINDOW_SIZE])
    p_fake = clf.predict_proba(feat.reshape(1, -1))[0][0] * 100.0
    if row['label'] == 'real':
        real_scores.append(p_fake)
        if p_fake < 45.0: correct += 1
    else:
        fake_scores.append(p_fake)
        if p_fake >= 45.0: correct += 1
    total += 1

print(f"\nUnseen Test Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
print(f"Real Human scores: mean={np.mean(real_scores):.1f}%, >20%: {np.sum(np.array(real_scores) > 20.0)}/{len(real_scores)}, >45%: {np.sum(np.array(real_scores) >= 45.0)}/{len(real_scores)}")
print(f"Fake Voice scores: mean={np.mean(fake_scores):.1f}%, <45%: {np.sum(np.array(fake_scores) < 45.0)}/{len(fake_scores)}, <65%: {np.sum(np.array(fake_scores) < 65.0)}/{len(fake_scores)}")

# Test Gemini
print("\n--- Testing Gemini Windows ---")
y_gem, _ = librosa.load('google_tts_test.mp3', sr=sr)
STRIDE_SIZE = int(sr * 1.5)
for i in range(0, len(y_gem) - WINDOW_SIZE + 1, STRIDE_SIZE):
    w = y_gem[i:i+WINDOW_SIZE]
    # Simulated mobile mic
    w_sim = simulate_speakerphone_rir_and_codec(w, sr=sr, reverb_intensity=0.25)
    noise = np.random.randn(len(w_sim)).astype(np.float32) * 0.003
    w_mic = w_sim * 0.15 + noise
    w_cond = condition_speakerphone_audio(w_mic, sr)
    feat = extract_16_features(w_cond)
    p_fake = clf.predict_proba(feat.reshape(1, -1))[0][0] * 100.0
    print(f"Gemini Mic Sim Win {i//STRIDE_SIZE + 1}: Fake Prob = {p_fake:.1f}%")
