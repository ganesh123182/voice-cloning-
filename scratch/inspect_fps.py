import sys
sys.path.insert(0, '.')
import os
import pandas as pd
import librosa
import numpy as np
import joblib
from backend.audio_utils import extract_robust_vocoder_features, condition_speakerphone_audio

test_csv = "E:/VoiceDeepfakeAI/voice_deepfake_dataset/metadata/test.csv"
df = pd.read_csv(test_csv)
model = joblib.load("backend/models/vocoder_defense_head.joblib")

print("Checking false positives on real files...")
fps = []
for idx, row in df[df['label'] == 'real'].iterrows():
    fpath = row['file_path']
    if not os.path.exists(fpath): continue
    y, sr = librosa.load(fpath, sr=16000)
    y_cond = condition_speakerphone_audio(y, sr)
    feat = extract_robust_vocoder_features(y_cond, sr)
    prob_fake = model.predict_proba(feat.reshape(1, -1))[0][0] * 100.0
    if prob_fake >= 45.0:
        fps.append((os.path.basename(fpath), prob_fake, row['language'], feat))

print(f"Total real test files: {len(df[df['label'] == 'real'])}, False positives: {len(fps)}")
for name, prob, lang, feat in fps[:10]:
    print(f"{name:35s} ({lang:8s}): Fake Prob = {prob:.1f}%")
