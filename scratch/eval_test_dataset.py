import sys
sys.path.insert(0, '.')
import os
import pandas as pd
import librosa
import numpy as np
from backend.voice_detector import detector

test_csv = "E:/VoiceDeepfakeAI/voice_deepfake_dataset/metadata/test.csv"
df = pd.read_csv(test_csv)

print(f"Testing on {len(df)} unseen test files...")

correct = 0
total = 0
real_scores = []
fake_scores = []

for idx, row in df.iterrows():
    fpath = row['file_path']
    label = row['label']
    if not os.path.exists(fpath): continue
    
    res = detector.analyze(fpath)
    ai_prob = res['ai_probability']
    pred = res['prediction']
    
    if label == 'real':
        real_scores.append(ai_prob)
        # Real should be < 45% (ideally <= 20%)
        if ai_prob < 45.0: correct += 1
    else:
        fake_scores.append(ai_prob)
        # Fake should be >= 45% (ideally >= 65%)
        if ai_prob >= 45.0: correct += 1
    total += 1

print(f"\nTotal tested: {total}")
print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")
print(f"Real Human scores: mean={np.mean(real_scores):.1f}%, min={np.min(real_scores):.1f}%, max={np.max(real_scores):.1f}%, >20%: {np.sum(np.array(real_scores) > 20.0)}/{len(real_scores)}, >45%: {np.sum(np.array(real_scores) >= 45.0)}/{len(real_scores)}")
print(f"Fake Voice scores: mean={np.mean(fake_scores):.1f}%, min={np.min(fake_scores):.1f}%, max={np.max(fake_scores):.1f}%, <45%: {np.sum(np.array(fake_scores) < 45.0)}/{len(fake_scores)}, <65%: {np.sum(np.array(fake_scores) < 65.0)}/{len(fake_scores)}")
