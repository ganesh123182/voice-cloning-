import sys
sys.path.insert(0, '.')
import joblib
import numpy as np

head = joblib.load('backend/models/vocoder_defense_head.joblib')
# The pipeline is StandardScaler -> LogisticRegression
scaler = head.steps[0][1]
clf = head.steps[1][1]

coefs = clf.coef_[0] # shape (782,)
w2v_coefs = coefs[:768]
ac_coefs = coefs[768:]

print(f"Total features: {len(coefs)}")
print(f"Wav2Vec2 weights (768): mean_abs={np.mean(np.abs(w2v_coefs)):.4f}, max_abs={np.max(np.abs(w2v_coefs)):.4f}, sum_abs={np.sum(np.abs(w2v_coefs)):.4f}")
print(f"Acoustic weights (14):  mean_abs={np.mean(np.abs(ac_coefs)):.4f}, max_abs={np.max(np.abs(ac_coefs)):.4f}, sum_abs={np.sum(np.abs(ac_coefs)):.4f}")

feature_names = [
    'harm_mean', 'mod_ratio', 'jitter', 'pitch_std', 'crest/100',
    'harmonic_decay', 'mfcc_d1_var/50', 'mfcc_d2_var/15', 'pitch_strength',
    'spectral_flatness', 'rolloff/5000', 'centroid/4000', 'bandwidth/3000', 'zcr*5'
]
print("\nAcoustic feature coefficients (negative means class 0/Fake, positive means class 1/Real):")
for name, w in zip(feature_names, ac_coefs):
    print(f"  {name:20s}: {w:+.4f}")
