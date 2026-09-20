import sys
sys.path.insert(0, '.')
import librosa
import numpy as np
from backend.audio_utils import extract_robust_vocoder_features, condition_speakerphone_audio
from backend.voice_detector import detector

y_gem, sr = librosa.load('google_tts_test.mp3', sr=16000)
WINDOW_SIZE = int(sr * 3.0)
STRIDE_SIZE = int(sr * 1.5)

feature_names = [
    'harm_mean', 'mod_ratio', 'jitter', 'pitch_std', 'crest/100',
    'harmonic_decay', 'mfcc_d1_var/50', 'mfcc_d2_var/15', 'pitch_strength',
    'spectral_flatness', 'rolloff/5000', 'centroid/4000', 'bandwidth/3000', 'zcr*5'
]

print(f"{'Feature':20s} | {'Win 2 (voc=100%)':18s} | {'Win 4 (voc=16%)':18s} | {'Win 5 (voc=1.2%)':18s} | {'Real Human':18s}")
print("-" * 105)

w2 = condition_speakerphone_audio(y_gem[STRIDE_SIZE:STRIDE_SIZE+WINDOW_SIZE], sr)
w4 = condition_speakerphone_audio(y_gem[3*STRIDE_SIZE:3*STRIDE_SIZE+WINDOW_SIZE], sr)
w5 = condition_speakerphone_audio(y_gem[4*STRIDE_SIZE:4*STRIDE_SIZE+WINDOW_SIZE], sr)

y_real, _ = librosa.load('backend/storage/enrolled_voices/72e8c2a7-422f-4974-bab0-1119c544e39b.wav', sr=16000)
w_real = condition_speakerphone_audio(y_real[:WINDOW_SIZE], sr)

f2 = extract_robust_vocoder_features(w2, sr)
f4 = extract_robust_vocoder_features(w4, sr)
f5 = extract_robust_vocoder_features(w5, sr)
f_real = extract_robust_vocoder_features(w_real, sr)

for i, name in enumerate(feature_names):
    print(f"{name:20s} | {f2[i]:18.4f} | {f4[i]:18.4f} | {f5[i]:18.4f} | {f_real[i]:18.4f}")
