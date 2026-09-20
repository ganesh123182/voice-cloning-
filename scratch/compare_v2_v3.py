import sys
sys.path.insert(0, '.')
import torch
import librosa
import numpy as np
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

v2_path = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-deepfake-v2"
v3_path = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-multilingual-v3"

print("Comparing v2 checkpoint vs v3 checkpoint...")

ext_v2 = AutoFeatureExtractor.from_pretrained(v2_path)
model_v2 = Wav2Vec2ForSequenceClassification.from_pretrained(v2_path)
model_v2.eval()

ext_v3 = AutoFeatureExtractor.from_pretrained(v3_path)
model_v3 = Wav2Vec2ForSequenceClassification.from_pretrained(v3_path)
model_v3.eval()

files = [
    'google_tts_test.mp3',
    'edge_tts_test.mp3',
    'test.wav',
    'voice1.wav',
    'backend/storage/enrolled_voices/72e8c2a7-422f-4974-bab0-1119c544e39b.wav'
]

def eval_model(model, ext, y, sr=16000):
    # Standard 3.0s padding/truncation as done in train
    max_length = int(sr * 3.0)
    if len(y) > max_length:
        y = y[:max_length]
    elif len(y) < max_length:
        y = np.pad(y, (0, max_length - len(y)))
    inp = ext(y, sampling_rate=sr, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
        probs = torch.softmax(out.logits.float(), dim=-1)[0].numpy()
    # Fake idx: check id2label
    fake_idx = 0
    if hasattr(model.config, 'id2label'):
        for idx, lbl in model.config.id2label.items():
            if 'fake' in lbl.lower(): fake_idx = int(idx)
    return probs[fake_idx] * 100.0, probs[1-fake_idx] * 100.0

for f in files:
    y, sr = librosa.load(f, sr=16000)
    f_v2, r_v2 = eval_model(model_v2, ext_v2, y, sr)
    f_v3, r_v3 = eval_model(model_v3, ext_v3, y, sr)
    print(f"\nFile: {f}")
    print(f"   V2: Fake={f_v2:.1f}%, Real={r_v2:.1f}%")
    print(f"   V3: Fake={f_v3:.1f}%, Real={r_v3:.1f}%")
