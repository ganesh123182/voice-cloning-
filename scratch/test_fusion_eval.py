import sys
sys.path.insert(0, '.')
import glob, os
import librosa
import numpy as np
import joblib
import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
from backend.audio_utils import extract_robust_vocoder_features, condition_speakerphone_audio, simulate_speakerphone_rir_and_codec

# Load models
v3_path = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-multilingual-v3"
ext = AutoFeatureExtractor.from_pretrained(v3_path)
model = Wav2Vec2ForSequenceClassification.from_pretrained(v3_path)
model.eval()

voc_model = joblib.load("backend/models/vocoder_defense_head.joblib")

def evaluate_audio(y, sr=16000):
    y_cond = condition_speakerphone_audio(y, sr)
    
    # 1. Base Wav2Vec2
    max_length = int(sr * 3.0)
    y_crop = y_cond[:max_length]
    if len(y_crop) < max_length:
        y_crop = np.pad(y_crop, (0, max_length - len(y_crop)))
    inp = ext(y_crop, sampling_rate=sr, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
    with torch.no_grad():
        out = model(**inp)
        probs = torch.softmax(out.logits.float(), dim=-1)[0].numpy()
    base_fake = float(probs[0] * 100.0)
    
    # 2. Vocoder defense head (14 acoustic features)
    ac = extract_robust_vocoder_features(y_cond, sr)
    voc_fake = float(voc_model.predict_proba(ac.reshape(1, -1))[0][0] * 100.0)
    
    # Fusion options:
    # Option A: max(base_fake, voc_fake)
    # Option B: If base_fake >= 50 or voc_fake >= 50, take max. Else weighted average.
    fused = max(base_fake, voc_fake)
    
    return base_fake, voc_fake, fused

print("=== REAL HUMAN SAMPLES ===")
for rf in glob.glob('backend/storage/enrolled_voices/*.wav')[:5]:
    y, _ = librosa.load(rf, sr=16000)
    # Clean
    b, v, f = evaluate_audio(y[:48000])
    print(f"Real (Clean)       {os.path.basename(rf)[:20]}: base={b:5.1f}%, voc={v:5.1f}% -> fused={f:5.1f}%")
    # Speakerphone RIR
    y_sim = simulate_speakerphone_rir_and_codec(y[:48000], sr=16000, reverb_intensity=0.35)
    b, v, f = evaluate_audio(y_sim)
    print(f"Real (Spk RIR)     {os.path.basename(rf)[:20]}: base={b:5.1f}%, voc={v:5.1f}% -> fused={f:5.1f}%")

print("\n=== GEMINI ASSISTANT SAMPLES ===")
y_gem, _ = librosa.load('google_tts_test.mp3', sr=16000)
for i in range(0, len(y_gem) - 48000 + 1, 24000):
    w = y_gem[i:i+48000]
    b, v, f = evaluate_audio(w)
    print(f"Gemini Clean Win {i//24000 + 1}: base={b:5.1f}%, voc={v:5.1f}% -> fused={f:5.1f}%")
    # Simulated mobile mic
    w_sim = simulate_speakerphone_rir_and_codec(w, sr=16000, reverb_intensity=0.25)
    noise = np.random.randn(len(w_sim)).astype(np.float32) * 0.003
    w_mic = w_sim * 0.15 + noise
    b, v, f = evaluate_audio(w_mic)
    print(f"Gemini Mic Sim Win {i//24000 + 1}: base={b:5.1f}%, voc={v:5.1f}% -> fused={f:5.1f}%")

print("\n=== EDGE TTS SAMPLES ===")
y_edge, _ = librosa.load('edge_tts_test.mp3', sr=16000)
for i in range(0, min(len(y_edge) - 48000 + 1, 48000*2), 24000):
    w = y_edge[i:i+48000]
    b, v, f = evaluate_audio(w)
    print(f"Edge TTS Clean Win {i//24000 + 1}: base={b:5.1f}%, voc={v:5.1f}% -> fused={f:5.1f}%")

print("\n=== VOICE CLONE (FISH AUDIO / TEST.WAV) ===")
y_clone, _ = librosa.load('test.wav', sr=16000)
b, v, f = evaluate_audio(y_clone[:48000])
print(f"Test.wav Clean:  base={b:5.1f}%, voc={v:5.1f}% -> fused={f:5.1f}%")
y_clone_sim = simulate_speakerphone_rir_and_codec(y_clone[:48000], sr=16000, reverb_intensity=0.35)
b, v, f = evaluate_audio(y_clone_sim)
print(f"Test.wav SpkRIR: base={b:5.1f}%, voc={v:5.1f}% -> fused={f:5.1f}%")
