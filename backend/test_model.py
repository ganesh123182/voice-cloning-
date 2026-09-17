import os
import torch
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import sys

def test_audio(filepath):
    print("=" * 30)
    print("AUDIO")
    print("=" * 30)
    y, sr = librosa.load(filepath, sr=16000)
    y_trim, _ = librosa.effects.trim(y, top_db=20)
    
    print(f"duration: {len(y)/sr:.2f}s")
    print(f"sample_rate: {sr}")
    print(f"channels: 1")
    
    print("=" * 30)
    print("WAV2VEC2")
    print("=" * 30)
    model_path = r'C:\Users\Ganesh\OneDrive\Documents\voice cloning\deepfake-voice-detector\models\wav2vec2-deepfake-finetuned'
    print(f"model: {model_path}")
    
    extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(model_path)
    
    import time
    start = time.time()
    inputs = extractor(y_trim, sampling_rate=16000, return_tensors='pt', padding='max_length', max_length=32000, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)[0].numpy()
    
    inf_ms = (time.time() - start) * 1000
    fake_prob = float(probs[0]) * 100
    real_prob = float(probs[1]) * 100
    
    print(f"logits: {logits.numpy().tolist()}")
    print(f"bonafide_probability: {real_prob:.2f}%")
    print(f"spoof_probability: {fake_prob:.2f}%")
    print(f"prediction: {'SPOOF' if fake_prob > 25 else 'BONAFIDE'}")
    print(f"inference_ms: {inf_ms:.2f} ms")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_audio(sys.argv[1])
    else:
        print("Provide a path")
