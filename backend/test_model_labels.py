import torch
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

model_name = "garystafford/wav2vec2-deepfake-voice-detector"
print(f"Loading {model_name}...")
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
print("ID2LABEL:")
print(model.config.id2label)
