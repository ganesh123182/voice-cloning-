import time
import traceback
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa

# Use the local fine-tuned Wav2Vec2 model from the user's workspace
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake-voice-detector", "models", "wav2vec2-deepfake-finetuned")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(4)

class VoiceDetector:
    def __init__(self):
        self.model = None
        self.extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing VoiceDetector on {self.device}...")
        
        # Use our fine-tuned checkpoint!
        self.MODEL_ID = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-deepfake-v2"
        self.model_name = self.MODEL_ID
        self.use_ml = False
        self.fake_idx = 0
        self.real_idx = 1
        self._load_model()

    def _load_model(self):
        try:
            print(f"[INFO] Loading final Wav2Vec2 detector from {self.model_name} on {self.device}...")
            # Use AutoFeatureExtractor from transformers
            self.extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name, low_cpu_mem_usage=True)
            self.model.to(self.device)
            self.model.eval()
            self.use_ml = True
            
            # Dynamically determine which index is fake and which is real
            if hasattr(self.model.config, 'id2label'):
                id2label = self.model.config.id2label
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if 'fake' in label_lower or 'spoof' in label_lower or 'synthetic' in label_lower:
                        self.fake_idx = int(idx)
                    if 'real' in label_lower or 'bonafide' in label_lower or 'human' in label_lower:
                        self.real_idx = int(idx)
                        
            print(f"[OK] Pretrained Wav2Vec2 Model loaded successfully. FAKE={self.fake_idx}, REAL={self.real_idx}")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load Wav2Vec2 final model: {e}")
            self.use_ml = False
            self.model = None

    def analyze(self, filepath: str) -> Dict[str, Any]:
        """Analyze an audio file using ONLY the fine-tuned Wav2Vec2 model."""
        start_time = time.time()
        
        if not self.use_ml or self.model is None:
            return {
                "prediction": "error",
                "confidence": 0,
                "ai_probability": 0,
                "risk_level": "UNKNOWN",
                "details": ["CRITICAL: ML Model is not loaded. Cannot process audio."],
                "disclaimer": "Deepfake detection requires the Wav2Vec2 model."
            }
            
        try:
            # 1. Load and Preprocess
            y, sr = librosa.load(filepath, sr=16000, mono=True)
            
            # Trim leading/trailing silence (matches train_v2.py preprocessing)
            y, _ = librosa.effects.trim(y, top_db=20)
            
            # Pad to at least 0.5s if too short
            min_length = int(16000 * 0.5)
            if len(y) < min_length:
                y = np.pad(y, (0, min_length - len(y)), mode='constant')
                
            # Limit to 4.0s (matches train_v2.py training duration)
            max_length = int(16000 * 4.0)
            if len(y) > max_length:
                y = y[:max_length]
                
            inputs = self.extractor(y, sampling_rate=16000, max_length=max_length, truncation=True, padding="max_length", return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 2. Inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
                # Standard softmax directly from the fine-tuned model (no artificial shift or temperature scaling)
                probs = torch.nn.functional.softmax(logits.float(), dim=-1)[0].cpu().numpy()
                
            raw_fake = float(probs[self.fake_idx]) if len(probs) > self.fake_idx else 0.0
            raw_real = float(probs[self.real_idx]) if len(probs) > self.real_idx else 0.0
            
            # Calibrate risk score to standard 0-100% scale:
            # - Real human baseline is centered around raw_fake ~0.30 - 0.38 (maps to 15 - 35% risk)
            # - Borderline / ambiguous region is ~0.40 - 0.44 (maps to 40 - 60% risk)
            # - Synthetic voice is >= 0.45 (maps to 70 - 99% risk)
            if raw_fake < 0.35:
                calibrated_risk = max(5.0, (raw_fake - 0.20) / (0.35 - 0.20) * 25.0)
            elif raw_fake < 0.43:
                calibrated_risk = 25.0 + (raw_fake - 0.35) / (0.43 - 0.35) * 30.0
            else:
                calibrated_risk = min(99.0, 55.0 + (raw_fake - 0.43) / (0.53 - 0.43) * 44.0)
                
            fake_prob = round(float(calibrated_risk), 1)
            real_prob = round(100.0 - fake_prob, 1)
            
            fake_logit = float(logits[0][self.fake_idx])
            real_logit = float(logits[0][self.real_idx])
            print(f"[Wav2Vec2] Raw Probs: Fake={raw_fake*100:.1f}%, Real={raw_real*100:.1f}% | Risk={fake_prob:.1f}% | Logits: Fake={fake_logit:.3f}, Real={real_logit:.3f}")
            
            # 3. Format Output
            if fake_prob >= 50:
                prediction = "likely_ai_generated"
                confidence = fake_prob
                risk_level = "CRITICAL" if fake_prob >= 85 else "HIGH" if fake_prob >= 70 else "MEDIUM"
                details = [
                    f"Wav2Vec2 classifies as SYNTHETIC ({fake_prob:.1f}% risk)",
                    f"Raw Model: Real={raw_real*100:.1f}%, Fake={raw_fake*100:.1f}% (Logits: Fake={fake_logit:.3f}, Real={real_logit:.3f})"
                ]
            else:
                prediction = "likely_human"
                confidence = real_prob
                risk_level = "LOW" if real_prob >= 70 else "MEDIUM"
                details = [
                    f"Wav2Vec2 classifies as HUMAN ({real_prob:.1f}% confidence)",
                    f"Raw Model: Real={raw_real*100:.1f}%, Fake={raw_fake*100:.1f}% (Logits: Fake={fake_logit:.3f}, Real={real_logit:.3f})"
                ]

            elapsed = round(time.time() - start_time, 2)
            
            return {
                "prediction": prediction,
                "confidence": round(confidence, 1),
                "ai_probability": round(fake_prob, 1),
                "risk_level": risk_level,
                "scores": {
                    "fake_prob": round(fake_prob / 100.0, 4),
                    "real_prob": round(real_prob / 100.0, 4),
                },
                "details": details,
                "processing_time": elapsed,
                "disclaimer": "Analysis uses the fine-tuned Wav2Vec2 deepfake detector."
            }
            
        except Exception as e:
            traceback.print_exc()
            return {
                "prediction": "error",
                "confidence": 0,
                "ai_probability": 0,
                "details": [f"Wav2Vec2 Error: {str(e)}"]
            }

detector = VoiceDetector()
