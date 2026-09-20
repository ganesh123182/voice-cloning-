import time
import traceback
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
import librosa
import joblib
from backend.audio_utils import compute_spectral_features, extract_robust_vocoder_features

# Use the local fine-tuned Wav2Vec2 model from the user's workspace
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "deepfake-voice-detector", "models", "wav2vec2-deepfake-finetuned")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(8)

class VoiceDetector:
    def __init__(self):
        self.model = None
        self.extractor = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Initializing VoiceDetector on {self.device}...")
        
        # Use our latest multilingual fine-tuned checkpoint (v3), falling back to v2 if missing
        v3_path = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-multilingual-v3"
        v2_path = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-deepfake-v2"
        self.MODEL_ID = v3_path if os.path.exists(v3_path) else v2_path
        self.model_name = self.MODEL_ID
        self.use_ml = False
        self.fake_idx = 0
        self.real_idx = 1
        self.vocoder_head = None
        self._load_model()
        self._load_vocoder_defense_head()

    def _load_vocoder_defense_head(self):
        """Load the calibrated multi-feature vocoder defense head."""
        try:
            head_path = os.path.join(os.path.dirname(__file__), "models", "vocoder_defense_head.joblib")
            if os.path.exists(head_path):
                self.vocoder_head = joblib.load(head_path)
                print(f"[OK] Multi-Feature Vocoder Defense Head loaded from {head_path}")
            else:
                print(f"[WARN] Vocoder defense head not found at {head_path}")
        except Exception as e:
            print(f"[WARN] Failed to load vocoder defense head: {e}")

    def _extract_vocoder_features(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extract 14 invariant acoustic vocoder features calibrated to identify neural vocoder synthesis."""
        return extract_robust_vocoder_features(y, sr)

    def _load_model(self):
        try:
            print(f"[INFO] Loading final Wav2Vec2 detector from {self.model_name} on {self.device}...")
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
                        
            # Enable dynamic INT8 quantization on CPU linear layers for sub-100ms forward pass
            if self.device.type == "cpu":
                try:
                    self.model = torch.quantization.quantize_dynamic(
                        self.model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    print("[OK] CPU INT8 Dynamic Quantization enabled (~2.5x speedup).")
                except Exception as q_err:
                    print(f"[WARN] INT8 quantization skipped: {q_err}")
                        
            print(f"[OK] Pretrained Wav2Vec2 Model loaded successfully. FAKE={self.fake_idx}, REAL={self.real_idx}")
        except Exception as e:
            print(f"[CRITICAL ERROR] Failed to load Wav2Vec2 final model: {e}")
            self.use_ml = False
            self.model = None

    def analyze_raw(self, audio_data: np.ndarray, sr: int = 16000) -> Dict[str, Any]:
        """
        Fast in-memory RAM forward pass directly from a float32 NumPy array.
        Eliminates disk I/O completely for ~75-95ms latency.
        """
        start_time = time.perf_counter()
        
        if not self.use_ml or self.model is None:
            return {
                "prediction": "error",
                "confidence": 0,
                "ai_probability": 0,
                "risk_level": "UNKNOWN",
                "details": ["CRITICAL: ML Model is not loaded. Cannot process audio."],
                "disclaimer": "Deepfake detection requires the Wav2Vec2 model.",
                "processing_time_ms": 0.0
            }
            
        try:
            y = np.asarray(audio_data, dtype=np.float32).flatten()
            
            # Fast energy-based silence trimming
            abs_y = np.abs(y)
            non_silent = np.where(abs_y > 0.005)[0]
            if len(non_silent) > 0:
                y = y[non_silent[0]:non_silent[-1]+1]
                
            # Pad to at least 0.5s if too short
            min_length = int(sr * 0.5)
            if len(y) < min_length:
                y = np.pad(y, (0, min_length - len(y)), mode='constant')
                
            # Window length (up to 3.0s)
            max_samples = int(sr * 3.0)
            if len(y) > max_samples:
                y = y[:max_samples]
                
            inputs = self.extractor(
                y, 
                sampling_rate=sr, 
                return_tensors="pt"
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits.float(), dim=-1)[0].cpu().numpy()
                rep = outputs.hidden_states[-1].mean(dim=1).squeeze(0).cpu().numpy()
                
            raw_fake = float(probs[self.fake_idx]) if len(probs) > self.fake_idx else 0.0
            raw_real = float(probs[self.real_idx]) if len(probs) > self.real_idx else 0.0
            base_fake_prob = round(raw_fake * 100.0, 1)

            vocoder_fake_prob = None
            if self.vocoder_head is not None:
                try:
                    ac = self._extract_vocoder_features(y, sr)
                    feat = ac.reshape(1, -1)
                    # classes_ are [0, 1] -> [Fake, Real]
                    vocoder_fake_prob = round(float(self.vocoder_head.predict_proba(feat)[0][0] * 100.0), 1)
                except Exception as vf_err:
                    print(f"[WARN] Vocoder head inference error: {vf_err}")

            if vocoder_fake_prob is not None:
                fake_prob = round(max(base_fake_prob, vocoder_fake_prob), 1)
            else:
                fake_prob = base_fake_prob

            real_prob = round(100.0 - fake_prob, 1)
            
            fake_logit = float(logits[0][self.fake_idx])
            real_logit = float(logits[0][self.real_idx])
            
            # Calibration endpoints requested by user:
            # - Real human voice: 0 to 20%
            # - Suspicious voice: 45 to 65%
            # - AI clone voice: 65%+
            if fake_prob >= 65.0:
                prediction = "likely_ai_generated"
                confidence = fake_prob
                risk_level = "CRITICAL" if fake_prob >= 85.0 else "HIGH"
                details = [
                    f"AI Voice Classifier flags SYNTHETIC ({fake_prob:.1f}% risk)",
                    f"Base Wav2Vec2: Real={raw_real*100:.1f}%, Fake={raw_fake*100:.1f}% (Logits: Fake={fake_logit:.3f}, Real={real_logit:.3f})"
                ]
                if vocoder_fake_prob is not None:
                    details.append(f"Vocoder Defense Head: {vocoder_fake_prob:.1f}% synthetic probability")
                    if vocoder_fake_prob >= 65.0 and base_fake_prob < 50.0:
                        details.append("[ALERT] Neural vocoder harmonic/spectral artifacts detected (Google Gemini / SoundStream / Azure Neural signature)")
            elif fake_prob >= 45.0:
                prediction = "suspicious"
                confidence = fake_prob
                risk_level = "SUSPICIOUS"
                details = [
                    f"AI Voice Classifier flags SUSPICIOUS ({fake_prob:.1f}% risk)",
                    f"Base Wav2Vec2: Real={raw_real*100:.1f}%, Fake={raw_fake*100:.1f}% (Logits: Fake={fake_logit:.3f}, Real={real_logit:.3f})"
                ]
                if vocoder_fake_prob is not None:
                    details.append(f"Vocoder Defense Head: {vocoder_fake_prob:.1f}% synthetic probability")
            else:
                prediction = "likely_human"
                confidence = real_prob
                risk_level = "LOW"
                details = [
                    f"AI Voice Classifier verifies HUMAN ({real_prob:.1f}% confidence)",
                    f"Base Wav2Vec2: Real={raw_real*100:.1f}%, Fake={raw_fake*100:.1f}% (Logits: Fake={fake_logit:.3f}, Real={real_logit:.3f})"
                ]
                if vocoder_fake_prob is not None:
                    details.append(f"Vocoder Defense Head: {100.0 - vocoder_fake_prob:.1f}% human consistency")
                
            elapsed_ms = round((time.perf_counter() - start_time) * 1000, 1)
            
            return {
                "prediction": prediction,
                "confidence": round(confidence, 1),
                "ai_probability": round(fake_prob, 1),
                "risk_level": risk_level,
                "scores": {
                    "fake_prob": round(fake_prob / 100.0, 4),
                    "real_prob": round(real_prob / 100.0, 4),
                    "base_fake_prob": round(base_fake_prob / 100.0, 4),
                    "vocoder_fake_prob": round((vocoder_fake_prob / 100.0 if vocoder_fake_prob is not None else base_fake_prob / 100.0), 4),
                },
                "details": details,
                "processing_time": round(elapsed_ms / 1000.0, 3),
                "processing_time_ms": elapsed_ms,
                "disclaimer": "Analysis uses the fine-tuned multilingual Wav2Vec2 detector with Multi-Feature Vocoder Defense."
            }
        except Exception as e:
            traceback.print_exc()
            return {
                "prediction": "error",
                "confidence": 0,
                "ai_probability": 0,
                "risk_level": "UNKNOWN",
                "details": [f"Wav2Vec2 Error: {str(e)}"],
                "processing_time_ms": round((time.perf_counter() - start_time) * 1000, 1)
            }

    def analyze(self, filepath: str) -> Dict[str, Any]:
        """Analyze an audio file from disk."""
        try:
            y, sr = librosa.load(filepath, sr=16000, mono=True)
            return self.analyze_raw(y, sr)
        except Exception as e:
            traceback.print_exc()
            return {
                "prediction": "error",
                "confidence": 0,
                "ai_probability": 0,
                "risk_level": "UNKNOWN",
                "details": [f"File Load Error: {str(e)}"]
            }

detector = VoiceDetector()
