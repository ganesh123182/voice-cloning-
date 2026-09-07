"""
Deepfake Voice Detector - Custom Model Engine
Loads the custom trained Random Forest model and scaler.
"""
import os
import time
import traceback
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, Any

from .audio_utils import extract_features

class DeepfakeDetector:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_name = "Custom Random Forest MFCC"
        self._load_model()

    def _load_model(self):
        print(f"[INFO] Loading deepfake detection model...")
        base_dir = Path(__file__).resolve().parent.parent
        model_path = base_dir / "models" / "model.pkl"
        scaler_path = base_dir / "models" / "scaler.pkl"
        
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"[OK] Custom Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load model from {model_path}.")
            print(f"Error: {e}")
            self.model = None

    def detect(self, filepath: str) -> Dict[str, Any]:
        start_time = time.time()

        if self.model is None or self.scaler is None:
            return self._error_result("Model not found. Please run 'python train.py' first.", time.time() - start_time)

        try:
            # Extract features exactly as in training
            feat = extract_features(filepath)
            if feat is None:
                return self._error_result("Audio too short or could not be processed.", time.time() - start_time)

            # Scale features
            feat_scaled = self.scaler.transform([feat])
            
            # Predict (0 = REAL, 1 = FAKE)
            pred_class = self.model.predict(feat_scaled)[0]
            probs = self.model.predict_proba(feat_scaled)[0]
            
            real_prob = probs[0] * 100
            fake_prob = probs[1] * 100

            ai_probability = fake_prob
            real_probability = real_prob

            return self._build_result(
                ai_probability=ai_probability,
                real_probability=real_probability,
                mode="custom_rf",
                processing_time=time.time() - start_time
            )

        except Exception as e:
            traceback.print_exc()
            return self._error_result(str(e), time.time() - start_time)

    def _build_result(self, ai_probability: float, real_probability: float, mode: str, processing_time: float) -> Dict[str, Any]:
        if ai_probability >= 65:
            prediction = "AI_GENERATED"
            confidence = ai_probability
        elif ai_probability <= 35:
            prediction = "REAL"
            confidence = real_probability
        else:
            prediction = "UNCERTAIN"
            confidence = max(ai_probability, real_probability)

        if prediction == "AI_GENERATED":
            risk_level = "CRITICAL" if confidence >= 85 else "HIGH" if confidence >= 70 else "MEDIUM"
        elif prediction == "UNCERTAIN":
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW" if confidence >= 80 else "MEDIUM"

        explanation = []
        if prediction == "AI_GENERATED":
            explanation.append("The acoustic MFCC patterns and spectral features strongly align with known synthetic voices in our training data.")
        elif prediction == "REAL":
            explanation.append("The acoustic features exhibit natural human variability and align with real human speech samples.")
        else:
            explanation.append("The acoustic features are borderline and exhibit mixed patterns.")

        return {
            "prediction": prediction,
            "confidence": round(confidence, 1),
            "ai_probability": round(ai_probability, 1),
            "real_probability": round(real_probability, 1),
            "risk_level": risk_level,
            "explanation": explanation,
            "raw_scores": {"rf_fake": ai_probability/100, "rf_real": real_probability/100},
            "disclaimer": "Analysis is probabilistic based on our custom trained Random Forest.",
            "mode": mode,
            "model": self.model_name,
            "duration": 0.0,
            "processing_time": round(processing_time, 2),
        }

    def _error_result(self, message: str, elapsed: float) -> Dict[str, Any]:
        return {
            "prediction": "ERROR",
            "confidence": 0,
            "ai_probability": 0,
            "real_probability": 0,
            "risk_level": "UNKNOWN",
            "explanation": [f"Error: {message}"],
            "raw_scores": {},
            "disclaimer": "Analysis could not be completed.",
            "mode": "error",
            "model": "N/A",
            "duration": 0,
            "processing_time": round(elapsed, 2),
        }

    def get_status(self) -> Dict[str, Any]:
        return {
            "demo_mode": False,
            "model_name": self.model_name,
            "device": "cpu",
            "ready": self.model is not None,
        }

detector = None

def get_detector() -> DeepfakeDetector:
    global detector
    if detector is None:
        detector = DeepfakeDetector()
    return detector
