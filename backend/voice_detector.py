"""
Voice Detection Module - Analyzes audio to determine if it's AI-generated or human.

Uses a trained Random Forest model on MFCC + spectral features for accurate
deepfake voice detection. Falls back to spectral heuristics if model is unavailable.
"""
import time
import traceback
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from backend.audio_utils import convert_to_wav, read_wav, compute_spectral_features


def _extract_ml_features(filepath: str) -> Optional[np.ndarray]:
    """Extract the same feature set used during model training.
    Must match train.py / deepfake-voice-detector exactly."""
    try:
        import av
        import librosa

        TARGET_SR = 16000

        # Decode audio with PyAV (handles mp3, mp4, ogg, etc.)
        container = av.open(filepath)
        try:
            stream = container.streams.audio[0]
            sr = stream.rate

            samples = []
            for frame in container.decode(stream):
                arr = frame.to_ndarray()
                if arr.shape[0] > 1:
                    arr = np.mean(arr, axis=0)
                else:
                    arr = arr[0]
                samples.append(arr)
        finally:
            container.close()

        y = np.concatenate(samples).astype(np.float32)

        # Resample to 16 kHz if necessary
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR

        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < sr * 0.3:  # Skip files shorter than 0.3 seconds
            return None

        features = []

        # 1. MFCC (40 coefficients) - mean + std
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc.T, axis=0))  # 40
        features.extend(np.std(mfcc.T, axis=0))    # 40
        
        # 2. Delta and Delta-Delta MFCC
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(delta_mfcc.T, axis=0))  # 40
        features.extend(np.mean(delta2_mfcc.T, axis=0)) # 40
        
        # 2b. Pitch (F0) tracking to catch synthetic artifacts
        f0 = librosa.yin(y, fmin=50, fmax=400, frame_length=2048)
        features.append(np.nanmean(f0))
        features.append(np.nanstd(f0))

        # 3. Spectral Centroid
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.append(np.mean(cent))
        features.append(np.std(cent))

        # 4. Spectral Bandwidth
        bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.append(np.mean(bw))
        features.append(np.std(bw))

        # 5. Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.append(np.mean(rolloff))
        features.append(np.std(rolloff))

        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))

        # 7. Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma.T, axis=0))  # 12

        # 8. RMS Energy
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))

        # 9. Spectral Contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(contrast.T, axis=0))  # 7

        # 10. Spectral Flatness
        flatness = librosa.feature.spectral_flatness(y=y)
        features.append(np.mean(flatness))
        features.append(np.std(flatness))

        # 11. Mel spectrogram statistics
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db.T, axis=0))  # 20

        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        traceback.print_exc()
        return None


class VoiceDetector:
    """
    Detects whether an audio file is likely AI-generated or human speech.

    Primary: Trained Random Forest model on MFCC + spectral features.
    Fallback: Heuristic spectral analysis if model files are missing.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.model_name = "Custom Random Forest MFCC"
        self.use_ml = False
        self._load_model()

    def _load_model(self):
        """Try to load the trained Random Forest model from deepfake-voice-detector."""
        try:
            import joblib
        except ImportError:
            print("[WARN] joblib not installed, using heuristic detector")
            return

        # Look for model in deepfake-voice-detector/models/
        base_dir = Path(__file__).resolve().parent.parent
        model_locations = [
            base_dir / "deepfake-voice-detector" / "models",
            base_dir / "models",
        ]

        for model_dir in model_locations:
            model_path = model_dir / "model.pkl"
            scaler_path = model_dir / "scaler.pkl"
            if model_path.exists() and scaler_path.exists():
                try:
                    self.model = joblib.load(model_path)
                    self.scaler = joblib.load(scaler_path)
                    self.use_ml = True
                    print(f"[OK] ML Model loaded from {model_dir}")
                    return
                except Exception as e:
                    print(f"[ERROR] Failed to load model from {model_dir}: {e}")

        print("[WARN] No trained model found. Using heuristic spectral analysis.")
        print("       Train a model with: cd deepfake-voice-detector && python train.py")

    def analyze(self, filepath: str) -> Dict[str, Any]:
        """Analyze an audio file and return detection results."""
        start_time = time.time()

        if self.use_ml and self.model is not None and self.scaler is not None:
            return self._analyze_ml(filepath, start_time)
        else:
            return self._analyze_heuristic(filepath, start_time)

    def _analyze_ml(self, filepath: str, start_time: float) -> Dict[str, Any]:
        """Analyze using the trained Random Forest model."""
        try:
            # Extract features (same pipeline as training)
            feat = _extract_ml_features(filepath)
            if feat is None:
                return {
                    "prediction": "insufficient_data",
                    "confidence": 0,
                    "ai_probability": 0,
                    "features": {},
                    "scores": {},
                    "details": ["Audio too short or could not be processed."],
                    "disclaimer": "Audio could not be analyzed. Please provide at least 0.5 seconds of speech.",
                }

            # Scale features and predict
            feat_scaled = self.scaler.transform([feat])
            pred_class = self.model.predict(feat_scaled)[0]
            probs = self.model.predict_proba(feat_scaled)[0]

            real_prob = probs[0] * 100
            fake_prob = probs[1] * 100

            # Determine prediction and confidence
            if fake_prob >= 65:
                prediction = "likely_ai_generated"
                confidence = fake_prob
                risk_level = "CRITICAL" if fake_prob >= 85 else "HIGH" if fake_prob >= 70 else "MEDIUM"
                details = [
                    "The acoustic MFCC patterns and spectral features strongly align with known synthetic voices.",
                    f"AI probability: {fake_prob:.1f}% | Real probability: {real_prob:.1f}%",
                ]
            elif fake_prob <= 35:
                prediction = "likely_human"
                confidence = real_prob
                risk_level = "LOW" if real_prob >= 80 else "MEDIUM"
                details = [
                    "The acoustic features exhibit natural human variability and align with real human speech.",
                    f"Real probability: {real_prob:.1f}% | AI probability: {fake_prob:.1f}%",
                ]
            else:
                prediction = "inconclusive"
                confidence = max(fake_prob, real_prob)
                risk_level = "MEDIUM"
                details = [
                    "The acoustic features show mixed patterns — borderline result.",
                    f"AI probability: {fake_prob:.1f}% | Real probability: {real_prob:.1f}%",
                ]

            elapsed = round(time.time() - start_time, 2)
            return {
                "prediction": prediction,
                "confidence": round(confidence, 1),
                "ai_probability": round(fake_prob, 1),
                "risk_level": risk_level,
                "features": {
                    "model": self.model_name,
                    "mode": "ml_random_forest",
                },
                "scores": {
                    "rf_fake": round(fake_prob / 100, 3),
                    "rf_real": round(real_prob / 100, 3),
                },
                "details": details,
                "processing_time": elapsed,
                "disclaimer": (
                    "Analysis uses a trained Random Forest model on MFCC + spectral features. "
                    "Results are probabilistic. No detection system is 100% accurate."
                ),
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "prediction": "error",
                "confidence": 0,
                "ai_probability": 0,
                "features": {},
                "scores": {},
                "details": [f"ML analysis error: {str(e)}"],
                "disclaimer": f"Analysis failed: {str(e)}",
            }

    def _analyze_heuristic(self, filepath: str, start_time: float) -> Dict[str, Any]:
        """Fallback: heuristic spectral analysis when no ML model is available."""
        try:
            # Convert to WAV if needed
            wav_path = filepath
            if not filepath.lower().endswith('.wav'):
                wav_path = filepath + ".analysis.wav"
                wav_path = convert_to_wav(filepath, wav_path)

            # Read audio
            samples, sample_rate = read_wav(wav_path)

            if len(samples) < sample_rate * 0.5:
                return {
                    "prediction": "insufficient_data",
                    "confidence": 0,
                    "ai_probability": 0,
                    "features": {},
                    "scores": {},
                    "details": ["Audio duration is less than 0.5 seconds"],
                    "disclaimer": "Audio is too short for reliable analysis.",
                }

            # Compute features
            features = compute_spectral_features(samples, sample_rate)

            if "error" in features:
                return {
                    "prediction": "error",
                    "confidence": 0,
                    "ai_probability": 0,
                    "features": features,
                    "scores": {},
                    "details": [features["error"]],
                    "disclaimer": f"Could not analyze audio: {features['error']}",
                }

            # Score each dimension (higher = more likely AI)
            scores = {}
            details = []

            # 1. Spectral flatness
            flatness = features.get("spectral_flatness_mean", 0)
            flatness_std = features.get("spectral_flatness_std", 0)
            if flatness > 0.15:
                scores["spectral_flatness"] = 0.3
                details.append("High spectral flatness — more noise-like characteristics")
            elif flatness < 0.01:
                scores["spectral_flatness"] = 0.7
                details.append("Very low spectral flatness — unusually tonal, possible synthesis")
            else:
                scores["spectral_flatness"] = 0.4 + (0.05 - flatness) * 2
                details.append(f"Spectral flatness: {flatness:.4f} — within speech range")

            if flatness_std < 0.005:
                scores["spectral_flatness"] = min(scores["spectral_flatness"] + 0.15, 1.0)
                details.append("Very consistent spectral flatness — possible AI artifact")

            # 2. Energy variation
            energy_var = features.get("energy_variation", 0)
            if energy_var < 0.2:
                scores["energy_variation"] = 0.7
                details.append("Low energy variation — unnaturally consistent volume")
            elif energy_var > 0.8:
                scores["energy_variation"] = 0.3
                details.append("High energy variation — natural speech dynamics detected")
            else:
                scores["energy_variation"] = 0.5 - (energy_var - 0.4) * 0.5
                details.append(f"Energy variation: {energy_var:.3f} — moderate dynamics")

            # 3. Spectral consistency
            centroid_std = features.get("spectral_centroid_std", 0)
            centroid_mean = features.get("spectral_centroid_mean", 1)
            consistency_ratio = centroid_std / (centroid_mean + 1e-10)

            if consistency_ratio < 0.15:
                scores["spectral_consistency"] = 0.7
                details.append("Very consistent spectral centroid — possible AI synthesis")
            elif consistency_ratio > 0.4:
                scores["spectral_consistency"] = 0.3
                details.append("Natural spectral variation — consistent with human speech")
            else:
                scores["spectral_consistency"] = 0.5 - (consistency_ratio - 0.25) * 1.3
                details.append(f"Spectral consistency ratio: {consistency_ratio:.3f}")

            # 4. Bandwidth patterns
            bw_mean = features.get("spectral_bandwidth_mean", 0)
            bw_std = features.get("spectral_bandwidth_std", 0)
            bw_ratio = bw_std / (bw_mean + 1e-10)

            if bw_ratio < 0.1:
                scores["bandwidth_pattern"] = 0.65
                details.append("Narrow bandwidth variation — possible compression artifact")
            elif bw_ratio > 0.35:
                scores["bandwidth_pattern"] = 0.35
                details.append("Wide bandwidth variation — natural speech pattern")
            else:
                scores["bandwidth_pattern"] = 0.5 - (bw_ratio - 0.2) * 1.0
                details.append(f"Bandwidth variation ratio: {bw_ratio:.3f}")

            # 5. Pitch analysis
            pitch_strength = features.get("pitch_strength", 0)
            if pitch_strength > 0.7:
                scores["pitch_analysis"] = 0.55
                details.append("Strong pitch detected — clear tonal quality")
            elif pitch_strength > 0.3:
                scores["pitch_analysis"] = 0.4
                details.append("Moderate pitch strength — consistent with natural speech")
            else:
                scores["pitch_analysis"] = 0.6
                details.append("Weak pitch detection — possible synthesis or noise")

            # 6. Zero crossing rate
            zcr = features.get("zero_crossing_rate", 0)
            if zcr > 0.15:
                scores["zero_crossing"] = 0.45
                details.append("High zero-crossing rate — noisy or fricative-heavy")
            elif zcr < 0.02:
                scores["zero_crossing"] = 0.6
                details.append("Low zero-crossing rate — very tonal")
            else:
                scores["zero_crossing"] = 0.5
                details.append(f"Zero-crossing rate: {zcr:.4f} — typical range")

            # Weighted final score
            weights = {
                "spectral_flatness": 0.20,
                "energy_variation": 0.20,
                "spectral_consistency": 0.15,
                "bandwidth_pattern": 0.15,
                "pitch_analysis": 0.15,
                "zero_crossing": 0.15,
            }
            final_score = sum(scores.get(k, 0.5) * w for k, w in weights.items())

            # Convert to prediction
            if final_score > 0.55:
                prediction = "likely_ai_generated"
                confidence = min(92, (final_score - 0.5) * 200)
            elif final_score < 0.45:
                prediction = "likely_human"
                confidence = min(92, (0.5 - final_score) * 200)
            else:
                prediction = "inconclusive"
                confidence = max(10, 50 - abs(final_score - 0.5) * 200)

            confidence = max(15, min(confidence, 92))
            ai_probability = round(final_score * 100, 1)

            elapsed = round(time.time() - start_time, 2)
            return {
                "prediction": prediction,
                "confidence": round(confidence, 1),
                "ai_probability": ai_probability,
                "risk_level": "HIGH" if ai_probability > 65 else "LOW" if ai_probability < 35 else "MEDIUM",
                "features": {k: round(v, 4) if isinstance(v, float) else v for k, v in features.items()},
                "scores": {k: round(v, 3) for k, v in scores.items()},
                "details": details,
                "processing_time": elapsed,
                "disclaimer": (
                    "WARNING: This analysis uses spectral heuristics (no ML model loaded). "
                    "Results are rough estimates. Train a model for better accuracy."
                ),
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "prediction": "error",
                "confidence": 0,
                "ai_probability": 0,
                "features": {},
                "scores": {},
                "details": [f"Error during analysis: {str(e)}"],
                "disclaimer": f"Analysis failed: {str(e)}",
            }


# Singleton instance
detector = VoiceDetector()
