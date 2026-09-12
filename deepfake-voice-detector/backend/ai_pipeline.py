import torch
import torchaudio
import numpy as np
import librosa
import logging
from typing import Dict, Union, List
import os

try:
    from speechbrain.inference.speaker import SpeakerRecognition
except ImportError:
    logging.warning("speechbrain is not installed. Please install it using `pip install speechbrain`")
    SpeakerRecognition = None

# Initialize logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- VAD Initialization ---
try:
    # Load Silero VAD
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                                      model='silero_vad',
                                      force_reload=False,
                                      trust_repo=True)
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
except Exception as e:
    logger.error(f"Failed to load Silero VAD: {e}")
    vad_model = None


def is_speech(audio_tensor: torch.Tensor, threshold: float = 0.5) -> bool:
    """
    Detects if there is speech in the 16kHz audio tensor using Silero VAD.
    """
    if vad_model is None or get_speech_timestamps is None:
        logger.warning("VAD model not loaded, assuming speech is present.")
        return True
    
    try:
        # Silero VAD expects 1D tensor for a single channel
        if audio_tensor.ndim > 1:
            audio_tensor = audio_tensor.squeeze()
        
        # Ensure it's float32
        audio_tensor = audio_tensor.to(torch.float32)
        
        # get_speech_timestamps processes the full 3-second chunk seamlessly
        # rather than manually chunking into 512-sample blocks
        speech_timestamps = get_speech_timestamps(audio_tensor, vad_model, sampling_rate=16000, threshold=threshold)
        
        # If any speech segment is detected, return True
        return len(speech_timestamps) > 0
    except Exception as e:
        logger.error(f"Error in VAD: {e}")
        return False


# --- Speaker Biometrics (Voice ID) ---
spk_model = None
def _load_spk_model():
    global spk_model
    if SpeakerRecognition is not None and spk_model is None:
        try:
            import os
            # Fix Windows symlink issue: force HuggingFace to copy instead of symlink
            os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
            os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")
            
            # Use absolute path for savedir to avoid relative path issues
            savedir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp_speechbrain")
            os.makedirs(savedir, exist_ok=True)
            
            # Attempt to load with symlinks disabled via huggingface_hub config
            try:
                from huggingface_hub import constants as hf_constants
                hf_constants.HF_HUB_ENABLE_HF_TRANSFER = False
                # Force local strategy (copy files, no symlinks)
                if hasattr(hf_constants, 'HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD'):
                    hf_constants.HF_HUB_LOCAL_DIR_AUTO_SYMLINK_THRESHOLD = float('inf')
            except Exception:
                pass  # Older huggingface_hub versions may not have these
            
            spk_model = SpeakerRecognition.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb", 
                savedir=savedir,
                run_opts={"symlink_strategy": "copy"}
            )
            logger.info("SpeechBrain ECAPA-TDNN speaker model loaded successfully!")
        except Exception as e:
            logger.error(f"Failed to load SpeechBrain ECAPA-TDNN: {e}")
            logger.warning("Speaker verification will use deterministic fallback.")

def extract_embedding(audio_tensor: torch.Tensor) -> np.ndarray:
    """
    Extracts a 192-dimensional normalized embedding vector using ECAPA-TDNN.
    """
    _load_spk_model()
    if spk_model is None:
        logger.warning("Speaker model not loaded. Returning deterministic embedding for fallback.")
        # Hash the tensor so the same audio returns the same embedding for testing
        tensor_sum = int(torch.sum(audio_tensor).item() * 1000)
        np.random.seed(tensor_sum % 2**32)
        emb = np.random.randn(192).astype(np.float32)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb
    
    try:
        # ECAPA-TDNN expects shape [batch, time]
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
            
        audio_tensor = audio_tensor.to(torch.float32)
        
        with torch.no_grad():
            embeddings = spk_model.encode_batch(audio_tensor)
            # embeddings shape is usually [batch, 1, embedding_dim]
            emb = embeddings.squeeze().cpu().numpy()
            
        # Normalize
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
            
        return emb
    except Exception as e:
        logger.error(f"Error extracting embedding: {e}")
        return np.zeros(192, dtype=np.float32)

def calculate_similarity(emb1: Union[np.ndarray, List[float]], emb2: Union[np.ndarray, List[float]]) -> float:
    """
    Computes Cosine Similarity between two embeddings (-1.0 to 1.0).
    """
    try:
        e1 = np.array(emb1, dtype=np.float32)
        e2 = np.array(emb2, dtype=np.float32)
        
        dot_product = np.dot(e1, e2)
        norm_product = np.linalg.norm(e1) * np.linalg.norm(e2)
        
        if norm_product == 0:
            return 0.0
            
        return float(dot_product / norm_product)
    except Exception as e:
        logger.error(f"Error calculating similarity: {e}")
        return 0.0

def is_user_speaking(chunk_embedding: np.ndarray, enrolled_embedding: np.ndarray, threshold: float = 0.75) -> bool:
    """
    Verifies if the current speaker matches the enrolled user.
    """
    similarity = calculate_similarity(chunk_embedding, enrolled_embedding)
    return similarity >= threshold


# --- Deepfake Detection Engine (Wav2Vec2 Powered) ---

# Lazy-loaded Wav2Vec2 model globals
_w2v_model = None
_w2v_processor = None
_w2v_load_attempted = False

def _load_wav2vec2():
    """Lazily load the pretrained Wav2Vec2 deepfake detector from HuggingFace."""
    global _w2v_model, _w2v_processor, _w2v_load_attempted
    if _w2v_load_attempted:
        return
    _w2v_load_attempted = True
    try:
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        MODEL_ID = "garystafford/wav2vec2-deepfake-voice-detector"
        logger.info(f"Loading Wav2Vec2 deepfake model: {MODEL_ID}...")
        _w2v_processor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_ID)
        _w2v_model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_ID)
        _w2v_model.eval()
        logger.info("Wav2Vec2 deepfake model loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load Wav2Vec2 model: {e}")
        logger.warning("Falling back to spectral feature analysis for deepfake detection.")
        _w2v_model = None
        _w2v_processor = None


class DeepfakeInferenceEngine:
    """
    Production deepfake audio detector.
    Primary: Wav2Vec2ForSequenceClassification (garystafford/wav2vec2-deepfake-voice-detector)
    Fallback: Spectral MFCC analysis if the transformer model fails to load.
    """
    def __init__(self):
        self.sample_rate = 16000
        # Trigger lazy loading of the Wav2Vec2 model
        _load_wav2vec2()
        self.use_wav2vec2 = (_w2v_model is not None and _w2v_processor is not None)
        if self.use_wav2vec2:
            logger.info("DeepfakeInferenceEngine initialized with Wav2Vec2 (REAL model).")
        else:
            logger.warning("DeepfakeInferenceEngine initialized with spectral fallback.")

    def extract_spectral_features(self, audio_numpy: np.ndarray) -> np.ndarray:
        """
        Extracts MFCC, Spectral Centroid, and Roll-off features.
        Used as fallback when Wav2Vec2 is unavailable.
        """
        try:
            mfcc = librosa.feature.mfcc(y=audio_numpy, sr=self.sample_rate, n_mfcc=20)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            cent = librosa.feature.spectral_centroid(y=audio_numpy, sr=self.sample_rate)
            cent_mean = np.mean(cent)
            
            rolloff = librosa.feature.spectral_rolloff(y=audio_numpy, sr=self.sample_rate)
            rolloff_mean = np.mean(rolloff)
            
            features = np.concatenate([mfcc_mean, [cent_mean, rolloff_mean]])
            return features
        except Exception as e:
            logger.error(f"Error in spectral feature extraction: {e}")
            return np.zeros(22, dtype=np.float32)

    def _predict_wav2vec2(self, audio_numpy: np.ndarray) -> Dict[str, Union[bool, float]]:
        """
        Run inference through the pretrained Wav2Vec2 deepfake classifier.
        Returns: {is_fake: bool, fake_prob: float (0-1), confidence: float (0-1)}
        """
        # Ensure minimum length (~0.5 sec) for the model to work
        min_samples = int(self.sample_rate * 0.5)
        if len(audio_numpy) < min_samples:
            # Pad with zeros if too short
            audio_numpy = np.pad(audio_numpy, (0, min_samples - len(audio_numpy)))
        
        # Process through the feature extractor
        inputs = _w2v_processor(
            audio_numpy, 
            sampling_rate=self.sample_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        with torch.no_grad():
            logits = _w2v_model(**inputs).logits
        
        # Apply softmax to get probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).squeeze()
        
        # Model labels: typically index 0 = "bonafide"/"real", index 1 = "spoof"/"fake"
        # Check the model's config to confirm label mapping
        id2label = getattr(_w2v_model.config, 'id2label', {0: 'bonafide', 1: 'spoof'})
        
        # Find which index corresponds to fake/spoof
        fake_idx = None
        real_idx = None
        for idx, label in id2label.items():
            label_lower = str(label).lower()
            if 'spoof' in label_lower or 'fake' in label_lower:
                fake_idx = int(idx)
            elif 'bonafide' in label_lower or 'real' in label_lower:
                real_idx = int(idx)
        
        # Default mapping if labels aren't clear
        if fake_idx is None:
            fake_idx = 1
        if real_idx is None:
            real_idx = 0
        
        fake_prob = float(probs[fake_idx].item()) if len(probs.shape) > 0 else 0.0
        real_prob = float(probs[real_idx].item()) if len(probs.shape) > 0 else 1.0
        
        is_fake = fake_prob > 0.5
        confidence = float(abs(fake_prob - 0.5) * 2.0)  # 0.0 to 1.0 scale
        
        return {
            "is_fake": is_fake,
            "fake_prob": fake_prob,
            "confidence": confidence,
            "real_prob": real_prob,
            "engine": "wav2vec2"
        }

    def _predict_spectral_fallback(self, audio_numpy: np.ndarray) -> Dict[str, Union[bool, float]]:
        """
        Fallback detector using spectral features when Wav2Vec2 is unavailable.
        Uses MFCC variance analysis — real speech has higher micro-variation
        than synthetic speech which tends to be unnaturally smooth.
        """
        features = self.extract_spectral_features(audio_numpy)
        
        # Real speech has more variance in MFCC coefficients than synthetic speech
        mfcc = librosa.feature.mfcc(y=audio_numpy, sr=self.sample_rate, n_mfcc=20)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Natural speech typically has higher std deviation in higher-order MFCCs
        high_mfcc_variance = np.mean(mfcc_std[10:])  # Higher coefficients
        low_mfcc_variance = np.mean(mfcc_std[:5])     # Lower coefficients
        
        # Synthetic audio tends to have unnaturally low variance in higher MFCCs
        variance_ratio = high_mfcc_variance / (low_mfcc_variance + 1e-8)
        
        # Spectral flatness: synthetic audio often has different flatness profile
        flatness = librosa.feature.spectral_flatness(y=audio_numpy)
        mean_flatness = float(np.mean(flatness))
        
        # Heuristic scoring (tuned to approximate real vs fake patterns)
        score = 0.5  # Start neutral
        if variance_ratio < 0.15:
            score += 0.2  # Low high-MFCC variance suggests synthetic
        if mean_flatness > 0.1:
            score += 0.1  # Higher flatness can indicate synthesis artifacts
        if variance_ratio > 0.3:
            score -= 0.2  # High variance suggests natural speech
        
        fake_prob = float(np.clip(score, 0.05, 0.95))
        is_fake = fake_prob > 0.5
        confidence = float(abs(fake_prob - 0.5) * 2.0)
        
        return {
            "is_fake": is_fake,
            "fake_prob": fake_prob,
            "confidence": confidence,
            "real_prob": 1.0 - fake_prob,
            "engine": "spectral_fallback"
        }

    def predict_chunk(self, audio_tensor: torch.Tensor) -> Dict[str, Union[bool, float]]:
        """
        Takes a 16kHz audio tensor and outputs deepfake probability.
        Uses Wav2Vec2 if available, otherwise falls back to spectral analysis.
        """
        try:
            audio_numpy = audio_tensor.squeeze().cpu().numpy().astype(np.float32)
            
            if self.use_wav2vec2:
                return self._predict_wav2vec2(audio_numpy)
            else:
                return self._predict_spectral_fallback(audio_numpy)
        except Exception as e:
            logger.error(f"Error in Deepfake inference: {e}")
            return {
                "is_fake": False,
                "fake_prob": 0.0,
                "confidence": 0.0,
                "real_prob": 1.0,
                "engine": "error"
            }


# --- Temporal Smoothing (Anti-Flicker) ---
class RollingRiskScore:
    """
    Maintains an Exponential Moving Average (EMA) over caller chunks 
    to prevent false positive alert flickering.
    """
    def __init__(self, alpha: float = 0.4):
        self.alpha = alpha  # Smoothing factor (0 < alpha <= 1)
        self.current_ema = None

    def update(self, new_score: float) -> float:
        """
        Updates the EMA with the new chunk risk score and returns the smoothed score.
        """
        if self.current_ema is None:
            self.current_ema = new_score
        else:
            self.current_ema = (self.alpha * new_score) + ((1 - self.alpha) * self.current_ema)
        
        return float(self.current_ema)
    
    def reset(self):
        """Resets the rolling average (e.g., when a new call starts)."""
        self.current_ema = None
