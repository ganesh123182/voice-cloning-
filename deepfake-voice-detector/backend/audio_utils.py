"""
Audio preprocessing utilities.
Handles loading, resampling, and normalizing audio files for the detector model.
"""
import numpy as np
from typing import Tuple, Optional
import librosa
import soundfile as sf
import av
import wave
import struct

from .config import TARGET_SAMPLE_RATE


def load_audio(filepath: str, target_sr: int = TARGET_SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """
    Load an audio file and resample to target sample rate.
    Tries multiple backends for maximum compatibility.
    Returns (waveform_numpy_array, sample_rate).
    """
    # Try librosa first (handles most formats)
    try:
        waveform, sr = librosa.load(filepath, sr=target_sr, mono=True)
        return waveform, sr
    except Exception as e:
        print(f"[WARN] librosa load failed: {e}")

    # Try soundfile
    try:
        waveform, sr = sf.read(filepath)
        if len(waveform.shape) > 1:
            waveform = np.mean(waveform, axis=1)  # Convert to mono
        # Resample if needed
        if sr != target_sr:
            waveform = _resample(waveform, sr, target_sr)
        return waveform.astype(np.float32), target_sr
    except Exception as e:
        print(f"[WARN] soundfile load failed: {e}")

    # Try torchaudio
    try:
        import torchaudio  # pyrefly: ignore [missing-import]
        waveform, sr = torchaudio.load(filepath)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        return waveform.squeeze().numpy(), target_sr
    except ImportError:
        pass  # torchaudio not installed, skip
    except Exception as e:
        print(f"[WARN] torchaudio load failed: {e}")

    # Fallback: wave module (WAV only)
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sr = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    if sampwidth == 2:
        samples = np.array(struct.unpack(f"<{n_frames * n_channels}h", raw), dtype=np.float32) / 32768.0
    elif sampwidth == 1:
        samples = np.array(struct.unpack(f"{n_frames * n_channels}B", raw), dtype=np.float32) / 128.0 - 1.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    if sr != target_sr:
        samples = _resample(samples, sr, target_sr)

    return samples.astype(np.float32), target_sr


def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Simple linear resampling."""
    duration = len(audio) / orig_sr
    target_len = int(duration * target_sr)
    indices = np.linspace(0, len(audio) - 1, target_len)
    return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)


def get_audio_duration(waveform: np.ndarray, sr: int) -> float:
    """Get duration in seconds."""
    return len(waveform) / sr


def normalize_audio(waveform: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range."""
    max_val = np.max(np.abs(waveform))
    if max_val > 0:
        return waveform / max_val
    return waveform


def trim_silence(waveform: np.ndarray, sr: int, threshold: float = 0.01) -> np.ndarray:
    """Trim leading and trailing silence."""
    abs_wav = np.abs(waveform)

    # Find first and last non-silent sample
    above_threshold = abs_wav > threshold
    if not np.any(above_threshold):
        return waveform

    first = np.argmax(above_threshold)
    last = len(waveform) - np.argmax(above_threshold[::-1])

    # Add small padding (50ms)
    pad = int(0.05 * sr)
    first = max(0, first - pad)
    last = min(len(waveform), last + pad)

    return waveform[first:last]


def preprocess_for_model(filepath: str, max_duration: float = 30.0) -> Tuple[np.ndarray, int, float]:
    """
    Full preprocessing pipeline for the detection model.
    Returns (processed_waveform, sample_rate, duration_seconds).
    """
    waveform, sr = load_audio(filepath)
    waveform = normalize_audio(waveform)
    waveform = trim_silence(waveform, sr)

    # Truncate to max duration
    max_samples = int(max_duration * sr)
    if len(waveform) > max_samples:
        waveform = waveform[:max_samples]

    duration = get_audio_duration(waveform, sr)
    return waveform, sr, duration


def extract_features(filepath: str) -> Optional[np.ndarray]:
    """Extract a comprehensive set of audio features for deepfake detection.
    Must match the features used in train.py exactly."""
    try:
        # Robust decoding with PyAV (works for mp3, mp4, etc without system FFmpeg)
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
        
        # Resample to TARGET_SAMPLE_RATE if necessary
        if sr != TARGET_SAMPLE_RATE:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE
        
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
        print(f"Error processing {filepath}: {e}")
        return None

