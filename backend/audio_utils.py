"""
Audio utility functions for processing, converting, and analyzing audio files.
"""
import wave
import struct
import math
import os
import io
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from backend.config import SAMPLE_RATE, UPLOAD_DIR, GENERATED_DIR


def read_wav(filepath: str) -> Tuple[np.ndarray, int]:
    """Read a WAV file and return (samples_array, sample_rate)."""
    with wave.open(filepath, 'rb') as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sampwidth == 1:
        fmt = f"{n_frames * n_channels}B"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 128.0 - 1.0
    elif sampwidth == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 32768.0
    elif sampwidth == 4:
        fmt = f"<{n_frames * n_channels}i"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width: {sampwidth}")

    # Convert to mono if stereo
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    return samples, framerate


def write_wav(filepath: str, samples: np.ndarray, sample_rate: int = SAMPLE_RATE):
    """Write a numpy array of samples to a WAV file."""
    # Normalize to [-1, 1]
    if samples.max() > 1.0 or samples.min() < -1.0:
        samples = samples / max(abs(samples.max()), abs(samples.min()))

    # Convert to 16-bit PCM
    int_samples = (samples * 32767).astype(np.int16)

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(int_samples.tobytes())


def convert_to_wav(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Convert an audio file to WAV format.
    Uses PyAV (preferred, handles mp3/mp4/etc) or pydub as fallback.
    """
    if output_path is None:
        output_path = str(Path(input_path).with_suffix('.wav'))

    if input_path.lower().endswith('.wav'):
        if input_path != output_path:
            import shutil
            shutil.copy2(input_path, output_path)
        return output_path

    # Try PyAV first (handles mp3, mp4, ogg, etc. without system ffmpeg)
    try:
        import av
        container = av.open(input_path)
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

        audio_data = np.concatenate(samples).astype(np.float32)

        # Normalize to [-1, 1]
        max_val = np.max(np.abs(audio_data))
        if max_val > 0:
            audio_data = audio_data / max_val

        # Write as 16-bit WAV
        int_samples = (audio_data * 32767).astype(np.int16)
        with wave.open(output_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(int_samples.tobytes())

        return output_path
    except ImportError:
        pass  # PyAV not installed, try next
    except Exception as e:
        print(f"[WARN] PyAV conversion failed: {e}")

   
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        audio.export(output_path, format="wav")
        return output_path
    except ImportError:
        raise RuntimeError(
            "Neither PyAV nor pydub is available for audio conversion. "
            "Install with: pip install av  OR  pip install pydub"
        )


def get_audio_duration(filepath: str) -> float:
    """Get the duration of a WAV file in seconds."""
    try:
        with wave.open(filepath, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
    except Exception:
        return 0.0


def compute_spectral_features(samples: np.ndarray, sample_rate: int) -> dict:
    """
    Compute spectral features from audio samples for voice detection.
    Returns a dictionary of features useful for AI vs human classification.
    """
    # Ensure we have enough samples
    if len(samples) < 1024:
        return {"error": "Audio too short for analysis"}

    features = {}

    # 1. Overall statistics
    features["rms_energy"] = float(np.sqrt(np.mean(samples ** 2)))
    features["zero_crossing_rate"] = float(
        np.sum(np.abs(np.diff(np.sign(samples)))) / (2 * len(samples))
    )

    # 2. Spectral analysis using FFT
    n_fft = 2048
    hop_length = 512
    n_frames = max(1, (len(samples) - n_fft) // hop_length + 1)

    spectral_centroids = []
    spectral_bandwidths = []
    spectral_rolloffs = []
    spectral_flatnesses = []

    for i in range(min(n_frames, 200)):  # Limit frames for performance
        start = i * hop_length
        frame = samples[start:start + n_fft]
        if len(frame) < n_fft:
            frame = np.pad(frame, (0, n_fft - len(frame)))

        # Apply Hanning window
        window = np.hanning(n_fft)
        windowed = frame * window

        # FFT
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        # Avoid division by zero
        spectrum_sum = spectrum.sum()
        if spectrum_sum == 0:
            continue

        # Spectral centroid
        centroid = np.sum(freqs * spectrum) / spectrum_sum
        spectral_centroids.append(centroid)

        # Spectral bandwidth
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * spectrum) / spectrum_sum)
        spectral_bandwidths.append(bandwidth)

        # Spectral rolloff (85th percentile)
        cumsum = np.cumsum(spectrum)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff_freq = freqs[min(rolloff_idx, len(freqs) - 1)]
        spectral_rolloffs.append(rolloff_freq)

        # Spectral flatness (geometric mean / arithmetic mean)
        log_spectrum = np.log(spectrum + 1e-10)
        geo_mean = np.exp(np.mean(log_spectrum))
        arith_mean = np.mean(spectrum)
        flatness = geo_mean / (arith_mean + 1e-10)
        spectral_flatnesses.append(flatness)

    if not spectral_centroids:
        return {"error": "Could not compute spectral features"}

    features["spectral_centroid_mean"] = float(np.mean(spectral_centroids))
    features["spectral_centroid_std"] = float(np.std(spectral_centroids))
    features["spectral_bandwidth_mean"] = float(np.mean(spectral_bandwidths))
    features["spectral_bandwidth_std"] = float(np.std(spectral_bandwidths))
    features["spectral_rolloff_mean"] = float(np.mean(spectral_rolloffs))
    features["spectral_rolloff_std"] = float(np.std(spectral_rolloffs))
    features["spectral_flatness_mean"] = float(np.mean(spectral_flatnesses))
    features["spectral_flatness_std"] = float(np.std(spectral_flatnesses))

    # 3. Pitch variation (autocorrelation method)
    # Human voices have more natural pitch variation
    frame_size = min(4096, len(samples))
    mid_start = len(samples) // 2 - frame_size // 2
    frame = samples[mid_start:mid_start + frame_size]
    autocorr = np.correlate(frame, frame, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    autocorr = autocorr / (autocorr[0] + 1e-10)

    # Find peaks in autocorrelation (pitch periods)
    min_period = int(sample_rate / 500)  # Max 500 Hz
    max_period = int(sample_rate / 50)   # Min 50 Hz

    if max_period < len(autocorr):
        pitch_region = autocorr[min_period:max_period]
        if len(pitch_region) > 0:
            features["pitch_strength"] = float(np.max(pitch_region))
        else:
            features["pitch_strength"] = 0.0
    else:
        features["pitch_strength"] = 0.0

    # 4. Temporal variation - how much the energy changes over time
    frame_energies = []
    frame_len = sample_rate // 20  # 50ms frames
    for i in range(0, len(samples) - frame_len, frame_len):
        e = np.sqrt(np.mean(samples[i:i + frame_len] ** 2))
        frame_energies.append(e)

    if frame_energies:
        frame_energies = np.array(frame_energies)
        features["energy_variation"] = float(np.std(frame_energies) / (np.mean(frame_energies) + 1e-10))
        features["energy_range"] = float(np.max(frame_energies) - np.min(frame_energies))
    else:
        features["energy_variation"] = 0.0
        features["energy_range"] = 0.0

    return features


def generate_demo_speech(text: str, reference_path: Optional[str] = None) -> Tuple[np.ndarray, int]:
    """
    Generate demo speech audio from text.
    This creates a synthetic waveform that sounds like a simple tone-based
    speech synthesis (clearly labelled as demo).
    It does NOT pretend to be real AI voice cloning.
    """
    sample_rate = SAMPLE_RATE
    words = text.split()
    if not words:
        words = ["demo"]

    # Generate a simple speech-like signal
    all_samples = []

    # Base frequency (will be modulated per word)
    base_freq = 150  # Hz, typical voice range

    # If we have a reference file, try to extract some characteristics
    if reference_path and os.path.exists(reference_path):
        try:
            ref_samples, ref_rate = read_wav(reference_path)
            # Extract rough pitch from reference
            frame = ref_samples[:min(4096, len(ref_samples))]
            autocorr = np.correlate(frame, frame, mode='full')
            autocorr = autocorr[len(autocorr) // 2:]
            min_period = int(ref_rate / 400)
            max_period = int(ref_rate / 80)
            if max_period < len(autocorr) and min_period < max_period:
                pitch_region = autocorr[min_period:max_period]
                if len(pitch_region) > 0:
                    peak_idx = np.argmax(pitch_region) + min_period
                    if peak_idx > 0:
                        base_freq = ref_rate / peak_idx
        except Exception:
            pass

    for i, word in enumerate(words):
        # Duration per word (100-200ms)
        word_duration = 0.1 + 0.02 * len(word)
        n_samples = int(word_duration * sample_rate)
        t = np.linspace(0, word_duration, n_samples, endpoint=False)

        # Create a speech-like signal with harmonics
        freq = base_freq * (1 + 0.1 * np.sin(2 * np.pi * 3 * t))  # Vibrato
        phase = 2 * np.pi * np.cumsum(freq) / sample_rate

        # Fundamental + harmonics
        signal = np.sin(phase) * 0.5
        signal += np.sin(2 * phase) * 0.25
        signal += np.sin(3 * phase) * 0.125
        signal += np.sin(4 * phase) * 0.0625

        # Formant-like envelope
        formant1 = np.exp(-((t * sample_rate - n_samples * 0.3) ** 2) / (n_samples * 50))
        formant2 = np.exp(-((t * sample_rate - n_samples * 0.6) ** 2) / (n_samples * 80))
        envelope = 0.6 * formant1 + 0.4 * formant2

        # Apply envelope
        signal *= envelope

        # Add slight noise for realism
        signal += np.random.randn(n_samples) * 0.01

        all_samples.append(signal)

        # Add gap between words
        gap_duration = 0.05 + random.random() * 0.05
        gap_samples = int(gap_duration * sample_rate)
        all_samples.append(np.zeros(gap_samples))

    result = np.concatenate(all_samples).astype(np.float32)

    # Normalize
    max_val = np.max(np.abs(result))
    if max_val > 0:
        result = result / max_val * 0.8

    return result, sample_rate
