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


def condition_speakerphone_audio(audio: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Acoustically conditions audio captured when a call is on speakerphone.
    - High-pass filter at 80 Hz eliminates phone chassis rumble & desk thump without clipping fundamental voice formants.
    - Preserves wideband speech cues up to 7800 Hz so neural vocoder synthesis artifacts are retained.
    - AGC normalizes peak amplitude to nominal speech range ~0.7.
    """
    import scipy.signal as sig
    
    y = np.asarray(audio, dtype=np.float32).copy()
    if len(y) == 0:
        return y
        
    # 1. 2nd-order High-pass filter at 80 Hz (eliminates desk thump & loudspeaker cabinet resonance)
    b_hp, a_hp = sig.butter(2, 80 / (sr / 2), btype='highpass')
    y_filtered = sig.filtfilt(b_hp, a_hp, y)
    
    # 2. Anti-aliasing high guard filter at 7800 Hz
    b_lp, a_lp = sig.butter(2, 7800 / (sr / 2), btype='lowpass')
    y_filtered = sig.filtfilt(b_lp, a_lp, y_filtered)
    
    # 3. Standardized AGC normalization (targets nominal speech peak ~0.75)
    peak = np.max(np.abs(y_filtered))
    if peak > 1e-4:
        y_filtered = (y_filtered / peak) * 0.75
        
    return y_filtered.astype(np.float32)


def classify_user_vs_caller(
    audio_chunk: np.ndarray, 
    sr: int = 16000, 
    user_energy_threshold: float = 0.030
) -> Tuple[str, float, float]:
    """
    Isolates User speech from Speakerphone Caller speech using acoustic energy profiling.
    
    Returns:
        (speaker_category, rms, peak)
        where speaker_category is:
          - 'user'    : Near-field high energy (holding phone close to mouth)
          - 'caller'  : Far-field loudspeaker acoustic leakage into the mic
          - 'silence' : Background noise / silence below active speech floor
    """
    y = np.asarray(audio_chunk, dtype=np.float32)
    if len(y) == 0:
        return "silence", 0.0, 0.0
        
    rms = float(np.sqrt(np.mean(y**2)))
    peak = float(np.max(np.abs(y)))
    
    # Genuine ambient silence threshold: quiet room background / digital silence is < 0.0022 RMS and < 0.010 Peak
    if rms < 0.0022 and peak < 0.010:
        return "silence", rms, peak
    elif rms >= user_energy_threshold or peak > 0.16:
        # Near-field user speaking directly into the phone's primary microphone
        return "user", rms, peak
    else:
        # Caller voice originating from loudspeaker and captured by microphone
        return "caller", rms, peak


def simulate_speakerphone_rir_and_codec(
    audio: np.ndarray, 
    sr: int = 16000, 
    reverb_intensity: float = 0.35
) -> np.ndarray:
    """
    Simulates acoustic speakerphone conditions for stress testing:
    1. Room Impulse Response (RIR) with realistic early wall reflections and exponential late decay.
    2. Mobile loudspeaker acoustic coupling.
    """
    import scipy.signal as sig
    
    y = np.asarray(audio, dtype=np.float32).copy()
    if len(y) == 0:
        return y
        
    # 1. Synthesize realistic Room Impulse Response (RIR)
    rir_len = int(sr * 0.15) # 150ms impulse response typical of room acoustics
    rir = np.zeros(rir_len, dtype=np.float32)
    rir[0] = 1.0 # Direct sound
    
    # Early reflections
    d1 = int(sr * 0.015)
    d2 = int(sr * 0.030)
    if d1 < rir_len: rir[d1] = 0.25 * reverb_intensity
    if d2 < rir_len: rir[d2] = 0.15 * reverb_intensity
    
    # Late exponential reverberation tail
    t = np.linspace(0, 0.15, rir_len)
    tail = np.random.randn(rir_len).astype(np.float32) * np.exp(-t / 0.05) * (0.03 * reverb_intensity)
    rir += tail
    rir = rir / np.max(np.abs(rir))
    
    # Convolve with RIR
    y_rev = sig.fftconvolve(y, rir, mode='same')
    
    # Normalize peak to match original
    peak = np.max(np.abs(y_rev))
    if peak > 1e-4:
        y_rev = y_rev / peak * min(np.max(np.abs(y)), 0.8)
        
    return y_rev.astype(np.float32)


def extract_robust_vocoder_features(y: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Extracts 12 invariant acoustic features that separate neural vocoder synthesis from human voice,
    even when captured over a smartphone microphone with room acoustics and background noise.
    """
    import librosa
    
    y = np.asarray(y, dtype=np.float32).flatten()
    if len(y) < int(sr * 0.5):
        y = np.pad(y, (0, int(sr * 0.5) - len(y)))
    elif len(y) > int(sr * 3.0):
        y = y[:int(sr * 3.0)]
        
    # Standardize amplitude peak to nominal 0.7
    peak = np.max(np.abs(y))
    if peak > 1e-4:
        y = (y / peak) * 0.7
        
    stft = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=512)
    
    # 1. Band energies & Spectral Tilt
    e_low = float(np.mean(stft[(freqs < 1000), :])) + 1e-6
    e_mid = float(np.mean(stft[(freqs >= 1000) & (freqs < 3000), :])) + 1e-6
    e_high = float(np.mean(stft[(freqs >= 3000) & (freqs < 6000), :])) + 1e-6
    e_vhigh = float(np.mean(stft[(freqs >= 6000), :])) + 1e-6
    
    hf_ratio = float(e_high / e_mid)
    vhigh_ratio = float(e_vhigh / e_mid)
    spectral_tilt = float(e_low / (e_high + 1e-6))
    
    # 2. Spectral statistics
    centroid = float(np.mean(librosa.feature.spectral_centroid(S=stft, sr=sr)))
    bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(S=stft, sr=sr)))
    rolloff = float(np.mean(librosa.feature.spectral_rolloff(S=stft, sr=sr, roll_percent=0.85)))
    flatness = float(np.mean(librosa.feature.spectral_flatness(S=stft)))
    
    # 3. Temporal envelope modulation (neural vocoder frame transitions)
    env = np.mean(stft, axis=0)
    env_diff = np.abs(np.diff(env))
    env_smoothness = float(np.std(env_diff) / (np.mean(env) + 1e-6))
    
    # 4. Zero crossing rate
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y, frame_length=512, hop_length=160)))
    
    # 5. MFCC deltas
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(stft**2 + 1e-6), sr=sr, n_mfcc=13)
    d1_var = float(np.mean(np.var(librosa.feature.delta(mfcc), axis=1)))
    d2_var = float(np.mean(np.var(librosa.feature.delta(mfcc, order=2), axis=1)))
    
    # 6. Autocorrelation Harmonic Peak Strength
    corr = np.correlate(y, y, mode='full')
    corr = corr[len(corr)//2:]
    min_lag = int(sr / 400)
    max_lag = int(sr / 70)
    if max_lag < len(corr):
        r_max = float(np.max(corr[min_lag:max_lag]) / (corr[0] + 1e-8))
    else:
        r_max = 0.0

    return np.array([
        hf_ratio,
        vhigh_ratio,
        spectral_tilt / 20.0,
        centroid / 4000.0,
        bandwidth / 3000.0,
        rolloff / 5000.0,
        flatness * 10.0,
        env_smoothness,
        zcr * 5.0,
        d1_var / 50.0,
        d2_var / 15.0,
        r_max
    ], dtype=np.float32)


