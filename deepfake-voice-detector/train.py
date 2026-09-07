"""
Deepfake Voice Detector - Custom Model Training (Enhanced)
This script:
1. Loads all audio files from dataset/real and dataset/fake.
2. Applies DATA AUGMENTATION to create many training samples from few files.
3. Extracts robust audio features (MFCC, Spectral, Chroma, RMS).
4. Trains a Random Forest Classifier with optimized hyperparameters.
5. Evaluates the model and saves it.
"""
import os
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import warnings
warnings.filterwarnings("ignore")

from backend.config import TARGET_SAMPLE_RATE

# Configuration
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
REAL_DIR = DATASET_DIR / "real"
FAKE_DIR = DATASET_DIR / "fake"
MODEL_DIR = BASE_DIR / "models"

# Ensure directories exist
REAL_DIR.mkdir(parents=True, exist_ok=True)
FAKE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset():
    """Disabled: Training strictly on the exact user-provided files in dataset/real and dataset/fake."""
    pass


def load_audio_robust(filepath: str, sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    """Load audio file robustly using multiple backends."""
    import av
    
    # Try PyAV first (handles mp3, mp4, etc. without system FFmpeg)
    try:
        container = av.open(filepath)
        stream = container.streams.audio[0]
        orig_sr = stream.rate
        
        samples = []
        for frame in container.decode(stream):
            arr = frame.to_ndarray()
            if arr.shape[0] > 1:
                arr = np.mean(arr, axis=0)
            else:
                arr = arr[0]
            samples.append(arr)
        
        y = np.concatenate(samples).astype(np.float32)
        
        if orig_sr != sr:
            y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)
        
        return y
    except Exception as e:
        print(f"  [WARN] PyAV failed for {filepath}: {e}")
    
    # Fallback to librosa
    try:
        y, _ = librosa.load(filepath, sr=sr, mono=True)
        return y
    except Exception as e:
        print(f"  [WARN] librosa failed for {filepath}: {e}")
    
    return None


def augment_audio(y: np.ndarray, sr: int) -> list:
    """
    Generate multiple augmented versions of an audio clip.
    This is critical when training with very few samples.
    Returns a list of (augmented_waveform, description) tuples.
    """
    augmented = []
    
    # 1. Original (full)
    augmented.append((y, "original"))
    
    # 2. Time-stretched versions
    for rate in [0.85, 0.92, 1.08, 1.15]:
        try:
            stretched = librosa.effects.time_stretch(y, rate=rate)
            augmented.append((stretched, f"time_stretch_{rate}"))
        except Exception:
            pass
    
    # 3. Pitch-shifted versions
    for steps in [-2, -1, 1, 2]:
        try:
            shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            augmented.append((shifted, f"pitch_shift_{steps}"))
        except Exception:
            pass
    
    # 4. Add white noise at different levels
    for noise_level in [0.002, 0.005, 0.01]:
        noise = np.random.normal(0, noise_level, len(y)).astype(np.float32)
        augmented.append((y + noise, f"noise_{noise_level}"))
    
    # 5. Volume variations
    for gain in [0.6, 0.8, 1.2, 1.5]:
        augmented.append((np.clip(y * gain, -1.0, 1.0).astype(np.float32), f"gain_{gain}"))
    
    # 6. Segment extractions (different parts of the audio)
    duration_samples = len(y)
    if duration_samples > sr * 2:  # Only if longer than 2 seconds
        segment_len = duration_samples // 3
        for i in range(3):
            start = i * segment_len
            end = min(start + segment_len + sr, duration_samples)  # overlap
            segment = y[start:end]
            if len(segment) > sr:  # at least 1 second
                augmented.append((segment, f"segment_{i}"))
    
    # 7. Combined augmentations (pitch + noise, stretch + noise)
    for rate in [0.9, 1.1]:
        try:
            stretched = librosa.effects.time_stretch(y, rate=rate)
            noise = np.random.normal(0, 0.003, len(stretched)).astype(np.float32)
            augmented.append((stretched + noise, f"combined_stretch_{rate}_noise"))
        except Exception:
            pass
    
    for steps in [-1, 1]:
        try:
            shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            noise = np.random.normal(0, 0.003, len(shifted)).astype(np.float32)
            augmented.append((shifted + noise, f"combined_pitch_{steps}_noise"))
        except Exception:
            pass
    
    return augmented


def extract_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Extract a comprehensive set of audio features from a waveform."""
    try:
        # Trim silence
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < sr * 0.3:  # Skip clips shorter than 0.3 seconds
            return None
        
        features = []
        
        # 1. MFCC (40 coefficients) + deltas
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.extend(np.mean(mfcc.T, axis=0))  # 40 features
        features.extend(np.std(mfcc.T, axis=0))    # 40 features (variance captures naturalness)
        
        # 2. Delta and Delta-Delta MFCC (captures temporal dynamics - important for detecting synthetic patterns)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        features.extend(np.mean(delta_mfcc.T, axis=0))  # 40 features
        features.extend(np.mean(delta2_mfcc.T, axis=0)) # 40 features
        
        # 2b. Pitch (F0) tracking to catch synthetic artifacts
        f0 = librosa.yin(y, fmin=50, fmax=400, frame_length=2048)
        features.append(np.nanmean(f0))
        features.append(np.nanstd(f0))
        
        # 3. Spectral Centroid (brightness)
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
        
        # 6. Zero Crossing Rate (texture indicator)
        zcr = librosa.feature.zero_crossing_rate(y)
        features.append(np.mean(zcr))
        features.append(np.std(zcr))
        
        # 7. Chroma features (12 pitch classes)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features.extend(np.mean(chroma.T, axis=0))  # 12 features
        
        # 8. RMS Energy
        rms = librosa.feature.rms(y=y)
        features.append(np.mean(rms))
        features.append(np.std(rms))
        
        # 9. Spectral Contrast (7 bands)
        contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        features.extend(np.mean(contrast.T, axis=0))  # 7 features
        
        # 10. Spectral Flatness (tonality vs noise - key for AI detection)
        flatness = librosa.feature.spectral_flatness(y=y)
        features.append(np.mean(flatness))
        features.append(np.std(flatness))
        
        # 11. Mel spectrogram statistics
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=20)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        features.extend(np.mean(mel_db.T, axis=0))  # 20 features
        
        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"  Feature extraction error: {e}")
        return None


def prepare_data():
    """Process all audio files with data augmentation and create the feature matrix."""
    print("\nLoading and augmenting audio files...")
    X, y_labels = [], []
    
    real_files = list(REAL_DIR.glob("*.wav")) + list(REAL_DIR.glob("*.mp3")) + list(REAL_DIR.glob("*.flac")) + list(REAL_DIR.glob("*.mp4")) + list(REAL_DIR.glob("*.m4a")) + list(REAL_DIR.glob("*.ogg"))
    fake_files = list(FAKE_DIR.glob("*.wav")) + list(FAKE_DIR.glob("*.mp3")) + list(FAKE_DIR.glob("*.flac")) + list(FAKE_DIR.glob("*.mp4")) + list(FAKE_DIR.glob("*.m4a")) + list(FAKE_DIR.glob("*.ogg"))
    
    print(f"  Found {len(real_files)} real files, {len(fake_files)} fake files")
    
    # Process REAL files (label 0)
    for file in real_files:
        print(f"  Processing REAL: {file.name}")
        audio = load_audio_robust(str(file))
        if audio is None:
            print(f"    [SKIP] Could not load {file.name}")
            continue
        
        augmented_samples = augment_audio(audio, TARGET_SAMPLE_RATE)
        for aug_audio, aug_desc in augmented_samples:
            feat = extract_features(aug_audio, TARGET_SAMPLE_RATE)
            if feat is not None:
                X.append(feat)
                y_labels.append(0)
        print(f"    Generated {len(augmented_samples)} augmented samples")
    
    # Process FAKE files (label 1)
    for file in fake_files:
        print(f"  Processing FAKE: {file.name}")
        audio = load_audio_robust(str(file))
        if audio is None:
            print(f"    [SKIP] Could not load {file.name}")
            continue
        
        augmented_samples = augment_audio(audio, TARGET_SAMPLE_RATE)
        for aug_audio, aug_desc in augmented_samples:
            feat = extract_features(aug_audio, TARGET_SAMPLE_RATE)
            if feat is not None:
                X.append(feat)
                y_labels.append(1)
        print(f"    Generated {len(augmented_samples)} augmented samples")
    
    return np.array(X), np.array(y_labels)


def main():
    print("=" * 60)
    print("  Deepfake Voice Detector - Enhanced Training Pipeline")
    print("  (with Data Augmentation for small datasets)")
    print("=" * 60)
    
    # 1. Download dataset if needed
    download_dataset()
    
    # 2. Extract features with augmentation
    X, y = prepare_data()
    if len(X) < 4:
        print("[ERROR] Not enough data to train. Please ensure at least 1 real and 1 fake audio file.")
        return
    
    print(f"\nTotal augmented samples: {len(X)} ({np.sum(y==0)} Real, {np.sum(y==1)} Fake)")
    print(f"Feature vector size: {X.shape[1]}")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 4. Train an Ensemble Model (more robust than single RF)
    print("\nTraining Ensemble Classifier (RF + Gradient Boosting)...")
    
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
        class_weight='balanced'
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42
    )
    
    # Voting ensemble for better generalization
    model = VotingClassifier(
        estimators=[('rf', rf), ('gb', gb)],
        voting='soft'  # Use probability averaging
    )
    
    model.fit(X_train_scaled, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    
    print("\n" + "=" * 40)
    print("        EVALUATION METRICS")
    print("=" * 40)
    print(f"  Validation Accuracy:  {acc*100:.2f}%")
    print(f"  Precision:            {prec*100:.2f}%")
    print(f"  Recall:               {rec*100:.2f}%")
    print(f"  F1 Score:             {f1*100:.2f}%")
    print(f"\n  Confusion Matrix (Test Set):")
    print(f"                     Predicted REAL    Predicted FAKE")
    print(f"  Actual REAL          {cm[0][0]:<15} {cm[0][1]}")
    print(f"  Actual FAKE          {cm[1][0]:<15} {cm[1][1]}")
    print("=" * 40)
    
    # 6. Save Model
    model_path = MODEL_DIR / "model.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\n[OK] Model saved to: {model_path}")
    print(f"[OK] Scaler saved to: {scaler_path}")
    print(f"\nTraining complete! You can now start the server.")

if __name__ == "__main__":
    main()
