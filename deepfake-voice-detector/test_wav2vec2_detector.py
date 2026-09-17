import os
import time
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import librosa
from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification

os.environ["HF_HOME"] = "E:/VoiceDeepfakeAI/cache"
MODEL_PATH = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-deepfake-finetuned"
TEST_DIR = Path("E:/VoiceDeepfakeAI/dataset/test")
SR = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_audio(path, duration=2.0):
    try:
        y, sr = librosa.load(path, sr=SR)
        y, _ = librosa.effects.trim(y, top_db=20)
        return y
    except:
        return None

def main():
    print(f"Loading Model from {MODEL_PATH}...")
    if not Path(MODEL_PATH).exists():
        print("Model checkpoint not found. Did training complete?")
        return
        
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()

    y_true = []
    y_pred = []
    y_probs = []
    latencies = []

    print(f"Evaluating on independent test set in {TEST_DIR}...")
    
    # 0 = Fake, 1 = Real
    files = []
    for f in (TEST_DIR / "fake").glob("*.wav"):
        files.append((f, 0))
    for f in (TEST_DIR / "real").glob("*.wav"):
        files.append((f, 1))
        
    if not files:
        print("Test dataset is empty!")
        return

    for path, label in files:
        audio = load_audio(str(path))
        if audio is None or len(audio) < SR * 0.5:
            continue
            
        t0 = time.time()
        inputs = extractor(audio, sampling_rate=SR, return_tensors="pt", padding="max_length", max_length=int(SR*2.0), truncation=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)[0].cpu().numpy()
            
        latencies.append(time.time() - t0)
        
        # probs[0] = Fake, probs[1] = Real
        pred_label = 1 if probs[1] > probs[0] else 0
        
        y_true.append(label)
        y_pred.append(pred_label)
        y_probs.append(probs[1]) # Probability of being REAL
        
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    
    # Try ROC AUC
    try:
        auc = roc_auc_score(y_true, y_probs)
    except:
        auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # cm: [[TN, FP], [FN, TP]] -> True Fake is 0, True Real is 1
    # TN = Correctly Fake, FP = Falsely Real, FN = Falsely Fake, TP = Correctly Real
    TN, FP = cm[0]
    FN, TP = cm[1]
    
    far = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    frr = FN / (FN + TP) if (FN + TP) > 0 else 0.0
    
    report = f"""# FINAL EVALUATION REPORT

## Model Details
- **Architecture:** Wav2Vec2ForSequenceClassification
- **Checkpoint:** `{MODEL_PATH}`
- **Device:** {DEVICE}
- **Average Inference Latency:** {np.mean(latencies)*1000:.1f} ms

## Independent Test Set Results (N={len(y_true)})
- **Accuracy:** {acc*100:.1f}%
- **Precision:** {p:.4f}
- **Recall:** {r:.4f}
- **F1 Score:** {f1:.4f}
- **ROC-AUC:** {auc:.4f}

## Security Metrics
- **FAR (False Acceptance Rate - Fake classified as Real):** {far*100:.1f}%
- **FRR (False Rejection Rate - Real classified as Fake):** {frr*100:.1f}%

## Confusion Matrix
- **True FAKE (Deepfake detected):** {TN}
- **True REAL (Human detected):** {TP}
- **False REAL (Deepfake missed):** {FP}
- **False FAKE (Human flagged):** {FN}

## Final Verdict
{"PASS" if acc >= 0.8 and far < 0.2 else "FAIL"} - Model shows {"strong" if acc >= 0.8 else "weak"} generalization on unseen data.
"""

    report_path = "E:/VoiceDeepfakeAI/results/FINAL_EVALUATION_REPORT.md"
    with open(report_path, "w") as f:
        f.write(report)
        
    print(report)
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
