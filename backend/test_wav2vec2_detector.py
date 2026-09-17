import os
import json
import torch
import numpy as np
import librosa
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix

# Point to the E: drive where the best model was saved
MODEL_PATH = "E:/VoiceDeepfakeAI/checkpoints/wav2vec2-deepfake-finetuned"
TEST_JSON = "E:/VoiceDeepfakeAI/dataset/test.json"
RESULTS_DIR = "E:/VoiceDeepfakeAI/results"

def load_test_data():
    with open(TEST_JSON, "r") as f:
        return json.load(f)

def run_evaluation():
    print(f"Loading Model from {MODEL_PATH}...")
    from transformers import AutoFeatureExtractor, Wav2Vec2ForSequenceClassification
    
    extractor = AutoFeatureExtractor.from_pretrained(MODEL_PATH)
    model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    test_data = load_test_data()
    print(f"Found {len(test_data)} test samples.")
    
    y_true = []
    y_scores = []
    y_pred = []
    
    fake_idx = 0
    real_idx = 1
    
    print("Running Inference on Test Set...")
    for idx, item in enumerate(test_data):
        y, sr = librosa.load(item["path"], sr=16000, mono=True)
        # Pad to at least 0.5s
        min_length = int(16000 * 0.5)
        if len(y) < min_length:
            y = np.pad(y, (0, min_length - len(y)), mode='constant')
            
        # Truncate to 2s
        if len(y) > int(16000 * 2.0):
            y = y[:int(16000 * 2.0)]
            
        inputs = extractor(y, sampling_rate=16000, max_length=int(16000 * 2.0), truncation=True, padding="max_length", return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probs = torch.nn.functional.softmax(logits, dim=-1)
            fake_prob = float(probs[0][fake_idx])
            
            pred = 0 if fake_prob > 0.50 else 1
            
            y_true.append(item["label"])
            y_scores.append(fake_prob)
            y_pred.append(pred)
            
            result_str = "FAKE" if pred == 0 else "REAL"
            true_str = "FAKE" if item["label"] == 0 else "REAL"
            print(f"[{idx+1}/{len(test_data)}] True: {true_str} | Pred: {result_str} (FakeProb: {fake_prob*100:.1f}%)")
            
    # Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, pos_label=0) # Fake is positive class here
    
    # y_true is 0 for Fake, 1 for Real. 
    # y_scores is probability of being Fake.
    # To use roc_auc_score, let's map true_binary: Fake=1, Real=0
    true_binary = [1 if label == 0 else 0 for label in y_true]
    
    try:
        auc = roc_auc_score(true_binary, y_scores)
    except:
        auc = 0.5 # fallback if only one class
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # False Acceptance Rate (Real classified as Fake)
    frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0 # False Rejection Rate (Fake classified as Real)
    
    print("\n" + "="*40)
    print("FINAL EVALUATION METRICS")
    print("="*40)
    print(f"Accuracy: {acc*100:.1f}%")
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC:  {auc:.3f}")
    print(f"FAR:      {far*100:.1f}%")
    print(f"FRR:      {frr*100:.1f}%")
    
    # Save Report
    report_path = os.path.join(RESULTS_DIR, "FINAL_EVALUATION_REPORT.md")
    with open(report_path, "w") as f:
        f.write("# Final Wav2Vec2 Evaluation Report\n\n")
        f.write("This report provides independent evaluation metrics on the isolated `test` dataset split using pure softmax probability (no calibration margin scaling).\n\n")
        f.write("## Metrics\n")
        f.write(f"- **Accuracy:** {acc*100:.1f}%\n")
        f.write(f"- **F1-Score:** {f1:.3f}\n")
        f.write(f"- **ROC-AUC:** {auc:.3f}\n")
        f.write(f"- **False Acceptance Rate (FAR):** {far*100:.1f}%\n")
        f.write(f"- **False Rejection Rate (FRR):** {frr*100:.1f}%\n\n")
        
        f.write("## Confusion Matrix\n")
        f.write("| | Predicted FAKE | Predicted REAL |\n")
        f.write("|---|---|---|\n")
        f.write(f"| **Actual FAKE** | {cm[0][0]} | {cm[0][1]} |\n")
        f.write(f"| **Actual REAL** | {cm[1][0]} | {cm[1][1]} |\n")
        
    print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    run_evaluation()
