import json
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

JSONL_PATH = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/drowsiness_risk_10_clean.jsonl"
THRESHOLDS = np.arange(0.0, 1.01, 0.05)

scores, labels = [], []

with open(JSONL_PATH, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        sample = json.loads(line)
        scores.append(float(sample["neuralnetwork_score"]))
        labels.append(int(sample["label"]))

scores = np.asarray(scores)
labels = np.asarray(labels)

print(f"Total samples: {len(labels)}")
print(f"{'Threshold':<12} {'Accuracy':<12} {'Recall':<12} {'Precision':<12} {'F1-score':<12}")
print("-" * 60)

for threshold in THRESHOLDS:
    predictions = (scores >= threshold).astype(int)

    acc = accuracy_score(labels, predictions)
    recall = recall_score(labels, predictions, zero_division=0)
    precision = precision_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)

    print(f"{threshold:<12.2f} {acc:<12.4f} {recall:<12.4f} {precision:<12.4f} {f1:<12.4f}")