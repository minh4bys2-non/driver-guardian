import json
import numpy as np

FEATURES = [
    "blink_frequency",
    "blink_duration",
    "perclos",
    "yawn_frequency",
    "nod_duration",
    "nod_frequency",
    "dominant_head_motion_frequency",
    "cnn_lstm_score",
]

def load_jsonl(path):
    X, y = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            X.append([row[name] for name in FEATURES])
            y.append(row["label"])
    return (
        np.asarray(X, dtype=np.float32),
        np.asarray(y, dtype=np.int8),
    )

class Fitness:
    def __init__(self, X):
        self.X = np.asarray(X, dtype=np.float32)
    @staticmethod
    def _candidate(candidate):
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.shape != (9,):
            raise ValueError("Candidate must be [w1, ..., w8, T_cls]")
        if np.any((candidate < 0) | (candidate > 1)):
            raise ValueError("Candidate values must be in [0, 1]")
        w = candidate[:8]
        threshold = candidate[8]
        mask = w >= 0.5
        return w, threshold, mask

    def weighted_mean(self, candidate):
        w, threshold, mask = self._candidate(candidate)
        if not mask.any():
            return np.zeros(len(self.X), dtype=np.int8)
        score = (
            self.X[:, mask] * w[mask]
        ).sum(axis=1) / w[mask].sum()
        return (score >= threshold).astype(np.int8)

    def noisy_or(self, candidate):
        w, threshold, mask = self._candidate(candidate)
        if not mask.any():
            return np.zeros(len(self.X), dtype=np.int8)
        z = self.X[:, mask] * w[mask]
        score = 1.0 - np.prod(1.0 - z, axis=1)
        return (score >= threshold).astype(np.int8)

    def predict(self, candidate, method="weighted_mean"):
        fn = getattr(self, method, None)
        if fn is None or not callable(fn):
            raise ValueError(f"Unknown fitness method: {method}")
        return fn(candidate)

    @staticmethod
    def evaluate(candidate, y_pred, y_true):
        candidate = np.asarray(candidate, dtype=np.float32)
        y_pred = np.asarray(y_pred, dtype=np.int8)
        y_true = np.asarray(y_true, dtype=np.int8)
        if y_pred.shape != y_true.shape:
            raise ValueError("y_pred and y_true must have the same shape")

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0

        num_feature = np.sum(candidate[:8] >= 0.5)

        return np.asarray([
            1.0 - recall,
            1.0 - precision,
            num_feature / 8.0,
        ], dtype=np.float32)