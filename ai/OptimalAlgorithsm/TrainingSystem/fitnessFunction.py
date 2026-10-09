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


def validate_labels(values, n_samples=None):
    values = np.asarray(values)
    if (values.ndim != 1 or not values.size or not np.isin(values, (0, 1)).all()
            or (n_samples is not None and len(values) != n_samples)):
        raise ValueError("Labels must be a nonempty binary vector matching the samples")
    return values.astype(np.int8)


def load_jsonl(path):
    X, y = [], []
    encoding = None
    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or type(row["label"]) is not int or row["label"] not in (0, 1):
                    raise ValueError("Each record requires an integer label 0 or 1")
                row_encoding = row.get("feature_encoding", "scale_v1")
                if row_encoding not in ("scale_v1", "risk_v1", "risk_v2"):
                    raise ValueError("Unknown feature encoding")
                if encoding is not None and row_encoding != encoding:
                    raise ValueError("Cannot mix feature encodings in one dataset")
                encoding = row_encoding
                values = np.asarray([row[name] for name in FEATURES], dtype=np.float64)
                if values.shape != (8,) or not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
                    raise ValueError("Features must be eight finite values in [0, 1]")
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError(f"{path}: line {line_number}: {exc}") from exc
            X.append(values)
            y.append(row["label"])
    if not X:
        raise ValueError(f"Dataset is empty: {path}")
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int8)

class Fitness:
    def __init__(self, X):
        self.X = np.asarray(X, dtype=np.float32)
        if (self.X.ndim != 2 or self.X.shape[1] != 8 or len(self.X) == 0
                or not np.isfinite(self.X).all() or np.any((self.X < 0) | (self.X > 1))):
            raise ValueError("X must be a nonempty [N,8] array of finite values in [0,1]")
    @staticmethod
    def _candidate(candidate):
        candidate = np.asarray(candidate, dtype=np.float32)
        if candidate.shape != (9,):
            raise ValueError("Candidate must be [w1, ..., w8, T_cls]")
        if not np.isfinite(candidate).all() or np.any((candidate < 0) | (candidate > 1)):
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
        if method not in ("weighted_mean", "noisy_or"):
            raise ValueError(f"Unknown fitness method: {method}")
        return getattr(self, method)(candidate)

    @staticmethod
    def evaluate(candidate, y_pred, y_true):
        _, _, mask = Fitness._candidate(candidate)
        y_true = validate_labels(y_true)
        y_pred = validate_labels(y_pred, len(y_true))

        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0

        num_feature = mask.sum()

        return np.asarray([
            1.0 - recall,
            1.0 - precision,
            num_feature / 8.0,
        ], dtype=np.float32)
