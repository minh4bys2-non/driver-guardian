"""Trích xuất một mẫu đặc trưng cho mỗi cửa sổ video [start, end).

Input JSONL: {"video_path": "Fold1_part1/04/0.mp4", "label": 0}
Video lấy từ loadvideo2handle.read_video: (RGB uint8 [3,H,W], timestamp).
"""
import json
import sys
from collections import deque
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import torch

if __package__:
    from .loadvideo2handle import read_video
    from .neuralnetwork import NeuralNetwork, letterbox
    from .physicalbranch import CameraMetrics
else:
    from loadvideo2handle import read_video
    from neuralnetwork import NeuralNetwork, letterbox
    from physicalbranch import CameraMetrics

DEFAULT_RISK_CONFIG = {
    "window_sec": 60,
    "blink_frequency": {
        "normal": (15, 20), "low_danger": 4, "high_danger": 35,
        "gate_low_with_eye_metrics": True,
    },
    "blink_duration": {"safe": 400, "danger": 800, "unit": "ms", "aggregation": "p90"},
    "perclos": {"safe": 0.05, "danger": 0.15},
    "yawn_frequency": {"safe": 0, "danger": 2, "unit": "events/min", "min_event_sec": 4.0},
    "nod_duration": {"safe": 0.5, "danger": 2.0, "unit": "sec", "aggregation": "max"},
    "nod_frequency": {"safe": 0, "danger": 3, "unit": "events/min", "min_event_sec": 0.8},
    "dominant_head_motion_frequency": {
        "method": "fft_pitch", "unit": "Hz", "window_sec": 60,
        "reference_band": (0.05, 0.20), "low_danger": 0.0, "high_danger": 0.60,
        "require_reliable_motion": True,
    },
    "cnn_lstm_score": {"safe": 0.0, "danger": 1.0},
}


class Engine:
    def __init__(self, video_root, fps=30, window_sec=None, stride_sec=10,
                 neural_sec=10, neural_size=640, min_observed_ratio=0.8, risk_config=None,
                 neural_fps=5, device=None, cnn_path=None, conv_gru_path=None,
                 chunk_size=32):
        self.root = Path(video_root).expanduser().resolve()
        self.risk_config = deepcopy(DEFAULT_RISK_CONFIG)
        for key, value in (risk_config or {}).items():
            if key not in self.risk_config:
                raise ValueError(f"Unknown risk configuration: {key}")
            if key == "window_sec":
                self.risk_config[key] = value
            else:
                if not isinstance(value, dict) or value.keys() - self.risk_config[key].keys():
                    raise ValueError(f"Invalid risk configuration: {key}")
                self.risk_config[key].update(deepcopy(value))
        self._validate_risk_config()
        if window_sec is not None and window_sec != self.risk_config["window_sec"]:
            raise ValueError("window_sec must match risk_config['window_sec']")
        window_sec = self.risk_config["window_sec"]
        values = (fps, window_sec, stride_sec, neural_sec, neural_size, neural_fps)
        if not np.isfinite(values).all() or any(v <= 0 for v in values):
            raise ValueError("Engine timing and sizes must be finite and positive")
        if fps != int(fps) or neural_size != int(neural_size) or fps % neural_fps:
            raise ValueError("fps and neural_size must be integers; neural_fps must divide fps")
        self.neural_step = int(fps / neural_fps)
        self.fps = int(fps)
        self.window = round(window_sec * fps)
        self.stride = round(stride_sec * fps)
        self.neural_count = round(neural_sec * neural_fps)
        self.neural_size = int(neural_size)
        self.min_observed_ratio = float(min_observed_ratio)
        if (fps <= 0 or self.window <= 0 or self.stride <= 0 or self.neural_count <= 0
                or self.neural_size <= 0 or not 0 <= min_observed_ratio <= 1):
            raise ValueError("Invalid engine configuration")
        if abs(self.window / fps - window_sec) > 1e-6 or abs(self.stride / fps - stride_sec) > 1e-6:
            raise ValueError("window_sec and stride_sec must be multiples of 1/fps")
        if (neural_sec > window_sec or abs(self.neural_count / neural_fps - neural_sec) > 1e-6
                or self.window % self.neural_step or self.stride % self.neural_step):
            raise ValueError("Neural sampling must align with complete video windows")
        self.window_sec = self.window / fps
        self.neural = NeuralNetwork(
            cnn_path=cnn_path, conv_gru_path=conv_gru_path,
            device=device, chunk_size=chunk_size,
        ).eval()

    def _validate_risk_config(self):
        cfg = self.risk_config
        motion = cfg["dominant_head_motion_frequency"]
        for value in (cfg["window_sec"], motion["window_sec"]):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("Risk windows must be finite and positive")
        for key, rule in cfg.items():
            if key == "window_sec":
                continue
            if "safe" in rule:
                points = (rule["safe"], rule["danger"])
            else:
                band = rule["normal"] if key == "blink_frequency" else rule["reference_band"]
                if len(band) != 2:
                    raise ValueError(f"Expected a two-point reference band: {key}")
                points = (rule["low_danger"], *band, rule["high_danger"])
            if not np.isfinite(points).all() or points[0] < 0 or np.any(np.diff(points) <= 0):
                raise ValueError(f"Risk thresholds must be finite and strictly increasing: {key}")
            if "unit" in rule and rule["unit"] != DEFAULT_RISK_CONFIG[key]["unit"]:
                raise ValueError(f"Unsupported unit: {key}")
            if "aggregation" in rule and rule["aggregation"] not in ("p90", "max"):
                raise ValueError(f"Unsupported aggregation: {key}")
        for key, upper_sec in (("yawn_frequency", 7.5), ("nod_frequency", 3.5)):
            value = cfg[key]["min_event_sec"]
            if not np.isfinite(value) or not 0 < value < upper_sec:
                raise ValueError(f"min_event_sec must be below the detector maximum {upper_sec}s: {key}")
        if cfg["perclos"]["danger"] > 1 or cfg["cnn_lstm_score"]["danger"] > 1:
            raise ValueError("PERCLOS and neural thresholds must be fractions in [0,1]")
        if (motion["method"] != "fft_pitch"
                or type(motion["require_reliable_motion"]) is not bool
                or type(cfg["blink_frequency"]["gate_low_with_eye_metrics"]) is not bool):
            raise ValueError("Invalid FFT method or risk gate")

    def _norm(self, key, value):
        if value is None or not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid raw feature: {key}")
        rule = self.risk_config[key]
        if "safe" in rule:
            return float(np.clip((value - rule["safe"]) / (rule["danger"] - rule["safe"]), 0, 1))
        band = rule["normal"] if key == "blink_frequency" else rule["reference_band"]
        return float(np.interp(value, (rule["low_danger"], *band, rule["high_danger"]), (1, 0, 0, 1)))

    def _duration(self, key, history):
        values = [duration for _, duration in history]
        if not values:
            return 0.0
        rule = self.risk_config[key]
        duration_ms = float(np.percentile(values, 90 if rule["aggregation"] == "p90" else 100))
        return duration_ms / 1000 if rule["unit"] == "sec" else duration_ms

    def _features(self, m, blink_durations, nod_durations, nn_frames, source, start, label):
        window = self.window_sec
        if (
                not m["eye_ready"] or not m["mouth_ready"] or not m["head_calibrated"]
                or m["perclos_pct"] is None
                or any((m[k] or 0) < window * self.min_observed_ratio
                       for k in ("eye_observed_sec", "mouth_observed_sec", "head_observed_sec"))):
            return None

        motion = self.risk_config["dominant_head_motion_frequency"]
        frequency = m["head_motion_frequency_hz"]
        resolution = m["head_motion_resolution_hz"]
        observed = m["head_motion_observed_sec"]
        reliable_motion = (
            frequency is not None and resolution is not None
            and np.isfinite((frequency, resolution, observed)).all()
            and frequency > 0 and resolution > 0
            and observed >= motion["window_sec"] - 1e-9
        )
        if motion["require_reliable_motion"] and not reliable_motion:
            return None

        with torch.inference_mode():
            score = float(self.neural(np.stack(nn_frames)))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid neural score: {score}")

        raw = {
            "blink_frequency": m["blink_rate_per_min"],
            "blink_duration": self._duration("blink_duration", blink_durations),
            "perclos": m["perclos_pct"] / 100,
            "yawn_frequency": m["yawning_frequency_per_min"],
            "nod_duration": self._duration("nod_duration", nod_durations),
            "nod_frequency": m["nodding_frequency_per_min"],
            "dominant_head_motion_frequency": frequency if reliable_motion else 0.0,
            "cnn_lstm_score": score,
        }
        risk = {k: self._norm(k, v) for k, v in raw.items()}
        if not reliable_motion:
            risk["dominant_head_motion_frequency"] = 0.0
        blink_rule = self.risk_config["blink_frequency"]
        if blink_rule["gate_low_with_eye_metrics"] and raw["blink_frequency"] < blink_rule["normal"][0]:
            risk["blink_frequency"] *= max(risk["perclos"], risk["blink_duration"])
        return {**{k: round(v, 4) for k, v in risk.items()}, "source_video": source,
                "start_second": start, "end_second": start + window, "label": label,
                "feature_encoding": "risk_v2"}

    def process_video(self, video_path, label):
        """Stream video một lần; yield từng dòng dữ liệu sau mỗi 10 giây."""
        if not isinstance(video_path, str) or not video_path:
            raise ValueError("video_path must be a nonempty relative path")
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be 0 (alert) or 1 (drowsy)")
        relative = Path(video_path)
        if relative.is_absolute() or not (self.root / relative).resolve().is_relative_to(self.root):
            raise ValueError(f"Invalid relative video path: {video_path}")
        path = self.root / relative
        camera = CameraMetrics(
            fps=self.fps, window_sec=self.window_sec,
            windows={"head_motion": self.risk_config["dominant_head_motion_frequency"]["window_sec"]},
            mouth_options={"min_duration_ms": self.risk_config["yawn_frequency"]["min_event_sec"] * 1000},
            nod_duration_ms=(self.risk_config["nod_frequency"]["min_event_sec"] * 1000, 3500),
        )
        nn_frames = deque(maxlen=self.neural_count)
        blink_durations, nod_durations = deque(), deque()
        try:
            for i, (frame, timestamp) in enumerate(read_video(str(path), target_fps=self.fps)):
                if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[
                    0] != 3 or min(frame.shape[1:]) == 0:
                    raise ValueError(f"Invalid RGB frame [3,H,W]: {path}")
                rgb = np.moveaxis(frame, 0, -1)
                metrics = camera.process(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), timestamp)
                if i % self.neural_step == 0:
                    nn_frames.append(np.moveaxis(letterbox(rgb, self.neural_size), -1, 0))

                if metrics["blink_detected"] and metrics["blink_duration_ms"] is not None:
                    blink_durations.append((timestamp, metrics["blink_duration_ms"]))
                if metrics["nod_detected"] and metrics["nod_event_duration_ms"] is not None:
                    nod_durations.append((timestamp, metrics["nod_event_duration_ms"]))
                cutoff = timestamp - self.window_sec
                for history in (blink_durations, nod_durations):
                    while history and history[0][0] <= cutoff:
                        history.popleft()

                count = i + 1
                if count >= self.window and (count - self.window) % self.stride == 0:
                    start = (count - self.window) / self.fps
                    sample = self._features(metrics, blink_durations, nod_durations,
                                            nn_frames, video_path, start, label)
                    if sample is not None:
                        yield sample
        finally:
            camera.close()

    def run(self, input_jsonl, output_jsonl, strict=False):
        src, dst = Path(input_jsonl).resolve(), Path(output_jsonl).resolve()
        if src == dst:
            raise ValueError("Input and output JSONL must be different files")
        dst.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with src.open(encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
            for line_number, line in enumerate(fin, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Each JSONL record must be an object")
                    for sample in self.process_video(record["video_path"], record["label"]):
                        fout.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                        fout.flush()  # giữ kết quả đã xử lý nếu video tiếp theo lỗi
                        count += 1
                except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as exc:
                    if strict:
                        raise
                    print(f"[line {line_number}] {exc}", file=sys.stderr)
        return count

if __name__ == "__main__":
    INPUT_JSONL = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/UTA-RLDD/dataset.jsonl"
    VIDEO_ROOT = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/UTA-RLDD"
    OUTPUT_JSONL = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/drowsiness_risk.jsonl"
    CNN_PATH = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/model_cnn/best.pt"
    CONV_GRU_PATH = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/model_convgru/best.pt"

    FPS = 30
    STRIDE_SEC = 10
    NEURAL_SEC = 10
    NEURAL_FPS = 5
    NEURAL_SIZE = 640
    CHUNK_SIZE = 32
    DEVICE = "cuda"  # None: tự chọn CUDA/CPU; hoặc "cuda", "cpu".
    MIN_OBSERVED_RATIO = 0.8
    RISK_CONFIG = deepcopy(DEFAULT_RISK_CONFIG)
    STRICT = False

    engine = Engine(
        video_root=VIDEO_ROOT,
        fps=FPS,
        stride_sec=STRIDE_SEC,
        neural_sec=NEURAL_SEC,
        neural_fps=NEURAL_FPS,
        neural_size=NEURAL_SIZE,
        cnn_path=CNN_PATH,
        conv_gru_path=CONV_GRU_PATH,
        chunk_size=CHUNK_SIZE,
        device=DEVICE,
        min_observed_ratio=MIN_OBSERVED_RATIO,
        risk_config=RISK_CONFIG,
    )
    print(engine.run(INPUT_JSONL, OUTPUT_JSONL, strict=STRICT))
