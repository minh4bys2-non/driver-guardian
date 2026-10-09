"""Trích xuất một mẫu đặc trưng cho mỗi cửa sổ video [start, end).

Input JSONL: {"video_path": "Fold1_part1/04/0.mp4", "label": 0}
Video lấy từ loadvideo2handle.read_video: (RGB uint8 [3,H,W], timestamp).
"""
import json
import sys
from collections import deque
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

# Các mẫu xuất ra nằm trong [0,1]. Đây là thang tham chiếu cấu hình,
# KHÔNG phải giới hạn sinh lý đã được hiệu chuẩn trên tập dữ liệu.
SCALES = {
    "blink_frequency": 30.0,  # lần/phút
    "blink_duration": 2000.0,  # ms
    "perclos": 100.0,  # %
    "yawn_frequency": 5.0,  # lần/phút
    "nod_duration": 3500.0,  # ms
    "nod_frequency": 5.0,  # lần/phút
    "dominant_head_motion_frequency": 2.0,  # Hz
}


class Engine:
    def __init__(self, video_root, fps=30, window_sec=60, stride_sec=10,
                 neural_sec=10, neural_size=640, min_observed_ratio=0.8, scales=None,
                 neural_fps=5, device=None, cnn_path=None, conv_gru_path=None,
                 chunk_size=32):
        self.root = Path(video_root).expanduser().resolve()
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
        self.scales = SCALES | (scales or {})
        if (fps <= 0 or self.window <= 0 or self.stride <= 0 or self.neural_count <= 0
                or self.neural_size <= 0 or not 0 <= min_observed_ratio <= 1
                or any(v <= 0 or not np.isfinite(v) for v in self.scales.values())):
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

    def _norm(self, key, value):
        # Một sự kiện chưa xảy ra được biểu diễn bằng 0.
        if value is None:
            value = 0.0
        if not np.isfinite(value):
            raise ValueError(f"Non-finite feature: {key}")
        return round(float(np.clip(value / self.scales[key], 0, 1)), 4)

    def _features(self, m, blink_durations, nod_durations, nn_frames, source, start, label):
        window = self.window_sec
        if (
                not m["eye_ready"] or not m["mouth_ready"] or not m["head_calibrated"]
                or m["perclos_pct"] is None or m["head_motion_resolution_hz"] is None
                or any((m[k] or 0) < window * self.min_observed_ratio
                       for k in ("eye_observed_sec", "mouth_observed_sec", "head_observed_sec"))):
            return None

        def mean_duration(history):
            return sum(duration for _, duration in history) / len(history) if history else 0.0

        with torch.inference_mode():
            score = float(self.neural(np.stack(nn_frames)))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid neural score: {score}")

        raw = {
            "blink_frequency": m["blink_rate_per_min"],
            "blink_duration": mean_duration(blink_durations),
            "perclos": m["perclos_pct"],
            "yawn_frequency": m["yawning_frequency_per_min"],
            "nod_duration": mean_duration(nod_durations),
            "nod_frequency": m["nodding_frequency_per_min"],
            "dominant_head_motion_frequency": m["head_motion_frequency_hz"],
        }
        return {**{k: self._norm(k, v) for k, v in raw.items()},
                "cnn_lstm_score": round(score, 4), "source_video": source,
                "start_second": start, "end_second": start + window, "label": label}

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
            windows={"head_motion": (self.window - 1) / self.fps} if self.window > 1 else None,
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
    OUTPUT_JSONL = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/drowsiness_sample.jsonl"
    CNN_PATH = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/model_cnn/best.pt"
    CONV_GRU_PATH = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/model_convgru/best.pt"

    FPS = 30
    WINDOW_SEC = 60
    STRIDE_SEC = 10
    NEURAL_SEC = 10
    NEURAL_FPS = 5
    NEURAL_SIZE = 640
    CHUNK_SIZE = 32
    DEVICE = "cuda"  # None: tự chọn CUDA/CPU; hoặc "cuda", "cpu".
    MIN_OBSERVED_RATIO = 0.8
    FEATURE_SCALES = SCALES.copy()
    STRICT = False

    engine = Engine(
        video_root=VIDEO_ROOT,
        fps=FPS,
        window_sec=WINDOW_SEC,
        stride_sec=STRIDE_SEC,
        neural_sec=NEURAL_SEC,
        neural_fps=NEURAL_FPS,
        neural_size=NEURAL_SIZE,
        cnn_path=CNN_PATH,
        conv_gru_path=CONV_GRU_PATH,
        chunk_size=CHUNK_SIZE,
        device=DEVICE,
        min_observed_ratio=MIN_OBSERVED_RATIO,
        scales=FEATURE_SCALES,
    )
    print(engine.run(INPUT_JSONL, OUTPUT_JSONL, strict=STRICT))
