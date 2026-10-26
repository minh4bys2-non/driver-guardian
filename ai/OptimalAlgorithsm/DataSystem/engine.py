"""Trích xuất một mẫu đặc trưng cho mỗi cửa sổ video [start, end).

Input JSONL: {"video_path": "Fold1_part1/04/0.mp4", "label": 0}
Video lấy từ loadvideo2handle.read_video: (RGB uint8 [3,H,W], timestamp).
"""

import json
import sys
import time
from collections import deque
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

if __package__:
    from .loadvideo2handle import read_video
    from .neuralnetwork import NeuralNetwork, letterbox
    from .physicalbranch import CameraMetrics
    from .normalization import PhysicalNormalizer, FEATURE_ENCODING
    from .config import DEFAULT_CONFIG, PROGRESS_INTERVAL_SEC
else:
    from neuralnetwork import NeuralNetwork, letterbox
    from loadvideo2handle import read_video
    from physicalbranch import CameraMetrics
    from normalization import PhysicalNormalizer, FEATURE_ENCODING
    from config import DEFAULT_CONFIG, PROGRESS_INTERVAL_SEC


class Engine:
    def __init__(
        self,
        video_root,
        fps=None,
        window_sec=None,
        stride_sec=None,
        neural_sec=None,
        neural_size=None,
        min_observed_ratio=None,
        neural_fps=None,
        device=None,
        checkpoint_path=None,
        chunk_size=None,
        *, config=DEFAULT_CONFIG,
    ):
        self.config = config
        fps = config.fps if fps is None else fps
        stride_sec = config.stride_sec if stride_sec is None else stride_sec
        neural_sec = config.neural_sec if neural_sec is None else neural_sec
        neural_size = config.neural_size if neural_size is None else neural_size
        neural_fps = config.neural_fps if neural_fps is None else neural_fps
        min_observed_ratio = config.min_observed_ratio if min_observed_ratio is None else min_observed_ratio
        chunk_size = config.chunk_size if chunk_size is None else chunk_size
        device = config.device if device is None else device
        self.root = Path(video_root).expanduser().resolve()
        window_sec = config.window_sec if window_sec is None else window_sec
        values = (fps, window_sec, stride_sec, neural_sec, neural_size, neural_fps)
        if not np.isfinite(values).all() or any(v <= 0 for v in values):
            raise ValueError("Engine timing and sizes must be finite and positive")
        if fps != int(fps) or neural_size != int(neural_size) or fps % neural_fps:
            raise ValueError(
                "fps and neural_size must be integers; neural_fps must divide fps"
            )
        self.neural_step = int(fps / neural_fps)
        self.fps = int(fps)
        self.window = round(window_sec * fps)
        self.stride = round(stride_sec * fps)
        self.neural_count = round(neural_sec * neural_fps)
        self.neural_size = int(neural_size)
        self.min_observed_ratio = float(min_observed_ratio)
        if (
            fps <= 0
            or self.window <= 0
            or self.stride <= 0
            or self.neural_count <= 0
            or self.neural_size <= 0
            or not 0 <= min_observed_ratio <= 1
        ):
            raise ValueError("Invalid engine configuration")
        if (
            abs(self.window / fps - window_sec) > 1e-6
            or abs(self.stride / fps - stride_sec) > 1e-6
        ):
            raise ValueError("window_sec and stride_sec must be multiples of 1/fps")
        if (
            neural_sec > window_sec
            or abs(self.neural_count / neural_fps - neural_sec) > 1e-6
            or self.window % self.neural_step
            or self.stride % self.neural_step
        ):
            raise ValueError("Neural sampling must align with complete video windows")
        self.window_sec = self.window / fps
        self.config = replace(
            config, fps=self.fps, window_sec=self.window_sec, stride_sec=stride_sec,
            neural_sec=neural_sec, neural_fps=neural_fps, neural_size=neural_size,
            min_observed_ratio=min_observed_ratio, chunk_size=chunk_size, device=device,
            checkpoint_path=str(checkpoint_path or config.checkpoint_path),
        )
        self.normalizer = PhysicalNormalizer(self.config)
        self.neural = NeuralNetwork(
            checkpoint_path=checkpoint_path,
            config=self.config,
            device=device,
            chunk_size=chunk_size,
        ).eval()

    @staticmethod
    def _duration(history, percentile):
        values = [duration for _, duration in history]
        return float(np.percentile(values, percentile)) if values else 0.0

    def _features(
        self, m, blink_durations, nod_durations, nn_frames, source, start, label
    ):
        window = self.window_sec
        coverage = [
            m[k]
            for k in ("eye_observed_sec", "mouth_observed_sec", "head_observed_sec")
        ]
        windows = m.get("windows_sec", self.config.windows)
        required = [windows.get(k, window) * self.min_observed_ratio
                    for k in ("perclos", "pom", "over_angle")]
        if (
            not m["eye_ready"]
            or not m["mouth_ready"]
            or not m["head_calibrated"]
            or m["perclos_pct"] is None
            or any(
                v is None or not np.isfinite(v) or v < minimum - 1e-9
                for v, minimum in zip(coverage, required)
            )
        ):
            tqdm.write(f"[skip] {source} [{start:g}, {start + window:g}): "
                  f"ready eye/mouth/head={m['eye_ready']}/{m['mouth_ready']}/{m['head_calibrated']}, "
                  f"coverage eye/mouth/head={coverage}s (need {required}s), "
                  f"perclos={m['perclos_pct']}", file=sys.stderr)
            return None

        motion_window = windows.get("head_motion", self.window_sec)
        frequency = m["head_motion_frequency_hz"]
        resolution = m["head_motion_resolution_hz"]
        observed = m["head_motion_observed_sec"]
        reliable_motion = (
            frequency is not None
            and resolution is not None
            and observed is not None
            and np.isfinite((frequency, resolution, observed)).all()
            and frequency >= 0
            and resolution > 0
            and observed >= motion_window - 1e-9
        )
        if not reliable_motion:
            tqdm.write(f"[skip] {source} [{start:g}, {start + window:g}): "
                  f"FFT observed={observed}s (need {motion_window:g}s), "
                  f"frequency={frequency}Hz, resolution={resolution}Hz",
                  file=sys.stderr)
            return None

        with torch.inference_mode():
            score = float(self.neural(np.stack(nn_frames)))
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid neural score: {score}")

        raw = {
            "blink_frequency": m["blink_rate_per_min"],
            "blink_duration": self._duration(blink_durations, self.config.blink_duration_percentile),
            "perclos": m["perclos_pct"] / 100,
            "yawn_frequency": m["yawning_frequency_per_min"],
            "nod_duration": self._duration(nod_durations, self.config.nod_duration_percentile) / 1000,
            "nod_frequency": m["nodding_frequency_per_min"],
            "dominant_head_motion_frequency": frequency,
            "neuralnetwork_score": score,
        }
        return {
            **self.normalizer.transform(raw),
            "raw_features": raw,
            "normalization_id": self.normalizer.id,
            "normalization_scales": self.normalizer.scales,
            "source_video": source,
            "start_second": start,
            "end_second": start + window,
            "label": label,
            "feature_encoding": FEATURE_ENCODING,
        }

    def _source(self, video_path):
        if not isinstance(video_path, str) or not video_path.strip():
            raise ValueError("video_path must be a nonempty relative path")
        relative = Path(video_path)
        if relative.is_absolute() or not (
            self.root / relative
        ).resolve().is_relative_to(self.root):
            raise ValueError(f"Invalid relative video path: {video_path}")
        return (self.root / relative).resolve().relative_to(self.root).as_posix()

    def process_video(self, video_path, label, saved_windows=None):
        """Stream video một lần; yield mẫu hợp lệ theo stride đã cấu hình."""
        video_path = self._source(video_path)
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be 0 (alert) or 1 (drowsy)")
        path = self.root / video_path
        tqdm.write(f"[video] {video_path}: starting face tracking; window={self.window_sec:g}s, "
              f"stride={self.stride / self.fps:g}s", file=sys.stderr)
        camera = CameraMetrics(config=self.config)
        nn_frames = deque(maxlen=self.neural_count)
        blink_durations, nod_durations = deque(), deque()
        started = last_progress = time.monotonic()
        frame_count = emitted = skipped = resumed = 0
        progress = tqdm(
            read_video(str(path), target_fps=self.fps, target_luma=self.config.target_luma,
                       min_duration_sec=self.window_sec),
            desc=video_path, unit="frame", position=1, leave=False, dynamic_ncols=True,
        )
        try:
            for i, (frame, timestamp) in enumerate(progress):
                if (
                    not isinstance(frame, np.ndarray)
                    or frame.dtype != np.uint8
                    or frame.ndim != 3
                    or frame.shape[0] != 3
                    or min(frame.shape[1:]) == 0
                ):
                    raise ValueError(f"Invalid RGB frame [3,H,W]: {path}")
                rgb = np.moveaxis(frame, 0, -1)
                metrics = camera.process(
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), timestamp
                )
                if i % self.neural_step == 0:
                    nn_frames.append(
                        np.moveaxis(letterbox(rgb, self.neural_size), -1, 0)
                    )

                if (
                    metrics["blink_detected"]
                    and metrics["blink_duration_ms"] is not None
                ):
                    blink_durations.append((timestamp, metrics["blink_duration_ms"]))
                if (
                    metrics["nod_detected"]
                    and metrics["nod_event_duration_ms"] is not None
                ):
                    nod_durations.append((timestamp, metrics["nod_event_duration_ms"]))
                cutoff = timestamp - self.window_sec
                for history in (blink_durations, nod_durations):
                    while history and history[0][0] <= cutoff:
                        history.popleft()

                count = i + 1
                frame_count = count
                now = time.monotonic()
                if PROGRESS_INTERVAL_SEC > 0 and now - last_progress >= PROGRESS_INTERVAL_SEC:
                    progress.set_postfix(video=f"{timestamp:.1f}s", samples=emitted,
                                         skipped=skipped, resumed=resumed)
                    last_progress = now
                if count >= self.window and (count - self.window) % self.stride == 0:
                    start = (count - self.window) / self.fps
                    if (
                        saved_windows
                        and (start, start + self.window_sec) in saved_windows
                    ):
                        resumed += 1
                        continue
                    sample = self._features(
                        metrics,
                        blink_durations,
                        nod_durations,
                        nn_frames,
                        video_path,
                        start,
                        label,
                    )
                    if sample is not None:
                        emitted += 1
                        yield sample
                    else:
                        skipped += 1
        finally:
            progress.close()
            camera.close()
            tqdm.write(f"[video summary] {video_path}: frames={frame_count}, samples={emitted}, "
                  f"skipped={skipped}, resumed={resumed}, elapsed={time.monotonic() - started:.1f}s",
                  file=sys.stderr)

    def run(self, input_jsonl, output_jsonl, strict=False, resume=False):
        src, dst = Path(input_jsonl).resolve(), Path(output_jsonl).resolve()
        if src == dst:
            raise ValueError("Input and output JSONL must be different files")
        saved = {}
        video_labels = {}
        video_subjects = {}
        needs_newline = False
        if resume and dst.exists():
            with dst.open(encoding="utf-8") as existing:
                for line_number, line in enumerate(existing, 1):
                    needs_newline = not line.endswith("\n")
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        source, label = row["source_video"], row["label"]
                        source = self._source(source)
                        start, end = row["start_second"], row["end_second"]
                        if (
                            row["feature_encoding"] != FEATURE_ENCODING
                            or row["normalization_id"] != self.normalizer.id
                            or row["normalization_scales"] != self.normalizer.scales
                            or any(type(v) not in (int, float)
                                   for v in row["normalization_scales"].values())
                            or not isinstance(source, str)
                            or not source
                            or type(label) is not int
                            or label not in (0, 1)
                            or type(start) not in (int, float)
                            or type(end) not in (int, float)
                            or not np.isfinite((start, end)).all()
                            or start < 0
                            or abs(end - start - self.window_sec) > 1e-6
                            or abs(
                                start * self.fps / self.stride
                                - round(start * self.fps / self.stride)
                            )
                            > 1e-6
                            or any(
                                type(row[k]) not in (int, float)
                                or not np.isfinite(row[k]) or not 0 <= row[k] <= 1
                                for k in self.normalizer.scales
                            )
                        ):
                            raise ValueError("Incompatible saved sample")
                        if source in video_labels and video_labels[source] != label:
                            raise ValueError(f"Conflicting labels for video: {source}")
                        video_labels[source] = label
                        subject = row.get("subject_id")
                        if subject is not None:
                            if type(subject) not in (str, int) or not str(subject).strip():
                                raise ValueError("Invalid saved subject_id")
                            if source in video_subjects and video_subjects[source] != str(subject).strip():
                                raise ValueError(f"Conflicting subject IDs for video: {source}")
                            video_subjects[source] = str(subject).strip()
                        windows = saved.setdefault((source, label), set())
                        if (start, end) in windows:
                            raise ValueError(
                                f"Duplicate saved window: {source}, {start}, {end}"
                            )
                        windows.add((start, end))
                    except (ValueError, KeyError, TypeError) as exc:
                        raise ValueError(
                            f"Cannot resume {dst}, line {line_number}: {exc}"
                        ) from exc
        dst.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        seen = set()
        with src.open(encoding="utf-8") as manifest:
            total = sum(bool(line.strip()) for line in manifest)
        with src.open(encoding="utf-8") as fin, dst.open(
            "a" if resume else "w", encoding="utf-8"
        ) as fout, tqdm(
            ((n, line) for n, line in enumerate(fin, 1) if line.strip()),
            total=total, desc="Dataset", unit="video", position=0, dynamic_ncols=True,
        ) as progress:
            if needs_newline:
                fout.write("\n")
            for line_number, line in progress:
                try:
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise ValueError("Each JSONL record must be an object")
                    source, label = record["video_path"], record["label"]
                    subject = record.get("subject_id")
                    if subject is not None and (
                        type(subject) not in (str, int) or not str(subject).strip()
                    ):
                        raise ValueError("subject_id must be a nonempty string or integer")
                    source = self._source(source)
                    if subject is not None:
                        if source in video_subjects and video_subjects[source] != str(subject).strip():
                            raise ValueError(f"Conflicting subject IDs for video: {source}")
                        video_subjects[source] = str(subject).strip()
                    elif source in video_subjects:
                        subject = video_subjects[source]
                    if type(label) is not int or label not in (0, 1):
                        raise ValueError("label must be an integer 0 or 1")
                    if source in video_labels and video_labels[source] != label:
                        raise ValueError(f"Conflicting labels for video: {source}")
                    video_labels[source] = label
                    key = (source, label)
                    if key in seen:
                        continue
                    seen.add(key)
                    windows = saved.setdefault(key, set()) if resume else None
                    samples = (
                        self.process_video(*key, saved_windows=windows)
                        if resume
                        else self.process_video(*key)
                    )
                    for sample in samples:
                        if subject is not None:
                            sample["subject_id"] = subject
                        window = (
                            (sample["start_second"], sample["end_second"])
                            if resume
                            else None
                        )
                        if windows is not None and window in windows:
                            continue
                        fout.write(
                            json.dumps(sample, ensure_ascii=False, allow_nan=False)
                            + "\n"
                        )
                        fout.flush()  # giữ kết quả đã xử lý nếu video tiếp theo lỗi
                        if windows is not None:
                            windows.add(window)
                        count += 1
                        progress.set_postfix(saved=count, refresh=False)
                        tqdm.write(f"[saved] {source}: [{sample.get('start_second')}, "
                              f"{sample.get('end_second')}), total_new={count}",
                              file=sys.stderr)
                except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as exc:
                    if strict:
                        raise
                    tqdm.write(f"[line {line_number}] {exc}", file=sys.stderr)
        if count == 0:
            tqdm.write("[summary] No new samples saved. Check video duration, [skip] quality errors, "
                       "and whether resume already contains all valid windows.", file=sys.stderr)
        return count


if __name__ == "__main__":
    from ai.OptimalAlgorithsm.DataSystem.config import (
        INPUT_JSONL, OUTPUT_JSONL, RESUME,
        RUN_DEVICE, STRICT, VIDEO_ROOT,
    )

    print("[init] Loading DrowsyClassifier checkpoint...", file=sys.stderr, flush=True)
    engine = Engine(
        VIDEO_ROOT, config=replace(DEFAULT_CONFIG, device=RUN_DEVICE),
    )
    print(engine.run(INPUT_JSONL, OUTPUT_JSONL, strict=STRICT, resume=RESUME))
