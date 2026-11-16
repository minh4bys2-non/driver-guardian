"""Trích xuất lại nhánh neuralnetwork trên dataset vật lý đã có.

Đọc file JSONL chứa các đặc trưng vật lý đã trích xuất (ví dụ: drowsiness_physical_60_novisual.jsonl),
giữ nguyên 100% các đặc trưng vật lý (cả ở dạng chuẩn hóa lẫn trong raw_features),
chỉ chạy lại mô hình NeuralNetwork (LiteFat Multimodal) trên chuỗi khung hình video
tại đúng các cửa sổ [start, end) và ghi ra file JSONL mới sau khi so khớp nghiêm ngặt.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict, deque
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, Union

import cv2
import numpy as np
import torch
from tqdm import tqdm

_CURRENT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _CURRENT_DIR.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if __package__:
    from .loadvideo2handle import read_video
    from .neuralnetwork_v1 import NeuralNetwork, letterbox
    from .normalization import PhysicalNormalizer, FEATURE_ENCODING
    from .config import DEFAULT_CONFIG, PROGRESS_INTERVAL_SEC, VIDEO_ROOT
else:
    from loadvideo2handle import read_video
    from neuralnetwork_v1 import NeuralNetwork, letterbox
    from normalization import PhysicalNormalizer, FEATURE_ENCODING
    from config import DEFAULT_CONFIG, PROGRESS_INTERVAL_SEC, VIDEO_ROOT

DEFAULT_INPUT_JSONL = str(_CURRENT_DIR / "drowsiness_physical_60_novisual.jsonl")
DEFAULT_OUTPUT_JSONL = str(_CURRENT_DIR / "drowsiness_physical_60_visual.jsonl")

PHYSICAL_FEATURES: Tuple[str, ...] = (
    "blink_frequency",
    "blink_duration",
    "perclos",
    "yawn_frequency",
    "nod_duration",
    "nod_frequency",
    "dominant_head_motion_frequency",
)


class Engine:
    def __init__(
            self,
            video_root: Optional[Union[str, Path]] = None,
            fps: Optional[float] = None,
            window_sec: Optional[float] = None,
            stride_sec: Optional[float] = None,
            neural_sec: Optional[float] = None,
            neural_size: Optional[int] = None,
            min_observed_ratio: Optional[float] = None,
            neural_fps: Optional[float] = None,
            device: Optional[str] = None,
            checkpoint_path: Optional[Union[str, Path]] = None,
            chunk_size: Optional[int] = None,
            neural_class: Optional[Any] = None,
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
        
        root_path = video_root or VIDEO_ROOT
        self.root = Path(root_path).expanduser().resolve()
        window_sec = config.window_sec if window_sec is None else window_sec
        values = (fps, window_sec, stride_sec, neural_sec, neural_size, neural_fps)
        if not np.isfinite(values).all() or any(v <= 0 for v in values):
            raise ValueError("Engine timing and sizes must be finite and positive")
        
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

        target_ckpt = Path(checkpoint_path or self.config.checkpoint_path).expanduser().resolve()
        self.neural = NeuralNetwork(checkpoint_path=target_ckpt).eval()

    def _source(self, video_path: str) -> str:
        if not isinstance(video_path, str) or not video_path.strip():
            raise ValueError("video_path must be a nonempty relative path")
        relative = Path(video_path)
        if relative.is_absolute() or not (
                self.root / relative
        ).resolve().is_relative_to(self.root):
            raise ValueError(f"Invalid relative video path: {video_path}")
        return (self.root / relative).resolve().relative_to(self.root).as_posix()

    @staticmethod
    def _verify_sample_match(original: Dict[str, Any], updated: Dict[str, Any]) -> None:
        """Kiểm tra đối chiếu nghiêm ngặt: mẫu mới cập nhật phải khớp hoàn toàn với mẫu gốc."""
        if updated["source_video"] != original["source_video"]:
            raise ValueError(
                f"Source video mismatch: {updated['source_video']} vs {original['source_video']}"
            )
        if abs(updated["start_second"] - original["start_second"]) > 1e-6:
            raise ValueError(
                f"Start second mismatch: {updated['start_second']} vs {original['start_second']}"
            )
        if abs(updated["end_second"] - original["end_second"]) > 1e-6:
            raise ValueError(
                f"End second mismatch: {updated['end_second']} vs {original['end_second']}"
            )
        if updated["label"] != original["label"]:
            raise ValueError(
                f"Label mismatch: {updated['label']} vs {original['label']}"
            )
        if updated.get("subject_id") != original.get("subject_id"):
            raise ValueError(
                f"Subject ID mismatch: {updated.get('subject_id')} vs {original.get('subject_id')}"
            )
        if updated.get("feature_encoding") != original.get("feature_encoding"):
            raise ValueError(
                f"Feature encoding mismatch: {updated.get('feature_encoding')} vs {original.get('feature_encoding')}"
            )
        
        orig_raw = original.get("raw_features", {})
        upd_raw = updated.get("raw_features", {})
        for k in PHYSICAL_FEATURES:
            if abs(updated[k] - original[k]) > 1e-6:
                raise ValueError(
                    f"Normalized physical feature {k} mismatch: {updated[k]} vs {original[k]}"
                )
            if abs(upd_raw[k] - orig_raw[k]) > 1e-6:
                raise ValueError(
                    f"Raw physical feature {k} mismatch: {upd_raw[k]} vs {orig_raw[k]}"
                )
        
        score = updated["neuralnetwork_score"]
        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid updated neural score: {score}")

    def process_video(
            self,
            video_path: str,
            label: int,
            target_records: Union[Dict[Tuple[float, float], Dict[str, Any]], List[Dict[str, Any]]],
            saved_windows: Optional[Set[Tuple[float, float]]] = None,
    ) -> Iterator[Dict[str, Any]]:
        """Stream video một lần; yield mẫu sau khi chạy lại NeuralNetwork và so khớp."""
        video_path = self._source(video_path)
        if type(label) is not int or label not in (0, 1):
            raise ValueError("label must be 0 (alert) or 1 (drowsy)")
        
        if isinstance(target_records, list):
            target_records = {
                (float(r["start_second"]), float(r["end_second"])): r
                for r in target_records
            }
        
        pending_keys = [
            k for k in target_records.keys()
            if not (saved_windows and k in saved_windows)
        ]
        if not pending_keys:
            return

        max_end_sec = max(end for _, end in pending_keys)
        max_end_count = round(max_end_sec * self.fps)
        
        path = self.root / video_path
        tqdm.write(
            f"[video] {video_path}: re-running neural network; "
            f"target_windows={len(target_records)}, pending={len(pending_keys)}",
            file=sys.stderr,
        )
        
        nn_frames: deque = deque(maxlen=self.neural_count)
        started = last_progress = time.monotonic()
        frame_count = emitted = skipped = resumed = 0
        
        progress = tqdm(
            read_video(
                str(path),
                target_fps=self.fps,
                target_luma=self.config.target_luma,
                min_duration_sec=self.window_sec,
            ),
            desc=video_path,
            unit="frame",
            position=1,
            leave=False,
            dynamic_ncols=True,
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
                if i % self.neural_step == 0:
                    nn_frames.append(
                        np.moveaxis(letterbox(rgb, self.neural_size), -1, 0)
                    )
                
                count = i + 1
                frame_count = count
                now = time.monotonic()
                if PROGRESS_INTERVAL_SEC > 0 and now - last_progress >= PROGRESS_INTERVAL_SEC:
                    progress.set_postfix(
                        video=f"{timestamp:.1f}s",
                        samples=emitted,
                        skipped=skipped,
                        resumed=resumed,
                    )
                    last_progress = now
                
                if count >= self.window and (count - self.window) % self.stride == 0:
                    start = (count - self.window) / self.fps
                    end = start + self.window_sec
                    window_key = (start, end)
                    
                    if window_key not in target_records:
                        skipped += 1
                        continue
                    
                    if saved_windows and window_key in saved_windows:
                        resumed += 1
                        continue
                    
                    if len(nn_frames) < self.neural_count:
                        raise ValueError(
                            f"Neural buffer underflow: {len(nn_frames)} frames < {self.neural_count} at [{start}, {end})"
                        )
                    
                    with torch.inference_mode():
                        score = float(self.neural(np.stack(nn_frames)))
                    if not np.isfinite(score) or not 0 <= score <= 1:
                        raise ValueError(f"Invalid neural score: {score}")
                    
                    orig_record = target_records[window_key]
                    orig_raw = orig_record["raw_features"]
                    raw = {k: orig_raw[k] for k in PHYSICAL_FEATURES}
                    raw["neuralnetwork_score"] = score
                    
                    transformed = self.normalizer.transform(raw)
                    sample = {
                        **transformed,
                        "raw_features": raw,
                        "normalization_id": self.normalizer.id,
                        "normalization_scales": self.normalizer.scales,
                        "source_video": orig_record["source_video"],
                        "start_second": orig_record["start_second"],
                        "end_second": orig_record["end_second"],
                        "label": orig_record["label"],
                        "feature_encoding": orig_record.get("feature_encoding", FEATURE_ENCODING),
                    }
                    if "subject_id" in orig_record:
                        sample["subject_id"] = orig_record["subject_id"]
                    
                    self._verify_sample_match(orig_record, sample)
                    emitted += 1
                    yield sample
                    
                    if count >= max_end_count:
                        break
        finally:
            progress.close()
            tqdm.write(
                f"[video summary] {video_path}: frames={frame_count}, emitted={emitted}, "
                f"skipped={skipped}, resumed={resumed}, elapsed={time.monotonic() - started:.1f}s",
                file=sys.stderr,
            )

    def run(
            self,
            input_jsonl: Union[str, Path] = DEFAULT_INPUT_JSONL,
            output_jsonl: Union[str, Path] = DEFAULT_OUTPUT_JSONL,
            strict: bool = False,
            resume: bool = False,
    ) -> int:
        src, dst = Path(input_jsonl).resolve(), Path(output_jsonl).resolve()
        if src == dst:
            raise ValueError("Input and output JSONL must be different files")
        if not src.is_file():
            raise FileNotFoundError(f"Input JSONL file not found: {src}")

        # 1. Đọc và kiểm tra tính toàn vẹn của file input
        tqdm.write(f"[load] Reading and validating input records from {src}...", file=sys.stderr)
        grouped_records: Dict[Tuple[str, int], Dict[Tuple[float, float], Dict[str, Any]]] = OrderedDict()
        video_subjects: Dict[str, str] = {}
        total_input_records = 0

        with src.open(encoding="utf-8") as fin:
            for line_number, line in enumerate(fin, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("Each JSONL record must be an object")
                    source = self._source(row["source_video"])
                    label = row["label"]
                    if type(label) is not int or label not in (0, 1):
                        raise ValueError("label must be 0 or 1")
                    start, end = float(row["start_second"]), float(row["end_second"])
                    if not np.isfinite((start, end)).all() or start < 0 or abs(end - start - self.window_sec) > 1e-6:
                        raise ValueError(f"Invalid window timing [{start}, {end})")
                    if row.get("feature_encoding") != FEATURE_ENCODING:
                        raise ValueError(f"Unsupported feature encoding: {row.get('feature_encoding')}")
                    
                    raw = row.get("raw_features")
                    if not isinstance(raw, dict):
                        raise ValueError("Missing or invalid raw_features")
                    for k in PHYSICAL_FEATURES:
                        if k not in raw or k not in row:
                            raise ValueError(f"Missing physical feature {k}")
                    
                    # Kiểm tra transform các đặc trưng vật lý khớp với row
                    tf = self.normalizer.transform(raw)
                    for k in PHYSICAL_FEATURES:
                        if abs(tf[k] - row[k]) > 1e-6:
                            raise ValueError(f"Normalized feature {k} does not match raw feature transform")
                    
                    subject = row.get("subject_id")
                    if subject is not None:
                        subject = str(subject).strip()
                        if source in video_subjects and video_subjects[source] != subject:
                            raise ValueError(f"Conflicting subject IDs for video: {source}")
                        video_subjects[source] = subject
                    
                    key = (source, label)
                    win = (start, end)
                    windows_dict = grouped_records.setdefault(key, OrderedDict())
                    if win in windows_dict:
                        raise ValueError(f"Duplicate window in input: {source} [{start}, {end})")
                    windows_dict[win] = row
                    total_input_records += 1
                except Exception as exc:
                    raise ValueError(f"Error validating {src} line {line_number}: {exc}") from exc

        tqdm.write(
            f"[load] Validated {total_input_records} samples across {len(grouped_records)} videos.",
            file=sys.stderr,
        )

        # 2. Xử lý resume nếu file output đã tồn tại
        saved: Dict[Tuple[str, int], Set[Tuple[float, float]]] = {}
        last_saved_key = None
        needs_newline = False
        saved_count = 0

        if resume and dst.exists():
            with dst.open(encoding="utf-8") as existing:
                for line_number, line in enumerate(existing, 1):
                    needs_newline = not line.endswith("\n")
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                        source = self._source(row["source_video"])
                        label = row["label"]
                        start, end = float(row["start_second"]), float(row["end_second"])
                        key = (source, label)
                        win = (start, end)
                        
                        if key not in grouped_records or win not in grouped_records[key]:
                            raise ValueError(f"Saved window not present in input dataset: {source} [{start}, {end})")
                        
                        # So khớp với bản ghi gốc
                        orig = grouped_records[key][win]
                        self._verify_sample_match(orig, row)
                        
                        windows_set = saved.setdefault(key, set())
                        if win in windows_set:
                            raise ValueError(f"Duplicate saved window: {source} [{start}, {end})")
                        windows_set.add(win)
                        saved_count += 1
                        last_saved_key = key
                    except Exception as exc:
                        raise ValueError(f"Cannot resume {dst}, line {line_number}: {exc}") from exc
            tqdm.write(f"[resume] Loaded and verified {saved_count} existing samples from {dst}.", file=sys.stderr)

        dst.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        
        with dst.open("a" if resume else "w", encoding="utf-8") as fout, tqdm(
            grouped_records.items(),
            total=len(grouped_records),
            desc="Dataset Re-extract",
            unit="video",
            position=0,
            dynamic_ncols=True,
        ) as progress:
            if needs_newline:
                fout.write("\n")
            
            for (source, label), target_dict in progress:
                key = (source, label)
                
                # Nếu resume và video này không phải video cuối cùng bị ngắt quãng, và đã lưu đủ mọi window:
                if resume and key in saved and key != last_saved_key:
                    if set(target_dict.keys()).issubset(saved[key]):
                        continue
                
                saved_windows = saved.get(key, set()) if resume else None
                try:
                    samples = self.process_video(
                        source,
                        label,
                        target_dict,
                        saved_windows=saved_windows,
                    )
                    for sample in samples:
                        win = (sample["start_second"], sample["end_second"])
                        if saved_windows is not None and win in saved_windows:
                            continue
                        fout.write(json.dumps(sample, ensure_ascii=False, allow_nan=False) + "\n")
                        fout.flush()
                        if saved_windows is not None:
                            saved_windows.add(win)
                        count += 1
                        progress.set_postfix(saved=count, refresh=False)
                        tqdm.write(
                            f"[saved] {source}: [{sample.get('start_second')}, "
                            f"{sample.get('end_second')}), new={count}",
                            file=sys.stderr,
                        )
                except (OSError, ValueError, KeyError, RuntimeError, cv2.error) as exc:
                    if strict:
                        raise
                    tqdm.write(f"[video {source}] {exc}", file=sys.stderr)

        tqdm.write(f"[summary] Finished re-extraction. Total new samples written: {count}.", file=sys.stderr)
        return count


Engine1 = Engine


if __name__ == "__main__":
    from ai.OptimalAlgorithsm.DataSystem.config import (
        RESUME, RUN_DEVICE, STRICT, VIDEO_ROOT,
    )

    parser = argparse.ArgumentParser(description="Re-extract neural network scores while keeping physical features.")
    parser.add_argument("--input", default=DEFAULT_INPUT_JSONL, help="Path to input physical JSONL")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_JSONL, help="Path to output visual JSONL")
    parser.add_argument("--video-root", default=VIDEO_ROOT, help="Root directory containing dataset videos")
    parser.add_argument("--device", default=RUN_DEVICE, help="Inference device (cuda or cpu)")
    parser.add_argument("--strict", action="store_true", default=STRICT, help="Raise on errors instead of skipping")
    parser.add_argument("--resume", action="store_true", default=RESUME, help="Resume from existing output JSONL")
    args = parser.parse_args()

    print("[init] Loading NeuralNetwork checkpoint for re-extraction...", file=sys.stderr, flush=True)
    engine = Engine(
        video_root=args.video_root,
        config=replace(DEFAULT_CONFIG, device=args.device),
    )
    total_written = engine.run(
        input_jsonl=args.input,
        output_jsonl=args.output,
        strict=args.strict,
        resume=args.resume,
    )
    print(f"[done] Extracted: {total_written} samples to {args.output}")
