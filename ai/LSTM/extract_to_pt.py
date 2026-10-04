#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: extract_to_pt.py
Mục đích:
    Trích xuất đặc trưng không gian đa tỉ lệ từ bộ dữ liệu video (chuẩn struct_dataset.md)
    thông qua mô hình ONNX PAFPN (checkpoints/backbone_neck.onnx), lưu trữ lũy tiến vào
    1 tệp HDF5 (.h5) duy nhất có tích hợp nén và chunking theo khung hình.

Đặc tính cốt lõi:
    1. Không qua Spatial Pooling: Bảo toàn nguyên vẹn 3 tensor đặc trưng 4D:
       - p3: [T, 64, 80, 80]
       - p4: [T, 128, 40, 40]
       - p5: [T, 256, 20, 20]
    2. Lưu trữ HDF5 (.h5) có nén: Ghi theo cơ chế Append ("a"), chunking (1, C, H, W)
       tối ưu cho PyTorch DataLoader lát cắt theo thời gian, thuật toán nén LZF hoặc GZIP.
    3. Tùy chỉnh tốc độ lấy mẫu: sample_interval (mặc định 0.1s ~ 10 FPS).
    4. Tăng cường dữ liệu: Tích hợp DetectionAugmenter từ src.augment với tính nhất quán
       thời gian (chung 1 random seed cho toàn bộ khung hình trong 1 video), áp dụng cho tập train.
    5. Cơ chế Resume & Chống chịu lỗi: Tự động bỏ qua các video đã hoàn thành thành công,
       tự động xóa và trích xuất lại các nhóm bị ngắt dở dang.
    6. Truy vết chi tiết: Ghi nhận thời gian thực vào file CSV theo dõi.
    7. Hiển thị tiến độ: Thanh tiến trình đa tầng trực quan qua tqdm.
"""

import os
import sys
import time
import json
import csv
import random
import math
import argparse
import hashlib
import gc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union, Set

# Đảm bảo console Windows hỗ trợ in tiếng Việt UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc vào sys.path để import an toàn
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Ngăn chặn xung đột OpenMP runtime và tối ưu hóa cấp phát CUDA
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ORT_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import h5py
from tqdm import tqdm

import torch

# Nạp thư viện CUDA DLL từ PyTorch nếu chạy trên môi trường Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError("Vui lòng cài đặt onnxruntime-gpu: pip install onnxruntime-gpu") from exc

from src.augment import DetectionAugmenter, config as DEFAULT_AUG_CONFIG


# ==============================================================================
# 1. CẤU TRÚC DỮ LIỆU ĐẠI DIỆN VIDEO (DATA STRUCTURES)
# ==============================================================================
@dataclass
class VideoClipItem:
    """Đại diện cho một video clip trong bộ dữ liệu."""
    path: Path
    split: str          # "train" hoặc "val"
    label: int          # 0 (alert) hoặc 1 (drowsy)
    label_name: str     # "0_alert" hoặc "1_drowsy"
    clip_name: str      # Tên tệp (vd: sust_n_1.mp4)
    video_id: str       # Khóa định danh (vd: train_0_alert_sust_n_1)
    source_dataset: str # Nguồn dataset (sust, uta-rldd, vbddd, unknown)
    subject_id: Optional[str] = None


# ==============================================================================
# 2. TIỀN XỬ LÝ ẢNH & LẤY MẪU KHUNG HÌNH (VIDEO SAMPLING)
# ==============================================================================
def letterbox(image: np.ndarray, new_size: int = 640, color: Tuple[int, int, int] = (114, 114, 114)) -> np.ndarray:
    """Resize ảnh giữ nguyên tỉ lệ (aspect ratio) với padding đồng màu (chuẩn 640x640)."""
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
    return canvas


class VideoSampler:
    """Đọc video và lấy mẫu khung hình theo chu kỳ thời gian (sample_interval)."""

    @staticmethod
    def sample_frames(
        video_path: Union[str, Path],
        sample_interval: float = 0.1,
        img_size: int = 640
    ) -> Tuple[List[np.ndarray], float, float]:
        """
        Đọc và lấy mẫu các khung hình trong video.

        Args:
            video_path: Đường dẫn tệp video.
            sample_interval: Khoảng cách thời gian (giây) giữa 2 frame lấy mẫu.
            img_size: Kích thước cạnh ảnh sau letterbox (mặc định 640).

        Returns:
            frames_rgb: Danh sách các ảnh RGB numpy uint8 [img_size, img_size, 3].
            fps: FPS gốc của video.
            duration_s: Thời lượng video tính bằng giây.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Không thể mở video qua OpenCV: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0.0

        step = max(1, int(round(fps * sample_interval)))
        frames_rgb: List[np.ndarray] = []
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            if frame_idx % step == 0:
                frame_lb = letterbox(frame_bgr, new_size=img_size)
                frame_rgb = cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)

            frame_idx += 1

        cap.release()
        return frames_rgb, float(fps), float(duration_s)


# ==============================================================================
# 3. ĐỘNG CƠ SUY LUẬN ONNX (STREAMING MINI-CHUNK FEATURE EXTRACTOR)
# ==============================================================================
class ONNXFeatureExtractor:
    """Quản lý suy luận mô hình ONNX Backbone+Neck trên GPU CUDA / CPU theo mini-chunk."""

    def __init__(self, model_path: Union[str, Path], device_id: int = 0):
        self.model_path = str(model_path)
        self.device_id = device_id

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số ONNX: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" in available_providers:
            cuda_options = {
                "device_id": self.device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            providers = [("CUDAExecutionProvider", cuda_options), "CPUExecutionProvider"]
            self.device_type = f"cuda:{self.device_id}"
        else:
            providers = ["CPUExecutionProvider"]
            self.device_type = "cpu"

        self.session = ort.InferenceSession(self.model_path, sess_options=sess_options, providers=providers)
        self.active_providers = self.session.get_providers()

        # Đọc thông tin I/O
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Xác định index vị trí cho p3, p4, p5
        self.idx_p3 = self.output_names.index("p3") if "p3" in self.output_names else 0
        self.idx_p4 = self.output_names.index("p4") if "p4" in self.output_names else 1
        self.idx_p5 = self.output_names.index("p5") if "p5" in self.output_names else 2

    def extract_chunks(
        self,
        frames_rgb: List[np.ndarray],
        chunk_size: int = 16,
        use_fp16: bool = True,
        show_inner_progress: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất đặc trưng nguyên bản p3, p4, p5 theo từng mini-chunk chống tràn VRAM.

        Returns:
            p3: np.ndarray [T, 64, 80, 80]
            p4: np.ndarray [T, 128, 40, 40]
            p5: np.ndarray [T, 256, 20, 20]
        """
        T = len(frames_rgb)
        dtype = np.float16 if use_fp16 else np.float32

        if T == 0:
            return (
                np.empty((0, 64, 80, 80), dtype=dtype),
                np.empty((0, 128, 40, 40), dtype=dtype),
                np.empty((0, 256, 20, 20), dtype=dtype)
            )

        p3_chunks: List[np.ndarray] = []
        p4_chunks: List[np.ndarray] = []
        p5_chunks: List[np.ndarray] = []

        chunk_range = range(0, T, chunk_size)
        if show_inner_progress and T > chunk_size:
            chunk_range = tqdm(
                chunk_range,
                desc="    -> Forwarding Chunks",
                leave=False,
                unit="chunk",
                dynamic_ncols=True
            )

        for i in chunk_range:
            batch_slice = frames_rgb[i : i + chunk_size]
            # Stack ảnh [B, H, W, 3] -> transpose sang [B, 3, H, W]
            batch_np = np.stack(batch_slice, axis=0).transpose(0, 3, 1, 2)
            batch_np = np.ascontiguousarray(batch_np, dtype=np.float32) / 255.0

            outputs = self.session.run(self.output_names, {self.input_name: batch_np})

            p3_chunk = outputs[self.idx_p3]
            p4_chunk = outputs[self.idx_p4]
            p5_chunk = outputs[self.idx_p5]

            if use_fp16:
                p3_chunk = p3_chunk.astype(np.float16)
                p4_chunk = p4_chunk.astype(np.float16)
                p5_chunk = p5_chunk.astype(np.float16)

            p3_chunks.append(p3_chunk)
            p4_chunks.append(p4_chunk)
            p5_chunks.append(p5_chunk)

            del batch_np, outputs

        p3_full = np.concatenate(p3_chunks, axis=0)
        p4_full = np.concatenate(p4_chunks, axis=0)
        p5_full = np.concatenate(p5_chunks, axis=0)
        return p3_full, p4_full, p5_full


# ==============================================================================
# 4. QUẢN LÝ LƯU TRỮ HDF5 (HDF5 PERSISTENCE & CHUNKING)
# ==============================================================================
class HDF5StorageManager:
    """Quản lý tệp HDF5 ghi lũy tiến (append mode), chunking theo frame và nén dữ liệu."""

    def __init__(self, h5_path: Union[str, Path], compression: str = "lzf", use_fp16: bool = True):
        self.h5_path = Path(h5_path)
        self.h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.compression = compression
        self.use_fp16 = use_fp16

        # Mở ở chế độ append ("a")
        self.h5_file = h5py.File(str(self.h5_path), "a")

    def is_video_completed(self, split: str, label_name: str, video_id: str) -> bool:
        """Kiểm tra video đã được ghi trọn vẹn và an toàn vào HDF5 hay chưa."""
        group_path = f"{split}/{label_name}/{video_id}"
        if group_path in self.h5_file:
            grp = self.h5_file[group_path]
            is_comp = grp.attrs.get("is_completed", False)
            if is_comp and ("p3" in grp) and ("p4" in grp) and ("p5" in grp):
                return True
        return False

    def save_video_features(
        self,
        split: str,
        label_name: str,
        video_id: str,
        p3: np.ndarray,
        p4: np.ndarray,
        p5: np.ndarray,
        attrs: Dict[str, Any]
    ) -> str:
        """
        Lưu 3 tensor đặc trưng p3, p4, p5 vào nhóm /{split}/{label_name}/{video_id}.

        Chunking (1, C, H, W) cho phép đọc từng frame độc lập cực nhanh khi train PyTorch.
        """
        group_path = f"{split}/{label_name}/{video_id}"

        # Nếu nhóm đã tồn tại dở dang do sự cố trước đó, xóa sạch để ghi lại
        if group_path in self.h5_file:
            del self.h5_file[group_path]

        grp = self.h5_file.create_group(group_path)
        T = p3.shape[0]

        # Cấu hình chunking theo từng frame
        chunk_p3 = (1, 64, 80, 80) if T > 0 else None
        chunk_p4 = (1, 128, 40, 40) if T > 0 else None
        chunk_p5 = (1, 256, 20, 20) if T > 0 else None

        comp_kwargs: Dict[str, Any] = {"compression": self.compression}
        if self.compression == "gzip":
            comp_kwargs["compression_opts"] = 4

        grp.create_dataset("p3", data=p3, chunks=chunk_p3, **comp_kwargs)
        grp.create_dataset("p4", data=p4, chunks=chunk_p4, **comp_kwargs)
        grp.create_dataset("p5", data=p5, chunks=chunk_p5, **comp_kwargs)

        # Ghi các thuộc tính metadata
        for k, v in attrs.items():
            if v is not None:
                grp.attrs[k] = v

        # Đánh dấu hoàn tất nguyên tử (Atomic transaction flag)
        grp.attrs["is_completed"] = True

        # Đẩy dữ liệu ngay xuống đĩa
        self.h5_file.flush()
        return group_path

    def flush(self):
        """Cam kết toàn bộ buffer xuống tệp đĩa cứng."""
        if self.h5_file:
            self.h5_file.flush()

    def close(self):
        """Đóng tệp HDF5 an toàn."""
        if self.h5_file:
            try:
                self.h5_file.flush()
                self.h5_file.close()
            except Exception:
                pass


# ==============================================================================
# 5. QUẢN LÝ TỆP CSV TRUY VẾT & PHỤC HỒI (MANIFEST CSV TRACKER)
# ==============================================================================
CSV_FIELDNAMES = [
    "video_id", "split", "label", "label_name", "orig_file",
    "source_dataset", "orig_duration_s", "orig_fps", "sample_interval",
    "num_frames", "p3_shape", "p4_shape", "p5_shape", "dtype",
    "compression", "is_augmented", "aug_seed", "h5_group_path",
    "status", "error_msg", "timestamp"
]


class ManifestCSVTracker:
    """Quản lý tệp CSV truy vết tiến trình và tra cứu phục hồi (Resume)."""

    def __init__(self, csv_path: Union[str, Path]):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.completed_video_ids: Set[str] = set()

        # Đọc dữ liệu cũ nếu tệp đã tồn tại
        if self.csv_path.exists() and self.csv_path.stat().st_size > 0:
            with open(self.csv_path, mode="r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("status") == "SUCCESS" and "video_id" in row:
                        self.completed_video_ids.add(row["video_id"])

        # Mở file ở chế độ append
        file_exists = self.csv_path.exists() and (self.csv_path.stat().st_size > 0)
        self._file = open(self.csv_path, mode="a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=CSV_FIELDNAMES)

        if not file_exists:
            self._writer.writeheader()
            self._file.flush()

    def is_completed(self, video_id: str) -> bool:
        """Tra cứu nhanh O(1) trạng thái đã hoàn thành."""
        return video_id in self.completed_video_ids

    def record_success(
        self,
        video_id: str,
        split: str,
        label: int,
        label_name: str,
        orig_file: str,
        source_dataset: str,
        orig_duration_s: float,
        orig_fps: float,
        sample_interval: float,
        num_frames: int,
        p3_shape: Tuple,
        p4_shape: Tuple,
        p5_shape: Tuple,
        dtype: str,
        compression: str,
        is_augmented: bool,
        aug_seed: int,
        h5_group_path: str
    ):
        """Ghi nhận bản ghi thành công."""
        row = {
            "video_id": video_id,
            "split": split,
            "label": label,
            "label_name": label_name,
            "orig_file": orig_file,
            "source_dataset": source_dataset,
            "orig_duration_s": round(orig_duration_s, 2),
            "orig_fps": round(orig_fps, 2),
            "sample_interval": sample_interval,
            "num_frames": num_frames,
            "p3_shape": str(p3_shape),
            "p4_shape": str(p4_shape),
            "p5_shape": str(p5_shape),
            "dtype": dtype,
            "compression": compression,
            "is_augmented": is_augmented,
            "aug_seed": aug_seed,
            "h5_group_path": h5_group_path,
            "status": "SUCCESS",
            "error_msg": "",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._writer.writerow(row)
        self._file.flush()
        self.completed_video_ids.add(video_id)

    def record_failure(
        self,
        video_id: str,
        split: str,
        label: int,
        label_name: str,
        orig_file: str,
        source_dataset: str,
        error_msg: str
    ):
        """Ghi nhận bản ghi thất bại."""
        row = {
            "video_id": video_id,
            "split": split,
            "label": label,
            "label_name": label_name,
            "orig_file": orig_file,
            "source_dataset": source_dataset,
            "orig_duration_s": 0.0,
            "orig_fps": 0.0,
            "sample_interval": 0.0,
            "num_frames": 0,
            "p3_shape": "",
            "p4_shape": "",
            "p5_shape": "",
            "dtype": "",
            "compression": "",
            "is_augmented": False,
            "aug_seed": -1,
            "h5_group_path": "",
            "status": "FAILED",
            "error_msg": str(error_msg),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        self._writer.writerow(row)
        self._file.flush()

    def close(self):
        """Đóng file CSV."""
        if self._file and not self._file.closed:
            self._file.flush()
            self._file.close()


# ==============================================================================
# 6. QUÉT VÀ THU THẬP BỘ DỮ LIỆU (DATASET SCANNING)
# ==============================================================================
def infer_source_dataset(filename: str) -> str:
    """Suy đoán nguồn dataset từ tiền tố tên video."""
    fn_lower = filename.lower()
    if fn_lower.startswith("sust_"):
        return "sust"
    if fn_lower.startswith("uta_") or "rldd" in fn_lower:
        return "uta-rldd"
    if fn_lower.startswith("vbddd_"):
        return "vbddd"
    return "unknown"


def scan_structured_dataset(
    data_dir: Union[str, Path],
    video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
) -> Tuple[List[VideoClipItem], List[VideoClipItem]]:
    """
    Quét trực tiếp cấu trúc cây thư mục:
        data_dir/train/0_alert/*.mp4, *.avi...
        data_dir/train/1_drowsy/*.mp4, *.avi...
        data_dir/val/0_alert/*.mp4, *.avi...
        data_dir/val/1_drowsy/*.mp4, *.avi...
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {data_dir}")

    train_items: List[VideoClipItem] = []
    val_items: List[VideoClipItem] = []

    def _scan_folder(folder_path: Path, split_name: str) -> List[VideoClipItem]:
        items = []
        if not folder_path.exists():
            return items

        label_subdirs = [
            ("0_alert", 0),
            ("1_drowsy", 1)
        ]

        for folder_name, label_val in label_subdirs:
            sub_p = folder_path / folder_name
            if not sub_p.exists():
                continue

            for ext in video_exts:
                for video_file in sub_p.glob(f"*{ext}"):
                    if not video_file.is_file():
                        continue

                    stem = video_file.stem
                    video_id = f"{split_name}_{folder_name}_{stem}"
                    source = infer_source_dataset(video_file.name)

                    item = VideoClipItem(
                        path=video_file,
                        split=split_name,
                        label=label_val,
                        label_name=folder_name,
                        clip_name=video_file.name,
                        video_id=video_id,
                        source_dataset=source
                    )
                    items.append(item)

        items.sort(key=lambda x: str(x.path))
        return items

    train_items = _scan_folder(data_dir / "train", "train")
    val_items = _scan_folder(data_dir / "val", "val")

    # Nếu không tìm thấy thư mục train/val, thử quét trực tiếp theo nhãn tại root
    if len(train_items) == 0 and len(val_items) == 0:
        print("[!] Không tìm thấy thư mục train/val, quét theo nhãn trực tiếp tại thư mục gốc...")
        direct_items = _scan_folder(data_dir, "train")
        train_items = direct_items

    return train_items, val_items


# ==============================================================================
# 7. ĐIỀU PHỐI TRÍCH XUẤT CHO TỪNG PHÂN TẬP (SPLIT EXTRACTION CONTROLLER)
# ==============================================================================
def process_split(
    split_name: str,
    items: List[VideoClipItem],
    extractor: ONNXFeatureExtractor,
    h5_manager: HDF5StorageManager,
    tracker: ManifestCSVTracker,
    sample_interval: float = 0.1,
    img_size: int = 640,
    chunk_size: int = 16,
    use_fp16: bool = True,
    augmenter: Optional[DetectionAugmenter] = None,
    num_aug: int = 1,
    include_original: bool = True,
    base_seed: int = 42,
    force_recompute: bool = False,
    gc_interval: int = 25
) -> Dict[str, int]:
    """
    Xử lý trích xuất và ghi HDF5 cho toàn bộ video trong 1 phân tập (train hoặc val).
    """
    stats = {"total": len(items), "processed": 0, "skipped": 0, "failed": 0, "aug_created": 0}
    if len(items) == 0:
        print(f"[*] Phân tập [{split_name.upper()}]: Không có video nào để xử lý.")
        return stats

    dtype_str = "float16" if use_fp16 else "float32"

    # Khởi tạo thanh tiến trình cấp cao qua tqdm
    pbar = tqdm(
        items,
        desc=f"[{split_name.upper()}] Extraction",
        unit="video",
        dynamic_ncols=True
    )

    for item_idx, item in enumerate(pbar):
        # Xác định các biến thể cần xử lý cho video này
        variants: List[Tuple[str, bool, int]] = []
        # Định dạng: (variant_id, is_aug, seed)

        if split_name == "train" and augmenter is not None:
            if include_original:
                variants.append((item.video_id, False, -1))
            for a_idx in range(1, num_aug + 1):
                aug_id = f"{item.video_id}_aug{a_idx:02d}"
                var_seed = (base_seed + abs(hash(item.video_id)) + a_idx) % (2**31 - 1)
                variants.append((aug_id, True, var_seed))
        else:
            variants.append((item.video_id, False, -1))

        # Kiểm tra xem toàn bộ các biến thể của video này đã hoàn thành chưa
        all_variants_done = True
        for var_id, _, _ in variants:
            is_csv_done = tracker.is_completed(var_id)
            is_h5_done = h5_manager.is_video_completed(item.split, item.label_name, var_id)
            if force_recompute or not (is_csv_done and is_h5_done):
                all_variants_done = False
                break

        if all_variants_done:
            stats["skipped"] += len(variants)
            pbar.set_postfix({
                "video": item.clip_name[:14],
                "skip": stats["skipped"],
                "status": "Skipped (Cached)"
            })
            continue

        # 1. Đọc và lấy mẫu video gốc
        try:
            raw_frames_rgb, orig_fps, duration_s = VideoSampler.sample_frames(
                video_path=item.path,
                sample_interval=sample_interval,
                img_size=img_size
            )
            T = len(raw_frames_rgb)
        except Exception as e:
            stats["failed"] += len(variants)
            for var_id, _, _ in variants:
                tracker.record_failure(
                    video_id=var_id,
                    split=item.split,
                    label=item.label,
                    label_name=item.label_name,
                    orig_file=str(item.path),
                    source_dataset=item.source_dataset,
                    error_msg=f"Lỗi đọc video: {e}"
                )
            pbar.set_postfix({"video": item.clip_name[:14], "status": f"Read Err: {str(e)[:15]}"})
            continue

        # 2. Xử lý từng biến thể (Gốc & Augment)
        for var_id, is_aug, seed in variants:
            # Kiểm tra riêng từng biến thể
            if not force_recompute and tracker.is_completed(var_id) and h5_manager.is_video_completed(item.split, item.label_name, var_id):
                stats["skipped"] += 1
                continue

            try:
                # Áp dụng Augment nếu là bản sao tăng cường
                if is_aug and augmenter is not None:
                    target_frames, _, _, _ = augmenter.augment_video(raw_frames_rgb, seed=seed)
                else:
                    target_frames = raw_frames_rgb

                # 3. Trích xuất đặc trưng nguyên bản qua ONNX
                p3, p4, p5 = extractor.extract_chunks(
                    frames_rgb=target_frames,
                    chunk_size=chunk_size,
                    use_fp16=use_fp16,
                    show_inner_progress=False
                )

                # 4. Ghi trực tiếp vào tệp HDF5
                attrs = {
                    "label": item.label,
                    "label_name": item.label_name,
                    "seq_len": T,
                    "sample_interval": sample_interval,
                    "orig_fps": orig_fps,
                    "orig_duration_s": duration_s,
                    "is_augmented": is_aug,
                    "aug_seed": seed,
                    "source_dataset": item.source_dataset,
                    "orig_file": str(item.path),
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
                }

                h5_group_path = h5_manager.save_video_features(
                    split=item.split,
                    label_name=item.label_name,
                    video_id=var_id,
                    p3=p3,
                    p4=p4,
                    p5=p5,
                    attrs=attrs
                )

                # 5. Ghi nhận thành công vào CSV truy vết
                tracker.record_success(
                    video_id=var_id,
                    split=item.split,
                    label=item.label,
                    label_name=item.label_name,
                    orig_file=str(item.path),
                    source_dataset=item.source_dataset,
                    orig_duration_s=duration_s,
                    orig_fps=orig_fps,
                    sample_interval=sample_interval,
                    num_frames=T,
                    p3_shape=p3.shape,
                    p4_shape=p4.shape,
                    p5_shape=p5.shape,
                    dtype=dtype_str,
                    compression=h5_manager.compression,
                    is_augmented=is_aug,
                    aug_seed=seed,
                    h5_group_path=h5_group_path
                )

                if is_aug:
                    stats["aug_created"] += 1
                stats["processed"] += 1

                del p3, p4, p5
                if is_aug:
                    del target_frames

            except Exception as e:
                stats["failed"] += 1
                tracker.record_failure(
                    video_id=var_id,
                    split=item.split,
                    label=item.label,
                    label_name=item.label_name,
                    orig_file=str(item.path),
                    source_dataset=item.source_dataset,
                    error_msg=f"Lỗi forward/lưu H5: {e}"
                )

        del raw_frames_rgb

        # Cập nhật thông số thời gian thực lên tqdm postfix
        gpu_mem_str = "N/A"
        if torch.cuda.is_available():
            alloc_mb = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_mem_str = f"{alloc_mb:.0f}MB"

        pbar.set_postfix({
            "video": item.clip_name[:14],
            "frames": T,
            "skip": stats["skipped"],
            "gpu": gpu_mem_str,
            "status": "OK"
        })

        # Thu gom rác bộ nhớ định kỳ
        if (item_idx + 1) % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()
    return stats


# ==============================================================================
# 8. HÀM ĐIỀU KHIỂN CHÍNH (MAIN CLI ENTRYPOINT)
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Trích xuất đặc trưng video không gian nguyên bản (p3, p4, p5) sang tệp HDF5 (.h5) nén."
    )

    # Đường dẫn thư mục & tệp
    parser.add_argument(
        "--data_dir", type=str, default=r"E:\LSTM\data_processed",
        help="Đường dẫn thư mục dataset phân tầng theo struct_dataset.md (mặc định: E:\\LSTM\\data_processed)"
    )
    parser.add_argument(
        "--output_h5", type=str, default="checkpoints/dataset_features.h5",
        help="Đường dẫn tệp .h5 đầu ra (mặc định: checkpoints/dataset_features.h5)"
    )
    parser.add_argument(
        "--manifest_csv", type=str, default="checkpoints/dataset_manifest.csv",
        help="Đường dẫn tệp .csv truy vết và phục hồi tiến trình (mặc định: checkpoints/dataset_manifest.csv)"
    )
    parser.add_argument(
        "--onnx_path", type=str, default="checkpoints/backbone_neck.onnx",
        help="Đường dẫn tệp trọng số ONNX Backbone + Neck"
    )

    # Tham số trích xuất & mô hình
    parser.add_argument(
        "--sample_interval", type=float, default=0.1,
        help="Khoảng thời gian (giây) giữa 2 khung hình lấy mẫu (mặc định: 0.1s ~ 10 FPS)"
    )
    parser.add_argument(
        "--img_size", type=int, default=640,
        help="Kích thước ảnh letterbox vuông đưa vào ONNX (mặc định: 640)"
    )
    parser.add_argument(
        "--chunk_size", type=int, default=16,
        help="Kích thước mini-batch khi forward ONNX trên GPU (mặc định: 16)"
    )
    parser.add_argument(
        "--device_id", type=int, default=0,
        help="CUDA GPU Device ID (mặc định: 0)"
    )
    parser.add_argument(
        "--compression", type=str, default="lzf", choices=["lzf", "gzip"],
        help="Thuật toán nén HDF5: 'lzf' (tối ưu tốc độ) hoặc 'gzip' (tối ưu dung lượng)"
    )
    parser.add_argument(
        "--fp16", action=argparse.BooleanOptionalAction, default=True,
        help="Lưu trữ đặc trưng dạng float16 để giảm 50%% dung lượng đĩa (mặc định: True)"
    )

    # Tăng cường dữ liệu (Augmentation từ src/augment.py)
    parser.add_argument(
        "--augment", action=argparse.BooleanOptionalAction, default=True,
        help="Bật/tắt tăng cường dữ liệu cho tập train (mặc định: True)"
    )
    parser.add_argument(
        "--num_aug", type=int, default=1,
        help="Số bản sao tăng cường cho mỗi video trong tập train (mặc định: 1)"
    )
    parser.add_argument(
        "--include_original", action=argparse.BooleanOptionalAction, default=True,
        help="Giữ lại video gốc bên cạnh các bản sao tăng cường (mặc định: True)"
    )
    parser.add_argument(
        "--aug_seed", type=int, default=42,
        help="Random seed cơ sở cho việc tăng cường dữ liệu (mặc định: 42)"
    )

    # Cơ chế Resume & Giới hạn chạy thử nghiệm
    parser.add_argument(
        "--force_recompute", action="store_true", default=False,
        help="Bắt buộc trích xuất lại từ đầu, bỏ qua cache và bản ghi đã có"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Giới hạn số lượng video để chạy thử nghiệm nhanh (ví dụ: --limit 2)"
    )
    parser.add_argument(
        "--gc_interval", type=int, default=25,
        help="Tần suất gọi gc.collect() và empty_cache (sau mỗi N video)"
    )

    args = parser.parse_args()

    # Chuyển đổi các đường dẫn
    data_dir = Path(args.data_dir)
    output_h5_path = Path(args.output_h5)
    manifest_csv_path = Path(args.manifest_csv)
    onnx_path = Path(args.onnx_path)

    print("=" * 80)
    print("      HỆ THỐNG TRÍCH XUẤT ĐẶC TRƯNG NGUYÊN BẢN SANG HDF5 (.H5) CÓ NÉN       ")
    print("       (KHÔNG SPATIAL POOLING - BẢO TOÀN ĐA TẦNG ĐẶC TRƯNG P3, P4, P5)       ")
    print("=" * 80)
    print(f"[*] Thư mục Dataset   : {data_dir}")
    print(f"[*] File HDF5 xuất ra : {output_h5_path}")
    print(f"[*] File CSV truy vết : {manifest_csv_path}")
    print(f"[*] Trọng số ONNX     : {onnx_path}")
    print(f"[*] Chu kỳ lấy mẫu    : {args.sample_interval}s (~{1.0 / args.sample_interval:.1f} FPS)")
    print(f"[*] Kích thước ảnh    : {args.img_size}x{args.img_size}")
    print(f"[*] Kích thước chunk  : {args.chunk_size} frames/batch")
    print(f"[*] Định dạng dữ liệu : {'Float16' if args.fp16 else 'Float32'}")
    print(f"[*] Thuật toán nén    : {args.compression.upper()}")
    print(f"[*] Tăng cường (Train): {'BẬT' if args.augment else 'TẮT'} (num_aug={args.num_aug}, include_orig={args.include_original})")
    print(f"[*] Chế độ Resume     : {'GHI ĐÈ (Force Recompute)' if args.force_recompute else 'TỰ ĐỘNG BỎ QUA ĐÃ CÓ (Auto Skip)'}")
    if args.limit:
        print(f"[*] CHẾ ĐỘ THỬ NGHIỆM : Giới hạn tối đa {args.limit} video mỗi tập!")
    print("=" * 80)

    # 1. Quét bộ dữ liệu
    print("[1/4] Đang quét cấu trúc thư mục dữ liệu...")
    train_items, val_items = scan_structured_dataset(data_dir)
    print(f"      Tìm thấy: {len(train_items)} clips tập Train, {len(val_items)} clips tập Val.")

    if args.limit:
        train_items = train_items[:args.limit]
        val_items = val_items[:args.limit]
        print(f"      Áp dụng --limit: Train={len(train_items)} clips, Val={len(val_items)} clips.")

    if len(train_items) == 0 and len(val_items) == 0:
        print("[!] Không tìm thấy video clip nào hợp lệ. Vui lòng kiểm tra lại đường dẫn dataset.")
        return

    # 2. Khởi tạo ONNX Runtime
    print(f"[2/4] Đang khởi tạo mô hình ONNX Runtime...")
    extractor = ONNXFeatureExtractor(model_path=onnx_path, device_id=args.device_id)
    print(f"      Active Execution Providers: {extractor.active_providers}")

    # 3. Khởi tạo Augmenter
    augmenter = None
    if args.augment and args.num_aug > 0:
        augmenter = DetectionAugmenter(DEFAULT_AUG_CONFIG)
        print(f"      Đã nạp DetectionAugmenter với cấu hình mặc định từ src.augment.")

    # 4. Khởi tạo HDF5 Storage Manager & Manifest CSV Tracker
    print(f"[3/4] Đang kết nối tệp HDF5 và CSV Manifest...")
    h5_manager = HDF5StorageManager(
        h5_path=output_h5_path,
        compression=args.compression,
        use_fp16=args.fp16
    )
    tracker = ManifestCSVTracker(csv_path=manifest_csv_path)
    print(f"      Đã nạp sẵn {len(tracker.completed_video_ids)} bản ghi đã hoàn thành từ CSV.")

    # 5. Thực thi trích xuất tuần tự
    print("\n[4/4] Bắt đầu quá trình trích xuất đặc trưng...")
    start_time = time.time()

    try:
        # Xử lý tập Train
        print(f"\n--- TIẾN HÀNH TRÍCH XUẤT TẬP TRAIN ({len(train_items)} CLIPS) ---")
        train_stats = process_split(
            split_name="train",
            items=train_items,
            extractor=extractor,
            h5_manager=h5_manager,
            tracker=tracker,
            sample_interval=args.sample_interval,
            img_size=args.img_size,
            chunk_size=args.chunk_size,
            use_fp16=args.fp16,
            augmenter=augmenter,
            num_aug=args.num_aug,
            include_original=args.include_original,
            base_seed=args.aug_seed,
            force_recompute=args.force_recompute,
            gc_interval=args.gc_interval
        )

        # Xử lý tập Val
        print(f"\n--- TIẾN HÀNH TRÍCH XUẤT TẬP VAL ({len(val_items)} CLIPS) ---")
        val_stats = process_split(
            split_name="val",
            items=val_items,
            extractor=extractor,
            h5_manager=h5_manager,
            tracker=tracker,
            sample_interval=args.sample_interval,
            img_size=args.img_size,
            chunk_size=args.chunk_size,
            use_fp16=args.fp16,
            augmenter=None,  # Không augment tập val
            num_aug=0,
            include_original=True,
            base_seed=args.aug_seed,
            force_recompute=args.force_recompute,
            gc_interval=args.gc_interval
        )

    except KeyboardInterrupt:
        print("\n[!] Phát hiện thao tác ngắt người dùng (Ctrl+C). Đang lưu và đóng file an toàn...")
    finally:
        h5_manager.close()
        tracker.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print("                  BÁO CÁO TỔNG KẾT TRÍCH XUẤT ĐẶC TRƯNG                  ")
    print("=" * 80)
    print(f"[*] Tổng thời gian thực thi: {elapsed:.2f} giây ({elapsed / 60.0:.2f} phút)")
    print(f"[*] Tập Train:")
    print(f"    - Thành công : {train_stats['processed']} mẫu")
    print(f"    - Bỏ qua     : {train_stats['skipped']} mẫu (đã có sẵn)")
    print(f"    - Lỗi        : {train_stats['failed']} mẫu")
    print(f"[*] Tập Val:")
    print(f"    - Thành công : {val_stats['processed']} mẫu")
    print(f"    - Bỏ qua     : {val_stats['skipped']} mẫu (đã có sẵn)")
    print(f"    - Lỗi        : {val_stats['failed']} mẫu")
    print(f"[*] Tệp HDF5 lưu trữ       : {output_h5_path.resolve()}")
    if output_h5_path.exists():
        size_mb = output_h5_path.stat().st_size / (1024 ** 2)
        print(f"    Dung lượng tệp HDF5    : {size_mb:.2f} MB")
    print(f"[*] Tệp CSV theo dõi       : {manifest_csv_path.resolve()}")
    print("=" * 80)
    print("[SUCCESS] Hoàn tất pipeline trích xuất đặc trưng an toàn!")


if __name__ == "__main__":
    main()
