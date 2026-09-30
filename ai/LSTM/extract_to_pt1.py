#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
File: extract_to_pt1.py
Mục đích:
    Triển khai GIAI ĐOẠN 1 của Pipeline Huấn Luyện Nhận Diện Tài Xế Buồn Ngủ (Deep LSTM),
    hoạt động trực tiếp trên cấu trúc thư mục dữ liệu chuẩn đã phân chia theo tài liệu struct_dataset.md,
    kết hợp CƠ CHẾ TỐI ƯU HÓA BỘ NHỚ RAM NÂNG CAO (Zero RAM Accumulation):

    Cấu trúc thư mục hỗ trợ:
    E:\LSTM\data_processed\
    ├── dataset_merged_split.csv               # (Tùy chọn) Bảng tra cứu toàn bộ clip
    ├── dataset_merged_split.json              # (Tùy chọn) Metadata nạp cho Dataset
    ├── train\                                 # TẬP HUẤN LUYỆN (~80% clips)
    │   ├── 0_alert\                           # Nhãn 0: Tỉnh táo / lái xe bình thường (.mp4, .avi, ...)
    │   └── 1_drowsy\                          # Nhãn 1: Buồn ngủ / ngủ gật
    └── val\                                   # TẬP KIỂM ĐỊNH (~20% clips)
        ├── 0_alert\                           # Nhãn 0: Kiểm định tỉnh táo
        └── 1_drowsy\                          # Nhãn 1: Kiểm định buồn ngủ

    Các cơ chế tối ưu hóa RAM chính (Zero RAM Accumulation Architecture):
    1. Tách rời hoàn toàn Pha Trích Xuất (Phase 1: Extract & Cache) và Pha Đóng Gói (Phase 2: Packaging):
       - Trong suốt vòng lặp duyệt hàng nghìn video: CHỈ ghi đặc trưng xuống tệp đĩa tạm `cache/{split}/{video_id}.pt`.
       - Không tích lũy các Tensor p3, p4, p5 vào RAM trong vòng lặp video, giữ RAM ở mức O(1).
    2. Streaming Mini-Chunk Processing:
       - Không chuyển đổi toàn bộ danh sách ảnh RGB của video thành Tensor Float32 khổng lồ [T, 3, 640, 640] gây đỉnh RAM.
       - Chỉ chuyển đổi và đẩy lên GPU đúng kích thước `chunk_size` (24 khung hình) cho mỗi bước forward.
    3. Ép kiểu Float16 sớm (Early FP16):
       - Nếu bật `--fp16`, ép kiểu sang half() ngay khi ra khỏi AdaptiveAvgPool2d và lưu trực tiếp vào cache.
       - Giảm 50% dung lượng đĩa cache và triệt tiêu hoàn toàn đỉnh RAM do Double Buffering ở cuối.
    4. Thu gom rác chủ động (Garbage Collection) & Dọn PyTorch CUDA Cache:
       - Tự động kích hoạt gc.collect() và torch.cuda.empty_cache() định kỳ sau mỗi `gc_interval` video (mặc định 50).
       - Hiển thị thông số RAM thực tế (RSS Memory) thời gian thực trên thanh tiến trình tqdm.
    5. Hỗ trợ Phân Mảnh Tệp (.pt Sharding) & Cache-Only Mode:
       - Hỗ trợ cờ `--shard_size` để phân mảnh khi dataset lên tới hàng chục nghìn clip.
       - Hỗ trợ cờ `--skip_package` / `--cache_only` nếu muốn dừng sau khi trích xuất cache.
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
from typing import List, Tuple, Optional, Dict, Any, Union

# Đảm bảo console Windows hỗ trợ in UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục hiện tại vào sys.path để các tiến trình multiprocessing nạp module an toàn
current_file_dir = str(Path(__file__).resolve().parent)
if current_file_dir not in sys.path:
    sys.path.insert(0, current_file_dir)

# Ngăn chặn xung đột OpenMP runtime và tối ưu cấp phát bộ nhớ CUDA
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["ORT_LOG_LEVEL"] = "3"

# Tích hợp theo dõi bộ nhớ qua psutil nếu có sẵn
try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import cv2
from tqdm import tqdm

from config import TrainConfig
from augment import DetectionAugmenter, config as DEFAULT_AUG_CONFIG

# Nạp thư viện CUDA DLL từ PyTorch nếu chạy trên môi trường Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception as e:
        print(f"[WARN] Không thể thêm torch DLL directory: {e}")

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("Vui lòng cài đặt onnxruntime-gpu: pip install onnxruntime-gpu") from e


def get_current_ram_gb() -> float:
    """Lấy dung lượng RAM thực tế (Resident Set Size - RSS) đang được tiến trình sử dụng (GB)."""
    if _PSUTIL_AVAILABLE:
        try:
            return psutil.Process().memory_info().rss / (1024 ** 3)
        except Exception:
            return 0.0
    return 0.0


# Xác định đường dẫn dataset mặc định theo tài liệu struct_dataset.md
STRUCT_DATASET_DEFAULT_PATH = r"E:\LSTM\data_processed"
if os.path.exists(STRUCT_DATASET_DEFAULT_PATH):
    DEFAULT_DATA_DIR = STRUCT_DATASET_DEFAULT_PATH
elif hasattr(TrainConfig, "dataset_dir") and os.path.exists(TrainConfig.dataset_dir):
    DEFAULT_DATA_DIR = TrainConfig.dataset_dir
else:
    DEFAULT_DATA_DIR = STRUCT_DATASET_DEFAULT_PATH

DEFAULT_SEQ_LEN = getattr(TrainConfig, "seq_len", None)  # Mặc định None (chuỗi động)
DEFAULT_SAMPLE_INTERVAL = 0.1  # Mặc định 0.1s (Target FPS = 10.0 theo kế hoạch tăng tốc)
DEFAULT_IMAGE_SIZE = TrainConfig.image_size[0] if isinstance(TrainConfig.image_size, (tuple, list)) else TrainConfig.image_size
DEFAULT_VIDEO_EXTS = ",".join(TrainConfig.video_exts) if isinstance(TrainConfig.video_exts, (tuple, list)) else ".mp4,.avi,.mkv,.mov"


# ==============================================================================
# DATA STRUCTURE: ĐẠI DIỆN MỖI MẪU VIDEO CLIP
# ==============================================================================
@dataclass
class VideoClipItem:
    video_path: Path
    label: int  # 0: Alert, 1: Drowsy
    clip_id: str
    video_id: str
    split: str  # 'train' hoặc 'val'
    dataset: str = "unknown"  # 'sust', 'uta-rldd', 'vbddd', etc.
    subject_id: str = "unknown"


# ==============================================================================
# 1. TIỀN XỬ LÝ ẢNH & RUNTIME ONNX VỚI CUDA ZERO-COPY
# ==============================================================================
def letterbox(image: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """Resize ảnh giữ nguyên tỉ lệ (aspect ratio) với padding đồng màu (mặc định 640x640)."""
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


class CloneONNXCUDARuntime:
    """Quản lý suy luận ONNX Runtime trên GPU CUDA với I/O Binding (hỗ trợ CPU fallback)."""

    def __init__(self, onnx_model_path: str, device_id: int = 0, cudnn_algo: str = "EXHAUSTIVE"):
        self.onnx_model_path = str(onnx_model_path)
        self.device_id = device_id
        self.cudnn_algo = str(cudnn_algo).upper()

        if not os.path.exists(self.onnx_model_path):
            raise FileNotFoundError(f"Không tìm thấy file ONNX: {self.onnx_model_path}")

        self.session_options = ort.SessionOptions()
        self.session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session_options.log_severity_level = 3

        cuda_options = {
            "device_id": self.device_id,
            "arena_extend_strategy": "kNextPowerOfTwo",
            "cudnn_conv_algo_search": self.cudnn_algo,
            "cudnn_conv_use_max_workspace": "1",
            "do_copy_in_default_stream": True,
        }
        providers = [
            ("CUDAExecutionProvider", cuda_options),
            "CPUExecutionProvider"
        ]

        print(f"[+] Khởi tạo ONNX Runtime InferenceSession trên CUDA:{self.device_id}...")
        self.session = ort.InferenceSession(
            self.onnx_model_path,
            sess_options=self.session_options,
            providers=providers,
        )

        active = self.session.get_providers()
        print(f"[+] Active Providers: {active}")
        self.use_cuda = "CUDAExecutionProvider" in active
        if not self.use_cuda:
            print(f"[WARN] CUDAExecutionProvider không khả dụng. Fallback sang CPUExecutionProvider.")

        input_meta = self.session.get_inputs()[0]
        self.input_name = input_meta.name
        self.expected_h = input_meta.shape[2] if len(input_meta.shape) > 2 and isinstance(input_meta.shape[2], int) else None
        self.expected_w = input_meta.shape[3] if len(input_meta.shape) > 3 and isinstance(input_meta.shape[3], int) else None
        self.output_names = [o.name for o in self.session.get_outputs()]

    def forward(self, input_tensor: torch.Tensor) -> List[torch.Tensor]:
        """Suy luận trực tiếp: Zero-copy CUDA I/O Binding nếu có CUDA, hoặc session.run thông thường nếu CPU."""
        if self.use_cuda:
            if not input_tensor.is_cuda:
                input_tensor = input_tensor.to(f"cuda:{self.device_id}", non_blocking=True)
            if not input_tensor.is_contiguous():
                input_tensor = input_tensor.contiguous()

            io_binding = self.session.io_binding()
            io_binding.bind_input(
                name=self.input_name,
                device_type="cuda",
                device_id=self.device_id,
                element_type=np.float32,
                shape=tuple(input_tensor.shape),
                buffer_ptr=input_tensor.data_ptr(),
            )
            for out_name in self.output_names:
                io_binding.bind_output(out_name, device_type="cuda", device_id=self.device_id)

            self.session.run_with_iobinding(io_binding)
            raw_outputs = io_binding.get_outputs()
            return [torch.from_dlpack(out) for out in raw_outputs]
        else:
            np_in = input_tensor.detach().cpu().numpy()
            if not np_in.flags.c_contiguous:
                np_in = np.ascontiguousarray(np_in)
            outs = self.session.run(self.output_names, {self.input_name: np_in})
            return [torch.from_numpy(out) for out in outs]


# ==============================================================================
# 2. HÀM ĐỌC VIDEO & TRÍCH XUẤT ĐẶC TRƯNG TỐI ƯU BỘ NHỚ
# ==============================================================================
def read_and_sample_video_frames(
        video_path: Path,
        seq_len: Optional[int] = None,
        sample_interval: float = 0.1,
        img_size: int = 640
) -> Tuple[List[np.ndarray], int]:
    """
    Đọc nhanh video clip, letterbox sang kích thước cố định và định dạng RGB (uint8).
    Áp dụng Sequential Stream Decoding một chiều (loại bỏ hoàn toàn cap.set overhead).
    Đồng bộ logic lấy mẫu thời gian:
    - Cứ mỗi khoảng thời gian sample_interval (mặc định 0.1s - Target FPS 10.0) lấy 1 khung hình qua frame_step = sample_interval * fps.
    - Nếu seq_len is None: Lấy toàn bộ các khung hình thực tế của clip theo chu kỳ t (không padding).
    - Nếu seq_len is not None: Pad lặp lại khung hình cuối cùng hoặc cắt ngắn về đúng seq_len.

    Returns:
        frames_rgb: Danh sách ảnh numpy RGB uint8 [img_size, img_size, 3]
        actual_frame_count: Số khung hình thực tế trích xuất được từ video trước khi padding
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0 or math.isnan(fps) or math.isinf(fps):
        fps = 30.0  # Mặc định 30 FPS nếu header không chứa thông tin hợp lệ

    frame_step = max(sample_interval * fps, 1.0)
    frames = []

    if total_frames > 0:
        indices = []
        k = 0
        while True:
            idx = int(round(k * frame_step))
            if idx >= total_frames:
                break
            indices.append(idx)
            k += 1
            if seq_len is not None and len(indices) >= seq_len:
                break

        if len(indices) == 0 and total_frames > 0:
            indices = [0]

        indices_set = set(indices)
        max_target_idx = max(indices) if indices else 0

        cur_idx = 0
        while cur_idx <= max_target_idx:
            ret, frame = cap.read()
            if not ret:
                break
            if cur_idx in indices_set:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(letterbox(rgb, new_size=img_size))
            cur_idx += 1
        cap.release()

        actual_frames = len(frames)
        if actual_frames == 0:
            actual_frames = 1
            frames = [np.zeros((img_size, img_size, 3), dtype=np.uint8)]

        # Chỉ thực hiện padding nếu seq_len được chỉ định cụ thể
        if seq_len is not None:
            while len(frames) < seq_len:
                frames.append(frames[-1].copy())
            frames = frames[:seq_len]

        return frames, actual_frames

    # Fallback đọc tuần tự nếu header không ghi tổng số frame (stream không rõ độ dài)
    raw_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    n_raw = len(raw_frames)
    if n_raw == 0:
        raise ValueError(f"Video không chứa khung hình nào: {video_path}")

    indices = []
    k = 0
    while True:
        idx = int(round(k * frame_step))
        if idx >= n_raw:
            break
        indices.append(idx)
        k += 1
        if seq_len is not None and len(indices) >= seq_len:
            break

    for idx in indices:
        rgb = cv2.cvtColor(raw_frames[idx], cv2.COLOR_BGR2RGB)
        lb = letterbox(rgb, new_size=img_size)
        frames.append(lb)

    del raw_frames

    actual_frames = len(frames)
    if actual_frames == 0:
        actual_frames = 1
        frames = [np.zeros((img_size, img_size, 3), dtype=np.uint8)]

    if seq_len is not None:
        while len(frames) < seq_len:
            frames.append(frames[-1].copy())
        frames = frames[:seq_len]

    return frames, actual_frames


def forward_video_chunks(
        frames_rgb: List[np.ndarray],
        runner: CloneONNXCUDARuntime,
        chunk_size: int = 64,
        device: str = "cuda:0",
        use_fp16: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward danh sách khung hình RGB qua mô hình ONNX theo từng mini-chunk
    kết hợp Adaptive Average Pooling (1, 1).

    Cơ chế tối ưu hóa tốc độ & RAM:
    - Truyền trực tiếp mảng NumPy uint8 qua bus PCIe (giảm 75% tải bus so với Float32).
    - Chuẩn hóa .float().div_(255.0) song song trực tiếp trên GPU CUDA Tensor Cores.
    - Áp dụng Early FP16 ngay sau pooling nếu use_fp16=True để tiết kiệm 50% RAM/Disk.

    Returns:
        p3_tensor: [T, 64] on CPU (float16 nếu use_fp16, ngược lại float32)
        p4_tensor: [T, 128] on CPU
        p5_tensor: [T, 256] on CPU
    """
    seq_len = len(frames_rgb)
    target_dtype = torch.float16 if use_fp16 else torch.float32

    if seq_len == 0:
        return (
            torch.empty((0, 64), dtype=target_dtype),
            torch.empty((0, 128), dtype=target_dtype),
            torch.empty((0, 256), dtype=target_dtype)
        )

    # Ánh xạ index theo tên output thực tế ('p3', 'p4', 'p5')
    name_to_idx = {name: i for i, name in enumerate(runner.output_names)}
    idx_p3 = name_to_idx.get("p3", 0)
    idx_p4 = name_to_idx.get("p4", 1)
    idx_p5 = name_to_idx.get("p5", 2)

    target_device = device if (runner.use_cuda and torch.cuda.is_available()) else "cpu"

    p3_chunks, p4_chunks, p5_chunks = [], [], []
    for i in range(0, seq_len, chunk_size):
        chunk_frames = frames_rgb[i : i + chunk_size]
        # Gom mảng uint8 dạng NCHW
        chunk_np = np.stack([np.ascontiguousarray(f.transpose(2, 0, 1)) for f in chunk_frames], axis=0)
        # Chuyển uint8 sang GPU qua PCIe (tiết kiệm 75% PCIe bandwidth)
        chunk_tensor = torch.from_numpy(chunk_np).to(target_device, non_blocking=True)
        # Chuẩn hóa trên GPU
        chunk_tensor = chunk_tensor.float().div_(255.0)

        outs = runner.forward(chunk_tensor)

        # Adaptive Average Pooling về vector 1D
        p3_pool = F.adaptive_avg_pool2d(outs[idx_p3], (1, 1)).flatten(1).cpu()
        p4_pool = F.adaptive_avg_pool2d(outs[idx_p4], (1, 1)).flatten(1).cpu()
        p5_pool = F.adaptive_avg_pool2d(outs[idx_p5], (1, 1)).flatten(1).cpu()

        # Ép kiểu Float16 sớm ngay sau pooling để giảm áp lực bộ nhớ
        if use_fp16:
            p3_pool = p3_pool.half()
            p4_pool = p4_pool.half()
            p5_pool = p5_pool.half()

        p3_chunks.append(p3_pool)
        p4_chunks.append(p4_pool)
        p5_chunks.append(p5_pool)

        del chunk_tensor, chunk_np, outs

    p3_tensor = torch.cat(p3_chunks, dim=0)  # [T, 64]
    p4_tensor = torch.cat(p4_chunks, dim=0)  # [T, 128]
    p5_tensor = torch.cat(p5_chunks, dim=0)  # [T, 256]
    return p3_tensor, p4_tensor, p5_tensor


def extract_single_video_features(
        video_path: Path,
        runner: CloneONNXCUDARuntime,
        seq_len: Optional[int] = None,
        sample_interval: float = 0.1,
        img_size: int = 640,
        chunk_size: int = 64,
        device: str = "cuda:0",
        use_fp16: bool = True,
        augmenter: Optional[DetectionAugmenter] = None,
        aug_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Hàm phụ trợ trích xuất đặc trưng cho một video clip đơn lẻ."""
    frames_rgb, actual_frames = read_and_sample_video_frames(
        video_path=video_path,
        seq_len=seq_len,
        sample_interval=sample_interval,
        img_size=img_size
    )

    if augmenter is not None:
        frames_rgb, _, _, _ = augmenter.augment_video(
            frames=frames_rgb,
            boxes_list=None,
            labels_list=None,
            seed=aug_seed
        )

    p3_tensor, p4_tensor, p5_tensor = forward_video_chunks(
        frames_rgb=frames_rgb,
        runner=runner,
        chunk_size=chunk_size,
        device=device,
        use_fp16=use_fp16
    )

    del frames_rgb
    return p3_tensor, p4_tensor, p5_tensor, actual_frames


# ==============================================================================
# 3. QUẢN LÝ DỮ LIỆU THEO CẤU TRÚC STRUCT_DATASET.MD
# ==============================================================================
def resolve_label_from_folder_or_name(item_path: Path) -> int:
    """
    Xác định nhãn nhị phân (0: Tỉnh táo/Alert, 1: Buồn ngủ/Drowsy)
    dựa vào tên thư mục cha hoặc tên tệp theo quy ước của struct_dataset.md.
    """
    parent = item_path.parent.name.lower()
    if parent.startswith("0") or "alert" in parent or "normal" in parent or "driving" in parent:
        return 0
    if parent.startswith("1") or "drowsy" in parent or "drowsiness" in parent:
        return 1

    # Fallback kiểm tra stem nếu thư mục cha không chuẩn
    stem = item_path.stem.lower()
    if "drowsiness" in stem or "drowsy" in stem or stem.startswith("d_") or stem == "d" or "_label10_" in stem or "_label1_" in stem:
        return 1
    if "driving" in stem or "alert" in stem or "normal" in stem or stem.startswith("n_") or stem == "n" or "_label0_" in stem:
        return 0

    raise ValueError(f"Không thể xác định nhãn từ thư mục cha hoặc tên video: {item_path}")


def infer_dataset_source(clip_name: str) -> str:
    """Suy đoán tập dữ liệu gốc từ tiền tố tên clip (sust, uta-rldd, vbddd)."""
    name_l = clip_name.lower()
    if name_l.startswith("sust"):
        return "sust"
    if name_l.startswith("uta"):
        return "uta-rldd"
    if name_l.startswith("vbddd"):
        return "vbddd"
    return "other"


def load_dataset_from_json(
        json_path: Path,
        data_root: Path
) -> Tuple[List[VideoClipItem], List[VideoClipItem], Dict[str, Any]]:
    """Nạp danh sách clip Train và Validation từ tệp metadata dataset_merged_split.json."""
    with open(json_path, "r", encoding="utf-8") as f:
        meta_data = json.load(f)

    train_clips_raw = meta_data.get("train_clips", [])
    val_clips_raw = meta_data.get("val_clips", [])

    train_items: List[VideoClipItem] = []
    val_items: List[VideoClipItem] = []

    for c in train_clips_raw:
        rel_p = c.get("processed_path", "")
        vid_path = data_root / rel_p
        if not vid_path.exists():
            # Thử tìm trực tiếp theo tên file trong data_root/train
            alt_path = data_root / "train" / ("1_drowsy" if c.get("binary_label", 0) == 1 else "0_alert") / Path(rel_p).name
            if alt_path.exists():
                vid_path = alt_path

        train_items.append(VideoClipItem(
            video_path=vid_path,
            label=int(c.get("binary_label", 0)),
            clip_id=str(c.get("clip_id", vid_path.stem)),
            video_id=vid_path.stem,
            split="train",
            dataset=str(c.get("dataset", infer_dataset_source(vid_path.name))),
            subject_id=str(c.get("subject_id", "unknown"))
        ))

    for c in val_clips_raw:
        rel_p = c.get("processed_path", "")
        vid_path = data_root / rel_p
        if not vid_path.exists():
            alt_path = data_root / "val" / ("1_drowsy" if c.get("binary_label", 0) == 1 else "0_alert") / Path(rel_p).name
            if alt_path.exists():
                vid_path = alt_path

        val_items.append(VideoClipItem(
            video_path=vid_path,
            label=int(c.get("binary_label", 0)),
            clip_id=str(c.get("clip_id", vid_path.stem)),
            video_id=vid_path.stem,
            split="val",
            dataset=str(c.get("dataset", infer_dataset_source(vid_path.name))),
            subject_id=str(c.get("subject_id", "unknown"))
        ))

    return train_items, val_items, meta_data


def scan_dataset_from_directory(
        data_root: Path,
        video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
) -> Tuple[List[VideoClipItem], List[VideoClipItem], Dict[str, Any]]:
    """
    Quét trực tiếp cấu trúc cây thư mục:
        data_root/train/0_alert/*.mp4, *.avi
        data_root/train/1_drowsy/*.mp4, *.avi
        data_root/val/0_alert/*.mp4, *.avi
        data_root/val/1_drowsy/*.mp4, *.avi
    """
    norm_exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in video_exts}

    def _scan_split_folder(split_folder: Path, split_name: str) -> List[VideoClipItem]:
        items = []
        if not split_folder.exists():
            return items

        for sub_dir in sorted(split_folder.iterdir()):
            if not sub_dir.is_dir():
                continue

            sub_name = sub_dir.name.lower()
            if sub_name.startswith("0") or "alert" in sub_name or "normal" in sub_name or "driving" in sub_name:
                sub_label = 0
            elif sub_name.startswith("1") or "drowsy" in sub_name or "drowsiness" in sub_name:
                sub_label = 1
            else:
                continue

            for vf in sorted(sub_dir.rglob("*")):
                if vf.is_file() and vf.suffix.lower() in norm_exts:
                    items.append(VideoClipItem(
                        video_path=vf,
                        label=sub_label,
                        clip_id=vf.stem,
                        video_id=vf.stem,
                        split=split_name,
                        dataset=infer_dataset_source(vf.name),
                        subject_id=vf.stem.split("-")[0] if "-" in vf.stem else vf.stem.split("_")[0]
                    ))
        return items

    train_folder = data_root / "train"
    val_folder = data_root / "val"
    if not val_folder.exists():
        val_folder = data_root / "validation"

    train_items = _scan_split_folder(train_folder, "train")
    val_items = _scan_split_folder(val_folder, "val")

    metadata = {
        "source": "folder_scan",
        "train_clips_count": len(train_items),
        "val_clips_count": len(val_items),
        "total_clips": len(train_items) + len(val_items),
    }

    return train_items, val_items, metadata


def prepare_structured_dataset(
        data_dir: Path,
        split_file: Optional[Path] = None,
        force_folder_scan: bool = False,
        video_exts: Tuple[str, ...] = (".mp4", ".avi", ".mkv", ".mov")
) -> Tuple[List[VideoClipItem], List[VideoClipItem], Dict[str, Any]]:
    """
    Điểm vào quản lý dữ liệu theo cấu trúc struct_dataset.md:
    1. Ưu tiên nạp metadata từ dataset_merged_split.json (nếu có và không ép scan).
    2. Fallback quét trực tiếp thư mục train/ và val/ (0_alert vs 1_drowsy).
    """
    if not data_dir.exists():
        raise FileNotFoundError(f"Thư mục chứa dữ liệu không tồn tại: {data_dir.resolve()}")

    target_json = split_file if (split_file and split_file.exists()) else (data_dir / "dataset_merged_split.json")

    if not force_folder_scan and target_json.exists():
        print(f"[+] Tìm thấy tệp metadata cấu hình: {target_json.resolve()}")
        try:
            train_items, val_items, meta = load_dataset_from_json(target_json, data_dir)
            existing_count = sum(1 for item in train_items[:10] if item.video_path.exists())
            if existing_count > 0:
                print(f"[+] Nạp thành công từ JSON: {len(train_items)} train clips, {len(val_items)} val clips.")
                return train_items, val_items, meta
            else:
                print(f"[WARN] Đường dẫn trong JSON không khớp tệp thực tế trên đĩa, chuyển sang quét trực tiếp thư mục...")
        except Exception as e:
            print(f"[WARN] Lỗi khi nạp từ JSON ({e}), chuyển sang quét trực tiếp thư mục...")

    print(f"[*] Đang quét cây thư mục dữ liệu tại: {data_dir.resolve()}...")
    train_items, val_items, meta = scan_dataset_from_directory(data_dir, video_exts=video_exts)
    print(f"[+] Quét hoàn tất: {len(train_items)} train clips, {len(val_items)} val clips.")

    if len(train_items) == 0 and len(val_items) == 0:
        raise FileNotFoundError(
            f"Không tìm thấy video clip nào trong {data_dir.resolve()}.\n"
            f"Vui lòng đảm bảo cấu trúc gồm các thư mục:\n"
            f"  {data_dir / 'train' / '0_alert'}\n"
            f"  {data_dir / 'train' / '1_drowsy'}\n"
            f"  {data_dir / 'val' / '0_alert'}\n"
            f"  {data_dir / 'val' / '1_drowsy'}"
        )

    return train_items, val_items, meta


# ==============================================================================
# 4. QUY TRÌNH ĐÓNG GÓI TIẾT KIỆM BỘ NHỚ (CONSOLIDATION PHASE)
# ==============================================================================
def consolidate_split_features(
        split_name: str,
        cache_files: List[Path],
        output_pt_path: Path,
        seq_len: Optional[int] = None,
        sample_interval: float = 0.1,
        use_fp16: bool = True,
        is_augmented_split: bool = False,
        num_aug: int = 0,
        actual_include_original: bool = True,
        shard_size: Optional[int] = 5000
) -> None:
    """
    Pha 2: Đóng gói các đặc trưng từ cache trên đĩa thành tệp .pt hoàn chỉnh.
    Chỉ nạp tensor vào RAM trong pha này và giải phóng ngay sau khi lưu.
    Hỗ trợ cả chế độ gộp 1 file đơn lẻ lẫn chế độ phân mảnh (--shard_size).
    """
    if len(cache_files) == 0:
        print(f"[WARN] Không có tệp cache nào để đóng gói cho tập {split_name.upper()}!")
        return

    print(f"\n{'-' * 75}")
    print(f"[*] BẮT ĐẦU PHA 2: ĐÓNG GÓI TENSOR CHO PHÂN TẬP: {split_name.upper()} ({len(cache_files)} MẪU)")
    print(f"[*] Đường dẫn xuất mục tiêu: {output_pt_path.resolve()}")
    if shard_size and shard_size > 0:
        print(f"[*] Chế độ phân mảnh       : BẬT (Mỗi shard tối đa {shard_size} mẫu)")
    else:
        print(f"[*] Chế độ phân mảnh       : TẮT (Gộp 1 file duy nhất tương thích PreloadedTensorDataset)")
    print(f"{'-' * 75}")

    def _package_single_shard(sub_files: List[Path], target_path: Path, part_idx: Optional[int] = None) -> Dict[str, Any]:
        p3_list, p4_list, p5_list = [], [], []
        labels_list, video_ids_list, seq_lens_list = [], [], []
        datasets_list, subjects_list, clip_ids_list = [], [], []

        desc_label = f"[Đóng gói {split_name.upper()}{f' Part {part_idx}' if part_idx is not None else ''}]"
        for cf in tqdm(sub_files, desc=desc_label):
            if not cf.exists():
                continue
            cdata = torch.load(cf, map_location="cpu")
            p3 = cdata["p3"]
            p4 = cdata["p4"]
            p5 = cdata["p5"]

            if use_fp16:
                if p3.dtype != torch.float16:
                    p3 = p3.half()
                    p4 = p4.half()
                    p5 = p5.half()

            p3_list.append(p3)
            p4_list.append(p4)
            p5_list.append(p5)
            labels_list.append(cdata["label"])
            video_ids_list.append(cdata["video_id"])
            clip_ids_list.append(cdata.get("clip_id", cdata["video_id"]))
            seq_lens_list.append(cdata.get("actual_frames", p3.shape[0]))
            if "dataset" in cdata:
                datasets_list.append(cdata["dataset"])
            if "subject_id" in cdata:
                subjects_list.append(cdata["subject_id"])

            del cdata

        if len(video_ids_list) == 0:
            return {}

        final_labels = torch.tensor(labels_list, dtype=torch.long)
        final_seq_lens = torch.tensor(seq_lens_list, dtype=torch.int32)

        if seq_len is None:
            final_p3 = p3_list  # List[Tensor [T_i, 64]]
            final_p4 = p4_list  # List[Tensor [T_i, 128]]
            final_p5 = p5_list  # List[Tensor [T_i, 256]]
            is_var_len = True
        else:
            final_p3 = torch.stack(p3_list, dim=0)  # [N, seq_len, 64]
            final_p4 = torch.stack(p4_list, dim=0)  # [N, seq_len, 128]
            final_p5 = torch.stack(p5_list, dim=0)  # [N, seq_len, 256]
            is_var_len = False

        save_dict = {
            "p3": final_p3,
            "p4": final_p4,
            "p5": final_p5,
            "labels": final_labels,
            "video_ids": video_ids_list,
            "clip_ids": clip_ids_list,
            "seq_lens": final_seq_lens,
            "is_variable_len": is_var_len,
            "sample_interval": sample_interval,
            "dtype": "float16" if use_fp16 else "float32",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "augmented": is_augmented_split,
            "num_aug": num_aug if is_augmented_split else 0,
            "include_original": actual_include_original,
            "datasets": datasets_list,
            "subjects": subjects_list
        }

        target_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, str(target_path))

        num_alerts = (final_labels == 0).sum().item()
        num_drowsy = (final_labels == 1).sum().item()
        num_orig = sum(1 for vid in video_ids_list if "_aug" not in vid)
        num_aug_count = sum(1 for vid in video_ids_list if "_aug" in vid)
        file_size_mb = target_path.stat().st_size / (1024 * 1024)

        print(f"[SUCCESS] Đã lưu tệp: {target_path.resolve()} ({file_size_mb:.2f} MB)")
        print(f"          - Tổng số mẫu: {len(video_ids_list)} (Gốc: {num_orig}, Augment: {num_aug_count})")
        print(f"          - Phân bố nhãn: Tỉnh táo (0) = {num_alerts}, Buồn ngủ (1) = {num_drowsy}")
        if is_var_len:
            print(f"          - Cấu trúc Tensor: List[{len(final_p3)}] (min={final_seq_lens.min().item()}, max={final_seq_lens.max().item()}, mean={final_seq_lens.float().mean().item():.1f})")
        else:
            print(f"          - Tensor p3: {list(final_p3.shape)}, p4: {list(final_p4.shape)}, p5: {list(final_p5.shape)}")

        del p3_list, p4_list, p5_list, final_p3, final_p4, final_p5, save_dict
        gc.collect()

        return {
            "file": str(target_path.resolve()),
            "total_samples": len(video_ids_list),
            "size_mb": file_size_mb,
            "num_alerts": num_alerts,
            "num_drowsy": num_drowsy
        }

    if shard_size and shard_size > 0 and len(cache_files) > shard_size:
        total_shards = math.ceil(len(cache_files) / shard_size)
        shards_info = []
        for s_idx in range(total_shards):
            start_i = s_idx * shard_size
            end_i = min(start_i + shard_size, len(cache_files))
            shard_cache_files = cache_files[start_i:end_i]
            shard_path = output_pt_path.parent / f"{output_pt_path.stem}_part{s_idx + 1:03d}.pt"
            info = _package_single_shard(shard_cache_files, shard_path, part_idx=s_idx + 1)
            shards_info.append(info)

        # Lưu master index JSON cho các shard
        master_json_path = output_pt_path.parent / f"{output_pt_path.stem}_shards_index.json"
        with open(master_json_path, "w", encoding="utf-8") as f:
            json.dump({
                "split": split_name,
                "total_shards": total_shards,
                "total_samples": len(cache_files),
                "shard_size": shard_size,
                "shards": shards_info,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }, f, indent=4, ensure_ascii=False)
        print(f"[+] Đã lưu chỉ mục phân mảnh: {master_json_path.resolve()}")
    else:
        _package_single_shard(cache_files, output_pt_path)


# ==============================================================================
# 5. QUY TRÌNH TRÍCH XUẤT ĐẶC TRƯNG TỪNG PHÂN TẬP (ZERO RAM ACCUMULATION)
# ==============================================================================
def process_split_set(
        split_name: str,
        items: List[VideoClipItem],
        runner: CloneONNXCUDARuntime,
        output_pt_path: Path,
        cache_dir: Path,
        seq_len: Optional[int] = None,
        sample_interval: float = 0.1,
        img_size: int = 640,
        chunk_size: int = 64,
        use_fp16: bool = True,
        augmenter: Optional[DetectionAugmenter] = None,
        num_aug: int = 4,
        include_original: bool = True,
        base_seed: int = 42,
        force_recompute: bool = False,
        gc_interval: int = 50,
        shard_size: Optional[int] = 5000,
        skip_package: bool = False
) -> None:
    """
    Trích xuất đặc trưng cho một phân tập (Train hoặc Val) theo cơ chế Zero RAM Accumulation.
    Pha 1: Trích xuất & ghi thẳng ra cache trên đĩa (RAM giữ nguyên ở mức O(1)).
    Pha 2: Đóng gói thành tệp .pt đơn lẻ (hoặc phân mảnh).
    """
    is_augmented_split = (augmenter is not None and num_aug > 0)
    actual_include_original = include_original if is_augmented_split else True

    print(f"\n{'=' * 75}")
    print(f"[*] BẮT ĐẦU PHA 1: TRÍCH XUẤT & CACHE (PHÂN TẬP: {split_name.upper()} - {len(items)} CLIPS)")
    print(f"[*] Cơ chế lấy mẫu      : t = {sample_interval}s cố định (Target FPS = {1.0 / sample_interval:.1f})")
    print(f"[*] Cấu hình seq_len     : {seq_len if seq_len is not None else 'None (Cách 1: List[torch.Tensor] chuỗi động)'}")
    print(f"[*] Tăng cường dữ liệu   : {'BẬT (Số bản=' + str(num_aug) + ')' if is_augmented_split else 'TẮT'}")
    print(f"[*] Cơ chế quản lý RAM   : Zero RAM Accumulation (gc_interval={gc_interval})")
    print(f"[*] Định dạng lưu cache  : {'Float16 (Tiết kiệm 50% RAM/Disk)' if use_fp16 else 'Float32'}")
    print(f"{'=' * 75}")

    split_cache_dir = cache_dir / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    # Khởi động trước (Warm-up) mô hình: warm up cả batch chính và batch dư nếu có (ví dụ 64 và 36 cho 100 frames)
    print(f"[+] Khởi động (Warm-up) ONNX cuDNN Execution Provider (batch={chunk_size})...")
    warmup_device = f"cuda:{runner.device_id}" if (runner.use_cuda and torch.cuda.is_available()) else "cpu"
    dummy_main = torch.zeros(chunk_size, 3, img_size, img_size, device=warmup_device)
    runner.forward(dummy_main)
    del dummy_main
    remainder = (seq_len % chunk_size) if (seq_len is not None and seq_len % chunk_size != 0) else (100 % chunk_size)
    if remainder > 0 and remainder != chunk_size:
        dummy_rem = torch.zeros(remainder, 3, img_size, img_size, device=warmup_device)
        runner.forward(dummy_rem)
        del dummy_rem
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("[+] Khởi động hoàn tất!")

    cache_files_for_packaging: List[Path] = []
    start_time = time.time()

    tqdm_bar = tqdm(items, desc=f"[{split_name.upper()}]")
    for idx, item in enumerate(tqdm_bar):
        video_path = item.video_path
        label = item.label
        video_id = item.video_id
        clip_id = item.clip_id

        if not video_path.exists():
            print(f"\n[WARN] Tệp không tồn tại trên đĩa, bỏ qua: {video_path}")
            continue

        # Xác định danh sách các biến thể cần xử lý cho video này
        targets = []
        if actual_include_original:
            targets.append({
                "sub_id": video_id,
                "is_aug": False,
                "aug_idx": 0,
                "seed": None,
            })

        if is_augmented_split:
            for k in range(1, num_aug + 1):
                aug_id = f"{video_id}_aug{k}"
                aug_seed = (base_seed + int(hashlib.md5(aug_id.encode("utf-8")).hexdigest()[:8], 16)) % (2 ** 31 - 1)
                targets.append({
                    "sub_id": aug_id,
                    "is_aug": True,
                    "aug_idx": k,
                    "seed": aug_seed,
                })

        # Kiểm tra xem những target nào chưa có trong cache hoặc cache không khớp sample_interval
        needed_targets = []
        for t in targets:
            cache_file = split_cache_dir / f"{t['sub_id']}.pt"
            if force_recompute or not cache_file.exists():
                needed_targets.append(t)
            else:
                try:
                    cdata = torch.load(cache_file, map_location="cpu")
                    c_interval = cdata.get("sample_interval", None)
                    if c_interval is not None and abs(c_interval - sample_interval) > 1e-4:
                        needed_targets.append(t)
                except Exception:
                    needed_targets.append(t)

        # Nếu cần trích xuất ít nhất một target, chỉ đọc và giải mã video một lần duy nhất
        if len(needed_targets) > 0:
            try:
                raw_frames, actual_frames = read_and_sample_video_frames(
                    video_path=video_path,
                    seq_len=seq_len,
                    sample_interval=sample_interval,
                    img_size=img_size
                )
            except Exception as e:
                print(f"\n[ERROR] Lỗi khi đọc video {video_path.name}: {e}")
                continue

            for t in needed_targets:
                try:
                    if t["is_aug"]:
                        aug_frames, _, _, _ = augmenter.augment_video(
                            frames=raw_frames,
                            boxes_list=None,
                            labels_list=None,
                            seed=t["seed"]
                        )
                        p3, p4, p5 = forward_video_chunks(
                            frames_rgb=aug_frames,
                            runner=runner,
                            chunk_size=chunk_size,
                            device=f"cuda:{runner.device_id}",
                            use_fp16=use_fp16
                        )
                        del aug_frames
                    else:
                        p3, p4, p5 = forward_video_chunks(
                            frames_rgb=raw_frames,
                            runner=runner,
                            chunk_size=chunk_size,
                            device=f"cuda:{runner.device_id}",
                            use_fp16=use_fp16
                        )

                    cache_file = split_cache_dir / f"{t['sub_id']}.pt"
                    torch.save({
                        "video_id": t["sub_id"],
                        "clip_id": clip_id,
                        "label": label,
                        "p3": p3,
                        "p4": p4,
                        "p5": p5,
                        "actual_frames": actual_frames,
                        "sample_interval": sample_interval,
                        "is_augmented": t["is_aug"],
                        "aug_seed": t["seed"],
                        "dataset": item.dataset,
                        "subject_id": item.subject_id
                    }, cache_file)
                    del p3, p4, p5
                except Exception as e:
                    print(f"\n[ERROR] Lỗi khi xử lý mẫu {t['sub_id']}: {e}")

            del raw_frames

        # Chỉ ghi nhận đường dẫn tệp cache để đóng gói, KHÔNG nạp tensor vào RAM!
        for t in targets:
            cache_file = split_cache_dir / f"{t['sub_id']}.pt"
            if cache_file.exists():
                cache_files_for_packaging.append(cache_file)

        # Định kỳ thu gom rác chủ động & cập nhật trạng thái RAM trên thanh tiến trình
        if (idx + 1) % gc_interval == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if _PSUTIL_AVAILABLE and (idx % 10 == 0 or idx == len(items) - 1):
            ram_gb = get_current_ram_gb()
            tqdm_bar.set_postfix({"RAM": f"{ram_gb:.2f}GB"})

    total_duration = time.time() - start_time
    print(f"\n[+] Hoàn tất Pha 1 (Trích xuất & Cache) cho {len(cache_files_for_packaging)} mẫu tập {split_name.upper()} trong {total_duration / 60:.2f} phút!")

    if len(cache_files_for_packaging) == 0:
        print(f"[WARN] Không có mẫu nào được trích xuất cho phân tập {split_name.upper()}!")
        return

    # Pha 2: Đóng gói
    if skip_package:
        print(f"[*] Bỏ qua pha đóng gói theo tùy chọn --skip_package (--cache_only). Tệp cache sẵn sàng tại: {split_cache_dir.resolve()}")
        return

    consolidate_split_features(
        split_name=split_name,
        cache_files=cache_files_for_packaging,
        output_pt_path=output_pt_path,
        seq_len=seq_len,
        sample_interval=sample_interval,
        use_fp16=use_fp16,
        is_augmented_split=is_augmented_split,
        num_aug=num_aug,
        actual_include_original=actual_include_original,
        shard_size=shard_size
    )


# ==============================================================================
# 5.1. TIẾN TRÌNH CON ĐA GPU (MULTI-GPU WORKER LOOP & DISPATCHER)
# ==============================================================================
def gpu_worker_loop(
        worker_id: int,
        gpu_id: int,
        task_queue,
        result_queue,
        onnx_model_path: str,
        split_cache_dir_str: str,
        seq_len: Optional[int] = None,
        sample_interval: float = 0.1,
        img_size: int = 640,
        chunk_size: int = 64,
        use_fp16: bool = True,
        augment_enabled: bool = False,
        num_aug: int = 4,
        include_original: bool = True,
        base_seed: int = 42,
        aug_config: Optional[dict] = None,
        force_recompute: bool = False,
        gc_interval: int = 50,
        cudnn_algo: str = "EXHAUSTIVE"
):
    """
    Vòng lặp worker chạy độc lập trên 1 GPU CUDA riêng biệt trong hồ đa tiến trình.
    Thực hiện trích xuất theo 5 chiến lược kỹ thuật tăng tốc đột phá:
    1. Sequential Stream Decoding một chiều (loại bỏ cap.set).
    2. Tái sử dụng raw_frames cho toàn bộ num_aug trong RAM (In-Memory Augmentation).
    3. Mở rộng batch chunk_size (mặc định 64/100).
    4. Uint8 H2D PCIe Transfer + chuẩn hóa song song trên GPU CUDA cores.
    5. Cache-First Fault Tolerance & Intermediate Commit.
    """
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(gpu_id)
        device_str = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

        runner = CloneONNXCUDARuntime(onnx_model_path=onnx_model_path, device_id=gpu_id, cudnn_algo=cudnn_algo)
        # Khởi động trước (Warm-up) mô hình: warm up cả chunk_size và batch dư (ví dụ 64 và 36)
        dummy_main = torch.zeros(chunk_size, 3, img_size, img_size, device=device_str)
        runner.forward(dummy_main)
        del dummy_main
        remainder = (seq_len % chunk_size) if (seq_len is not None and seq_len % chunk_size != 0) else (100 % chunk_size)
        if remainder > 0 and remainder != chunk_size:
            dummy_rem = torch.zeros(remainder, 3, img_size, img_size, device=device_str)
            runner.forward(dummy_rem)
            del dummy_rem
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        augmenter = None
        if augment_enabled and num_aug > 0:
            cfg_dict = aug_config if aug_config else dict(DEFAULT_AUG_CONFIG)
            augmenter = DetectionAugmenter(cfg_dict)

        split_cache_dir = Path(split_cache_dir_str)
        split_cache_dir.mkdir(parents=True, exist_ok=True)
        is_augmented = (augmenter is not None and num_aug > 0)
        actual_include_original = include_original if is_augmented else True

        result_queue.put({"type": "ready", "worker_id": worker_id, "gpu_id": gpu_id})
        processed_count = 0

        while True:
            task = task_queue.get()
            if task is None:
                break

            video_path_str, label, clip_id, video_id, dataset, subject_id = task
            video_path = Path(video_path_str)

            targets = []
            if actual_include_original:
                targets.append({"sub_id": video_id, "is_aug": False, "aug_idx": 0, "seed": None})
            if is_augmented:
                for k in range(1, num_aug + 1):
                    aug_id = f"{video_id}_aug{k}"
                    aug_seed = (base_seed + int(hashlib.md5(aug_id.encode("utf-8")).hexdigest()[:8], 16)) % (2 ** 31 - 1)
                    targets.append({"sub_id": aug_id, "is_aug": True, "aug_idx": k, "seed": aug_seed})

            needed_targets = []
            for t in targets:
                cache_file = split_cache_dir / f"{t['sub_id']}.pt"
                if force_recompute or not cache_file.exists():
                    needed_targets.append(t)
                else:
                    try:
                        cdata = torch.load(cache_file, map_location="cpu")
                        c_interval = cdata.get("sample_interval", None)
                        if c_interval is not None and abs(c_interval - sample_interval) > 1e-4:
                            needed_targets.append(t)
                    except Exception:
                        needed_targets.append(t)

            if len(needed_targets) == 0:
                result_queue.put({
                    "type": "done",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "video_id": video_id,
                    "status": "cached",
                    "targets": [t["sub_id"] for t in targets]
                })
                continue

            try:
                # Đọc và giải mã video gốc DUY NHẤT 1 LẦN (In-Memory Augmentation)
                raw_frames, actual_frames = read_and_sample_video_frames(
                    video_path=video_path,
                    seq_len=seq_len,
                    sample_interval=sample_interval,
                    img_size=img_size
                )

                for t in needed_targets:
                    if t["is_aug"]:
                        aug_frames, _, _, _ = augmenter.augment_video(frames=raw_frames, seed=t["seed"])
                        p3, p4, p5 = forward_video_chunks(
                            frames_rgb=aug_frames,
                            runner=runner,
                            chunk_size=chunk_size,
                            device=device_str,
                            use_fp16=use_fp16
                        )
                        del aug_frames
                    else:
                        p3, p4, p5 = forward_video_chunks(
                            frames_rgb=raw_frames,
                            runner=runner,
                            chunk_size=chunk_size,
                            device=device_str,
                            use_fp16=use_fp16
                        )

                    cache_file = split_cache_dir / f"{t['sub_id']}.pt"
                    torch.save({
                        "video_id": t["sub_id"],
                        "clip_id": clip_id,
                        "label": label,
                        "p3": p3,
                        "p4": p4,
                        "p5": p5,
                        "actual_frames": actual_frames,
                        "sample_interval": sample_interval,
                        "is_augmented": t["is_aug"],
                        "aug_seed": t["seed"],
                        "dataset": dataset,
                        "subject_id": subject_id
                    }, cache_file)
                    del p3, p4, p5

                del raw_frames
                processed_count += 1

                if processed_count % gc_interval == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                result_queue.put({
                    "type": "done",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "video_id": video_id,
                    "status": "extracted",
                    "targets": [t["sub_id"] for t in targets]
                })

            except Exception as e:
                result_queue.put({
                    "type": "error",
                    "worker_id": worker_id,
                    "gpu_id": gpu_id,
                    "video_id": video_id,
                    "error": str(e)
                })

    except Exception as fatal_e:
        result_queue.put({
            "type": "fatal_error",
            "worker_id": worker_id,
            "gpu_id": gpu_id,
            "error": str(fatal_e)
        })


def run_multi_gpu_extraction(
        split_name: str,
        items: List[VideoClipItem],
        device_ids: List[int],
        onnx_path: Path,
        output_dir: Path,
        cache_dir: Path,
        seq_len: Optional[int] = None,
        sample_interval: float = 0.1,
        img_size: int = 640,
        chunk_size: int = 64,
        use_fp16: bool = True,
        augment_enabled: bool = False,
        num_aug: int = 4,
        include_original: bool = True,
        base_seed: int = 42,
        aug_config: Optional[dict] = None,
        force_recompute: bool = False,
        gc_interval: int = 50,
        shard_size: Optional[int] = 5000,
        skip_package: bool = False,
        output_pt_path: Optional[Path] = None,
        cudnn_algo: str = "EXHAUSTIVE"
) -> List[Path]:
    """
    Điều phối trích xuất đặc trưng song song trên tất cả các GPU CUDA được chỉ định (ví dụ: Kaggle 2x Tesla T4).
    Sử dụng hàng đợi công việc Task Queue và giám sát tiến độ thời gian thực qua tqdm.
    """
    num_workers = len(device_ids)
    split_cache_dir = cache_dir / split_name
    split_cache_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print(f"[*] BẮT ĐẦU PHA 1: TRÍCH XUẤT ĐA GPU CHO PHÂN TẬP: {split_name.upper()} ({len(items)} CLIPS)")
    print(f"[*] Thiết bị tham gia : {num_workers} GPUs ({['CUDA:' + str(i) for i in device_ids]})")
    print(f"[*] Khoảng lấy mẫu     : t = {sample_interval}s (Target FPS = {1.0 / sample_interval:.1f})")
    print(f"[*] Mini-chunk size   : {chunk_size} frames/batch")
    print(f"[*] Precision         : {'Float16 (Tiết kiệm 50% RAM/Disk)' if use_fp16 else 'Float32'}")
    print(f"[*] Tăng cường dữ liệu: {'BẬT (Số bản=' + str(num_aug) + ')' if augment_enabled else 'TẮT'}")
    print(f"[*] cuDNN Conv Algo   : {cudnn_algo}")
    print(f"[*] Thư mục Cache     : {split_cache_dir.resolve()}")
    print("=" * 80)

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()

    # Nạp toàn bộ task vào queue
    for it in items:
        task_queue.put((
            str(it.video_path),
            it.label,
            it.clip_id,
            it.video_id,
            it.dataset,
            it.subject_id
        ))
    for _ in range(num_workers):
        task_queue.put(None)  # Tín hiệu kết thúc cho từng worker

    workers = []
    for worker_id, gpu_id in enumerate(device_ids):
        p = ctx.Process(
            target=gpu_worker_loop,
            kwargs=dict(
                worker_id=worker_id,
                gpu_id=gpu_id,
                task_queue=task_queue,
                result_queue=result_queue,
                onnx_model_path=str(onnx_path),
                split_cache_dir_str=str(split_cache_dir),
                seq_len=seq_len,
                sample_interval=sample_interval,
                img_size=img_size,
                chunk_size=chunk_size,
                use_fp16=use_fp16,
                augment_enabled=augment_enabled,
                num_aug=num_aug if augment_enabled else 0,
                include_original=include_original,
                base_seed=base_seed,
                aug_config=aug_config,
                force_recompute=force_recompute,
                gc_interval=gc_interval,
                cudnn_algo=cudnn_algo
            )
        )
        p.start()
        workers.append(p)

    # Chờ tất cả worker gửi tín hiệu ready
    ready_count = 0
    while ready_count < num_workers:
        msg = result_queue.get()
        if msg.get("type") == "ready":
            ready_count += 1
            print(f"    -> Worker {msg['worker_id']} sẵn sàng trên GPU CUDA:{msg['gpu_id']}")
        elif msg.get("type") == "fatal_error":
            print(f"[FATAL ERROR] Worker {msg['worker_id']} trên GPU CUDA:{msg['gpu_id']} lỗi: {msg['error']}")
            for p in workers:
                p.terminate()
            raise RuntimeError(f"Worker khởi động thất bại: {msg['error']}")

    print(f"[+] Tất cả {num_workers} GPU Workers đã sẵn sàng! Bắt đầu xử lý song song...")

    pbar = tqdm(total=len(items), desc=f"[{split_name.upper()} - {num_workers} GPUs]")
    cached_count, extracted_count, error_count, completed_videos = 0, 0, 0, 0
    collected_cache_files: List[Path] = []

    while completed_videos < len(items):
        msg = result_queue.get()
        msg_type = msg.get("type")

        if msg_type == "done":
            completed_videos += 1
            status = msg.get("status")
            if status == "cached":
                cached_count += 1
            else:
                extracted_count += 1

            for target_id in msg.get("targets", []):
                cf = split_cache_dir / f"{target_id}.pt"
                if cf.exists():
                    collected_cache_files.append(cf)

            ram_gb = get_current_ram_gb()
            pbar.set_postfix({
                "Mới": extracted_count,
                "Cache": cached_count,
                "Lỗi": error_count,
                "RAM": f"{ram_gb:.2f}GB"
            })
            pbar.update(1)

        elif msg_type == "error":
            completed_videos += 1
            error_count += 1
            print(f"\n[WARN] Lỗi khi xử lý video {msg.get('video_id')} (GPU CUDA:{msg.get('gpu_id')}): {msg.get('error')}")
            pbar.update(1)

        elif msg_type == "fatal_error":
            print(f"\n[FATAL] Worker {msg.get('worker_id')} (GPU CUDA:{msg.get('gpu_id')}) dừng đột ngột: {msg.get('error')}")
            break

    pbar.close()

    for p in workers:
        p.join()

    print(f"[+] Hoàn tất Pha 1 ({split_name.upper()}): {completed_videos} videos | Mới: {extracted_count}, Cache: {cached_count}, Lỗi: {error_count}")

    if len(collected_cache_files) == 0:
        print(f"[WARN] Không có mẫu nào được thu thập cho phân tập {split_name.upper()}!")
        return []

    # Pha 2: Đóng gói
    if skip_package or output_pt_path is None:
        print(f"[*] Bỏ qua pha đóng gói theo tùy chọn --skip_package. Tệp cache tại: {split_cache_dir.resolve()}")
        return collected_cache_files

    consolidate_split_features(
        split_name=split_name,
        cache_files=collected_cache_files,
        output_pt_path=output_pt_path,
        seq_len=seq_len,
        sample_interval=sample_interval,
        use_fp16=use_fp16,
        is_augmented_split=augment_enabled,
        num_aug=num_aug if augment_enabled else 0,
        actual_include_original=include_original,
        shard_size=shard_size
    )
    return collected_cache_files


# ==============================================================================
# 6. ĐIỂM VÀO CHÍNH (MAIN ENTRY POINT)
# ==============================================================================
def parse_seq_len(val: Any) -> Optional[int]:
    """Helper chuyển đổi tham số CLI seq_len sang int hoặc None."""
    if val is None or str(val).strip().lower() in ("none", "null", ""):
        return None
    try:
        parsed = int(val)
        return parsed if parsed > 0 else None
    except ValueError:
        raise argparse.ArgumentTypeError(f"seq_len phải là số nguyên dương hoặc None, nhận được: {val}")


def parse_device_ids(val: Any) -> List[int]:
    """Phân giải danh sách GPU index khả dụng, tự động phát hiện trên hệ thống."""
    if not torch.cuda.is_available():
        return [0]
    total_gpus = torch.cuda.device_count()
    if val is None or str(val).strip().lower() in ("auto", "none", "all", ""):
        return list(range(total_gpus)) if total_gpus > 0 else [0]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    parts = [int(p.strip()) for p in str(val).split(",") if p.strip()]
    return parts if parts else [0]


def main():
    parser = argparse.ArgumentParser(
        description="Trích xuất đặc trưng video ra file .pt theo cấu trúc thư mục struct_dataset.md (Tối ưu hóa Tốc độ & RAM)"
    )
    # Cấu hình đường dẫn dữ liệu
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR,
                        help=f"Thư mục gốc chứa dataset phân tầng (mặc định: {DEFAULT_DATA_DIR})")
    parser.add_argument("--split_file", type=str, default=None,
                        help="Đường dẫn tới file metadata dataset_merged_split.json hoặc .csv (mặc định: tự động tìm trong data_dir)")
    parser.add_argument("--scan_folders", action="store_true", default=False,
                        help="Bắt buộc quét trực tiếp thư mục (bỏ qua tệp JSON metadata nếu có)")
    parser.add_argument("--video_exts", type=str, default=DEFAULT_VIDEO_EXTS,
                        help=f"Danh sách phần mở rộng video hỗ trợ (mặc định: {DEFAULT_VIDEO_EXTS})")

    # Mô hình ONNX & Thư mục xuất kết quả
    parser.add_argument("--onnx_path", type=str,
                        default=r"D:\Project\DATN\driver-guardian\ai\LSTM\backbone_neck.onnx",
                        help="Đường dẫn file mô hình ONNX backbone_neck.onnx")
    parser.add_argument("--output_dir", type=str, default="extracted_features_pt",
                        help="Thư mục xuất kết quả (mặc định: extracted_features_pt)")
    parser.add_argument("--train_name", type=str, default="features_merged_train.pt",
                        help="Tên file tensor train đầu ra (mặc định: features_merged_train.pt)")
    parser.add_argument("--val_name", type=str, default="features_merged_val.pt",
                        help="Tên file tensor val đầu ra (mặc định: features_merged_val.pt)")

    # Tham số lấy mẫu chuỗi thời gian
    parser.add_argument("--seq_len", type=parse_seq_len, default=DEFAULT_SEQ_LEN,
                        help="Số khung hình cố định mỗi clip (mặc định: None - Cách 1: giữ nguyên độ dài tự nhiên)")
    parser.add_argument("--sample_interval", type=float, default=DEFAULT_SAMPLE_INTERVAL,
                        help=f"Khoảng thời gian t (giây) giữa 2 lần lấy mẫu (mặc định: {DEFAULT_SAMPLE_INTERVAL}s)")
    parser.add_argument("--splits", type=str, default="train,val",
                        help="Các phân tập cần xử lý, phân cách bằng dấu phẩy (mặc định: 'train,val')")

    # Tham số phần cứng & kích thước ảnh
    parser.add_argument("--img_size", type=int, default=DEFAULT_IMAGE_SIZE, help="Kích thước resize letterbox (mặc định: 640)")
    parser.add_argument("--chunk_size", type=int, default=64, help="Batch size khi forward ONNX (mặc định: 64)")
    parser.add_argument("--device_ids", type=str, default="auto",
                        help="Danh sách GPU index (ví dụ: '0,1' hoặc 'auto' để tự động phát hiện toàn bộ GPU)")
    parser.add_argument("--device_id", type=int, default=None,
                        help="[Tương thích cũ] CUDA device index đơn lẻ (nếu truyền, ghi đè --device_ids)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Giới hạn số lượng video clip mỗi tập để chạy thử nghiệm (ví dụ: --limit 10)")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True,
                        help="Lưu đặc trưng dạng float16 (giảm 50%% dung lượng Disk/RAM, mặc định: True)")
    parser.add_argument("--cudnn_algo", type=str, default="EXHAUSTIVE",
                        choices=["EXHAUSTIVE", "HEURISTIC", "DEFAULT"],
                        help="Thuật toán cuDNN conv search (mặc định: EXHAUSTIVE, tránh fallback chậm)")

    # Tùy chọn tối ưu hóa RAM & Đóng gói nâng cao
    parser.add_argument("--gc_interval", type=int, default=50,
                        help="Chu kỳ số clip thực hiện gc.collect() và empty_cache() (mặc định: 50)")
    parser.add_argument("--shard_size", type=int, default=5000,
                        help="Số lượng mẫu tối đa trên mỗi tệp .pt phân mảnh (mặc định: 5000)")
    parser.add_argument("--skip_package", "--cache_only", action="store_true", default=False,
                        help="Chỉ trích xuất đặc trưng vào thư mục cache, bỏ qua pha gộp file .pt cuối cùng")

    # Tham số tăng cường dữ liệu (Data Augmentation) từ augment.py
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                        help="Bật/tắt tăng cường dữ liệu từ augment.py (mặc định: True)")
    parser.add_argument("--num_aug", type=int, default=4,
                        help="Số bản sao tăng cường cho mỗi video trong tập áp dụng augment (mặc định: 4)")
    parser.add_argument("--include_original", action=argparse.BooleanOptionalAction, default=True,
                        help="Giữ lại video gốc bên cạnh các bản sao tăng cường (mặc định: True)")
    parser.add_argument("--augment_splits", type=str, default="train",
                        help="Các phân tập áp dụng tăng cường, phân cách bởi dấu phẩy (mặc định: train)")
    parser.add_argument("--aug_seed", type=int, default=42,
                        help="Random seed cơ sở cho việc tăng cường dữ liệu (mặc định: 42)")
    parser.add_argument("--aug_config", type=str, default=None,
                        help="Đường dẫn file JSON cấu hình augment tùy chỉnh (mặc định: dùng config từ augment.py)")
    parser.add_argument("--force_recompute", action="store_true", default=False,
                        help="Bắt buộc trích xuất lại từ đầu, bỏ qua cache hiện có")

    # Giữ tương thích tham số cũ
    parser.add_argument("--split_by_subject", action=argparse.BooleanOptionalAction, default=True,
                        help="[Tương thích cũ] Dữ liệu theo struct_dataset.md đã được chia sẵn theo Subject-Independent Split")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="[Tương thích cũ] Tỷ lệ chia tập đã được cố định theo struct_dataset.md (~80/20)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    onnx_path = Path(args.onnx_path)
    output_dir = Path(args.output_dir)
    cache_dir = output_dir / "cache"

    video_exts = tuple(ext.strip() for ext in args.video_exts.split(",") if ext.strip())
    active_splits = [s.strip().lower() for s in args.splits.split(",") if s.strip()]
    augment_splits = [s.strip().lower() for s in args.augment_splits.split(",") if s.strip()]

    split_file_path = Path(args.split_file) if args.split_file else None
    train_pt_path = output_dir / args.train_name
    val_pt_path = output_dir / args.val_name

    # Xác định danh sách GPU xử lý
    if args.device_id is not None:
        device_ids = [args.device_id]
    else:
        device_ids = parse_device_ids(args.device_ids)

    print("=" * 75)
    print("      HỆ THỐNG TRÍCH XUẤT ĐẶC TRƯNG TĂNG TỐC SANG PYTORCH TENSOR (.PT)    ")
    print("      THEO ĐỊNH DẠNG CẤU TRÚC THƯ MỤC CHUẨN (STRUCT_DATASET.MD)         ")
    print("      KẾT HỢP DUAL-GPU WORKER POOL & SEQUENTIAL STREAM DECODING          ")
    print("=" * 75)
    print(f"[*] Dataset Directory : {data_dir.resolve()}")
    print(f"[*] Metadata Split    : {split_file_path.resolve() if split_file_path else 'Tự động phát hiện (Auto-detect)'}")
    print(f"[*] Video Exts        : {video_exts}")
    print(f"[*] ONNX Model Path   : {onnx_path.resolve()}")
    print(f"[*] Output Directory  : {output_dir.resolve()}")
    print(f"[*] Active Splits     : {active_splits}")
    print(f"[*] Sample Interval   : {args.sample_interval}s (Target FPS = {1.0 / args.sample_interval:.1f})")
    print(f"[*] Sequence Length   : {args.seq_len if args.seq_len is not None else 'None (Cách 1: List[torch.Tensor] tự nhiên)'}")
    print(f"[*] Chunk Size        : {args.chunk_size}")
    print(f"[*] Precision         : {'Float16 (Tiết kiệm 50% RAM/Disk)' if args.fp16 else 'Float32'}")
    print(f"[*] Thiết bị GPU xử lý: {device_ids} ({len(device_ids)} GPU song song)")
    print(f"[*] cuDNN Conv Algo   : {args.cudnn_algo}")
    print(f"[*] GC Interval       : Mỗi {args.gc_interval} clips")
    print(f"[*] Shard Size        : {args.shard_size if args.shard_size else 'Không phân mảnh (Gộp 1 file duy nhất)'}")
    print(f"[*] Skip Package      : {'BẬT (Chỉ lưu cache)' if args.skip_package else 'TẮT (Đóng gói tệp .pt)'}")
    print(f"[*] Augmentation      : {'BẬT' if args.augment else 'TẮT'}")
    if args.augment:
        print(f"    - Augment Splits  : {augment_splits}")
        print(f"    - Num Aug Copies  : {args.num_aug}")
        print(f"    - Include Original: {args.include_original}")
        print(f"    - Base Seed       : {args.aug_seed}")
    if args.force_recompute:
        print(f"[*] Force Recompute   : TRUE (Bỏ qua cache hiện có)")
    if args.limit:
        print(f"[*] TEST MODE         : Giới hạn xử lý tối đa {args.limit} clip mỗi tập!")
    print("=" * 75)

    # 1. Nạp danh sách clips theo cấu trúc struct_dataset.md
    train_items, val_items, dataset_meta = prepare_structured_dataset(
        data_dir=data_dir,
        split_file=split_file_path,
        force_folder_scan=args.scan_folders,
        video_exts=video_exts
    )

    if args.limit:
        train_items = train_items[:args.limit]
        val_items = val_items[:args.limit]
        print(f"[*] Áp dụng giới hạn: Train={len(train_items)} clips, Val={len(val_items)} clips.")

    # Lưu thông tin phân bổ để tra cứu provenance
    split_info_save_path = output_dir / "dataset_split_info.json"
    split_info_save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(split_info_save_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset_dir": str(data_dir.resolve()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_clips": len(train_items),
            "val_clips": len(val_items),
            "sample_interval": args.sample_interval,
            "train_samples_preview": [item.clip_id for item in train_items[:10]],
            "val_samples_preview": [item.clip_id for item in val_items[:10]],
        }, f, indent=4, ensure_ascii=False)
    print(f"[+] Đã lưu thông tin kiểm định phân bổ tại: {split_info_save_path.resolve()}")

    # 2. Kiểm tra input shape của mô hình ONNX an toàn qua CPU (không chạm context CUDA của parent)
    actual_img_size = args.img_size
    try:
        cpu_sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        in_meta = cpu_sess.get_inputs()[0]
        if len(in_meta.shape) > 2 and isinstance(in_meta.shape[2], int):
            if actual_img_size != in_meta.shape[2]:
                print(f"[!] Tự động điều chỉnh img_size từ {actual_img_size} -> {in_meta.shape[2]} để khớp với mô hình ONNX.")
                actual_img_size = in_meta.shape[2]
        del cpu_sess
    except Exception:
        pass

    # 3. Chuẩn bị cấu hình augmentation nếu bật
    aug_cfg = dict(DEFAULT_AUG_CONFIG)
    if args.augment and args.num_aug > 0 and len(augment_splits) > 0:
        if args.aug_config:
            aug_cfg_path = Path(args.aug_config)
            if aug_cfg_path.exists():
                with open(aug_cfg_path, "r", encoding="utf-8") as f:
                    custom_cfg = json.load(f)
                    aug_cfg.update(custom_cfg)
                print(f"[+] Đã nạp cấu hình augment tùy chỉnh từ: {aug_cfg_path}")
            else:
                print(f"[WARN] Không tìm thấy file {aug_cfg_path}, dùng cấu hình mặc định từ augment.py")

    # 4. Trích xuất đặc trưng cho Train và Validation
    # Sử dụng Multi-GPU parallel worker pool nếu có từ 2 GPU trở lên, hoặc chạy trực tiếp nếu đơn GPU
    use_multi_gpu = len(device_ids) > 1

    if "train" in active_splits and len(train_items) > 0:
        should_augment_train = args.augment and ("train" in augment_splits)
        if use_multi_gpu:
            run_multi_gpu_extraction(
                split_name="train",
                items=train_items,
                device_ids=device_ids,
                onnx_path=onnx_path,
                output_dir=output_dir,
                cache_dir=cache_dir,
                seq_len=args.seq_len,
                sample_interval=args.sample_interval,
                img_size=actual_img_size,
                chunk_size=args.chunk_size,
                use_fp16=args.fp16,
                augment_enabled=should_augment_train,
                num_aug=args.num_aug if should_augment_train else 0,
                include_original=args.include_original,
                base_seed=args.aug_seed,
                aug_config=aug_cfg if should_augment_train else None,
                force_recompute=args.force_recompute,
                gc_interval=args.gc_interval,
                shard_size=args.shard_size,
                skip_package=args.skip_package,
                output_pt_path=train_pt_path,
                cudnn_algo=args.cudnn_algo
            )
        else:
            runner = CloneONNXCUDARuntime(str(onnx_path), device_id=device_ids[0], cudnn_algo=args.cudnn_algo)
            augmenter = DetectionAugmenter(aug_cfg) if should_augment_train else None
            process_split_set(
                split_name="train",
                items=train_items,
                runner=runner,
                output_pt_path=train_pt_path,
                cache_dir=cache_dir,
                seq_len=args.seq_len,
                sample_interval=args.sample_interval,
                img_size=actual_img_size,
                chunk_size=args.chunk_size,
                use_fp16=args.fp16,
                augmenter=augmenter,
                num_aug=args.num_aug if should_augment_train else 0,
                include_original=args.include_original,
                base_seed=args.aug_seed,
                force_recompute=args.force_recompute,
                gc_interval=args.gc_interval,
                shard_size=args.shard_size,
                skip_package=args.skip_package
            )
            del runner

    if "val" in active_splits and len(val_items) > 0:
        should_augment_val = args.augment and ("val" in augment_splits)
        if use_multi_gpu:
            run_multi_gpu_extraction(
                split_name="val",
                items=val_items,
                device_ids=device_ids,
                onnx_path=onnx_path,
                output_dir=output_dir,
                cache_dir=cache_dir,
                seq_len=args.seq_len,
                sample_interval=args.sample_interval,
                img_size=actual_img_size,
                chunk_size=args.chunk_size,
                use_fp16=args.fp16,
                augment_enabled=should_augment_val,
                num_aug=args.num_aug if should_augment_val else 0,
                include_original=args.include_original,
                base_seed=args.aug_seed,
                aug_config=aug_cfg if should_augment_val else None,
                force_recompute=args.force_recompute,
                gc_interval=args.gc_interval,
                shard_size=args.shard_size,
                skip_package=args.skip_package,
                output_pt_path=val_pt_path,
                cudnn_algo=args.cudnn_algo
            )
        else:
            runner = CloneONNXCUDARuntime(str(onnx_path), device_id=device_ids[0], cudnn_algo=args.cudnn_algo)
            augmenter = DetectionAugmenter(aug_cfg) if should_augment_val else None
            process_split_set(
                split_name="val",
                items=val_items,
                runner=runner,
                output_pt_path=val_pt_path,
                cache_dir=cache_dir,
                seq_len=args.seq_len,
                sample_interval=args.sample_interval,
                img_size=actual_img_size,
                chunk_size=args.chunk_size,
                use_fp16=args.fp16,
                augmenter=augmenter,
                num_aug=args.num_aug if should_augment_val else 0,
                include_original=args.include_original,
                base_seed=args.aug_seed,
                force_recompute=args.force_recompute,
                gc_interval=args.gc_interval,
                shard_size=args.shard_size,
                skip_package=args.skip_package
            )
            del runner

    print("\n" + "=" * 75)
    print("      [HOÀN TẤT GIAI ĐOẠN 1] TRÍCH XUẤT ĐẶC TRƯNG THÀNH CÔNG!     ")
    print("=" * 75)
    print(f"1. Cấu hình phân bổ : {split_info_save_path.resolve()}")
    if not args.skip_package:
        if "train" in active_splits:
            if train_pt_path.exists():
                print(f"2. File Train Tensor: {train_pt_path.resolve()}")
            else:
                shards = sorted(output_dir.glob(f"{train_pt_path.stem}_part*.pt"))
                if shards:
                    print(f"2. Shards Train Tensor: {len(shards)} shards tại {output_dir.resolve()} (chỉ mục: {train_pt_path.stem}_shards_index.json)")
        if "val" in active_splits:
            if val_pt_path.exists():
                print(f"3. File Val Tensor  : {val_pt_path.resolve()}")
            else:
                shards = sorted(output_dir.glob(f"{val_pt_path.stem}_part*.pt"))
                if shards:
                    print(f"3. Shards Val Tensor  : {len(shards)} shards tại {output_dir.resolve()}")
    else:
        print(f"2. Toàn bộ tệp cache được lưu trữ tại: {cache_dir.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    mp.freeze_support()
    main()
