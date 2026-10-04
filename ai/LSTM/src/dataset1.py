#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: src/dataset1.py
Mục đích:
    Xây dựng PyTorch Dataset (RawVideoONNXDataset), hàm gom batch (collate_raw_video_features)
    và hàm factory (build_raw_video_dataloaders) để nạp trực tiếp video thô (.mp4, .avi, .mkv)
    và tích hợp động cơ suy luận ONNX Runtime trích xuất trực tiếp các tensor đặc trưng không gian
    đa tỉ lệ nguyên bản (p3, p4, p5) theo thời gian lấy mẫu tùy chỉnh (sample_interval).

Đặc tính kỹ thuật cốt lõi:
    1. Trích xuất Động Trực tiếp (Streaming On-The-Fly Extraction):
       Giải mã video thô qua OpenCV và trích xuất trực tiếp bản đồ đặc trưng qua mô hình
       ONNX Backbone+PAFPN Neck (checkpoints/backbone_neck.onnx).
       Bảo toàn nguyên vẹn 3 tensor đặc trưng 4D:
         - p3: [T, 64, 80, 80]
         - p4: [T, 128, 40, 40]
         - p5: [T, 256, 20, 20]
    2. Thời gian Lấy mẫu Tùy chỉnh (Customizable Sampling Interval):
       Tham số sample_interval (đơn vị giây, ví dụ 0.1s ~ 10 FPS, 0.05s ~ 20 FPS, 0.2s ~ 5 FPS)
       tự động thích nghi với FPS gốc của từng video: step = max(1, round(fps * sample_interval)).
       Căn chỉnh kích thước qua letterbox giữ nguyên tỷ lệ khung hình chuẩn 640x640.
    3. An toàn Tuyệt đối khi Chạy Đa Tiến trình (Multiprocessing Safe):
       Khởi tạo trễ (Lazy Session Initialization) đối tượng onnxruntime.InferenceSession
       bên trong từng tiến trình worker khi gọi __getitem__, triệt tiêu hoàn toàn lỗi
       TypeError: cannot pickle 'InferenceSession' trên cả Windows và Linux.
    4. Suy luận Mini-Chunk Chống tràn VRAM:
       Xử lý chuỗi khung hình theo từng mini-chunk (chunk_size=16) trên GPU CUDA / CPU fallback.
    5. Thiết kế Tinh gọn & Tiết kiệm Bộ nhớ (Stateless Data Pipeline):
       - Không tích hợp src/img_preprocess.py để tối ưu hóa thông lượng nạp dữ liệu.
       - Không sử dụng cơ chế Caching (RAM/Disk cache), bảo đảm tiết kiệm 100% RAM và không sinh file rác.
    6. Gom Batch Động (Dynamic Zero-Padding Collate):
       Tự động pad 0 theo trục thời gian về độ dài lớn nhất (T_max) trong từng batch và sinh
       tensor seq_lens [B], tích hợp hoàn hảo với cơ chế Attention Masking của TemporalAttentionPooling
       trong DeepGRUClassifier (src/models.py).
"""

import os
import sys
import math
import random
import shutil
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thư mục gốc dự án
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Thiết lập môi trường tối ưu cho OpenMP và ONNX Runtime
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["ORT_LOG_LEVEL"] = "3"

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Tải trước thư viện CUDA DLL từ PyTorch nếu chạy trên hệ điều hành Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

try:
    import onnxruntime as ort
except ImportError as exc:
    raise ImportError("Vui lòng cài đặt onnxruntime hoặc onnxruntime-gpu: pip install onnxruntime-gpu") from exc


# ==============================================================================
# 1. CẤU TRÚC DỮ LIỆU ĐẠI DIỆN MẪU VIDEO (RAW VIDEO SAMPLE ENTRY)
# ==============================================================================
@dataclass
class RawVideoSample:
    """Cấu trúc đại diện cho một mẫu video thô đã quét/lập chỉ mục."""
    path: Path              # Đường dẫn tệp video thô
    video_id: str           # Khóa định danh duy nhất (vd: "train_0_alert_sust_n_1")
    split: str              # "train", "val", hoặc "all"
    label: int              # 0 (alert) hoặc 1 (drowsy)
    label_name: str         # "0_alert" hoặc "1_drowsy"
    source_dataset: str = "custom"  # Nguồn dataset (vd: "sust", "uta-rldd", "vbddd")
    subject_id: Optional[str] = None # Mã định danh đối tượng người tham gia (nếu có)


# ==============================================================================
# 2. HÀM BIẾN ĐỔI ẢNH LETTERBOX (LETTERBOX RESIZING UTILS)
# ==============================================================================
def letterbox(
    image: np.ndarray,
    new_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỷ lệ (aspect ratio) với phần đệm padding đồng màu chuẩn 640x640.

    Args:
        image: Ảnh numpy mảng [H, W, 3] (BGR hoặc RGB).
        new_size: Kích thước cạnh vuông mục tiêu (mặc định 640).
        color: Màu đệm padding theo chuẩn YOLO/PAFPN (114, 114, 114).

    Returns:
        canvas: Ảnh sau khi letterbox [new_size, new_size, 3].
    """
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas


# ==============================================================================
# 3. ĐỘNG CƠ SUY LUẬN ONNX THEO MINI-CHUNK (STREAMING FEATURE EXTRACTOR)
# ==============================================================================
class ONNXRawFeatureExtractor:
    """
    Quản lý suy luận mô hình ONNX Backbone+PAFPN Neck theo từng mini-chunk.
    Hỗ trợ khởi tạo trễ (Lazy Initialization) để đảm bảo an toàn tuyệt đối khi serialize
    qua các worker processes của PyTorch DataLoader.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: str = "auto",
        device_id: int = 0,
        chunk_size: int = 16,
        use_fp16: bool = False
    ) -> None:
        self.model_path = Path(model_path)
        self.device = device.lower().strip()
        self.device_id = device_id
        self.chunk_size = max(1, chunk_size)
        self.use_fp16 = use_fp16

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: Optional[str] = None
        self.output_names: Optional[List[str]] = None
        self.idx_p3: int = 0
        self.idx_p4: int = 1
        self.idx_p5: int = 2
        self.active_provider: Optional[str] = None

    def _init_session(self) -> None:
        """Khởi tạo phiên làm việc ONNX Runtime an toàn bên trong tiến trình hiện tại."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp mô hình ONNX: {self.model_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3

        available_providers = ort.get_available_providers()
        providers = []

        # Tự động phát hiện hoặc cấu hình thiết bị thực thi
        use_cuda = False
        if self.device in ("cuda", "gpu", "auto") and "CUDAExecutionProvider" in available_providers:
            # Kiểm tra thêm tính khả dụng của torch CUDA nếu có
            if not torch.cuda.is_available() and self.device == "auto":
                use_cuda = False
            else:
                use_cuda = True

        if use_cuda:
            cuda_options = {
                "device_id": self.device_id,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "cudnn_conv_algo_search": "HEURISTIC",
                "do_copy_in_default_stream": True,
            }
            providers.append(("CUDAExecutionProvider", cuda_options))

        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_options, providers=providers)
        self.active_provider = self.session.get_providers()[0]

        # Đọc thông tin các cổng vào/ra
        inputs = self.session.get_inputs()
        self.input_name = inputs[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

        # Xác định chỉ mục vị trí cho p3, p4, p5
        self.idx_p3 = self.output_names.index("p3") if "p3" in self.output_names else 0
        self.idx_p4 = self.output_names.index("p4") if "p4" in self.output_names else 1
        self.idx_p5 = self.output_names.index("p5") if "p5" in self.output_names else 2

    def get_session(self) -> ort.InferenceSession:
        """Lấy phiên InferenceSession hiện tại (khởi tạo trễ nếu chưa có)."""
        if self.session is None:
            self._init_session()
        return self.session

    def extract_chunks(
        self,
        frames_rgb: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Trích xuất 3 bản đồ đặc trưng (p3, p4, p5) theo từng mini-chunk chống tràn VRAM.

        Args:
            frames_rgb: Danh sách các ảnh numpy RGB uint8 [640, 640, 3].

        Returns:
            p3: np.ndarray [T, 64, 80, 80]
            p4: np.ndarray [T, 128, 40, 40]
            p5: np.ndarray [T, 256, 20, 20]
        """
        session = self.get_session()
        T = len(frames_rgb)
        out_dtype = np.float16 if self.use_fp16 else np.float32

        if T == 0:
            return (
                np.empty((0, 64, 80, 80), dtype=out_dtype),
                np.empty((0, 128, 40, 40), dtype=out_dtype),
                np.empty((0, 256, 20, 20), dtype=out_dtype)
            )

        p3_chunks: List[np.ndarray] = []
        p4_chunks: List[np.ndarray] = []
        p5_chunks: List[np.ndarray] = []

        for i in range(0, T, self.chunk_size):
            batch_slice = frames_rgb[i : i + self.chunk_size]
            # [B, H, W, 3] -> [B, 3, H, W] chuẩn hóa [0.0, 1.0]
            batch_np = np.stack(batch_slice, axis=0).transpose(0, 3, 1, 2)
            batch_np = np.ascontiguousarray(batch_np, dtype=np.float32) / 255.0

            outputs = session.run(self.output_names, {self.input_name: batch_np})
            p3_c = outputs[self.idx_p3]
            p4_c = outputs[self.idx_p4]
            p5_c = outputs[self.idx_p5]

            if self.use_fp16:
                p3_c = p3_c.astype(np.float16)
                p4_c = p4_c.astype(np.float16)
                p5_c = p5_c.astype(np.float16)

            p3_chunks.append(p3_c)
            p4_chunks.append(p4_c)
            p5_chunks.append(p5_c)

            del batch_np, outputs

        p3_full = np.concatenate(p3_chunks, axis=0)
        p4_full = np.concatenate(p4_chunks, axis=0)
        p5_full = np.concatenate(p5_chunks, axis=0)
        return p3_full, p4_full, p5_full

    def close(self) -> None:
        """Giải phóng phiên suy luận ONNX."""
        self.session = None


# ==============================================================================
# 4. LỚP PYTORCH DATASET: RawVideoONNXDataset
# ==============================================================================
class RawVideoONNXDataset(Dataset):
    """
    PyTorch Dataset nạp video thô (.mp4, .avi, .mkv) và trích xuất trực tiếp
    đặc trưng không gian đa tỷ lệ (p3, p4, p5) qua mô hình ONNX Runtime.

    Args:
        dataset_dir: Thư mục chứa video hoặc đường dẫn manifest CSV/JSON.
        manifest_file: Đường dẫn tệp CSV/JSON manifest tùy chọn.
        split: Phân vùng dữ liệu ("train", "val", hoặc "all").
        sample_interval: Chu kỳ lấy mẫu khung hình thời gian (giây), mặc định 0.1s (~10 FPS).
        seq_len: Độ dài khung hình cố định (None = giữ trọn vẹn số khung hình thực tế).
        onnx_model_path: Đường dẫn tệp mô hình ONNX (mặc định "checkpoints/backbone_neck.onnx").
        img_size: Kích thước cạnh vuông ảnh sau letterbox (mặc định 640).
        chunk_size: Kích thước mini-chunk khi suy luận ONNX tránh tràn VRAM (mặc định 16).
        device: Thiết bị thực thi ONNX ("cuda", "cpu", hoặc "auto").
        device_id: Mã GPU ID nếu sử dụng CUDA (mặc định 0).
        use_fp16: Trả về tensor float16 để tiết kiệm bộ nhớ.
        augmenter: Đối tượng tăng cường khung hình từ src.augment (tùy chọn, áp dụng cho train).
        video_exts: Bộ đuôi tệp video hợp lệ (mặc định ('.mp4', '.avi', '.mkv', '.mov')).
        window_sampling: Chiến lược cắt lát thời gian ('random' cho train, 'center' cho val).
        filter_label: Chỉ lấy các mẫu có nhãn cụ thể (0 hoặc 1). None = lấy tất cả.
        min_frames: Số lượng khung hình tối thiểu hợp lệ của một video (mặc định 1).
        target_dtype: Kiểu dữ liệu PyTorch của tensor đầu ra (mặc định torch.float32).
    """

    def __init__(
        self,
        dataset_dir: Union[str, Path],
        manifest_file: Optional[Union[str, Path]] = None,
        split: str = "train",
        sample_interval: float = 0.1,
        seq_len: Optional[int] = None,
        onnx_model_path: Union[str, Path] = "checkpoints/backbone_neck.onnx",
        img_size: int = 640,
        chunk_size: int = 16,
        device: str = "auto",
        device_id: int = 0,
        use_fp16: bool = False,
        augmenter: Optional[Any] = None,
        video_exts: Sequence[str] = (".mp4", ".avi", ".mkv", ".mov"),
        window_sampling: str = "random",
        filter_label: Optional[int] = None,
        min_frames: int = 1,
        target_dtype: torch.dtype = torch.float32
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.manifest_file = Path(manifest_file) if manifest_file else None
        self.split = split.lower().strip()
        self.sample_interval = float(sample_interval)
        if self.sample_interval <= 0:
            raise ValueError(f"sample_interval phải là số dương, nhận được: {sample_interval}")

        self.seq_len = int(seq_len) if seq_len is not None else None
        self.onnx_model_path = Path(onnx_model_path)
        self.img_size = int(img_size)
        self.chunk_size = int(chunk_size)
        self.device = device
        self.device_id = int(device_id)
        self.use_fp16 = bool(use_fp16)
        self.augmenter = augmenter
        self.video_exts = tuple(ext.lower() for ext in video_exts)
        self.window_sampling = window_sampling.lower().strip()
        self.filter_label = int(filter_label) if filter_label is not None else None
        self.min_frames = max(1, int(min_frames))
        self.target_dtype = target_dtype

        # Khởi tạo trễ: đối tượng trích xuất ONNX sẽ được cấp phát trong _get_extractor()
        self._extractor: Optional[ONNXRawFeatureExtractor] = None

        # Quét và lập danh mục các mẫu video
        self.samples: List[RawVideoSample] = self._discover_samples()

    def _discover_samples(self) -> List[RawVideoSample]:
        """Tự động phát hiện và lập danh mục các mẫu video từ thư mục hoặc manifest."""
        samples: List[RawVideoSample] = []

        # 1. Ưu tiên đọc qua tệp manifest CSV/JSON nếu có
        manifest_path = self.manifest_file
        if manifest_path is None:
            # Tự động tìm manifest trong thư mục dataset_dir
            for candidate in ("dataset_manifest.csv", "dataset_merged_split.csv", "dataset_merged_split.json"):
                cand_path = self.dataset_dir / candidate
                if cand_path.exists():
                    manifest_path = cand_path
                    break

        if manifest_path is not None and manifest_path.exists():
            samples = self._load_from_manifest(manifest_path)
            if samples:
                return samples

        # 2. Quét thư mục phân tầng (Hierarchical Directory Scan)
        if not self.dataset_dir.exists():
            raise FileNotFoundError(f"Thư mục dữ liệu không tồn tại: {self.dataset_dir}")

        # Kiểm tra xem có cấu trúc split/label không (vd: dataset_dir/train/0_alert)
        splits_to_scan = [self.split] if self.split != "all" else ["train", "val"]

        for s in splits_to_scan:
            split_dir = self.dataset_dir / s
            target_base = split_dir if split_dir.exists() else self.dataset_dir

            for label_int, label_str in [(0, "0_alert"), (1, "1_drowsy")]:
                if self.filter_label is not None and self.filter_label != label_int:
                    continue

                label_dir = target_base / label_str
                if not label_dir.exists():
                    # Thử tìm nhãn theo tên rút gọn: "alert", "drowsy"
                    short_str = "alert" if label_int == 0 else "drowsy"
                    label_dir = target_base / short_str

                if label_dir.exists():
                    for v_path in sorted(label_dir.rglob("*")):
                        if v_path.is_file() and v_path.suffix.lower() in self.video_exts:
                            v_id = f"{s}_{label_str}_{v_path.stem}"
                            src = "sust" if "sust" in v_path.stem.lower() else ("uta-rldd" if "uta" in v_path.stem.lower() else "custom")
                            samples.append(
                                RawVideoSample(
                                    path=v_path,
                                    video_id=v_id,
                                    split=s,
                                    label=label_int,
                                    label_name=label_str,
                                    source_dataset=src
                                )
                            )

        # 3. Fallback: Nếu không tìm thấy theo cấu trúc phân tầng, quét toàn bộ thư mục phẳng
        if not samples:
            for v_path in sorted(self.dataset_dir.rglob("*")):
                if v_path.is_file() and v_path.suffix.lower() in self.video_exts:
                    fname = v_path.stem.lower()
                    # Suy luận nhãn từ tên file
                    if any(w in fname for w in ("drowsy", "drowzy", "sleep", "_d_")):
                        lbl = 1
                        lbl_name = "1_drowsy"
                    else:
                        lbl = 0
                        lbl_name = "0_alert"

                    if self.filter_label is not None and self.filter_label != lbl:
                        continue

                    # Suy luận split từ đường dẫn hoặc gán mặc định
                    path_str = str(v_path).lower()
                    if "val" in path_str or "test" in path_str:
                        cur_split = "val"
                    else:
                        cur_split = "train"

                    if self.split != "all" and cur_split != self.split:
                        continue

                    v_id = f"{cur_split}_{lbl_name}_{v_path.stem}"
                    samples.append(
                        RawVideoSample(
                            path=v_path,
                            video_id=v_id,
                            split=cur_split,
                            label=lbl,
                            label_name=lbl_name,
                            source_dataset="custom"
                        )
                    )

        return samples

    def _load_from_manifest(self, manifest_path: Path) -> List[RawVideoSample]:
        """Nạp danh mục mẫu từ tệp CSV/JSON."""
        samples: List[RawVideoSample] = []
        if manifest_path.suffix.lower() == ".csv":
            import csv
            with open(manifest_path, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_split = row.get("split", "train").strip().lower()
                    if self.split != "all" and row_split != self.split:
                        continue

                    label_val = int(row.get("label", 0))
                    if self.filter_label is not None and self.filter_label != label_val:
                        continue

                    # Tìm đường dẫn file video
                    p_str = row.get("orig_file") or row.get("path") or row.get("file_path") or ""
                    v_path = Path(p_str)
                    if not v_path.is_absolute():
                        v_path = self.dataset_dir / v_path

                    if not v_path.exists():
                        continue

                    v_id = row.get("video_id") or v_path.stem
                    lbl_name = row.get("label_name") or ("0_alert" if label_val == 0 else "1_drowsy")
                    src = row.get("source_dataset", "custom")
                    sub_id = row.get("subject_id")

                    samples.append(
                        RawVideoSample(
                            path=v_path,
                            video_id=v_id,
                            split=row_split,
                            label=label_val,
                            label_name=lbl_name,
                            source_dataset=src,
                            subject_id=sub_id
                        )
                    )
        return samples

    def _get_extractor(self) -> ONNXRawFeatureExtractor:
        """Cơ chế Lazy Loading khởi tạo ONNX Feature Extractor trong tiến trình hiện tại."""
        if self._extractor is None:
            self._extractor = ONNXRawFeatureExtractor(
                model_path=self.onnx_model_path,
                device=self.device,
                device_id=self.device_id,
                chunk_size=self.chunk_size,
                use_fp16=self.use_fp16
            )
        return self._extractor

    def _sample_video_frames(self, video_path: Path) -> Tuple[List[np.ndarray], float, float]:
        """
        Đọc và lấy mẫu các khung hình trong video qua OpenCV theo chu kỳ sample_interval tùy chỉnh.

        Returns:
            frames_rgb: Danh sách các ảnh numpy RGB uint8 [img_size, img_size, 3].
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

        # Tính bước nhảy khung hình dựa trên chu kỳ lấy mẫu thời gian tùy chỉnh
        step = max(1, int(round(fps * self.sample_interval)))
        frames_rgb: List[np.ndarray] = []
        frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret or frame_bgr is None:
                break

            if frame_idx % step == 0:
                frame_lb = letterbox(frame_bgr, new_size=self.img_size)
                frame_rgb = cv2.cvtColor(frame_lb, cv2.COLOR_BGR2RGB)
                frames_rgb.append(frame_rgb)

            frame_idx += 1

        cap.release()
        return frames_rgb, float(fps), float(duration_s)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self,
        idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Truy xuất mẫu video thứ `idx`, lấy mẫu khung hình, suy luận ONNX và trả về tensor đặc trưng.

        Returns:
            features: Tuple (p3, p4, p5)
                - p3: torch.Tensor [T, 64, 80, 80]
                - p4: torch.Tensor [T, 128, 40, 40]
                - p5: torch.Tensor [T, 256, 20, 20]
            label: torch.LongTensor scalar (0 hoặc 1)
            seq_len: torch.LongTensor scalar (số lượng khung hình thực tế T)
            meta: Dict chứa thông tin siêu dữ liệu của mẫu
        """
        sample = self.samples[idx]

        # 1. Đọc và lấy mẫu khung hình theo chu kỳ thời gian tùy chỉnh
        frames_rgb, fps, duration_s = self._sample_video_frames(sample.path)
        orig_sampled_len = len(frames_rgb)

        # Xử lý trường hợp video quá ngắn (ít hơn min_frames)
        if orig_sampled_len < self.min_frames:
            if orig_sampled_len == 0:
                # Tạo khung hình rác nếu video rỗng
                dummy_frame = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
                frames_rgb = [dummy_frame] * self.min_frames
            else:
                # Lặp lại frame cuối cho đủ min_frames
                frames_rgb = frames_rgb + [frames_rgb[-1]] * (self.min_frames - orig_sampled_len)

        # 2. Cắt lát cửa sổ thời gian (Temporal Windowing) nếu seq_len được cấu hình
        if self.seq_len is not None and len(frames_rgb) > self.seq_len:
            if self.split == "train" and self.window_sampling == "random":
                start_idx = random.randint(0, len(frames_rgb) - self.seq_len)
            else:
                start_idx = (len(frames_rgb) - self.seq_len) // 2
            frames_rgb = frames_rgb[start_idx : start_idx + self.seq_len]

        # 3. Tăng cường dữ liệu thời gian (Temporal Augmentation) nếu có
        if self.augmenter is not None and self.split == "train":
            # Sử dụng chung 1 seed cho toàn bộ khung hình trong cùng 1 video để đảm bảo tính liên tục
            aug_seed = random.randint(0, 1000000)
            if hasattr(self.augmenter, "apply_sequence"):
                frames_rgb = self.augmenter.apply_sequence(frames_rgb, seed=aug_seed)
            else:
                frames_rgb = [self.augmenter(img, seed=aug_seed) if callable(self.augmenter) else img for img in frames_rgb]

        # 4. Trích xuất trực tiếp đặc trưng qua mô hình ONNX Runtime
        extractor = self._get_extractor()
        p3_np, p4_np, p5_np = extractor.extract_chunks(frames_rgb)

        # 5. Chuyển đổi sang PyTorch Tensors
        p3_t = torch.from_numpy(p3_np).to(dtype=self.target_dtype)
        p4_t = torch.from_numpy(p4_np).to(dtype=self.target_dtype)
        p5_t = torch.from_numpy(p5_np).to(dtype=self.target_dtype)

        label_t = torch.tensor(sample.label, dtype=torch.long)
        seq_len_t = torch.tensor(len(frames_rgb), dtype=torch.long)

        meta = {
            "video_id": sample.video_id,
            "video_path": str(sample.path),
            "split": sample.split,
            "label": sample.label,
            "label_name": sample.label_name,
            "source_dataset": sample.source_dataset,
            "subject_id": sample.subject_id,
            "fps": fps,
            "sample_interval": self.sample_interval,
            "duration_s": duration_s,
            "num_sampled_frames": len(frames_rgb),
            "orig_sampled_frames": orig_sampled_len
        }

        return (p3_t, p4_t, p5_t), label_t, seq_len_t, meta

    def close(self) -> None:
        """Giải phóng tài nguyên ONNX Session."""
        if self._extractor is not None:
            self._extractor.close()
            self._extractor = None


# ==============================================================================
# 5. HÀM GOM BATCH ĐỘNG & FACTORY DATALOADER (COLLATE & DATALOADER FACTORY)
# ==============================================================================
def collate_raw_video_features(
    batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Gom batch động với cơ chế Zero-Padding dọc theo trục thời gian về độ dài lớn nhất T_max.

    Args:
        batch: Danh sách các mẫu trả về từ RawVideoONNXDataset.__getitem__.

    Returns:
        features: Tuple (batch_p3, batch_p4, batch_p5)
            - batch_p3: [B, T_max, 64, 80, 80]
            - batch_p4: [B, T_max, 128, 40, 40]
            - batch_p5: [B, T_max, 256, 20, 20]
        labels: torch.LongTensor [B]
        seq_lens: torch.LongTensor [B] chứa độ dài thực tế của từng mẫu
        metas: List[Dict[str, Any]] chứa siêu dữ liệu truy vết
    """
    batch_size = len(batch)
    if batch_size == 0:
        raise ValueError("Không thể gom batch từ danh sách mẫu rỗng.")

    seq_lens = [item[2].item() for item in batch]
    t_max = max(seq_lens)
    dtype = batch[0][0][0].dtype

    # Cấp phát sẵn bộ nhớ đệm Zero-Padded
    b_p3 = torch.zeros(batch_size, t_max, 64, 80, 80, dtype=dtype)
    b_p4 = torch.zeros(batch_size, t_max, 128, 40, 40, dtype=dtype)
    b_p5 = torch.zeros(batch_size, t_max, 256, 20, 20, dtype=dtype)

    for i, item in enumerate(batch):
        feat, _, s_len, _ = item
        t_i = s_len.item()
        b_p3[i, :t_i] = feat[0][:t_i]
        b_p4[i, :t_i] = feat[1][:t_i]
        b_p5[i, :t_i] = feat[2][:t_i]

    b_labels = torch.stack([item[1] for item in batch])
    b_seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    b_metas = [item[3] for item in batch]

    return (b_p3, b_p4, b_p5), b_labels, b_seq_lens, b_metas


def _seed_worker(worker_id: int) -> None:
    """Hàm worker_init_fn đảm bảo tính tái lập (reproducibility) cho multi-processing."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_raw_video_dataloaders(
    dataset_dir: Union[str, Path],
    manifest_file: Optional[Union[str, Path]] = None,
    sample_interval: float = 0.1,
    seq_len: Optional[int] = None,
    onnx_model_path: Union[str, Path] = "checkpoints/backbone_neck.onnx",
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
    device: str = "auto",
    use_fp16: bool = False,
    augmenter: Optional[Any] = None,
    seed: int = 42,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Hàm Factory khởi tạo cặp DataLoader (train_loader, val_loader) nạp video thô.

    Returns:
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập kiểm định
    """
    train_dataset = RawVideoONNXDataset(
        dataset_dir=dataset_dir,
        manifest_file=manifest_file,
        split="train",
        sample_interval=sample_interval,
        seq_len=seq_len,
        onnx_model_path=onnx_model_path,
        device=device,
        use_fp16=use_fp16,
        augmenter=augmenter,
        window_sampling="random"
    )

    val_dataset = RawVideoONNXDataset(
        dataset_dir=dataset_dir,
        manifest_file=manifest_file,
        split="val",
        sample_interval=sample_interval,
        seq_len=seq_len,
        onnx_model_path=onnx_model_path,
        device=device,
        use_fp16=use_fp16,
        augmenter=None,
        window_sampling="center"
    )

    # Nếu tập train hoặc val rỗng do quét từ thư mục phẳng, tự động chia theo tỷ lệ train_ratio
    if len(val_dataset) == 0 and len(train_dataset) > 1:
        all_dataset = RawVideoONNXDataset(
            dataset_dir=dataset_dir,
            manifest_file=manifest_file,
            split="all",
            sample_interval=sample_interval,
            seq_len=seq_len,
            onnx_model_path=onnx_model_path,
            device=device,
            use_fp16=use_fp16
        )
        total_len = len(all_dataset)
        train_len = max(1, int(round(total_len * train_ratio)))
        val_len = max(1, total_len - train_len)

        generator = torch.Generator().manual_seed(seed)
        from torch.utils.data import random_split
        train_sub, val_sub = random_split(all_dataset, [train_len, val_len], generator=generator)
        train_dataset = train_sub
        val_dataset = val_sub

    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        collate_fn=collate_raw_video_features,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_raw_video_features,
        pin_memory=pin_memory,
        worker_init_fn=_seed_worker if num_workers > 0 else None
    )

    return train_loader, val_loader


# ==============================================================================
# 6. BỘ KIỂM THỬ TỰ LẬP HOÀN CHỈNH (SELF-CONTAINED UNIT TESTS)
# ==============================================================================
def _create_mock_video(
    video_path: Path,
    num_frames: int = 30,
    fps: float = 30.0,
    width: int = 320,
    height: int = 240,
    color_base: Tuple[int, int, int] = (100, 150, 200)
) -> None:
    """Tạo video giả lập (.mp4) bằng OpenCV VideoWriter."""
    video_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not out.isOpened():
        # Fallback codec nếu mp4v không khả dụng
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Tạo khung hình với gradient nhẹ để mô phỏng chuyển động
        frame = np.full((height, width, 3), color_base, dtype=np.uint8)
        # Vẽ một hình tròn chuyển động
        center_x = int((i / num_frames) * (width - 40)) + 20
        center_y = height // 2
        cv2.circle(frame, (center_x, center_y), 20, (255, 255, 255), -1)
        out.write(frame)

    out.release()


def main() -> None:
    """Hàm thực thi các bài kiểm thử xác minh toàn diện cho src/dataset1.py."""
    print("=" * 80)
    print("   KIỂM THỬ XÁC MINH TOÀN DIỆN CHO MODULE: src/dataset1.py")
    print("=" * 80)

    temp_dir = Path(tempfile.mkdtemp(prefix="test_raw_video_onnx_"))
    onnx_path = ROOT_DIR / "checkpoints" / "backbone_neck.onnx"

    if not onnx_path.exists():
        print(f"[!] CẢNH BÁO: Không tìm thấy tệp trọng số ONNX tại {onnx_path}. Bỏ qua kiểm thử ONNX.")
        return

    try:
        # 1. Khởi tạo cấu trúc dữ liệu mock (train/val với 2 video mỗi tập)
        print("\n[*] [Giai đoạn 1] Khởi tạo các tệp video mock giả lập đa tần số (30 FPS & 20 FPS)...")
        v1_train = temp_dir / "train" / "0_alert" / "clip_alert_30fps.mp4"
        v2_train = temp_dir / "train" / "1_drowsy" / "clip_drowsy_20fps.mp4"
        v1_val = temp_dir / "val" / "0_alert" / "clip_val_alert_30fps.mp4"
        v2_val = temp_dir / "val" / "1_drowsy" / "clip_val_drowsy_30fps.mp4"

        # Clip 1: 30 FPS, 60 frames = 2.0s
        _create_mock_video(v1_train, num_frames=60, fps=30.0, color_base=(50, 100, 150))
        # Clip 2: 20 FPS, 50 frames = 2.5s
        _create_mock_video(v2_train, num_frames=50, fps=20.0, color_base=(150, 80, 50))
        # Val clips: 30 FPS, 30 frames = 1.0s
        _create_mock_video(v1_val, num_frames=30, fps=30.0, color_base=(60, 120, 180))
        _create_mock_video(v2_val, num_frames=45, fps=30.0, color_base=(180, 60, 60))

        print(f"    Thư mục dữ liệu mock: {temp_dir}")
        print("    [✓] Đã tạo thành công 4 video clips giả lập chuẩn cấu trúc thư mục.")

        # 2. Kiểm thử lấy mẫu thời gian tùy chỉnh (sample_interval)
        print("\n[*] [Giai đoạn 2] Kiểm thử chu kỳ lấy mẫu khung hình thời gian tùy chỉnh (sample_interval):")
        # 2.1. Lấy mẫu với interval = 0.1s (~10 FPS)
        ds_10fps = RawVideoONNXDataset(
            dataset_dir=temp_dir,
            split="train",
            sample_interval=0.1,
            onnx_model_path=onnx_path,
            device="auto"
        )
        print(f"    -> ds_10fps nạp được {len(ds_10fps)} mẫu train.")
        assert len(ds_10fps) == 2, f"Mong đợi 2 mẫu, nhận được {len(ds_10fps)}"

        # Kiểm tra mẫu 0 (Clip 1: 60 frames @ 30 FPS = 2.0s. Với interval=0.1s -> step=3 -> 20 frames)
        frames_sample0, fps0, dur0 = ds_10fps._sample_video_frames(ds_10fps.samples[0].path)
        print(f"    Mẫu 0: duration={dur0:.1f}s, fps={fps0}, step=max(1, round({fps0}*0.1))={max(1, int(round(fps0*0.1)))}")
        print(f"    Số frame lấy mẫu (interval=0.1s): {len(frames_sample0)} (Mong đợi: ~20 frames)")
        assert abs(len(frames_sample0) - 20) <= 1, f"Sai lệch số frame lấy mẫu: {len(frames_sample0)}"

        # 2.2. Lấy mẫu với interval = 0.2s (~5 FPS)
        ds_5fps = RawVideoONNXDataset(
            dataset_dir=temp_dir,
            split="train",
            sample_interval=0.2,
            onnx_model_path=onnx_path,
            device="auto"
        )
        frames_sample0_5fps, _, _ = ds_5fps._sample_video_frames(ds_5fps.samples[0].path)
        print(f"    Số frame lấy mẫu (interval=0.2s): {len(frames_sample0_5fps)} (Mong đợi: ~10 frames)")
        assert abs(len(frames_sample0_5fps) - 10) <= 1, f"Sai lệch số frame lấy mẫu 5fps: {len(frames_sample0_5fps)}"
        print("    [✓] ĐẠT: Chu kỳ lấy mẫu tùy chỉnh sample_interval hoạt động chính xác 100%!")

        # 3. Kiểm thử trích xuất đặc trưng ONNX Runtime trực tiếp
        print("\n[*] [Giai đoạn 3] Kiểm thử trích xuất trực tiếp đặc trưng qua ONNX Runtime (__getitem__):")
        (p3, p4, p5), lbl, slen, meta = ds_10fps[0]
        print(f"    Mẫu 0: video_id={meta['video_id']}, T={slen.item()}, label={lbl.item()}")
        print(f"    Shapes: p3={p3.shape}, p4={p4.shape}, p5={p5.shape}, dtype={p3.dtype}")
        T_actual = slen.item()
        assert p3.shape == (T_actual, 64, 80, 80), f"Sai shape p3: {p3.shape}"
        assert p4.shape == (T_actual, 128, 40, 40), f"Sai shape p4: {p4.shape}"
        assert p5.shape == (T_actual, 256, 20, 20), f"Sai shape p5: {p5.shape}"
        assert not torch.isnan(p3).any(), "p3 chứa giá trị NaN!"
        print("    [✓] ĐẠT: Trích xuất đặc trưng ONNX sinh đúng 3 tensor đa tỷ lệ p3, p4, p5!")

        # 4. Kiểm thử Gom Batch Động (Dynamic Zero-Padding Collate)
        print("\n[*] [Giai đoạn 4] Kiểm thử gom batch động collate_raw_video_features (T1 != T2):")
        sample0 = ds_10fps[0] # T1 ~ 20
        sample1 = ds_10fps[1] # Clip 2: 50 frames @ 20 FPS = 2.5s -> step=2 -> 25 frames
        t1 = sample0[2].item()
        t2 = sample1[2].item()
        print(f"    Độ dài chuỗi 2 mẫu trong batch: T1={t1}, T2={t2}")

        (b_p3, b_p4, b_p5), b_lbls, b_lens, b_metas = collate_raw_video_features([sample0, sample1])
        t_max = max(t1, t2)
        print(f"    Batch shapes: p3={b_p3.shape}, p4={b_p4.shape}, p5={b_p5.shape}")
        print(f"    Batch labels={b_lbls.tolist()}, seq_lens={b_lens.tolist()}")
        assert b_p3.shape == (2, t_max, 64, 80, 80), f"Sai shape b_p3: {b_p3.shape}"
        assert b_lens.tolist() == [t1, t2], f"Sai seq_lens: {b_lens.tolist()}"

        # Kiểm tra phần padding ở cuối mẫu ngắn hơn phải bằng 0.0
        min_idx = 0 if t1 < t2 else 1
        min_t = min(t1, t2)
        pad_sum = b_p3[min_idx, min_t:].abs().sum().item()
        assert pad_sum == 0.0, f"Phát hiện padding rác không bằng 0: {pad_sum}"
        print("    [✓] ĐẠT: Dynamic Zero-Padding và tạo seq_lens chuẩn xác tuyệt đối!")

        # 5. Kiểm thử DataLoader Factory
        print("\n[*] [Giai đoạn 5] Kiểm thử DataLoader factory build_raw_video_dataloaders:")
        train_loader, val_loader = build_raw_video_dataloaders(
            dataset_dir=temp_dir,
            sample_interval=0.1,
            batch_size=2,
            num_workers=0,
            onnx_model_path=onnx_path,
            device="auto"
        )
        batch_cnt = 0
        for batch_data in train_loader:
            feats, targets, lens, metas = batch_data
            batch_cnt += 1
            print(f"    Nạp Batch {batch_cnt}: p3={feats[0].shape}, seq_lens={lens.tolist()}, labels={targets.tolist()}")
        assert batch_cnt == 1, f"Mong đợi 1 batch train, nhận được {batch_cnt}"
        print("    [✓] ĐẠT: build_raw_video_dataloaders vận hành mượt mà!")

        # 6. Kiểm thử Tích hợp Toàn diện End-to-End với CNNAdapter, DeepGRUClassifier & DrowsinessLoss
        print("\n[*] [Giai đoạn 6] Kiểm thử End-to-End Forward & Backward Pass với mô hình hạ tầng:")
        from src.models import DeepGRUClassifier
        from src.loss import DrowsinessLoss

        model = DeepGRUClassifier(
            input_dim=128,
            hidden_dim=96,
            num_layers=2,
            num_classes=2,
            spatial_in_channels=(64, 128, 256),
            fusion="concat"
        )
        loss_fn = DrowsinessLoss()

        # Forward Pass
        logits = model((b_p3, b_p4, b_p5), seq_lens=b_lens)
        print(f"    Model Logits Shape: {logits.shape} (Mong đợi: [2, 2])")
        assert logits.shape == (2, 2), f"Sai shape logits: {logits.shape}"

        # Loss calculation
        loss = loss_fn(logits, b_lbls)
        print(f"    Computed Loss: {loss.item():.4f}")
        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss bị NaN hoặc Inf!"

        # Backward Pass
        loss.backward()
        print("    Backward Pass: Đã lan truyền ngược thành công qua toàn bộ mạng!")
        conv_grad = model.spatial_adapter.pyramid_stage1[0].weight.grad
        assert conv_grad is not None, "Không tìm thấy gradient tại spatial_adapter!"
        print(f"    Gradient Stage1 Conv2d Norm: {conv_grad.norm().item():.6f}")
        print("    [✓] ĐẠT: Tích hợp hoàn hảo 100% với DeepGRUClassifier và DrowsinessLoss!")

        print("\n" + "=" * 80)
        print("    >>> TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ ĐẠT 100% TIÊU CHÍ CHẤT LƯỢNG! <<<")
        print("=" * 80)

    finally:
        # Dọn dẹp tài nguyên
        if 'ds_10fps' in locals():
            ds_10fps.close()
        if 'ds_5fps' in locals():
            ds_5fps.close()
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n[*] Đã dọn dẹp sạch sẽ thư mục mock tạm: {temp_dir}")


if __name__ == "__main__":
    main()
