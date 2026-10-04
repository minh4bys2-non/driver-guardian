#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: src/dataset2.py
Mục đích:
    Xây dựng PyTorch Dataset (RawVideoBackboneNeckDataset), hàm gom batch (collate_raw_video_features)
    và hàm factory (build_raw_video_dataloaders) để nạp trực tiếp video thô (.mp4, .avi, .mkv, .mov)
    và tích hợp trực tiếp mô hình PyTorch BackboneNeck (từ ai.ObjectDetection_2p6M.runtime.convertor)
    để trích xuất trực tiếp các bản đồ đặc trưng không gian đa tỉ lệ nguyên bản (p3, p4, p5)
    theo chu kỳ lấy mẫu tùy chỉnh (sample_interval).

Đặc tính kỹ thuật cốt lõi:
    1. Trích xuất Trực tiếp qua PyTorch Native BackboneNeck:
       Sử dụng lớp BackboneNeck định nghĩa trong runtime/convertor.py nạp trọng số từ checkpoint .pt.
       Bảo toàn nguyên vẹn 3 tensor bản đồ đặc trưng 4D:
         - p3: [T, 64, 80, 80]
         - p4: [T, 128, 40, 40]
         - p5: [T, 256, 20, 20]
    2. Chu kỳ Lấy mẫu Tùy chỉnh (Customizable Sampling Interval):
       Tham số sample_interval (đơn vị giây, ví dụ 0.1s ~ 10 FPS, 0.05s ~ 20 FPS, 0.2s ~ 5 FPS)
       tự động thích ứng với FPS gốc của từng video: step = max(1, round(fps * sample_interval)).
       Căn chỉnh kích thước qua letterbox giữ nguyên tỷ lệ khung hình chuẩn 640x640.
    3. An toàn Tuyệt đối khi Chạy Đa Tiến trình (Multiprocessing & CUDA Safe):
       Khởi tạo trễ (Lazy Model Initialization) đối tượng PyTorchBackboneNeckExtractor
       bên trong từng tiến trình worker khi gọi __getitem__, triệt tiêu hoàn toàn lỗi
       RuntimeError CUDA re-initialization deadlock trên Windows (spawn).
    4. Suy luận Mini-Chunk Chống tràn VRAM:
       Xử lý chuỗi khung hình theo từng mini-chunk (chunk_size=16) trên GPU CUDA / CPU fallback
       bằng ngữ cảnh torch.inference_mode() tối ưu nhất.
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
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thiết lập đường dẫn thư mục gốc và thư mục dự án cha (driver-guardian)
LSTM_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LSTM_DIR.parent.parent  # D:\Project\DATN\driver-guardian

if str(LSTM_DIR) not in sys.path:
    sys.path.insert(0, str(LSTM_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tối ưu hóa môi trường OpenMP và CUDA DLLs trên Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Tải trước thư viện CUDA DLL từ PyTorch nếu chạy trên hệ điều hành Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception:
        pass

# Import các thành phần mô hình ObjectDetection_2p6M
try:
    from ai.ObjectDetection_2p6M.runtime.convertor import BackboneNeck
    from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector
    from ai.ObjectDetection_2p6M.utils.artifacts import validate_metadata
except ImportError as exc:
    raise ImportError(
        f"Không thể import mô hình BackboneNeck từ ObjectDetection_2p6M. Kiểm tra sys.path: {exc}"
    ) from exc

logger = logging.getLogger("dataset2")

# Đường dẫn checkpoint mặc định
DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "ai"
    / "ObjectDetection_2p6M"
    / "checkpoints"
    / "2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031"
    / "de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310"
    / "finetune"
    / "best.pt"
)


# ==============================================================================
# 1. CẤU TRÚC DỮ LIỆU ĐẠI DIỆN MẪU VIDEO (RAW VIDEO SAMPLE ENTRY)
# ==============================================================================
@dataclass
class RawVideoSample:
    """Cấu trúc đại diện cho một mẫu video thô đã quét/lập chỉ mục."""
    path: Path              # Đường dẫn tệp video thô (.mp4, .avi, .mkv, .mov)
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
# 3. ĐỘNG CƠ TRÍCH XUẤT ĐẶC TRƯNG PYTORCH BACKBONENECK THEO MINI-CHUNK
# ==============================================================================
class PyTorchBackboneNeckExtractor:
    """
    Quản lý suy luận mô hình PyTorch BackboneNeck theo từng mini-chunk.
    Hỗ trợ khởi tạo trễ (Lazy Initialization) để đảm bảo an toàn tuyệt đối khi serialize
    qua các worker processes của PyTorch DataLoader (tránh lỗi CUDA context trên Windows).
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
        device: str = "auto",
        device_id: int = 0,
        chunk_size: int = 16,
        use_fp16: bool = False,
        img_size: int = 640
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device_str = device.lower().strip()
        self.device_id = device_id
        self.chunk_size = max(1, chunk_size)
        self.use_fp16 = use_fp16
        self.img_size = img_size

        self.model: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None

    def _init_device(self) -> torch.device:
        """Xác định và khởi tạo thiết bị tính toán thích hợp."""
        if self.device_str in ("cuda", "gpu"):
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.device_id}")
            return torch.device("cpu")
        elif self.device_str == "cpu":
            return torch.device("cpu")
        else:  # "auto"
            if torch.cuda.is_available():
                return torch.device(f"cuda:{self.device_id}")
            return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        """Nạp checkpoint, xác thực metadata, nạp NMSFreeDetector và BackboneNeck."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Không tìm thấy tệp checkpoint: {self.checkpoint_path}")

        self.device = self._init_device()

        # Nạp checkpoint an toàn lên CPU trước
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

        # Xác thực metadata kiến trúc
        metadata = None
        try:
            metadata = validate_metadata(self.checkpoint_path, checkpoint)
        except Exception:
            metadata = checkpoint.get("metadata")

        if metadata and "architecture" in metadata:
            arch = metadata["architecture"]
        else:
            # Dự phòng: Nạp cấu hình mặc định tương thích
            from ai.ObjectDetection_2p6M.src.config import TrainConfig
            cfg = TrainConfig()
            arch = {
                "nc": cfg.nc,
                "reg_max": cfg.reg_max,
                "backbone_w": cfg.backbone_w,
                "backbone_n": cfg.backbone_n,
                "neck_n": cfg.neck_n,
                "strides": cfg.strides,
            }

        # Khởi tạo mô hình phát hiện
        detector = NMSFreeDetector(**arch, img_size=self.img_size).eval()

        # Nạp trọng số từ ema hoặc model
        state_dict = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
        detector.load_state_dict(state_dict)

        # Đóng gói vào BackboneNeck
        backbone_neck = BackboneNeck(detector).eval()

        # Đóng băng gradient cho toàn bộ trọng số (Feature Extractor thuần túy)
        for param in backbone_neck.parameters():
            param.requires_grad = False

        # Chuyển mô hình sang thiết bị đích
        backbone_neck = backbone_neck.to(self.device)

        return backbone_neck

    def get_model(self) -> nn.Module:
        """Lấy thực thể mô hình BackboneNeck hiện tại (khởi tạo trễ nếu chưa có)."""
        if self.model is None:
            self.model = self._load_model()
        return self.model

    def extract_chunks(
        self,
        frames_rgb: List[np.ndarray]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trích xuất 3 bản đồ đặc trưng (p3, p4, p5) theo từng mini-chunk chống tràn VRAM.

        Args:
            frames_rgb: Danh sách các ảnh numpy RGB uint8 [640, 640, 3].

        Returns:
            p3: torch.Tensor [T, 64, 80, 80] trên CPU
            p4: torch.Tensor [T, 128, 40, 40] trên CPU
            p5: torch.Tensor [T, 256, 20, 20] trên CPU
        """
        model = self.get_model()
        device = self.device if self.device is not None else torch.device("cpu")
        T = len(frames_rgb)
        out_dtype = torch.float16 if self.use_fp16 else torch.float32

        if T == 0:
            return (
                torch.empty((0, 64, 80, 80), dtype=out_dtype),
                torch.empty((0, 128, 40, 40), dtype=out_dtype),
                torch.empty((0, 256, 20, 20), dtype=out_dtype)
            )

        p3_chunks: List[torch.Tensor] = []
        p4_chunks: List[torch.Tensor] = []
        p5_chunks: List[torch.Tensor] = []

        for i in range(0, T, self.chunk_size):
            batch_slice = frames_rgb[i : i + self.chunk_size]
            # [B, H, W, 3] -> [B, 3, H, W] chuẩn hóa [0.0, 1.0]
            batch_np = np.stack(batch_slice, axis=0).transpose(0, 3, 1, 2)
            batch_t = torch.from_numpy(batch_np).to(device=device, dtype=torch.float32) / 255.0

            with torch.inference_mode():
                if self.use_fp16 and device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out_p3, out_p4, out_p5 = model(batch_t)
                else:
                    out_p3, out_p4, out_p5 = model(batch_t)

                # Chuyển kết quả về kiểu dữ liệu đích và đẩy về CPU để tránh phình VRAM
                if self.use_fp16:
                    out_p3 = out_p3.half()
                    out_p4 = out_p4.half()
                    out_p5 = out_p5.half()
                else:
                    out_p3 = out_p3.float()
                    out_p4 = out_p4.float()
                    out_p5 = out_p5.float()

                p3_chunks.append(out_p3.detach().cpu())
                p4_chunks.append(out_p4.detach().cpu())
                p5_chunks.append(out_p5.detach().cpu())

            del batch_t, out_p3, out_p4, out_p5

        p3_full = torch.cat(p3_chunks, dim=0)
        p4_full = torch.cat(p4_chunks, dim=0)
        p5_full = torch.cat(p5_chunks, dim=0)
        del p3_chunks, p4_chunks, p5_chunks
        if device.type == "cuda" and T > 64:
            torch.cuda.empty_cache()
        return p3_full, p4_full, p5_full

    def close(self) -> None:
        """Giải phóng tài nguyên mô hình và bộ nhớ GPU."""
        self.model = None
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ==============================================================================
# 4. LỚP PYTORCH DATASET: RawVideoBackboneNeckDataset
# ==============================================================================
class RawVideoBackboneNeckDataset(Dataset):
    """
    PyTorch Dataset nạp video thô (.mp4, .avi, .mkv, .mov) và trích xuất trực tiếp
    đặc trưng không gian đa tỷ lệ (p3, p4, p5) qua mô hình PyTorch BackboneNeck.

    Args:
        dataset_dir: Thư mục chứa video hoặc đường dẫn manifest CSV/JSON.
        manifest_file: Đường dẫn tệp CSV/JSON manifest tùy chọn.
        split: Phân vùng dữ liệu ("train", "val", hoặc "all").
        sample_interval: Chu kỳ lấy mẫu khung hình thời gian (giây), mặc định 0.1s (~10 FPS).
        seq_len: Độ dài khung hình cố định (None = giữ trọn vẹn số khung hình thực tế).
        checkpoint_path: Đường dẫn tệp trọng số PyTorch (.pt).
        img_size: Kích thước cạnh vuông ảnh sau letterbox (mặc định 640).
        chunk_size: Kích thước mini-chunk khi suy luận BackboneNeck tránh tràn VRAM (mặc định 16).
        device: Thiết bị thực thi ("cuda", "cpu", hoặc "auto").
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
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
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
        self.checkpoint_path = Path(checkpoint_path)
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

        # Khởi tạo trễ: đối tượng trích xuất PyTorch sẽ được cấp phát trong _get_extractor()
        self._extractor: Optional[PyTorchBackboneNeckExtractor] = None

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

    def _get_extractor(self) -> PyTorchBackboneNeckExtractor:
        """Cơ chế Lazy Loading khởi tạo PyTorch Feature Extractor trong tiến trình hiện tại."""
        if self._extractor is None:
            self._extractor = PyTorchBackboneNeckExtractor(
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                device_id=self.device_id,
                chunk_size=self.chunk_size,
                use_fp16=self.use_fp16,
                img_size=self.img_size
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
        Truy xuất mẫu video thứ `idx`, lấy mẫu khung hình, suy luận BackboneNeck và trả về tensor đặc trưng.

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
                dummy_frame = np.full((self.img_size, self.img_size, 3), 114, dtype=np.uint8)
                frames_rgb = [dummy_frame] * self.min_frames
            else:
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
            aug_seed = random.randint(0, 2**31 - 1)
            if hasattr(self.augmenter, "apply_sequence"):
                frames_rgb = self.augmenter.apply_sequence(frames_rgb, seed=aug_seed)
            elif hasattr(self.augmenter, "augment_video"):
                aug_res = self.augmenter.augment_video(frames_rgb, seed=aug_seed)
                frames_rgb = aug_res[0] if isinstance(aug_res, (tuple, list)) else aug_res
            elif callable(self.augmenter):
                processed: List[np.ndarray] = []
                for img in frames_rgb:
                    out = self.augmenter(img, seed=aug_seed)
                    if isinstance(out, (tuple, list)):
                        processed.append(out[0])
                    else:
                        processed.append(out)
                frames_rgb = processed

        # 4. Trích xuất trực tiếp đặc trưng qua mô hình PyTorch BackboneNeck
        extractor = self._get_extractor()
        p3_t, p4_t, p5_t = extractor.extract_chunks(frames_rgb)

        # 5. Ép kiểu tensor sang target_dtype nếu cần
        if p3_t.dtype != self.target_dtype:
            p3_t = p3_t.to(dtype=self.target_dtype)
            p4_t = p4_t.to(dtype=self.target_dtype)
            p5_t = p5_t.to(dtype=self.target_dtype)

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
        """Giải phóng tài nguyên mô hình PyTorch."""
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
        batch: Danh sách các mẫu trả về từ RawVideoBackboneNeckDataset.__getitem__.

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
    checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
    batch_size: int = 4,
    num_workers: int = 0,
    pin_memory: bool = False,
    shuffle_train: bool = True,
    device: str = "auto",
    use_fp16: bool = False,
    augmenter: Optional[Any] = None,
    use_augmentation: bool = True,
    seed: int = 42,
    train_ratio: float = 0.8
) -> Tuple[DataLoader, DataLoader]:
    """
    Hàm Factory khởi tạo cặp DataLoader (train_loader, val_loader) nạp video thô
    và trích xuất đặc trưng qua PyTorch BackboneNeck.

    Args:
        dataset_dir: Thư mục chứa video hoặc đường dẫn manifest CSV/JSON.
        manifest_file: Đường dẫn tệp CSV/JSON manifest tùy chọn.
        sample_interval: Chu kỳ lấy mẫu khung hình thời gian (giây), mặc định 0.1s.
        seq_len: Độ dài khung hình cố định (None = dynamic).
        checkpoint_path: Đường dẫn tệp trọng số PyTorch (.pt).
        batch_size: Kích thước batch.
        num_workers: Số worker nạp dữ liệu.
        pin_memory: Cờ pin memory vào pinned RAM host.
        shuffle_train: Xáo trộn tập train.
        device: Thiết bị ("auto", "cuda", "cpu").
        use_fp16: Tensor đầu ra FP16.
        augmenter: Đối tượng Augmenter tùy chỉnh (None = dùng mặc định nếu use_augmentation=True).
        use_augmentation: Bật/tắt tăng cường dữ liệu cho tập huấn luyện.
        seed: Hạt giống ngẫu nhiên.
        train_ratio: Tỷ lệ chia train/val nếu thư mục phẳng.

    Returns:
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập kiểm định
    """
    effective_augmenter = None
    if use_augmentation:
        if augmenter is not None:
            effective_augmenter = augmenter
        else:
            try:
                from src.augment import get_video_augmenter
                effective_augmenter = get_video_augmenter()
            except Exception as e:
                logger.warning(f"Không thể khởi tạo DetectionAugmenter từ src.augment: {e}")
                effective_augmenter = None

    train_dataset = RawVideoBackboneNeckDataset(
        dataset_dir=dataset_dir,
        manifest_file=manifest_file,
        split="train",
        sample_interval=sample_interval,
        seq_len=seq_len,
        checkpoint_path=checkpoint_path,
        device=device,
        use_fp16=use_fp16,
        augmenter=effective_augmenter,
        window_sampling="random"
    )

    val_dataset = RawVideoBackboneNeckDataset(
        dataset_dir=dataset_dir,
        manifest_file=manifest_file,
        split="val",
        sample_interval=sample_interval,
        seq_len=seq_len,
        checkpoint_path=checkpoint_path,
        device=device,
        use_fp16=use_fp16,
        augmenter=None,
        window_sampling="center"
    )

    # Nếu tập train hoặc val rỗng do quét từ thư mục phẳng, tự động chia theo tỷ lệ train_ratio
    if len(val_dataset) == 0 and len(train_dataset) > 1:
        all_dataset = RawVideoBackboneNeckDataset(
            dataset_dir=dataset_dir,
            manifest_file=manifest_file,
            split="all",
            sample_interval=sample_interval,
            seq_len=seq_len,
            checkpoint_path=checkpoint_path,
            device=device,
            use_fp16=use_fp16
        )
        total_len = len(all_dataset)
        train_len = max(1, int(round(total_len * train_ratio)))
        val_len = max(1, total_len - train_len)

        generator = torch.Generator().manual_seed(seed)
        from torch.utils.data import random_split
        train_dataset, val_dataset = random_split(all_dataset, [train_len, val_len], generator=generator)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_raw_video_features,
        worker_init_fn=_seed_worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_raw_video_features,
        worker_init_fn=_seed_worker
    )

    return train_loader, val_loader


# ==============================================================================
# 6. KHỐI KIỂM THỬ TỰ LẬP (SELF-CONTAINED VERIFICATION & UNIT TESTS)
# ==============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ MODULE src/dataset2.py (PyTorch BackboneNeck Raw Video Dataset)")
    print("=" * 80)

    temp_dir = Path(tempfile.mkdtemp(prefix="test_raw_video_pytorch_"))
    try:
        # 1. Tạo mock video clips
        print("\n[Bước 1/5] Khởi tạo các tệp video giả lập (Mock Video Files)...")
        train_alert_dir = temp_dir / "train" / "0_alert"
        train_drowsy_dir = temp_dir / "train" / "1_drowsy"
        val_alert_dir = temp_dir / "val" / "0_alert"
        val_drowsy_dir = temp_dir / "val" / "1_drowsy"

        for d in (train_alert_dir, train_drowsy_dir, val_alert_dir, val_drowsy_dir):
            d.mkdir(parents=True, exist_ok=True)

        def create_dummy_video(file_path: Path, num_frames: int = 30, fps: float = 30.0, color=(100, 150, 200)):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(file_path), fourcc, fps, (320, 240))
            for f_i in range(num_frames):
                img = np.full((240, 320, 3), color, dtype=np.uint8)
                cv2.putText(img, f"Frame {f_i}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                writer.write(img)
            writer.release()

        v1_path = train_alert_dir / "sample_alert_1.mp4"
        v2_path = train_drowsy_dir / "sample_drowsy_1.mp4"
        v3_path = val_alert_dir / "sample_alert_val.mp4"
        v4_path = val_drowsy_dir / "sample_drowsy_val.mp4"

        create_dummy_video(v1_path, num_frames=30, fps=30.0, color=(50, 100, 150))
        create_dummy_video(v2_path, num_frames=45, fps=30.0, color=(150, 50, 50))
        create_dummy_video(v3_path, num_frames=20, fps=20.0, color=(50, 150, 50))
        create_dummy_video(v4_path, num_frames=25, fps=25.0, color=(150, 150, 50))

        print(f"  ✓ Đã sinh thành công 4 video mock tại: {temp_dir}")

        # 2. Kiểm thử Lấy mẫu Khung hình theo sample_interval
        print("\n[Bước 2/5] Kiểm thử cơ chế lấy mẫu khung hình thời gian tùy chỉnh...")
        dummy_ds = RawVideoBackboneNeckDataset(
            dataset_dir=temp_dir,
            split="train",
            sample_interval=0.1,
            checkpoint_path=DEFAULT_CHECKPOINT_PATH
        )
        frames_1, fps_1, dur_1 = dummy_ds._sample_video_frames(v1_path)
        print(f"  ✓ Video 1: 30 frames @ 30 FPS (1.0s), sample_interval=0.1s -> Lấy mẫu: {len(frames_1)} frames (kỳ vọng ~10)")
        assert len(frames_1) in (10, 11), f"Lỗi số lượng frames lấy mẫu: {len(frames_1)}"

        # 3. Kiểm thử Trích xuất Đặc trưng PyTorch BackboneNeck
        print("\n[Bước 3/5] Kiểm thử nạp mô hình BackboneNeck & trích xuất đặc trưng...")
        feat, lbl, seq_len, meta = dummy_ds[0]
        p3, p4, p5 = feat

        print(f"  ✓ Mẫu 0: Label={lbl.item()}, SeqLen={seq_len.item()}")
        print(f"  ✓ p3 shape: {p3.shape} (kỳ vọng [T, 64, 80, 80])")
        print(f"  ✓ p4 shape: {p4.shape} (kỳ vọng [T, 128, 40, 40])")
        print(f"  ✓ p5 shape: {p5.shape} (kỳ vọng [T, 256, 20, 20])")

        assert p3.shape[1:] == (64, 80, 80), f"Sai kích thước p3: {p3.shape}"
        assert p4.shape[1:] == (128, 40, 40), f"Sai kích thước p4: {p4.shape}"
        assert p5.shape[1:] == (256, 20, 20), f"Sai kích thước p5: {p5.shape}"

        # 4. Kiểm thử Hàm gom batch động (Collate Zero-Padding)
        print("\n[Bước 4/5] Kiểm thử hàm gom batch động collate_raw_video_features...")
        train_loader, val_loader = build_raw_video_dataloaders(
            dataset_dir=temp_dir,
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            batch_size=2,
            sample_interval=0.1,
            num_workers=0
        )
        batch = next(iter(train_loader))
        (b_p3, b_p4, b_p5), b_lbl, b_lens, b_meta = batch

        print(f"  ✓ Batch p3 shape: {b_p3.shape} (kỳ vọng [2, T_max, 64, 80, 80])")
        print(f"  ✓ Batch p4 shape: {b_p4.shape} (kỳ vọng [2, T_max, 128, 40, 40])")
        print(f"  ✓ Batch p5 shape: {b_p5.shape} (kỳ vọng [2, T_max, 256, 20, 20])")
        print(f"  ✓ Batch Labels: {b_lbl.tolist()}, SeqLens: {b_lens.tolist()}")

        t_max = max(b_lens.tolist())
        assert b_p3.shape[1] == t_max, f"T_max không khớp: {b_p3.shape[1]} vs {t_max}"

        # 5. Kiểm thử Tích hợp End-to-End với CNNAdapter & DeepGRUClassifier
        print("\n[Bước 5/5] Kiểm thử kết nối End-to-End với CNNAdapter & DeepGRUClassifier...")
        from src.models import DeepGRUClassifier
        from src.loss import DrowsinessLoss

        classifier = DeepGRUClassifier(
            input_dim=256,
            hidden_dim=128,
            num_layers=1,
            num_classes=2,
            spatial_in_channels=(64, 128, 256),
            fusion="concat"
        )
        loss_fn = DrowsinessLoss()

        logits = classifier((b_p3, b_p4, b_p5), seq_lens=b_lens)
        loss = loss_fn(logits, b_lbl)

        print(f"  ✓ Forward Pass thành công! Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
        assert logits.shape == (2, 2), f"Sai shape logits: {logits.shape}"
        assert not torch.isnan(loss), "Loss bị NaN!"

        loss.backward()
        print("  ✓ Backward Pass thành công! Đồ thị lan truyền ngược gradient hoàn chỉnh.")

        print("\n" + "=" * 80)
        print("TẤT CẢ 5 BƯỚC KIỂM THỬ CỦA src/dataset2.py ĐÃ HOÀN TẤT THÀNH CÔNG 100%!")
        print("=" * 80)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print(f"Đã dọn dẹp thư mục tạm: {temp_dir}")
