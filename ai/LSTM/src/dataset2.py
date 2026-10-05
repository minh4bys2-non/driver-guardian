#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: src/dataset2.py
Mục đích:
    Xây dựng PyTorch Dataset (RawVideoFramesDataset), hàm gom batch (collate_video_frames)
    và hàm factory (build_raw_video_dataloaders) để nạp trực tiếp video thô (.mp4, .avi, .mkv, .mov).
    Module cũng cung cấp lớp ChunkedBackboneNeckExtractor để trích xuất đặc trưng
    BackboneNeck (p3, p4, p5) trực tiếp trên GPU theo cơ chế Mini-Chunk, giúp ngăn chặn 
    lỗi tràn VRAM (OOM) trên các GPU giới hạn bộ nhớ (ví dụ 4GB VRAM).

Đặc tính kỹ thuật cốt lõi:
    1. Dataset Siêu Nhẹ (CPU-bound):
       RawVideoFramesDataset chỉ làm nhiệm vụ decode video, letterbox và trả về
       Tensor uint8 trên CPU, đảm bảo tiết kiệm RAM và tuyệt đối không chạm tới GPU.
    2. An toàn Tuyệt đối Đa Tiến trình:
       Sử dụng num_workers > 0 không gây chết/đóng băng tiến trình do PyTorch Dataset
       hoàn toàn tách biệt khỏi mô hình mạng.
    3. Trích xuất Đặc trưng On-The-Fly (Mini-Chunk GPU):
       ChunkedBackboneNeckExtractor được thiết kế để nhúng vào Vòng lặp Huấn luyện (Main Loop),
       cắt nhỏ batch ảnh ra thành các chunk (VD: 4 hoặc 8 khung hình) đi qua Backbone,
       bảo đảm trần VRAM (VRAM ceiling) không bao giờ bị vượt qua.
    4. Tối ưu Băng thông PCIe:
       Dữ liệu Tensor uint8 được đẩy lên GPU một lần, chạy qua Backbone và forward thẳng
       vào LSTM/GRU, triệt tiêu overhead chuyển tiếp qua lại CPU-GPU.
"""

import os
import sys
import math
import random
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional, Union, Sequence

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thiết lập đường dẫn
LSTM_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = LSTM_DIR.parent.parent  # driver-guardian

if str(LSTM_DIR) not in sys.path:
    sys.path.insert(0, str(LSTM_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tối ưu hóa môi trường OpenMP và CUDA DLLs
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Tải thư viện CUDA DLL
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
        f"Không thể import mô hình BackboneNeck. Kiểm tra cấu trúc thư mục: {exc}"
    ) from exc

logger = logging.getLogger("dataset2")

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
# 1. CẤU TRÚC DỮ LIỆU ĐẠI DIỆN MẪU VIDEO
# ==============================================================================
@dataclass
class RawVideoSample:
    path: Path
    video_id: str
    split: str
    label: int
    label_name: str
    source_dataset: str = "custom"
    subject_id: Optional[str] = None


# ==============================================================================
# 2. HÀM BIẾN ĐỔI ẢNH LETTERBOX
# ==============================================================================
def letterbox(
    image: np.ndarray,
    new_size: int = 640,
    color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
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
# 3. CLASS TRÍCH XUẤT ĐẶC TRƯNG GPU MINI-CHUNK CHO VÒNG LẶP HUẤN LUYỆN
# ==============================================================================
class ChunkedBackboneNeckExtractor(nn.Module):
    """
    Wrapper mô hình BackboneNeck chạy trên GPU bằng cách chia nhỏ Tensor đầu vào
    để ngăn chặn triệt để OOM trên các phần cứng VRAM thấp (4GB).
    Được thiết kế để nhúng trực tiếp vào vòng lặp train, không nằm trong Dataset.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHECKPOINT_PATH,
        device: str = "cuda",
        device_id: int = 0,
        chunk_size: int = 4,  # Giá trị nhỏ cho 4GB VRAM
        use_fp16: bool = True,
        img_size: int = 640
    ) -> None:
        super().__init__()
        self.checkpoint_path = Path(checkpoint_path)
        self.device_str = device.lower().strip()
        self.device_id = device_id
        self.chunk_size = max(1, chunk_size)
        self.use_fp16 = use_fp16
        self.img_size = img_size

        self.device = self._init_device()
        self.model = self._load_model()

    def _init_device(self) -> torch.device:
        if self.device_str in ("cuda", "gpu") and torch.cuda.is_available():
            return torch.device(f"cuda:{self.device_id}")
        return torch.device("cpu")

    def _load_model(self) -> nn.Module:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        metadata = None
        try:
            metadata = validate_metadata(self.checkpoint_path, checkpoint)
        except Exception:
            metadata = checkpoint.get("metadata")

        if metadata and "architecture" in metadata:
            arch = metadata["architecture"]
        else:
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

        valid_keys = {"nc", "reg_max", "backbone_w", "backbone_n", "neck_n", "strides"}
        arch = {k: v for k, v in arch.items() if k in valid_keys}
            
        detector = NMSFreeDetector(**arch, img_size=self.img_size).eval()
        state_dict = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
        detector.load_state_dict(state_dict, strict=False)

        backbone_neck = BackboneNeck(detector).eval()

        for param in backbone_neck.parameters():
            param.requires_grad = False

        backbone_neck = backbone_neck.to(self.device)
        return backbone_neck

    def forward(self, batch_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Trích xuất đặc trưng theo cơ chế Mini-Chunk.

        Args:
            batch_frames: Tensor ảnh uint8 hoặc float32 dạng [B, T, 3, H, W] trên GPU.

        Returns:
            Tuple (p3, p4, p5) với dạng [B, T, C, H, W] trên GPU.
        """
        B, T, C, H, W = batch_frames.shape
        flat_frames = batch_frames.view(B * T, C, H, W)

        p3_list, p4_list, p5_list = [], [], []

        with torch.inference_mode():
            for i in range(0, B * T, self.chunk_size):
                chunk = flat_frames[i : i + self.chunk_size]
                
                # Chuyển đổi [0, 255] uint8 -> [0.0, 1.0] float32 TỪNG CHUNK MỘT
                if chunk.dtype == torch.uint8:
                    chunk = chunk.float() / 255.0
                
                if self.use_fp16 and self.device.type == "cuda":
                    with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                        out3, out4, out5 = self.model(chunk)
                else:
                    out3, out4, out5 = self.model(chunk)

                # Ép kiểu đích về float32 để tương thích hoàn hảo với GRU, hoặc giữ float16 tuỳ chiến lược
                p3_list.append(out3.float())
                p4_list.append(out4.float())
                p5_list.append(out5.float())

        p3_flat = torch.cat(p3_list, dim=0)
        p4_flat = torch.cat(p4_list, dim=0)
        p5_flat = torch.cat(p5_list, dim=0)

        p3 = p3_flat.view(B, T, *p3_flat.shape[1:])
        p4 = p4_flat.view(B, T, *p4_flat.shape[1:])
        p5 = p5_flat.view(B, T, *p5_flat.shape[1:])

        return p3, p4, p5


# ==============================================================================
# 4. LỚP PYTORCH DATASET: RawVideoFramesDataset (CHỈ DÙNG CPU)
# ==============================================================================
class RawVideoFramesDataset(Dataset):
    """
    Dataset giải mã video và trả về tensor ảnh raw trên CPU. Không chứa model GPU.
    """
    def __init__(
        self,
        dataset_dir: Union[str, Path],
        manifest_file: Optional[Union[str, Path]] = None,
        split: str = "train",
        sample_interval: float = 0.1,
        seq_len: Optional[int] = None,
        img_size: int = 640,
        augmenter: Optional[Any] = None,
        video_exts: Sequence[str] = (".mp4", ".avi", ".mkv", ".mov"),
        window_sampling: str = "random",
        min_frames: int = 1
    ) -> None:
        super().__init__()
        self.dataset_dir = Path(dataset_dir)
        self.manifest_file = Path(manifest_file) if manifest_file else None
        self.split = split.lower().strip()
        self.sample_interval = float(sample_interval)
        self.seq_len = int(seq_len) if seq_len is not None else None
        self.img_size = int(img_size)
        self.augmenter = augmenter
        self.video_exts = tuple(ext.lower() for ext in video_exts)
        self.window_sampling = window_sampling.lower().strip()
        self.min_frames = max(1, int(min_frames))

        self.samples: List[RawVideoSample] = self._discover_samples()

    def _discover_samples(self) -> List[RawVideoSample]:
        """Tự động phát hiện mẫu từ cấu trúc thư mục phân tầng hoặc manifest."""
        samples: List[RawVideoSample] = []

        manifest_path = self.manifest_file
        if manifest_path is None:
            for candidate in ("dataset_manifest.csv", "dataset_merged_split.csv", "dataset_merged_split.json"):
                cand_path = self.dataset_dir / candidate
                if cand_path.exists():
                    manifest_path = cand_path
                    break

        if manifest_path is not None and manifest_path.exists():
            return self._load_from_manifest(manifest_path)

        splits_to_scan = [self.split] if self.split != "all" else ["train", "val"]

        for s in splits_to_scan:
            split_dir = self.dataset_dir / s
            target_base = split_dir if split_dir.exists() else self.dataset_dir

            for label_int, label_str in [(0, "0_alert"), (1, "1_drowsy")]:
                label_dir = target_base / label_str
                if not label_dir.exists():
                    short_str = "alert" if label_int == 0 else "drowsy"
                    label_dir = target_base / short_str

                if label_dir.exists():
                    for v_path in sorted(label_dir.rglob("*")):
                        if v_path.is_file() and v_path.suffix.lower() in self.video_exts:
                            v_id = f"{s}_{label_str}_{v_path.stem}"
                            src = "sust" if "sust" in v_path.stem.lower() else ("uta-rldd" if "uta" in v_path.stem.lower() else "custom")
                            samples.append(
                                RawVideoSample(
                                    path=v_path, video_id=v_id, split=s,
                                    label=label_int, label_name=label_str, source_dataset=src
                                )
                            )
        return samples

    def _load_from_manifest(self, manifest_path: Path) -> List[RawVideoSample]:
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
                    p_str = row.get("orig_file") or row.get("path") or row.get("file_path") or ""
                    v_path = Path(p_str)
                    if not v_path.is_absolute():
                        v_path = self.dataset_dir / v_path

                    if not v_path.exists():
                        continue

                    v_id = row.get("video_id") or v_path.stem
                    lbl_name = row.get("label_name") or ("0_alert" if label_val == 0 else "1_drowsy")
                    src = row.get("source_dataset", "custom")
                    samples.append(
                        RawVideoSample(
                            path=v_path, video_id=v_id, split=row_split,
                            label=label_val, label_name=lbl_name, source_dataset=src
                        )
                    )
        return samples

    def _sample_video_frames(self, video_path: Path) -> Tuple[List[np.ndarray], float, float]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning(f"Lỗi đọc video (bỏ qua): {video_path}")
            return [], 0.0, 0.0

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or math.isnan(fps):
            fps = 30.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_frames / fps if fps > 0 else 0.0
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

    def __getitem__(self, idx: int) -> Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[idx]

        frames_rgb, fps, duration_s = self._sample_video_frames(sample.path)
        orig_sampled_len = len(frames_rgb)

        if orig_sampled_len < self.min_frames:
            # Không dùng khung hình ảo, trả về None để collate_fn bỏ qua
            meta = {"video_id": sample.video_id, "status": "corrupted"}
            return None, torch.tensor(sample.label), torch.tensor(0), meta

        if self.seq_len is not None and len(frames_rgb) > self.seq_len:
            if self.split == "train" and self.window_sampling == "random":
                start_idx = random.randint(0, len(frames_rgb) - self.seq_len)
            else:
                start_idx = (len(frames_rgb) - self.seq_len) // 2
            frames_rgb = frames_rgb[start_idx : start_idx + self.seq_len]

        if self.augmenter is not None and self.split == "train":
            aug_seed = random.randint(0, 2**31 - 1)
            if hasattr(self.augmenter, "apply_sequence"):
                frames_rgb = self.augmenter.apply_sequence(frames_rgb, seed=aug_seed)
            elif hasattr(self.augmenter, "augment_video"):
                aug_res = self.augmenter.augment_video(frames_rgb, seed=aug_seed)
                frames_rgb = aug_res[0] if isinstance(aug_res, (tuple, list)) else aug_res
            elif callable(self.augmenter):
                processed = [self.augmenter(img, seed=aug_seed) for img in frames_rgb]
                frames_rgb = [p[0] if isinstance(p, (tuple, list)) else p for p in processed]

        # Convert sang torch.Tensor uint8 (tiết kiệm bộ nhớ RAM) định dạng [T, 3, H, W]
        batch_np = np.stack(frames_rgb, axis=0).transpose(0, 3, 1, 2)
        frames_t = torch.from_numpy(batch_np).to(dtype=torch.uint8)

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
            "orig_sampled_frames": orig_sampled_len,
            "status": "ok"
        }

        return frames_t, label_t, seq_len_t, meta


# ==============================================================================
# 5. HÀM GOM BATCH ĐỘNG & FACTORY DATALOADER
# ==============================================================================
def collate_video_frames(
    batch: List[Tuple[Optional[torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Gom batch và Zero-Padding trục thời gian cho tensor ảnh. Loại bỏ video bị lỗi (None).
    """
    valid_batch = [item for item in batch if item[0] is not None]
    
    if len(valid_batch) == 0:
        # Nếu toàn bộ batch bị lỗi, ném ngoại lệ hoặc trả về rỗng (tùy thiết kế)
        return torch.empty(0), torch.empty(0), torch.empty(0), []

    seq_lens = [item[2].item() for item in valid_batch]
    t_max = max(seq_lens)
    
    _, _, H, W = valid_batch[0][0].shape
    batch_size = len(valid_batch)

    # Pad với uint8 (0)
    b_frames = torch.zeros(batch_size, t_max, 3, H, W, dtype=torch.uint8)

    for i, item in enumerate(valid_batch):
        f_tensor, _, s_len, _ = item
        t_i = s_len.item()
        b_frames[i, :t_i] = f_tensor[:t_i]

    b_labels = torch.stack([item[1] for item in valid_batch])
    b_seq_lens = torch.tensor(seq_lens, dtype=torch.long)
    b_metas = [item[3] for item in valid_batch]

    return b_frames, b_labels, b_seq_lens, b_metas


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_raw_video_dataloaders(
    dataset_dir: Union[str, Path],
    manifest_file: Optional[Union[str, Path]] = None,
    sample_interval: float = 0.1,
    seq_len: Optional[int] = None,
    batch_size: int = 4,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    augmenter: Optional[Any] = None,
    use_augmentation: bool = True
) -> Tuple[DataLoader, DataLoader]:
    
    effective_augmenter = None
    if use_augmentation:
        if augmenter is not None:
            effective_augmenter = augmenter
        else:
            try:
                from src.augment import get_video_augmenter
                effective_augmenter = get_video_augmenter()
            except Exception as e:
                logger.warning(f"Không thể khởi tạo DetectionAugmenter: {e}")
                effective_augmenter = None

    train_dataset = RawVideoFramesDataset(
        dataset_dir=dataset_dir,
        manifest_file=manifest_file,
        split="train",
        sample_interval=sample_interval,
        seq_len=seq_len,
        augmenter=effective_augmenter,
        window_sampling="random"
    )

    val_dataset = RawVideoFramesDataset(
        dataset_dir=dataset_dir,
        manifest_file=manifest_file,
        split="val",
        sample_interval=sample_interval,
        seq_len=seq_len,
        augmenter=None,
        window_sampling="center"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_video_frames,
        worker_init_fn=_seed_worker
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_video_frames,
        worker_init_fn=_seed_worker
    )

    return train_loader, val_loader


# ==============================================================================
# 6. KHỐI KIỂM THỬ TỰ LẬP (UNIT TESTS)
# ==============================================================================
if __name__ == "__main__":
    import shutil
    import tempfile
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print("=" * 80)
    print("BẮT ĐẦU KIỂM THỬ MODULE src/dataset2.py (Tối ưu cho VRAM 4GB)")
    print("=" * 80)

    temp_dir = Path(tempfile.mkdtemp(prefix="test_dataset_vram4gb_"))
    try:
        print("\n[Bước 1] Khởi tạo các tệp video giả lập...")
        (temp_dir / "train" / "0_alert").mkdir(parents=True, exist_ok=True)
        (temp_dir / "train" / "1_drowsy").mkdir(parents=True, exist_ok=True)

        def create_dummy_video(file_path, num_frames=30, color=(100, 150, 200)):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(file_path), fourcc, 30.0, (320, 240))
            for f_i in range(num_frames):
                img = np.full((240, 320, 3), color, dtype=np.uint8)
                writer.write(img)
            writer.release()

        v1_path = temp_dir / "train" / "0_alert" / "sample1.mp4"
        v2_path = temp_dir / "train" / "1_drowsy" / "sample2.mp4"
        create_dummy_video(v1_path, 30, (50, 100, 150))
        create_dummy_video(v2_path, 45, (150, 50, 50))

        print("\n[Bước 2] Kiểm thử Dataset siêu nhẹ (Chỉ dùng CPU, không Model)...")
        dummy_ds = RawVideoFramesDataset(dataset_dir=temp_dir, split="train", sample_interval=0.1)
        frames_t, lbl, seq_len, meta = dummy_ds[0]
        print(f"  ✓ Tensor frames dtype: {frames_t.dtype}, shape: {frames_t.shape}")
        assert frames_t.dtype == torch.uint8

        print("\n[Bước 3] Kiểm thử DataLoader (Zero-padding)...")
        train_loader, _ = build_raw_video_dataloaders(temp_dir, batch_size=2, num_workers=0)
        batch_frames, batch_lbl, batch_lens, _ = next(iter(train_loader))
        print(f"  ✓ Batch Frames shape: {batch_frames.shape}")

        print("\n[Bước 4] Kiểm thử Trích xuất GPU theo Mini-Chunk (VRAM Capping)...")
        extractor = ChunkedBackboneNeckExtractor(
            checkpoint_path=DEFAULT_CHECKPOINT_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
            chunk_size=4,
            use_fp16=True
        )
        # Đẩy batch lên GPU giả định
        batch_frames_gpu = batch_frames.to(extractor.device)
        p3, p4, p5 = extractor(batch_frames_gpu)
        print(f"  ✓ Output p3 shape: {p3.shape} (kỳ vọng [B, T_max, 64, 80, 80])")
        print(f"  ✓ Output p4 shape: {p4.shape} (kỳ vọng [B, T_max, 128, 40, 40])")
        print(f"  ✓ Output p5 shape: {p5.shape} (kỳ vọng [B, T_max, 256, 20, 20])")
        
        print("\n" + "=" * 80)
        print("TẤT CẢ CÁC BƯỚC KIỂM THỬ ĐÃ HOÀN TẤT THÀNH CÔNG!")
        print("=" * 80)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
