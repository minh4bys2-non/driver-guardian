#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tệp: src/dataset.py
Mục đích:
    Xây dựng PyTorch Dataset (HDF5FeatureDataset) và hàm gom batch (collate_h5_features)
    nạp trực tiếp các tensor đặc trưng không gian đa tỉ lệ nguyên bản (p3, p4, p5)
    từ tệp HDF5 (.h5) được trích xuất bởi extract_to_pt.py.

Đặc tính kỹ thuật cốt lõi:
    1. An toàn Multi-processing (Safe Multiprocessing):
       Áp dụng cơ chế Lazy File Opening per-process kết hợp cờ SWMR (Single-Writer-Multiple-Reader),
       triệt tiêu hoàn toàn nguy cơ deadlock hoặc lỗi chia sẻ file handle khi num_workers > 0.
    2. Cơ chế Lập chỉ mục kép (Dual Indexing):
       Ưu tiên đọc siêu dữ liệu từ dataset_manifest.csv nếu có; tự động quét cây nhóm HDF5
       khi không có file CSV đi kèm, chỉ nạp các video có cờ 'is_completed == True'.
    3. Cắt lát cửa sổ thời gian (Temporal Windowing):
       Hỗ trợ cắt lát cửa sổ ngẫu nhiên (train) và chính giữa (val) nếu đặt cố định seq_len.
    4. Gom batch động (Dynamic Zero-Padding Collate):
       Tự động pad 0 theo trục thời gian về độ dài lớn nhất (T_max) trong từng batch và sinh
       tensor seq_lens [B], tích hợp hoàn hảo với cơ chế Attention Masking của TemporalAttentionPooling
       trong DeepGRUClassifier (triệt tiêu 100% gradient rác).
    5. Đồng bộ cấu hình:
       Cung cấp hàm factory build_h5_dataloaders tương thích với TrainConfig từ configs/config.py.
"""

import os
import sys
import csv
import random
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any, Optional, Union

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset, DataLoader


# ==============================================================================
# 1. CẤU TRÚC DỮ LIỆU ĐẠI DIỆN MẪU ĐÃ LẬP CHỈ MỤC (INDEXED SAMPLE ENTRY)
# ==============================================================================
@dataclass
class H5VideoSample:
    """Cấu trúc đại diện cho một mẫu video đã lập chỉ mục từ tệp HDF5 hoặc CSV."""
    group_path: str         # Đường dẫn nhóm trong H5 (vd: "train/0_alert/train_0_alert_sust_n_1")
    video_id: str           # Khóa định danh duy nhất (vd: "train_0_alert_sust_n_1")
    split: str              # "train" hoặc "val"
    label: int              # 0 (alert) hoặc 1 (drowsy)
    label_name: str         # "0_alert" hoặc "1_drowsy"
    seq_len: int            # Số lượng khung hình thời gian thực tế T
    is_augmented: bool      # True nếu là bản sao tăng cường
    source_dataset: str     # Nguồn dữ liệu (vd: "sust", "uta-rldd", "vbddd")
    orig_file: str          # Đường dẫn video gốc nếu có


# ==============================================================================
# 2. LỚP PYTORCH DATASET: HDF5FeatureDataset
# ==============================================================================
class HDF5FeatureDataset(Dataset):
    """
    PyTorch Dataset nạp các tensor đặc trưng không gian (p3, p4, p5) trực tiếp từ tệp HDF5 (.h5).

    Đặc trưng đầu ra cho mỗi mẫu:
      - p3: [T_win, 64, 80, 80]
      - p4: [T_win, 128, 40, 40]
      - p5: [T_win, 256, 20, 20]
      - label: torch.LongTensor scalar
      - seq_len: torch.LongTensor scalar (độ dài thực tế)
      - meta: dict chứa siêu dữ liệu bổ trợ
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        manifest_csv: Optional[Union[str, Path]] = None,
        split: str = "train",
        seq_len: Optional[int] = None,
        stride: int = 1,
        include_augmented: bool = True,
        source_dataset: Optional[Union[str, List[str]]] = None,
        filter_label: Optional[int] = None,
        min_seq_len: int = 1,
        target_dtype: torch.dtype = torch.float32,
        cache_in_ram: bool = False,
        window_sampling: str = "random",
        transform: Optional[Any] = None
    ) -> None:
        """
        Khởi tạo HDF5FeatureDataset.

        Args:
            h5_path: Đường dẫn tới tệp .h5 (bắt buộc).
            manifest_csv: Đường dẫn tệp CSV manifest đi kèm (tùy chọn).
            split: Phân tập dữ liệu: "train", "val", hoặc "all".
            seq_len: Độ dài cố định của chuỗi thời gian mong muốn (ví dụ 120). Nếu None, giữ nguyên độ dài thực tế.
            stride: Bước nhảy khi lấy mẫu thời gian (mặc định: 1).
            include_augmented: Cho phép nạp các bản sao tăng cường (mặc định: True cho train, nên đặt False cho val).
            source_dataset: Lọc theo nguồn dữ liệu (ví dụ: "sust", "uta-rldd").
            filter_label: Lọc theo nhãn cụ thể (0 hoặc 1).
            min_seq_len: Bỏ qua các clip có số khung hình ít hơn ngưỡng này.
            target_dtype: Kiểu dữ liệu PyTorch xuất ra (mặc định: torch.float32).
            cache_in_ram: Nạp trước toàn bộ tensor vào RAM nếu bộ nhớ cho phép (mặc định: False).
            window_sampling: Chiến lược cắt lát thời gian khi T > seq_len: "random", "center", hoặc "start".
            transform: Biến đổi bổ trợ trên tensor nếu cần.
        """
        super().__init__()
        self.h5_path = Path(h5_path)
        if not self.h5_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy tệp HDF5 tại: {self.h5_path}")

        self.manifest_csv = Path(manifest_csv) if manifest_csv is not None else None
        self.split = split.lower().strip()
        if self.split not in ("train", "val", "all"):
            raise ValueError(f"Tham số split '{split}' không hợp lệ. Chọn: 'train', 'val', hoặc 'all'.")

        self.seq_len = seq_len
        self.stride = max(1, int(stride))
        self.include_augmented = include_augmented
        self.min_seq_len = max(1, int(min_seq_len))
        self.target_dtype = target_dtype
        self.cache_in_ram = cache_in_ram
        self.window_sampling = window_sampling.lower().strip()
        self.transform = transform

        # Chuẩn hóa bộ lọc source_dataset
        if isinstance(source_dataset, str):
            self.source_dataset = {source_dataset.lower()}
        elif isinstance(source_dataset, (list, tuple, set)):
            self.source_dataset = {s.lower() for s in source_dataset}
        else:
            self.source_dataset = None

        self.filter_label = filter_label

        # Quản lý file handle HDF5 an toàn cho multi-processing worker
        self._h5_file: Optional[h5py.File] = None
        self._pid: Optional[int] = None

        # Bộ nhớ đệm RAM (nếu cache_in_ram=True)
        self._ram_cache: Dict[int, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # 1. Lập chỉ mục toàn bộ các mẫu thỏa điều kiện
        self.samples: List[H5VideoSample] = self._build_index()

        if len(self.samples) == 0:
            print(f"[!] CẢNH BÁO: Không tìm thấy mẫu video nào thỏa mãn điều kiện lọc (split='{self.split}')!")

        # 2. Nếu bật chế độ nạp trước vào RAM, tải toàn bộ dữ liệu
        if self.cache_in_ram and len(self.samples) > 0:
            self._preload_into_ram()

    def _get_h5_file(self) -> h5py.File:
        """
        Lazy-loader mở tệp HDF5 per-process an toàn cho PyTorch multi-worker DataLoader.
        Tự động nhận diện khi tiến trình con (worker) được sinh ra để tạo file handle độc lập.
        """
        curr_pid = os.getpid()
        if self._h5_file is None or self._pid != curr_pid:
            self._pid = curr_pid
            # Mở ở chế độ Read-Only an toàn, SWMR tối ưu truy cập đồng thời nhiều tiến trình
            try:
                self._h5_file = h5py.File(str(self.h5_path), mode="r", swmr=True, libver="latest")
            except Exception:
                # Fallback sang chế độ đọc thông thường nếu hệ thống không hỗ trợ SWMR
                self._h5_file = h5py.File(str(self.h5_path), mode="r")
        return self._h5_file

    def _build_index(self) -> List[H5VideoSample]:
        """
        Lập chỉ mục danh sách các mẫu video:
        Ưu tiên nạp qua file CSV manifest nếu có, hoặc tự động quét cây HDF5.
        """
        # Nhánh 1: Nạp qua CSV manifest nếu tệp tồn tại và hợp lệ
        if self.manifest_csv is not None and self.manifest_csv.is_file() and self.manifest_csv.stat().st_size > 0:
            try:
                samples = self._build_index_from_csv(self.manifest_csv)
                if len(samples) > 0:
                    return samples
            except Exception as e:
                print(f"[!] Không thể đọc tệp CSV manifest ({e}), chuyển sang quét trực tiếp tệp HDF5...")

        # Nhánh 2: Tự động quét cây phân cấp HDF5
        return self._build_index_from_h5()

    def _build_index_from_csv(self, csv_path: Path) -> List[H5VideoSample]:
        """Lập chỉ mục nhanh chóng từ tệp CSV manifest."""
        samples: List[H5VideoSample] = []
        with open(csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # 1. Chỉ lấy các bản ghi trích xuất thành công
                if row.get("status") != "SUCCESS":
                    continue

                split_val = row.get("split", "").lower().strip()
                if self.split != "all" and split_val != self.split:
                    continue

                # 2. Lọc cờ tăng cường (Augmentation)
                is_aug_str = row.get("is_augmented", "False").lower()
                is_aug = is_aug_str in ("true", "1", "yes")
                if not self.include_augmented and is_aug:
                    continue

                # 3. Lọc nhãn
                label = int(row.get("label", 0))
                if self.filter_label is not None and label != self.filter_label:
                    continue

                # 4. Lọc theo nguồn dữ liệu (Source Dataset)
                src = row.get("source_dataset", "unknown").lower().strip()
                if self.source_dataset is not None and src not in self.source_dataset:
                    continue

                # 5. Lọc theo số khung hình tối thiểu
                num_frames = int(row.get("num_frames", 0))
                if num_frames < self.min_seq_len:
                    continue

                group_path = row.get("h5_group_path", "")
                if not group_path:
                    # Tự tái tạo group_path nếu CSV không có
                    label_name = row.get("label_name", "0_alert" if label == 0 else "1_drowsy")
                    group_path = f"{split_val}/{label_name}/{row['video_id']}"

                sample = H5VideoSample(
                    group_path=group_path,
                    video_id=row.get("video_id", Path(group_path).name),
                    split=split_val,
                    label=label,
                    label_name=row.get("label_name", "0_alert" if label == 0 else "1_drowsy"),
                    seq_len=num_frames,
                    is_augmented=is_aug,
                    source_dataset=src,
                    orig_file=row.get("orig_file", "")
                )
                samples.append(sample)

        samples.sort(key=lambda s: s.video_id)
        return samples

    def _build_index_from_h5(self) -> List[H5VideoSample]:
        """Tự động duyệt cây nhóm HDF5 tìm các video có cờ is_completed == True."""
        samples: List[H5VideoSample] = []
        with h5py.File(str(self.h5_path), mode="r") as h5_temp:
            splits_to_check = [self.split] if self.split != "all" else ["train", "val"]

            for sp in splits_to_check:
                if sp not in h5_temp:
                    continue

                sp_grp = h5_temp[sp]
                for label_name in ("0_alert", "1_drowsy"):
                    if label_name not in sp_grp:
                        continue

                    lbl_grp = sp_grp[label_name]
                    default_label = 0 if "0" in label_name else 1

                    for video_id in lbl_grp.keys():
                        v_grp = lbl_grp[video_id]
                        if not isinstance(v_grp, h5py.Group):
                            continue

                        # Kiểm tra cờ giao dịch nguyên tử is_completed và đủ 3 dataset p3, p4, p5
                        is_comp = v_grp.attrs.get("is_completed", False)
                        if not is_comp or not ("p3" in v_grp and "p4" in v_grp and "p5" in v_grp):
                            continue

                        # Đọc siêu dữ liệu
                        label = int(v_grp.attrs.get("label", default_label))
                        if self.filter_label is not None and label != self.filter_label:
                            continue

                        is_aug = bool(v_grp.attrs.get("is_augmented", False))
                        if not self.include_augmented and is_aug:
                            continue

                        src = str(v_grp.attrs.get("source_dataset", "unknown")).lower().strip()
                        if self.source_dataset is not None and src not in self.source_dataset:
                            continue

                        # Xác định số khung hình từ shape của p3
                        seq_len = int(v_grp.attrs.get("seq_len", v_grp["p3"].shape[0]))
                        if seq_len < self.min_seq_len:
                            continue

                        group_path = f"{sp}/{label_name}/{video_id}"
                        sample = H5VideoSample(
                            group_path=group_path,
                            video_id=video_id,
                            split=sp,
                            label=label,
                            label_name=label_name,
                            seq_len=seq_len,
                            is_augmented=is_aug,
                            source_dataset=src,
                            orig_file=str(v_grp.attrs.get("orig_file", ""))
                        )
                        samples.append(sample)

        samples.sort(key=lambda s: s.video_id)
        return samples

    def _preload_into_ram(self) -> None:
        """Nạp trước toàn bộ các tensor đặc trưng vào bộ nhớ RAM."""
        print(f"[*] Đang nạp trước {len(self.samples)} mẫu vào bộ nhớ RAM...")
        h5_file = self._get_h5_file()
        for idx, sample in enumerate(self.samples):
            grp = h5_file[sample.group_path]
            p3 = torch.from_numpy(grp["p3"][:]).to(dtype=self.target_dtype)
            p4 = torch.from_numpy(grp["p4"][:]).to(dtype=self.target_dtype)
            p5 = torch.from_numpy(grp["p5"][:]).to(dtype=self.target_dtype)
            self._ram_cache[idx] = (p3, p4, p5)
        print(f"[*] Hoàn tất nạp RAM cho tập {self.split.upper()} ({len(self._ram_cache)} mẫu).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Nạp một mẫu video đặc trưng theo chỉ số index.

        Returns:
            features: Tuple gồm (p3, p4, p5)
            label: Tensor scalar kiểu torch.long
            seq_len: Tensor scalar kiểu torch.long chứa độ dài khung hình thực tế
            meta: Dict chứa siêu dữ liệu của mẫu
        """
        sample = self.samples[idx]

        # 1. Đọc dữ liệu từ RAM Cache hoặc trực tiếp từ HDF5
        if self.cache_in_ram and idx in self._ram_cache:
            p3, p4, p5 = self._ram_cache[idx]
            total_t = p3.shape[0]

            # Xử lý temporal slicing trên RAM
            if self.seq_len is not None and total_t > self.seq_len:
                if self.split == "train" and self.window_sampling == "random":
                    t0 = random.randint(0, total_t - self.seq_len)
                elif self.window_sampling == "center" or self.split == "val":
                    t0 = (total_t - self.seq_len) // 2
                else:
                    t0 = 0
                p3 = p3[t0 : t0 + self.seq_len : self.stride]
                p4 = p4[t0 : t0 + self.seq_len : self.stride]
                p5 = p5[t0 : t0 + self.seq_len : self.stride]
            elif self.stride > 1:
                p3 = p3[::self.stride]
                p4 = p4[::self.stride]
                p5 = p5[::self.stride]

        else:
            # Đọc trực tiếp từ tệp HDF5 theo lát cắt chunking tối ưu
            h5_file = self._get_h5_file()
            grp = h5_file[sample.group_path]

            ds_p3 = grp["p3"]
            ds_p4 = grp["p4"]
            ds_p5 = grp["p5"]
            total_t = ds_p3.shape[0]

            # Xác định phạm vi khung hình cần đọc
            if self.seq_len is not None and total_t > self.seq_len:
                if self.split == "train" and self.window_sampling == "random":
                    t0 = random.randint(0, total_t - self.seq_len)
                elif self.window_sampling == "center" or self.split == "val":
                    t0 = (total_t - self.seq_len) // 2
                else:
                    t0 = 0
                t1 = t0 + self.seq_len
                p3_np = ds_p3[t0 : t1 : self.stride]
                p4_np = ds_p4[t0 : t1 : self.stride]
                p5_np = ds_p5[t0 : t1 : self.stride]
            else:
                p3_np = ds_p3[::self.stride]
                p4_np = ds_p4[::self.stride]
                p5_np = ds_p5[::self.stride]

            # Chuyển đổi sang PyTorch Tensor với kiểu dữ liệu mong muốn
            p3 = torch.from_numpy(p3_np).to(dtype=self.target_dtype)
            p4 = torch.from_numpy(p4_np).to(dtype=self.target_dtype)
            p5 = torch.from_numpy(p5_np).to(dtype=self.target_dtype)

        # 2. Áp dụng transform bổ trợ nếu có
        if self.transform is not None:
            p3, p4, p5 = self.transform((p3, p4, p5))

        cur_t = p3.shape[0]
        label_tensor = torch.tensor(sample.label, dtype=torch.long)
        seq_len_tensor = torch.tensor(cur_t, dtype=torch.long)

        meta = {
            "video_id": sample.video_id,
            "split": sample.split,
            "label_name": sample.label_name,
            "orig_seq_len": sample.seq_len,
            "cur_seq_len": cur_t,
            "is_augmented": sample.is_augmented,
            "source_dataset": sample.source_dataset,
            "orig_file": sample.orig_file,
            "group_path": sample.group_path
        }

        return (p3, p4, p5), label_tensor, seq_len_tensor, meta

    def close(self) -> None:
        """Đóng an toàn file descriptor HDF5 nếu đang mở."""
        if self._h5_file is not None:
            try:
                self._h5_file.close()
            except Exception:
                pass
            self._h5_file = None
            self._pid = None

    def __del__(self) -> None:
        self.close()


# ==============================================================================
# 3. HÀM GOM BATCH TÙY BIẾN: collate_h5_features (DYNAMIC ZERO-PADDING)
# ==============================================================================
def collate_h5_features(
    batch: List[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]]
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    """
    Hàm gom batch tùy biến thực hiện Zero-Padding theo trục thời gian về độ dài max trong batch.

    Args:
        batch: Danh sách các phần tử trả về từ Dataset:
               [((p3, p4, p5), label, seq_len, meta), ...]

    Returns:
        features: Tuple gồm 3 tensor 5D:
                  - p3_batch: [B, T_max, 64, 80, 80]
                  - p4_batch: [B, T_max, 128, 40, 40]
                  - p5_batch: [B, T_max, 256, 20, 20]
        labels: Tensor 1D [B] (torch.long)
        seq_lens: Tensor 1D [B] (torch.long) chứa độ dài thực tế trước khi padding
        metas: Danh sách B phần tử metadata
    """
    if len(batch) == 0:
        raise ValueError("Không thể gom batch rỗng.")

    b_size = len(batch)

    # 1. Tìm độ dài thời gian lớn nhất T_max trong batch hiện tại
    seq_lens_list = [item[2].item() for item in batch]
    max_t = max(seq_lens_list)

    # Lấy thông tin kích thước không gian và kiểu dữ liệu từ mẫu đầu tiên
    first_p3, first_p4, first_p5 = batch[0][0]
    dtype = first_p3.dtype
    device = first_p3.device

    # 2. Khởi tạo sẵn batch tensor đệm số 0 (Zero-Padding)
    p3_batch = torch.zeros((b_size, max_t, 64, 80, 80), dtype=dtype, device=device)
    p4_batch = torch.zeros((b_size, max_t, 128, 40, 40), dtype=dtype, device=device)
    p5_batch = torch.zeros((b_size, max_t, 256, 20, 20), dtype=dtype, device=device)

    labels_list: List[torch.Tensor] = []
    metas_list: List[Dict[str, Any]] = []

    # 3. Điền dữ liệu thực tế vào batch tensor
    for i, (features, label, seq_len, meta) in enumerate(batch):
        p3, p4, p5 = features
        t_i = p3.shape[0]

        if t_i > 0:
            p3_batch[i, :t_i] = p3
            p4_batch[i, :t_i] = p4
            p5_batch[i, :t_i] = p5

        labels_list.append(label)
        metas_list.append(meta)

    labels_batch = torch.stack(labels_list, dim=0)
    seq_lens_batch = torch.tensor(seq_lens_list, dtype=torch.long, device=device)

    return (p3_batch, p4_batch, p5_batch), labels_batch, seq_lens_batch, metas_list


# ==============================================================================
# 4. HÀM FACTORY KHỞI TẠO BỘ NẠP: build_h5_dataloaders
# ==============================================================================
def seed_worker(worker_id: int) -> None:
    """Hàm khởi tạo seed độc lập cho từng worker đảm bảo tính lặp lại (reproducibility)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_h5_dataloaders(
    h5_path: Union[str, Path],
    manifest_csv: Optional[Union[str, Path]] = None,
    batch_size: int = 64,
    num_workers: int = 4,
    seq_len: Optional[int] = None,
    stride: int = 1,
    include_augmented_train: bool = True,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    cache_in_ram: bool = False,
    seed: int = 42,
    config: Optional[Any] = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Khởi tạo cặp (train_loader, val_loader) chuẩn hóa từ tệp HDF5.
    Hỗ trợ đồng bộ tự động với đối tượng TrainConfig từ configs/config.py nếu được truyền vào.

    Returns:
        (train_loader, val_loader)
    """
    # Nếu truyền vào đối tượng config, tự động trích xuất các tham số tương ứng
    if config is not None:
        batch_size = getattr(config, "batch_size", batch_size)
        num_workers = getattr(config, "num_workers", num_workers)
        pin_memory = getattr(config, "pin_memory", pin_memory)
        persistent_workers = getattr(config, "persistent_workers", persistent_workers)
        seed = getattr(config, "seed", seed)
        seq_len = getattr(config, "seq_len", seq_len)

    # Thiết lập generator cho DataLoader
    g = torch.Generator()
    g.manual_seed(seed)

    # 1. Khởi tạo Train Dataset & DataLoader
    train_dataset = HDF5FeatureDataset(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        split="train",
        seq_len=seq_len,
        stride=stride,
        include_augmented=include_augmented_train,
        cache_in_ram=cache_in_ram,
        window_sampling="random"
    )

    # 2. Khởi tạo Val Dataset & DataLoader (Luôn tắt Augmentation để chống Data Leakage)
    val_dataset = HDF5FeatureDataset(
        h5_path=h5_path,
        manifest_csv=manifest_csv,
        split="val",
        seq_len=seq_len,
        stride=stride,
        include_augmented=False,
        cache_in_ram=cache_in_ram,
        window_sampling="center"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_h5_features,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=seed_worker,
        generator=g
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_h5_features,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        worker_init_fn=seed_worker
    )

    return train_loader, val_loader


# ==============================================================================
# 5. HÀM TẠO DỮ LIỆU TẠM PHỤC VỤ KIỂM THỬ (MOCK HDF5 GENERATOR)
# ==============================================================================
def create_mock_h5_dataset(h5_path: Union[str, Path], csv_path: Optional[Union[str, Path]] = None) -> None:
    """
    Tạo một tệp HDF5 tạm (và CSV manifest tùy chọn) với đúng 100% định dạng
    sinh ra bởi extract_to_pt.py để phục vụ kiểm thử độc lập.
    """
    h5_path = Path(h5_path)
    h5_path.parent.mkdir(parents=True, exist_ok=True)

    # Danh sách các mẫu mô phỏng: (split, label, label_name, video_id, T, is_aug, aug_seed)
    mock_samples = [
        ("train", 0, "0_alert", "train_0_alert_sust_n_1", 25, False, -1),
        ("train", 0, "0_alert", "train_0_alert_sust_n_1_aug01", 25, True, 101),
        ("train", 1, "1_drowsy", "train_1_drowsy_sust_d_1", 40, False, -1),
        ("val", 0, "0_alert", "val_0_alert_sust_n_2", 30, False, -1),
        ("val", 1, "1_drowsy", "val_1_drowsy_sust_d_2", 15, False, -1),
    ]

    csv_rows = []

    with h5py.File(str(h5_path), "w") as f:
        for split, label, label_name, video_id, T, is_aug, seed in mock_samples:
            group_path = f"{split}/{label_name}/{video_id}"
            grp = f.create_group(group_path)

            # Khởi tạo dữ liệu tensor ngẫu nhiên FP16 chunked
            p3 = (np.random.randn(T, 64, 80, 80) * 0.1).astype(np.float16)
            p4 = (np.random.randn(T, 128, 40, 40) * 0.1).astype(np.float16)
            p5 = (np.random.randn(T, 256, 20, 20) * 0.1).astype(np.float16)

            grp.create_dataset("p3", data=p3, chunks=(1, 64, 80, 80), compression="lzf")
            grp.create_dataset("p4", data=p4, chunks=(1, 128, 40, 40), compression="lzf")
            grp.create_dataset("p5", data=p5, chunks=(1, 256, 20, 20), compression="lzf")

            # Thuộc tính metadata
            grp.attrs["label"] = label
            grp.attrs["label_name"] = label_name
            grp.attrs["seq_len"] = T
            grp.attrs["sample_interval"] = 0.1
            grp.attrs["orig_fps"] = 30.0
            grp.attrs["orig_duration_s"] = T * 0.1
            grp.attrs["is_augmented"] = is_aug
            grp.attrs["aug_seed"] = seed
            grp.attrs["source_dataset"] = "sust"
            grp.attrs["orig_file"] = f"mock_{video_id}.mp4"
            grp.attrs["created_at"] = "2026-10-02 10:00:00"
            grp.attrs["is_completed"] = True

            if csv_path is not None:
                csv_rows.append({
                    "video_id": video_id,
                    "split": split,
                    "label": label,
                    "label_name": label_name,
                    "orig_file": f"mock_{video_id}.mp4",
                    "source_dataset": "sust",
                    "orig_duration_s": f"{T * 0.1:.2f}",
                    "orig_fps": "30.0",
                    "sample_interval": "0.1",
                    "num_frames": str(T),
                    "p3_shape": str(p3.shape),
                    "p4_shape": str(p4.shape),
                    "p5_shape": str(p5.shape),
                    "dtype": "float16",
                    "compression": "lzf",
                    "is_augmented": str(is_aug),
                    "aug_seed": str(seed),
                    "h5_group_path": group_path,
                    "status": "SUCCESS",
                    "error_msg": "",
                    "timestamp": "2026-10-02 10:00:00"
                })

    if csv_path is not None:
        csv_path = Path(csv_path)
        csv_fieldnames = [
            "video_id", "split", "label", "label_name", "orig_file",
            "source_dataset", "orig_duration_s", "orig_fps", "sample_interval",
            "num_frames", "p3_shape", "p4_shape", "p5_shape", "dtype",
            "compression", "is_augmented", "aug_seed", "h5_group_path",
            "status", "error_msg", "timestamp"
        ]
        with open(csv_path, mode="w", encoding="utf-8", newline="") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)


# ==============================================================================
# 6. KHỐI KIỂM THỬ TỰ ĐỘNG CHUYÊN SÂU (SELF-CONTAINED UNIT & INTEGRATION TESTS)
# ==============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("      KIỂM THỬ TỰ ĐỘNG MODULE SRC/DATASET.PY VỚI FILE .H5 TẠM CHUẨN HÓA      ")
    print("=" * 80)

    # Thêm thư mục gốc vào path để nạp src.models và src.loss
    root_dir = Path(__file__).resolve().parent.parent
    if str(root_dir) not in sys.path:
        sys.path.insert(0, str(root_dir))

    from src.models import DeepGRUClassifier
    from src.loss import DrowsinessLoss

    # Tạo thư mục tạm an toàn
    temp_dir = tempfile.mkdtemp(prefix="test_h5_dataset_")
    test_h5 = Path(temp_dir) / "test_features.h5"
    test_csv = Path(temp_dir) / "test_manifest.csv"

    try:
        # 1. Khởi tạo dữ liệu mock HDF5 và CSV
        print("\n[*] [Giai đoạn 1] Đang sinh file .h5 và .csv tạm chuẩn định dạng extract_to_pt.py...")
        create_mock_h5_dataset(test_h5, test_csv)
        print(f"    Tệp H5 tạm: {test_h5} ({test_h5.stat().st_size / 1024:.1f} KB)")
        print(f"    Tệp CSV tạm: {test_csv} ({test_csv.stat().st_size} bytes)")

        # 2. Test Lập chỉ mục kép (Dual Indexing)
        print("\n[*] [Giai đoạn 2] Kiểm thử cơ chế Lập chỉ mục kép (Dual Indexing):")
        # 2.1. Đọc qua CSV
        ds_train_csv = HDF5FeatureDataset(h5_path=test_h5, manifest_csv=test_csv, split="train", include_augmented=True)
        ds_val_csv = HDF5FeatureDataset(h5_path=test_h5, manifest_csv=test_csv, split="val", include_augmented=False)
        print(f"    -> Đọc qua CSV: Train có {len(ds_train_csv)} mẫu, Val có {len(ds_val_csv)} mẫu.")
        assert len(ds_train_csv) == 3, f"Mong đợi 3 mẫu train, nhận được {len(ds_train_csv)}"
        assert len(ds_val_csv) == 2, f"Mong đợi 2 mẫu val, nhận được {len(ds_val_csv)}"

        # 2.2. Đọc trực tiếp từ cây HDF5 (không có file CSV)
        ds_train_h5 = HDF5FeatureDataset(h5_path=test_h5, manifest_csv=None, split="train", include_augmented=True)
        ds_val_h5 = HDF5FeatureDataset(h5_path=test_h5, manifest_csv=None, split="val", include_augmented=False)
        print(f"    -> Quét cây HDF5 trực tiếp: Train có {len(ds_train_h5)} mẫu, Val có {len(ds_val_h5)} mẫu.")
        assert len(ds_train_h5) == 3, f"Mong đợi 3 mẫu train qua H5, nhận được {len(ds_train_h5)}"
        assert len(ds_val_h5) == 2, f"Mong đợi 2 mẫu val qua H5, nhận được {len(ds_val_h5)}"
        print("    [✓] ĐẠT: Cơ chế Lập chỉ mục kép hoạt động chính xác 100%!")

        # 3. Test Lấy mẫu đơn lẻ và Cắt lát thời gian
        print("\n[*] [Giai đoạn 3] Kiểm thử __getitem__ và Temporal Windowing:")
        feat, lbl, slen, meta = ds_train_csv[0]
        p3, p4, p5 = feat
        print(f"    Mẫu 0: video_id={meta['video_id']}, T={slen.item()}, label={lbl.item()}")
        print(f"    Shapes: p3={p3.shape}, p4={p4.shape}, p5={p5.shape}, dtype={p3.dtype}")
        assert p3.shape == (25, 64, 80, 80), f"Sai shape p3: {p3.shape}"
        assert p4.shape == (25, 128, 40, 40), f"Sai shape p4: {p4.shape}"
        assert p5.shape == (25, 256, 20, 20), f"Sai shape p5: {p5.shape}"

        # Thử nghiệm cắt lát seq_len = 20
        ds_window = HDF5FeatureDataset(h5_path=test_h5, split="train", seq_len=20, window_sampling="center")
        feat_w, _, slen_w, _ = ds_window[2]  # Mẫu gốc có T=40
        print(f"    Mẫu 2 (T gốc=40) qua cắt lát cố định seq_len=20: T={slen_w.item()}, shape={feat_w[0].shape}")
        assert slen_w.item() == 20, f"Mong đợi T=20, nhận được {slen_w.item()}"
        print("    [✓] ĐẠT: Nạp tensor đơn lẻ và cắt lát thời gian hoạt động hoàn hảo!")

        # 4. Test Gom batch động (Dynamic Zero-Padding Collate)
        print("\n[*] [Giai đoạn 4] Kiểm thử collate_h5_features với độ dài chuỗi lệch nhau (T1=25, T2=40):")
        sample_batch = [ds_train_csv[0], ds_train_csv[2]]  # Mẫu 0 có T=25, Mẫu 2 có T=40
        (b_p3, b_p4, b_p5), b_lbls, b_lens, b_metas = collate_h5_features(sample_batch)
        print(f"    Batch Tensor Shapes: p3={b_p3.shape}, p4={b_p4.shape}, p5={b_p5.shape}")
        print(f"    Batch Labels: {b_lbls}, Seq Lens: {b_lens}")
        assert b_p3.shape == (2, 40, 64, 80, 80), f"Sai shape b_p3: {b_p3.shape}"
        assert b_lens.tolist() == [25, 40], f"Sai seq_lens: {b_lens}"
        # Xác nhận các frame từ 25 đến 40 của mẫu đầu tiên được pad số 0 tuyệt đối
        pad_sum = b_p3[0, 25:].abs().sum().item()
        assert pad_sum == 0.0, f"Padding rác phát hiện: {pad_sum}"
        print("    [✓] ĐẠT: Gom batch động tự động Zero-Padding về T_max=40 không tì vết!")

        # 5. Test PyTorch DataLoader với Multi-processing (num_workers=2)
        print("\n[*] [Giai đoạn 5] Kiểm thử DataLoader an toàn Multi-processing (num_workers=2):")
        train_loader, val_loader = build_h5_dataloaders(
            h5_path=test_h5,
            manifest_csv=test_csv,
            batch_size=2,
            num_workers=2,
            pin_memory=False
        )
        total_train_batches = 0
        for batch_data in train_loader:
            features, labels, seq_lens, metas = batch_data
            total_train_batches += 1
            print(f"    -> Đã nạp batch {total_train_batches}: p3={features[0].shape}, seq_lens={seq_lens.tolist()}")
        print(f"    -> Hoàn thành nạp {total_train_batches} batches qua 2 worker processes song song.")
        print("    [✓] ĐẠT: Multi-processing an toàn tuyệt đối, không deadlock, không lỗi file handle!")

        # 6. Test Tích hợp Toàn diện End-to-End với DeepGRUClassifier và DrowsinessLoss
        print("\n[*] [Giai đoạn 6] Kiểm thử End-to-End Forward & Backward Pass với DeepGRUClassifier:")
        model = DeepGRUClassifier(
            input_dim=128,
            hidden_dim=96,
            num_layers=2,
            num_classes=2,
            spatial_in_channels=(64, 128, 256),
            fusion="concat"
        )
        loss_fn = DrowsinessLoss()

        # Lấy một batch từ dataloader
        (p3_in, p4_in, p5_in), targets, lens_in, _ = next(iter(train_loader))

        # Forward pass
        logits = model((p3_in, p4_in, p5_in), seq_lens=lens_in)
        print(f"    Model Logits Shape: {logits.shape} (Mong đợi: [2, 2])")
        assert logits.shape == (p3_in.shape[0], 2), f"Sai shape logits: {logits.shape}"

        # Loss calculation
        loss = loss_fn(logits, targets)
        print(f"    Computed Loss: {loss.item():.4f}")
        assert not torch.isnan(loss) and not torch.isinf(loss), "Loss bị NaN hoặc Inf!"

        # Backward pass
        loss.backward()
        print("    Backward Pass: Đã tính toán gradient thành công cho toàn bộ mạng!")
        # Kiểm tra gradient tại tầng đầu tiên của spatial_adapter
        conv1_grad = model.spatial_adapter.pyramid_stage1[0].weight.grad
        assert conv1_grad is not None, "Không tìm thấy gradient tại spatial_adapter!"
        print(f"    Gradient Stage1 Conv2d Norm: {conv1_grad.norm().item():.6f}")
        print("    [✓] ĐẠT: Tích hợp hoàn hảo với DeepGRUClassifier và DrowsinessLoss!")

        print("\n" + "=" * 80)
        print("    >>> TẤT CẢ CÁC BÀI KIỂM THỬ ĐÃ ĐẠT 100% TIÊU CHÍ CHẤT LƯỢNG! <<<    ")
        print("=" * 80)

    finally:
        # Đóng an toàn tất cả các file handle đang mở để tránh lock file trên Windows
        try:
            if 'train_loader' in locals() and hasattr(train_loader.dataset, 'close'):
                train_loader.dataset.close()
            if 'val_loader' in locals() and hasattr(val_loader.dataset, 'close'):
                val_loader.dataset.close()
            if 'ds_train_csv' in locals():
                ds_train_csv.close()
            if 'ds_val_csv' in locals():
                ds_val_csv.close()
            if 'ds_train_h5' in locals():
                ds_train_h5.close()
            if 'ds_val_h5' in locals():
                ds_val_h5.close()
            if 'ds_window' in locals():
                ds_window.close()
        except Exception:
            pass

        # Dọn dẹp an toàn các file và thư mục tạm
        if Path(temp_dir).exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"\n[*] Đã dọn dẹp sạch sẽ thư mục tạm: {temp_dir}")

