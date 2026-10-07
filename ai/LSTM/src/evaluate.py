#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Đánh Giá & Trực Quan Hóa Mô Hình ConvGRUClassifier (Driver Guardian AI)
Phiên bản: 2.0 (Chuẩn hóa cấu hình tập trung EvalConfig - Không dùng CLI)

Mô tả chức năng:
  1. Nạp checkpoint của mô hình Spatio-Temporal ConvGRUClassifier (best.pt, last.pt hoặc epoch_*.pt).
  2. Khởi tạo Validation DataLoader từ dữ liệu video thô hoặc tập kiểm định đã tách.
  3. Trích xuất đặc trưng NMSFreeDetector PAFPN qua GPU Mini-Chunking (chống tràn VRAM).
  4. Tính toán toàn diện hệ thống chỉ số: Loss, Accuracy, Macro/Per-class Precision, Recall, F1, Latency.
  5. Trực quan hóa chuyên nghiệp:
     - Ma trận nhầm lẫn (Confusion Matrix): Heatmap chi tiết số mẫu và tỷ lệ chuẩn hóa (%) từng lớp.
     - Biểu đồ so sánh Loss, Accuracy, Recall, F1 qua mỗi giai đoạn/epoch từ nhật ký huấn luyện.
     - Đồ thị đường cong ROC-AUC và Precision-Recall (PR).
  6. Xuất báo cáo tóm tắt ra tệp JSON và hiển thị trực quan trên giao diện dòng lệnh.

Người dùng chỉ cần điều chỉnh các hằng số cấu hình tại lớp `EvalConfig` bên dưới
và thực thi trực tiếp: python src/evaluate.py
"""

import os
import sys
import time
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List, Union, Sequence

# Đảm bảo console Windows hỗ trợ hiển thị tiếng Việt UTF-8
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Thiết lập đường dẫn thư mục gốc
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Tối ưu môi trường DLL & OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Chế độ headless an toàn cho máy chủ / nền tảng không có GUI
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import các thành phần nội bộ từ package src
from configs.config import load_config, TrainConfig
from src.models import ConvGRUClassifier
from src.dataset import build_raw_video_dataloaders, ChunkedBackboneNeckExtractor


# ==============================================================================
# 1. NƠI TẬP TRUNG TOÀN BỘ HẰNG SỐ & THAM SỐ CẤU HÌNH (EVALCONFIG)
# ==============================================================================
@dataclass
class EvalConfig:
    """
    NƠI TẬP TRUNG TOÀN BỘ HẰNG SỐ VÀ THAM SỐ ĐÁNH GIÁ MÔ HÌNH CONVGRU.
    Chỉnh sửa trực tiếp các tham số dưới đây theo nhu cầu trước khi chạy script.
    """
    # ---- 1. ĐƯỜNG DẪN TỆP & THƯ MỤC ----
    # Đường dẫn file checkpoint cần đánh giá (có thể dùng 'best', 'last', hoặc đường dẫn cụ thể)
    checkpoint_path: str = "checkpoints/experiments/deepgru_raw_nmsfree/epoch_5.pt"
    # Đường dẫn file cấu hình hệ thống
    config_yaml_path: str = "configs/config.yaml"
    # Đường dẫn file lịch sử huấn luyện qua các epoch
    history_csv_path: str = "logs/training_history.csv"
    # Thư mục lưu trữ hình ảnh biểu đồ và báo cáo JSON
    output_dir: str = "logs/evaluation"

    # ---- 2. PHẦN CỨNG & DATALOADER ----
    # Thiết bị tính toán ('cuda' nếu có GPU NVIDIA, tự động fallback sang 'cpu')
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Batch size khi chạy validation
    batch_size: int = 4
    # Số tiến trình worker tải dữ liệu (0 là an toàn nhất trên Windows)
    num_workers: int = 0
    # Kích thước chunk chia nhỏ khung hình khi trích xuất đặc trưng Backbone (chống tràn VRAM)
    chunk_size: int = 32
    # Bật chế độ Mix Precision (FP16) khi chạy trên GPU
    amp: bool = True
    # Giới hạn số batch đánh giá (None: toàn bộ tập val; đặt int > 0 nếu muốn kiểm tra nhanh)
    max_eval_batches: Optional[int] = None

    # ---- 3. TÙY CHỌN TRỰC QUAN HÓA & XUẤT BÁO CÁO ----
    # Vẽ và lưu biểu đồ Ma trận nhầm lẫn (Confusion Matrix)
    plot_confusion_matrix: bool = True
    # Vẽ và lưu biểu đồ so sánh Loss, Acc, Recall qua mỗi giai đoạn từ history CSV
    plot_stage_metrics: bool = True
    # Vẽ và lưu đồ thị đường cong ROC-AUC và Precision-Recall
    plot_roc_pr: bool = True
    # Xuất tệp báo cáo tổng hợp JSON (evaluation_summary.json)
    save_json_summary: bool = True
    # Tùy chọn duyệt và đánh giá toàn bộ checkpoint epoch_*.pt có trong thư mục
    eval_all_checkpoints: bool = False

    # ---- 4. THÔNG TIN NHÃN LỚP ----
    # Tên hiển thị các lớp phân loại
    class_names: Tuple[str, str] = ("Alert (Tỉnh táo)", "Drowsy (Buồn ngủ)")
    # Seed ngẫu nhiên để tái lập kết quả
    seed: int = 42


# ==============================================================================
# 2. TIỆN ÍCH HỆ THỐNG & TÁI LẬP (REPRODUCIBILITY)
# ==============================================================================
def seed_everything(seed: int = 42) -> None:
    """Cố định seed ngẫu nhiên cho toàn bộ hệ thống để đảm bảo tính tái lập."""
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logger(name: str = "EvaluateConvGRU") -> logging.Logger:
    """Khởi tạo logger ghi thông tin ra màn hình với định dạng chuẩn."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = setup_logger()


def resolve_path(path_str: Union[str, Path], base_dir: Optional[Path] = None) -> Path:
    """Chuyển đổi đường dẫn linh hoạt tương thích đa nền tảng."""
    p = Path(path_str)
    if p.is_absolute():
        return p
    if base_dir is None:
        base_dir = PROJECT_ROOT
    return (base_dir / p).resolve()


# ==============================================================================
# 3. BỘ NẠP CHECKPOINT VÀ MÔ HÌNH (MODEL & CHECKPOINT LOADER)
# ==============================================================================
def resolve_checkpoint_file(checkpoint_path: Union[str, Path], ckpt_dir: Optional[Path] = None) -> Path:
    """Phân giải bí danh checkpoint ('best', 'last') hoặc tìm file trong thư mục mặc định."""
    p_str = str(checkpoint_path).strip()
    alias = p_str.lower()

    if ckpt_dir is None:
        ckpt_dir = PROJECT_ROOT / "checkpoints" / "experiments" / "deepgru_raw_nmsfree"

    if alias in ("best", "best.pt"):
        target = ckpt_dir / "best.pt"
        if target.exists():
            return target
    elif alias in ("last", "last.pt"):
        target = ckpt_dir / "last.pt"
        if target.exists():
            return target

    # Kiểm tra đường dẫn trực tiếp
    p = Path(checkpoint_path)
    if p.is_absolute() and p.exists():
        return p
    
    cand = (PROJECT_ROOT / p).resolve()
    if cand.exists():
        return cand
        
    cand_in_dir = (ckpt_dir / p.name).resolve()
    if cand_in_dir.exists():
        return cand_in_dir

    raise FileNotFoundError(
        f"[LỖI] Không tìm thấy file checkpoint: '{checkpoint_path}'. "
        f"Đã tìm kiếm tại: {cand} và {cand_in_dir}"
    )


def load_eval_model(
    checkpoint_path: Union[str, Path],
    device: torch.device,
    fallback_config: Optional[TrainConfig] = None
) -> Tuple[ConvGRUClassifier, Dict[str, Any], Path]:
    """
    Nạp mô hình ConvGRUClassifier từ file checkpoint .pt an toàn.

    Args:
        checkpoint_path: Đường dẫn tới checkpoint hoặc bí danh ('best', 'last').
        device: Thiết bị tính toán (CUDA/CPU).
        fallback_config: Cấu hình mặc định nếu checkpoint không lưu config.

    Returns:
        Tuple gồm (mô hình đã nạp weights, dict nội dung checkpoint, đường dẫn thực tế).
    """
    ckpt_file = resolve_checkpoint_file(checkpoint_path)
    logger.info(f"[*] Đang nạp checkpoint từ: {ckpt_file}")

    ckpt_data = torch.load(str(ckpt_file), map_location=device)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data.get("state_dict", ckpt_data))
    saved_cfg = ckpt_data.get("config", {})

    # Tự động suy luận kích thước từ weights nếu có
    weight_key = "convgru.cells.0.conv_gates.weight"
    if weight_key in state_dict:
        w_shape = state_dict[weight_key].shape  # [2 * H, in + H, K, K]
        out_c, in_total, _, _ = w_shape
        hid_dim = out_c // 2
        in_dim = in_total - hid_dim
    else:
        in_dim = saved_cfg.get("input_dim", getattr(fallback_config, "input_dim", 64))
        hid_dim = saved_cfg.get("hidden_dim", getattr(fallback_config, "hidden_dim", 64))

    # Tự động tính số tầng ConvGRU
    layer_keys = {
        int(k.split(".")[2])
        for k in state_dict
        if k.startswith("convgru.cells.") and k.split(".")[2].isdigit()
    }
    num_layers = len(layer_keys) if layer_keys else saved_cfg.get("num_layers", 2)
    num_classes = int(saved_cfg.get("num_classes", getattr(fallback_config, "num_classes", 2)))
    spatial_in = saved_cfg.get("cnn_neck_channels", getattr(fallback_config, "cnn_neck_channels", (64, 128, 256)))

    logger.info(
        f"[*] Kiến trúc mô hình: in_dim={in_dim}, hidden_dim={hid_dim}, "
        f"num_layers={num_layers}, num_classes={num_classes}"
    )

    model = ConvGRUClassifier(
        input_dim=in_dim,
        hidden_dim=hid_dim,
        num_layers=num_layers,
        num_classes=num_classes,
        spatial_in_channels=tuple(spatial_in)
    ).to(device)

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    logger.info(f"[✓] Đã nạp thành công trọng số mô hình ConvGRUClassifier ({device.type.upper()}).")
    return model, ckpt_data, ckpt_file


def load_backbone_extractor(
    config: TrainConfig,
    device: torch.device,
    chunk_size: int = 32,
    use_fp16: bool = True
) -> Optional[ChunkedBackboneNeckExtractor]:
    """Khởi tạo bộ trích xuất đặc trưng NMSFreeDetector PAFPN."""
    ckpt_path = Path(config.backbone_neck_checkpoint)
    if not ckpt_path.exists():
        logger.warning(f"[CẢNH BÁO] Không tìm thấy backbone checkpoint tại: {ckpt_path}")
        return None

    img_size = getattr(config, "image_size", (640, 640))[0]
    logger.info(f"[*] Đang khởi tạo ChunkedBackboneNeckExtractor (chunk_size={chunk_size}, img_size={img_size})...")
    extractor = ChunkedBackboneNeckExtractor(
        checkpoint_path=str(ckpt_path),
        device=str(device),
        chunk_size=chunk_size,
        use_fp16=(use_fp16 and device.type == "cuda"),
        img_size=img_size
    ).to(device)
    extractor.eval()
    for param in extractor.parameters():
        param.requires_grad = False
    return extractor


# ==============================================================================
# 4. BỘ NẠP DỮ LIỆU KIỂM ĐỊNH (VALIDATION DATALOADER)
# ==============================================================================
def build_val_dataloader(
    config: TrainConfig,
    batch_size: int = 4,
    num_workers: int = 0
) -> Optional[DataLoader]:
    """Khởi tạo Validation DataLoader độc lập dựa trên cấu hình dự án."""
    ds_dir = Path(config.dataset_dir)
    if not ds_dir.exists():
        logger.warning(f"[CẢNH BÁO] Thư mục dataset không tồn tại: {ds_dir}. Không thể chạy validation trực tiếp.")
        return None

    img_size = getattr(config, "image_size", (640, 640))[0]
    min_frames = getattr(config, "min_frames", 10)

    logger.info(f"[*] Khởi tạo Validation DataLoader từ '{ds_dir}' (batch_size={batch_size})...")
    try:
        from src.dataset import RawVideoFramesDataset, collate_video_frames, _seed_worker

        val_dataset = RawVideoFramesDataset(
            dataset_dir=str(ds_dir),
            manifest_file=config.manifest_file,
            split="val",
            sample_interval=config.sample_interval,
            seq_len=config.seq_len,
            img_size=img_size,
            min_frames=min_frames,
            augmenter=None,
            window_sampling="center"
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            collate_fn=collate_video_frames,
            worker_init_fn=_seed_worker
        )
        logger.info(f"[✓] Đã tạo thành công Validation DataLoader: {len(val_dataset)} video clips.")
        return val_loader
    except Exception as e:
        logger.error(f"[LỖI] Không thể khởi tạo Validation DataLoader: {e}")
        return None


# ==============================================================================
# 5. ĐỘNG CƠ SUY LUẬN & TÍNH TOÁN METRIC (EVALUATION ENGINE)
# ==============================================================================
def compute_classification_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: Optional[np.ndarray] = None,
    num_classes: int = 2
) -> Dict[str, Any]:
    """Tính toán đầy đủ hệ thống chỉ số phân loại nhị phân / đa lớp."""
    if len(targets) == 0 or len(preds) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int).tolist(),
        }

    acc = float(accuracy_score(targets, preds))
    prec_macro = float(precision_score(targets, preds, average="macro", zero_division=0))
    rec_macro = float(recall_score(targets, preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(targets, preds, average="macro", zero_division=0))

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    # Chỉ số chi tiết từng lớp
    per_class_rec = recall_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_prec = precision_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)

    res: Dict[str, Any] = {
        "total_samples": int(len(targets)),
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "alert": {
                "precision": float(per_class_prec[0]) if len(per_class_prec) > 0 else 0.0,
                "recall": float(per_class_rec[0]) if len(per_class_rec) > 0 else 0.0,
                "f1": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
                "support": int(np.sum(targets == 0)),
            },
            "drowsy": {
                "precision": float(per_class_prec[1]) if len(per_class_prec) > 1 else 0.0,
                "recall": float(per_class_rec[1]) if len(per_class_rec) > 1 else 0.0,
                "f1": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
                "support": int(np.sum(targets == 1)),
            }
        }
    }

    # Tính ROC-AUC nếu có phân phối xác suất
    if probs is not None and num_classes == 2 and len(np.unique(targets)) > 1:
        try:
            drowsy_probs = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
            fpr, tpr, _ = roc_curve(targets, drowsy_probs)
            roc_auc_val = float(auc(fpr, tpr))
            res["roc_auc"] = roc_auc_val
        except Exception:
            res["roc_auc"] = None
    else:
        res["roc_auc"] = None

    return res


@torch.no_grad()
def run_evaluation_loop(
    model: ConvGRUClassifier,
    extractor: Optional[ChunkedBackboneNeckExtractor],
    val_loader: DataLoader,
    device: torch.device,
    amp: bool = True,
    max_batches: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Thực hiện suy luận toàn diện trên tập Validation.

    Returns:
        Tuple: (all_targets, all_preds, all_probs, avg_loss, avg_latency_ms)
    """
    model.eval()
    if extractor is not None:
        extractor.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[np.ndarray] = []
    batch_latencies: List[float] = []
    valid_batches = 0

    total_samples_processed = 0
    total_time_inference = 0.0

    pbar_total = len(val_loader) if max_batches is None else min(len(val_loader), max_batches)
    logger.info(f"[*] Bắt đầu suy luận trên {pbar_total} batches kiểm định...")

    for batch_idx, batch in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        frames_t, labels, seq_lens, _ = batch
        if frames_t.size(0) == 0:
            continue

        b_size = frames_t.size(0)
        frames_t = frames_t.to(device)
        labels = labels.to(device)
        seq_lens = seq_lens.to(device)

        start_t = time.perf_counter()

        try:
            autocast_enabled = (amp and device.type == "cuda")
            with torch.amp.autocast(device_type=device.type, enabled=autocast_enabled):
                if extractor is not None:
                    p3, p4, p5 = extractor(frames_t)
                    logits = model((p3, p4, p5), seq_lens=seq_lens)
                else:
                    logits = model(frames_t, seq_lens=seq_lens)

                loss = criterion(logits, labels)

            # Đồng bộ GPU để đo chính xác thời gian
            if device.type == "cuda":
                torch.cuda.synchronize()

            latency_ms = (time.perf_counter() - start_t) * 1000.0 / max(b_size, 1)
            batch_latencies.append(latency_ms)

            # Tính xác suất và nhãn dự đoán
            probs = F.softmax(logits, dim=-1)
            preds = torch.argmax(probs, dim=-1)

            total_loss += loss.item()
            valid_batches += 1
            total_samples_processed += b_size

            all_targets.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy())

            if (batch_idx + 1) % max(1, pbar_total // 5) == 0 or (batch_idx + 1) == pbar_total:
                logger.info(
                    f"  -> Tiến độ: [{batch_idx + 1}/{pbar_total}] | "
                    f"Loss: {loss.item():.4f} | Latency: {latency_ms:.2f}ms/clip"
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"[!] CẢNH BÁO OOM tại batch {batch_idx}. Đang xả cache và bỏ qua...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise e
        finally:
            # Giải phóng tensor trung gian
            del frames_t, labels, seq_lens

    avg_loss = total_loss / max(valid_batches, 1)
    avg_latency = float(np.mean(batch_latencies)) if batch_latencies else 0.0

    logger.info(f"[✓] Hoàn tất suy luận: {total_samples_processed} mẫu | Loss: {avg_loss:.4f} | Latency: {avg_latency:.2f}ms/clip")

    return (
        np.array(all_targets, dtype=int),
        np.array(all_preds, dtype=int),
        np.array(all_probs, dtype=np.float32),
        float(avg_loss),
        float(avg_latency)
    )


# ==============================================================================
# 6. ĐỘNG CƠ TRỰC QUAN HÓA BIỂU ĐỒ (VISUALIZATION ENGINE)
# ==============================================================================
def set_plot_style() -> None:
    """Thiết lập cấu hình giao diện đồ thị thẩm mỹ cao và tương thích font chữ."""
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Segoe UI", "Tahoma"]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", palette="muted")


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Sequence[str],
    save_path: Path,
    metrics: Optional[Dict[str, Any]] = None,
    title: str = "Ma Trận Nhầm Lẫn - ConvGRUClassifier"
) -> None:
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap trực quan chuyên nghiệp.
    Hiển thị cả số lượng mẫu tuyệt đối và tỷ lệ phần trăm chuẩn hóa (%) theo hàng.
    """
    set_plot_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm_counts = np.array(cm, dtype=int)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm_counts.astype(float), row_sums, out=np.zeros_like(cm_counts, dtype=float), where=row_sums != 0)

    # Chuẩn bị chuỗi nhãn chú thích cho từng ô (Số lượng + Tỷ lệ %)
    annot_matrix = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = cm_counts[i, j]
            pct = cm_norm[i, j] * 100.0
            annot_matrix[i, j] = f"{count}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

    # Heatmap với bảng màu Blues sang trọng
    sns.heatmap(
        cm_norm,
        annot=annot_matrix,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0.0,
        vmax=1.0,
        annot_kws={"fontsize": 13, "fontweight": "bold"},
        linewidths=1.5,
        linecolor="white",
        ax=ax
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted Label)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=12, fontweight="bold", labelpad=10)

    # Hộp tóm tắt các chỉ số cốt lõi ở góc dưới
    if metrics is not None:
        acc = metrics.get("accuracy", 0.0) * 100.0
        rec_drowsy = metrics.get("per_class", {}).get("drowsy", {}).get("recall", 0.0) * 100.0
        rec_alert = metrics.get("per_class", {}).get("alert", {}).get("recall", 0.0) * 100.0
        f1_macro = metrics.get("macro_f1", 0.0) * 100.0
        total = metrics.get("total_samples", int(np.sum(cm_counts)))

        summary_box = (
            f"Tổng số mẫu: {total}\n"
            f"Độ chính xác (Acc): {acc:.2f}%\n"
            f"Độ nhạy Buồn ngủ (Drowsy Recall): {rec_drowsy:.2f}%\n"
            f"Độ nhạy Tỉnh táo (Alert Recall): {rec_alert:.2f}%\n"
            f"F1-Score (Macro): {f1_macro:.2f}%"
        )
        plt.figtext(
            0.5, -0.05, summary_box,
            wrap=True, horizontalalignment="center", fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8", edgecolor="#b0bec5", alpha=0.9)
        )

    plt.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"[✓] Đã lưu đồ thị Ma trận nhầm lẫn tại: {save_path}")


def plot_stage_metrics(
    history_csv_path: Union[str, Path],
    save_path: Path,
    best_epoch: Optional[int] = None
) -> bool:
    """
    Vẽ lưới biểu đồ 2x2 so sánh chi tiết: Loss, Accuracy, Recall, F1
    qua mỗi giai đoạn/epoch huấn luyện (Train vs Val) từ training_history.csv.
    """
    set_plot_style()
    csv_file = Path(history_csv_path)
    if not csv_file.exists():
        logger.warning(f"[CẢNH BÁO] Không tìm thấy tệp lịch sử huấn luyện: {csv_file}")
        return False

    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        logger.error(f"[LỖI] Không thể đọc tệp CSV {csv_file}: {e}")
        return False

    # Lọc bỏ các dòng test dry-run (nơi cả train_loss và val_loss đều bằng 0)
    if "train_loss" in df.columns and "val_loss" in df.columns:
        valid_mask = (df["train_loss"] > 0) | (df["val_loss"] > 0)
        df_valid = df[valid_mask].copy()
        if len(df_valid) > 0:
            df = df_valid

    # Nếu có nhiều lượt huấn luyện trùng epoch, giữ lại bản ghi sau cùng của mỗi epoch
    if "epoch" in df.columns:
        df = df.drop_duplicates(subset=["epoch"], keep="last").sort_values("epoch").reset_index(drop=True)

    if len(df) == 0:
        logger.warning("[CẢNH BÁO] Tệp CSV không có dữ liệu epoch hợp lệ để vẽ biểu đồ.")
        return False

    epochs = df["epoch"].values
    train_loss = df["train_loss"].values if "train_loss" in df.columns else None
    val_loss = df["val_loss"].values if "val_loss" in df.columns else None

    train_acc = df["train_acc"].values if "train_acc" in df.columns else None
    val_acc = df["val_acc"].values if "val_acc" in df.columns else None

    train_recall = df["train_recall"].values if "train_recall" in df.columns else None
    val_recall = df["val_recall"].values if "val_recall" in df.columns else None

    train_f1 = df["train_f1"].values if "train_f1" in df.columns else None
    val_f1 = df["val_f1"].values if "val_f1" in df.columns else None

    # Tự động xác định Best Epoch nếu chưa chỉ định (dựa trên val_f1 cao nhất)
    if best_epoch is None and val_f1 is not None and len(val_f1) > 0:
        best_idx = int(np.argmax(val_f1))
        best_epoch = int(epochs[best_idx])

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # 1. Biểu đồ LOSS qua mỗi giai đoạn
    ax = axes[0, 0]
    if train_loss is not None:
        ax.plot(epochs, train_loss, marker="o", linewidth=2.2, label="Train Loss", color="#1976D2")
    if val_loss is not None:
        ax.plot(epochs, val_loss, marker="s", linewidth=2.2, label="Validation Loss", color="#D32F2F")
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(x=best_epoch, color="#E65100", linestyle="--", linewidth=1.8, label=f"Best Epoch ({best_epoch})")
    ax.set_title("1. So Sánh Loss Qua Các Giai Đoạn (Loss Curve)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch / Giai đoạn", fontsize=11)
    ax.set_ylabel("Mất Mát (Loss)", fontsize=11)
    ax.legend(frameon=True, loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 2. Biểu đồ ACCURACY qua mỗi giai đoạn
    ax = axes[0, 1]
    if train_acc is not None:
        ax.plot(epochs, train_acc * 100.0, marker="o", linewidth=2.2, label="Train Accuracy", color="#388E3C")
    if val_acc is not None:
        ax.plot(epochs, val_acc * 100.0, marker="s", linewidth=2.2, label="Validation Accuracy", color="#F57C00")
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(x=best_epoch, color="#E65100", linestyle="--", linewidth=1.8, label=f"Best Epoch ({best_epoch})")
    ax.set_title("2. So Sánh Độ Chính Xác (Accuracy Curve)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch / Giai đoạn", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 3. Biểu đồ RECALL qua mỗi giai đoạn (ĐIỂM NHẤN CỐT LÕI)
    ax = axes[1, 0]
    if train_recall is not None:
        ax.plot(epochs, train_recall * 100.0, marker="o", linewidth=2.2, label="Train Recall", color="#7B1FA2")
    if val_recall is not None:
        ax.plot(epochs, val_recall * 100.0, marker="s", linewidth=2.2, label="Validation Recall", color="#C2185B")
        if best_epoch is not None and best_epoch in epochs:
            best_idx = np.where(epochs == best_epoch)[0][0]
            ax.scatter([best_epoch], [val_recall[best_idx] * 100.0], color="#C2185B", s=130, zorder=5, marker="*")
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(x=best_epoch, color="#E65100", linestyle="--", linewidth=1.8, label=f"Best Epoch ({best_epoch})")
    ax.set_title("3. So Sánh Độ Nhạy (Recall Curve - Macro)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch / Giai đoạn", fontsize=11)
    ax.set_ylabel("Recall (%)", fontsize=11)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    # 4. Biểu đồ F1-SCORE & TIẾN TRÌNH qua mỗi giai đoạn
    ax = axes[1, 1]
    if train_f1 is not None:
        ax.plot(epochs, train_f1 * 100.0, marker="o", linewidth=2.2, label="Train F1-Score", color="#00796B")
    if val_f1 is not None:
        ax.plot(epochs, val_f1 * 100.0, marker="s", linewidth=2.2, label="Validation F1-Score", color="#E64A19")
        if best_epoch is not None and best_epoch in epochs:
            best_idx = np.where(epochs == best_epoch)[0][0]
            ax.scatter([best_epoch], [val_f1[best_idx] * 100.0], color="#E64A19", s=130, zorder=5, marker="*")
    if best_epoch is not None and best_epoch in epochs:
        ax.axvline(x=best_epoch, color="#E65100", linestyle="--", linewidth=1.8, label=f"Best Epoch ({best_epoch})")
    ax.set_title("4. So Sánh F1-Score Qua Các Giai Đoạn", fontsize=13, fontweight="bold")
    ax.set_xlabel("Epoch / Giai đoạn", fontsize=11)
    ax.set_ylabel("F1-Score (%)", fontsize=11)
    ax.legend(frameon=True, loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("TIẾN TRÌNH ĐÁNH GIÁ CHỈ SỐ HUẤN LUYỆN QUA TỪNG GIAI ĐOẠN (CONVGRU)", fontsize=16, fontweight="bold", y=0.995)
    plt.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"[✓] Đã lưu biểu đồ so sánh các giai đoạn tại: {save_path}")
    return True


def plot_roc_pr_curves(
    targets: np.ndarray,
    probs: np.ndarray,
    save_path: Path
) -> None:
    """Vẽ đồ thị đường cong ROC (Receiver Operating Characteristic) và Precision-Recall."""
    set_plot_style()
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return

    drowsy_probs = probs[:, 1] if probs.ndim == 2 and probs.shape[1] > 1 else probs.ravel()
    fpr, tpr, _ = roc_curve(targets, drowsy_probs)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(targets, drowsy_probs)
    avg_prec = average_precision_score(targets, drowsy_probs)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Đồ thị ROC
    ax1.plot(fpr, tpr, color="#E65100", lw=2.5, label=f"ConvGRU ROC (AUC = {roc_auc_val:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Classifier")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("Đường Cong ROC (Receiver Operating Characteristic)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Đồ thị Precision-Recall
    ax2.plot(recall, precision, color="#00897B", lw=2.5, label=f"ConvGRU PR (AP = {avg_prec:.4f})")
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    ax2.set_xlabel("Recall (Độ Nhạy)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Precision (Độ Chuẩn Xác)", fontsize=11, fontweight="bold")
    ax2.set_title("Đường Cong Precision - Recall (PR Curve)", fontsize=12, fontweight="bold")
    ax2.legend(loc="lower left", frameon=True)
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"[✓] Đã lưu đồ thị ROC & PR Curves tại: {save_path}")


# ==============================================================================
# 7. BỘ XUẤT BÁO CÁO & GIAO DIỆN CONSOLE (REPORT EXPORTER)
# ==============================================================================
def save_evaluation_summary(summary_data: Dict[str, Any], save_path: Path) -> None:
    """Lưu trữ báo cáo tổng hợp đánh giá ra file định dạng JSON."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    logger.info(f"[✓] Đã lưu báo cáo đánh giá JSON tại: {save_path}")


def print_evaluation_summary(metrics: Dict[str, Any], class_names: Sequence[str]) -> None:
    """Hiển thị bảng tổng kết các chỉ số đo đạc trên giao diện dòng lệnh."""
    print("\n" + "=" * 78)
    print("      KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH CONVGRUCLASSIFIER (DRIVER GUARDIAN AI)     ")
    print("=" * 78)

    acc = metrics.get("accuracy", 0.0) * 100.0
    prec = metrics.get("macro_precision", 0.0) * 100.0
    rec = metrics.get("macro_recall", 0.0) * 100.0
    f1 = metrics.get("macro_f1", 0.0) * 100.0
    loss = metrics.get("loss", 0.0)
    latency = metrics.get("latency_ms_per_clip", 0.0)
    samples = metrics.get("total_samples", 0)

    print(f" [*] Tổng số mẫu kiểm định (Validation Samples): {samples} video clips")
    print(f" [*] Mất mát trung bình (Validation Loss)       : {loss:.4f}")
    print(f" [*] Độ chính xác tổng thể (Accuracy)          : {acc:.2f}%")
    print(f" [*] Độ chuẩn xác vĩ mô (Macro Precision)      : {prec:.2f}%")
    print(f" [*] Độ nhạy vĩ mô (Macro Recall)             : {rec:.2f}%")
    print(f" [*] Chỉ số F1-Score vĩ mô (Macro F1)          : {f1:.2f}%")
    if metrics.get("roc_auc") is not None:
        print(f" [*] Diện tích dưới đường ROC (ROC-AUC)        : {metrics['roc_auc']:.4f}")
    print(f" [*] Độ trễ suy luận trung bình (Latency)      : {latency:.2f} ms/clip")

    print("-" * 78)
    print(f" {'Lớp':<22} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<12} | {'Mẫu':<6}")
    print("-" * 78)

    per_cls = metrics.get("per_class", {})
    alert_info = per_cls.get("alert", {})
    drowsy_info = per_cls.get("drowsy", {})

    print(
        f" {class_names[0]:<22} | "
        f"{alert_info.get('precision', 0.0) * 100.0:>10.2f}% | "
        f"{alert_info.get('recall', 0.0) * 100.0:>10.2f}% | "
        f"{alert_info.get('f1', 0.0) * 100.0:>10.2f}% | "
        f"{alert_info.get('support', 0):>6d}"
    )
    print(
        f" {class_names[1]:<22} | "
        f"{drowsy_info.get('precision', 0.0) * 100.0:>10.2f}% | "
        f"{drowsy_info.get('recall', 0.0) * 100.0:>10.2f}% | "
        f"{drowsy_info.get('f1', 0.0) * 100.0:>10.2f}% | "
        f"{drowsy_info.get('support', 0):>6d}"
    )

    print("-" * 78)
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    print(" [*] Ma Trận Nhầm Lẫn (Confusion Matrix - Counts):")
    print(f"       Dự đoán -> {class_names[0]:<15} {class_names[1]:<15}")
    print(f"   Thực tế {class_names[0]:<12}: {cm[0][0]:<15} {cm[0][1]:<15}")
    print(f"   Thực tế {class_names[1]:<12}: {cm[1][0]:<15} {cm[1][1]:<15}")
    print("=" * 78 + "\n")


# ==============================================================================
# 8. QUY TRÌNH ĐÁNH GIÁ CHÍNH (PIPELINE ENTRY POINT)
# ==============================================================================
def evaluate(config: Optional[EvalConfig] = None) -> Dict[str, Any]:
    """
    Hàm thực thi toàn bộ pipeline đánh giá và trực quan hóa mô hình.
    Có thể gọi trực tiếp từ script hoặc import vào Notebooks.

    Args:
        config: Đối tượng EvalConfig chứa các hằng số tham số. Nếu None, dùng mặc định.

    Returns:
        Dict chứa toàn bộ kết quả đo đạc và thông tin tệp hình ảnh đã tạo.
    """
    if config is None:
        config = EvalConfig()

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 78)
    logger.info(f"KHỞI ĐỘNG TIẾN TRÌNH ĐÁNH GIÁ MÔ HÌNH CONVGRU (Thiết bị: {device})")
    logger.info("=" * 78)

    # 1. Đọc file cấu hình TrainConfig
    train_cfg_path = resolve_path(config.config_yaml_path)
    if train_cfg_path.exists():
        logger.info(f"[*] Đang nạp cấu hình hệ thống từ: {train_cfg_path}")
        train_cfg = load_config(str(train_cfg_path))
    else:
        logger.warning(f"[CẢNH BÁO] Không tìm thấy {train_cfg_path}, sử dụng cấu hình mặc định.")
        train_cfg = TrainConfig()

    # 2. Vẽ biểu đồ so sánh các giai đoạn từ history CSV nếu bật
    stage_plot_created = False
    best_epoch_num: Optional[int] = None
    if config.plot_stage_metrics:
        hist_csv_file = resolve_path(config.history_csv_path)
        stage_img_path = out_dir / "stage_metrics_comparison.png"
        logger.info(f"[*] Tiến hành vẽ biểu đồ so sánh Loss, Acc, Recall qua mỗi giai đoạn từ: {hist_csv_file}")
        stage_plot_created = plot_stage_metrics(hist_csv_file, stage_img_path)

    # 3. Nạp mô hình ConvGRUClassifier từ checkpoint
    eval_metrics: Dict[str, Any] = {}
    try:
        model, ckpt_dict, ckpt_file = load_eval_model(config.checkpoint_path, device, fallback_config=train_cfg)
        best_epoch_num = ckpt_dict.get("epoch", None)
        eval_metrics["checkpoint_file"] = str(ckpt_file)
        eval_metrics["checkpoint_epoch"] = best_epoch_num
        eval_metrics["checkpoint_saved_f1"] = ckpt_dict.get("best_val_f1", ckpt_dict.get("val_f1", None))
    except Exception as e:
        logger.error(f"[LỖI] Không thể nạp checkpoint mô hình: {e}")
        return {"status": "error", "error_message": str(e)}

    # Cập nhật lại biểu đồ giai đoạn với Best Epoch chính xác từ checkpoint
    if config.plot_stage_metrics and best_epoch_num is not None:
        hist_csv_file = resolve_path(config.history_csv_path)
        stage_img_path = out_dir / "stage_metrics_comparison.png"
        plot_stage_metrics(hist_csv_file, stage_img_path, best_epoch=best_epoch_num)

    # 4. Khởi tạo Validation DataLoader và trích xuất Backbone
    val_loader = build_val_dataloader(train_cfg, batch_size=config.batch_size, num_workers=config.num_workers)

    if val_loader is not None and len(val_loader.dataset) > 0:
        extractor = load_backbone_extractor(train_cfg, device, chunk_size=config.chunk_size, use_fp16=config.amp)

        # 5. Chạy suy luận trên tập Validation
        targets, preds, probs, avg_loss, avg_latency = run_evaluation_loop(
            model=model,
            extractor=extractor,
            val_loader=val_loader,
            device=device,
            amp=config.amp,
            max_batches=config.max_eval_batches
        )

        # 6. Tính toán các chỉ số phân loại
        metrics_dict = compute_classification_metrics(targets, preds, probs, num_classes=2)
        metrics_dict["loss"] = avg_loss
        metrics_dict["latency_ms_per_clip"] = avg_latency
        eval_metrics.update(metrics_dict)

        # 7. Vẽ ma trận nhầm lẫn
        if config.plot_confusion_matrix:
            cm_img_path = out_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                cm=np.array(metrics_dict["confusion_matrix"]),
                class_names=config.class_names,
                save_path=cm_img_path,
                metrics=metrics_dict
            )
            eval_metrics["confusion_matrix_image"] = str(cm_img_path)

        # 8. Vẽ đồ thị ROC & PR Curves
        if config.plot_roc_pr and len(targets) > 0:
            roc_pr_img_path = out_dir / "roc_pr_curves.png"
            plot_roc_pr_curves(targets, probs, roc_pr_img_path)
            eval_metrics["roc_pr_image"] = str(roc_pr_img_path)

        # 9. In bảng tóm tắt trên terminal
        print_evaluation_summary(eval_metrics, config.class_names)
    else:
        logger.warning(
            "[*] Tập dữ liệu video kiểm định không sẵn sàng trên máy này. "
            "Đã hoàn thành vẽ biểu đồ so sánh qua các giai đoạn từ file lịch sử huấn luyện (training_history.csv)."
        )

    # 10. Xuất file JSON kết quả
    if config.save_json_summary:
        summary_file = out_dir / "evaluation_summary.json"
        save_evaluation_summary(eval_metrics, summary_file)

    logger.info("=" * 78)
    logger.info(f"[THÀNH CÔNG] Toàn bộ kết quả và hình ảnh đánh giá đã được lưu tại: {out_dir}")
    logger.info("=" * 78)

    return eval_metrics


def main() -> None:
    """Điểm khởi chạy chính khi thực thi tệp: python src/evaluate.py."""
    config = EvalConfig()
    evaluate(config)


if __name__ == "__main__":
    main()
