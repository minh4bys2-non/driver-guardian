#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module Đánh Giá & Benchmark Toàn Diện Mô Hình ModelInference (Driver Guardian AI)
Phiên bản: 3.0 (Benchmark ModelInference: Acc, Recall, F1, F2, Độ Trễ Đa Độ Dài, Tùy Biến Dataset)

Mô tả chức năng:
  1. Nạp mô hình suy luận đầu-cuối ModelInference (kết hợp NMSFreeDetector PAFPN và ConvGRUClassifier).
  2. Đo đạc hệ thống chỉ số phân loại chuyên sâu: Accuracy, Recall (Macro & Per-class), F1-Score,
     F2-Score (ưu tiên độ nhạy cảnh báo an toàn cho lớp Buồn ngủ), Confusion Matrix, ROC-AUC, PR-AUC.
  3. Khảo sát tốc độ xử lý (Latency ms/clip, Throughput FPS, ms/frame, Peak VRAM) với các mẫu có độ dài
     chuỗi khung hình khác nhau theo 2 chế độ:
     - Synthetic / Controlled Sequence Length Sweep: Quét qua dải độ dài (8, 16, 24, 32, 48, 64, 96, 128).
     - Dataset Sequence Length Binning: Phân nhóm video thực tế trong dataset theo các khoảng thời lượng.
  4. Tự do lựa chọn và tùy biến các bộ dữ liệu khác nhau (chỉ định thư mục dataset_dir, manifest_file,
     chọn split 'val'/'test'/'train', hoặc bộ lọc nguồn con như 'ul-dd', 'uta-rldd', 'sust').
  5. Tối ưu hóa ngưỡng quyết định (Threshold Tuning) cho F1 và F2.
  6. Trực quan hóa chuyên nghiệp:
     - Ma trận nhầm lẫn (Confusion Matrix Heatmap với count + % + Acc, Rec, F1, F2 stats).
     - Biểu đồ tốc độ đa trục Speed vs Length (4 panels: Latency, FPS, ms/frame, VRAM).
     - Đồ thị đường cong ROC và Precision-Recall Curves.
     - Biểu đồ phân tích độ nhạy ngưỡng quyết định F1/F2 vs Threshold.
  7. Xuất báo cáo tổng hợp chi tiết ra tệp JSON và hiển thị định dạng bảng đẹp mắt trên Console.

Người dùng có thể cấu hình trực tiếp tại lớp `EvalBenchmarkConfig` bên dưới
và thực thi: python src/evaluate.py
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

# Thiết lập đường dẫn môi trường và sys.path
CURRENT_DIR = Path(__file__).resolve().parent
LSTM_DIR = CURRENT_DIR.parent
PROJECT_ROOT = LSTM_DIR.parent.parent  # driver-guardian

for p in [str(PROJECT_ROOT), str(PROJECT_ROOT / "ai" / "ObjectDetection_2p6M"), str(LSTM_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

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
    fbeta_score,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import các thành phần dữ liệu từ src
from src.dataset import RawVideoFramesDataset


# ==============================================================================
# 1. NƠI TẬP TRUNG TOÀN BỘ HẰNG SỐ & THAM SỐ CẤU HÌNH (EVALBENCHMARKCONFIG)
# ==============================================================================
@dataclass
class EvalBenchmarkConfig:
    """
    NƠI TẬP TRUNG TOÀN BỘ THAM SỐ CẤU HÌNH BENCHMARK VÀ ĐÁNH GIÁ MÔ HÌNH.
    Người dùng có thể tự do chỉnh sửa trực tiếp các tham số dưới đây.
    """
    # ---- 1. CHECKPOINTS & MÔ HÌNH MODELINFERENCE ----
    # Trọng số CNN Backbone/PAFPN của NMSFreeDetector
    cnn_checkpoint_path: str = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best.pt"
    # Trọng số Spatio-Temporal ConvGRUClassifier
    conv_gru_checkpoint_path: str = r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_convgru\best.pt"
    # Kích thước chunk chia nhỏ khung hình khi trích xuất đặc trưng Backbone (chống tràn VRAM)
    chunk_size: int = 32
    # Thiết bị tính toán ('cuda' nếu có GPU NVIDIA, tự động fallback sang 'cpu')
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- 2. CẤU HÌNH BỘ DỮ LIỆU TÙY CHỌN (DATASET FLEXIBILITY) ----
    # Thư mục gốc chứa bộ dữ liệu video (người dùng tự do trỏ đến bất kỳ dataset nào)
    dataset_dir: str = r"E:\LSTM\data_processed"
    # Tệp manifest CSV chứa danh sách video clip, nhãn và trường phân chia split (None nếu quét thư mục)
    manifest_file: Optional[str] = "dataset_merged_split.csv"
    # Lựa chọn tập kiểm định: 'val', 'test', 'train', hoặc 'all'
    split: str = "val"
    # Bộ lọc nguồn dữ liệu con: 'all', 'ul-dd', 'uta-rldd', 'sust', ...
    dataset_source: str = "all"
    # Khoảng thời gian (giây) giữa các khung hình lấy mẫu (0.2s tương đương 5.0 FPS)
    sample_interval: float = 0.2
    # Cố định số lượng khung hình đưa vào mô hình (None = giữ nguyên độ dài tự nhiên của clip)
    seq_len: Optional[int] = None
    # Kích thước khung hình đưa vào mô hình (Height, Width = 640x640)
    img_size: int = 640
    # Số khung hình tối thiểu để coi là một video hợp lệ
    min_frames: int = 5
    # Giới hạn số mẫu đánh giá (None = đánh giá toàn bộ tập; int > 0 = kiểm tra nhanh/dry-run)
    max_eval_samples: Optional[int] = None

    # ---- 3. BENCHMARK TỐC ĐỘ THEO ĐỘ DÀI MẪU (SPEED PROFILING) ----
    # Bật/tắt benchmark tốc độ quét qua các độ dài chuỗi cố định
    run_synthetic_speed_benchmark: bool = True
    # Danh sách các độ dài chuỗi cần khảo sát tốc độ (ngắn đến dài)
    benchmark_seq_lengths: Tuple[int, ...] = (8, 16, 24, 32, 48, 64, 96, 128)
    # Kích thước khung hình chuẩn dùng cho kiểm thử tốc độ
    benchmark_img_size: int = 640
    # Số lượt chạy khởi động GPU làm ấm cache (warm-up)
    warmup_runs: int = 3
    # Số lượt lặp lại đo thời gian cho mỗi độ dài
    repeat_runs: int = 10

    # ---- 4. THRESHOLD & CHỈ SỐ PHÂN LOẠI ----
    # Ngưỡng phân loại mặc định (xác suất >= threshold -> Drowsy)
    classification_threshold: float = 0.5
    # Bật/tắt quét tìm ngưỡng tối ưu cho F1 và F2
    tune_threshold: bool = True
    # Tên hiển thị các lớp phân loại
    class_names: Tuple[str, str] = ("Alert (Tỉnh táo)", "Drowsy (Buồn ngủ)")
    # Hạt giống ngẫu nhiên để tái lập kết quả
    seed: int = 42

    # ---- 5. THƯ MỤC LƯU TRỮ & TRỰC QUAN HÓA ----
    output_dir: str = "logs/benchmark_val_SUST"
    plot_confusion_matrix: bool = True
    plot_speed_curves: bool = True
    plot_roc_pr: bool = True
    plot_threshold_curves: bool = True
    save_json_summary: bool = True


# Giữ alias EvalConfig để tương thích ngược 100% với các import cũ
EvalConfig = EvalBenchmarkConfig


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


def setup_logger(name: str = "BenchmarkModel") -> logging.Logger:
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
        base_dir = LSTM_DIR
    return (base_dir / p).resolve()


# ==============================================================================
# 3. BỘ NẠP MÔ HÌNH VÀ DỮ LIỆU (MODEL & DATASET LOADERS)
# ==============================================================================
def load_benchmark_model(config: EvalBenchmarkConfig) -> Any:
    """
    Nạp mô hình ModelInference từ các file checkpoint đã cấu hình.
    """
    try:
        from model_inference import ModelInference
    except ImportError:
        from ai.LSTM.model_inference import ModelInference

    cnn_path = resolve_path(config.cnn_checkpoint_path)
    conv_gru_path = resolve_path(config.conv_gru_checkpoint_path)

    if not cnn_path.exists():
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file checkpoint CNN: {cnn_path}")
    if not conv_gru_path.exists():
        raise FileNotFoundError(f"[LỖI] Không tìm thấy file checkpoint ConvGRU: {conv_gru_path}")

    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"[*] Đang khởi tạo ModelInference trên thiết bị: {device.type.upper()}")
    logger.info(f"    - CNN Checkpoint: {cnn_path}")
    logger.info(f"    - ConvGRU Checkpoint: {conv_gru_path}")
    logger.info(f"    - Chunk size: {config.chunk_size}")

    model = ModelInference(
        cnn_path=cnn_path,
        conv_gru_path=conv_gru_path,
        device=device,
        chunk_size=config.chunk_size,
    )
    model.eval()
    logger.info("[✓] Đã nạp thành công mô hình ModelInference.")
    return model


def build_benchmark_dataset(config: EvalBenchmarkConfig) -> Optional[RawVideoFramesDataset]:
    """
    Khởi tạo RawVideoFramesDataset dựa trên cấu hình bộ dữ liệu tùy chọn.
    """
    ds_dir = resolve_path(config.dataset_dir)
    if not ds_dir.exists():
        logger.warning(
            f"[CẢNH BÁO] Thư mục dataset không tồn tại: {ds_dir}. "
            "Bỏ qua đánh giá trên video thực tế, sẽ chỉ thực hiện synthetic speed benchmark."
        )
        return None

    manifest_path = None
    if config.manifest_file:
        cand_manifest = resolve_path(config.manifest_file, base_dir=ds_dir)
        if cand_manifest.exists():
            manifest_path = cand_manifest
        else:
            cand_manifest_lstm = resolve_path(config.manifest_file, base_dir=LSTM_DIR)
            if cand_manifest_lstm.exists():
                manifest_path = cand_manifest_lstm

    logger.info(
        f"[*] Khởi tạo Dataset từ '{ds_dir}' (Split='{config.split}', Manifest={manifest_path.name if manifest_path else 'None'})..."
    )

    try:
        dataset = RawVideoFramesDataset(
            dataset_dir=str(ds_dir),
            manifest_file=str(manifest_path) if manifest_path else None,
            split=config.split,
            sample_interval=config.sample_interval,
            seq_len=config.seq_len,
            img_size=config.img_size,
            min_frames=config.min_frames,
            augmenter=None,
            window_sampling="center",
        )

        # Lọc theo nguồn dữ liệu con nếu người dùng yêu cầu
        if config.dataset_source != "all":
            src_filter = config.dataset_source.lower().strip()
            filtered_samples = [s for s in dataset.samples if src_filter in s.source_dataset.lower()]
            dataset.samples = filtered_samples
            logger.info(f"[*] Đã lọc theo nguồn '{src_filter}': còn lại {len(dataset.samples)} mẫu.")

        # Cắt giảm số lượng mẫu nếu có giới hạn (Lấy mẫu cân bằng 2 lớp Alert và Drowsy)
        if config.max_eval_samples is not None and config.max_eval_samples > 0:
            if config.max_eval_samples < len(dataset.samples):
                alert_samples = [s for s in dataset.samples if s.label == 0]
                drowsy_samples = [s for s in dataset.samples if s.label == 1]
                n_half = config.max_eval_samples // 2
                sampled = alert_samples[:n_half] + drowsy_samples[:(config.max_eval_samples - n_half)]
                dataset.samples = sampled
            logger.info(f"[*] Giới hạn kiểm tra nhanh (cân bằng 2 lớp): {len(dataset.samples)} mẫu.")

        logger.info(f"[✓] Đã nạp thành công RawVideoFramesDataset: {len(dataset)} video clips.")
        return dataset

    except Exception as e:
        logger.error(f"[LỖI] Không thể khởi tạo RawVideoFramesDataset: {e}")
        return None


# ==============================================================================
# 4. ĐỘNG CƠ TÍNH TOÁN HỆ THỐNG CHỈ SỐ (METRICS ENGINE)
# ==============================================================================
def compute_benchmark_classification_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    probs: Optional[np.ndarray] = None,
    num_classes: int = 2,
    beta: float = 2.0,
) -> Dict[str, Any]:
    """
    Tính toán đầy đủ hệ thống chỉ số phân loại nhị phân:
    Accuracy, Recall (Macro & Per-class), F1, F2 (với beta=2.0), Precision, Confusion Matrix, ROC-AUC, PR-AUC.
    """
    if len(targets) == 0 or len(preds) == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "macro_f2": 0.0,
            "confusion_matrix": np.zeros((num_classes, num_classes), dtype=int).tolist(),
        }

    acc = float(accuracy_score(targets, preds))
    prec_macro = float(precision_score(targets, preds, average="macro", zero_division=0))
    rec_macro = float(recall_score(targets, preds, average="macro", zero_division=0))
    f1_macro = float(f1_score(targets, preds, average="macro", zero_division=0))
    f2_macro = float(fbeta_score(targets, preds, beta=beta, average="macro", zero_division=0))

    cm = confusion_matrix(targets, preds, labels=list(range(num_classes)))

    # Chỉ số chi tiết từng lớp
    per_class_rec = recall_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_prec = precision_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f1 = f1_score(targets, preds, average=None, labels=list(range(num_classes)), zero_division=0)
    per_class_f2 = fbeta_score(targets, preds, beta=beta, average=None, labels=list(range(num_classes)), zero_division=0)

    res: Dict[str, Any] = {
        "total_samples": int(len(targets)),
        "accuracy": acc,
        "macro_precision": prec_macro,
        "macro_recall": rec_macro,
        "macro_f1": f1_macro,
        "macro_f2": f2_macro,
        "confusion_matrix": cm.tolist(),
        "per_class": {
            "alert": {
                "precision": float(per_class_prec[0]) if len(per_class_prec) > 0 else 0.0,
                "recall": float(per_class_rec[0]) if len(per_class_rec) > 0 else 0.0,
                "f1": float(per_class_f1[0]) if len(per_class_f1) > 0 else 0.0,
                "f2": float(per_class_f2[0]) if len(per_class_f2) > 0 else 0.0,
                "support": int(np.sum(targets == 0)),
            },
            "drowsy": {
                "precision": float(per_class_prec[1]) if len(per_class_prec) > 1 else 0.0,
                "recall": float(per_class_rec[1]) if len(per_class_rec) > 1 else 0.0,
                "f1": float(per_class_f1[1]) if len(per_class_f1) > 1 else 0.0,
                "f2": float(per_class_f2[1]) if len(per_class_f2) > 1 else 0.0,
                "support": int(np.sum(targets == 1)),
            },
        },
    }

    # Tính ROC-AUC và PR-AUC (Average Precision)
    if probs is not None and len(np.unique(targets)) > 1:
        try:
            drowsy_probs = probs.ravel()
            fpr, tpr, _ = roc_curve(targets, drowsy_probs)
            res["roc_auc"] = float(auc(fpr, tpr))
            res["pr_auc"] = float(average_precision_score(targets, drowsy_probs))
        except Exception:
            res["roc_auc"] = None
            res["pr_auc"] = None
    else:
        res["roc_auc"] = None
        res["pr_auc"] = None

    return res


def tune_classification_thresholds(
    targets: np.ndarray,
    probs: np.ndarray,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """
    Khảo sát biến thiên của F1 và F2 trên 100 ngưỡng từ 0.01 đến 0.99,
    tự động tìm ra ngưỡng tối ưu nhất cho F1 và F2.
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    f1_scores = []
    f2_scores = []
    rec_drowsy_scores = []
    prec_drowsy_scores = []

    drowsy_probs = probs.ravel()

    best_thresh_f1 = 0.5
    best_f1_val = -1.0
    best_thresh_f2 = 0.5
    best_f2_val = -1.0

    for th in thresholds:
        bin_preds = (drowsy_probs >= th).astype(int)
        f1_val = float(f1_score(targets, bin_preds, pos_label=1, zero_division=0))
        f2_val = float(fbeta_score(targets, bin_preds, beta=2.0, pos_label=1, zero_division=0))
        rec_val = float(recall_score(targets, bin_preds, pos_label=1, zero_division=0))
        prec_val = float(precision_score(targets, bin_preds, pos_label=1, zero_division=0))

        f1_scores.append(f1_val)
        f2_scores.append(f2_val)
        rec_drowsy_scores.append(rec_val)
        prec_drowsy_scores.append(prec_val)

        if f1_val > best_f1_val:
            best_f1_val = f1_val
            best_thresh_f1 = float(th)

        if f2_val > best_f2_val:
            best_f2_val = f2_val
            best_thresh_f2 = float(th)

    return {
        "thresholds": thresholds.tolist(),
        "f1_scores": f1_scores,
        "f2_scores": f2_scores,
        "recall_scores": rec_drowsy_scores,
        "precision_scores": prec_drowsy_scores,
        "best_thresh_f1": best_thresh_f1,
        "best_f1_val": best_f1_val,
        "best_thresh_f2": best_thresh_f2,
        "best_f2_val": best_f2_val,
    }


# ==============================================================================
# 5. ĐỘNG CƠ KHẢO SÁT TỐC ĐỘ THEO ĐỘ DÀI (SPEED PROFILING ENGINES)
# ==============================================================================
def profile_synthetic_speed_sweep(
    model: Any,
    config: EvalBenchmarkConfig,
) -> Dict[str, Any]:
    """
    Thực hiện benchmark tốc độ có kiểm soát (Synthetic Sequence Length Sweep)
    trên danh sách các độ dài chuỗi quy định.
    """
    logger.info("=" * 78)
    logger.info("[*] KHỞI CHẠY BENCHMARK TỐC ĐỘ THEO CÁC ĐỘ DÀI CHUỖI CỐ ĐỊNH (SYNTHETIC SWEEP)")
    logger.info("=" * 78)

    device = next(model.cnn_backbone.parameters()).device
    img_size = config.benchmark_img_size
    results: List[Dict[str, Any]] = []

    for seq_len in config.benchmark_seq_lengths:
        logger.info(f"  -> Đang kiểm thử độ dài chuỗi T = {seq_len:3d} frames...")

        # Tạo dummy tensor trên CPU dạng uint8 [T, 3, H, W]
        dummy_input = torch.zeros((seq_len, 3, img_size, img_size), dtype=torch.uint8)

        # Chạy khởi động GPU làm ấm cache (Warmup)
        for _ in range(config.warmup_runs):
            _ = model(dummy_input)

        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

        # Đo lặp lại nhiều lần
        latencies_ms: List[float] = []
        for _ in range(config.repeat_runs):
            t_start = time.perf_counter()
            _ = model(dummy_input)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            latencies_ms.append(t_elapsed_ms)

        mean_latency = float(np.mean(latencies_ms))
        std_latency = float(np.std(latencies_ms))
        min_latency = float(np.min(latencies_ms))
        max_latency = float(np.max(latencies_ms))
        latency_per_frame = mean_latency / max(seq_len, 1)
        fps = (seq_len * 1000.0) / mean_latency if mean_latency > 0 else 0.0

        peak_vram_mb = 0.0
        if device.type == "cuda":
            peak_vram_mb = float(torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0))

        logger.info(
            f"     [T={seq_len:3d}] Latency: {mean_latency:6.2f} ms (±{std_latency:4.2f}) | "
            f"Frame: {latency_per_frame:5.2f} ms | FPS: {fps:6.1f} | Peak VRAM: {peak_vram_mb:6.1f} MB"
        )

        results.append({
            "seq_len": int(seq_len),
            "mean_latency_ms": mean_latency,
            "std_latency_ms": std_latency,
            "min_latency_ms": min_latency,
            "max_latency_ms": max_latency,
            "latency_per_frame_ms": latency_per_frame,
            "fps": fps,
            "peak_vram_mb": peak_vram_mb,
        })

    logger.info("[✓] Đã hoàn thành đo đạc tốc độ trên toàn bộ các độ dài chuỗi.")
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "benchmark_records": results,
    }


def evaluate_dataset(
    model: Any,
    dataset: RawVideoFramesDataset,
    config: EvalBenchmarkConfig,
) -> Dict[str, Any]:
    """
    Chạy suy luận và đo đạc toàn diện trên tập dữ liệu video thực tế,
    thu thập nhãn, xác suất, thời gian xử lý và phân nhóm theo độ dài chuỗi.
    """
    total_samples = len(dataset)
    logger.info("=" * 78)
    logger.info(f"[*] BẮT ĐẦU ĐÁNH GIÁ TRÊN TẬP DỮ LIỆU VIDEO: {total_samples} MẪU (Split: '{config.split}')")
    logger.info("=" * 78)

    all_targets: List[int] = []
    all_preds: List[int] = []
    all_probs: List[float] = []
    all_seq_lens: List[int] = []
    all_latencies_ms: List[float] = []
    sample_records: List[Dict[str, Any]] = []

    corrupted_count = 0
    t_start_total = time.perf_counter()

    for idx in range(total_samples):
        frames_t, label_t, seq_len_t, meta = dataset[idx]

        if frames_t is None or frames_t.size(0) == 0:
            corrupted_count += 1
            continue

        target = int(label_t.item())
        seq_len = int(frames_t.size(0))

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        try:
            score = model(frames_t)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t_lat = (time.perf_counter() - t0) * 1000.0

            pred = 1 if score >= config.classification_threshold else 0

            all_targets.append(target)
            all_preds.append(pred)
            all_probs.append(score)
            all_seq_lens.append(seq_len)
            all_latencies_ms.append(t_lat)

            sample_records.append({
                "video_id": meta.get("video_id", f"sample_{idx}"),
                "target": target,
                "pred": pred,
                "prob": score,
                "seq_len": seq_len,
                "latency_ms": t_lat,
            })

            # Hiển thị tiến độ định kỳ
            if (idx + 1) % max(1, total_samples // 10) == 0 or (idx + 1) == total_samples:
                cur_fps = (seq_len * 1000.0) / t_lat if t_lat > 0 else 0.0
                logger.info(
                    f"  -> Tiến độ: [{idx + 1:4d}/{total_samples:4d}] | "
                    f"Clip: {meta.get('video_id', '')[:32]:<32} | "
                    f"T={seq_len:3d} | Score: {score:.4f} (GT={target}) | "
                    f"Latency: {t_lat:6.2f}ms ({cur_fps:5.1f} FPS)"
                )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"[!] CẢNH BÁO OOM tại mẫu {idx}. Đang xả GPU cache và bỏ qua...")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            raise e
        finally:
            del frames_t

    total_eval_time = time.perf_counter() - t_start_total
    valid_count = len(all_targets)
    logger.info(
        f"[✓] Hoàn thành suy luận: {valid_count}/{total_samples} mẫu hợp lệ "
        f"({corrupted_count} mẫu lỗi/bỏ qua) trong {total_eval_time:.2f}s."
    )

    targets_np = np.array(all_targets, dtype=int)
    preds_np = np.array(all_preds, dtype=int)
    probs_np = np.array(all_probs, dtype=np.float32)
    seq_lens_np = np.array(all_seq_lens, dtype=int)
    latencies_np = np.array(all_latencies_ms, dtype=np.float32)

    # 1. Tính toán hệ thống chỉ số phân loại tổng thể
    metrics = compute_benchmark_classification_metrics(targets_np, preds_np, probs_np, num_classes=2, beta=2.0)
    metrics["avg_latency_ms"] = float(np.mean(latencies_np)) if len(latencies_np) > 0 else 0.0
    metrics["std_latency_ms"] = float(np.std(latencies_np)) if len(latencies_np) > 0 else 0.0
    metrics["avg_fps"] = (
        float(np.mean((seq_lens_np * 1000.0) / latencies_np)) if len(latencies_np) > 0 else 0.0
    )
    metrics["corrupted_samples"] = corrupted_count

    # 2. Phân tích tối ưu hóa ngưỡng quyết định (Threshold Tuning)
    threshold_analysis = None
    if config.tune_threshold and len(targets_np) > 0:
        threshold_analysis = tune_classification_thresholds(targets_np, probs_np, num_steps=100)
        metrics["threshold_tuning"] = {
            "default_threshold": config.classification_threshold,
            "best_thresh_f1": threshold_analysis["best_thresh_f1"],
            "best_f1_val": threshold_analysis["best_f1_val"],
            "best_thresh_f2": threshold_analysis["best_thresh_f2"],
            "best_f2_val": threshold_analysis["best_f2_val"],
        }

    # 3. Phân nhóm theo độ dài chuỗi thực tế trong dataset (Dataset Length Binning)
    bins_config = [
        ("T <= 16", lambda t: t <= 16),
        ("16 < T <= 32", lambda t: (t > 16) & (t <= 32)),
        ("32 < T <= 64", lambda t: (t > 32) & (t <= 64)),
        ("T > 64", lambda t: t > 64),
    ]

    length_bins_report: List[Dict[str, Any]] = []
    for bin_name, bin_filter in bins_config:
        mask = bin_filter(seq_lens_np)
        bin_samples = int(np.sum(mask))
        if bin_samples == 0:
            continue

        b_targets = targets_np[mask]
        b_preds = preds_np[mask]
        b_probs = probs_np[mask]
        b_lats = latencies_np[mask]
        b_lens = seq_lens_np[mask]

        b_metrics = compute_benchmark_classification_metrics(b_targets, b_preds, b_probs, num_classes=2, beta=2.0)
        b_mean_lat = float(np.mean(b_lats))
        b_fps = float(np.mean((b_lens * 1000.0) / b_lats))

        length_bins_report.append({
            "bin_name": bin_name,
            "sample_count": bin_samples,
            "mean_latency_ms": b_mean_lat,
            "throughput_fps": b_fps,
            "accuracy": b_metrics["accuracy"],
            "macro_recall": b_metrics["macro_recall"],
            "drowsy_recall": b_metrics["per_class"]["drowsy"]["recall"],
            "macro_f1": b_metrics["macro_f1"],
            "macro_f2": b_metrics["macro_f2"],
            "drowsy_f2": b_metrics["per_class"]["drowsy"]["f2"],
        })

    metrics["length_bins_analysis"] = length_bins_report

    return {
        "metrics": metrics,
        "targets": targets_np,
        "preds": preds_np,
        "probs": probs_np,
        "seq_lens": seq_lens_np,
        "latencies_ms": latencies_np,
        "threshold_analysis": threshold_analysis,
    }


# ==============================================================================
# 6. ĐỘNG CƠ TRỰC QUAN HÓA ĐỒ THỊ (VISUALIZATION ENGINES)
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
    title: str = "Ma Trận Nhầm Lẫn - ModelInference",
) -> None:
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix) dạng Heatmap trực quan.
    Hiển thị số lượng mẫu tuyệt đối, tỷ lệ phần trăm (%) và hộp chỉ số Acc, Rec, F1, F2.
    """
    set_plot_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    cm_counts = np.array(cm, dtype=int)
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm_counts.astype(float),
        row_sums,
        out=np.zeros_like(cm_counts, dtype=float),
        where=row_sums != 0,
    )

    annot_matrix = np.empty_like(cm_counts, dtype=object)
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = cm_counts[i, j]
            pct = cm_norm[i, j] * 100.0
            annot_matrix[i, j] = f"{count}\n({pct:.1f}%)"

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=300)

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
        ax=ax,
    )

    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.set_xlabel("Nhãn Dự Đoán (Predicted Label)", fontsize=12, fontweight="bold", labelpad=10)
    ax.set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=12, fontweight="bold", labelpad=10)

    if metrics is not None:
        acc = metrics.get("accuracy", 0.0) * 100.0
        rec_drowsy = metrics.get("per_class", {}).get("drowsy", {}).get("recall", 0.0) * 100.0
        rec_alert = metrics.get("per_class", {}).get("alert", {}).get("recall", 0.0) * 100.0
        f1_macro = metrics.get("macro_f1", 0.0) * 100.0
        f2_macro = metrics.get("macro_f2", 0.0) * 100.0
        f2_drowsy = metrics.get("per_class", {}).get("drowsy", {}).get("f2", 0.0) * 100.0
        total = metrics.get("total_samples", int(np.sum(cm_counts)))

        summary_box = (
            f"Tổng số mẫu: {total}\n"
            f"Độ chính xác (Acc): {acc:.2f}%\n"
            f"Độ nhạy Buồn ngủ (Drowsy Recall): {rec_drowsy:.2f}%\n"
            f"Độ nhạy Tỉnh táo (Alert Recall): {rec_alert:.2f}%\n"
            f"F1-Score (Macro): {f1_macro:.2f}% | F2-Score (Macro): {f2_macro:.2f}%\n"
            f"F2-Score Buồn ngủ (Drowsy F2 - Beta=2.0): {f2_drowsy:.2f}%"
        )
        plt.figtext(
            0.5,
            -0.08,
            summary_box,
            wrap=True,
            horizontalalignment="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.6", facecolor="#f0f4f8", edgecolor="#b0bec5", alpha=0.9),
        )

    plt.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"[✓] Đã lưu biểu đồ Ma trận nhầm lẫn tại: {save_path}")


def plot_speed_vs_sequence_length(
    speed_sweep_data: Dict[str, Any],
    save_path: Path,
) -> None:
    """
    Vẽ lưới biểu đồ 2x2 khảo sát tốc độ xử lý theo độ dài chuỗi khung hình:
    1. Clip Latency (ms) vs T
    2. Throughput (FPS) vs T
    3. Frame Latency (ms/frame) vs T
    4. Peak GPU VRAM (MB) vs T
    """
    set_plot_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    records = speed_sweep_data.get("benchmark_records", [])
    if not records:
        logger.warning("[!] Không có dữ liệu benchmark tốc độ để vẽ biểu đồ.")
        return

    seq_lens = [r["seq_len"] for r in records]
    latencies = [r["mean_latency_ms"] for r in records]
    latency_stds = [r["std_latency_ms"] for r in records]
    fps_vals = [r["fps"] for r in records]
    frame_lats = [r["latency_per_frame_ms"] for r in records]
    vrams = [r["peak_vram_mb"] for r in records]
    dev_name = speed_sweep_data.get("device_name", "GPU/CPU")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)

    # 1. Clip Latency vs T (ms)
    ax = axes[0, 0]
    ax.errorbar(
        seq_lens,
        latencies,
        yerr=latency_stds,
        fmt="o-",
        linewidth=2.2,
        capsize=5,
        color="#1976D2",
        ecolor="#90CAF9",
        label="Latency (ms/clip)",
    )
    ax.set_title("1. Thời Gian Suy Luận Trên Mỗi Clip Video", fontsize=13, fontweight="bold")
    ax.set_xlabel("Độ Dài Chuỗi Khung Hình (Sequence Length T)", fontsize=11)
    ax.set_ylabel("Latency (ms)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")

    # 2. Throughput vs T (FPS)
    ax = axes[0, 1]
    ax.plot(seq_lens, fps_vals, marker="s", linewidth=2.2, color="#388E3C", label="Model Throughput (FPS)")
    ax.axhline(y=30.0, color="#E65100", linestyle="--", linewidth=1.5, label="Real-time Baseline (30 FPS)")
    ax.axhline(y=5.0, color="#7B1FA2", linestyle=":", linewidth=1.5, label="Sampling Rate (5 FPS / 0.2s)")
    ax.set_title("2. Tốc Độ Xử Lý Khung Hình (Throughput FPS)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Độ Dài Chuỗi Khung Hình (Sequence Length T)", fontsize=11)
    ax.set_ylabel("Tốc độ (Frames / Second)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")

    # 3. Frame Latency vs T (ms/frame)
    ax = axes[1, 0]
    ax.plot(seq_lens, frame_lats, marker="^", linewidth=2.2, color="#F57C00", label="Latency / Frame (ms)")
    ax.set_title("3. Thời Gian Xử Lý Trung Bình Mỗi Khung Hình", fontsize=13, fontweight="bold")
    ax.set_xlabel("Độ Dài Chuỗi Khung Hình (Sequence Length T)", fontsize=11)
    ax.set_ylabel("Latency (ms/frame)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper right")

    # 4. Peak VRAM vs T (MB)
    ax = axes[1, 1]
    ax.plot(seq_lens, vrams, marker="d", linewidth=2.2, color="#D32F2F", label="Peak VRAM (MB)")
    ax.set_title(f"4. Bộ Nhớ GPU Tiêu Thụ Đỉnh ({dev_name})", fontsize=13, fontweight="bold")
    ax.set_xlabel("Độ Dài Chuỗi Khung Hình (Sequence Length T)", fontsize=11)
    ax.set_ylabel("Peak VRAM (MB)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="upper left")

    plt.suptitle(
        f"BENCHMARK TỐC ĐỘ MÔ HÌNH MODELINFERENCE THEO ĐỘ DÀI KHUNG HÌNH ({dev_name})",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"[✓] Đã lưu biểu đồ khảo sát tốc độ tại: {save_path}")


def plot_roc_pr_curves(
    targets: np.ndarray,
    probs: np.ndarray,
    save_path: Path,
) -> None:
    """Vẽ đồ thị đường cong ROC và Precision-Recall."""
    set_plot_style()
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return

    save_path.parent.mkdir(parents=True, exist_ok=True)
    drowsy_probs = probs.ravel()
    fpr, tpr, _ = roc_curve(targets, drowsy_probs)
    roc_auc_val = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(targets, drowsy_probs)
    avg_prec = average_precision_score(targets, drowsy_probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Đồ thị ROC
    ax1.plot(fpr, tpr, color="#E65100", lw=2.5, label=f"ModelInference ROC (AUC = {roc_auc_val:.4f})")
    ax1.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--", label="Random Guess")
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11, fontweight="bold")
    ax1.set_title("Đường Cong ROC (Receiver Operating Characteristic)", fontsize=12, fontweight="bold")
    ax1.legend(loc="lower right", frameon=True)
    ax1.grid(True, linestyle="--", alpha=0.6)

    # Đồ thị Precision-Recall
    ax2.plot(recall, precision, color="#00897B", lw=2.5, label=f"ModelInference PR (AP = {avg_prec:.4f})")
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


def plot_threshold_curves(
    threshold_data: Dict[str, Any],
    save_path: Path,
) -> None:
    """
    Vẽ đồ thị phân tích sự biến thiên của F1-Score, F2-Score, Precision và Recall
    theo các ngưỡng quyết định (Decision Threshold).
    """
    set_plot_style()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = threshold_data.get("thresholds", [])
    f1_scores = threshold_data.get("f1_scores", [])
    f2_scores = threshold_data.get("f2_scores", [])
    rec_scores = threshold_data.get("recall_scores", [])
    prec_scores = threshold_data.get("precision_scores", [])

    best_th_f1 = threshold_data.get("best_thresh_f1", 0.5)
    best_val_f1 = threshold_data.get("best_f1_val", 0.0)
    best_th_f2 = threshold_data.get("best_thresh_f2", 0.5)
    best_val_f2 = threshold_data.get("best_f2_val", 0.0)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.plot(thresholds, f1_scores, color="#1976D2", lw=2.4, label=f"F1-Score (Max: {best_val_f1:.3f} @ Th={best_th_f1:.2f})")
    ax.plot(thresholds, f2_scores, color="#D32F2F", lw=2.4, label=f"F2-Score (Max: {best_val_f2:.3f} @ Th={best_th_f2:.2f})")
    ax.plot(thresholds, rec_scores, color="#7B1FA2", lw=1.8, linestyle="--", label="Drowsy Recall")
    ax.plot(thresholds, prec_scores, color="#388E3C", lw=1.8, linestyle=":", label="Drowsy Precision")

    # Đánh dấu các ngưỡng tối ưu
    ax.axvline(x=best_th_f1, color="#1976D2", linestyle="--", alpha=0.7)
    ax.scatter([best_th_f1], [best_val_f1], color="#1976D2", s=120, zorder=5, marker="*")

    ax.axvline(x=best_th_f2, color="#D32F2F", linestyle="--", alpha=0.7)
    ax.scatter([best_th_f2], [best_val_f2], color="#D32F2F", s=140, zorder=5, marker="X")

    ax.set_title("PHÂN TÍCH BIẾN THIÊN F1 VÀ F2 THEO NGƯỠNG QUYẾT ĐỊNH (THRESHOLD TUNING)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Ngưỡng Quyết Định Phân Loại (Decision Threshold)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Giá Trị Chỉ Số (Metric Score)", fontsize=11, fontweight="bold")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(frameon=True, loc="lower center")

    plt.tight_layout()
    fig.savefig(str(save_path), bbox_inches="tight", dpi=300)
    plt.close(fig)
    logger.info(f"[✓] Đã lưu đồ thị phân tích ngưỡng tại: {save_path}")


# ==============================================================================
# 7. BỘ XUẤT BÁO CÁO & GIAO DIỆN DÒNG LỆNH (REPORT EXPORTERS)
# ==============================================================================
def save_benchmark_summary(summary_data: Dict[str, Any], save_path: Path) -> None:
    """Lưu trữ báo cáo tổng hợp kết quả đánh giá ra tệp JSON."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    logger.info(f"[✓] Đã lưu báo cáo đánh giá JSON tại: {save_path}")


def print_benchmark_summary(
    eval_results: Dict[str, Any],
    speed_sweep_results: Optional[Dict[str, Any]],
    config: EvalBenchmarkConfig,
) -> None:
    """Hiển thị bảng tổng kết đẹp mắt trên giao diện dòng lệnh."""
    metrics = eval_results.get("metrics", {})
    class_names = config.class_names

    print("\n" + "=" * 82)
    print("      KẾT QUẢ BENCHMARK MÔ HÌNH MODELINFERENCE (DRIVER GUARDIAN AI)     ")
    print("=" * 82)

    samples = metrics.get("total_samples", 0)
    acc = metrics.get("accuracy", 0.0) * 100.0
    prec = metrics.get("macro_precision", 0.0) * 100.0
    rec = metrics.get("macro_recall", 0.0) * 100.0
    f1 = metrics.get("macro_f1", 0.0) * 100.0
    f2 = metrics.get("macro_f2", 0.0) * 100.0
    latency = metrics.get("avg_latency_ms", 0.0)
    fps = metrics.get("avg_fps", 0.0)

    print(f" [*] Bộ dữ liệu đánh giá       : '{config.dataset_dir}' (Split: '{config.split}')")
    print(f" [*] Tổng số mẫu kiểm định     : {samples} video clips")
    print(f" [*] Ngưỡng quyết định mặc định : {config.classification_threshold:.2f}")
    print(f" [*] Độ chính xác tổng thể (Acc): {acc:.2f}%")
    print(f" [*] Độ chuẩn xác vĩ mô (Macro Prec): {prec:.2f}%")
    print(f" [*] Độ nhạy vĩ mô (Macro Recall)  : {rec:.2f}%")
    print(f" [*] F1-Score vĩ mô (Macro F1)     : {f1:.2f}%")
    print(f" [*] F2-Score vĩ mô (Macro F2 - B=2): {f2:.2f}%  <-- [ĐẶC BIỆT QUAN TRỌNG]")
    if metrics.get("roc_auc") is not None:
        print(f" [*] Diện tích ROC (ROC-AUC)        : {metrics['roc_auc']:.4f}")
    if metrics.get("pr_auc") is not None:
        print(f" [*] Diện tích PR (PR-AUC / AP)     : {metrics['pr_auc']:.4f}")
    print(f" [*] Độ trễ suy luận trung bình     : {latency:.2f} ms/clip ({fps:.1f} FPS)")

    # Bảng chi tiết từng lớp
    print("-" * 82)
    print(f" {'Lớp Phân Loại':<22} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'F2-Score':<10} | {'Mẫu':<6}")
    print("-" * 82)

    per_cls = metrics.get("per_class", {})
    alert_info = per_cls.get("alert", {})
    drowsy_info = per_cls.get("drowsy", {})

    print(
        f" {class_names[0]:<22} | "
        f"{alert_info.get('precision', 0.0) * 100.0:>8.2f}% | "
        f"{alert_info.get('recall', 0.0) * 100.0:>8.2f}% | "
        f"{alert_info.get('f1', 0.0) * 100.0:>8.2f}% | "
        f"{alert_info.get('f2', 0.0) * 100.0:>8.2f}% | "
        f"{alert_info.get('support', 0):>6d}"
    )
    print(
        f" {class_names[1]:<22} | "
        f"{drowsy_info.get('precision', 0.0) * 100.0:>8.2f}% | "
        f"{drowsy_info.get('recall', 0.0) * 100.0:>8.2f}% | "
        f"{drowsy_info.get('f1', 0.0) * 100.0:>8.2f}% | "
        f"{drowsy_info.get('f2', 0.0) * 100.0:>8.2f}% | "
        f"{drowsy_info.get('support', 0):>6d}"
    )

    # Ma trận nhầm lẫn
    cm = metrics.get("confusion_matrix", [[0, 0], [0, 0]])
    print("-" * 82)
    print(" [*] Ma Trận Nhầm Lẫn (Confusion Matrix - Counts):")
    print(f"       Dự đoán -> {class_names[0]:<18} {class_names[1]:<18}")
    print(f"   Thực tế {class_names[0]:<14}: {cm[0][0]:<18} {cm[0][1]:<18}")
    print(f"   Thực tế {class_names[1]:<14}: {cm[1][0]:<18} {cm[1][1]:<18}")

    # Tối ưu ngưỡng
    if "threshold_tuning" in metrics:
        tt = metrics["threshold_tuning"]
        print("-" * 82)
        print(" [*] Phân Tích Ngưỡng Tối Ưu (Optimal Thresholds):")
        print(f"     - Ngưỡng đạt Max F1-Score: Threshold = {tt['best_thresh_f1']:.2f} (F1 = {tt['best_f1_val'] * 100.0:.2f}%)")
        print(f"     - Ngưỡng đạt Max F2-Score: Threshold = {tt['best_thresh_f2']:.2f} (F2 = {tt['best_f2_val'] * 100.0:.2f}%)")

    # Bảng phân nhóm độ dài thực tế
    length_bins = metrics.get("length_bins_analysis", [])
    if length_bins:
        print("-" * 82)
        print(" [*] Phân Tích Hiệu Năng & Tốc Độ Theo Nhóm Độ Dài Thực Tế (Length Bins):")
        print(f" {'Nhóm Độ Dài':<16} | {'Mẫu':<6} | {'Latency':<12} | {'FPS':<8} | {'Acc':<8} | {'Drowsy Rec':<11} | {'F2 Drowsy':<10}")
        print("-" * 82)
        for b in length_bins:
            print(
                f" {b['bin_name']:<16} | "
                f"{b['sample_count']:>6d} | "
                f"{b['mean_latency_ms']:>8.2f} ms | "
                f"{b['throughput_fps']:>6.1f} | "
                f"{b['accuracy'] * 100.0:>6.2f}% | "
                f"{b['drowsy_recall'] * 100.0:>9.2f}% | "
                f"{b['drowsy_f2'] * 100.0:>8.2f}%"
            )

    # Bảng benchmark tốc độ sweep (nếu có)
    if speed_sweep_results and "benchmark_records" in speed_sweep_results:
        records = speed_sweep_results["benchmark_records"]
        print("-" * 82)
        print(" [*] Benchmark Tốc Độ Có Kiểm Soát (Synthetic Sequence Length Sweep):")
        print(f" {'Độ Dài T':<10} | {'Latency Clip':<16} | {'Latency Frame':<15} | {'Throughput':<12} | {'Peak VRAM':<10}")
        print("-" * 82)
        for r in records:
            print(
                f" {r['seq_len']:>6d} f   | "
                f"{r['mean_latency_ms']:>8.2f} ± {r['std_latency_ms']:<4.1f} ms | "
                f"{r['latency_per_frame_ms']:>10.2f} ms/f | "
                f"{r['fps']:>8.1f} FPS | "
                f"{r['peak_vram_mb']:>8.1f} MB"
            )

    print("=" * 82 + "\n")


# ==============================================================================
# 8. TIẾN TRÌNH THỰC THI CHÍNH (BENCHMARK PIPELINE ENTRY POINT)
# ==============================================================================
def evaluate_benchmark(config: Optional[EvalBenchmarkConfig] = None) -> Dict[str, Any]:
    """
    Hàm thực thi toàn bộ pipeline benchmark và đánh giá mô hình ModelInference.
    Có thể gọi trực tiếp từ script hoặc import vào Notebooks.
    """
    if config is None:
        config = EvalBenchmarkConfig()

    seed_everything(config.seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    out_dir = resolve_path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 78)
    logger.info(f"KHỞI ĐỘNG TIẾN TRÌNH BENCHMARK MÔ HÌNH MODELINFERENCE (Thiết bị: {device})")
    logger.info("=" * 78)

    # 1. Nạp mô hình ModelInference
    try:
        model = load_benchmark_model(config)
    except Exception as e:
        logger.error(f"[LỖI] Không thể nạp mô hình ModelInference: {e}")
        return {"status": "error", "error_message": str(e)}

    # 2. Chạy benchmark tốc độ có kiểm soát (Synthetic Sequence Length Sweep)
    speed_sweep_data = None
    if config.run_synthetic_speed_benchmark:
        try:
            speed_sweep_data = profile_synthetic_speed_sweep(model, config)
            if config.plot_speed_curves:
                speed_img_path = out_dir / "benchmark_speed_vs_length.png"
                plot_speed_vs_sequence_length(speed_sweep_data, speed_img_path)
        except Exception as e:
            logger.error(f"[LỖI] Lỗi khi chạy benchmark tốc độ sweep: {e}")

    # 3. Nạp bộ dữ liệu video kiểm định tùy chọn
    dataset = build_benchmark_dataset(config)
    eval_results: Dict[str, Any] = {"metrics": {}}

    if dataset is not None and len(dataset) > 0:
        # 4. Chạy suy luận trên video thực tế
        eval_results = evaluate_dataset(model, dataset, config)
        metrics = eval_results["metrics"]

        # 5. Vẽ Ma trận nhầm lẫn
        if config.plot_confusion_matrix and "confusion_matrix" in metrics:
            cm_img_path = out_dir / "confusion_matrix.png"
            plot_confusion_matrix(
                cm=np.array(metrics["confusion_matrix"]),
                class_names=config.class_names,
                save_path=cm_img_path,
                metrics=metrics,
            )
            metrics["confusion_matrix_image"] = str(cm_img_path)

        # 6. Vẽ đồ thị ROC & PR Curves
        if config.plot_roc_pr and len(eval_results["targets"]) > 0:
            roc_pr_img_path = out_dir / "roc_pr_curves.png"
            plot_roc_pr_curves(eval_results["targets"], eval_results["probs"], roc_pr_img_path)
            metrics["roc_pr_image"] = str(roc_pr_img_path)

        # 7. Vẽ đồ thị phân tích ngưỡng quyết định
        if config.plot_threshold_curves and eval_results.get("threshold_analysis"):
            thresh_img_path = out_dir / "threshold_analysis.png"
            plot_threshold_curves(eval_results["threshold_analysis"], thresh_img_path)
            metrics["threshold_analysis_image"] = str(thresh_img_path)

        # 8. Hiển thị bảng tổng kết trên terminal
        print_benchmark_summary(eval_results, speed_sweep_data, config)

    else:
        logger.warning(
            "[*] Bỏ qua đánh giá tập dữ liệu video do thư mục không sẵn sàng. "
            "Đã hoàn thành phần Benchmark tốc độ Synthetic Sweep."
        )
        if speed_sweep_results := speed_sweep_data:
            print_benchmark_summary({"metrics": {}}, speed_sweep_results, config)

    # 9. Xuất tệp báo cáo tổng hợp JSON
    final_summary = {
        "config": asdict(config),
        "device_info": {
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        },
        "evaluation_metrics": eval_results.get("metrics", {}),
        "speed_sweep_benchmark": speed_sweep_data,
    }

    if config.save_json_summary:
        summary_file = out_dir / "benchmark_summary.json"
        save_benchmark_summary(final_summary, summary_file)

    logger.info("=" * 78)
    logger.info(f"[THÀNH CÔNG] Toàn bộ kết quả và đồ thị benchmark đã được lưu tại: {out_dir}")
    logger.info("=" * 78)

    return final_summary


# Giữ alias evaluate để tương thích ngược 100%
evaluate = evaluate_benchmark


def main() -> None:
    """Điểm khởi chạy chính khi thực thi tệp: python src/evaluate.py."""
    config = EvalBenchmarkConfig()
    evaluate_benchmark(config)


if __name__ == "__main__":
    main()
