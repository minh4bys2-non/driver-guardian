#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Tệp: train1.py
Mục đích:
    Điểm thực thi chính (Entry Point Wrapper) của pipeline huấn luyện mô hình Deep GRU
    với dữ liệu video thô trích xuất qua PyTorch native BackboneNeck (src/dataset2.py),
    đánh giá kết quả mô hình (Model Evaluation) và chẩn đoán quá trình huấn luyện
    (Training Process Diagnostics).

Luồng dữ liệu:
    - Tập Huấn luyện (Train): RawVideoBackboneNeckDataset từ src/dataset2.py
    - Tập Kiểm định (Val): RawVideoBackboneNeckDataset từ src/dataset2.py
    - Mô hình: DeepGRUClassifier từ src/models.py (CNNAdapter + Deep GRU + TemporalAttentionPooling + FC)
    - Hàm mất mát: DrowsinessLoss từ src/loss.py (CrossEntropyLoss hỗ trợ pos_weight)

Cơ chế cấu hình:
    - 100% CẤU HÌNH TẬP TRUNG QUA configs/config.py (TrainConfig)
    - KHÔNG DÙNG GIAO DIỆN DÒNG LỆNH (No CLI / No Argparse)
    - Thực thi trực tiếp: python train1.py
"""

import os
import sys
from pathlib import Path

# Đảm bảo console Windows hỗ trợ in tiếng Việt UTF-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Thêm thư mục gốc dự án vào sys.path
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Tối ưu hóa bộ cấp phát CUDA Caching Allocator: Chống phân mảnh bộ nhớ khi chuỗi video có độ dài biến thiên
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Tái xuất (re-export) toàn bộ các lớp và hàm từ src.train1
from src.train1 import (
    seed_everything,
    setup_logger,
    get_vram_info,
    cleanup_cuda_memory,
    MetricsTracker,
    TrainingVisualizer,
    CheckpointManager,
    DrowsinessTrainer1,
    DrowsinessTrainer,
    run_dry_run_test,
    train_pipeline,
    main
)

__all__ = [
    "seed_everything",
    "setup_logger",
    "get_vram_info",
    "cleanup_cuda_memory",
    "MetricsTracker",
    "TrainingVisualizer",
    "CheckpointManager",
    "DrowsinessTrainer1",
    "DrowsinessTrainer",
    "run_dry_run_test",
    "train_pipeline",
    "main"
]

if __name__ == "__main__":
    main()
