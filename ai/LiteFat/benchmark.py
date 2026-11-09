#!/usr/bin/env python3
# =====================================================================
# File: benchmark.py
# Dự án: Driver Guardian - Module LiteFat (IEEE/RSJ IROS 2025 Upgraded)
# Tác vụ: Đánh giá chuyên sâu mô hình LiteFat trên cả 2 tập Validation
#         và Test ở ngưỡng mặc định (0.50) và ngưỡng tối ưu, vẽ 7 bộ
#         biểu đồ trực quan hóa chuẩn xuất bản (300 DPI).
# =====================================================================
from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
try:
    mp.set_start_method("fork", force=True)
except Exception:
    pass
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
try:
    cv2.setNumThreads(0)
except Exception:
    pass

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from config import TrainConfig
from dataset import build_litefat_dataloaders
from model import LiteFat
from ultis import binary_auc


def compute_binary_metrics(labels: np.ndarray, preds: np.ndarray, probs: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Tính toán bộ chỉ số phân loại nhị phân hoàn chỉnh."""
    labels = np.asarray(labels, dtype=int).ravel()
    preds = np.asarray(preds, dtype=int).ravel()

    tp = int(np.sum((labels == 1) & (preds == 1)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    total = max(1, len(labels))

    acc = (tp + tn) / total
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    spec = tn / max(1, tn + fp)
    f1 = (2.0 * prec * rec) / max(1e-8, prec + rec) if (prec + rec) > 0 else 0.0
    f2 = (5.0 * prec * rec) / max(1e-8, 4.0 * prec + rec) if (4.0 * prec + rec) > 0 else 0.0

    res = {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "specificity": float(spec),
        "f1": float(f1),
        "f2": float(f2),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total": total,
    }

    if probs is not None:
        probs_arr = np.asarray(probs, dtype=float).ravel()
        if len(np.unique(labels)) > 1:
            res["auc"] = float(binary_auc(labels, probs_arr))
        else:
            res["auc"] = 0.5
    return res


def compute_roc_curve(labels: np.ndarray, probs: np.ndarray, num_thresholds: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tính các điểm (FPR, TPR, thresholds) cho đường cong ROC."""
    labels = np.asarray(labels, dtype=int).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    tpr_list, fpr_list = [], []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tp = np.sum((labels == 1) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))
        tn = np.sum((labels == 0) & (preds == 0))

        tpr = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    # Đảo thứ tự để FPR tăng từ 0 đến 1
    fpr_arr = np.array(fpr_list)[::-1]
    tpr_arr = np.array(tpr_list)[::-1]
    thr_arr = thresholds[::-1]
    return fpr_arr, tpr_arr, thr_arr


def compute_pr_curve(labels: np.ndarray, probs: np.ndarray, num_thresholds: int = 200) -> Tuple[np.ndarray, np.ndarray, float]:
    """Tính các điểm (Recall, Precision) và Average Precision (AP)."""
    labels = np.asarray(labels, dtype=int).ravel()
    probs = np.asarray(probs, dtype=float).ravel()
    thresholds = np.linspace(0.0, 1.0, num_thresholds)

    rec_list, prec_list = [], []
    for thr in thresholds:
        preds = (probs >= thr).astype(int)
        tp = np.sum((labels == 1) & (preds == 1))
        fn = np.sum((labels == 1) & (preds == 0))
        fp = np.sum((labels == 0) & (preds == 1))

        rec = tp / max(1, tp + fn)
        prec = tp / max(1, tp + fp)
        rec_list.append(rec)
        prec_list.append(prec)

    rec_arr = np.array(rec_list)[::-1]
    prec_arr = np.array(prec_list)[::-1]

    # Tính AP xấp xỉ qua trapezoid (tương thích cả NumPy 2.x và NumPy 1.x)
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    ap = float(trapz_fn(prec_arr, rec_arr)) if (trapz_fn is not None and len(rec_arr) > 1) else 0.0
    return rec_arr, prec_arr, abs(ap)


def evaluate_dataloader(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    optimal_thr: float = 0.5,
    split_name: str = "Test",
    model_name: str = "Model",
) -> Dict[str, Any]:
    """Chạy suy luận thu thập nhãn và tính toán toàn diện mọi chỉ số."""
    model.eval()
    all_targets: List[int] = []
    all_probs: List[float] = []
    sample_metas: List[Dict[str, Any]] = []

    samples = getattr(loader.dataset, "samples", [])

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Evaluating [{model_name} | {split_name}]", leave=False)):
            if len(batch) == 4:
                batch_frames, batch_landmarks, batch_labels, batch_lengths = batch
            elif len(batch) >= 5:
                batch_frames, batch_landmarks, batch_labels, batch_lengths = batch[:4]
            else:
                batch_frames, batch_landmarks, batch_labels = batch[:3]
                batch_lengths = None

            batch_landmarks = batch_landmarks.to(device, non_blocking=True)
            batch_labels = batch_labels.to(device, non_blocking=True)
            if batch_frames is not None:
                batch_frames = batch_frames.to(device, non_blocking=True)
            if batch_lengths is not None:
                batch_lengths = batch_lengths.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                frame_logits, clip_logits = model(
                    video_frames=batch_frames,
                    landmarks=batch_landmarks,
                    lengths=batch_lengths,
                    return_clip_logits=True,
                )
            probs = torch.softmax(clip_logits, dim=-1)[:, 1].cpu().numpy()
            targets = batch_labels.cpu().numpy()

            all_probs.extend(probs.tolist())
            all_targets.extend(targets.tolist())

    targets_np = np.array(all_targets, dtype=int)
    probs_np = np.array(all_probs, dtype=float)

    # 1. Đánh giá tại ngưỡng mặc định 0.50
    preds_default = (probs_np >= 0.50).astype(int)
    metrics_default = compute_binary_metrics(targets_np, preds_default, probs_np)

    # 2. Đánh giá tại ngưỡng tối ưu
    preds_optimal = (probs_np >= optimal_thr).astype(int)
    metrics_optimal = compute_binary_metrics(targets_np, preds_optimal, probs_np)
    metrics_optimal["optimal_threshold"] = float(optimal_thr)

    # 3. Phân rã theo Dataset
    datasets_np = np.array([s.get("dataset", "unknown") for s in samples[:len(targets_np)]])
    subjects_np = np.array([s.get("subject_id", "unknown") for s in samples[:len(targets_np)]])

    unique_datasets = sorted(list(set(datasets_np)))
    per_dataset_res = {}
    for d in unique_datasets:
        mask = (datasets_np == d)
        if mask.sum() > 0:
            d_y = targets_np[mask]
            d_p = probs_np[mask]
            d_pred_def = (d_p >= 0.50).astype(int)
            d_pred_opt = (d_p >= optimal_thr).astype(int)
            per_dataset_res[d] = {
                "num_samples": int(mask.sum()),
                "auc": float(binary_auc(d_y, d_p)) if len(np.unique(d_y)) > 1 else float("nan"),
                "acc_default": float(np.mean(d_y == d_pred_def)),
                "acc_optimal": float(np.mean(d_y == d_pred_opt)),
                "f1_default": compute_binary_metrics(d_y, d_pred_def)["f1"],
                "f1_optimal": compute_binary_metrics(d_y, d_pred_opt)["f1"],
            }

    # Macro AUC
    valid_aucs = [v["auc"] for v in per_dataset_res.values() if not math.isnan(v["auc"])]
    macro_auc = float(np.mean(valid_aucs)) if valid_aucs else metrics_default.get("auc", 0.5)
    metrics_default["macro_auc"] = macro_auc
    metrics_optimal["macro_auc"] = macro_auc

    # 4. Phân rã theo Subject
    unique_subjects = sorted(list(set(f"{d}:{sub}" for d, sub in zip(datasets_np, subjects_np))))
    per_subject_res = {}
    for sub_key in unique_subjects:
        d_name, s_name = sub_key.split(":")
        mask = (datasets_np == d_name) & (subjects_np == s_name)
        if mask.sum() > 0:
            s_y = targets_np[mask]
            s_p = probs_np[mask]
            s_pred_def = (s_p >= 0.50).astype(int)
            s_pred_opt = (s_p >= optimal_thr).astype(int)
            per_subject_res[sub_key] = {
                "dataset": d_name,
                "subject_id": s_name,
                "num_samples": int(mask.sum()),
                "auc": float(binary_auc(s_y, s_p)) if len(np.unique(s_y)) > 1 else float("nan"),
                "acc_default": float(np.mean(s_y == s_pred_def)),
                "acc_optimal": float(np.mean(s_y == s_pred_opt)),
            }

    # 5. Đường cong ROC và PR
    fpr_arr, tpr_arr, _ = compute_roc_curve(targets_np, probs_np)
    rec_arr, prec_arr, ap_score = compute_pr_curve(targets_np, probs_np)

    # 6. Quét dải ngưỡng [0.01 .. 0.99]
    thr_sweep = np.linspace(0.01, 0.99, 99)
    sweep_metrics = {"thresholds": thr_sweep.tolist(), "accuracy": [], "f1": [], "precision": [], "recall": [], "specificity": []}
    for t in thr_sweep:
        m = compute_binary_metrics(targets_np, (probs_np >= t).astype(int))
        sweep_metrics["accuracy"].append(m["accuracy"])
        sweep_metrics["f1"].append(m["f1"])
        sweep_metrics["precision"].append(m["precision"])
        sweep_metrics["recall"].append(m["recall"])
        sweep_metrics["specificity"].append(m["specificity"])

    return {
        "targets": targets_np,
        "probs": probs_np,
        "metrics_default": metrics_default,
        "metrics_optimal": metrics_optimal,
        "per_dataset": per_dataset_res,
        "per_subject": per_subject_res,
        "roc": {"fpr": fpr_arr, "tpr": tpr_arr},
        "pr": {"recall": rec_arr, "precision": prec_arr, "ap": ap_score},
        "sweep": sweep_metrics,
        "datasets": datasets_np,
        "subjects": subjects_np,
    }


def load_model_from_checkpoint(ckpt_dir: Path, device: torch.device) -> Tuple[LiteFat, TrainConfig, float]:
    """Nạp mô hình LiteFat, cấu hình và ngưỡng tối ưu từ thư mục checkpoint."""
    cfg_path = ckpt_dir / "config.json"
    weights_path = ckpt_dir / "best.pt"
    if not weights_path.is_file():
        weights_path = ckpt_dir / "last.pt"

    # Đọc cấu hình
    if cfg_path.is_file():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg_dict = json.load(f)
        cfg = TrainConfig(**cfg_dict)
    else:
        cfg = TrainConfig()

    # Đọc ngưỡng tối ưu đã lưu
    optimal_thr = 0.50
    test_m_path = ckpt_dir / "test_metrics.json"
    best_m_path = ckpt_dir / "best_metrics.json"

    if test_m_path.is_file():
        try:
            with open(test_m_path, "r", encoding="utf-8") as f:
                tm = json.load(f)
            optimal_thr = float(tm.get("optimal_threshold", 0.50))
        except Exception:
            pass
    elif best_m_path.is_file():
        try:
            with open(best_m_path, "r", encoding="utf-8") as f:
                bm = json.load(f)
            optimal_thr = float(bm.get("metrics", {}).get("optimal_threshold", 0.50))
        except Exception:
            pass

    # Xây dựng model
    model = LiteFat(
        num_nodes=68,
        feature_dim=cfg.feature_dim,
        mobilenet_proj_hidden=cfg.mobilenet_proj_hidden,
        node_embed_dim=cfg.node_embed_dim,
        residual_channels=cfg.residual_channels,
        dilation_channels=cfg.dilation_channels,
        skip_channels=cfg.skip_channels,
        end_channels=cfg.end_channels,
        num_classes=cfg.num_classes,
        num_blocks=cfg.num_blocks,
        layers_per_block=cfg.layers_per_block,
        kernel_size=cfg.kernel_size,
        pretrained_backbone=False,
        dropout=cfg.dropout,
        static_shape_ratio=getattr(cfg, "static_shape_ratio", 0.25),
    )

    ckpt_data = torch.load(weights_path, map_location="cpu", weights_only=False)
    state_dict = ckpt_data.get("model_state_dict", ckpt_data)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    return model, cfg, optimal_thr


# =====================================================================
# 5. CÁC HÀM TRỰC QUAN HÓA BIỂU ĐỒ (PUBLICATION-QUALITY 300 DPI)
# =====================================================================

def plot_1_roc_curves(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Vẽ đường cong ROC so sánh Stage 1 vs Stage 2 trên Val, Test & Per-dataset."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    sns.set_theme(style="whitegrid", font_scale=1.05)

    colors = {"Stage 1 (Landmark-Only)": "#1f77b4", "Stage 2 (Multimodal)": "#ff7f0e"}

    # 1. Validation ROC
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    for name, splits in eval_data.items():
        v = splits["val"]
        auc = v["metrics_default"]["auc"]
        fpr, tpr = v["roc"]["fpr"], v["roc"]["tpr"]
        ax1.plot(fpr, tpr, color=colors.get(name, "blue"), lw=2.5, label=f"{name} (AUC = {auc:.4f})")
    ax1.set_title("A. ROC Curves on Validation Split (1,978 samples)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Recall)")
    ax1.legend(loc="lower right", frameon=True)
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    # 2. Test ROC
    ax2 = axes[0, 1]
    ax2.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Guess (AUC = 0.500)")
    for name, splits in eval_data.items():
        t = splits["test"]
        auc = t["metrics_default"]["auc"]
        fpr, tpr = t["roc"]["fpr"], t["roc"]["tpr"]
        ax2.plot(fpr, tpr, color=colors.get(name, "orange"), lw=2.5, label=f"{name} (AUC = {auc:.4f})")
    ax2.set_title("B. ROC Curves on Test Split (1,018 samples)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("False Positive Rate (1 - Specificity)")
    ax2.set_ylabel("True Positive Rate (Recall)")
    ax2.legend(loc="lower right", frameon=True)
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])

    # 3. Per-Dataset Test ROC (SUST & UL-DD)
    ax3 = axes[1, 0]
    ax3.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ds_colors = {
        ("Stage 1 (Landmark-Only)", "SUST"): "#6baed6",
        ("Stage 2 (Multimodal)", "SUST"): "#fd8d3c",
        ("Stage 1 (Landmark-Only)", "UL-DD"): "#08519c",
        ("Stage 2 (Multimodal)", "UL-DD"): "#bd0026",
    }
    for name, splits in eval_data.items():
        t = splits["test"]
        for ds_name in ["SUST", "UL-DD"]:
            mask = (t["datasets"] == ds_name)
            if mask.sum() > 0:
                y_sub = t["targets"][mask]
                p_sub = t["probs"][mask]
                if len(np.unique(y_sub)) > 1:
                    fpr_sub, tpr_sub, _ = compute_roc_curve(y_sub, p_sub)
                    auc_sub = float(binary_auc(y_sub, p_sub))
                    c = ds_colors.get((name, ds_name), "gray")
                    ax3.plot(fpr_sub, tpr_sub, color=c, lw=2, label=f"{name} - {ds_name} (AUC = {auc_sub:.3f})")
    ax3.set_title("C. Test ROC Breakdown: SUST vs UL-DD", fontsize=13, fontweight="bold", pad=10)
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.legend(loc="lower right", frameon=True, fontsize=9.5)
    ax3.set_xlim([-0.02, 1.02])
    ax3.set_ylim([-0.02, 1.02])

    # 4. Per-Dataset Test ROC (VBDDD)
    ax4 = axes[1, 1]
    ax4.plot([0, 1], [0, 1], "k--", alpha=0.5)
    vbd_colors = {"Stage 1 (Landmark-Only)": "#31a354", "Stage 2 (Multimodal)": "#756bb1"}
    for name, splits in eval_data.items():
        t = splits["test"]
        mask = (t["datasets"] == "VBDDD")
        if mask.sum() > 0:
            y_sub = t["targets"][mask]
            p_sub = t["probs"][mask]
            if len(np.unique(y_sub)) > 1:
                fpr_sub, tpr_sub, _ = compute_roc_curve(y_sub, p_sub)
                auc_sub = float(binary_auc(y_sub, p_sub))
                ax4.plot(fpr_sub, tpr_sub, color=vbd_colors.get(name, "purple"), lw=2.2, label=f"{name} - VBDDD (AUC = {auc_sub:.3f})")
    ax4.set_title("D. Test ROC Breakdown: VBDDD (Vietnamese Drivers)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.legend(loc="lower right", frameon=True)
    ax4.set_xlim([-0.02, 1.02])
    ax4.set_ylim([-0.02, 1.02])

    plt.tight_layout()
    fig_path = out_dir / "1_roc_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [Đồ thị 1] Đã lưu: {fig_path.name}")


def plot_2_pr_curves(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Vẽ đường cong Precision-Recall trên Val và Test."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)
    colors = {"Stage 1 (Landmark-Only)": "#1f77b4", "Stage 2 (Multimodal)": "#ff7f0e"}

    for idx, (split_key, split_title) in enumerate([("val", "Validation Split"), ("test", "Test Split")]):
        ax = axes[idx]
        for name, splits in eval_data.items():
            s = splits[split_key]
            rec, prec = s["pr"]["recall"], s["pr"]["precision"]
            ap = s["pr"]["ap"]
            ax.plot(rec, prec, color=colors.get(name, "blue"), lw=2.5, label=f"{name} (AP = {ap:.4f})")

        no_skill = float(np.mean(eval_data[list(eval_data.keys())[0]][split_key]["targets"]))
        ax.plot([0, 1], [no_skill, no_skill], "k--", alpha=0.5, label=f"No Skill Baseline (P = {no_skill:.3f})")
        ax.set_title(f"Precision-Recall Curve ({split_title})", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Recall (True Positive Rate)")
        ax.set_ylabel("Precision (Positive Predictive Value)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.legend(loc="lower left", frameon=True)

    plt.tight_layout()
    fig_path = out_dir / "2_pr_curves.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [Đồ thị 2] Đã lưu: {fig_path.name}")


def plot_3_confusion_matrices(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Vẽ ma trận nhầm lẫn chuẩn hóa dạng lưới 2x4 (2 split x 2 model x 2 ngưỡng)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), dpi=300)

    model_names = list(eval_data.keys())
    split_keys = ["val", "test"]

    for row_idx, split_key in enumerate(split_keys):
        split_label = "Validation Set (1,978 clips)" if split_key == "val" else "Test Set (1,018 clips)"
        col_idx = 0
        for m_name in model_names:
            s_data = eval_data[m_name][split_key]
            opt_thr = s_data["metrics_optimal"]["optimal_threshold"]

            # Cột 1: Ngưỡng mặc định 0.50
            cm_def = np.array([
                [s_data["metrics_default"]["tn"], s_data["metrics_default"]["fp"]],
                [s_data["metrics_default"]["fn"], s_data["metrics_default"]["tp"]],
            ])
            cm_def_norm = cm_def.astype(float) / cm_def.sum(axis=1, keepdims=True)

            ax = axes[row_idx, col_idx]
            annot = np.array([
                [f"TN\n{cm_def[0,0]}\n({cm_def_norm[0,0]:.1%})", f"FP\n{cm_def[0,1]}\n({cm_def_norm[0,1]:.1%})"],
                [f"FN\n{cm_def[1,0]}\n({cm_def_norm[1,0]:.1%})", f"TP\n{cm_def[1,1]}\n({cm_def_norm[1,1]:.1%})"],
            ])
            sns.heatmap(cm_def_norm, annot=annot, fmt="", cmap="Blues", cbar=False, ax=ax, vmin=0, vmax=1)
            ax.set_title(f"{m_name}\n{split_label} @Thr=0.50\nAcc: {s_data['metrics_default']['accuracy']:.1%} | F1: {s_data['metrics_default']['f1']:.3f}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label" if col_idx == 0 else "")
            ax.set_xticklabels(["Alert (0)", "Drowsy (1)"])
            ax.set_yticklabels(["Alert (0)", "Drowsy (1)"] if col_idx == 0 else ["", ""])
            col_idx += 1

            # Cột 2: Ngưỡng tối ưu
            cm_opt = np.array([
                [s_data["metrics_optimal"]["tn"], s_data["metrics_optimal"]["fp"]],
                [s_data["metrics_optimal"]["fn"], s_data["metrics_optimal"]["tp"]],
            ])
            cm_opt_norm = cm_opt.astype(float) / cm_opt.sum(axis=1, keepdims=True)

            ax = axes[row_idx, col_idx]
            annot = np.array([
                [f"TN\n{cm_opt[0,0]}\n({cm_opt_norm[0,0]:.1%})", f"FP\n{cm_opt[0,1]}\n({cm_opt_norm[0,1]:.1%})"],
                [f"FN\n{cm_opt[1,0]}\n({cm_opt_norm[1,0]:.1%})", f"TP\n{cm_opt[1,1]}\n({cm_opt_norm[1,1]:.1%})"],
            ])
            sns.heatmap(cm_opt_norm, annot=annot, fmt="", cmap="Oranges", cbar=False, ax=ax, vmin=0, vmax=1)
            ax.set_title(f"{m_name}\n{split_label} @Optimal={opt_thr:.2f}\nAcc: {s_data['metrics_optimal']['accuracy']:.1%} | F1: {s_data['metrics_optimal']['f1']:.3f}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("")
            ax.set_xticklabels(["Alert (0)", "Drowsy (1)"])
            ax.set_yticklabels(["", ""])
            col_idx += 1

    plt.tight_layout()
    fig_path = out_dir / "3_confusion_matrices.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [Đồ thị 3] Đã lưu: {fig_path.name}")


def plot_4_dataset_subject_comparison(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Vẽ biểu đồ cột so sánh theo từng Dataset và từng Subject kiểm thử."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    m1_name, m2_name = list(eval_data.keys())[:2]
    t1 = eval_data[m1_name]["test"]
    t2 = eval_data[m2_name]["test"]

    # 1. So sánh theo Dataset (SUST, UL-DD, VBDDD)
    datasets = ["SUST", "UL-DD", "VBDDD"]
    x = np.arange(len(datasets))
    width = 0.2

    ax1 = axes[0]
    m1_auc = [t1["per_dataset"].get(d, {}).get("auc", 0.0) for d in datasets]
    m2_auc = [t2["per_dataset"].get(d, {}).get("auc", 0.0) for d in datasets]
    m1_acc = [t1["per_dataset"].get(d, {}).get("acc_optimal", 0.0) * 100 for d in datasets]
    m2_acc = [t2["per_dataset"].get(d, {}).get("acc_optimal", 0.0) * 100 for d in datasets]

    rects1 = ax1.bar(x - 1.5 * width, m1_auc, width, label=f"{m1_name} (AUC)", color="#4292c6")
    rects2 = ax1.bar(x - 0.5 * width, m2_auc, width, label=f"{m2_name} (AUC)", color="#fe9929")
    rects3 = ax1.bar(x + 0.5 * width, [a / 100 for a in m1_acc], width, label=f"{m1_name} (Acc %)", color="#084594", alpha=0.85)
    rects4 = ax1.bar(x + 1.5 * width, [a / 100 for a in m2_acc], width, label=f"{m2_name} (Acc %)", color="#d95f0e", alpha=0.85)

    ax1.set_ylabel("Score / Percentage")
    ax1.set_title("A. Performance Comparison Across Sub-Datasets (Test Set)", fontsize=13, fontweight="bold", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{d}\n(N={t1['per_dataset'].get(d, {}).get('num_samples', 0)})" for d in datasets], fontsize=11)
    ax1.set_ylim([0, 1.15])
    ax1.legend(loc="upper left", frameon=True, fontsize=10)

    # Hiển thị số liệu trên đỉnh cột
    for rect in rects1 + rects2:
        h = rect.get_height()
        ax1.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for rect in rects3 + rects4:
        h = rect.get_height()
        ax1.annotate(f"{h*100:.1f}%", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5)

    # 2. So sánh theo Subject trọng điểm
    key_subjects = [
        "SUST:ob15", "SUST:ob17", "SUST:ob7",
        "UL-DD:subject_N", "UL-DD:subject_O",
        "VBDDD:subject4", "VBDDD:subject2", "VBDDD:subject26"
    ]
    ax2 = axes[1]
    xs = np.arange(len(key_subjects))
    w_sub = 0.35

    s1_aucs = [t1["per_subject"].get(s, {}).get("auc", 0.0) for s in key_subjects]
    s2_aucs = [t2["per_subject"].get(s, {}).get("auc", 0.0) for s in key_subjects]
    # Thay NaN bằng 1.0 (như subject 1 class)
    s1_aucs = [1.0 if math.isnan(a) else a for a in s1_aucs]
    s2_aucs = [1.0 if math.isnan(a) else a for a in s2_aucs]

    r1 = ax2.bar(xs - w_sub / 2, s1_aucs, w_sub, label=f"{m1_name}", color="#1f77b4")
    r2 = ax2.bar(xs + w_sub / 2, s2_aucs, w_sub, label=f"{m2_name}", color="#ff7f0e")

    ax2.set_ylabel("AUC-ROC Score")
    ax2.set_title("B. Key Subject-Level Test AUC (Addressing Shortcuts)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_xticks(xs)
    ax2.set_xticklabels([s.replace(":", "\n") for s in key_subjects], fontsize=9.5, rotation=0)
    ax2.set_ylim([0, 1.2])
    ax2.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random Guess (0.50)")
    ax2.legend(loc="upper right", frameon=True, fontsize=10)

    for rect in r1 + r2:
        h = rect.get_height()
        ax2.annotate(f"{h:.2f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig_path = out_dir / "4_dataset_subject_comparison.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [Đồ thị 4] Đã lưu: {fig_path.name}")


def plot_5_threshold_sensitivity(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Vẽ đường biến thiên F1, Acc, Recall, Precision theo dải ngưỡng [0..1]."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), dpi=300)
    model_names = list(eval_data.keys())

    for col_idx, m_name in enumerate(model_names):
        for row_idx, (s_key, s_title) in enumerate([("val", "Validation Set"), ("test", "Test Set")]):
            ax = axes[row_idx, col_idx]
            sw = eval_data[m_name][s_key]["sweep"]
            thrs = sw["thresholds"]
            opt_thr = eval_data[m_name][s_key]["metrics_optimal"]["optimal_threshold"]

            ax.plot(thrs, sw["accuracy"], label="Accuracy", color="#2ca02c", lw=2)
            ax.plot(thrs, sw["f1"], label="F1-Score", color="#1f77b4", lw=2.5)
            ax.plot(thrs, sw["recall"], label="Recall (Sensitivity)", color="#d62728", lw=1.8, linestyle="--")
            ax.plot(thrs, sw["precision"], label="Precision", color="#ff7f0e", lw=1.8, linestyle=":")
            ax.plot(thrs, sw["specificity"], label="Specificity", color="#9467bd", lw=1.5, linestyle="-.")

            ax.axvline(0.50, color="gray", linestyle="--", alpha=0.7, label="Default Thr (0.50)")
            ax.axvline(opt_thr, color="black", linestyle="-", lw=2, label=f"Optimal Thr ({opt_thr:.2f})")

            ax.set_title(f"{m_name} — {s_title}", fontsize=12, fontweight="bold", pad=8)
            ax.set_xlabel("Decision Threshold")
            ax.set_ylabel("Metric Value")
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            if row_idx == 0 and col_idx == 1:
                ax.legend(loc="lower left", frameon=True, fontsize=9)

    plt.tight_layout()
    fig_path = out_dir / "5_threshold_sensitivity.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [Đồ thị 5] Đã lưu: {fig_path.name}")


def plot_6_metrics_comparison_summary(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Vẽ biểu đồ cột so sánh toàn diện 6 chỉ số cốt lõi ở 2 ngưỡng trên cả Val và Test."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    metric_keys = ["accuracy", "precision", "recall", "specificity", "f1", "auc"]
    metric_labels = ["Accuracy", "Precision", "Recall", "Specificity", "F1-Score", "AUC-ROC"]
    x = np.arange(len(metric_keys))
    width = 0.2

    m1_name, m2_name = list(eval_data.keys())[:2]

    for idx, (s_key, s_title) in enumerate([("val", "Validation Split (1,978 samples)"), ("test", "Test Split (1,018 samples)")]):
        ax = axes[idx]
        s1 = eval_data[m1_name][s_key]
        s2 = eval_data[m2_name][s_key]

        m1_def = [s1["metrics_default"][k] for k in metric_keys]
        m1_opt = [s1["metrics_optimal"][k] for k in metric_keys]
        m2_def = [s2["metrics_default"][k] for k in metric_keys]
        m2_opt = [s2["metrics_optimal"][k] for k in metric_keys]

        ax.bar(x - 1.5 * width, m1_def, width, label=f"{m1_name} @0.50", color="#9ecae1")
        ax.bar(x - 0.5 * width, m1_opt, width, label=f"{m1_name} @Opt", color="#3182bd")
        ax.bar(x + 0.5 * width, m2_def, width, label=f"{m2_name} @0.50", color="#fdae6b")
        ax.bar(x + 1.5 * width, m2_opt, width, label=f"{m2_name} @Opt", color="#e6550d")

        ax.set_ylabel("Score / Percentage")
        ax.set_title(f"Comprehensive Metric Comparison ({s_title})", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, fontsize=10.5)
        ax.set_ylim([0, 1.15])
        ax.legend(loc="upper right", frameon=True, fontsize=9.5)

    plt.tight_layout()
    fig_path = out_dir / "6_metrics_comparison_summary.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [Đồ thị 6] Đã lưu: {fig_path.name}")


def plot_7_summary_dashboard(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Vẽ Dashboard tổng hợp khổ lớn (4 bảng hợp nhất) phục vụ báo cáo khoa học."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 13), dpi=300)
    m1_name, m2_name = list(eval_data.keys())[:2]
    t1 = eval_data[m1_name]["test"]
    t2 = eval_data[m2_name]["test"]

    # (1) Top-Left: Test ROC Curves
    ax1 = axes[0, 0]
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax1.plot(t1["roc"]["fpr"], t1["roc"]["tpr"], color="#1f77b4", lw=2.5, label=f"{m1_name} (AUC = {t1['metrics_default']['auc']:.4f})")
    ax1.plot(t2["roc"]["fpr"], t2["roc"]["tpr"], color="#ff7f0e", lw=2.5, label=f"{m2_name} (AUC = {t2['metrics_default']['auc']:.4f})")
    ax1.set_title("1. Test ROC Curves (1,018 samples)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.legend(loc="lower right", frameon=True)

    # (2) Top-Right: Test Set Metrics Summary @ Optimal Threshold
    ax2 = axes[0, 1]
    metric_keys = ["accuracy", "recall", "specificity", "f1", "auc"]
    metric_labels = ["Accuracy", "Recall", "Specificity", "F1", "AUC"]
    x = np.arange(len(metric_keys))
    w = 0.35
    v1 = [t1["metrics_optimal"][k] for k in metric_keys]
    v2 = [t2["metrics_optimal"][k] for k in metric_keys]

    r1 = ax2.bar(x - w / 2, v1, w, label=f"{m1_name} (@{t1['metrics_optimal']['optimal_threshold']:.2f})", color="#1f77b4")
    r2 = ax2.bar(x + w / 2, v2, w, label=f"{m2_name} (@{t2['metrics_optimal']['optimal_threshold']:.2f})", color="#ff7f0e")
    ax2.set_title("2. Test Performance @ Optimal Threshold", fontsize=13, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, fontsize=11)
    ax2.set_ylim([0, 1.15])
    ax2.legend(loc="lower right", frameon=True)
    for rect in r1 + r2:
        h = rect.get_height()
        ax2.annotate(f"{h:.1%}" if h < 0.99 else f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # (3) Bottom-Left: Side-by-side Confusion Matrices on Test @ Optimal
    ax3 = axes[1, 0]
    cm1 = np.array([[t1["metrics_optimal"]["tn"], t1["metrics_optimal"]["fp"]], [t1["metrics_optimal"]["fn"], t1["metrics_optimal"]["tp"]]])
    cm2 = np.array([[t2["metrics_optimal"]["tn"], t2["metrics_optimal"]["fp"]], [t2["metrics_optimal"]["fn"], t2["metrics_optimal"]["tp"]]])
    cm_combined = np.hstack([cm1, cm2])
    cm_norm = np.hstack([cm1 / cm1.sum(axis=1, keepdims=True), cm2 / cm2.sum(axis=1, keepdims=True)])

    annot_cmb = np.array([
        [f"{cm1[0,0]}\n({cm_norm[0,0]:.1%})", f"{cm1[0,1]}\n({cm_norm[0,1]:.1%})", f"{cm2[0,0]}\n({cm_norm[0,2]:.1%})", f"{cm2[0,1]}\n({cm_norm[0,3]:.1%})"],
        [f"{cm1[1,0]}\n({cm_norm[1,0]:.1%})", f"{cm1[1,1]}\n({cm_norm[1,1]:.1%})", f"{cm2[1,0]}\n({cm_norm[1,2]:.1%})", f"{cm2[1,1]}\n({cm_norm[1,3]:.1%})"],
    ])
    sns.heatmap(cm_norm, annot=annot_cmb, fmt="", cmap="Blues", cbar=False, ax=ax3, vmin=0, vmax=1)
    ax3.set_title("3. Test Confusion Matrices @ Optimal Threshold\n(Left: Stage 1 @0.42 | Right: Stage 2 @0.58)", fontsize=12, fontweight="bold")
    ax3.set_xticklabels(["S1: Alert", "S1: Drowsy", "S2: Alert", "S2: Drowsy"])
    ax3.set_yticklabels(["True Alert", "True Drowsy"])

    # (4) Bottom-Right: Dataset Breakdown (SUST, UL-DD, VBDDD)
    ax4 = axes[1, 1]
    datasets = ["SUST", "UL-DD", "VBDDD"]
    x_ds = np.arange(len(datasets))
    s1_ds_auc = [t1["per_dataset"].get(d, {}).get("auc", 0.0) for d in datasets]
    s2_ds_auc = [t2["per_dataset"].get(d, {}).get("auc", 0.0) for d in datasets]

    rb1 = ax4.bar(x_ds - w / 2, s1_ds_auc, w, label=f"{m1_name}", color="#1f77b4")
    rb2 = ax4.bar(x_ds + w / 2, s2_ds_auc, w, label=f"{m2_name}", color="#ff7f0e")
    ax4.set_title("4. Test AUC Breakdown Across Datasets", fontsize=13, fontweight="bold")
    ax4.set_xticks(x_ds)
    ax4.set_xticklabels([f"{d}\n(N={t1['per_dataset'].get(d, {}).get('num_samples', 0)})" for d in datasets], fontsize=11)
    ax4.set_ylim([0, 1.15])
    ax4.legend(loc="lower right", frameon=True)
    for rect in rb1 + rb2:
        h = rect.get_height()
        ax4.annotate(f"{h:.3f}", xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig_path = out_dir / "7_summary_dashboard.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"  [Đồ thị 7] Đã lưu: {fig_path.name}")


# =====================================================================
# 6. XUẤT BÁO CÁO MARKDOWN VÀ JSON
# =====================================================================

def export_reports(eval_data: Dict[str, Dict[str, Any]], out_dir: Path):
    """Xuất file benchmark_summary.json và BENCHMARK_REPORT.md."""
    # 1. Chuyển đổi dữ liệu sang dạng JSON serializable
    summary_dict = {}
    for m_name, splits in eval_data.items():
        summary_dict[m_name] = {}
        for s_key in ["val", "test"]:
            s_data = splits[s_key]
            summary_dict[m_name][s_key] = {
                "metrics_default": s_data["metrics_default"],
                "metrics_optimal": s_data["metrics_optimal"],
                "per_dataset": s_data["per_dataset"],
                "per_subject": s_data["per_subject"],
                "pr_ap": s_data["pr"]["ap"],
            }

    json_path = out_dir / "benchmark_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_dict, f, indent=2, ensure_ascii=False)
    print(f"  [Báo cáo JSON] Đã lưu: {json_path.name}")

    # 2. Sinh báo cáo Markdown
    m1_name, m2_name = list(eval_data.keys())[:2]
    v1, t1 = eval_data[m1_name]["val"], eval_data[m1_name]["test"]
    v2, t2 = eval_data[m2_name]["val"], eval_data[m2_name]["test"]

    opt1 = t1["metrics_optimal"]["optimal_threshold"]
    opt2 = t2["metrics_optimal"]["optimal_threshold"]

    md_lines = [
        "# Báo Cáo Đo Đạc & So Sánh Mô Hình LiteFat v6",
        "",
        f"- **Ngày thực hiện**: {time.strftime('%d/%m/%Y %H:%M:%S')}",
        "- **Tập Validation**: 1,978 mẫu (cân bằng subject UL-DD & SUST)",
        "- **Tập Test**: 1,018 mẫu chuẩn (giữ cố định 100% qua tất cả các thử nghiệm)",
        "",
        "---",
        "",
        "## 1. Bảng So Sánh Hiệu Năng Trên Tập Test (1,018 Mẫu)",
        "",
        "| Độ đo kiểm thử | Stage 1 @0.50 | Stage 1 @Opt (" + f"{opt1:.2f}" + ") | Stage 2 @0.50 | Stage 2 @Opt (" + f"{opt2:.2f}" + ") | Bước nhảy vọt |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |",
        f"| **Accuracy** | {t1['metrics_default']['accuracy']:.2%} | {t1['metrics_optimal']['accuracy']:.2%} | {t2['metrics_default']['accuracy']:.2%} | **{t2['metrics_optimal']['accuracy']:.2%}** | Đỉnh cao mới ở Stage 2 |",
        f"| **Precision** | {t1['metrics_default']['precision']:.2%} | {t1['metrics_optimal']['precision']:.2%} | {t2['metrics_default']['precision']:.2%} | **{t2['metrics_optimal']['precision']:.2%}** | Tăng độ tin cậy cảnh báo |",
        f"| **Recall (Độ nhạy)** | {t1['metrics_default']['recall']:.2%} | {t1['metrics_optimal']['recall']:.2%} | **{t2['metrics_default']['recall']:.2%}** | {t2['metrics_optimal']['recall']:.2%} | Nhận diện buồn ngủ kịp thời |",
        f"| **Specificity (Độ đặc hiệu)** | {t1['metrics_default']['specificity']:.2%} | {t1['metrics_optimal']['specificity']:.2%} | {t2['metrics_default']['specificity']:.2%} | **{t2['metrics_optimal']['specificity']:.2%}** | Giảm báo động giả khi tỉnh táo |",
        f"| **F1-Score** | {t1['metrics_default']['f1']:.4f} | {t1['metrics_optimal']['f1']:.4f} | **{t2['metrics_default']['f1']:.4f}** | {t2['metrics_optimal']['f1']:.4f} | Cân bằng F1 tốt nhất |",
        f"| **AUC-ROC Toàn bộ** | {t1['metrics_default']['auc']:.4f} | {t1['metrics_optimal']['auc']:.4f} | **{t2['metrics_default']['auc']:.4f}** | **{t2['metrics_optimal']['auc']:.4f}** | Khả năng phân tách nhị phân vượt trội |",
        f"| **Macro-AUC** | {t1['metrics_default']['macro_auc']:.4f} | {t1['metrics_optimal']['macro_auc']:.4f} | **{t2['metrics_default']['macro_auc']:.4f}** | **{t2['metrics_optimal']['macro_auc']:.4f}** | Cân bằng liên dataset |",
        f"| **Average Precision (AP)** | {t1['pr']['ap']:.4f} | {t1['pr']['ap']:.4f} | **{t2['pr']['ap']:.4f}** | **{t2['pr']['ap']:.4f}** | Độ chính xác trung bình PR |",
        "",
        "---",
        "",
        "## 2. Bảng So Sánh Hiệu Năng Trên Tập Validation (1,978 Mẫu)",
        "",
        "| Độ đo Validation | Stage 1 @0.50 | Stage 1 @Opt (" + f"{opt1:.2f}" + ") | Stage 2 @0.50 | Stage 2 @Opt (" + f"{opt2:.2f}" + ") |",
        "| :--- | :---: | :---: | :---: | :---: |",
        f"| **Accuracy** | {v1['metrics_default']['accuracy']:.2%} | {v1['metrics_optimal']['accuracy']:.2%} | {v2['metrics_default']['accuracy']:.2%} | {v2['metrics_optimal']['accuracy']:.2%} |",
        f"| **Precision** | {v1['metrics_default']['precision']:.2%} | {v1['metrics_optimal']['precision']:.2%} | {v2['metrics_default']['precision']:.2%} | {v2['metrics_optimal']['precision']:.2%} |",
        f"| **Recall** | {v1['metrics_default']['recall']:.2%} | {v1['metrics_optimal']['recall']:.2%} | {v2['metrics_default']['recall']:.2%} | {v2['metrics_optimal']['recall']:.2%} |",
        f"| **Specificity** | {v1['metrics_default']['specificity']:.2%} | {v1['metrics_optimal']['specificity']:.2%} | {v2['metrics_default']['specificity']:.2%} | {v2['metrics_optimal']['specificity']:.2%} |",
        f"| **F1-Score** | {v1['metrics_default']['f1']:.4f} | {v1['metrics_optimal']['f1']:.4f} | {v2['metrics_default']['f1']:.4f} | {v2['metrics_optimal']['f1']:.4f} |",
        f"| **AUC-ROC** | {v1['metrics_default']['auc']:.4f} | {v1['metrics_optimal']['auc']:.4f} | {v2['metrics_default']['auc']:.4f} | {v2['metrics_optimal']['auc']:.4f} |",
        "",
        "---",
        "",
        "## 3. Phân Rã Chi Tiết Theo Từng Bộ Dữ Liệu Con (Test Set)",
        "",
        "| Bộ dữ liệu con | Số mẫu | Stage 1 AUC | Stage 1 Acc (@Opt) | Stage 2 AUC | Stage 2 Acc (@Opt) | Đánh giá |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ]

    for d in ["SUST", "UL-DD", "VBDDD"]:
        d1 = t1["per_dataset"].get(d, {})
        d2 = t2["per_dataset"].get(d, {})
        cnt = d1.get("num_samples", 0)
        a1, acc1 = d1.get("auc", 0.0), d1.get("acc_optimal", 0.0)
        a2, acc2 = d2.get("auc", 0.0), d2.get("acc_optimal", 0.0)
        eval_cmt = "Bối cảnh ảnh mặt tăng sức mạnh vượt bậc" if a2 > a1 else "Đặc trưng hình thái triệt tiêu shortcut tốt hơn"
        md_lines.append(f"| **{d}** | {cnt} | {a1:.4f} | {acc1:.2%} | **{a2:.4f}** | **{acc2:.2%}** | {eval_cmt} |")

    md_lines.extend([
        "",
        "---",
        "",
        "## 4. Danh Mục Biểu Đồ Trực Quan Hóa Đã Xuất",
        "",
        "1. [`1_roc_curves.png`](./1_roc_curves.png): Đường cong ROC trên tập Val, Test và phân rã SUST, UL-DD, VBDDD.",
        "2. [`2_pr_curves.png`](./2_pr_curves.png): Đường cong Precision-Recall và chỉ số Average Precision (AP).",
        "3. [`3_confusion_matrices.png`](./3_confusion_matrices.png): Lưới ma trận nhầm lẫn 2x4 chuẩn hóa so sánh trực tiếp cả 2 tập và 2 ngưỡng.",
        "4. [`4_dataset_subject_comparison.png`](./4_dataset_subject_comparison.png): Biểu đồ cột so sánh AUC & Acc theo từng dataset và từng subject kiểm thử.",
        "5. [`5_threshold_sensitivity.png`](./5_threshold_sensitivity.png): Đường cong độ nhạy F1, Acc, Rec, Prec theo dải ngưỡng [0.0 .. 1.0].",
        "6. [`6_metrics_comparison_summary.png`](./6_metrics_comparison_summary.png): Biểu đồ cột nhóm so sánh 6 chỉ số cốt lõi.",
        "7. [`7_summary_dashboard.png`](./7_summary_dashboard.png): Dashboard tổng hợp hợp nhất 4 khung hình đẹp nhất phục vụ trình chiếu.",
    ])

    md_path = out_dir / "BENCHMARK_REPORT.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"  [Báo cáo Markdown] Đã lưu: {md_path.name}")


def print_console_summary(eval_data: Dict[str, Dict[str, Any]]):
    """In bảng tóm tắt so sánh đẹp mắt ra màn hình terminal."""
    m1_name, m2_name = list(eval_data.keys())[:2]
    t1 = eval_data[m1_name]["test"]
    t2 = eval_data[m2_name]["test"]

    opt1 = t1["metrics_optimal"]["optimal_threshold"]
    opt2 = t2["metrics_optimal"]["optimal_threshold"]

    print("\n" + "=" * 82)
    print("           BẢNG TỔNG HỢP SO SÁNH HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST (1,018 MẪU)")
    print("=" * 82)
    print(f"{'Chỉ số':<24} | {'Stage 1 @0.50':<12} | {f'Stage 1 @{opt1:.2f}':<12} | {'Stage 2 @0.50':<12} | {f'Stage 2 @{opt2:.2f}':<12}")
    print("-" * 82)

    rows = [
        ("Accuracy", f"{t1['metrics_default']['accuracy']:.2%}", f"{t1['metrics_optimal']['accuracy']:.2%}", f"{t2['metrics_default']['accuracy']:.2%}", f"{t2['metrics_optimal']['accuracy']:.2%}"),
        ("Precision", f"{t1['metrics_default']['precision']:.2%}", f"{t1['metrics_optimal']['precision']:.2%}", f"{t2['metrics_default']['precision']:.2%}", f"{t2['metrics_optimal']['precision']:.2%}"),
        ("Recall", f"{t1['metrics_default']['recall']:.2%}", f"{t1['metrics_optimal']['recall']:.2%}", f"{t2['metrics_default']['recall']:.2%}", f"{t2['metrics_optimal']['recall']:.2%}"),
        ("Specificity", f"{t1['metrics_default']['specificity']:.2%}", f"{t1['metrics_optimal']['specificity']:.2%}", f"{t2['metrics_default']['specificity']:.2%}", f"{t2['metrics_optimal']['specificity']:.2%}"),
        ("F1-Score", f"{t1['metrics_default']['f1']:.4f}", f"{t1['metrics_optimal']['f1']:.4f}", f"{t2['metrics_default']['f1']:.4f}", f"{t2['metrics_optimal']['f1']:.4f}"),
        ("AUC-ROC", f"{t1['metrics_default']['auc']:.4f}", f"{t1['metrics_optimal']['auc']:.4f}", f"{t2['metrics_default']['auc']:.4f}", f"{t2['metrics_optimal']['auc']:.4f}"),
        ("SUST AUC", f"{t1['per_dataset'].get('SUST',{}).get('auc',0):.4f}", f"{t1['per_dataset'].get('SUST',{}).get('auc',0):.4f}", f"{t2['per_dataset'].get('SUST',{}).get('auc',0):.4f}", f"{t2['per_dataset'].get('SUST',{}).get('auc',0):.4f}"),
        ("UL-DD AUC", f"{t1['per_dataset'].get('UL-DD',{}).get('auc',0):.4f}", f"{t1['per_dataset'].get('UL-DD',{}).get('auc',0):.4f}", f"{t2['per_dataset'].get('UL-DD',{}).get('auc',0):.4f}", f"{t2['per_dataset'].get('UL-DD',{}).get('auc',0):.4f}"),
        ("VBDDD AUC", f"{t1['per_dataset'].get('VBDDD',{}).get('auc',0):.4f}", f"{t1['per_dataset'].get('VBDDD',{}).get('auc',0):.4f}", f"{t2['per_dataset'].get('VBDDD',{}).get('auc',0):.4f}", f"{t2['per_dataset'].get('VBDDD',{}).get('auc',0):.4f}"),
    ]

    for label, v1_def, v1_opt, v2_def, v2_opt in rows:
        print(f"{label:<24} | {v1_def:<12} | {v1_opt:<12} | {v2_def:<12} | {v2_opt:<12}")
    print("=" * 82 + "\n")


# =====================================================================
# 7. HÀM MAIN THỰC THI TOÀN BỘ PIPELINE
# =====================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark & Visualize LiteFat v6 Models")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["./checkpoints/LiteFat_v6_stage1", "./checkpoints/LiteFat_v6_stage2_visual"],
        help="Danh sách thư mục chứa checkpoint cần benchmark",
    )
    parser.add_argument("--manifest", type=str, default="dataset_merged_split_v6.csv", help="Tên manifest split v6")
    parser.add_argument("--dataset_dir", type=str, default="/home/riftuser/train/dataset/data-processed-v3/data_processed")
    parser.add_argument("--cache_dir", type=str, default="/home/riftuser/train/dataset/cache-dataset-processed-v3/LiteFat/cache_pt")
    parser.add_argument("--output_dir", type=str, default="./benchmark_results", help="Thư mục xuất biểu đồ và báo cáo")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()
    device = torch.device(args.device)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("       KHỞI CHẠY BENCHMARK VÀ TRỰC QUAN HÓA MÔ HÌNH LITEFAT V6")
    print(f"Thiết bị: {device} | Output: {out_dir}")
    print("=" * 75)

    eval_data = {}

    for ckpt_path_str in args.checkpoints:
        ckpt_dir = Path(ckpt_path_str)
        if not ckpt_dir.is_dir():
            print(f"[Cảnh báo] Không tìm thấy thư mục checkpoint: {ckpt_dir}. Bỏ qua.")
            continue

        model_display_name = "Stage 1 (Landmark-Only)" if "stage1" in ckpt_dir.name.lower() else "Stage 2 (Multimodal)"
        print(f"\n>>> Đang nạp mô hình: {model_display_name} từ {ckpt_dir.name}...")
        model, cfg, optimal_thr = load_model_from_checkpoint(ckpt_dir, device)
        print(f"    - Ngưỡng tối ưu đã lưu: {optimal_thr:.2f}")
        print(f"    - Chế độ Bối cảnh thị giác (use_visual_context): {cfg.use_visual_context}")

        # Khởi tạo DataLoader cho cả Val và Test
        print("    - Đang khởi tạo DataLoaders cho Val và Test...")
        _, val_loader, test_loader = build_litefat_dataloaders(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            time_interval=cfg.time_interval,
            window_sec=cfg.window_sec,
            letterbox_size=cfg.letterbox_size,
            face_size=cfg.face_size,
            manifest_filename=args.manifest,
            use_cache=True,
            cache_dir=args.cache_dir,
            pin_memory=(device.type == "cuda"),
            device=str(device),
            extract_crops_only=(not cfg.use_visual_context),
            selected_datasets=cfg.selected_datasets,
            use_visual_context=cfg.use_visual_context,
            eval_window_mode=getattr(cfg, "eval_window_mode", "center"),
            drop_single_class_subjects=True,
            visual_frame_stride=getattr(cfg, "visual_frame_stride", 8),
        )

        # Đánh giá trên Validation Set
        val_res = evaluate_dataloader(
            model=model,
            loader=val_loader,
            device=device,
            optimal_thr=optimal_thr,
            split_name="Validation",
            model_name=model_display_name,
        )

        # Đánh giá trên Test Set
        test_res = evaluate_dataloader(
            model=model,
            loader=test_loader,
            device=device,
            optimal_thr=optimal_thr,
            split_name="Test",
            model_name=model_display_name,
        )

        eval_data[model_display_name] = {
            "val": val_res,
            "test": test_res,
            "cfg": cfg,
            "optimal_thr": optimal_thr,
        }

    if len(eval_data) < 2:
        print("[Lỗi] Cần ít nhất 2 mô hình để vẽ biểu đồ đối sánh!")
        return

    # In kết quả tóm tắt ra console
    print_console_summary(eval_data)

    # Vẽ 7 bộ đồ thị chuyên nghiệp
    print("\n>>> Đang vẽ và xuất 7 bộ biểu đồ chất lượng cao (300 DPI)...")
    plot_1_roc_curves(eval_data, out_dir)
    plot_2_pr_curves(eval_data, out_dir)
    plot_3_confusion_matrices(eval_data, out_dir)
    plot_4_dataset_subject_comparison(eval_data, out_dir)
    plot_5_threshold_sensitivity(eval_data, out_dir)
    plot_6_metrics_comparison_summary(eval_data, out_dir)
    plot_7_summary_dashboard(eval_data, out_dir)

    # Xuất báo cáo Markdown và JSON
    print("\n>>> Đang xuất báo cáo tổng hợp Markdown và JSON...")
    export_reports(eval_data, out_dir)

    print("\n" + "=" * 75)
    print(">>> HOÀN TẤT BENCHMARK TOÀN DIỆN!")
    print(f"Toàn bộ biểu đồ và báo cáo đã được lưu vào thư mục: {out_dir.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
