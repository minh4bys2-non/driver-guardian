# =====================================================================
# File: diagnose.py
# Dự án: Driver Guardian - Module LiteFat_v1 (KSS 1-9, Sliding 60s/30s)
# Tác vụ: Script chẩn đoán lỗi mô hình từ checkpoint trên tập Val / Test:
#           - Ma trận nhầm lẫn 3 Regime và 9 mức KSS theo từng dataset.
#           - Phân bố dự đoán (mean ± std, min..max, MAE) theo từng nhãn thật.
#           - Phân rã từng thành phần loss theo miền (UL-DD vs UTA-RLDD).
#           - So sánh từng mẫu giữa 2 checkpoint (vd: v6 vs v7) để định vị hồi quy.
# =====================================================================
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import TrainConfig
from dataset import LiteFatDataset, collate_litefat
from loss import LiteFatLoss
from model import LiteFatV1
from ultis import compute_kss_metrics, seed_everything


def _infer_legacy_flags(cfg_dict: Dict[str, Any], ckpt_path: Path) -> Dict[str, Any]:
    """
    Suy luận chính xác cờ hành vi cho các checkpoint cũ chưa lưu trường ablation mới:
      - v7: nhận diện theo tên thư mục 'novisual7' hoặc cặp (boundary=0.05, dann=0.05).
      - v3..v6: mặc định dùng hành vi v6.
    """
    merged = dict(cfg_dict)
    if "eye_vel_smooth" in merged and "uta_reg_mode" in merged:
        merged.setdefault("use_alert_boundary_margin", False)
        merged.setdefault("boundary_alert_max", 4.20)
        merged.setdefault("boundary_mid_min", 4.70)
        return merged

    is_v7 = (
        "novisual7" in str(ckpt_path)
        or "novisual7" in str(merged.get("checkpoint_dir", ""))
        or (
            abs(float(merged.get("boundary_margin_weight", 0.15)) - 0.05) < 1e-6
            and abs(float(merged.get("dann_weight", 0.15)) - 0.05) < 1e-6
        )
    )
    if is_v7:
        merged.setdefault("uta_reg_mode", "interval")
        merged.setdefault("boundary_drowsy_min", 6.65)
        merged.setdefault("boundary_nondrowsy_max", 6.30)
        merged.setdefault("eye_vel_smooth", True)
        merged.setdefault("eye_y_scale_mode", "v7")
        merged.setdefault("window_attn_weighted_logits", True)
        merged.setdefault("sampler_uta_quota", 0.35)
        merged.setdefault("sampler_weight_exponents", [0.70, 0.50, 0.50])
    else:
        merged.setdefault("uta_reg_mode", "point")
        merged.setdefault("boundary_drowsy_min", 6.70)
        merged.setdefault("boundary_nondrowsy_max", 6.20)
        merged.setdefault("eye_vel_smooth", False)
        merged.setdefault("eye_y_scale_mode", "v6")
        merged.setdefault("window_attn_weighted_logits", False)
        merged.setdefault("sampler_uta_quota", 0.40)
        merged.setdefault("sampler_weight_exponents", [1.0, 1.0, 0.5])
    merged.setdefault("use_alert_boundary_margin", False)
    merged.setdefault("boundary_alert_max", 4.20)
    merged.setdefault("boundary_mid_min", 4.70)
    return merged


def _build_config_from_checkpoint(
        ckpt: Dict[str, Any],
        ckpt_path: Path,
        overrides: argparse.Namespace,
) -> TrainConfig:
    """Khôi phục TrainConfig từ checkpoint và áp dụng các tham số ghi đè từ CLI."""
    raw_cfg: Dict[str, Any] = {}
    cfg_json = ckpt_path.parent / "config.json"
    if cfg_json.is_file():
        with open(cfg_json, "r", encoding="utf-8") as f:
            raw_cfg = json.load(f)
    if isinstance(ckpt.get("config"), dict):
        raw_cfg.update(ckpt["config"])

    raw_cfg = _infer_legacy_flags(raw_cfg, ckpt_path)
    valid_fields = {f.name for f in dataclasses.fields(TrainConfig)}
    filtered = {k: v for k, v in raw_cfg.items() if k in valid_fields}

    for size_key in ("letterbox_size", "face_size"):
        if size_key in filtered and isinstance(filtered[size_key], list):
            filtered[size_key] = tuple(int(x) for x in filtered[size_key])
    for pairs_key in ("vertical_pairs", "cross_region_pairs"):
        if pairs_key in filtered and isinstance(filtered[pairs_key], list):
            filtered[pairs_key] = [(int(p[0]), int(p[1])) for p in filtered[pairs_key]]
    if "sampler_weight_exponents" in filtered and isinstance(filtered["sampler_weight_exponents"], list):
        filtered["sampler_weight_exponents"] = tuple(float(x) for x in filtered["sampler_weight_exponents"])

    cfg = TrainConfig(**filtered)
    if overrides.dataset_dir is not None:
        cfg.dataset_dir = overrides.dataset_dir
    if overrides.cache_dir is not None:
        cfg.cache_dir = overrides.cache_dir
    if overrides.manifest is not None:
        cfg.manifest_filename = overrides.manifest
    if overrides.use_cache is not None:
        cfg.use_cache = overrides.use_cache
    if overrides.batch_size is not None:
        cfg.batch_size = overrides.batch_size
    return cfg


def _kss_to_regime(val: float) -> int:
    if val >= 6.5:
        return 2
    if val >= 4.5:
        return 1
    return 0


def _confusion_matrix(yt: List[int], yp: List[int], labels: List[int]) -> List[List[int]]:
    idx_map = {lbl: i for i, lbl in enumerate(labels)}
    mat = [[0 for _ in labels] for _ in labels]
    for a, b in zip(yt, yp):
        if a in idx_map and b in idx_map:
            mat[idx_map[a]][idx_map[b]] += 1
    return mat


def _logits_to_drowsy_prob(pred_kss: np.ndarray, logits: np.ndarray, kss_min: float, kss_max: float) -> np.ndarray:
    if logits is not None and len(logits) == len(pred_kss) and logits.ndim == 2 and logits.shape[1] >= 2:
        x = logits.astype(np.float64)
        x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(x)
        probs = exp_x / np.sum(exp_x, axis=-1, keepdims=True).clip(min=1e-12)
        start_idx = max(0, min(probs.shape[1] - 1, int(round(7.0 - kss_min))))
        return probs[:, start_idx:].sum(axis=-1)
    span = max(1e-6, float(kss_max - kss_min))
    return np.clip((pred_kss.astype(np.float64) - kss_min) / span, 0.0, 1.0)


@torch.no_grad()
def evaluate_checkpoint_split(
        ckpt_path: Path,
        split: str,
        overrides: argparse.Namespace,
        device: torch.device,
) -> Dict[str, Any]:
    """Nạp checkpoint, chạy suy luận trên một split và thu thập thống kê chẩn đoán chi tiết."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _build_config_from_checkpoint(ckpt, ckpt_path, overrides)
    seed_everything(cfg.seed)

    ds = LiteFatDataset(
        dataset_dir=cfg.dataset_dir,
        split=split,
        time_interval=cfg.time_interval,
        window_sec=cfg.dataset_window_sec,
        letterbox_size=cfg.letterbox_size,
        face_size=cfg.face_size,
        manifest_filename=cfg.manifest_filename,
        use_cache=cfg.use_cache,
        cache_dir=cfg.cache_dir,
        pipnet_backbone=cfg.pipnet_backbone,
        conf_threshold=cfg.conf_threshold,
        nms_threshold=cfg.nms_threshold,
        bbox_margin=cfg.bbox_margin,
        box_smooth_alpha=cfg.box_smooth_alpha,
        extract_crops_only=(not cfg.use_visual_context),
        selected_datasets=cfg.selected_datasets,
        use_visual_context=cfg.use_visual_context,
        eval_window_mode=cfg.eval_window_mode,
        visual_frame_stride=cfg.visual_frame_stride,
        augment_train=False,
        return_meta=True,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_litefat,
    )

    model = LiteFatV1(
        num_nodes=cfg.num_nodes,
        feature_dim=cfg.feature_dim,
        mobilenet_proj_hidden=cfg.mobilenet_proj_hidden,
        node_embed_dim=cfg.node_embed_dim,
        residual_channels=cfg.residual_channels,
        dilation_channels=cfg.dilation_channels,
        skip_channels=cfg.skip_channels,
        end_channels=cfg.end_channels,
        num_classes=cfg.num_classes,
        kss_min=cfg.kss_min,
        kss_max=cfg.kss_max,
        fps=cfg.fps,
        window_sec=cfg.window_sec,
        stride_sec=cfg.stride_sec,
        num_blocks=cfg.num_blocks,
        layers_per_block=cfg.layers_per_block,
        kernel_size=cfg.kernel_size,
        pretrained_backbone=False,
        dropout=cfg.dropout,
        freeze_backbone=(cfg.freeze_backbone_epochs > 0),
        vertical_pairs=cfg.vertical_pairs,
        cross_region_pairs=cfg.cross_region_pairs,
        static_shape_ratio=cfg.static_shape_ratio,
        use_dann=bool(getattr(cfg, "use_dann", False)),
        eye_vel_smooth=bool(getattr(cfg, "eye_vel_smooth", False)),
        eye_y_scale_mode=str(getattr(cfg, "eye_y_scale_mode", "v6")),
        window_attn_weighted_logits=bool(getattr(cfg, "window_attn_weighted_logits", False)),
    ).to(device)

    use_ema = (not overrides.no_ema) and ("ema_state_dict" in ckpt)
    sd = ckpt["ema_state_dict"] if use_ema else ckpt.get("model_state_dict", ckpt)
    if isinstance(sd, dict) and "shadow" in sd and isinstance(sd["shadow"], dict):
        sd = sd["shadow"]
    model.load_state_dict(sd, strict=True)
    model.eval()

    criterion = LiteFatLoss(
        num_classes=cfg.num_classes,
        kss_min=cfg.kss_min,
        kss_max=cfg.kss_max,
        clip_reg_weight=cfg.clip_reg_weight,
        clip_cls_weight=cfg.clip_cls_weight,
        frame_loss_weight=cfg.frame_loss_weight,
        window_loss_weight=cfg.window_loss_weight,
        window_smooth_weight=cfg.window_smooth_weight,
        boundary_margin_weight=float(getattr(cfg, "boundary_margin_weight", 0.15)),
        uta_ordinal_sigma=float(getattr(cfg, "uta_ordinal_sigma", 1.25)),
        dann_weight=float(getattr(cfg, "dann_weight", 0.15)),
        huber_beta=cfg.huber_beta,
        ordinal_sigma=cfg.ordinal_sigma,
        label_smoothing=cfg.label_smoothing,
        alert_kss_threshold=cfg.alert_kss_threshold,
        mid_kss_threshold=cfg.mid_kss_threshold,
        topk_mid_ratio=cfg.topk_mid_ratio,
        topk_drowsy_ratio=cfg.topk_drowsy_ratio,
        fps=cfg.fps,
        window_sec=cfg.window_sec,
        stride_sec=cfg.stride_sec,
        smooth_margin=cfg.smooth_margin,
        uta_reg_mode=str(getattr(cfg, "uta_reg_mode", "point")),
        uta_interval_center_weight=float(getattr(cfg, "uta_interval_center_weight", 0.50)),
        boundary_drowsy_min=float(getattr(cfg, "boundary_drowsy_min", 6.70)),
        boundary_nondrowsy_max=float(getattr(cfg, "boundary_nondrowsy_max", 6.20)),
        use_alert_boundary_margin=bool(getattr(cfg, "use_alert_boundary_margin", False)),
        boundary_alert_max=float(getattr(cfg, "boundary_alert_max", 4.20)),
        boundary_mid_min=float(getattr(cfg, "boundary_mid_min", 4.70)),
    ).to(device)

    records: List[Dict[str, Any]] = []
    all_targets: List[float] = []
    all_preds: List[float] = []
    all_logits: List[np.ndarray] = []
    domain_loss_sums: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    domain_loss_counts: Dict[str, int] = defaultdict(int)

    for batch in loader:
        b_frames = batch[0].to(device, non_blocking=True) if batch[0] is not None else None
        b_lm = batch[1].to(device, non_blocking=True)
        b_lbl = batch[2].to(device, non_blocking=True)
        b_len = batch[3].to(device, non_blocking=True)
        metas = batch[4]

        out = model(
            video_frames=b_frames,
            landmarks=b_lm,
            lengths=b_len,
            return_dict=True,
        )
        c_kss = out["clip_kss"].float().cpu().numpy()
        c_logits = out["clip_logits"].float().cpu().numpy()
        y_np = b_lbl.float().cpu().numpy()

        all_targets.extend(y_np.tolist())
        all_preds.extend(c_kss.tolist())
        all_logits.append(c_logits)

        # Tính thành phần loss riêng cho từng mẫu nhằm tách chính xác theo từng domain
        B_cur = len(metas)
        for i in range(B_cur):
            m = metas[i]
            ds_name = str(m.get("dataset", "unknown"))
            dom_id = int(m.get("domain_id", 0 if ds_name.upper().startswith("UL") else 1))
            single_out = {
                k: (v[i:i + 1] if isinstance(v, torch.Tensor) else None)
                for k, v in out.items()
                if k != "state"
            }
            _, l_details = criterion(
                single_out,
                b_lbl[i:i + 1],
                lengths=b_len[i:i + 1],
                domain_targets=torch.tensor([dom_id], dtype=torch.long, device=device),
                return_dict=True,
            )
            domain_loss_counts[ds_name] += 1
            for lk, lv in l_details.items():
                domain_loss_sums[ds_name][lk] += float(lv)

            records.append({
                "key": str(m.get("key", "")),
                "dataset": ds_name,
                "subject_id": str(m.get("subject_id", "")),
                "video_id": str(m.get("video_id", "")),
                "domain_id": dom_id,
                "target_kss": float(y_np[i]),
                "pred_kss": float(c_kss[i]),
                "abs_err": float(abs(float(c_kss[i]) - float(y_np[i]))),
            })

    targets_np = np.asarray(all_targets, dtype=np.float64)
    preds_np = np.asarray(all_preds, dtype=np.float64)
    logits_np = np.concatenate(all_logits, axis=0) if all_logits else np.empty((0, cfg.num_classes))
    drowsy_probs = _logits_to_drowsy_prob(preds_np, logits_np, cfg.kss_min, cfg.kss_max)
    for idx_r, r in enumerate(records):
        r["drowsy_prob"] = float(drowsy_probs[idx_r])

    overall_m = compute_kss_metrics(
        targets_np, preds_np, pred_logits=logits_np if len(logits_np) else None,
        kss_min=cfg.kss_min, kss_max=cfg.kss_max,
    )

    per_ds: Dict[str, Any] = {}
    ds_names = sorted({r["dataset"] for r in records})
    for ds_name in ds_names:
        mask = np.array([r["dataset"] == ds_name for r in records])
        dm = compute_kss_metrics(
            targets_np[mask], preds_np[mask],
            pred_logits=logits_np[mask] if len(logits_np) == len(targets_np) else None,
            kss_min=cfg.kss_min, kss_max=cfg.kss_max,
        )
        yt_cls = [int(np.clip(np.rint(v), 1, 9)) for v in targets_np[mask]]
        yp_cls = [int(np.clip(np.rint(v), 1, 9)) for v in preds_np[mask]]
        yt_reg = [_kss_to_regime(v) for v in targets_np[mask]]
        yp_reg = [_kss_to_regime(v) for v in preds_np[mask]]

        by_true_kss: Dict[str, Any] = {}
        for lbl in sorted(set(yt_cls)):
            lm = np.array([a == lbl for a in yt_cls])
            p_sub = preds_np[mask][lm]
            t_sub = targets_np[mask][lm]
            by_true_kss[str(lbl)] = {
                "n": int(len(p_sub)),
                "pred_mean": float(np.mean(p_sub)),
                "pred_std": float(np.std(p_sub)),
                "pred_min": float(np.min(p_sub)),
                "pred_max": float(np.max(p_sub)),
                "mae": float(np.mean(np.abs(p_sub - t_sub))),
                "within_1": float(np.mean(np.abs(p_sub - t_sub) <= 1.0)),
            }

        cnt = max(1, domain_loss_counts[ds_name])
        avg_losses = {k: round(v / cnt, 4) for k, v in domain_loss_sums[ds_name].items()}

        per_ds[ds_name] = {
            "metrics": dm,
            "regime_cm": _confusion_matrix(yt_reg, yp_reg, [0, 1, 2]),
            "kss_cm": _confusion_matrix(yt_cls, yp_cls, list(range(1, 10))),
            "pred_by_true_kss": by_true_kss,
            "avg_losses_train_formula": avg_losses,
        }

    return {
        "checkpoint": str(ckpt_path),
        "epoch": int(ckpt.get("epoch", -1)),
        "split": split,
        "flags": {
            "uta_reg_mode": cfg.uta_reg_mode,
            "use_alert_boundary_margin": cfg.use_alert_boundary_margin,
            "boundary_alert_max": cfg.boundary_alert_max,
            "boundary_mid_min": cfg.boundary_mid_min,
            "eye_vel_smooth": cfg.eye_vel_smooth,
            "eye_y_scale_mode": cfg.eye_y_scale_mode,
            "window_attn_weighted_logits": cfg.window_attn_weighted_logits,
            "boundary_margin_weight": cfg.boundary_margin_weight,
            "dann_weight": cfg.dann_weight,
        },
        "overall_metrics": overall_m,
        "per_dataset": per_ds,
        "records": records,
    }


def _print_report(res: Dict[str, Any]) -> None:
    om = res["overall_metrics"]
    print("=" * 78)
    print(f"CKPT: {res['checkpoint']} (ep={res['epoch']}) | SPLIT: {res['split']} | FLAGS: {res['flags']}")
    print(
        f"  OVERALL: MAE={om['mae']:.4f} | RMSE={om['rmse']:.4f} | W1={om['within_1_acc']*100:.1f}% | "
        f"QWK={om['qwk']:.4f} | RegimeAcc={om['regime_acc']*100:.1f}% | AUC={om['auc']:.4f} | F1={om['f1']:.4f}"
    )
    for ds_name, d_info in res["per_dataset"].items():
        dm = d_info["metrics"]
        print(
            f"\n  [{ds_name}] N={dm['num_samples']} | MAE={dm['mae']:.4f} | QWK={dm['qwk']:.4f} | "
            f"RegimeAcc={dm['regime_acc']*100:.1f}% | AUC={dm['auc']:.4f} | F1={dm['f1']:.4f}"
        )
        print(f"    Loss decomposition (with domain_targets): {d_info['avg_losses_train_formula']}")
        print("    Regime CM (rows=true [Alert,Mid,Drowsy], cols=pred):")
        for rname, row in zip(("Alert ", "Mid   ", "Drowsy"), d_info["regime_cm"]):
            print(f"      {rname}: {row}")
        print("    Pred distribution by true KSS:")
        for lbl, st in d_info["pred_by_true_kss"].items():
            print(
                f"      KSS={lbl} (N={st['n']:2d}): pred={st['pred_mean']:.2f}±{st['pred_std']:.2f} "
                f"[{st['pred_min']:.2f}..{st['pred_max']:.2f}] | MAE={st['mae']:.3f} | W1={st['within_1']*100:.0f}%"
            )


def _compare_pair(base: Dict[str, Any], target: Dict[str, Any], out_dir: Path, split: str) -> None:
    base_map = {r["key"]: r for r in base["records"]}
    diffs: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[str, int], List[Tuple[float, float, float, float]]] = defaultdict(list)

    for r in target["records"]:
        b = base_map.get(r["key"])
        if b is None:
            continue
        delta_err = r["abs_err"] - b["abs_err"]
        diffs.append({
            "key": r["key"],
            "dataset": r["dataset"],
            "subject_id": r["subject_id"],
            "target_kss": r["target_kss"],
            "base_pred": b["pred_kss"],
            "target_pred": r["pred_kss"],
            "base_err": b["abs_err"],
            "target_err": r["abs_err"],
            "delta_err": delta_err,
        })
        grouped[(r["dataset"], int(round(r["target_kss"])))].append(
            (b["pred_kss"], r["pred_kss"], b["abs_err"], r["abs_err"])
        )

    print("\n" + "-" * 78)
    print(f"PAIRWISE COMPARISON [{split}]: {Path(base['checkpoint']).parent.name} -> {Path(target['checkpoint']).parent.name}")
    print("  By (dataset, true_KSS):")
    for (ds_name, lbl), vals in sorted(grouped.items()):
        bp = np.mean([v[0] for v in vals])
        tp = np.mean([v[1] for v in vals])
        be = np.mean([v[2] for v in vals])
        te = np.mean([v[3] for v in vals])
        print(
            f"    {ds_name:8s} KSS={lbl} (N={len(vals):2d}): pred {bp:.2f} -> {tp:.2f} | MAE {be:.3f} -> {te:.3f} ({te-be:+.3f})"
        )

    diffs.sort(key=lambda x: x["delta_err"], reverse=True)
    print("  Top 10 worst regressed samples (delta_err = target_err - base_err):")
    for d in diffs[:10]:
        print(
            f"    {d['dataset']:8s} {d['subject_id']:8s} tgt={d['target_kss']:.0f} | "
            f"pred {d['base_pred']:.2f}->{d['target_pred']:.2f} | err {d['base_err']:.2f}->{d['target_err']:.2f} ({d['delta_err']:+.2f})"
        )

    cmp_path = out_dir / f"comparison_{split}.csv"
    if diffs:
        with open(cmp_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(diffs[0].keys()))
            writer.writeheader()
            writer.writerows(diffs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chẩn đoán sai số & phân rã loss cho các checkpoint LiteFat_v1.")
    parser.add_argument(
        "--ckpts",
        nargs="+",
        default=[
            "checkpoints/LiteFat_v1_novisual6/best.pt",
            "checkpoints/LiteFat_v1_novisual7/best.pt",
        ],
        help="Danh sách đường dẫn checkpoint (.pt) cần chẩn đoán.",
    )
    parser.add_argument("--splits", nargs="+", default=["val", "test"], help="Các split cần đánh giá.")
    parser.add_argument("--dataset-dir", type=str, default=None, help="Ghi đè dataset_dir.")
    parser.add_argument("--cache-dir", type=str, default=None, help="Ghi đè cache_dir.")
    parser.add_argument("--manifest", type=str, default=None, help="Ghi đè manifest_filename.")
    parser.add_argument("--batch-size", type=int, default=None, help="Ghi đè batch_size.")
    parser.add_argument("--use-cache", dest="use_cache", action="store_true", default=None)
    parser.add_argument("--no-cache", dest="use_cache", action="store_false")
    parser.add_argument("--no-ema", action="store_true", help="Dùng model_state_dict thay vì ema_state_dict.")
    parser.add_argument("--out", type=str, default="./diagnose_out", help="Thư mục lưu báo cáo JSON/CSV.")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    full_report: Dict[str, Any] = {}
    for split in args.splits:
        split_results: List[Dict[str, Any]] = []
        for ckpt_str in args.ckpts:
            ckpt_path = Path(ckpt_str)
            if not ckpt_path.is_file():
                print(f"[SKIP] Không tìm thấy checkpoint: {ckpt_path}")
                continue
            res = evaluate_checkpoint_split(ckpt_path, split, args, device)
            split_results.append(res)
            _print_report(res)

            tag = f"{ckpt_path.parent.name}_{ckpt_path.stem}_{split}"
            pred_csv = out_dir / f"{tag}_preds.csv"
            if res["records"]:
                with open(pred_csv, "w", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(res["records"][0].keys()))
                    writer.writeheader()
                    writer.writerows(res["records"])

        if len(split_results) >= 2:
            for i in range(1, len(split_results)):
                _compare_pair(split_results[0], split_results[i], out_dir, split)

        # Lưu report không kèm list records dài dòng để dễ đọc JSON
        full_report[split] = [
            {k: v for k, v in r.items() if k != "records"}
            for r in split_results
        ]

    report_path = out_dir / "diagnose_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"\nĐã lưu báo cáo chẩn đoán tại: {report_path}")


if __name__ == "__main__":
    main()
