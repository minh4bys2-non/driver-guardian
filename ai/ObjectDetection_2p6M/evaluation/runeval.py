"""Chỉnh main(), rồi chạy từ project root: python -m evaluation.runeval."""

import json
from dataclasses import replace
from pathlib import Path

import torch
import ultralytics
from torch.utils.data import DataLoader

from ai.ObjectDetection_2p6M.evaluation.ultralytics_evaluation import compute_map_metrics
from ai.ObjectDetection_2p6M.src.config import TrainConfig
from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector
from ai.ObjectDetection_2p6M.train_.dataloader_ import build_split_dataset, collate_fn, load_categories
from ai.ObjectDetection_2p6M.utils.artifacts import validate_metadata

def run_evaluation(checkpoint_path, cfg, *, img_size=None, use_ema=True,
                   score_thres=0.001, max_det=300, output_path=None):
    if not checkpoint_path:
        raise ValueError("Hãy đặt checkpoint_path trong main()")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = validate_metadata(checkpoint_path, checkpoint)
    categories_path = Path(cfg.labels_root) / cfg.train_subdir / cfg.categories_filename
    mapping, categories, nc = load_categories(categories_path)
    if categories != metadata["categories"]:
        raise ValueError("Categories của dataset không khớp mapping trong checkpoint")

    if img_size is None:
        img_size = checkpoint.get("cfg", {}).get("img_size", cfg.img_size)
    if img_size <= 0 or img_size % max(metadata["architecture"]["strides"]):
        raise ValueError("img_size phải dương và chia hết cho stride lớn nhất")
    cfg = replace(cfg, img_size=img_size)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[Eval] CUDA không khả dụng, dùng CPU")
        device = torch.device("cpu")

    dataset = build_split_dataset(
        cfg, str(Path(cfg.labels_root) / cfg.val_subdir), is_train=False,
        image_path_map_filename=cfg.val_image_path_map_filename,
        images_split_dir=cfg.images_val_subdir, cat_id_to_idx=mapping)
    if not len(dataset):
        raise ValueError("Validation dataset rỗng")
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, drop_last=False,
                        num_workers=cfg.num_workers, pin_memory=device.type == "cuda",
                        collate_fn=collate_fn)
    model = NMSFreeDetector(**metadata["architecture"], img_size=img_size)
    weights = "ema" if use_ema and checkpoint.get("ema") is not None else "model"
    model.load_state_dict(checkpoint[weights])
    model.to(device).eval()
    print(f"[Eval] {checkpoint_path} | weights={weights} | device={device} | "
          f"img_size={img_size} | images={len(dataset)} | classes={nc}")
    scores = compute_map_metrics(model, loader, device, nc,
                                 score_thres=score_thres, max_det=max_det)
    print("[Eval] " + " | ".join(f"{key}={scores[key]:.6f}" for key in
                                ("map_50", "map_50_95", "precision", "recall")))
    if output_path:
        report = {"checkpoint": str(checkpoint_path), "weights": weights,
                  "ultralytics_version": ultralytics.__version__, "images": len(dataset),
                  "img_size": img_size, "score_thres": score_thres, "max_det": max_det,
                  "labels_root": str(cfg.labels_root), "images_root_dir": str(cfg.images_root_dir),
                  "val_subdir": cfg.val_subdir, "images_val_subdir": cfg.images_val_subdir,
                  "skip_iscrowd": cfg.skip_iscrowd, "skip_isfake": cfg.skip_isfake,
                  "include_images_without_annotations": cfg.include_images_without_annotations,
                  "categories": categories, **scores}
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[Eval] Đã lưu: {path}")
    return scores


def main():
    checkpoint_path = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt"  # best.pt/last.pt kèm architecture.json và categories.json.
    cfg = TrainConfig(
        labels_root="/run/media/tranmanhduy/Data/MSCOCO/labels",
        images_root_dir="/run/media/tranmanhduy/Data/MSCOCO/images",
        index_cache_dir="/run/media/tranmanhduy/Data/MSCOCO/cache",  # Đổi cache khi đổi dataset.
        log_dir="runs/evaluation",
        batch_size=4,
        num_workers=4,
        device="cuda",
        include_images_without_annotations=True,  # Tính cả false positives trên ảnh không có GT.
        max_load_retries=1,  # Dừng nếu ảnh hỏng, tránh thay bằng ảnh khác khi đánh giá.
    )
    run_evaluation(checkpoint_path, cfg, img_size=640, use_ema=True,
                   score_thres=0.001, max_det=300, output_path="runs/evaluation/metrics.json")

if __name__ == "__main__":
    main()
