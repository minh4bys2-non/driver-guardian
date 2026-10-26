"""Evaluate a self-contained checkpoint; edit the main() call at the bottom to run."""
import csv
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.LSTM.src.dataloader import RawVideoFramesDataset, collate_video_frames
from ai.LSTM.src.checkpoint import load_model
from ai.LSTM.src.training import configure_logging, logger

@torch.inference_mode()
def evaluate(model, loader, device, amp=True):
    model.eval()
    labels_all, preds_all, scores_all, rows = [], [], [], []
    total_loss = 0.0

    for frames, labels, lengths, metas in tqdm(loader, desc="Test", unit="batch"):
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(frames, seq_lens=lengths)
            if logits.shape != (labels.numel(), 2) or not torch.isfinite(logits).all():
                raise ValueError("Expected finite binary clip logits [B,2]")
            total_loss += F.cross_entropy(logits.float(), labels, reduction="sum").item()

        predictions = logits.argmax(dim=1).cpu().tolist()
        scores = logits.float().softmax(dim=1)[:, 1].cpu().tolist()
        targets = labels.cpu().tolist()
        labels_all.extend(targets)
        preds_all.extend(predictions)
        scores_all.extend(scores)

        for meta, target, pred, score in zip(metas, targets, predictions, scores):
            rows.append({
                "video_id": meta["video_id"],
                "video_path": meta["video_path"],
                "subject_id": meta["subject_id"],
                "source_video_id": meta["source_video_id"],
                "start_sec": meta["start_sec"],
                "end_sec": meta["end_sec"],
                "label": target,
                "prediction": pred,
                "prob_drowsy": score,
            })

    if not labels_all:
        raise ValueError("Empty evaluation loader")
    tn, fp, fn, tp = confusion_matrix(labels_all, preds_all, labels=[0, 1]).ravel()
    has_both_classes = len(set(labels_all)) == 2
    results = {
        "n_samples": len(labels_all),
        "support_alert": int(tn + fp),
        "support_drowsy": int(tp + fn),
        "loss": total_loss / len(labels_all),
        "accuracy": accuracy_score(labels_all, preds_all),
        "precision": precision_score(labels_all, preds_all, zero_division=0),
        "recall": recall_score(labels_all, preds_all, zero_division=0),
        "f1": f1_score(labels_all, preds_all, zero_division=0),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "roc_auc": roc_auc_score(labels_all, scores_all) if has_both_classes else None,
        "average_precision": average_precision_score(labels_all, scores_all) if has_both_classes else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }
    return results, rows

def main(checkpoint_path, *, dataset_dir=None, split="test", output_dir=None,
         device="auto", batch_size=None, num_workers=None):
    """None keeps checkpoint settings; runtime overrides do not change preprocessing."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"Invalid split: {split}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
    model, checkpoint = load_model(checkpoint_path, device)
    cfg = checkpoint["config"]
    dataset_dir = cfg["dataset_dir"] if dataset_dir is None else dataset_dir
    batch_size = cfg["batch_size"] if batch_size is None else batch_size
    num_workers = cfg["num_workers"] if num_workers is None else num_workers
    output_dir = Path(output_dir or Path(checkpoint_path).parent / f"evaluation_{split}")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / "evaluation.log")
    dataset = RawVideoFramesDataset(dataset_dir, split=split,
                                    img_size=cfg["image_size"], window_sec=cfg["window_sec"],
                                    target_fps=cfg["target_fps"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=device.type == "cuda",
                        collate_fn=collate_video_frames)
    metrics, predictions = evaluate(model, loader, device, amp=cfg["amp"])
    metrics.update(checkpoint_epoch=checkpoint["epoch"], split=split,
                   window_sec=cfg["window_sec"], target_fps=cfg["target_fps"], image_size=cfg["image_size"])
    evaluation_config = dict(checkpoint_path=str(Path(checkpoint_path).resolve()),
                             model_config=checkpoint["model_config"], config=cfg,
                             dataset_dir=str(dataset_dir), split=split, device=str(device),
                             batch_size=batch_size, num_workers=num_workers, amp=cfg["amp"] and device.type == "cuda")
    (output_dir / "evaluation_config.json").write_text(json.dumps(evaluation_config, indent=2), encoding="utf-8")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    logger.info("Checkpoint=%s split=%s metrics=%s", checkpoint_path, split, json.dumps(metrics))
    return metrics, predictions

if __name__ == "__main__":
    main(
        checkpoint_path="/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/LSTM/checkpoints/scratchv1/best.pt",
        dataset_dir="/home/tranmanhduy/Workspace/dataE/DataLSTM_v2",  # None: dùng đường dẫn dataset trong checkpoint.
        split="test",
        output_dir="/home/tranmanhduy/Workspace/ptithcm/driver-guardian/runs",  # None: lưu cạnh checkpoint trong evaluation_<split>.
        device="auto",
        batch_size=1,  # None: dùng batch_size và num_workers trong checkpoint.
        num_workers=2,
    )
