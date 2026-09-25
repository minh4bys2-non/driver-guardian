"""Ultralytics validation metrics for NMSFreeDetector (requires ultralytics).

Uses the installed Ultralytics matcher and AP implementation, not COCOeval.
Boxes and targets must be xyxy in the same image coordinates. The loader
controls image/annotation filtering; use an unaugmented validation loader.
"""

import numpy as np
import torch
from tqdm import tqdm
from ultralytics.engine.validator import BaseValidator
from ultralytics.utils.metrics import ap_per_class, box_iou


class MetricAccumulator:
    def __init__(self, nc, score_thres=0.001, max_det=300):
        if nc <= 0 or max_det <= 0 or not 0 <= score_thres <= 1:
            raise ValueError("nc/max_det phải > 0; score_thres phải trong [0, 1]")
        self.nc, self.score_thres, self.max_det = nc, score_thres, max_det
        self.iouv = torch.linspace(0.5, 0.95, 10)
        self.stats = []
        self.img_id = 0

    @torch.no_grad()
    def update(self, preds, targets):
        boxes = preds["o2o"]["box"].detach().float()
        scores = preds["o2o"]["cls"].detach().float().sigmoid()
        for image_boxes, image_scores, target in zip(boxes, scores, targets):
            # Keep the project's NMS-free top-k over (box, class) pairs.
            flat = image_scores.flatten()
            conf, indices = flat.topk(min(self.max_det, flat.numel()))
            keep = conf > self.score_thres
            conf, indices = conf[keep].cpu(), indices[keep]
            pred_cls = (indices % self.nc).cpu()
            pred_boxes = image_boxes[indices // self.nc].cpu()
            gt_cls = target["labels"].detach().cpu().long()
            gt_boxes = target["boxes"].detach().cpu().float()
            tp = BaseValidator.match_predictions(
                self, pred_cls, gt_cls, box_iou(gt_boxes, pred_boxes))
            self.stats.append(tuple(x.numpy() for x in (tp, conf, pred_cls, gt_cls)))
            self.img_id += 1

    def compute(self):
        if not self.img_id:
            raise ValueError("Validation loader rỗng")
        tp, conf, pred_cls, gt_cls = (np.concatenate(x) for x in zip(*self.stats))
        per_class_ap = np.zeros(self.nc)
        result = {"map_50_95": 0.0, "map_50": 0.0, "precision": 0.0, "recall": 0.0,
                  "per_class_ap": per_class_ap.tolist()}
        if not tp.any():
            return result
        _, _, p, r, _, ap, classes, *_ = ap_per_class(tp, conf, pred_cls, gt_cls)
        per_class_ap[classes] = ap.mean(1)
        return {"map_50_95": float(ap.mean()), "map_50": float(ap[:, 0].mean()),
                "precision": float(p.mean()), "recall": float(r.mean()),
                "per_class_ap": per_class_ap.tolist()}


@torch.no_grad()
def compute_map_metrics(model, loader, device, nc, move_batch=None, **kwargs):
    """Evaluate an already-loaded model on its device; restore its train_/eval mode.

    Returns map_50, map_50_95, precision, recall and per_class_ap (class-indexed).
    Optional move_batch and metric kwargs match the existing evaluator API.
    """
    acc = MetricAccumulator(nc, **kwargs)
    training = model.training
    model.eval()
    try:
        for images, targets in tqdm(loader, desc="Ultralytics validation"):
            if move_batch is None:
                images = images.to(device)
            else:
                images, targets = move_batch(images, targets, device)
            acc.update(model(images, o2o_only=True), targets)
        return acc.compute()
    finally:
        model.train(training)
