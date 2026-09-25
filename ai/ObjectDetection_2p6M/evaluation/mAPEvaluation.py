from typing import Dict, List, Optional
import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm import tqdm

def _ap_101(recall: np.ndarray, precision: np.ndarray) -> float:
    recall = np.concatenate(([0.0], recall, [1.0]))
    precision = np.concatenate(([0.0], precision, [0.0]))
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    idx = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
    return float(precision[np.clip(idx, 0, len(precision) - 1)].mean())

class MetricAccumulator:
    def __init__( self, nc: int, iou_thresholds: Optional[List[float]] | None = None, score_thres: float = 0.001,
                  max_det: int = 300, pr_iou_thres: float = 0.5, pr_score_thres: float = 0.25
    ) -> None:
        self.nc = nc
        self.iou_thresholds = iou_thresholds or list(np.round(np.arange(0.5, 1.0, 0.05), 2))
        self.score_thres = score_thres
        self.max_det = max_det
        self.pr_iou_thres = pr_iou_thres
        self.pr_score_thres = pr_score_thres

        self.preds_by_class = {c: [] for c in range(nc)}
        self.gts_by_class = {c: {} for c in range(nc)}
        self.n_gt_per_class = np.zeros(nc, dtype=np.int64)
        self.tp_pr = self.fp_pr = self.fn_pr = 0
        self.img_id = 0

    @torch.no_grad()
    def update(self, preds: Dict, targets: list[Dict]) -> None:
        boxes = preds["o2o"]["box"].detach()
        scores = torch.sigmoid(preds["o2o"]["cls"].detach())

        for b in range(len(boxes)):
            gt_boxes = targets[b]["boxes"].detach().cpu()
            gt_labels = targets[b]["labels"].detach().cpu().long()

            for c in gt_labels.unique().tolist():
                c = int(c)
                cls_gt = gt_boxes[gt_labels == c]
                self.gts_by_class[c][self.img_id] = cls_gt
                self.n_gt_per_class[c] += len(cls_gt)

            # YOLO-style top-k over all (box, class) pairs.
            flat = scores[b].flatten()
            k = min(self.max_det, flat.numel())
            p_scores, idx = flat.topk(k)

            keep = p_scores >= self.score_thres
            p_scores, idx = p_scores[keep], idx[keep]
            box_idx, p_labels = idx // self.nc, (idx % self.nc).cpu()
            p_boxes = boxes[b][box_idx].cpu()
            p_scores = p_scores.cpu()

            for box, score, c in zip(p_boxes, p_scores, p_labels.tolist()):
                self.preds_by_class[c].append((self.img_id, float(score), box))

            keep = p_scores >= self.pr_score_thres
            pr_boxes, pr_labels = p_boxes[keep], p_labels[keep]

            matched = torch.zeros(len(gt_boxes), dtype=torch.bool)
            pr_ious = box_iou(pr_boxes, gt_boxes)

            for ious, c in zip(pr_ious, pr_labels.tolist()):
                cand = (gt_labels == c) & ~matched
                if cand.any():
                    ious[~cand] = -1
                    best_iou, best_j = ious.max(0)

                    if best_iou >= self.pr_iou_thres:
                        self.tp_pr += 1
                        matched[best_j] = True
                        continue

                self.fp_pr += 1

            self.fn_pr += int((~matched).sum())
            self.img_id += 1

    def compute(self) -> Dict:
        per_class_ap = [0.0] * self.nc
        per_class_ap50 = [0.0] * self.nc

        for c in tqdm(range(self.nc), desc="Validation mAP", leave=False):
            n_gt = self.n_gt_per_class[c]
            if not n_gt:
                continue

            preds = sorted(self.preds_by_class[c], key=lambda x: -x[1])
            aps = []
            thresholds = np.asarray(self.iou_thresholds)
            tp = np.zeros((len(preds), len(thresholds)), dtype=bool)
            by_image = {}
            for i, (img, _, box) in enumerate(preds):
                if img in self.gts_by_class[c]:
                    by_image.setdefault(img, []).append(i)

            for img, indices in by_image.items():
                gt = self.gts_by_class[c][img]
                if not len(gt):
                    continue
                boxes = torch.stack([preds[i][2] for i in indices])
                ious = box_iou(boxes, gt).numpy()
                image_thresholds = thresholds.astype(ious.dtype)
                matched = np.zeros((len(thresholds), len(gt)), dtype=bool)
                rows = np.arange(len(thresholds))
                # Giữ thứ tự confidence và trạng thái ghép riêng cho từng ngưỡng IoU.
                for i, overlaps in zip(indices, ious):
                    available = np.where(matched, -1., overlaps[None])
                    best = available.argmax(axis=1)
                    hits = available[rows, best] >= image_thresholds
                    tp[i] = hits
                    matched[rows[hits], best[hits]] = True

            for t, iou_thr in enumerate(self.iou_thresholds):
                tp_cum = np.cumsum(tp[:, t])
                recall = tp_cum / n_gt
                precision = tp_cum / np.arange(1, len(preds) + 1)
                ap = _ap_101(recall, precision)
                aps.append(ap)

                if np.isclose(iou_thr, 0.5):
                    per_class_ap50[c] = ap

            per_class_ap[c] = float(np.mean(aps))

        valid = self.n_gt_per_class > 0
        map_50_95 = float(np.mean(np.asarray(per_class_ap)[valid])) if valid.any() else 0.0
        map_50 = float(np.mean(np.asarray(per_class_ap50)[valid])) if valid.any() else 0.0

        precision = self.tp_pr / max(self.tp_pr + self.fp_pr, 1)
        recall = self.tp_pr / max(self.tp_pr + self.fn_pr, 1)

        return {
            "map_50_95": map_50_95,
            "map_50": map_50,
            "precision": precision,
            "recall": recall,
            "per_class_ap": per_class_ap,
        }

@torch.no_grad()
def compute_map_metrics(model, loader, device, nc: int, move_batch, **kwargs) -> Dict:
    model.eval()
    acc = MetricAccumulator(nc=nc, **kwargs)

    for images, targets in loader:
        images, targets = move_batch(images, targets, device)
        acc.update(model(images, o2o_only=True), targets)

    return acc.compute()
