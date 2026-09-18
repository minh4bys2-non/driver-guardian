import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def bbox_iou(box1, box2, xywh=False, GIoU=False, DIoU=False, CIoU=False, eps=1e-7):
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        w1_, h1_, w2_, h2_ = w1 / 2, h1 / 2, w2 / 2, h2 / 2
        b1_x1, b1_y1, b1_x2, b1_y2 = x1 - w1_, y1 - h1_, x1 + w1_, y1 + h1_
        b2_x1, b2_y1, b2_x2, b2_y2 = x2 - w2_, y2 - h2_, x2 + w2_, y2 + h2_
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
        b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1

    inter = ((b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp(0) *
             (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp(0))
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union
    if not (GIoU or DIoU or CIoU):
        return iou.squeeze(-1)

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    if GIoU and not (DIoU or CIoU):
        c_area = cw * ch + eps
        return (iou - (c_area - union) / c_area).squeeze(-1)

    c2 = cw.square() + ch.square() + eps
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2).square() +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2).square()) / 4
    if DIoU and not CIoU:
        return (iou - rho2 / c2).squeeze(-1)

    v = 4 / math.pi**2 * (torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps))).square()
    with torch.no_grad():
        alpha = v / (v - iou + 1 + eps)
    return (iou - rho2 / c2 - v * alpha).squeeze(-1)


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    lt, rb = distance.chunk(2, dim)
    x1y1, x2y2 = anchor_points - lt, anchor_points + rb
    if not xywh:
        return torch.cat((x1y1, x2y2), dim)
    return torch.cat(((x1y1 + x2y2) / 2, x2y2 - x1y1), dim)


def bbox2dist(anchor_points, bbox, reg_max):
    x1y1, x2y2 = bbox.chunk(2, -1)
    return torch.cat((anchor_points - x1y1, x2y2 - anchor_points), -1).clamp_(0, reg_max - 0.01)

class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super().__init__()
        self.topk, self.nc = topk, num_classes
        self.alpha, self.beta, self.eps = alpha, beta, eps

    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, anc_points, gt_labels, gt_bboxes, mask_gt):
        self.bs, self.n_max_boxes = pd_scores.shape[0], gt_bboxes.shape[1]
        A, device = pd_scores.shape[1], gt_bboxes.device
        if self.n_max_boxes == 0:
            return (
                torch.full((self.bs, A), self.nc, dtype=torch.long, device=device),
                torch.zeros((self.bs, A, 4), device=device),
                torch.zeros((self.bs, A, self.nc), device=device),
                torch.zeros((self.bs, A), dtype=torch.bool, device=device),
                torch.zeros((self.bs, A), dtype=torch.long, device=device),
            )

        mask_pos, align_metric, overlaps = self.get_pos_mask(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt
        )
        target_gt_idx, fg_mask, mask_pos = self.select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes
        )
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask
        )

        align_metric *= mask_pos
        max_align = align_metric.amax(-1, keepdim=True)
        max_iou = (overlaps * mask_pos).amax(-1, keepdim=True)
        norm = (align_metric * max_iou / (max_align + self.eps)).amax(-2).unsqueeze(-1)
        return target_labels, target_bboxes, target_scores * norm, fg_mask.bool(), target_gt_idx

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):
        mask_in_gts = self.select_candidates_in_gts(anc_points, gt_bboxes)
        align_metric, overlaps = self.get_box_metrics(
            pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_in_gts * mask_gt
        )
        mask_topk = self.select_topk_candidates(
            align_metric, mask_gt.expand(-1, -1, self.topk).bool()
        )
        return mask_topk * mask_in_gts * mask_gt, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, mask_gt):
        bs, M, A = self.bs, self.n_max_boxes, pd_scores.shape[1]
        mask_gt = mask_gt.bool()
        overlaps = torch.zeros((bs, M, A), dtype=pd_bboxes.dtype, device=pd_bboxes.device)
        bbox_scores = torch.zeros((bs, M, A), dtype=pd_scores.dtype, device=pd_scores.device)

        batch_idx = torch.arange(bs, device=pd_scores.device)[:, None].expand(-1, M)
        cls_idx = gt_labels.squeeze(-1).clamp(0, self.nc - 1)
        bbox_scores[mask_gt] = pd_scores[batch_idx, :, cls_idx][mask_gt]

        pd_boxes = pd_bboxes[:, None].expand(-1, M, -1, -1)[mask_gt]
        gt_boxes = gt_bboxes[:, :, None].expand(-1, -1, A, -1)[mask_gt]
        if pd_boxes.numel():
            overlaps[mask_gt] = bbox_iou(gt_boxes, pd_boxes, CIoU=True).clamp(0)
        return bbox_scores.pow(self.alpha) * overlaps.pow(self.beta), overlaps

    def select_topk_candidates(self, metrics, topk_mask=None):
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, dim=-1)
        if topk_mask is None:
            topk_mask = (topk_metrics.amax(-1, keepdim=True) > self.eps).expand_as(topk_idxs)
        topk_idxs = torch.where(topk_mask, topk_idxs, 0)
        count = torch.zeros_like(metrics, dtype=torch.int8)
        count.scatter_add_(-1, topk_idxs, torch.ones_like(topk_idxs, dtype=torch.int8))
        return count.masked_fill_(count > 1, 0).to(metrics.dtype)

    @staticmethod
    def select_candidates_in_gts(anc_points, gt_bboxes, eps=1e-9):
        bs, M, _ = gt_bboxes.shape
        lt, rb = gt_bboxes.view(-1, 1, 4).chunk(2, 2)
        deltas = torch.cat((anc_points[None] - lt, rb - anc_points[None]), 2)
        return deltas.view(bs, M, anc_points.shape[0], 4).amin(3).gt_(eps)

    @staticmethod
    def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            multi = (fg_mask[:, None] > 1).expand(-1, n_max_boxes, -1)
            best = F.one_hot(overlaps.argmax(1), n_max_boxes).permute(0, 2, 1).to(overlaps.dtype)
            mask_pos = torch.where(multi, best, mask_pos)
            fg_mask = mask_pos.sum(-2)
        return mask_pos.argmax(-2), fg_mask, mask_pos

    def get_targets(self, gt_labels, gt_bboxes, target_gt_idx, fg_mask):
        bs = gt_labels.shape[0]
        idx = target_gt_idx + torch.arange(bs, device=gt_labels.device)[:, None] * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[idx].clamp(0)
        target_bboxes = gt_bboxes.view(-1, 4)[idx]
        target_scores = F.one_hot(target_labels, self.nc).float() * fg_mask.unsqueeze(-1)
        return target_labels, target_bboxes, target_scores


class BboxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes,
                target_scores, target_scores_sum, fg_mask):
        if not fg_mask.any():
            return pred_bboxes.sum() * 0, pred_dist.sum() * 0

        weight = target_scores.sum(-1)[fg_mask, None]
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], CIoU=True)
        loss_iou = ((1 - iou)[:, None] * weight).sum() / target_scores_sum

        target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max - 1)
        loss_dfl = self._df_loss(
            pred_dist[fg_mask].view(-1, self.reg_max), target_ltrb[fg_mask]
        )
        return loss_iou, (loss_dfl * weight).sum() / target_scores_sum

    @staticmethod
    def _df_loss(pred_dist, target):
        tl = target.long()
        tr = tl + 1
        wl, wr = tr - target, target - tl
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none").view_as(target)
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none").view_as(target)
        return (loss_l * wl + loss_r * wr).mean(-1, keepdim=True)

class DetectionLoss(nn.Module):
    def __init__(self, nc, reg_max=16, topk_o2m=10, topk_o2o=1, alpha=0.5, beta=6.0,
                 box_gain=7.5, cls_gain=1.0, dfl_gain=1.5, o2m_weight=1.0, o2o_weight=1.0):
        super().__init__()
        self.nc, self.reg_max = nc, reg_max
        self.box_gain, self.cls_gain, self.dfl_gain = box_gain, cls_gain, dfl_gain
        self.o2m_weight, self.o2o_weight = o2m_weight, o2o_weight
        self.assigner_o2m = TaskAlignedAssigner(topk_o2m, nc, alpha, beta)
        self.assigner_o2o = TaskAlignedAssigner(topk_o2o, nc, alpha, beta)
        self.bbox_loss = BboxLoss(reg_max)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    @staticmethod
    def preprocess_targets(targets, batch_size, device):
        n_max = max(max((len(t["boxes"]) for t in targets), default=0), 1)
        boxes = torch.zeros(batch_size, n_max, 4, device=device)
        labels = torch.zeros(batch_size, n_max, 1, dtype=torch.long, device=device)
        mask = torch.zeros(batch_size, n_max, 1, dtype=torch.bool, device=device)
        for i, t in enumerate(targets):
            n = len(t["boxes"])
            if n:
                boxes[i, :n] = t["boxes"].to(device)
                labels[i, :n, 0] = t["labels"].to(device)
                mask[i, :n, 0] = True
        return boxes, labels, mask

    def _branch_loss(self, assigner, cls_raw, box_pixel, reg_raw, anchors, strides,
                     gt_bboxes, gt_labels, mask_gt):
        # TAL/IoU cần FP32 để tránh lệch dtype với GT và underflow khi dùng AMP.
        cls_raw, box_pixel, reg_raw = cls_raw.float(), box_pixel.float(), reg_raw.float()
        anchors, strides = anchors.float(), strides.float()
        # TAL uses PIXEL; CIoU/DFL below use GRID.
        _, target_boxes, target_scores, fg_mask, _ = assigner(
            cls_raw.detach().sigmoid(), box_pixel.detach(), anchors * strides,
            gt_labels, gt_bboxes, mask_gt,
        )
        score_sum = target_scores.sum().clamp_min(1)
        loss_cls = self.bce(cls_raw, target_scores).sum() / score_sum

        stride = strides.unsqueeze(0)
        loss_iou, loss_dfl = self.bbox_loss(
            reg_raw.transpose(1, 2).contiguous(), box_pixel / stride, anchors,
            target_boxes / stride, target_scores, score_sum, fg_mask,
        )
        return loss_iou, loss_cls, loss_dfl, fg_mask.sum().item()

    def forward(self, preds, targets):
        anchors, strides = preds["anchors"], preds["strides"]
        gt = self.preprocess_targets(targets, preds["o2o"]["cls"].shape[0], anchors.device)

        def branch(name, assigner):
            p = preds[name]
            return self._branch_loss(
                assigner, p["cls"], p["box"], p["reg_raw"], anchors, strides, *gt
            )

        iou_m, cls_m, dfl_m, npos_m = branch("o2m", self.assigner_o2m)
        iou_o, cls_o, dfl_o, npos_o = branch("o2o", self.assigner_o2o)
        loss_o2m = self.box_gain * iou_m + self.cls_gain * cls_m + self.dfl_gain * dfl_m
        loss_o2o = self.box_gain * iou_o + self.cls_gain * cls_o + self.dfl_gain * dfl_o
        total = self.o2m_weight * loss_o2m + self.o2o_weight * loss_o2o

        items = {
            "loss": total.item(), "loss_o2m": loss_o2m.item(), "loss_o2o": loss_o2o.item(),
            "o2m/iou": iou_m.item(), "o2m/cls": cls_m.item(), "o2m/dfl": dfl_m.item(),
            "o2o/iou": iou_o.item(), "o2o/cls": cls_o.item(), "o2o/dfl": dfl_o.item(),
            "o2m/n_pos": npos_m, "o2o/n_pos": npos_o,
        }
        return total, items
