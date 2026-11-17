# =====================================================================
# File: loss.py
# Dự án: Driver Guardian - Module LiteFat_v1 (KSS 1-9, Sliding 60s/30s)
# Tác vụ: Định nghĩa hàm mục tiêu huấn luyện LiteFatLoss (Hybrid KSS Loss)
#         kết hợp:
#           1. Clip Regression Loss (SmoothL1 + MSE trên thang điểm 1..9)
#           2. Clip Ordinal Gaussian Soft-Label + CrossEntropy (logits 9 mức)
#           3. Causal Time-Weighted Top-K MIL Frame Loss thích nghi mức KSS
#           4. Sliding-Window Causal & Smoothness Loss cho chuỗi dài > 60s
# =====================================================================
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteFatLoss(nn.Module):
    """
    Hàm mất mát Hybrid KSS Loss cho mô hình LiteFat_v1 (Karolinska Sleepiness Scale 1..9):
      L_total = clip_reg_weight * L_clip_reg
              + clip_cls_weight * L_clip_cls
              + frame_loss_weight * L_frame_mil
              + window_loss_weight * L_window
              + window_smooth_weight * L_smooth

    Hỗ trợ linh hoạt mọi kiểu đầu ra của LiteFat_v1.forward():
      - Dict trả về khi `return_dict=True`
      - Tuple trả về khi `return_dict=False` (`(frame_kss, clip_kss)` hoặc kèm logits/state)
      - Truyền trực tiếp từng tensor hoặc theo chữ ký cũ `(frame_logits, targets, clip_logits=..., lengths=...)`.
    """

    def __init__(
            self,
            num_classes: int = 9,
            kss_min: float = 1.0,
            kss_max: float = 9.0,
            clip_reg_weight: float = 1.0,
            clip_cls_weight: float = 0.5,
            frame_loss_weight: float = 0.3,
            window_loss_weight: float = 0.25,
            window_smooth_weight: float = 0.05,
            dann_weight: float = 0.05,
            huber_beta: float = 0.5,
            ordinal_sigma: float = 0.85,
            label_smoothing: float = 0.05,
            alert_kss_threshold: float = 4.0,
            mid_kss_threshold: float = 6.0,
            topk_mid_ratio: float = 0.70,
            topk_drowsy_ratio: float = 0.45,
            fps: int = 10,
            window_sec: float = 60.0,
            stride_sec: float = 30.0,
            smooth_margin: float = 1.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.kss_min = float(kss_min)
        self.kss_max = float(kss_max)

        self.clip_reg_weight = float(clip_reg_weight)
        self.clip_cls_weight = float(clip_cls_weight)
        self.frame_loss_weight = float(frame_loss_weight)
        self.window_loss_weight = float(window_loss_weight)
        self.window_smooth_weight = float(window_smooth_weight)
        self.dann_weight = float(dann_weight)

        self.huber_beta = float(huber_beta)
        self.ordinal_sigma = float(max(1e-3, ordinal_sigma))
        self.label_smoothing = float(label_smoothing)
        self.alert_kss_threshold = float(alert_kss_threshold)
        self.mid_kss_threshold = float(mid_kss_threshold)
        self.topk_mid_ratio = float(topk_mid_ratio)
        self.topk_drowsy_ratio = float(topk_drowsy_ratio)

        self.fps = int(max(1, fps))
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.window_size = max(1, int(round(self.window_sec * self.fps)))
        self.stride_size = max(1, int(round(self.stride_sec * self.fps)))
        self.smooth_margin = float(smooth_margin)

        kss_levels = torch.linspace(self.kss_min, self.kss_max, steps=self.num_classes, dtype=torch.float32)
        self.register_buffer("kss_levels", kss_levels)
        self.ce_raw = nn.CrossEntropyLoss(reduction="none", label_smoothing=self.label_smoothing)

    def _logits_to_kss_expectation(self, logits: torch.Tensor) -> torch.Tensor:
        """Quy đổi logits 9 mức KSS thành điểm kỳ vọng liên tục trong [kss_min, kss_max]."""
        probs = F.softmax(logits.float(), dim=-1)
        view_shape = [1] * (logits.dim() - 1) + [self.num_classes]
        levels = self.kss_levels.view(*view_shape).to(logits.device)
        return (probs * levels).sum(dim=-1)

    def _ordinal_soft_cross_entropy(
            self,
            logits: torch.Tensor,
            target_kss: torch.Tensor,
    ) -> torch.Tensor:
        """
        Tính Ordinal Gaussian Soft-Label Cross-Entropy giữa phân phối logits [..., C]
        và mục tiêu liên tục target_kss [...] trên lưới mức KSS [1..9].
        """
        view_shape = [1] * (logits.dim() - 1) + [self.num_classes]
        levels = self.kss_levels.view(*view_shape).to(logits.device)
        dist_sq = (levels - target_kss.float().unsqueeze(-1)).pow(2)
        soft_targets = F.softmax(-dist_sq / (2.0 * (self.ordinal_sigma ** 2)), dim=-1)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        return -(soft_targets * log_probs).sum(dim=-1)

    def _classification_ordinal_loss(
            self,
            logits: torch.Tensor,
            target_kss: torch.Tensor,
            target_cls: torch.Tensor,
    ) -> torch.Tensor:
        """Kết hợp 65% Ordinal Gaussian Soft-Label CE + 35% CrossEntropy chuẩn."""
        ord_ce = self._ordinal_soft_cross_entropy(logits, target_kss)
        flat_logits = logits.float().reshape(-1, self.num_classes)
        flat_cls = target_cls.reshape(-1)
        std_ce = self.ce_raw(flat_logits, flat_cls).reshape(target_cls.shape)
        return 0.65 * ord_ce + 0.35 * std_ce

    def _unpack_predictions(
            self,
            predictions: Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...]],
            clip_kss: Optional[torch.Tensor],
            frame_logits: Optional[torch.Tensor],
            clip_logits: Optional[torch.Tensor],
            window_kss: Optional[torch.Tensor],
            window_logits: Optional[torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Chuẩn hóa đầu vào dự đoán từ dict, tuple hoặc các tensor riêng lẻ."""
        out: Dict[str, Optional[torch.Tensor]] = {
            "frame_kss": None,
            "clip_kss": clip_kss,
            "frame_logits": frame_logits,
            "clip_logits": clip_logits,
            "window_kss": window_kss,
            "window_logits": window_logits,
            "window_valid_mask": None,
            "domain_logits": None,
            "window_domain_logits": None,
        }

        if isinstance(predictions, dict):
            out["frame_kss"] = predictions.get("frame_kss", out["frame_kss"])
            out["clip_kss"] = predictions.get("clip_kss", out["clip_kss"])
            out["frame_logits"] = predictions.get("frame_logits", out["frame_logits"])
            out["clip_logits"] = predictions.get("clip_logits", out["clip_logits"])
            out["window_kss"] = predictions.get("window_kss", out["window_kss"])
            out["window_logits"] = predictions.get("window_logits", out["window_logits"])
            out["window_valid_mask"] = predictions.get("window_valid_mask", out["window_valid_mask"])
            out["domain_logits"] = predictions.get("domain_logits", out["domain_logits"])
            out["window_domain_logits"] = predictions.get("window_domain_logits", out["window_domain_logits"])
        elif isinstance(predictions, (tuple, list)):
            items = list(predictions)
            if items and isinstance(items[-1], dict):
                items = items[:-1]
            if len(items) == 1:
                t0 = items[0]
                if t0.dim() == 3:
                    out["frame_logits"] = t0
                else:
                    out["frame_kss"] = t0
            elif len(items) == 2:
                a, b = items
                if a.dim() == 2 and b.dim() == 1:
                    out["frame_kss"], out["clip_kss"] = a, b
                elif a.dim() == 2 and b.dim() == 3:
                    out["frame_kss"], out["frame_logits"] = a, b
                elif a.dim() == 3 and b.dim() == 2:
                    out["frame_logits"], out["clip_logits"] = a, b
                else:
                    out["frame_kss"] = a
                    if out["clip_kss"] is None and b.dim() == 1:
                        out["clip_kss"] = b
            elif len(items) >= 4:
                out["frame_kss"], out["clip_kss"], out["frame_logits"], out["clip_logits"] = items[:4]
        elif isinstance(predictions, torch.Tensor):
            if predictions.dim() == 3:
                out["frame_logits"] = predictions
            elif predictions.dim() == 2:
                out["frame_kss"] = predictions
            elif predictions.dim() == 1:
                out["clip_kss"] = predictions

        if out["frame_kss"] is None and out["frame_logits"] is not None:
            out["frame_kss"] = self._logits_to_kss_expectation(out["frame_logits"])
        if out["clip_kss"] is None and out["clip_logits"] is not None:
            out["clip_kss"] = self._logits_to_kss_expectation(out["clip_logits"])

        return out

    def _prepare_targets(
            self,
            targets: torch.Tensor,
            lengths: Optional[torch.Tensor],
            device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Chuẩn hóa targets về cả KSS liên tục [1.0, 9.0] và chỉ số lớp 0..8.
        Hỗ trợ targets dạng [B], [B, 1], hoặc [B, T].
        """
        tgt = targets.to(device)
        if tgt.dim() == 2 and tgt.size(1) == 1:
            tgt = tgt.squeeze(1)

        if tgt.dim() == 1:
            clip_target_kss = tgt.float().clamp(min=self.kss_min, max=self.kss_max)
            clip_target_cls = (clip_target_kss.round().long() - int(round(self.kss_min))).clamp(
                0, self.num_classes - 1
            )
            return clip_target_kss, clip_target_cls, None, None

        # Trường hợp targets dạng chuỗi theo khung hình [B, T]
        B, T = tgt.shape
        if lengths is not None:
            last_idx = (lengths.to(device).long() - 1).clamp(min=0, max=T - 1)
            clip_raw = tgt[torch.arange(B, device=device), last_idx]
        else:
            clip_raw = tgt[:, -1]

        clip_target_kss = clip_raw.float().clamp(min=self.kss_min, max=self.kss_max)
        clip_target_cls = (clip_target_kss.round().long() - int(round(self.kss_min))).clamp(
            0, self.num_classes - 1
        )
        frame_target_kss = tgt.float().clamp(min=self.kss_min, max=self.kss_max)
        frame_target_cls = (frame_target_kss.round().long() - int(round(self.kss_min))).clamp(
            0, self.num_classes - 1
        )
        return clip_target_kss, clip_target_cls, frame_target_kss, frame_target_cls

    @staticmethod
    def _weighted_mean(
            values: torch.Tensor,
            sample_weights: Optional[torch.Tensor] = None,
            valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Tính trung bình có trọng số mẫu trên batch (hỗ trợ sample_weights và valid_mask)."""
        val = values.float()
        weights = torch.ones_like(val)
        if sample_weights is not None:
            weights = weights * sample_weights.float().clamp(min=0.0).to(val.device)
        if valid_mask is not None:
            weights = weights * valid_mask.float().to(val.device)
        denom = weights.sum().clamp(min=1e-6)
        return (val * weights).sum() / denom

    def _compute_frame_mil_loss(
            self,
            frame_kss: Optional[torch.Tensor],
            frame_logits: Optional[torch.Tensor],
            clip_target_kss: torch.Tensor,
            clip_target_cls: torch.Tensor,
            frame_target_kss: Optional[torch.Tensor],
            frame_target_cls: Optional[torch.Tensor],
            lengths: Optional[torch.Tensor],
            sample_weights: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Causal Time-Weighted Top-K MIL Frame Loss cho KSS 1..9:
          - Trọng số nhân quả tăng dần 0.5 -> 1.0 theo độ dài hợp lệ từng mẫu.
          - KSS <= 4: giám sát 100% frame hợp lệ.
          - 4 < KSS <= 6: giám sát Top-K (70%) frame xu hướng mệt mỏi cao nhất.
          - KSS > 6: giám sát Top-K (45%) frame xu hướng buồn ngủ cao nhất.
        """
        ref = frame_kss if frame_kss is not None else frame_logits
        assert ref is not None
        B, T = ref.shape[:2]
        dev = ref.device

        if lengths is not None:
            valid_lens = lengths.to(dev).long().clamp(min=0, max=T)
            t_idx = torch.arange(T, device=dev).unsqueeze(0)
            valid_mask = t_idx < valid_lens.unsqueeze(1)
            # Ramp nhân quả tính riêng trên độ dài hợp lệ của từng mẫu (bất biến với batch padding)
            ramp_denom = (valid_lens.float() - 1.0).clamp(min=1.0).unsqueeze(1)
            ramp = torch.where(
                valid_lens.unsqueeze(1) > 1,
                0.5 + 0.5 * (t_idx.float() / ramp_denom),
                torch.ones((B, T), device=dev, dtype=torch.float32),
            )
            weight_matrix = ramp * valid_mask.float()
        else:
            valid_lens = torch.full((B,), T, dtype=torch.long, device=dev)
            valid_mask = torch.ones((B, T), dtype=torch.bool, device=dev)
            ramp = (0.5 + 0.5 * torch.linspace(0.0, 1.0, T, device=dev, dtype=torch.float32)).unsqueeze(0).expand(B, T)
            weight_matrix = ramp.clone()

        # Điểm xu hướng buồn ngủ của từng frame dùng để xếp hạng MIL Top-K
        if frame_kss is not None:
            drowsy_scores = frame_kss.float()
        else:
            assert frame_logits is not None
            drowsy_scores = self._logits_to_kss_expectation(frame_logits)
        drowsy_scores = drowsy_scores.masked_fill(~valid_mask, -1e4)

        mil_mask = valid_mask.clone()
        for b in range(B):
            t_len = int(valid_lens[b].item())
            if t_len <= 1:
                continue
            y_kss = float(clip_target_kss[b].item())
            if y_kss <= self.alert_kss_threshold:
                ratio = 1.0
            elif y_kss <= self.mid_kss_threshold:
                ratio = self.topk_mid_ratio
            else:
                ratio = self.topk_drowsy_ratio

            if ratio < 0.999:
                k_b = max(1, min(t_len, int(round(t_len * ratio))))
                if k_b < t_len:
                    _, topk_idx = torch.topk(drowsy_scores[b, :t_len], k=k_b, dim=0)
                    sample_mask = torch.zeros(T, dtype=torch.bool, device=dev)
                    sample_mask[topk_idx] = True
                    mil_mask[b] = sample_mask

        weight_matrix = weight_matrix * mil_mask.float()
        norm_weights = weight_matrix / weight_matrix.sum(dim=1, keepdim=True).clamp(min=1e-6)

        tgt_f_kss = (
            frame_target_kss.to(dev)
            if frame_target_kss is not None
            else clip_target_kss.unsqueeze(1).expand(B, T)
        )
        tgt_f_cls = (
            frame_target_cls.to(dev)
            if frame_target_cls is not None
            else clip_target_cls.unsqueeze(1).expand(B, T)
        )

        loss_reg_per = None
        if frame_kss is not None:
            loss_reg_per = F.smooth_l1_loss(
                frame_kss.float(),
                tgt_f_kss,
                beta=self.huber_beta,
                reduction="none",
            )

        loss_cls_per = None
        if frame_logits is not None:
            loss_cls_per = self._classification_ordinal_loss(frame_logits, tgt_f_kss, tgt_f_cls)

        if loss_reg_per is not None and loss_cls_per is not None:
            per_frame_loss = 0.65 * loss_reg_per + 0.35 * loss_cls_per
        elif loss_reg_per is not None:
            per_frame_loss = loss_reg_per
        else:
            assert loss_cls_per is not None
            per_frame_loss = loss_cls_per

        per_sample_loss = (per_frame_loss * norm_weights).sum(dim=1)
        active_samples = valid_lens > 0
        total_frame_loss = self._weighted_mean(per_sample_loss, sample_weights=sample_weights, valid_mask=active_samples)
        return total_frame_loss, mil_mask

    def _compute_window_loss(
            self,
            window_kss: Optional[torch.Tensor],
            window_logits: Optional[torch.Tensor],
            clip_target_kss: torch.Tensor,
            clip_target_cls: torch.Tensor,
            lengths: Optional[torch.Tensor],
            sample_weights: Optional[torch.Tensor],
            window_valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sliding-Window Causal & Smoothness Loss cho chuỗi dài > 60s:
          - Phạt sai số KSS/Ordinal trên từng cửa sổ trượt hợp lệ với trọng số tăng dần theo thời gian.
          - Phạt dao động KSS đột ngột vượt quá `smooth_margin` giữa 2 cửa sổ chồng lấp liên tiếp.
        """
        ref = window_kss if window_kss is not None else window_logits
        if ref is None or ref.size(1) == 0:
            dev = clip_target_kss.device
            zero = torch.tensor(0.0, device=dev, dtype=torch.float32)
            return zero, zero

        B, W_num = ref.shape[:2]
        dev = ref.device

        if window_valid_mask is not None and window_valid_mask.shape == (B, W_num):
            win_valid = window_valid_mask.to(dev).bool()
        elif lengths is not None:
            lens = lengths.to(dev).long()
            win_starts = torch.arange(W_num, device=dev) * self.stride_size
            win_valid = lens.unsqueeze(1) > win_starts.unsqueeze(0)
            win_valid[:, 0] = win_valid[:, 0] | (lens > 0)
        else:
            win_valid = torch.ones((B, W_num), dtype=torch.bool, device=dev)

        valid_w_counts = win_valid.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        w_idx = torch.arange(W_num, device=dev, dtype=torch.float32).unsqueeze(0)
        w_ramp = torch.where(
            valid_w_counts > 1.0,
            0.6 + 0.4 * (w_idx / (valid_w_counts - 1.0).clamp(min=1.0)),
            torch.ones((B, W_num), device=dev, dtype=torch.float32),
        )
        w_weights = w_ramp * win_valid.float()
        w_norm = w_weights / w_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

        tgt_w_kss = clip_target_kss.unsqueeze(1).expand(B, W_num)
        tgt_w_cls = clip_target_cls.unsqueeze(1).expand(B, W_num)

        win_reg = None
        if window_kss is not None:
            win_reg = F.smooth_l1_loss(
                window_kss.float(),
                tgt_w_kss,
                beta=self.huber_beta,
                reduction="none",
            )

        win_cls = None
        if window_logits is not None:
            win_cls = self._classification_ordinal_loss(window_logits, tgt_w_kss, tgt_w_cls)

        if win_reg is not None and win_cls is not None:
            per_win_loss = 0.65 * win_reg + 0.35 * win_cls
        elif win_reg is not None:
            per_win_loss = win_reg
        else:
            assert win_cls is not None
            per_win_loss = win_cls

        per_sample_win_loss = (per_win_loss * w_norm).sum(dim=1)
        active_samples = win_valid.any(dim=1)
        loss_window = self._weighted_mean(
            per_sample_win_loss,
            sample_weights=sample_weights,
            valid_mask=active_samples,
        )

        # Điều chuẩn độ mượt liên cửa sổ (chỉ tính khi có ít nhất 2 cửa sổ hợp lệ)
        if W_num >= 2 and window_kss is not None and self.window_smooth_weight > 0:
            pair_valid = win_valid[:, 1:] & win_valid[:, :-1]
            if pair_valid.any():
                diff = torch.abs(window_kss.float()[:, 1:] - window_kss.float()[:, :-1])
                excess = torch.clamp(diff - self.smooth_margin, min=0.0).pow(2) * pair_valid.float()
                pair_counts = pair_valid.float().sum(dim=1).clamp(min=1.0)
                per_sample_smooth = excess.sum(dim=1) / pair_counts
                has_pairs = pair_valid.any(dim=1)
                loss_smooth = self._weighted_mean(
                    per_sample_smooth,
                    sample_weights=sample_weights,
                    valid_mask=has_pairs,
                )
            else:
                loss_smooth = torch.tensor(0.0, device=dev, dtype=torch.float32)
        else:
            loss_smooth = torch.tensor(0.0, device=dev, dtype=torch.float32)

        return loss_window, loss_smooth

    def forward(
            self,
            predictions: Union[torch.Tensor, Dict[str, Any], Tuple[Any, ...]],
            targets: torch.Tensor,
            clip_logits: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            clip_kss: Optional[torch.Tensor] = None,
            frame_logits: Optional[torch.Tensor] = None,
            window_kss: Optional[torch.Tensor] = None,
            window_logits: Optional[torch.Tensor] = None,
            sample_weights: Optional[torch.Tensor] = None,
            domain_targets: Optional[torch.Tensor] = None,
            return_dict: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, float]]]:
        """
        Tính tổng hàm mất mát Hybrid KSS Loss cho một batch dự đoán từ LiteFat_v1.

        Args:
            predictions: Dict (`return_dict=True`), Tuple từ `LiteFat_v1(...)`,
                         hoặc `frame_kss` [B, T] / `frame_logits` [B, T, 9].
            targets: Nhãn KSS mục tiêu [B] (int64 1..9 hoặc float 1.0..9.0), hoặc [B, T].
            clip_logits: Logits toàn mẫu [B, 9] (tùy chọn).
            lengths: Độ dài thực tế từng mẫu trong batch [B].
            clip_kss: Điểm KSS dự đoán toàn mẫu [B] (tùy chọn).
            frame_logits: Logits từng frame [B, T, 9] (tùy chọn).
            window_kss: Điểm KSS từng cửa sổ trượt [B, W_num] (tùy chọn).
            window_logits: Logits từng cửa sổ trượt [B, W_num, 9] (tùy chọn).
            sample_weights: Trọng số cân bằng mẫu [B] từ manifest (tùy chọn).
            domain_targets: Nhãn miền [B] (0=UL-DD, 1=UTA-RLDD) dùng cho DANN (tùy chọn).
            return_dict: True để trả kèm dict các thành phần loss chi tiết.
        """
        unpacked = self._unpack_predictions(
            predictions=predictions,
            clip_kss=clip_kss,
            frame_logits=frame_logits,
            clip_logits=clip_logits,
            window_kss=window_kss,
            window_logits=window_logits,
        )

        f_kss = unpacked["frame_kss"]
        c_kss = unpacked["clip_kss"]
        f_logits = unpacked["frame_logits"]
        c_logits = unpacked["clip_logits"]
        w_kss = unpacked["window_kss"]
        w_logits = unpacked["window_logits"]
        w_valid_mask = unpacked.get("window_valid_mask")
        d_logits = unpacked.get("domain_logits")
        w_d_logits = unpacked.get("window_domain_logits")

        ref_tensor = next(t for t in (c_kss, c_logits, f_kss, f_logits, w_kss, w_logits) if t is not None)
        dev = ref_tensor.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"

        with torch.amp.autocast(dev_type, enabled=False):
            clip_target_kss, clip_target_cls, frame_target_kss, frame_target_cls = self._prepare_targets(
                targets=targets,
                lengths=lengths,
                device=dev,
            )

            # Fallback khi caller chỉ trả về đầu ra cấp frame (return_clip_logits=False)
            if c_kss is None and f_kss is not None:
                if lengths is not None:
                    T_f = f_kss.size(1)
                    mask_f = (torch.arange(T_f, device=dev).unsqueeze(0) < lengths.to(dev).unsqueeze(1)).float()
                    c_kss = (f_kss.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
                else:
                    c_kss = f_kss.float().mean(dim=1)
            if c_logits is None and f_logits is not None:
                if lengths is not None:
                    T_f = f_logits.size(1)
                    mask_f = (
                        torch.arange(T_f, device=dev).unsqueeze(0) < lengths.to(dev).unsqueeze(1)
                    ).unsqueeze(-1).float()
                    c_logits = (f_logits.float() * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
                else:
                    c_logits = f_logits.float().mean(dim=1)

            # 1. Clip Regression Loss (SmoothL1 + MSE)
            if c_kss is not None and self.clip_reg_weight > 0:
                c_kss_32 = c_kss.float()
                smooth_l1 = F.smooth_l1_loss(c_kss_32, clip_target_kss, beta=self.huber_beta, reduction="none")
                mse = F.mse_loss(c_kss_32, clip_target_kss, reduction="none")
                loss_clip_reg = self._weighted_mean(0.75 * smooth_l1 + 0.25 * mse, sample_weights=sample_weights)
            else:
                loss_clip_reg = torch.tensor(0.0, device=dev, dtype=torch.float32)

            # 2. Clip Ordinal Gaussian Soft-Label + CrossEntropy Loss
            if c_logits is not None and self.clip_cls_weight > 0:
                cls_loss_vec = self._classification_ordinal_loss(c_logits, clip_target_kss, clip_target_cls)
                loss_clip_cls = self._weighted_mean(cls_loss_vec, sample_weights=sample_weights)
            else:
                loss_clip_cls = torch.tensor(0.0, device=dev, dtype=torch.float32)

            # 3. Causal Time-Weighted Top-K MIL Frame Loss
            if self.frame_loss_weight > 0 and (f_kss is not None or f_logits is not None):
                loss_frame, _ = self._compute_frame_mil_loss(
                    frame_kss=f_kss,
                    frame_logits=f_logits,
                    clip_target_kss=clip_target_kss,
                    clip_target_cls=clip_target_cls,
                    frame_target_kss=frame_target_kss,
                    frame_target_cls=frame_target_cls,
                    lengths=lengths,
                    sample_weights=sample_weights,
                )
            else:
                loss_frame = torch.tensor(0.0, device=dev, dtype=torch.float32)

            # 4. Sliding-Window Causal & Smoothness Loss
            if (self.window_loss_weight > 0 or self.window_smooth_weight > 0) and (
                w_kss is not None or w_logits is not None
            ):
                loss_window, loss_smooth = self._compute_window_loss(
                    window_kss=w_kss,
                    window_logits=w_logits,
                    clip_target_kss=clip_target_kss,
                    clip_target_cls=clip_target_cls,
                    lengths=lengths,
                    sample_weights=sample_weights,
                    window_valid_mask=w_valid_mask,
                )
            else:
                loss_window = torch.tensor(0.0, device=dev, dtype=torch.float32)
                loss_smooth = torch.tensor(0.0, device=dev, dtype=torch.float32)

            # 5. Regime-Conditioned Domain-Adversarial Loss (DANN)
            loss_domain = torch.tensor(0.0, device=dev, dtype=torch.float32)
            if self.dann_weight > 0 and domain_targets is not None and (
                d_logits is not None or w_d_logits is not None
            ):
                dom_tgt = domain_targets.to(dev).long().view(-1)
                dom_valid = dom_tgt >= 0
                if dom_valid.any():
                    loss_d_clip = None
                    if d_logits is not None:
                        per_sample_d_clip = F.cross_entropy(
                            d_logits.float(),
                            dom_tgt.clamp(min=0),
                            reduction="none",
                        )
                        loss_d_clip = self._weighted_mean(
                            per_sample_d_clip,
                            sample_weights=sample_weights,
                            valid_mask=dom_valid,
                        )

                    loss_d_win = None
                    if w_d_logits is not None and w_d_logits.size(1) > 0:
                        B_w, W_w, C_w = w_d_logits.shape
                        if w_valid_mask is not None and w_valid_mask.shape == (B_w, W_w):
                            win_v = w_valid_mask.to(dev).bool()
                        elif lengths is not None:
                            lens_w = lengths.to(dev).long()
                            ws_w = torch.arange(W_w, device=dev) * self.stride_size
                            win_v = lens_w.unsqueeze(1) > ws_w.unsqueeze(0)
                            win_v[:, 0] = win_v[:, 0] | (lens_w > 0)
                        else:
                            win_v = torch.ones((B_w, W_w), dtype=torch.bool, device=dev)

                        tgt_w_dom = dom_tgt.clamp(min=0).unsqueeze(1).expand(B_w, W_w)
                        per_win_d = F.cross_entropy(
                            w_d_logits.float().reshape(-1, C_w),
                            tgt_w_dom.reshape(-1),
                            reduction="none",
                        ).reshape(B_w, W_w)
                        per_sample_d_win = (per_win_d * win_v.float()).sum(dim=1) / win_v.float().sum(dim=1).clamp(min=1.0)
                        active_d_win = win_v.any(dim=1) & dom_valid
                        loss_d_win = self._weighted_mean(
                            per_sample_d_win,
                            sample_weights=sample_weights,
                            valid_mask=active_d_win,
                        )

                    if loss_d_clip is not None and loss_d_win is not None:
                        loss_domain = 0.5 * loss_d_clip + 0.5 * loss_d_win
                    elif loss_d_clip is not None:
                        loss_domain = loss_d_clip
                    elif loss_d_win is not None:
                        loss_domain = loss_d_win

            total_loss = (
                self.clip_reg_weight * loss_clip_reg
                + self.clip_cls_weight * loss_clip_cls
                + self.frame_loss_weight * loss_frame
                + self.window_loss_weight * loss_window
                + self.window_smooth_weight * loss_smooth
                + self.dann_weight * loss_domain
            )

        if return_dict:
            details = {
                "loss_total": float(total_loss.detach().item()),
                "loss_clip_reg": float(loss_clip_reg.detach().item()),
                "loss_clip_cls": float(loss_clip_cls.detach().item()),
                "loss_frame": float(loss_frame.detach().item()),
                "loss_window": float(loss_window.detach().item()),
                "loss_smooth": float(loss_smooth.detach().item()),
                "loss_domain": float(loss_domain.detach().item()),
            }
            return total_loss, details

        return total_loss


LiteFatV1Loss = LiteFatLoss
