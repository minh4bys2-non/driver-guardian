# =====================================================================
# File: loss.py
# Dự án: Driver Guardian - Module LiteFat
# Tác vụ: Định nghĩa hàm mục tiêu huấn luyện LiteFatLoss kết hợp
#         mất mát cấp đoạn video (Clip CrossEntropy) và mất mát
#         từng khung hình có trọng số thời gian nhân quả (Causal Frame Loss).
# =====================================================================
from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LiteFatLoss(nn.Module):
    """
    Hàm mất mát hỗn hợp (Hybrid Causal Loss & Multiple Instance Learning - MIL) cho mô hình LiteFat:
    L_total = clip_loss_weight * L_clip + frame_loss_weight * L_frame

    Trong đó:
      - L_clip: CrossEntropyLoss với label_smoothing trên clip_logits [B, num_classes].
      - L_frame: Multiple Instance Learning Frame Loss:
        + Với clip Alert (0): Giám sát 100% frame hợp lệ với nhãn 0.
        + Với clip Drowsy (1): Chỉ giám sát Top-K (45%) frame có xu hướng buồn ngủ cao nhất,
          tránh ép nhãn 1 cho các frame tỉnh táo bình thường trong clip buồn ngủ (ngăn ngừa kéo tụt bias âm).
        + Kết hợp trọng số nhân quả tăng dần w_t để TCN tích lũy đủ ngữ cảnh.
    """

    def __init__(
            self,
            clip_loss_weight: float = 1.0,
            frame_loss_weight: float = 0.3,
            label_smoothing: float = 0.1,
            topk_frame_ratio: float = 0.45,
    ):
        super().__init__()
        self.clip_loss_weight = float(clip_loss_weight)
        self.frame_loss_weight = float(frame_loss_weight)
        self.topk_frame_ratio = float(topk_frame_ratio)
        self.ce = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)
        self.frame_ce_raw = nn.CrossEntropyLoss(reduction="none", label_smoothing=label_smoothing)

    def forward(
            self,
            frame_logits: torch.Tensor,
            targets: torch.Tensor,
            clip_logits: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            frame_logits: Logits từng khung hình [B, T, num_classes].
            targets: Nhãn nhị phân / đa lớp [B] (hoặc [B, T]).
            clip_logits: Logits gộp toàn đoạn [B, num_classes] (tùy chọn).
            lengths: Độ dài frame thực tế của mỗi mẫu trong batch [B].
        """
        B, T, num_classes = frame_logits.shape
        device = frame_logits.device

        # Chuẩn hóa nhãn targets
        if targets.dim() == 1:
            clip_targets = targets
            frame_targets = targets.unsqueeze(1).expand(B, T)
        else:
            clip_targets = targets[:, -1]
            frame_targets = targets

        # 1. Tính Clip Loss
        if clip_logits is not None:
            loss_clip = self.ce(clip_logits, clip_targets)
        else:
            # Fallback nếu không truyền clip_logits: lấy trung bình frame_logits
            if lengths is not None:
                mask_t = (torch.arange(T, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)).unsqueeze(-1).float()
                avg_clip_logits = (frame_logits * mask_t).sum(dim=1) / mask_t.sum(dim=1).clamp(min=1.0)
            else:
                avg_clip_logits = frame_logits.mean(dim=1)
            loss_clip = self.ce(avg_clip_logits, clip_targets)

        # 2. Tính Causal Time-Weighted Top-K MIL Frame Loss
        if self.frame_loss_weight > 0:
            # Trọng số thời gian nhân quả mượt mà: tỷ lệ 0.5 ở frame đầu đến 1.0 ở frame cuối
            base_weights = 0.5 + 0.5 * torch.linspace(0.0, 1.0, T, device=device, dtype=torch.float32)
            time_weights = base_weights / base_weights.sum()  # [T]
            weight_matrix = time_weights.unsqueeze(0).expand(B, T).clone()  # [B, T]

            # Mặt nạ loại bỏ frame đệm (padding)
            if lengths is not None:
                valid_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)
            else:
                valid_mask = torch.ones((B, T), dtype=torch.bool, device=device)
            weight_matrix = weight_matrix * valid_mask.float()

            # Multiple Instance Learning (MIL) Frame Selection:
            # - Với mẫu Alert (0): toàn bộ valid frames là tỉnh táo -> giám sát 100% valid frames.
            # - Với mẫu Drowsy (1): chỉ giám sát Top-K (45%) frames có xác suất buồn ngủ cao nhất.
            drowsy_scores = frame_logits[:, :, 1] - frame_logits[:, :, 0]  # [B, T]
            drowsy_scores = drowsy_scores.masked_fill(~valid_mask, -1e4)

            mil_mask = valid_mask.clone()
            for b in range(B):
                if clip_targets[b] == 1:
                    t_len = int(lengths[b].item()) if lengths is not None else T
                    k_b = max(1, int(round(t_len * self.topk_frame_ratio)))
                    if k_b < t_len:
                        _, topk_idx = torch.topk(drowsy_scores[b, :t_len], k=k_b, dim=0)
                        sample_mask = torch.zeros(T, dtype=torch.bool, device=device)
                        sample_mask[topk_idx] = True
                        mil_mask[b] = sample_mask

            weight_matrix = weight_matrix * mil_mask.float()

            # Chuẩn hóa trọng số trên từng sample để mỗi video đóng góp đồng đều
            sample_weight_sum = weight_matrix.sum(dim=1, keepdim=True).clamp(min=1e-6)
            norm_weights = weight_matrix / sample_weight_sum  # [B, T]

            # Tính CE riêng cho từng frame [B * T] -> [B, T]
            ce_per_frame = self.frame_ce_raw(
                frame_logits.reshape(B * T, num_classes),
                frame_targets.reshape(B * T)
            ).reshape(B, T)

            loss_frame = (ce_per_frame * norm_weights).sum(dim=1).mean()
        else:
            loss_frame = torch.tensor(0.0, device=device)

        total_loss = self.clip_loss_weight * loss_clip + self.frame_loss_weight * loss_frame
        return total_loss
