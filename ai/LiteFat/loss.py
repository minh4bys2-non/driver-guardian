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
    Hàm mất mát hỗn hợp (Hybrid Causal Loss) cho mô hình LiteFat:
    L_total = clip_loss_weight * L_clip + frame_loss_weight * L_frame

    Trong đó:
      - L_clip: CrossEntropyLoss với label_smoothing trên clip_logits [B, num_classes].
      - L_frame: Causal Time-Weighted CrossEntropyLoss trên frame_logits [B, T, num_classes]:
        Trọng số w_t = (t + 1) / sum(t + 1) tăng dần theo thời gian nhân quả,
        phản ánh việc TCN nhân quả cần một vài frame ban đầu để tích lũy ngữ cảnh.
    """

    def __init__(
            self,
            clip_loss_weight: float = 1.0,
            frame_loss_weight: float = 0.3,
            label_smoothing: float = 0.1,
    ):
        super().__init__()
        self.clip_loss_weight = float(clip_loss_weight)
        self.frame_loss_weight = float(frame_loss_weight)
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

        # 2. Tính Causal Time-Weighted Frame Loss
        if self.frame_loss_weight > 0:
            # Trọng số thời gian nhân quả w_t: tăng tuyến tính từ frame 0 đến frame T-1
            time_weights = torch.arange(1, T + 1, device=device, dtype=torch.float32)
            time_weights = time_weights / time_weights.sum()  # Tổng = 1.0, shape: [T]
            weight_matrix = time_weights.unsqueeze(0).expand(B, T)  # [B, T]

            # Mặt nạ loại bỏ frame đệm (padding)
            if lengths is not None:
                valid_mask = torch.arange(T, device=device).unsqueeze(0) < lengths.to(device).unsqueeze(1)
                weight_matrix = weight_matrix * valid_mask.float()

            denom = weight_matrix.sum().clamp(min=1e-6)

            # Tính CE riêng cho từng frame [B * T] -> [B, T]
            ce_per_frame = self.frame_ce_raw(
                frame_logits.reshape(B * T, num_classes),
                frame_targets.reshape(B * T)
            ).reshape(B, T)

            loss_frame = (ce_per_frame * weight_matrix).sum() / denom
        else:
            loss_frame = torch.tensor(0.0, device=device)

        total_loss = self.clip_loss_weight * loss_clip + self.frame_loss_weight * loss_frame
        return total_loss
