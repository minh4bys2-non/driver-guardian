from typing import Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from configs.config import TrainConfig
except ImportError:
    try:
        from config import TrainConfig
    except ImportError:
        TrainConfig = Any


class DrowsinessLoss(nn.Module):
    """
    Hàm mất mát Cross-Entropy Loss tiêu chuẩn cho mô hình nhận diện buồn ngủ Deep LSTM / GRU.
    
    Hỗ trợ cả 2 trường hợp:
    1. Giám sát toàn bộ chuỗi thời gian (Sequence-level supervision): 
       Logits có shape [B, T, C=2], targets có shape [B].
       Tự động reshape logits thành [B * T, C] và mở rộng targets thành [B * T] để tính loss từng frame.
    2. Giám sát tại khung hình cuối (Clip-level / Final frame supervision):
       Logits có shape [B, C=2], targets có shape [B].
    """

    def __init__(self, weight: Optional[torch.Tensor] = None, reduction: str = "mean"):
        super(DrowsinessLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    @classmethod
    def from_config(cls, config: Any) -> "DrowsinessLoss":
        """Khởi tạo DrowsinessLoss từ TrainConfig."""
        pos_w = getattr(config, "pos_weight", None)
        weight = None
        if pos_w is not None and pos_w > 0:
            weight = torch.tensor([1.0, float(pos_w)], dtype=torch.float32)
        return cls(weight=weight, reduction="mean")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tính toán Cross-Entropy Loss.

        Args:
            logits (torch.Tensor): Raw logits đầu ra từ mô hình [B, T, C] hoặc [B, C].
            targets (torch.Tensor): Nhãn nguyên bản Ground Truth [B] hoặc [B, T].

        Returns:
            torch.Tensor: Giá trị loss vô hướng (scalar).
        """
        if logits.dim() == 3:
            b, t, c = logits.shape
            logits_flat = logits.reshape(b * t, c)
            if targets.dim() == 1:
                targets_expanded = targets.unsqueeze(1).expand(b, t).reshape(b * t)
            else:
                targets_expanded = targets.reshape(b * t)
            return self.criterion(logits_flat, targets_expanded)

        return self.criterion(logits, targets)


class DrowsinessBCELoss(nn.Module):
    """
    Hàm mất mát Binary Cross-Entropy Loss với Pos-Weight (dành cho bài toán phân loại nhị phân).
    """

    def __init__(self, pos_weight: Optional[float] = None, reduction: str = "mean", eps: float = 1e-7):
        super(DrowsinessBCELoss, self).__init__()
        pw_tensor = torch.tensor([pos_weight], dtype=torch.float32) if pos_weight is not None else None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor, reduction=reduction)
        self.eps = eps

    @classmethod
    def from_config(cls, config: Any) -> "DrowsinessBCELoss":
        """Khởi tạo DrowsinessBCELoss từ TrainConfig."""
        return cls(
            pos_weight=getattr(config, "pos_weight", None),
            reduction=getattr(config, "bce_reduction", "mean"),
            eps=getattr(config, "bce_eps", 1e-7)
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Tính toán BCEWithLogitsLoss sau khi làm sạch logits và targets."""
        if logits.shape[-1] == 2:
            logits = logits[..., 1]  # Lấy logit của class 1 (Drowsy)
        logits = logits.squeeze(-1) if logits.dim() > targets.dim() else logits

        if logits.dim() == 2 and targets.dim() == 1:
            b, t = logits.shape
            targets = targets.unsqueeze(1).expand(b, t)

        return self.criterion(logits, targets.float())


def build_loss(config: Any) -> nn.Module:
    """Factory function khởi tạo Loss function phù hợp theo config."""
    loss_type = getattr(config, "loss_type", "ce").lower()
    if loss_type == "ce":
        return DrowsinessLoss.from_config(config)
    elif loss_type == "bce":
        return DrowsinessBCELoss.from_config(config)
    else:
        raise ValueError(f"Loss type không hợp lệ: '{loss_type}'. Chọn 'ce' hoặc 'bce'.")
