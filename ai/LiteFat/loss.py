import torch
import torch.nn as nn


class LiteFatLoss(nn.Module):
    """
    Hàm mục tiêu huấn luyện theo Phương trình (6):
        L = - (1 / T) * sum_{i=1}^{T} y_i * log(y_hat_i)
    """

    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, frame_logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_logits: [B, T, num_classes]
            targets: Nhãn mức khung hình [B, T] hoặc nhãn mức đoạn video [B]
        """
        B, T, C = frame_logits.shape
        if targets.dim() == 1:
            # Nếu nhãn được gán cho cả đoạn clip, lặp nhãn cho T khung hình
            targets = targets.unsqueeze(1).expand(B, T)
        return self.ce(frame_logits.reshape(B * T, C), targets.reshape(B * T))
