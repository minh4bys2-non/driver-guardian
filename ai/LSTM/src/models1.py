import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

# Đảm bảo đường dẫn thư mục gốc dự án được thêm vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ConvGRUClassifier
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN


class DrowsyClassifier(nn.Module):
    """
    Mô hình Phân loại Trạng thái Buồn ngủ Toàn diện End-to-End Spatio-Temporal.
    
    Luồng xử lý (Data Flow):
      Video [B, T, 3, H, W]
        -> Reshape [B*T, 3, H, W]
        -> Backbone & PAFPN (Gradient Checkpointing tùy chọn)
        -> Trích xuất 3 mức đặc trưng không gian: p3, p4, p5
        -> Reshape [B, T, C, H_feat, W_feat]
        -> ConvGRUClassifier (SpatialReductionNeck -> ConvGRU -> Dual Attention -> FC Head)
        -> Logits phân loại buồn ngủ.
    """

    def __init__(
            self,
            num_classes: int = 2,
            backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256),
            backbone_n: Tuple[int, ...] = (1, 2, 2, 1),
            neck_n: int = 1,
            input_dim: int = 64,
            hidden_dim: Union[int, Tuple[int, ...], List[int]] = (128, 128),
            target_size: Tuple[int, int] = (14, 14),
            supervision_mode: str = "attention_pooling",
            gradient_checkpointing: bool = False,
            freeze_bn: bool = False,
    ):
        """
        Khởi tạo DrowsyClassifier End-to-End.

        Args:
            num_classes (int): Số lớp phân loại (mặc định 2: Tỉnh táo / Buồn ngủ).
            backbone_w (Tuple[int, ...]): Độ rộng các kênh của Backbone (mặc định (16, 32, 64, 128, 256)).
            backbone_n (Tuple[int, ...]): Số lượng khối lặp lại trong từng tầng Backbone.
            neck_n (int): Số lượng khối lặp lại trong PAFPN.
            input_dim (int): Số kênh nén không gian đầu vào của ConvGRU (mặc định 64).
            hidden_dim (Union[int, Tuple, List]): Số kênh ẩn trong các tầng ConvGRU.
            target_size (Tuple[int, int]): Kích thước lưới không gian hội tụ (mặc định (14, 14)).
            supervision_mode (str): 'attention_pooling' (clip-level) hoặc 'sequence' (frame-level).
            gradient_checkpointing (bool): Bật checkpointing cho Backbone/PAFPN để tiết kiệm VRAM khi train.
            freeze_bn (bool): Khóa thống kê running mean/var của BatchNorm trong Backbone/PAFPN.
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.backbone_w = tuple(backbone_w)
        self.backbone_n = tuple(backbone_n)
        self.neck_n = int(neck_n)
        self.input_dim = int(input_dim)
        self.hidden_dim = hidden_dim
        self.target_size = tuple(target_size)
        self.supervision_mode = str(supervision_mode).lower()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.freeze_bn = bool(freeze_bn)

        # 1. Trích xuất đặc trưng không gian (Spatial Backbone)
        self.backbone = Backbone(w=self.backbone_w, n=self.backbone_n)
        c3, c4, c5 = self.backbone_w[2], self.backbone_w[3], self.backbone_w[4]
        self.neck_chs = (c3, c4, c5)

        # 2. Cổ tổng hợp đặc trưng đa tỷ lệ (Feature Aggregation Neck)
        self.neck = PAFPN(chs=self.neck_chs, n=self.neck_n)

        # 3. Bộ phân loại không gian - thời gian (Spatio-Temporal Classifier)
        self.conv_gru = ConvGRUClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            spatial_in_channels=self.neck_chs,
            target_size=self.target_size,
            num_classes=self.num_classes,
            supervision_mode=self.supervision_mode,
        )

    def train(self, mode: bool = True):
        """Chuyển đổi trạng thái train/eval kèm hỗ trợ khóa BatchNorm tự động nếu freeze_bn=True."""
        super().train(mode)
        if mode and self.freeze_bn:
            self._freeze_backbone_neck_bn()
        return self

    def _freeze_backbone_neck_bn(self) -> None:
        """Đóng băng thống kê Batch Normalization trong Backbone và Neck để tránh ô nhiễm bởi video frames."""
        for m in self.backbone.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()
        for m in self.neck.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval()

    def _forward_backbone_neck(
            self,
            flat_frames: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lan truyền qua Backbone và PAFPN có tích hợp Gradient Checkpointing khi cần tiết kiệm VRAM."""
        if self.gradient_checkpointing and self.training and flat_frames.requires_grad:
            def _stage(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                o3, o4, o5 = self.backbone(f)
                return self.neck(o3, o4, o5)

            return torch.utils.checkpoint.checkpoint(_stage, flat_frames, use_reentrant=False)
        else:
            out3, out4, out5 = self.backbone(flat_frames)
            return self.neck(out3, out4, out5)

    def forward(
            self,
            x: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None,
            hidden_state: Optional[List[torch.Tensor]] = None,
            return_sequence: bool = False,
            return_weights: bool = False,
            return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        """
        Lan truyền xuôi toàn bộ mô hình End-to-End.

        Args:
            x (torch.Tensor): Tensor chuỗi video 5D dạng [Batch, Time, Channels, Height, Width].
                             Chấp nhận kiểu dữ liệu uint8 [0..255] hoặc float32 [0.0..1.0].
            seq_lens (Optional[torch.Tensor]): Độ dài thực tế của từng clip trong batch [Batch].
            hidden_state (Optional[List[torch.Tensor]]): Hidden state ngoại vi cho chế độ streaming.
            return_sequence (bool): Nếu True, trả về logits từng frame [B, T, num_classes].
            return_weights (bool): Nếu True, trả về dict trọng số chú ý không gian và thời gian.
            return_state (bool): Nếu True, trả về hidden state cuối cùng của ConvGRU.

        Returns:
            logits (torch.Tensor) hoặc Tuple chứa (logits, [weights], [states]).
        """
        if x.dim() != 5:
            raise ValueError(
                f"[DrowsyClassifier] Yêu cầu đầu vào Tensor 5D [B, T, C, H, W], nhưng nhận {x.dim()}D với shape {tuple(x.shape)}"
            )

        B, T, C, H, W = x.shape

        # 1. Tự động chuyển đổi uint8 sang float32 [0.0, 1.0] an toàn, không blocking GPU sync
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)

        # 2. Định hình lại không gian an toàn cho cả non-contiguous tensors
        flat_frames = x.reshape(B * T, C, H, W)

        # 3. Trích xuất đặc trưng qua Backbone & PAFPN End-to-End
        out3, out4, out5 = self._forward_backbone_neck(flat_frames)

        # 4. Định hình lại không gian - thời gian cho 3 tầng đặc trưng đa tỷ lệ
        p3 = out3.reshape(B, T, *out3.shape[1:])
        p4 = out4.reshape(B, T, *out4.shape[1:])
        p5 = out5.reshape(B, T, *out5.shape[1:])

        # 5. Phân loại qua SpatioTemporal ConvGRUClassifier
        outputs = self.conv_gru(
            (p3, p4, p5),
            seq_lens=seq_lens,
            hidden_state=hidden_state,
            return_sequence=return_sequence,
            return_weights=return_weights,
            return_state=return_state,
        )

        return outputs


# ==============================================================================
# BỘ KIỂM THỬ TỰ LẬP NGHIỆM THU TOÀN DIỆN (SELF-VERIFICATION SUITE)
# ==============================================================================
if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Khởi tạo mô hình
    model = DrowsyClassifier(
        num_classes=2,
        target_size=(14, 14),
        gradient_checkpointing=True,
        freeze_bn=True,
    ).to(device)
    model.train()
