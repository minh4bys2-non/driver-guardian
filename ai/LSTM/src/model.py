from pathlib import Path

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .model_backbone_neck import BackboneNeck
from .models_convgru import ConvGRUClassifier

class DrowsyClassifier(nn.Module):
    def __init__(
        self,
        backbone_checkpoint: str | Path | None = None,
        num_classes: int = 2,
        input_dim: int = 128,
        hidden_dim: int = 160,
        backbone_w: tuple[int, ...] = (16, 32, 64, 128, 256),
        backbone_n: tuple[int, ...] = (1, 2, 2, 1),
        neck_n: int = 1,
        chunk_size: int = 32,
        dropout: float = 0.35,
        supervision_mode: str = "attention_pooling",
        freeze_bn: bool | None = None,
        freeze_backbone: bool = False,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        if chunk_size < 0:
            raise ValueError("chunk_size phải >= 0 (0: xử lý toàn bộ frame).")
        self.chunk_size = chunk_size
        self.freeze_backbone = freeze_backbone
        self.freeze_bn = (backbone_checkpoint is not None) if freeze_bn is None else freeze_bn
        self.gradient_checkpointing = gradient_checkpointing
        self.backbone_neck = (
            BackboneNeck.from_checkpoint(backbone_checkpoint) if backbone_checkpoint is not None else
            BackboneNeck(backbone_w=backbone_w, backbone_n=backbone_n, neck_n=neck_n)
        )
        self.backbone_neck.requires_grad_(not freeze_backbone)
        self.model_config = dict(
            backbone_w=self.backbone_neck.backbone_w, backbone_n=self.backbone_neck.backbone_n,
            neck_n=self.backbone_neck.neck_n, input_dim=input_dim, hidden_dim=hidden_dim,
            num_classes=num_classes, chunk_size=chunk_size, dropout=dropout,
            supervision_mode=supervision_mode, freeze_bn=self.freeze_bn,
            freeze_backbone=freeze_backbone, gradient_checkpointing=gradient_checkpointing,
        )
        self.conv_gru = ConvGRUClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            spatial_in_channels=self.backbone_neck.neck_chs,
            dropout=dropout,
            supervision_mode=supervision_mode,
        )
        self.train()

    def train(self, mode: bool = True) -> "DrowsyClassifier":
        super().train(mode)
        if self.freeze_backbone:
            self.backbone_neck.eval()
        if mode and self.freeze_bn:
            for layer in self.backbone_neck.modules():
                if isinstance(layer, nn.BatchNorm2d):
                    layer.eval()
        return self

    def _extract_p4(self, frames: torch.Tensor) -> torch.Tensor:
        return self.backbone_neck(frames)[1]

    def forward(
        self,
        x: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
        hidden_state=None,
        return_sequence: bool = False,
        return_weights: bool = False,
        return_state: bool = False,
    ) -> torch.Tensor | tuple:
        if x.ndim != 5 or x.shape[2] != 3 or any(size == 0 for size in x.shape):
            raise ValueError("Đầu vào phải có dạng [B,T,3,H,W] và không rỗng.")
        b, t, c, h, w = x.shape
        lengths = torch.full((b,), t, device=x.device) if seq_lens is None else torch.as_tensor(seq_lens, device=x.device)
        if lengths.shape != (b,) or not torch.all((lengths >= 1) & (lengths <= t) & (lengths == lengths.long())):
            raise ValueError("seq_lens phải gồm B số nguyên trong [1,T].")
        valid = (torch.arange(t, device=x.device)[None] < lengths[:, None]).flatten()
        frames = x.reshape(b * t, c, h, w)[valid]
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255.0

        chunks = []
        for chunk in frames.split(self.chunk_size or len(frames)):
            if self.freeze_backbone:
                with torch.no_grad():
                    p4 = self._extract_p4(chunk)
            elif self.gradient_checkpointing and self.training and self.freeze_bn:
                p4 = checkpoint(self._extract_p4, chunk, use_reentrant=False)
            else:
                p4 = self._extract_p4(chunk)
            chunks.append(p4)
        p4 = torch.cat(chunks)
        # Exclude padded frames from backbone BatchNorm and restore the time axis afterward.
        features = p4.new_zeros((b * t, *p4.shape[1:]))
        features[valid] = p4
        return self.conv_gru(
            features.reshape(b, t, *p4.shape[1:]),
            seq_lens=lengths,
            hidden_state=hidden_state,
            return_sequence=return_sequence,
            return_weights=return_weights,
            return_state=return_state,
        )

if __name__ == "__main__":
    BACKBONE_CHECKPOINT = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11/da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9/landmark_finetune2/best.pt"
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    model = DrowsyClassifier(
        backbone_checkpoint=BACKBONE_CHECKPOINT
    ).to(DEVICE).eval()
    print("Video -> Backbone -> Neck (P4) -> ConvGRU -> Logits")
    modules = {
        "Backbone": model.backbone_neck.backbone,
        "Neck": model.backbone_neck.neck,
        "ConvGRU classifier": model.conv_gru,
        "Total": model,
    }
    for name, module in modules.items():
        params = sum(p.numel() for p in module.parameters())
        print(f"{name:20s}: {params:,} parameters")
    frames = torch.randint(
        0, 256, (2, 4, 3, 224, 224),
        dtype=torch.uint8, device=DEVICE
    )
    with torch.inference_mode():
        logits, weights, states = model(
            frames,
            seq_lens=torch.tensor([4, 2], device=DEVICE),
            return_weights=True,
            return_state=True,
        )
    outputs = {
        "Video": frames,
        "Logits": logits,
        "Spatial attention": weights["spatial"],
        "Temporal weights": weights["temporal"],
        "Hidden state": states[0],
    }
    for name, tensor in outputs.items():
        print(f"{name:20s}: {tuple(tensor.shape)}")
