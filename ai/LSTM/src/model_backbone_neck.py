import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN
from ai.ObjectDetection_2p6M.utils.init_weights import initialize_weights

class BackboneNeck(nn.Module):
    def __init__(
        self,
        backbone: Optional[nn.Module] = None,
        neck: Optional[nn.Module] = None,
        backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256),
        backbone_n: Tuple[int, ...] = (1, 2, 2, 1),
        neck_n: int = 1,
    ):
        super().__init__()
        self.backbone_w = tuple(backbone_w)
        self.backbone_n = tuple(backbone_n)
        self.neck_n = int(neck_n)
        c3, c4, c5 = self.backbone_w[2], self.backbone_w[3], self.backbone_w[4]
        self.neck_chs = (c3, c4, c5)

        self.backbone = Backbone(w=self.backbone_w, n=self.backbone_n)
        self.neck = PAFPN(chs=self.neck_chs, n=self.neck_n)
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.neck(*self.backbone(x))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path, Dict[str, Any]],
        map_location: Optional[Union[str, torch.device]] = "cpu",
        eval_mode: bool = True,
        strict: bool = True,
        use_ema: bool = True,
    ) -> "BackboneNeck":
        checkpoint = (
            torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if isinstance(checkpoint_path, (str, Path)) else checkpoint_path
        )
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint phải là dictionary hoặc đường dẫn file .pt.")
        metadata = checkpoint.get("metadata") or {}
        sources = (metadata.get("architecture") or {}, checkpoint.get("cfg") or {}, checkpoint)
        architecture = {}
        for name in ("backbone_w", "backbone_n", "neck_n"):
            value = next((source[key] for source in sources for key in (name, f"trunk_{name}")
                          if source.get(key) is not None), None)
            if value is None:
                raise ValueError(f"Checkpoint thiếu cấu hình {name}.")
            architecture[name] = value
        keys = ("ema", "model", "model_state_dict", "state_dict") if use_ema else (
            "model", "model_state_dict", "state_dict", "ema"
        )
        weight_source = next((key for key in keys
                              if isinstance(checkpoint.get(key), dict) and checkpoint[key]), None)
        state = checkpoint[weight_source] if weight_source else checkpoint
        weights = {key: value for key, value in state.items() if key.startswith(("backbone.", "neck."))}
        if not weights:
            raise ValueError("Checkpoint không chứa trọng số backbone/neck.")
        model = cls(**architecture)
        model.load_state_dict(weights, strict=strict)
        if map_location is not None:
            model.to(map_location)
        model.train(not eval_mode)
        source = checkpoint_path if isinstance(checkpoint_path, (str, Path)) else "checkpoint dict"
        logging.getLogger("ai.LSTM").info(f"[BackboneNeck] Đã nạp: {source}")
        logging.getLogger("ai.LSTM").info(f"  task={metadata.get('task', 'backbone_neck')} | weights={weight_source or 'state_dict'} | "
              f"backbone_w={model.backbone_w} | backbone_n={model.backbone_n} | "
              f"neck_n={model.neck_n} | device={next(model.parameters()).device}")
        return model

if __name__ == "__main__":
    import argparse

    checkpoint_path = Path("/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/ObjectDetection_2p6M/checkpoints/231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11/da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9/landmark_finetune2/best.pt")
    model = BackboneNeck.from_checkpoint(checkpoint_path)
    with torch.inference_mode():
        outputs = model(torch.randn(1, 3, 224, 224))
    for name, output in zip(("P3", "P4", "P5"), outputs):
        logging.getLogger("ai.LSTM").info(f"{name}: {tuple(output.shape)}")
