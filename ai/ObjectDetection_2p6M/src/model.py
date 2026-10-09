import os
import json
from pathlib import Path
from typing import Union, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn

from utils.artifacts import validate_metadata

try:
    from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN
    from ai.ObjectDetection_2p6M.src.head import DetectHead
    from ai.ObjectDetection_2p6M.src.config import TrainConfig
    from ai.ObjectDetection_2p6M.utils.init_weights import initialize_weights, initialize_detection_head
except ModuleNotFoundError:
    import sys

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "ai").is_dir():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            break
    from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN
    from ai.ObjectDetection_2p6M.src.head import DetectHead
    from ai.ObjectDetection_2p6M.src.config import TrainConfig
    from ai.ObjectDetection_2p6M.utils.init_weights import initialize_weights, initialize_detection_head


class NMSFreeDetector(nn.Module):
    def __init__(self, nc=TrainConfig().nc, reg_max=TrainConfig().reg_max,
                 backbone_w=TrainConfig().backbone_w,
                 backbone_n=TrainConfig().backbone_n,
                 neck_n=TrainConfig().neck_n,
                 strides=TrainConfig().strides,
                 img_size=TrainConfig().img_size):
        super().__init__()

        # Setting parameters
        self.nc = nc
        self.reg_max = reg_max
        self.backbone_w = backbone_w
        self.backbone_n = backbone_n
        self.neck_n = neck_n
        self.strides = strides
        self.img_size = img_size

        self.backbone = Backbone(w=backbone_w, n=backbone_n)
        c3, c4, c5 = backbone_w[2], backbone_w[3], backbone_w[4]
        self.neck_chs = (c3, c4, c5)
        self.neck = PAFPN(chs=(c3, c4, c5), n=neck_n)
        self.head = DetectHead(chs=(c3, c4, c5), nc=nc, reg_max=reg_max,
                               strides=strides, img_size=img_size)

        self._initialize_weights()

    def forward(self, x, o2o_only=False):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.neck(p3, p4, p5)
        results = self.head([p3, p4, p5], o2o_only=o2o_only)
        return results

    def replace_head(self, nc=None, reg_max=None, strides=None, img_size=None, replace_all=True):
        nc = self.nc if nc is None else nc
        reg_max = self.reg_max if reg_max is None else reg_max
        strides = self.strides if strides is None else strides
        img_size = self.img_size if img_size is None else img_size
        if not replace_all:
            if reg_max != self.reg_max or tuple(strides) != tuple(self.strides) or img_size != self.img_size:
                raise ValueError("replace_all=False only supports changing nc")
            self.head.replace_head(nc)
            self.nc = nc
            return self

        ref_param = next(self.backbone.parameters())

        self.head = DetectHead(
            chs=self.neck_chs,
            nc=nc,
            reg_max=reg_max,
            strides=strides,
            img_size=img_size
        ).to(
            device=ref_param.device,
            dtype=ref_param.dtype,
        )
        # Khởi tạo trọng số cho head mới (dùng đúng img_size)
        initialize_detection_head(self.head, img_size)

        self.nc = nc
        self.reg_max = reg_max
        self.strides = strides
        self.img_size = img_size

        return self

    def _initialize_weights(self):
        initialize_weights(self, self.img_size)
        return self

    def freeze_trunk(self, freeze=True):
        for p in self.backbone.parameters():
            p.requires_grad_(not freeze)
        for p in self.neck.parameters():
            p.requires_grad_(not freeze)
        return self

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)["model"]

        return cls(
            nc=cfg.get("num_classes", cfg.get("nc")),
            reg_max=cfg["reg_max"],
            backbone_w=tuple(cfg["backbone_w"]),
            backbone_n=tuple(cfg["backbone_n"]),
            neck_n=cfg["neck_n"],
            strides=tuple(cfg["strides"]),
            img_size=cfg.get("img_size", 640),
        )

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: Union[str, Path, Dict[str, Any]],
            map_location: Optional[Union[str, torch.device]] = "cpu",
            eval_mode: bool = False,
            # img_size: int = 640
    ) -> "NMSFreeDetector":
        """
        """
        checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        metadata = None
        img_size = checkpoint.get("cfg", {}).get("img_size", 290)
        try:
            metadata = validate_metadata(checkpoint_path, checkpoint)
        except Exception:
            metadata = checkpoint.get("metadata")

        if metadata and "architecture" in metadata:
            arch = metadata["architecture"]
        else:
            from ai.ObjectDetection_2p6M.src.config import TrainConfig
            cfg = TrainConfig()
            arch = {
                "nc": cfg.nc,
                "reg_max": cfg.reg_max,
                "backbone_w": cfg.backbone_w,
                "backbone_n": cfg.backbone_n,
                "neck_n": cfg.neck_n,
                "strides": cfg.strides,
            }
        valid_keys = {"nc", "reg_max", "backbone_w", "backbone_n", "neck_n", "strides"}
        arch = {k: v for k, v in arch.items() if k in valid_keys}

        model = cls(**arch, img_size=img_size)
        state_dict = checkpoint.get("ema") or checkpoint.get("model") or checkpoint
        model.load_state_dict(state_dict, strict=False)

        if eval_mode:
            model.eval()

        return model


if __name__ == "__main__":
    model = NMSFreeDetector.from_checkpoint(
        r"D:\Project\DATN\driver-guardian\ai\ObjectDetection_2p6M\checkpoints\231e35bb4061257f9bcb7cc5e3a0d0064e5e1beb6351a5e0adfc8d0c91ba4e11\da92589d1b67339be4491beaf147575e81a2699bbde06d654c05f852d7f0e9c9\landmark_train\best.pt",
        eval_mode=True
    )
    img = torch.rand((1, 3, 640, 640))
    preds = model.backbone(img)
    print(preds[1].shape)
