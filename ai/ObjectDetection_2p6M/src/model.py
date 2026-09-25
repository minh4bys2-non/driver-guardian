import torch
import torch.nn as nn
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN
from ai.ObjectDetection_2p6M.src.head import DetectHead
from ai.ObjectDetection_2p6M.src.config import TrainConfig
from ai.ObjectDetection_2p6M.utils.init_weights import initialize_weights, initialize_detection_head
import json

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
if __name__ == "__main__":
    m = NMSFreeDetector().to("cuda").eval()

    n_params = sum(p.numel() for p in m.parameters())
    print(f"Total parameters: {n_params:,} ({n_params/1e6:.2f}M)")
    def count_params(module):
        return sum(p.numel() for p in module.parameters()) / 1e6

    print(f"Backbone : {count_params(m.backbone):.3f} M")
    print(f"Neck     : {count_params(m.neck):.3f} M")
    print(f"Head     : {count_params(m.head):.3f} M")
    print(f"Total    : {count_params(m):.3f} M")
    import time
    x = torch.randn(1, 3, 640, 640).to("cuda")
    # Benchmark tốc độ inference
    with torch.inference_mode():
        start = time.time()
        for _ in range(100):
            out = m(x)
        end = time.time()
    print("Inference time:", end - start)
    # Kiểm tra kích thước đầu ra
    print("o2o cls:", out["o2o"]["cls"].shape)
    print("o2o box:", out["o2o"]["box"].shape)
    print("anchors:", out["anchors"].shape)
