import math
import torch
import torch.nn as nn
from src.blocks import Conv, DWConv, DFL

class ScaleHead(nn.Module):
    def __init__(self, c, nc, reg_max=16):
        super().__init__()
        self.nc = nc
        cc, cr = max(c // 2, 64), max(c // 4, 64)

        cls = lambda: nn.Sequential(
            DWConv(c, c, 3), Conv(c, cc, 1),
            DWConv(cc, cc, 3), Conv(cc, cc, 1)
        )
        reg = lambda: nn.Sequential(
            Conv(c, cr, 3), Conv(cr, cr, 3)
        )

        self.cls_o2m, self.reg_o2m = cls(), reg()
        self.cls_o2o, self.reg_o2o = cls(), reg()
        self.out_cls_o2m, self.out_reg_o2m = nn.Conv2d(cc, nc, 1), nn.Conv2d(cr, 4 * reg_max, 1)
        self.out_cls_o2o, self.out_reg_o2o = nn.Conv2d(cc, nc, 1), nn.Conv2d(cr, 4 * reg_max, 1)

    def init_bias(self, stride, img=640):
        b = math.log(5 / self.nc / (img / stride) ** 2)
        for m in self.out_cls_o2m, self.out_cls_o2o: nn.init.constant_(m.bias, b)
        for m in self.out_reg_o2m, self.out_reg_o2o: nn.init.constant_(m.bias, 1.)

    @staticmethod
    def _forward(x, cls, reg, out_cls, out_reg):
        return out_cls(cls(x)), out_reg(reg(x))

    def forward(self, x, o2o_only=False):
        o2o = self._forward(
            x.detach(), self.cls_o2o, self.reg_o2o,
            self.out_cls_o2o, self.out_reg_o2o
        )
        if not self.training and o2o_only:
            return None, o2o

        o2m = self._forward(
            x, self.cls_o2m, self.reg_o2m,
            self.out_cls_o2m, self.out_reg_o2m
        )
        return o2m, o2o

class DetectHead(nn.Module):
    def __init__(self, chs=(128, 256, 512), nc=80, reg_max=16,
                 strides=(8, 16, 32), img_size=640):
        super().__init__()
        self.nc, self.reg_max, self.strides = nc, reg_max, strides
        self.img_size = img_size
        self.heads = nn.ModuleList(ScaleHead(c, nc, reg_max) for c in chs)
        self.dfl = DFL(reg_max)

        for h, s in zip(self.heads, strides):
            h.init_bias(s, img_size)

    def replace_head(self, nc):
        if isinstance(nc, bool) or not isinstance(nc, int) or nc < 1:
            raise ValueError("nc must be a positive integer")
        for head, stride in zip(self.heads, self.strides):
            for name in ("out_cls_o2m", "out_cls_o2o"):
                old = getattr(head, name)
                layer = nn.Conv2d(old.in_channels, nc, 1,
                                  device=old.weight.device, dtype=old.weight.dtype)
                nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                nn.init.constant_(layer.bias, math.log(5 / nc / (self.img_size / stride) ** 2))
                layer.train(old.training)
                setattr(head, name, layer)
            head.nc = nc
        self.nc = nc
        return self

    @staticmethod
    def make_anchors(feats, strides, offset=.5):
        anchors, stride_t = [], []
        device, dtype = feats[0].device, feats[0].dtype

        for f, s in zip(feats, strides):
            h, w = f.shape[-2:]
            y = torch.arange(h, device=device, dtype=dtype) + offset
            x = torch.arange(w, device=device, dtype=dtype) + offset
            gy, gx = torch.meshgrid(y, x, indexing="ij")
            anchors.append(torch.stack((gx, gy), -1).reshape(-1, 2))
            stride_t.append(torch.full((h * w, 1), s, device=device, dtype=dtype))

        return torch.cat(anchors), torch.cat(stride_t)

    def decode_box(self, reg, anchors, strides):
        lt, rb = self.dfl(reg).chunk(2, 1)
        anchors, strides = anchors.T[None], strides.T[None]
        return (torch.cat((anchors - lt, anchors + rb), 1) * strides).transpose(1, 2)

    @staticmethod
    def merge(cls, reg):
        return torch.cat(cls, 2).transpose(1, 2), torch.cat(reg, 2)

    def forward(self, feats, o2o_only=False):
        o2m_cls, o2m_reg, o2o_cls, o2o_reg = [], [], [], []

        for x, head in zip(feats, self.heads):
            o2m, o2o = head(x, o2o_only=o2o_only)

            c, r = o2o
            o2o_cls.append(c.flatten(2))
            o2o_reg.append(r.flatten(2))

            if o2m is not None:
                c, r = o2m
                o2m_cls.append(c.flatten(2))
                o2m_reg.append(r.flatten(2))

        anchors, strides = self.make_anchors(feats, self.strides)

        c, r = self.merge(o2o_cls, o2o_reg)
        out = {
            "o2o": {"cls": c, "box": self.decode_box(r, anchors, strides), "reg_raw": r},
            "anchors": anchors,
            "strides": strides,
        }
        if not self.training and o2o_only:
            return out
        c, r = self.merge(o2m_cls, o2m_reg)
        out["o2m"] = {
            "cls": c,
            "box": self.decode_box(r, anchors, strides),
            "reg_raw": r,
        }
        return out
