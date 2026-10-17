import logging
import math
import torch
import torch.nn as nn

from ai.ObjectDetection_2p6M.src.blocks import Conv, DWConv, DFL
from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector
from ai.ObjectDetection_2p6M.transfer.config_lmk import (
    PIPELINE_IMAGE_SIZE,
    FaceLmkConfig,
    TrainConfig,
)
from ai.ObjectDetection_2p6M.utils.init_weights import initialize_detection_head

_BN_TYPES = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
logger = logging.getLogger('train_face_lmk')

class ScaleHeadFaceLmk(nn.Module):
    def __init__(self, c_in: int, cfg: FaceLmkConfig):
        super().__init__()
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.num_landmarks = cfg.require_num_landmarks()
        self.cfg = cfg
        c_cls = max(c_in // cfg.cls_channel_divisor, cfg.head_min_channels)
        c_reg = max(c_in // cfg.reg_channel_divisor, cfg.head_min_channels)
        c_lmk = max(c_in // cfg.lmk_channel_divisor, cfg.head_min_channels)
        c_lmk_hidden = max(
            c_lmk,
            min(self.num_landmarks * 2, cfg.lmk_hidden_max_channels),
        )

        def cls_stem():
            return nn.Sequential(DWConv(c_in, c_in, 3, 1), Conv(c_in, c_cls, 1, 1),
                                  DWConv(c_cls, c_cls, 3, 1), Conv(c_cls, c_cls, 1, 1))

        def reg_stem():
            return nn.Sequential(Conv(c_in, c_reg, 3, 1), Conv(c_reg, c_reg, 3, 1))

        def lmk_stem():
            return nn.Sequential(Conv(c_in, c_lmk, 3, 1), Conv(c_lmk, c_lmk, 3, 1), Conv(c_lmk, c_lmk_hidden, 1, 1))

        self.cls_stem_o2m, self.reg_stem_o2m, self.lmk_stem_o2m = cls_stem(), reg_stem(), lmk_stem()
        self.cls_o2m = nn.Conv2d(c_cls, cfg.nc, 1)
        self.reg_o2m = nn.Conv2d(c_reg, 4 * cfg.reg_max, 1)
        self.lmk_o2m = nn.Conv2d(c_lmk_hidden, self.num_landmarks * 2, 1)

        self.cls_stem_o2o, self.reg_stem_o2o, self.lmk_stem_o2o = cls_stem(), reg_stem(), lmk_stem()
        self.cls_o2o = nn.Conv2d(c_cls, cfg.nc, 1)
        self.reg_o2o = nn.Conv2d(c_reg, 4 * cfg.reg_max, 1)
        self.lmk_o2o = nn.Conv2d(c_lmk_hidden, self.num_landmarks * 2, 1)
        self._init_bias()

    def _init_bias(self):
        probability = self.cfg.cls_prior_probability
        prior = -math.log((1 - probability) / probability)
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, prior)
        for m in (self.reg_o2m, self.reg_o2o):
            nn.init.constant_(m.bias, self.cfg.reg_bias)
        for m in (self.lmk_o2m, self.lmk_o2o):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.lmk_weight_std)
            nn.init.constant_(m.bias, 0.0)

    def init_bias(self, stride, img=PIPELINE_IMAGE_SIZE):
        self._init_bias()
        value = math.log(
            self.cfg.stride_bias_expected_objects
            / self.nc
            / (img / stride) ** 2
        )
        for m in (self.cls_o2m, self.cls_o2o):
            nn.init.constant_(m.bias, value)

    def forward(self, x, return_o2m: bool = True):
        cf_o2o, rf_o2o, lf_o2o = self.cls_stem_o2o(x), self.reg_stem_o2o(x), self.lmk_stem_o2o(x)
        out_o2o = (self.cls_o2o(cf_o2o), self.reg_o2o(rf_o2o), self.lmk_o2o(lf_o2o))
        if not self.training and not return_o2m:
            return None, out_o2o
        cf_o2m, rf_o2m, lf_o2m = self.cls_stem_o2m(x), self.reg_stem_o2m(x), self.lmk_stem_o2m(x)
        out_o2m = (self.cls_o2m(cf_o2m), self.reg_o2m(rf_o2m), self.lmk_o2m(lf_o2m))
        return out_o2m, out_o2o

class DetectHeadFaceLmk(nn.Module):
    def __init__(
        self,
        chs=None,
        cfg: FaceLmkConfig = None,
        image_size: int = PIPELINE_IMAGE_SIZE,
    ):
        super().__init__()
        if cfg is None:
            raise ValueError('DetectHeadFaceLmk cần cfg: FaceLmkConfig dùng chung với FaceLandmarkDetectionLoss.')
        if chs is None:
            raise ValueError('DetectHeadFaceLmk cần danh sách channel P3/P4/P5 từ trunk.')
        self.cfg = cfg
        self.nc = cfg.nc
        self.reg_max = cfg.reg_max
        self.strides = cfg.strides
        self.num_landmarks = cfg.require_num_landmarks()
        self.heads = nn.ModuleList(ScaleHeadFaceLmk(c, cfg) for c in chs)
        self.dfl = DFL(cfg.reg_max)
        for head, stride in zip(self.heads, self.strides):
            head.init_bias(stride, image_size)

    def make_anchors(self, feats, strides, offset=None):
        if offset is None:
            offset = self.cfg.anchor_offset
        anchor_points, stride_tensor = [], []
        for (h, w), s in zip([f.shape[-2:] for f in feats], strides):
            sy = torch.arange(h, device=feats[0].device) + offset
            sx = torch.arange(w, device=feats[0].device) + offset
            gy, gx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((gx, gy), -1).view(-1, 2))
            stride_tensor.append(torch.full((h * w, 1), s, device=feats[0].device, dtype=torch.float))
        return torch.cat(anchor_points), torch.cat(stride_tensor)

    def decode_box(self, reg, anchors, stride):
        ltrb = self.dfl(reg)
        lt, rb = ltrb[:, :2], ltrb[:, 2:]
        anchors_t = anchors.transpose(0, 1).unsqueeze(0)
        xyxy = torch.cat([anchors_t - lt, anchors_t + rb], 1) * stride.transpose(0, 1).unsqueeze(0)
        return xyxy.transpose(1, 2)

    def decode_landmarks(self, lmk_raw, anchors, strides):
        """Decode signed landmark offsets relative to anchor centers.
        Args:
            lmk_raw: Tensor[B, 2*K, A], signed offsets in feature-grid units.
            anchors: Tensor[A, 2], anchor centers in feature-grid units.
            strides: Tensor[A, 1], pixel stride for each anchor.
        Returns:
            Tensor[B, A, K, 2] containing landmark coordinates in pixels.

        Landmark geometry is independent of the predicted bounding box. The
        box branch is still used by the assigner, but it no longer translates
        or rescales the final landmark coordinates.
        """
        if lmk_raw.ndim != 3:
            raise ValueError(f'lmk_raw phải có shape [B, 2K, A], nhận {tuple(lmk_raw.shape)}.')

        batch_size, channels, num_anchors = lmk_raw.shape
        expected_channels = self.num_landmarks * 2
        if channels != expected_channels:
            raise ValueError(
                f'lmk_raw có {channels} channel, cần {expected_channels} '
                f'cho {self.num_landmarks} landmark.'
            )
        if anchors.shape != (num_anchors, 2):
            raise ValueError(
                f'anchors phải có shape {(num_anchors, 2)}, nhận {tuple(anchors.shape)}.'
            )
        if strides.shape != (num_anchors, 1):
            raise ValueError(
                f'strides phải có shape {(num_anchors, 1)}, nhận {tuple(strides.shape)}.'
            )

        offsets_grid = (
            lmk_raw.transpose(1, 2)
            .contiguous()
            .view(batch_size, num_anchors, self.num_landmarks, 2)
        )
        anchor_grid = anchors.to(dtype=offsets_grid.dtype).view(1, num_anchors, 1, 2)
        stride_pixel = strides.to(dtype=offsets_grid.dtype).view(1, num_anchors, 1, 1)
        return (anchor_grid + offsets_grid) * stride_pixel

    def forward(self, feats, return_o2m: bool = True):
        o2m_cls, o2m_reg, o2m_lmk = [], [], []
        o2o_cls, o2o_reg, o2o_lmk = [], [], []
        for feat, head in zip(feats, self.heads):
            out_o2m, (c_o, r_o, l_o) = head(feat, return_o2m=return_o2m)
            if out_o2m is not None:
                c_m, r_m, l_m = out_o2m
                o2m_cls.append(c_m.flatten(2))
                o2m_reg.append(r_m.flatten(2))
                o2m_lmk.append(l_m.flatten(2))
            o2o_cls.append(c_o.flatten(2))
            o2o_reg.append(r_o.flatten(2))
            o2o_lmk.append(l_o.flatten(2))

        anchors, stride_t = self.make_anchors(feats, self.strides)
        o2o_cls_c = torch.cat(o2o_cls, 2).transpose(1, 2)
        o2o_reg_c = torch.cat(o2o_reg, 2)
        o2o_lmk_raw = torch.cat(o2o_lmk, 2)
        o2o_box = self.decode_box(o2o_reg_c, anchors, stride_t)

        if not self.training and not return_o2m:
            o2o_lmk_dec = self.decode_landmarks(o2o_lmk_raw, anchors, stride_t)
            return {'o2o': {'cls': o2o_cls_c, 'box': o2o_box, 'reg_raw': o2o_reg_c, 'lmk': o2o_lmk_dec, 'lmk_raw': o2o_lmk_raw},
                    'anchors': anchors, 'strides': stride_t}

        o2m_cls_c = torch.cat(o2m_cls, 2).transpose(1, 2)
        o2m_reg_c = torch.cat(o2m_reg, 2)
        o2m_lmk_raw = torch.cat(o2m_lmk, 2)
        o2m_box = self.decode_box(o2m_reg_c, anchors, stride_t)
        return {'o2m': {'cls': o2m_cls_c, 'box': o2m_box, 'reg_raw': o2m_reg_c, 'lmk_raw': o2m_lmk_raw},
                'o2o': {'cls': o2o_cls_c, 'box': o2o_box, 'reg_raw': o2o_reg_c, 'lmk_raw': o2o_lmk_raw},
                'anchors': anchors, 'strides': stride_t}

class FaceLmkDetector(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self._trunk_frozen = False
        scaffold = NMSFreeDetector(nc=cfg.face.nc, reg_max=cfg.face.reg_max, backbone_w=cfg.trunk_backbone_w,
                                    backbone_n=cfg.trunk_backbone_n, neck_n=cfg.trunk_neck_n, strides=cfg.face.strides)
        self.backbone = scaffold.backbone
        self.neck = scaffold.neck
        neck_channels = tuple(scaffold.neck_chs)
        if neck_channels != cfg.trunk_feat_channels:
            raise ValueError(
                f'Output channel của trunk {neck_channels} khác cấu hình suy ra {cfg.trunk_feat_channels}.'
            )
        if hasattr(scaffold, 'head'):
            del scaffold.head
        del scaffold
        self.head = DetectHeadFaceLmk(chs=neck_channels, cfg=cfg.face, image_size=cfg.image_size)
        initialize_detection_head(self.head, image_size=cfg.image_size)

    def load_trunk(self, path: str, map_location='cpu'):
        from utils.artifacts import validate_metadata

        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        metadata = validate_metadata(path, checkpoint)
        source = metadata['architecture']
        expected = dict(backbone_w=list(self.cfg.trunk_backbone_w),
                        backbone_n=list(self.cfg.trunk_backbone_n),
                        neck_n=self.cfg.trunk_neck_n, strides=list(self.cfg.face.strides))
        if any(source[key] != value for key, value in expected.items()):
            raise ValueError('Kiến trúc trunk không khớp checkpoint object detection.')
        state = checkpoint.get('ema') or checkpoint['model']
        for name in ('backbone', 'neck'):
            prefix = name + '.'
            weights = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
            getattr(self, name).load_state_dict(weights, strict=True)
        logger.info('Đã nạp backbone + neck từ %s', path)

    def freeze_trunk(self, freeze: bool = True):
        self._trunk_frozen = freeze
        for p in list(self.backbone.parameters()) + list(self.neck.parameters()):
            p.requires_grad = not freeze
        for m in list(self.backbone.modules()) + list(self.neck.modules()):
            if isinstance(m, _BN_TYPES):
                m.eval() if freeze else m.train()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self._trunk_frozen:
            for m in list(self.backbone.modules()) + list(self.neck.modules()):
                if isinstance(m, _BN_TYPES):
                    m.eval()
        return self

    def forward(self, images: torch.Tensor, return_o2m: bool = True):
        feats = self.neck(*self.backbone(images))
        return self.head(feats, return_o2m=return_o2m)

