# =====================================================================
# File: model.py
# Dự án: Driver Guardian - Module LiteFat (IEEE/RSJ IROS 2025 Upgraded)
# Tác vụ: Định nghĩa kiến trúc LiteFat v2 giải quyết triệt để sụp đổ
#         đồ thị mốc mặt (Landmark Collapse) và hiện tượng học thuộc
#         lòng khuôn mặt (Shortcut Identity Overfitting) của MobileNetV3.
# =====================================================================
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class KinematicFeatureExtractor(nn.Module):
    """
    Khối tiền xử lý chuẩn hóa động học 68 điểm mốc khuôn mặt (LiteFat v5):
    1. Chuẩn hóa hình thái đa cơ quan độc lập (Organ-Centered Local Coordinates):
       - Mắt & Lông mày: Quy chiếu về tâm khóe mắt, bảo toàn 100% độ mở mắt tĩnh/động (EAR).
       - Miệng: Quy chiếu về tâm khóe miệng, bảo toàn 100% độ mở miệng tĩnh/động (MAR).
       - Hàm & Mũi: Khử cấu trúc tĩnh (pos_dyn + 0.25*pos_norm) để ghi nhận tư thế đầu mà không học thuộc lòng xương sọ.
    2. Ổn định IOD tránh nổ thương số khi quay đầu sang ngang (Head Yaw).
    3. Đạo hàm vận tốc d(nose)/iod và gia tốc đúng chuẩn giải tích, loại bỏ triệt để gai giả ở t=1.
    4. Cân bằng năng lượng động học từng clip (Instance Kinematic Normalization) trước BatchNorm2d(6)
       để đồng bộ biên độ chuyển động giữa các bộ dữ liệu khác biệt FPS và độ rung cabin (UL-DD, SUST, VBDDD).
    """

    def __init__(
            self,
            nose_idx: int = 30,
            left_eye_idx: int = 36,
            right_eye_idx: int = 45,
            eps: float = 1e-2,
            static_shape_ratio: float = 0.25,
    ):
        super().__init__()
        self.nose_idx = nose_idx
        self.left_eye_idx = left_eye_idx
        self.right_eye_idx = right_eye_idx
        self.eps = eps
        self.static_shape_ratio = float(static_shape_ratio)

        # Chuẩn hóa phân phối riêng cho 6 kênh [x, y, dx, dy, d2x, d2y]
        self.channel_norm = nn.BatchNorm2d(6, affine=True)

    @staticmethod
    def build_temporal_mask(
            x: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Tạo binary float mask kích thước (B, 1, T, 1) biểu diễn các frame hợp lệ (1.0) và frame đệm (0.0).
        """
        B = x.size(0)
        T = x.size(2) if x.dim() == 4 else x.size(1)
        device = x.device
        if lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            mask = (t_idx < lengths.to(device).unsqueeze(1)).float()  # (B, T)
            return mask.view(B, 1, T, 1)
        else:
            frame_activity = x[:, :2].abs().sum(dim=(1, 3), keepdim=True) if x.dim() == 4 and x.size(1) == 2 else x.abs().sum(dim=(-2, -1), keepdim=True)
            return (frame_activity > eps).float().view(B, 1, T, 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input : (B, 2, T, V=68) - Tọa độ [0, 1] trong hệ tọa độ khung ảnh.
        mask  : (B, 1, T, 1)    - Float mask (1.0 = valid, 0.0 = padded/missing).
        Output: (B, 6, T, V=68) - [x_norm, y_norm, dx_norm, dy_norm, d2x_norm, d2y_norm].
        """
        if x.size(1) != 2:
            return x * mask if mask is not None else x

        B, _, T, V = x.shape

        # 1. Khoảng cách giải phẫu 2 mắt (Inter-Ocular Distance - IOD giữa điểm 36 và 45)
        p_left_eye = x[:, :, :, self.left_eye_idx: self.left_eye_idx + 1]    # (B, 2, T, 1)
        p_right_eye = x[:, :, :, self.right_eye_idx: self.right_eye_idx + 1]  # (B, 2, T, 1)
        raw_iod = torch.norm(p_right_eye - p_left_eye, p=2, dim=1, keepdim=True).clamp(min=self.eps)  # (B, 1, T, 1)

        # Ổn định IOD theo từng clip: kết hợp IOD tức thời với trung bình IOD của các frame hợp lệ
        # Khi tài xế quay đầu sang ngang (yaw), khoảng cách 2D co lại nhưng không làm nổ mẫu số
        if mask is not None:
            v_mask = mask.clamp(min=0.0, max=1.0)
            clip_iod = (raw_iod * v_mask).sum(dim=2, keepdim=True) / v_mask.sum(dim=2, keepdim=True).clamp(min=1.0)
        else:
            clip_iod = raw_iod.mean(dim=2, keepdim=True)
        clip_iod = clip_iod.clamp(min=self.eps)
        iod_scale = torch.maximum(0.5 * raw_iod + 0.5 * clip_iod, 0.65 * clip_iod).clamp(min=self.eps)

        # 2. Tọa độ chuẩn hóa theo Tâm Mũi (pos_norm) dùng cho biến dạng động lực học
        nose_pos = x[:, :, :, self.nose_idx: self.nose_idx + 1]  # (B, 2, T, 1)
        pos_norm = (x - nose_pos) / iod_scale                    # (B, 2, T, V)
        pos_norm = torch.clamp(pos_norm, min=-8.0, max=8.0)

        # 3. Chuẩn hóa hình thái đa cơ quan độc lập (Organ-Centered Local Coordinates):
        # Mắt trái (36..41) & Lông mày trái (17..21): Gốc tại trung điểm khóe mắt trái (36 & 39)
        c_left_eye = 0.5 * (x[:, :, :, 36:37] + x[:, :, :, 39:40])
        # Mắt phải (42..47) & Lông mày phải (22..26): Gốc tại trung điểm khóe mắt phải (42 & 45)
        c_right_eye = 0.5 * (x[:, :, :, 42:43] + x[:, :, :, 45:46])
        # Miệng (48..67): Gốc tại trung điểm 2 khóe miệng (48 & 54)
        c_mouth = 0.5 * (x[:, :, :, 48:49] + x[:, :, :, 54:55])

        pos_feat = pos_norm.clone()
        # Vùng mắt & lông mày trái
        pos_feat[:, :, :, 17:22] = (x[:, :, :, 17:22] - c_left_eye) / iod_scale
        pos_feat[:, :, :, 36:42] = (x[:, :, :, 36:42] - c_left_eye) / iod_scale
        # Vùng mắt & lông mày phải
        pos_feat[:, :, :, 22:27] = (x[:, :, :, 22:27] - c_right_eye) / iod_scale
        pos_feat[:, :, :, 42:48] = (x[:, :, :, 42:48] - c_right_eye) / iod_scale
        # Vùng miệng
        pos_feat[:, :, :, 48:68] = (x[:, :, :, 48:68] - c_mouth) / iod_scale

        # Giảm thiểu shortcut hình thái tĩnh trên mắt, lông mày, miệng nếu static_shape_ratio < 1.0
        if self.static_shape_ratio < 0.999:
            organ_indices = list(range(17, 27)) + list(range(36, 68))
            pos_org = pos_feat[:, :, :, organ_indices]
            if mask is not None:
                denom_org = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
                org_mean = (pos_org * mask).sum(dim=2, keepdim=True) / denom_org
            else:
                org_mean = pos_org.mean(dim=2, keepdim=True)
            pos_feat[:, :, :, organ_indices] = (pos_org - org_mean) + self.static_shape_ratio * org_mean

        # Vùng Hàm (0..16) & Mũi (27..35): Trừ trung bình thời gian để bắt gật đầu/quay đầu mà không học thuộc lòng xương sọ
        jaw_nose_indices = list(range(0, 17)) + list(range(27, 36))
        pos_jn = pos_norm[:, :, :, jaw_nose_indices]
        if mask is not None:
            denom_jn = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
            jn_mean = (pos_jn * mask).sum(dim=2, keepdim=True) / denom_jn
        else:
            jn_mean = pos_jn.mean(dim=2, keepdim=True)
        pos_feat[:, :, :, jaw_nose_indices] = (pos_jn - jn_mean) + 0.25 * pos_jn
        pos_feat = torch.clamp(pos_feat, min=-8.0, max=8.0)

        # 4. Tính Vận tốc (dx, dy) và Gia tốc (d2x, d2y) chuẩn hóa giải tích
        if T <= 1:
            vel_norm = torch.zeros_like(pos_norm)
            acc_norm = torch.zeros_like(pos_norm)
        else:
            # Vận tốc biến dạng cục bộ giữa các khung hình liên tiếp
            vel_local = torch.diff(pos_norm, n=1, dim=2)  # (B, 2, T-1, V)

            # Vận tốc tịnh tiến toàn đầu tại đỉnh tâm mũi (node 30):
            # Tính d(nose)/iod thay vì d(nose/iod) để triệt tiêu hoàn toàn lỗi nổ vận tốc do dao động mẫu số
            delta_nose = torch.diff(nose_pos, n=1, dim=2)  # (B, 2, T-1, 1)
            iod_mid = 0.5 * (iod_scale[:, :, 1:, :] + iod_scale[:, :, :-1, :])
            nose_vel = delta_nose / iod_mid.clamp(min=self.eps)
            nose_vel = torch.clamp(nose_vel, min=-4.0, max=4.0)
            vel_local = vel_local.clone()
            vel_local[:, :, :, self.nose_idx: self.nose_idx + 1] = nose_vel

            if mask is not None:
                pair_valid = mask[:, :, 1:, :] * mask[:, :, :-1, :]
                vel_local = vel_local * pair_valid

            # Gia tốc: Đạo hàm bậc 2 trên chuỗi vận tốc trước khi chèn số 0 (loại bỏ gai giả ở t=1)
            if T >= 3:
                acc_raw = torch.diff(vel_local, n=1, dim=2)  # (B, 2, T-2, V)
                if mask is not None:
                    triplet_valid = mask[:, :, 2:, :] * mask[:, :, 1:-1, :] * mask[:, :, :-2, :]
                    acc_raw = acc_raw * triplet_valid
                acc_raw = torch.clamp(acc_raw, min=-4.0, max=4.0)
                acc_norm = F.pad(acc_raw, (0, 0, 2, 0), mode="constant", value=0.0)  # (B, 2, T, V)
            else:
                acc_norm = torch.zeros_like(pos_norm)

            vel_norm = F.pad(vel_local, (0, 0, 1, 0), mode="constant", value=0.0)  # (B, 2, T, V)
            vel_norm = torch.clamp(vel_norm, min=-4.0, max=4.0)

        # 5. Cân bằng năng lượng động học từng clip (Instance Kinematic Normalization)
        # Giúp đồng bộ thang đo giữa UL-DD (camera 60fps tĩnh) và SUST/VBDDD (camera cabin rung)
        if mask is not None:
            m_valid = mask.clamp(min=0.0, max=1.0)
            n_valid = m_valid.sum(dim=(2, 3), keepdim=True).clamp(min=1.0) * float(V)
            vel_rms = torch.sqrt((vel_norm.pow(2) * m_valid).sum(dim=(2, 3), keepdim=True) / n_valid + 1e-6)
            acc_rms = torch.sqrt((acc_norm.pow(2) * m_valid).sum(dim=(2, 3), keepdim=True) / n_valid + 1e-6)
        else:
            vel_rms = torch.sqrt(vel_norm.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-6)
            acc_rms = torch.sqrt(acc_norm.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-6)

        # Chuẩn hóa mềm đưa RMS về dải tự nhiên ~0.035 IOD/frame
        vel_scale = torch.clamp(vel_rms, min=0.012, max=0.10) / 0.035
        acc_scale = torch.clamp(acc_rms, min=0.010, max=0.08) / 0.025
        vel_norm = vel_norm / vel_scale
        acc_norm = acc_norm / acc_scale

        out = torch.cat([pos_feat, vel_norm, acc_norm], dim=1)  # (B, 6, T, V)
        out = torch.nan_to_num(out, nan=0.0, posinf=4.0, neginf=-4.0)

        # 6. Chuẩn hóa kênh BatchNorm2d(6) kết hợp che mặt nạ chặt chẽ
        if mask is not None:
            out = out * mask
        out = self.channel_norm(out)
        if mask is not None:
            out = out * mask

        return out


class MobileNetV3ContextEncoder(nn.Module):
    """
    Trích xuất vector ngữ cảnh thị giác d in R^{D} cho mỗi khung hình khuôn mặt
    bằng xương sống MobileNetV3-Small.
    """

    def __init__(
            self,
            out_dim: int = 64,
            proj_hidden_dim: int = 192,
            pretrained: bool = True,
            dropout: float = 0.2,
            freeze_backbone: bool = False,
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)

        self.features = base_model.features
        self.avgpool = base_model.avgpool

        self.projector = nn.Sequential(
            nn.Linear(576, proj_hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(proj_hidden_dim, out_dim),
        )

        if freeze_backbone:
            self.freeze_backbone(freeze=True)

    def freeze_backbone(self, freeze: bool = True, freeze_projector: bool = False) -> None:
        """Bật/tắt đóng băng tham số của xương sống MobileNetV3-Small."""
        for param in self.features.parameters():
            param.requires_grad = not freeze
        for m in self.features.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                m.eval() if freeze else m.train(self.training)
        if freeze_projector:
            for param in self.projector.parameters():
                param.requires_grad = not freeze
        elif not freeze:
            for param in self.projector.parameters():
                param.requires_grad = True

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None) -> None:
        """
        Mở khóa đóng băng MobileNetV3-Small:
        - unfreeze_last_n_blocks = None: mở khóa toàn bộ features.
        - unfreeze_last_n_blocks = K: chỉ mở khóa K block tích chập cuối của features.
        """
        if unfreeze_last_n_blocks is None:
            self.freeze_backbone(freeze=False, freeze_projector=False)
            return

        # Khóa toàn bộ trước
        self.freeze_backbone(freeze=True, freeze_projector=False)
        total_blocks = len(self.features)
        start_idx = max(0, total_blocks - unfreeze_last_n_blocks)
        for i in range(start_idx, total_blocks):
            for param in self.features[i].parameters():
                param.requires_grad = True

    @property
    def is_backbone_frozen(self) -> bool:
        """Kiểm tra xem tất cả các tham số của MobileNetV3 features có đang bị đóng băng hay không."""
        return not any(p.requires_grad for p in self.features.parameters())

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.is_backbone_frozen:
            for m in self.features.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                    m.eval()
        return self

    def forward(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        feat = self.features(frame_tensor)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        d = self.projector(feat)
        return d


class MultimodalFusionModule(nn.Module):
    """
    Dung hợp Đa phương thức giải quyết triệt để Rank-1 Bottleneck:
    1. Chiếu 6 kênh động học [x, y, dx, dy, d2x, d2y] lên không gian đặc trưng đa chiều D (Linear + LayerNorm + GELU).
    2. Cộng Node Positional Embedding [1, 1, 68, D] giúp phân biệt rõ vai trò giải phẫu của từng điểm mốc.
    3. Điều biến Gated FiLM modulation với vector ngữ cảnh thị giác d in R^{D}:
       H = H_kin * (1.0 + tanh(W_scale(d))) + W_shift(d)
    4. Tương thích hoàn hảo cả khi d=None (chế độ Fast Landmark-Only).
    """

    def __init__(
            self,
            num_features: int = 6,
            feature_dim: int = 64,
            num_nodes: int = 68,
            nose_idx: int = 30,
            left_eye_idx: int = 36,
            right_eye_idx: int = 45,
            eps: float = 1e-2,
            static_shape_ratio: float = 0.25,
    ):
        super().__init__()
        self.left_eye_idx = left_eye_idx
        self.right_eye_idx = right_eye_idx
        self.eps = eps
        self.feature_dim = feature_dim
        self.num_nodes = num_nodes

        # Khối tiền xử lý động học chuẩn hóa theo IOD
        self.kinematic = KinematicFeatureExtractor(
            nose_idx=nose_idx,
            left_eye_idx=left_eye_idx,
            right_eye_idx=right_eye_idx,
            eps=eps,
            static_shape_ratio=static_shape_ratio,
        )

        # Giữ lại w để tương thích backward compatibility với checkpoint cũ
        self.w = nn.Parameter(torch.ones(num_features, 1) / float(num_features))

        # Phép chiếu động học đa kênh 6D -> D
        self.kin_proj = nn.Sequential(
            nn.Linear(num_features, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        # Vector nhúng vị trí đỉnh giải phẫu khả học cho 68 mốc mặt
        self.node_embed = nn.Parameter(torch.randn(1, 1, num_nodes, feature_dim) * 0.02)

        # Tầng điều biến FiLM từ vector ngữ cảnh thị giác d
        self.film_scale = nn.Linear(feature_dim, feature_dim)
        self.film_shift = nn.Linear(feature_dim, feature_dim)

    def reset_film_to_identity(self) -> None:
        """Khởi tạo zero-init cho FiLM scale và shift để bảo toàn 100% đặc trưng đồ thị tại thời điểm bắt đầu Stage 2."""
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)

    def detect_missing_mask(self, coords: torch.Tensor) -> torch.Tensor:
        p_left = coords[:, :, self.left_eye_idx, :]
        p_right = coords[:, :, self.right_eye_idx, :]
        raw_iod = torch.norm(p_right - p_left, p=2, dim=-1, keepdim=True).unsqueeze(-1)

        all_ones = (coords >= 0.999).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
        all_zeros = (coords <= 0.001).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True)
        is_missing = (raw_iod < 1e-2) | all_ones | all_zeros
        return is_missing

    def forward(
            self,
            coords: torch.Tensor,
            d: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coords: Chuỗi tọa độ mốc [B, T, N, 2] (hoặc [B, T, N, 3] từ cache cũ).
            d: Vector ngữ cảnh thị giác [B, T, D] hoặc None (Landmark-Only).
            mask: Temporal float mask [B, 1, T, 1].
        Returns:
            X: Tensor đặc trưng đồ thị [B, D, N, T].
        """
        if coords.dim() == 4:
            if coords.size(-1) >= 2:
                coords_xy = coords[..., :2]
            elif coords.size(1) >= 2:
                coords_xy = coords[:, :2, :, :].permute(0, 2, 3, 1).contiguous()
            else:
                coords_xy = coords
        else:
            coords_xy = coords

        orig_dtype = coords.dtype
        dev = coords.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"

        with torch.autocast(device_type=dev_type, enabled=False):
            coords_32 = coords_xy.float()

            is_missing = self.detect_missing_mask(coords_32)
            valid_mask = (~is_missing).permute(0, 3, 1, 2).float()

            if mask is not None:
                temporal_mask = mask.float() * valid_mask
                is_missing = (temporal_mask == 0.0).permute(0, 2, 3, 1)
            else:
                temporal_mask = valid_mask

            x_in = coords_32.permute(0, 3, 1, 2).contiguous()  # [B, 2, T, N]
            feat_kin = self.kinematic(x_in, mask=temporal_mask)  # [B, 6, T, N]
            C_kin = feat_kin.permute(0, 2, 3, 1).contiguous()  # [B, T, N, 6]

            # Chiếu đa kênh 6D -> D và cộng nhúng vị trí giải phẫu
            H_kin = self.kin_proj(C_kin) + self.node_embed  # [B, T, N, D]

            # Điều biến FiLM nếu có bối cảnh thị giác d
            if d is not None:
                d_32 = d.float()
                scale = torch.tanh(self.film_scale(d_32)).unsqueeze(2)  # [B, T, 1, D]
                shift = self.film_shift(d_32).unsqueeze(2)  # [B, T, 1, D]
                H = H_kin * (1.0 + scale) + shift
            else:
                H = H_kin

            H = torch.nan_to_num(H, nan=0.0)
            # Giữ các khung hình đệm (padding) bằng 0 tuyệt đối trước khi vào STGL
            H = H * temporal_mask.permute(0, 2, 3, 1)

            # Đưa về chuẩn đồ thị STGL: [B, D, N, T]
            X = H.permute(0, 3, 2, 1).contiguous()

        return X.to(orig_dtype)


class SelfAdaptiveAdjacency(nn.Module):
    """
    Ma trận kề kết hợp Cấu trúc Giải phẫu Khuôn mặt (A_prior) và Tự thích nghi (A_adp):
        A_total = A_prior + alpha * softmax(ReLU(E1 @ E2^T / sqrt(d)))
    """

    def __init__(
            self,
            num_nodes: int = 68,
            embed_dim: int = 16,
            vertical_pairs: Optional[List[Tuple[int, int]]] = None,
            cross_region_pairs: Optional[List[Tuple[int, int]]] = None,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embed_dim = embed_dim

        # Xây dựng ma trận kề giải phẫu A_prior chuẩn hóa bậc đối xứng
        A_prior = self._build_anatomical_prior(num_nodes, vertical_pairs, cross_region_pairs)
        self.register_buffer("A_prior", A_prior)

        # Tham số học ma trận kề thích nghi
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.alpha = nn.Parameter(torch.tensor(0.5))

    @staticmethod
    def _build_anatomical_prior(
            num_nodes: int,
            vertical_pairs: Optional[List[Tuple[int, int]]],
            cross_region_pairs: Optional[List[Tuple[int, int]]],
    ) -> torch.Tensor:
        A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

        # 1. Đường viền hàm dưới: 0..16
        for i in range(16):
            A[i, i + 1] = A[i + 1, i] = 1.0

        # 2. Lông mày trái (17..21) & phải (22..26)
        for i in range(17, 21):
            A[i, i + 1] = A[i + 1, i] = 1.0
        for i in range(22, 26):
            A[i, i + 1] = A[i + 1, i] = 1.0

        # 3. Sống mũi (27..30) & Cánh mũi (31..35)
        for i in range(27, 30):
            A[i, i + 1] = A[i + 1, i] = 1.0
        for i in range(31, 35):
            A[i, i + 1] = A[i + 1, i] = 1.0
        A[30, 33] = A[33, 30] = 1.0

        # 4. Mắt trái (36..41) & Mắt phải (42..47)
        for i in range(36, 41):
            A[i, i + 1] = A[i + 1, i] = 1.0
        A[41, 36] = A[36, 41] = 1.0
        for i in range(42, 47):
            A[i, i + 1] = A[i + 1, i] = 1.0
        A[47, 42] = A[42, 47] = 1.0

        # 5. Môi ngoài (48..59) & Môi trong (60..67)
        for i in range(48, 59):
            A[i, i + 1] = A[i + 1, i] = 1.0
        A[59, 48] = A[48, 59] = 1.0
        for i in range(60, 67):
            A[i, i + 1] = A[i + 1, i] = 1.0
        A[67, 60] = A[60, 67] = 1.0

        # 6. Các cặp dọc mở/nhắm mắt (EAR) & miệng (MAR)
        v_pairs = vertical_pairs or [
            (37, 41), (38, 40), (43, 47), (44, 46),
            (50, 58), (51, 57), (52, 56), (61, 67), (62, 66), (63, 65)
        ]
        for u, v in v_pairs:
            if u < num_nodes and v < num_nodes:
                A[u, v] = A[v, u] = 1.0

        # 7. Các cặp liên vùng khuôn mặt
        cr_pairs = cross_region_pairs or [
            (19, 37), (24, 44), (30, 39), (30, 42), (57, 8), (51, 33)
        ]
        for u, v in cr_pairs:
            if u < num_nodes and v < num_nodes:
                A[u, v] = A[v, u] = 1.0

        # Tự liên kết (Self-loops)
        A = A + np.eye(num_nodes, dtype=np.float32)

        # Chuẩn hóa bậc đối xứng: D^{-1/2} A D^{-1/2}
        degrees = np.sum(A, axis=1)
        d_inv_sqrt = np.power(np.maximum(degrees, 1e-8), -0.5)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt

        return torch.from_numpy(A_norm).float()

    def forward(self) -> torch.Tensor:
        orig_dtype = self.E1.dtype
        scale = 1.0 / math.sqrt(self.embed_dim)
        E1_f32 = self.E1.float()
        E2_f32 = self.E2.float()
        spatial_weights = F.relu(torch.matmul(E1_f32, E2_f32.transpose(0, 1)) * scale)
        A_adp = F.softmax(spatial_weights, dim=-1)
        # Giới hạn alpha qua tanh để ma trận kề thích nghi không tăng trưởng vô hạn
        alpha_bounded = torch.tanh(self.alpha.float())
        A_total = self.A_prior.float() + alpha_bounded * A_adp
        return A_total.to(orig_dtype)


class SpatialGCNLayer(nn.Module):
    """
    Tầng Tích chập Đồ thị Không gian (GCN) không làm triệt tiêu gradient:
    Thay thế F.softmax bằng F.gelu(LayerNorm) kèm kết nối phần dư.
    """

    def __init__(
            self,
            in_dim: int = 64,
            hidden_dim: int = 64,
            out_dim: int = 64,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.W0 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.res_proj = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x: torch.Tensor, A_adp: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        device_type = x.device.type
        # Bảo vệ số học: chạy phép nhân đồ thị và LayerNorm dưới float32 để chống tràn số float16 (max 65,504)
        with torch.amp.autocast(device_type, enabled=False):
            x_perm = x.float().permute(0, 3, 2, 1).contiguous()  # [B, T, N, D]
            A_f32 = A_adp.float()

            h1 = self.W0(x_perm)  # [B, T, N, H]
            # Nhân đồ thị chuẩn xác: đỉnh i tổng hợp đặc trưng từ các đỉnh láng giềng j theo A_adp[i, j]
            # torch.matmul(A_f32, h1) tự động broadcast qua [B, T], chạy bằng cuBLAS GEMM nhanh gấp 3-4 lần einsum
            h1 = torch.matmul(A_f32, h1)
            h1 = F.relu(h1)
            h1 = self.dropout(h1)

            h2 = self.W1(h1)  # [B, T, N, M]
            h2 = torch.matmul(A_f32, h2)
            z = F.gelu(self.norm(h2)) + self.res_proj(x_perm)

        return z.to(orig_dtype).permute(0, 3, 2, 1).contiguous()


class GatedDilatedCausalTCN(nn.Module):
    """
    Tầng Tích chập Thời gian Nhân quả Giãn nở có Cổng (Gated Dilated Causal TCN):
    h = tanh(Theta_1 * X) (x) sigmoid(Theta_2 * X).
    """

    def __init__(
            self,
            in_channels: int = 64,
            dilation_channels: int = 64,
            kernel_size: int = 2,
            dilation: int = 1,
    ):
        super().__init__()
        self.causal_padding = (kernel_size - 1) * dilation
        self.theta_1 = nn.Conv2d(
            in_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            bias=True,
        )
        self.theta_2 = nn.Conv2d(
            in_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pad = F.pad(x, (self.causal_padding, 0, 0, 0))
        tcn_a = torch.tanh(self.theta_1(x_pad))
        tcn_b = torch.sigmoid(self.theta_2(x_pad))
        return tcn_a * tcn_b


class STGLBlock(nn.Module):
    """
    Khối Không gian - Thời gian STGL Block kết hợp Gated TCN và Spatial GCN:
    Đường tắt skip_conv được tính SAU spatial_gcn để dòng gradient chảy xuyên suốt qua tầng đồ thị.
    """

    def __init__(
            self,
            residual_channels: int = 64,
            dilation_channels: int = 64,
            skip_channels: int = 128,
            kernel_size: int = 2,
            dilation: int = 1,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.gated_tcn = GatedDilatedCausalTCN(
            in_channels=residual_channels,
            dilation_channels=dilation_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.spatial_gcn = SpatialGCNLayer(
            in_dim=dilation_channels,
            hidden_dim=residual_channels,
            out_dim=residual_channels,
            dropout=dropout,
        )
        # Skip connection lấy từ h_gcn để truyền gradient trực tiếp qua GCN
        self.skip_conv = nn.Conv2d(residual_channels, skip_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(residual_channels)
        self.res_scale = nn.Parameter(torch.ones(1))

    def forward(
            self,
            x: torch.Tensor,
            A_adp: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        h_tcn = self.gated_tcn(x)
        h_gcn = self.spatial_gcn(h_tcn, A_adp)
        skip = self.skip_conv(h_gcn)
        x_res = self.bn(self.res_scale * h_gcn + residual)
        return x_res, skip


class LiteFat(nn.Module):
    """
    Mô hình LiteFat v2 Hoàn chỉnh:
      - Hỗ trợ linh hoạt cả 2 chế độ: Đa phương thức (Multimodal) và Chỉ dùng Landmark (Landmark-Only).
      - Học ma trận kề thích nghi kết hợp với kề giải phẫu A_prior.
      - Attentive Node Pooling tập trung vào mắt và miệng thay vì trung bình đều.
    """

    def __init__(
            self,
            num_nodes: int = 68,
            feature_dim: int = 64,
            mobilenet_proj_hidden: int = 192,
            node_embed_dim: int = 16,
            residual_channels: int = 64,
            dilation_channels: int = 64,
            skip_channels: int = 128,
            end_channels: int = 256,
            num_classes: int = 2,
            num_blocks: int = 3,
            layers_per_block: int = 2,
            kernel_size: int = 2,
            pretrained_backbone: bool = True,
            dropout: float = 0.3,
            freeze_backbone: bool = False,
            vertical_pairs: Optional[List[Tuple[int, int]]] = None,
            cross_region_pairs: Optional[List[Tuple[int, int]]] = None,
            static_shape_ratio: float = 0.25,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.layers_per_block = layers_per_block
        self.kernel_size = kernel_size
        self.static_shape_ratio = float(static_shape_ratio)

        # 1. MobileNetV3-Small trích xuất ngữ cảnh thị giác từ ảnh mặt
        self.context_encoder = MobileNetV3ContextEncoder(
            out_dim=feature_dim,
            proj_hidden_dim=mobilenet_proj_hidden,
            pretrained=pretrained_backbone,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

        # 2. Dung hợp đa phương thức FiLM chống Rank-1 Bottleneck
        self.fusion = MultimodalFusionModule(
            num_features=6,
            feature_dim=feature_dim,
            num_nodes=num_nodes,
            static_shape_ratio=static_shape_ratio,
        )

        # 3. Ma trận kề giải phẫu + tự thích nghi
        self.adaptive_adj = SelfAdaptiveAdjacency(
            num_nodes=num_nodes,
            embed_dim=node_embed_dim,
            vertical_pairs=vertical_pairs,
            cross_region_pairs=cross_region_pairs,
        )

        # Tầng chiếu đầu vào đồ thị
        self.start_conv = nn.Conv2d(feature_dim, residual_channels, kernel_size=(1, 1))

        # 4. Các khối Không gian - Thời gian (STGL Blocks)
        self.stgl_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            dilation = 1
            for _ in range(layers_per_block):
                self.stgl_blocks.append(
                    STGLBlock(
                        residual_channels=residual_channels,
                        dilation_channels=dilation_channels,
                        skip_channels=skip_channels,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        dropout=dropout,
                    )
                )
                dilation *= 2

        # Đầu phân loại dự đoán cấp khung hình
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, num_classes, kernel_size=(1, 1))

        # Attentive Node Pooling: Tự học trọng số chú ý trên đặc trưng 256D của 68 đỉnh khuôn mặt
        self.node_gate = nn.Sequential(
            nn.Conv2d(end_channels, 64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
        )
        # Prior giải phẫu thiên vị vùng mắt (36..47) và vùng miệng (48..67)
        node_prior = torch.zeros(1, 1, num_nodes, 1)
        node_prior[:, :, 36:48, :] = 1.0  # Mắt
        node_prior[:, :, 48:68, :] = 1.0  # Miệng
        self.register_buffer("node_prior_bias", node_prior)

        # Attentive Temporal Pooling: Tự học trọng số chú ý từ vector đặc trưng khung hình (end_channels) thay vì class logits
        self.temp_norm = nn.LayerNorm(end_channels)
        self.temporal_gate = nn.Sequential(
            nn.Linear(end_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.temp_scale = nn.Parameter(torch.tensor(2.0))

        # Đầu phân loại độc lập cấp đoạn video (Clip-Level Classifier):
        # Tách biệt không gian biểu diễn cấp đoạn và không gian dự đoán frame để không bị ảnh hưởng bởi bias âm của frame
        self.clip_classifier = nn.Sequential(
            nn.Linear(end_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward_stgl_from_fused(
            self,
            X: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chạy phần đồ thị STGL từ tensor đặc trưng X [B, D, N, T]."""
        A_adp = self.adaptive_adj()
        h = self.start_conv(X)

        B, _, _, T = X.shape
        t_mask = None
        if lengths is not None:
            t_idx = torch.arange(T, device=X.device).unsqueeze(0)  # [1, T]
            t_mask = (t_idx < lengths.to(X.device).unsqueeze(1)).float().view(B, 1, 1, T)

        skip_sum = None
        for block in self.stgl_blocks:
            h, skip = block(h, A_adp)
            if t_mask is not None:
                h = h * t_mask
                skip = skip * t_mask
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        out = F.relu(skip_sum)
        h_end = F.relu(self.end_conv_1(out))  # [B, end_channels, N, T]

        # Attentive Spatial Pooling trên N=68 đỉnh dựa trên không gian đặc trưng 256D và prior giải phẫu
        node_attn = self.node_gate(h_end) + self.node_prior_bias  # [B, 1, N, T]
        gate = F.softmax(node_attn, dim=2)  # [B, 1, N, T]

        # Đặc trưng cấp khung hình sau khi gom tụ các đỉnh [B, end_channels, T]
        pooled_node_feat = (h_end * gate).sum(dim=2)  # [B, end_channels, T]
        frame_feat = pooled_node_feat.permute(0, 2, 1).contiguous()  # [B, T, end_channels]

        # Dự đoán logits cấp khung hình [B, T, num_classes]
        frame_logits = self.end_conv_2(pooled_node_feat.unsqueeze(2)).squeeze(2).permute(0, 2, 1).contiguous()

        # Multi-Scale Event MIL Pooling cho Clip Logits
        feat_norm = self.temp_norm(frame_feat)
        temp_scores = self.temporal_gate(feat_norm) * self.temp_scale  # [B, T, 1]

        if lengths is not None:
            valid_bool = (torch.arange(T, device=frame_logits.device).unsqueeze(0) < lengths.to(frame_logits.device).unsqueeze(1)).unsqueeze(-1)
            valid_mask = valid_bool.type_as(frame_logits)
            denom = valid_mask.sum(dim=1).clamp(min=1.0)

            # 1. Trọng số chú ý sắc nét (Sharp Attention)
            masked_scores = temp_scores.masked_fill(~valid_bool, -1e4)
            temp_weights = F.softmax(masked_scores, dim=1) * valid_mask
            temp_weights = temp_weights / temp_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

            # 2. Gộp đặc trưng đa nhánh: Mean + Sharp Attention + Top-K Event
            feat_mean = (frame_feat * valid_mask).sum(dim=1) / denom  # [B, end_channels]
            feat_attn = (frame_feat * temp_weights).sum(dim=1)         # [B, end_channels]

            # Top-K event pooling (lấy top 25% frames có điểm chú ý sự kiện cao nhất theo độ dài từng mẫu)
            scores_2d = masked_scores.squeeze(-1)  # [B, T]
            k_per = (lengths.float() * 0.25).round().clamp(min=1.0).long()  # [B]
            ranks = scores_2d.argsort(dim=1, descending=True).argsort(dim=1)  # [B, T]
            topk_mask = (ranks < k_per.unsqueeze(1)) & valid_bool.squeeze(-1)  # [B, T]
            feat_topk = (frame_feat * topk_mask.unsqueeze(-1).float()).sum(dim=1) / topk_mask.sum(dim=1, keepdim=True).clamp(min=1.0).float()

            pooled_clip_feat = 0.50 * feat_attn + 0.30 * feat_topk + 0.20 * feat_mean
            clip_head_logits = self.clip_classifier(pooled_clip_feat)  # [B, num_classes]

            # Kết hợp với attentive frame logits
            attn_frame_logits = (frame_logits * temp_weights).sum(dim=1)
            clip_logits = 0.60 * clip_head_logits + 0.40 * attn_frame_logits
        else:
            temp_weights = F.softmax(temp_scores, dim=1)
            feat_mean = frame_feat.mean(dim=1)
            feat_attn = (frame_feat * temp_weights).sum(dim=1)

            k_val = max(1, int(round(T * 0.25)))
            scores_2d = temp_scores.squeeze(-1)
            topk_idx = torch.topk(scores_2d, k=k_val, dim=1).indices
            batch_indices = torch.arange(B, device=frame_feat.device).unsqueeze(1).expand(-1, k_val)
            feat_topk = frame_feat[batch_indices, topk_idx, :].mean(dim=1)

            pooled_clip_feat = 0.50 * feat_attn + 0.30 * feat_topk + 0.20 * feat_mean
            clip_head_logits = self.clip_classifier(pooled_clip_feat)
            attn_frame_logits = (frame_logits * temp_weights).sum(dim=1)
            clip_logits = 0.60 * clip_head_logits + 0.40 * attn_frame_logits

        return frame_logits, clip_logits

    def forward(
            self,
            video_frames: Optional[torch.Tensor] = None,
            landmarks: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            return_clip_logits: bool = True,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Chuẩn hóa nạp duy nhất một kiểu:
            model(video_frames=..., landmarks=..., lengths=..., return_clip_logits=...)

        - Khi video_frames is not None và video_frames.numel() > 0:
            Trích xuất vector ngữ cảnh d [B, T, D] từ MobileNetV3 Context Encoder.
        - Khi video_frames is None:
            Chạy chế độ Fast Landmark-Only (d = None, chỉ sử dụng nhánh động học mốc mặt).
        """
        if landmarks is None:
            raise ValueError("Tham số 'landmarks' [B, T, N, 2] bắt buộc phải được cung cấp!")

        d = None
        if video_frames is not None and video_frames.numel() > 0:
            B, T_v, C_img, H, W = video_frames.shape
            frames_flat = video_frames.view(B * T_v, C_img, H, W)
            d_flat = self.context_encoder(frames_flat)
            d_sparse = d_flat.view(B, T_v, self.feature_dim)
            # Xác định chiều dài thời gian T của landmark
            T_target = landmarks.shape[1] if (landmarks.dim() == 4 and landmarks.size(-1) in (2, 3)) else landmarks.shape[2]
            if T_v != T_target:
                # Nội suy đặc trưng ngữ cảnh thị giác từ T_v khung hình lên trọn vẹn T_target khung hình
                d = F.interpolate(d_sparse.permute(0, 2, 1), size=T_target, mode="linear", align_corners=False).permute(0, 2, 1)
            else:
                d = d_sparse

        if mask is None and lengths is not None:
            mask = KinematicFeatureExtractor.build_temporal_mask(
                landmarks.permute(0, 3, 1, 2) if landmarks.dim() == 4 and landmarks.size(-1) == 2 else landmarks,
                lengths=lengths,
            )

        X = self.fusion(landmarks, d=d, mask=mask)
        frame_logits, clip_logits = self.forward_stgl_from_fused(X, lengths=lengths)

        if return_clip_logits:
            return frame_logits, clip_logits
        return frame_logits

    @torch.no_grad()
    def forward_asymmetric_step(
            self,
            landmark_window: torch.Tensor,
            d_history: torch.Tensor,
            current_frame: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if current_frame is not None:
            d_new = self.context_encoder(current_frame).unsqueeze(1)
        else:
            d_new = d_history[:, -1:, :]

        d_updated = torch.cat([d_history[:, 1:, :], d_new], dim=1)
        X = self.fusion(landmark_window, d=d_updated)
        _, clip_logits = self.forward_stgl_from_fused(X)
        probs = F.softmax(clip_logits, dim=-1)
        return probs, d_updated

    def freeze_backbone(self, freeze: bool = True, freeze_projector: bool = False) -> None:
        self.context_encoder.freeze_backbone(freeze=freeze, freeze_projector=freeze_projector)

    def unfreeze_backbone(self, unfreeze_last_n_blocks: Optional[int] = None) -> None:
        self.context_encoder.unfreeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    def set_backbone_trainable(self, trainable: bool, unfreeze_last_n_blocks: Optional[int] = 3) -> None:
        """Đóng/mở khóa backbone MobileNetV3 theo epoch huấn luyện."""
        if not trainable:
            self.freeze_backbone(freeze=True, freeze_projector=False)
        else:
            self.unfreeze_backbone(unfreeze_last_n_blocks=unfreeze_last_n_blocks)

    @property
    def is_backbone_frozen(self) -> bool:
        return self.context_encoder.is_backbone_frozen

    def get_trainable_params_count(self) -> Dict[str, Any]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.context_encoder.features.parameters())
        backbone_trainable = sum(p.numel() for p in self.context_encoder.features.parameters() if p.requires_grad)
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
            "backbone_total": backbone_total,
            "backbone_trainable": backbone_trainable,
            "is_backbone_frozen": self.is_backbone_frozen,
        }
