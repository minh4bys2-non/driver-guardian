# =====================================================================
# File: model.py
# Dự án: Driver Guardian - Module LiteFat (IEEE/RSJ IROS 2025 Upgraded)
# Tác vụ: Định nghĩa kiến trúc LiteFat_v1 dự đoán điểm buồn ngủ KSS (1-9)
#         với cửa sổ mặc định 60s (10 FPS) và cơ chế trượt cửa sổ
#         stride 30s tích lũy đặc trưng kép (Overlap + Gated GRU Memory)
#         cho các chuỗi có độ dài lớn hơn 60s.
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
            fps: float = 10.0,
    ):
        super().__init__()
        self.nose_idx = nose_idx
        self.left_eye_idx = left_eye_idx
        self.right_eye_idx = right_eye_idx
        self.eps = eps
        self.static_shape_ratio = float(static_shape_ratio)
        self.fps = float(max(fps, 1.0))

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

        # Chuẩn hóa mềm đưa RMS về dải tự nhiên theo tần số lấy mẫu FPS (quy chiếu gốc 20 FPS ~0.035 IOD/frame)
        fps_ratio = 20.0 / self.fps
        vel_scale = torch.clamp(vel_rms, min=0.012 * fps_ratio, max=0.10 * fps_ratio) / (0.035 * fps_ratio)
        acc_scale = torch.clamp(acc_rms, min=0.010 * fps_ratio, max=0.08 * fps_ratio) / (0.025 * fps_ratio)
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
            fps: float = 10.0,
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
            fps=fps,
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


class CrossWindowFeatureAccumulator(nn.Module):
    """
    Khối Tích lũy Đặc trưng Kép qua các Cửa sổ Trượt (Dual-Level Forward Feature Accumulation):
      1. Overlap Frame Gated Fusion: Dung hợp đặc trưng khung hình ở vùng chồng lấp (mặc định 30s = 300 frames)
         giữa cuối cửa sổ trước (w-1) và đầu cửa sổ hiện tại (w).
      2. Cross-Window Gated GRU Memory: Cập nhật trạng thái tích lũy cấp cửa sổ xuyên suốt chiều dài mẫu (>60s)
         kèm bảo vệ độ dài thực tế (active_mask) để padding không làm trôi trí nhớ mệt mỏi.
      3. Causal Window Attention: Tổng hợp chuỗi trạng thái các cửa sổ lịch sử thành biểu diễn toàn mẫu.
    """

    def __init__(
            self,
            feature_dim: int = 256,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_dim = feature_dim

        # 1. Nhánh dung hợp đặc trưng khung hình vùng chồng lấp 30s
        self.overlap_norm_cur = nn.LayerNorm(feature_dim)
        self.overlap_norm_prev = nn.LayerNorm(feature_dim)
        self.overlap_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

        # 2. Nhánh bộ nhớ trạng thái có cổng cấp cửa sổ (Gated GRU Memory)
        self.input_norm = nn.LayerNorm(feature_dim)
        self.state_norm = nn.LayerNorm(feature_dim)
        self.gru_cell = nn.GRUCell(feature_dim, feature_dim)
        self.mem_gate = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.out_norm = nn.LayerNorm(feature_dim)

        # 3. Nhánh tổng hợp chuỗi trạng thái các cửa sổ lịch sử
        self.window_attn_gate = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def fuse_overlap_frames(
            self,
            cur_frame_feat: torch.Tensor,
            prev_tail_feat: Optional[torch.Tensor],
            cur_lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Dung hợp vùng khung hình chồng lấp giữa đầu cửa sổ hiện tại [B, T_w, D]
        và vùng tail đã tích lũy từ cửa sổ trước [B, T_ov, D].
        """
        if prev_tail_feat is None or prev_tail_feat.size(1) == 0 or cur_frame_feat.size(1) == 0:
            return cur_frame_feat

        L_ov = min(cur_frame_feat.size(1), prev_tail_feat.size(1))
        cur_ov = cur_frame_feat[:, :L_ov, :]
        prev_ov = prev_tail_feat[:, :L_ov, :]

        orig_dtype = cur_frame_feat.dtype
        dev = cur_frame_feat.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"

        with torch.amp.autocast(dev_type, enabled=False):
            cur_ov_32 = cur_ov.float()
            prev_ov_32 = prev_ov.float()
            gate_in = torch.cat([self.overlap_norm_cur(cur_ov_32), self.overlap_norm_prev(prev_ov_32)], dim=-1)
            gate = self.overlap_gate(gate_in)
            fused_ov = gate * cur_ov_32 + (1.0 - gate) * prev_ov_32

            if cur_lengths is not None:
                valid_ov = (
                    torch.arange(L_ov, device=dev).unsqueeze(0)
                    < cur_lengths.to(dev).unsqueeze(1)
                ).unsqueeze(-1).float()
                fused_ov = fused_ov * valid_ov + cur_ov_32 * (1.0 - valid_ov)

        fused_ov = fused_ov.to(orig_dtype)
        if L_ov < cur_frame_feat.size(1):
            return torch.cat([fused_ov, cur_frame_feat[:, L_ov:, :]], dim=1)
        return fused_ov

    def update_window_memory(
            self,
            window_feat: torch.Tensor,
            prev_memory: Optional[torch.Tensor] = None,
            active_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Cập nhật trạng thái tích lũy cấp cửa sổ h_k = GatedGRU(z_k, h_{k-1}).
        Nếu mẫu trong batch có active_mask == False (toàn bộ frame ở cửa sổ này là padding),
        giữ nguyên trạng thái prev_memory.
        """
        orig_dtype = window_feat.dtype
        dev = window_feat.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"

        with torch.amp.autocast(dev_type, enabled=False):
            w_32 = window_feat.float()
            if prev_memory is None:
                h_new = self.out_norm(w_32)
            else:
                p_32 = prev_memory.float()
                z = self.input_norm(w_32)
                h_prev = self.state_norm(p_32)
                h_gru = self.gru_cell(z, h_prev)
                alpha = self.mem_gate(torch.cat([z, h_prev, h_gru], dim=-1))
                h_new = self.out_norm(alpha * h_gru + (1.0 - alpha) * w_32)
                if active_mask is not None:
                    mask_col = active_mask.view(-1, 1).bool().to(dev)
                    h_new = torch.where(mask_col, h_new, p_32)

        return h_new.to(orig_dtype)

    def aggregate_window_states(
            self,
            window_states: torch.Tensor,
            window_valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Tổng hợp chuỗi trạng thái [B, W_num, D] qua các cửa sổ trượt thành vector đặc trưng cuối [B, D].
        Kết hợp trạng thái tích lũy cuối cùng hợp lệ (65%) và Causal Window Attention (35%).
        """
        B, W_num, _ = window_states.shape
        if W_num == 1:
            return window_states[:, 0, :]

        orig_dtype = window_states.dtype
        dev = window_states.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"

        with torch.amp.autocast(dev_type, enabled=False):
            ws_32 = window_states.float()
            if window_valid_mask is not None:
                valid_bool = window_valid_mask.bool().to(dev)
                last_idx = valid_bool.long().sum(dim=1).clamp(min=1) - 1
                batch_idx = torch.arange(B, device=dev)
                last_state = ws_32[batch_idx, last_idx, :]
            else:
                valid_bool = torch.ones((B, W_num), dtype=torch.bool, device=dev)
                last_state = ws_32[:, -1, :]

            recency_bias = torch.linspace(-0.5, 0.5, W_num, device=dev, dtype=ws_32.dtype).view(1, W_num)
            scores = self.window_attn_gate(ws_32).squeeze(-1) + recency_bias
            scores = scores.masked_fill(~valid_bool, -1e4)
            attn_w = F.softmax(scores, dim=1) * valid_bool.float()
            attn_w = attn_w / attn_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            attn_state = (ws_32 * attn_w.unsqueeze(-1)).sum(dim=1)

            final_state = 0.65 * last_state + 0.35 * attn_state

        return final_state.to(orig_dtype)


class HybridKSSHead(nn.Module):
    """
    Đầu dự đoán điểm KSS (Karolinska Sleepiness Scale, thang 1-9) theo cơ chế Lai ghép:
      - Kết hợp Kỳ vọng phân phối 9 mức KSS [1..9] với Nhánh hồi quy liên tục có chặn [1.0, 9.0].
      - Trả về đồng thời điểm KSS liên tục và logits 9 lớp ở cả cấp khung hình và cấp đoạn/cửa sổ.
    """

    def __init__(
            self,
            in_channels: int = 256,
            num_classes: int = 9,
            kss_min: float = 1.0,
            kss_max: float = 9.0,
            dropout: float = 0.3,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.kss_min = float(kss_min)
        self.kss_max = float(kss_max)

        kss_levels = torch.linspace(self.kss_min, self.kss_max, steps=num_classes, dtype=torch.float32)
        self.register_buffer("kss_values", kss_levels)

        # Cấp khung hình (Frame-level)
        self.frame_cls_conv = nn.Conv2d(in_channels, num_classes, kernel_size=(1, 1))
        self.frame_reg_head = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

        # Cấp cửa sổ / toàn mẫu (Clip-level)
        self.clip_classifier = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
        self.clip_reg_head = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )
        self.kss_mix_logit = nn.Parameter(torch.tensor(0.5))

    def forward_frames(
            self,
            pooled_node_feat: torch.Tensor,
            frame_feat: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            pooled_node_feat: [B, D, T]
            frame_feat: [B, T, D]
        Returns:
            frame_kss: [B, T] trong đoạn [kss_min, kss_max]
            frame_logits: [B, T, num_classes]
        """
        orig_dtype = frame_feat.dtype
        dev = frame_feat.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"

        frame_logits = self.frame_cls_conv(pooled_node_feat.unsqueeze(2)).squeeze(2).permute(0, 2, 1).contiguous()
        with torch.amp.autocast(dev_type, enabled=False):
            logits_32 = frame_logits.float()
            probs = F.softmax(logits_32, dim=-1)
            kss_vals = self.kss_values.view(1, 1, -1).to(dev)
            kss_exp = (probs * kss_vals).sum(dim=-1)
            offset = 0.5 * torch.tanh(self.frame_reg_head(frame_feat.float()).squeeze(-1))
            frame_kss = torch.clamp(kss_exp + offset, min=self.kss_min, max=self.kss_max)

        return frame_kss.to(orig_dtype), frame_logits

    def forward_clip(
            self,
            clip_feat: torch.Tensor,
            attn_frame_logits: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            clip_feat: [B, D]
            attn_frame_logits: [B, num_classes] (tùy chọn)
        Returns:
            clip_kss: [B] trong đoạn [kss_min, kss_max]
            clip_logits: [B, num_classes]
        """
        orig_dtype = clip_feat.dtype
        dev = clip_feat.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"

        head_logits = self.clip_classifier(clip_feat)
        if attn_frame_logits is not None:
            clip_logits = 0.65 * head_logits + 0.35 * attn_frame_logits
        else:
            clip_logits = head_logits

        with torch.amp.autocast(dev_type, enabled=False):
            logits_32 = clip_logits.float()
            probs = F.softmax(logits_32, dim=-1)
            kss_vals = self.kss_values.view(1, -1).to(dev)
            kss_exp = (probs * kss_vals).sum(dim=-1)

            span = self.kss_max - self.kss_min
            kss_reg = self.kss_min + span * torch.sigmoid(self.clip_reg_head(clip_feat.float()).squeeze(-1))
            mix_w = torch.sigmoid(self.kss_mix_logit.float())
            clip_kss = torch.clamp(mix_w * kss_exp + (1.0 - mix_w) * kss_reg, min=self.kss_min, max=self.kss_max)

        return clip_kss.to(orig_dtype), clip_logits


class LiteFat_v1(nn.Module):
    """
    Mô hình LiteFat_v1 Dự đoán Điểm Buồn ngủ KSS (thang 1-9):
      - Cửa sổ mặc định 60s ở 10 FPS (600 frames) và bước trượt mặc định 30s (300 frames).
      - Hỗ trợ tự động chạy Sliding Window với tích lũy đặc trưng kép (Overlap Frame Fusion +
        Cross-Window Gated GRU Memory) cho các mẫu có độ dài lớn hơn 60s.
      - Hỗ trợ linh hoạt cả 2 chế độ: Đa phương thức (Multimodal) và Chỉ dùng Landmark (Landmark-Only).
      - Đầu ra Hybrid KSS: hỗ trợ đồng thời điểm KSS liên tục [1.0, 9.0] và logits 9 mức KSS.
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
            num_classes: int = 9,
            kss_min: float = 1.0,
            kss_max: float = 9.0,
            fps: int = 10,
            window_sec: float = 60.0,
            stride_sec: float = 30.0,
            num_blocks: int = 3,
            layers_per_block: int = 4,
            kernel_size: int = 3,
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
        self.end_channels = end_channels
        self.num_classes = num_classes
        self.kss_min = float(kss_min)
        self.kss_max = float(kss_max)
        self.fps = int(max(1, fps))
        self.window_sec = float(window_sec)
        self.stride_sec = float(stride_sec)
        self.window_size = max(1, int(round(self.window_sec * self.fps)))
        self.stride_size = max(1, int(round(self.stride_sec * self.fps)))
        self.overlap_size = max(0, self.window_size - self.stride_size)
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

        # 2. Dung hợp đa phương thức FiLM chống Rank-1 Bottleneck (chuẩn hóa động học theo FPS)
        self.fusion = MultimodalFusionModule(
            num_features=6,
            feature_dim=feature_dim,
            num_nodes=num_nodes,
            static_shape_ratio=static_shape_ratio,
            fps=float(self.fps),
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

        # 4. Các khối Không gian - Thời gian (STGL Blocks) mở rộng dilation 1, 2, 4, 8 cho cửa sổ 60s
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

        # Chiếu đặc trưng đầu ra đồ thị
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))

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

        # Attentive Temporal Pooling: Tự học trọng số chú ý từ vector đặc trưng khung hình (end_channels)
        self.temp_norm = nn.LayerNorm(end_channels)
        self.temporal_gate = nn.Sequential(
            nn.Linear(end_channels, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )
        self.temp_scale = nn.Parameter(torch.tensor(2.0))

        # 5. Khối tích lũy đặc trưng kép xuyên suốt các cửa sổ trượt 60s (stride 30s)
        self.accumulator = CrossWindowFeatureAccumulator(
            feature_dim=end_channels,
            dropout=dropout,
        )

        # 6. Đầu dự đoán KSS Lai ghép (Hybrid 9-level expectation + bounded regression [1.0, 9.0])
        self.kss_head = HybridKSSHead(
            in_channels=end_channels,
            num_classes=num_classes,
            kss_min=self.kss_min,
            kss_max=self.kss_max,
            dropout=dropout,
        )

    @property
    def end_conv_2(self) -> nn.Conv2d:
        """Alias tương thích ngược trỏ tới tầng phân loại frame của kss_head."""
        return self.kss_head.frame_cls_conv

    @property
    def clip_classifier(self) -> nn.Sequential:
        """Alias tương thích ngược trỏ tới bộ phân loại clip của kss_head."""
        return self.kss_head.clip_classifier

    @staticmethod
    def _build_sliding_window_ranges(
            total_frames: int,
            window_size: int,
            stride_size: int,
    ) -> List[Tuple[int, int]]:
        """
        Tạo danh sách các khoảng cửa sổ (start, end) với độ dài window_size và bước trượt stride_size.
        Đảm bảo khoảng cách giữa các cửa sổ liên tiếp luôn đúng bằng stride_size để vùng chồng lấp khớp chuẩn.
        """
        if total_frames <= window_size:
            return [(0, total_frames)]

        starts = list(range(0, max(1, total_frames - window_size + 1), stride_size))
        if not starts:
            starts = [0]
        while starts[-1] + window_size < total_frames:
            starts.append(starts[-1] + stride_size)

        return [(s, min(total_frames, s + window_size)) for s in starts]

    @staticmethod
    def _normalize_landmarks_shape(landmarks: torch.Tensor) -> torch.Tensor:
        """Chuẩn hóa tensor landmarks về dạng [B, T, N, C]."""
        if landmarks.dim() == 4:
            if landmarks.size(-1) in (2, 3):
                return landmarks
            if landmarks.size(1) in (2, 3):
                return landmarks.permute(0, 2, 3, 1).contiguous()
        return landmarks

    def _encode_visual_context(
            self,
            video_frames: torch.Tensor,
            target_length: int,
            max_chunk: int = 64,
    ) -> torch.Tensor:
        """Trích xuất vector ngữ cảnh thị giác theo từng chunk nhỏ và nội suy về target_length."""
        B, T_v, C_img, H, W = video_frames.shape
        frames_flat = video_frames.reshape(B * T_v, C_img, H, W)
        total_imgs = frames_flat.size(0)

        if total_imgs <= max_chunk:
            d_flat = self.context_encoder(frames_flat)
        else:
            chunks = [self.context_encoder(frames_flat[i:i + max_chunk]) for i in range(0, total_imgs, max_chunk)]
            d_flat = torch.cat(chunks, dim=0)

        d_sparse = d_flat.view(B, T_v, self.feature_dim)
        if T_v != target_length:
            return F.interpolate(
                d_sparse.permute(0, 2, 1),
                size=target_length,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        return d_sparse

    def _extract_window_frame_features(
            self,
            X: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chạy đồ thị STGL và Attentive Spatial Node Pooling cho 1 cửa sổ."""
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

        # Attentive Spatial Pooling trên N=68 đỉnh dựa trên đặc trưng 256D và prior giải phẫu
        node_attn = self.node_gate(h_end) + self.node_prior_bias  # [B, 1, N, T]
        gate = F.softmax(node_attn, dim=2)  # [B, 1, N, T]

        pooled_node_feat = (h_end * gate).sum(dim=2)  # [B, end_channels, T]
        frame_feat = pooled_node_feat.permute(0, 2, 1).contiguous()  # [B, T, end_channels]
        return pooled_node_feat, frame_feat

    def _pool_window_temporal_features(
            self,
            frame_feat: torch.Tensor,
            frame_logits: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gom tụ đặc trưng thời gian đa nhánh (Attn + Top-K + Mean) cho 1 cửa sổ."""
        B, T, _ = frame_feat.shape
        feat_norm = self.temp_norm(frame_feat)
        temp_scores = self.temporal_gate(feat_norm) * self.temp_scale  # [B, T, 1]

        if lengths is not None:
            valid_bool = (
                torch.arange(T, device=frame_logits.device).unsqueeze(0)
                < lengths.to(frame_logits.device).unsqueeze(1)
            ).unsqueeze(-1)
            valid_mask = valid_bool.type_as(frame_logits)
            denom = valid_mask.sum(dim=1).clamp(min=1.0)

            masked_scores = temp_scores.masked_fill(~valid_bool, -1e4)
            temp_weights = F.softmax(masked_scores, dim=1) * valid_mask
            temp_weights = temp_weights / temp_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

            feat_mean = (frame_feat * valid_mask).sum(dim=1) / denom
            feat_attn = (frame_feat * temp_weights).sum(dim=1)

            scores_2d = masked_scores.squeeze(-1)  # [B, T]
            k_per = (lengths.float() * 0.25).round().clamp(min=1.0).long()  # [B]
            ranks = scores_2d.argsort(dim=1, descending=True).argsort(dim=1)  # [B, T]
            topk_mask = (ranks < k_per.unsqueeze(1)) & valid_bool.squeeze(-1)  # [B, T]
            feat_topk = (
                (frame_feat * topk_mask.unsqueeze(-1).type_as(frame_feat)).sum(dim=1)
                / topk_mask.sum(dim=1, keepdim=True).clamp(min=1.0).type_as(frame_feat)
            )

            pooled_window_feat = 0.50 * feat_attn + 0.30 * feat_topk + 0.20 * feat_mean
            attn_frame_logits = (frame_logits * temp_weights).sum(dim=1)

            active_col = (lengths > 0).to(frame_logits.device).view(B, 1)
            pooled_window_feat = pooled_window_feat * active_col.type_as(pooled_window_feat)
            attn_frame_logits = attn_frame_logits * active_col.type_as(attn_frame_logits)
        else:
            temp_weights = F.softmax(temp_scores, dim=1)
            feat_mean = frame_feat.mean(dim=1)
            feat_attn = (frame_feat * temp_weights).sum(dim=1)

            k_val = max(1, int(round(T * 0.25)))
            scores_2d = temp_scores.squeeze(-1)
            topk_idx = torch.topk(scores_2d, k=k_val, dim=1).indices
            batch_indices = torch.arange(B, device=frame_feat.device).unsqueeze(1).expand(-1, k_val)
            feat_topk = frame_feat[batch_indices, topk_idx, :].mean(dim=1)

            pooled_window_feat = 0.50 * feat_attn + 0.30 * feat_topk + 0.20 * feat_mean
            attn_frame_logits = (frame_logits * temp_weights).sum(dim=1)

        return pooled_window_feat, attn_frame_logits, temp_weights

    def _extract_overlap_tail(
            self,
            frame_feat: torch.Tensor,
            lengths: Optional[torch.Tensor],
            stride_size: int,
            overlap_size: int,
            prev_tail: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Trích xuất vùng đặc trưng khung hình chồng lấp (từ stride_size trở đi) cho cửa sổ kế tiếp."""
        if overlap_size <= 0:
            return None

        B, T_w, D = frame_feat.shape
        dev = frame_feat.device

        if T_w > stride_size:
            raw_tail = frame_feat[:, stride_size:min(T_w, stride_size + overlap_size), :]
            cur_len = raw_tail.size(1)
            if cur_len < overlap_size:
                overlap_tail = F.pad(raw_tail, (0, 0, 0, overlap_size - cur_len))
            else:
                overlap_tail = raw_tail

            if lengths is not None:
                valid_tail_len = (lengths.to(dev) - stride_size).clamp(min=0, max=overlap_size)
                tail_mask = (
                    torch.arange(overlap_size, device=dev).unsqueeze(0)
                    < valid_tail_len.unsqueeze(1)
                ).unsqueeze(-1).type_as(overlap_tail)
                overlap_tail = overlap_tail * tail_mask
        else:
            overlap_tail = frame_feat.new_zeros((B, overlap_size, D))

        if prev_tail is not None and lengths is not None and prev_tail.size(1) == overlap_size:
            active_col = (lengths.to(dev) > 0).view(B, 1, 1)
            overlap_tail = torch.where(active_col, overlap_tail, prev_tail.to(overlap_tail.dtype))

        return overlap_tail

    def forward_stgl_from_fused(
            self,
            X: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
            prev_state: Optional[Dict[str, torch.Tensor]] = None,
            overlap_size: Optional[int] = None,
            stride_size: Optional[int] = None,
            return_details: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Chạy phần đồ thị STGL từ tensor đặc trưng X [B, D, N, T_w] cho 1 cửa sổ,
        hỗ trợ tích lũy kép từ prev_state (vùng chồng lấp khung hình + bộ nhớ Gated GRU).
        """
        eff_stride = self.stride_size if stride_size is None else int(max(1, stride_size))
        eff_overlap = self.overlap_size if overlap_size is None else int(max(0, overlap_size))

        pooled_node_feat, frame_feat = self._extract_window_frame_features(X, lengths=lengths)

        prev_tail = prev_state.get("overlap_tail_feat") if prev_state is not None else None
        if prev_tail is not None:
            frame_feat = self.accumulator.fuse_overlap_frames(
                frame_feat,
                prev_tail,
                cur_lengths=lengths,
            )
            pooled_node_feat = frame_feat.permute(0, 2, 1).contiguous()

        frame_kss, frame_logits = self.kss_head.forward_frames(pooled_node_feat, frame_feat)
        pooled_window_feat, attn_frame_logits, _ = self._pool_window_temporal_features(
            frame_feat,
            frame_logits,
            lengths=lengths,
        )

        prev_mem = prev_state.get("memory_feat") if prev_state is not None else None
        active_mask = (lengths > 0) if lengths is not None else None
        memory_feat = self.accumulator.update_window_memory(
            pooled_window_feat,
            prev_memory=prev_mem,
            active_mask=active_mask,
        )

        clip_kss, clip_logits = self.kss_head.forward_clip(
            memory_feat,
            attn_frame_logits=attn_frame_logits,
        )

        if return_details:
            overlap_tail_feat = self._extract_overlap_tail(
                frame_feat=frame_feat,
                lengths=lengths,
                stride_size=eff_stride,
                overlap_size=eff_overlap,
                prev_tail=prev_tail,
            )
            new_state: Dict[str, Any] = {
                "memory_feat": memory_feat,
                "overlap_tail_feat": overlap_tail_feat,
                "window_feat": pooled_window_feat,
            }
            return {
                "frame_kss": frame_kss,
                "clip_kss": clip_kss,
                "frame_logits": frame_logits,
                "clip_logits": clip_logits,
                "attn_frame_logits": attn_frame_logits,
                "state": new_state,
            }

        return frame_kss, clip_kss

    def forward_sliding_windows(
            self,
            landmarks: torch.Tensor,
            video_frames: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            window_size: Optional[int] = None,
            stride_size: Optional[int] = None,
            prev_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, Any]:
        """
        Chạy cơ chế cửa sổ trượt (mặc định 60s = 600 frames, stride 30s = 300 frames @ 10 FPS)
        với tích lũy đặc trưng kép phía trước cho các mẫu có độ dài lớn hơn 60s.
        """
        lm_norm = self._normalize_landmarks_shape(landmarks)
        B, T, _, _ = lm_norm.shape
        dev = lm_norm.device

        win_size = self.window_size if window_size is None else int(max(1, window_size))
        str_size = self.stride_size if stride_size is None else int(max(1, stride_size))
        ov_size = max(0, win_size - str_size)

        d_full = None
        if video_frames is not None and video_frames.numel() > 0:
            d_full = self._encode_visual_context(video_frames, target_length=T)

        if mask is None and lengths is not None:
            mask = KinematicFeatureExtractor.build_temporal_mask(
                lm_norm.permute(0, 3, 1, 2),
                lengths=lengths,
            )

        ranges = self._build_sliding_window_ranges(T, win_size, str_size)

        full_frame_kss = lm_norm.new_full((B, T), self.kss_min, dtype=torch.float32)
        full_frame_logits = lm_norm.new_zeros((B, T, self.num_classes), dtype=torch.float32)
        written_mask = torch.zeros((B, T), dtype=torch.bool, device=dev)

        state = prev_state
        window_memories: List[torch.Tensor] = []
        window_kss_list: List[torch.Tensor] = []
        window_logits_list: List[torch.Tensor] = []
        window_attn_logits_list: List[torch.Tensor] = []
        window_valid_list: List[torch.Tensor] = []

        for w_idx, (s, e_valid) in enumerate(ranges):
            sub_len_span = e_valid - s
            sub_landmarks = lm_norm[:, s:e_valid]
            sub_d = d_full[:, s:e_valid] if d_full is not None else None

            if lengths is not None:
                sub_lengths = (lengths.to(dev) - s).clamp(min=0, max=sub_len_span)
                active_mask = sub_lengths > 0
            else:
                sub_lengths = None
                active_mask = torch.ones((B,), dtype=torch.bool, device=dev)

            if mask is not None:
                sub_mask = mask[:, :, s:e_valid, :]
            elif sub_lengths is not None:
                sub_mask = KinematicFeatureExtractor.build_temporal_mask(
                    sub_landmarks.permute(0, 3, 1, 2),
                    lengths=sub_lengths,
                )
            else:
                sub_mask = None

            X_sub = self.fusion(sub_landmarks, d=sub_d, mask=sub_mask)
            win_out = self.forward_stgl_from_fused(
                X_sub,
                lengths=sub_lengths,
                prev_state=state,
                overlap_size=ov_size,
                stride_size=str_size,
                return_details=True,
            )
            state = win_out["state"]

            w_frame_kss = win_out["frame_kss"].float()
            w_frame_logits = win_out["frame_logits"].float()

            if sub_lengths is not None:
                local_valid = (
                    torch.arange(sub_len_span, device=dev).unsqueeze(0)
                    < sub_lengths.unsqueeze(1)
                )
            else:
                local_valid = torch.ones((B, sub_len_span), dtype=torch.bool, device=dev)

            if w_idx == 0:
                full_frame_kss[:, s:e_valid] = w_frame_kss
                full_frame_logits[:, s:e_valid, :] = w_frame_logits
                written_mask[:, s:e_valid] = True
            else:
                update_mask = local_valid | (~written_mask[:, s:e_valid])
                full_frame_kss[:, s:e_valid] = torch.where(
                    update_mask,
                    w_frame_kss,
                    full_frame_kss[:, s:e_valid],
                )
                full_frame_logits[:, s:e_valid, :] = torch.where(
                    update_mask.unsqueeze(-1),
                    w_frame_logits,
                    full_frame_logits[:, s:e_valid, :],
                )
                written_mask[:, s:e_valid] = True

            window_memories.append(state["memory_feat"])
            window_kss_list.append(win_out["clip_kss"])
            window_logits_list.append(win_out["clip_logits"])
            window_attn_logits_list.append(win_out["attn_frame_logits"])
            window_valid_list.append(active_mask)

        window_states = torch.stack(window_memories, dim=1)  # [B, W_num, D]
        window_kss = torch.stack(window_kss_list, dim=1)  # [B, W_num]
        window_logits = torch.stack(window_logits_list, dim=1)  # [B, W_num, num_classes]
        window_attn_logits = torch.stack(window_attn_logits_list, dim=1)  # [B, W_num, num_classes]
        window_valid_mask = torch.stack(window_valid_list, dim=1)  # [B, W_num]

        final_feat = self.accumulator.aggregate_window_states(
            window_states,
            window_valid_mask=window_valid_mask,
        )
        valid_w_float = window_valid_mask.float().unsqueeze(-1)
        final_attn_logits = (
            (window_attn_logits.float() * valid_w_float).sum(dim=1)
            / valid_w_float.sum(dim=1).clamp(min=1.0)
        ).to(final_feat.dtype)

        clip_kss, clip_logits = self.kss_head.forward_clip(
            final_feat,
            attn_frame_logits=final_attn_logits,
        )

        orig_dtype = lm_norm.dtype
        return {
            "frame_kss": full_frame_kss.to(orig_dtype),
            "clip_kss": clip_kss,
            "frame_logits": full_frame_logits.to(orig_dtype),
            "clip_logits": clip_logits,
            "window_kss": window_kss,
            "window_logits": window_logits,
            "state": state,
        }

    def forward(
            self,
            video_frames: Optional[torch.Tensor] = None,
            landmarks: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            return_clip_logits: bool = True,
            return_logits: bool = False,
            return_dict: bool = False,
            use_sliding_window: Optional[bool] = None,
            window_sec: Optional[float] = None,
            stride_sec: Optional[float] = None,
            prev_state: Optional[Dict[str, torch.Tensor]] = None,
            return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[Any, ...], Dict[str, Any]]:
        """
        Dự đoán điểm buồn ngủ KSS (thang 1-9) với cửa sổ mặc định 60s @ 10 FPS (600 frames)
        và tự động chạy Sliding Window (stride 30s = 300 frames) tích lũy đặc trưng phía trước
        khi chuỗi đầu vào dài hơn 60s.

        Args:
            video_frames: Tensor ảnh khuôn mặt [B, T_v, C, H, W] hoặc None (Landmark-Only).
            landmarks: Tensor mốc mặt [B, T, N, 2] (hoặc [B, 2, T, N]).
            lengths: Độ dài thực tế từng mẫu trong batch [B].
            mask: Temporal float mask [B, 1, T, 1].
            return_clip_logits: True để trả kèm đầu ra cấp đoạn/mẫu; False để chỉ trả cấp frame.
            return_logits: True để trả thêm frame_logits và clip_logits 9 mức KSS.
            return_dict: True để trả toàn bộ kết quả dưới dạng Dict đầy đủ.
            use_sliding_window: None (tự động bật khi T > window_size), True (ép chạy sliding),
                                False (ép chạy 1 cửa sổ).
            window_sec: Độ dài cửa sổ tùy chỉnh (giây); mặc định dùng self.window_sec (60.0s).
            stride_sec: Bước trượt cửa sổ tùy chỉnh (giây); mặc định dùng self.stride_sec (30.0s).
            prev_state: Trạng thái tích lũy từ cửa sổ/đoạn video trước.
            return_state: True để trả kèm trạng thái tích lũy `state` ở cuối tuple.
        """
        if landmarks is None:
            raise ValueError("Tham số 'landmarks' [B, T, N, 2] bắt buộc phải được cung cấp!")

        lm_norm = self._normalize_landmarks_shape(landmarks)
        T_target = lm_norm.shape[1]

        win_frames = self.window_size if window_sec is None else max(1, int(round(float(window_sec) * self.fps)))
        str_frames = self.stride_size if stride_sec is None else max(1, int(round(float(stride_sec) * self.fps)))
        ov_frames = max(0, win_frames - str_frames)

        run_sliding = (T_target > win_frames) if use_sliding_window is None else bool(use_sliding_window)

        if run_sliding:
            out_dict = self.forward_sliding_windows(
                landmarks=lm_norm,
                video_frames=video_frames,
                lengths=lengths,
                mask=mask,
                window_size=win_frames,
                stride_size=str_frames,
                prev_state=prev_state,
            )
        else:
            d = None
            if video_frames is not None and video_frames.numel() > 0:
                d = self._encode_visual_context(video_frames, target_length=T_target)

            if mask is None and lengths is not None:
                mask = KinematicFeatureExtractor.build_temporal_mask(
                    lm_norm.permute(0, 3, 1, 2),
                    lengths=lengths,
                )

            X = self.fusion(lm_norm, d=d, mask=mask)
            win_out = self.forward_stgl_from_fused(
                X,
                lengths=lengths,
                prev_state=prev_state,
                overlap_size=ov_frames,
                stride_size=str_frames,
                return_details=True,
            )
            out_dict = {
                "frame_kss": win_out["frame_kss"],
                "clip_kss": win_out["clip_kss"],
                "frame_logits": win_out["frame_logits"],
                "clip_logits": win_out["clip_logits"],
                "window_kss": win_out["clip_kss"].unsqueeze(1),
                "window_logits": win_out["clip_logits"].unsqueeze(1),
                "state": win_out["state"],
            }

        if return_dict:
            return out_dict

        frame_kss = out_dict["frame_kss"]
        clip_kss = out_dict["clip_kss"]
        frame_logits = out_dict["frame_logits"]
        clip_logits = out_dict["clip_logits"]
        state = out_dict["state"]

        if return_logits:
            if return_clip_logits:
                base_out: Tuple[Any, ...] = (frame_kss, clip_kss, frame_logits, clip_logits)
            else:
                base_out = (frame_kss, frame_logits)
            return (*base_out, state) if return_state else base_out

        if return_clip_logits:
            base_out = (frame_kss, clip_kss)
            return (*base_out, state) if return_state else base_out

        return (frame_kss, state) if return_state else frame_kss

    @torch.no_grad()
    def forward_window_step(
            self,
            landmarks: torch.Tensor,
            video_frames: Optional[torch.Tensor] = None,
            lengths: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
            prev_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Suy luận cuốn chiếu (streaming) cho từng cửa sổ 60s mới (dịch bước 30s so với cửa sổ trước),
        tích lũy đặc trưng phía trước từ prev_state.

        Returns:
            clip_kss: Điểm KSS dự đoán [B] trong khoảng [1.0, 9.0].
            clip_probs: Phân phối xác suất 9 mức KSS [B, num_classes].
            new_state: Trạng thái tích lũy để truyền vào bước kế tiếp.
        """
        out = self.forward(
            video_frames=video_frames,
            landmarks=landmarks,
            lengths=lengths,
            mask=mask,
            return_dict=True,
            use_sliding_window=False,
            prev_state=prev_state,
        )
        clip_probs = F.softmax(out["clip_logits"].float(), dim=-1)
        return out["clip_kss"], clip_probs, out["state"]

    @torch.no_grad()
    def forward_asymmetric_step(
            self,
            landmark_window: torch.Tensor,
            d_history: torch.Tensor,
            current_frame: Optional[torch.Tensor] = None,
            prev_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if current_frame is not None:
            d_new = self.context_encoder(current_frame).unsqueeze(1)
        else:
            d_new = d_history[:, -1:, :]

        d_updated = torch.cat([d_history[:, 1:, :], d_new], dim=1)
        X = self.fusion(landmark_window, d=d_updated)
        win_out = self.forward_stgl_from_fused(
            X,
            prev_state=prev_state,
            return_details=True,
        )
        probs = F.softmax(win_out["clip_logits"].float(), dim=-1)
        return win_out["clip_kss"], probs, d_updated, win_out["state"]

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
