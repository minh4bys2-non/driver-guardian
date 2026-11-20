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


class MaskedBatchNorm2d(nn.Module):
    """
    BatchNorm2d có mặt nạ (Masked BatchNorm2d):
      - Khi training và có mask: chỉ tính mean/var và cập nhật running stats trên các vị trí hợp lệ (mask > 0.5),
        triệt tiêu hoàn toàn nhiễu từ các khung hình đệm (padding).
      - Khi eval hoặc không có mask: dùng running stats cố định và che lại đầu ra theo mask (bất biến batch_size & padding).
    """

    def __init__(
            self,
            num_features: int,
            eps: float = 1e-5,
            momentum: float = 0.01,
            affine: bool = True,
    ):
        super().__init__()
        self.bn = nn.BatchNorm2d(num_features, eps=eps, momentum=momentum, affine=affine)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if (not self.training) or (mask is None) or (not self.bn.track_running_stats):
            out = self.bn(x)
            if mask is not None:
                out = out * mask.to(dtype=out.dtype)
            return out

        B, C, H, W = x.shape
        dev_type = x.device.type if x.device.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(dev_type, enabled=False):
            x_32 = x.float()
            m_broad = (mask > 0.5).float().expand(B, 1, H, W)
            valid_count = m_broad.sum()

            if valid_count.item() < 2.0:
                out_32 = F.batch_norm(
                    x_32,
                    self.bn.running_mean,
                    self.bn.running_var,
                    self.bn.weight.float() if self.bn.affine else None,
                    self.bn.bias.float() if self.bn.affine else None,
                    training=False,
                    momentum=0.0,
                    eps=self.bn.eps,
                ) * m_broad
            else:
                mean = (x_32 * m_broad).sum(dim=(0, 2, 3)) / valid_count
                mean_4d = mean.view(1, C, 1, 1)
                var = (((x_32 - mean_4d) * m_broad).pow(2)).sum(dim=(0, 2, 3)) / valid_count

                with torch.no_grad():
                    mom = float(self.bn.momentum) if self.bn.momentum is not None else 0.1
                    self.bn.num_batches_tracked.add_(1)
                    var_unbiased = var * (valid_count / (valid_count - 1.0))
                    self.bn.running_mean.mul_(1.0 - mom).add_(mean.detach() * mom)
                    self.bn.running_var.mul_(1.0 - mom).add_(var_unbiased.detach() * mom)

                inv_std = torch.rsqrt(var.view(1, C, 1, 1) + self.bn.eps)
                out_32 = (x_32 - mean_4d) * inv_std
                if self.bn.affine:
                    out_32 = (
                        out_32 * self.bn.weight.float().view(1, C, 1, 1)
                        + self.bn.bias.float().view(1, C, 1, 1)
                    )
                out_32 = out_32 * m_broad

        return out_32.to(x.dtype)


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
            static_shape_ratio: float = 0.15,
            fps: float = 10.0,
            num_nodes: int = 68,
    ):
        super().__init__()
        self.nose_idx = nose_idx
        self.left_eye_idx = left_eye_idx
        self.right_eye_idx = right_eye_idx
        self.eps = eps
        self.static_shape_ratio = float(static_shape_ratio)
        self.fps = float(max(fps, 1.0))
        self.num_nodes = int(num_nodes)

        # Chuẩn hóa phân phối riêng cho 6 kênh [x, y, dx, dy, d2x, d2y] chỉ tính trên frame hợp lệ
        self.channel_norm = MaskedBatchNorm2d(6, momentum=0.01, affine=True)

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

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None,
            kin_state: Optional[Dict[str, torch.Tensor]] = None,
            return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Input : (B, 2, T, V=68) - Tọa độ [0, 1] trong hệ tọa độ khung ảnh.
        mask  : (B, 1, T, 1)    - Float mask (1.0 = valid, 0.0 = padded/missing).
        kin_state: Trạng thái chuẩn hóa nhân quả truyền từ cửa sổ trước (dùng cho streaming/mobile).
        Output: (B, 6, T, V=68) - [x_norm, y_norm, dx_norm, dy_norm, d2x_norm, d2y_norm].
        """
        if x.size(1) != 2:
            out_passthru = x * mask if mask is not None else x
            if return_state:
                return out_passthru, (kin_state if kin_state is not None else {})
            return out_passthru

        B, _, T, V = x.shape

        # 1. Khoảng cách giải phẫu 2 mắt (Inter-Ocular Distance - IOD giữa điểm 36 và 45)
        p_left_eye = x[:, :, :, self.left_eye_idx: self.left_eye_idx + 1]    # (B, 2, T, 1)
        p_right_eye = x[:, :, :, self.right_eye_idx: self.right_eye_idx + 1]  # (B, 2, T, 1)
        raw_iod = torch.norm(p_right_eye - p_left_eye, p=2, dim=1, keepdim=True).clamp(min=self.eps)  # (B, 1, T, 1)

        # Ổn định IOD nhân quả: lưu baseline_iod từ cửa sổ đầu và cập nhật EMA chậm theo luồng streaming
        if mask is not None:
            v_mask = mask.clamp(min=0.0, max=1.0)
            v_sum = v_mask.sum(dim=2, keepdim=True)
            has_valid = v_sum > 0.5
            cur_clip_iod = (raw_iod * v_mask).sum(dim=2, keepdim=True) / v_sum.clamp(min=1.0)
        else:
            has_valid = torch.ones((B, 1, 1, 1), dtype=torch.bool, device=x.device)
            cur_clip_iod = raw_iod.mean(dim=2, keepdim=True)
        cur_clip_iod = cur_clip_iod.clamp(min=self.eps)

        if kin_state is not None and "baseline_iod" in kin_state:
            ref_iod = kin_state["baseline_iod"].to(device=x.device, dtype=raw_iod.dtype).clamp(min=self.eps)
            next_iod = torch.where(has_valid, 0.90 * ref_iod + 0.10 * cur_clip_iod, ref_iod)
        else:
            ref_iod = cur_clip_iod
            next_iod = cur_clip_iod

        iod_scale = torch.maximum(0.5 * raw_iod + 0.5 * ref_iod, 0.65 * ref_iod).clamp(min=self.eps)

        # 2. Tọa độ chuẩn hóa theo Tâm Mũi (pos_norm) dùng cho biến dạng động lực học
        nose_pos = x[:, :, :, self.nose_idx: self.nose_idx + 1]  # (B, 2, T, 1)
        pos_norm = (x - nose_pos) / iod_scale                    # (B, 2, T, V)
        pos_norm = torch.clamp(pos_norm, min=-8.0, max=8.0)

        # 3. Chuẩn hóa hình thái đa cơ quan độc lập & tỷ lệ nội cơ quan (Intra-Organ Aspect Ratio):
        c_left_eye = 0.5 * (x[:, :, :, 36:37] + x[:, :, :, 39:40])
        c_right_eye = 0.5 * (x[:, :, :, 42:43] + x[:, :, :, 45:46])
        c_mouth = 0.5 * (x[:, :, :, 48:49] + x[:, :, :, 54:55])

        w_left_eye = torch.norm(x[:, :, :, 39:40] - x[:, :, :, 36:37], p=2, dim=1, keepdim=True)
        w_right_eye = torch.norm(x[:, :, :, 45:46] - x[:, :, :, 42:43], p=2, dim=1, keepdim=True)
        w_mouth = torch.norm(x[:, :, :, 54:55] - x[:, :, :, 48:49], p=2, dim=1, keepdim=True)

        # Biên độ mở mắt dọc chuẩn hóa theo phân vị cao P85 (Open-Eye Upper-Quantile Calibration)
        le_vert = 0.5 * (
            torch.abs(x[:, 1:2, :, 37:38] - x[:, 1:2, :, 41:42])
            + torch.abs(x[:, 1:2, :, 38:39] - x[:, 1:2, :, 40:41])
        )
        re_vert = 0.5 * (
            torch.abs(x[:, 1:2, :, 43:44] - x[:, 1:2, :, 47:48])
            + torch.abs(x[:, 1:2, :, 44:45] - x[:, 1:2, :, 46:47])
        )
        raw_eye_open = 0.5 * (le_vert + re_vert)  # (B, 1, T, 1)
        eye_open_2d = raw_eye_open.view(B, T)
        if mask is not None:
            valid_open_2d = eye_open_2d.masked_fill(mask[:, 0, :, 0] <= 0.5, float("nan"))
            cur_eye_p85 = torch.nanquantile(valid_open_2d, 0.85, dim=1)
            cur_eye_p85 = torch.where(torch.isnan(cur_eye_p85), 0.12 * ref_iod.view(B), cur_eye_p85)
        else:
            cur_eye_p85 = torch.quantile(eye_open_2d, 0.85, dim=1)
        cur_eye_p85 = torch.clamp(cur_eye_p85.view(B, 1, 1, 1), min=0.045 * ref_iod, max=0.45 * ref_iod)

        if kin_state is not None and "baseline_eye_open" in kin_state:
            ref_eye_open = kin_state["baseline_eye_open"].to(device=x.device, dtype=raw_eye_open.dtype)
            next_eye_open = torch.where(has_valid, 0.85 * ref_eye_open + 0.15 * cur_eye_p85, ref_eye_open)
            eff_eye_open = torch.maximum(0.75 * ref_eye_open + 0.25 * cur_eye_p85, 0.045 * ref_iod)
        else:
            eff_eye_open = cur_eye_p85
            next_eye_open = cur_eye_p85

        scale_left_eye = torch.maximum(w_left_eye, 0.22 * iod_scale).clamp(min=self.eps)
        scale_right_eye = torch.maximum(w_right_eye, 0.22 * iod_scale).clamp(min=self.eps)
        scale_mouth = torch.maximum(w_mouth, 0.32 * iod_scale).clamp(min=self.eps)
        eye_y_scale_l = torch.maximum(
            0.65 * eff_eye_open + 0.35 * (0.32 * scale_left_eye), 0.045 * iod_scale
        ).clamp(min=self.eps)
        eye_y_scale_r = torch.maximum(
            0.65 * eff_eye_open + 0.35 * (0.32 * scale_right_eye), 0.045 * iod_scale
        ).clamp(min=self.eps)

        pos_feat = pos_norm.clone()
        # Lông mày trái & phải (quy chiếu về tâm khóe mắt theo iod_scale)
        pos_feat[:, :, :, 17:22] = (x[:, :, :, 17:22] - c_left_eye) / iod_scale
        pos_feat[:, :, :, 22:27] = (x[:, :, :, 22:27] - c_right_eye) / iod_scale

        # Mắt trái (36..41): trục ngang theo iod_scale, trục dọc theo biên độ mở mắt P85 của tài xế
        rel_le = x[:, :, :, 36:42] - c_left_eye
        pos_feat[:, 0:1, :, 36:42] = rel_le[:, 0:1] / iod_scale
        pos_feat[:, 1:2, :, 36:42] = rel_le[:, 1:2] / eye_y_scale_l

        # Mắt phải (42..47): trục ngang theo iod_scale, trục dọc theo biên độ mở mắt P85 của tài xế
        rel_re = x[:, :, :, 42:48] - c_right_eye
        pos_feat[:, 0:1, :, 42:48] = rel_re[:, 0:1] / iod_scale
        pos_feat[:, 1:2, :, 42:48] = rel_re[:, 1:2] / eye_y_scale_r

        # Miệng (48..67): trục ngang theo iod_scale, trục dọc theo bề rộng miệng (bảo toàn MAR)
        rel_mo = x[:, :, :, 48:68] - c_mouth
        pos_feat[:, 0:1, :, 48:68] = rel_mo[:, 0:1] / iod_scale
        pos_feat[:, 1:2, :, 48:68] = rel_mo[:, 1:2] / (scale_mouth * 0.45)

        # Baseline nhân quả cho lông mày & miệng (giữ nguyên hệ quy chiếu P85 của mắt 36..47 để bảo toàn 100% tín hiệu PERCLOS)
        organ_indices = list(range(17, 27)) + list(range(48, 68))
        pos_org = pos_feat[:, :, :, organ_indices]
        if mask is not None:
            denom_org = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
            cur_org_mean = (pos_org * mask).sum(dim=2, keepdim=True) / denom_org
        else:
            cur_org_mean = pos_org.mean(dim=2, keepdim=True)

        jaw_nose_indices = list(range(0, 17)) + list(range(27, 36))
        pos_jn = pos_norm[:, :, :, jaw_nose_indices]
        if mask is not None:
            denom_jn = mask.sum(dim=2, keepdim=True).clamp(min=1.0)
            cur_jn_mean = (pos_jn * mask).sum(dim=2, keepdim=True) / denom_jn
        else:
            cur_jn_mean = pos_jn.mean(dim=2, keepdim=True)

        # Neo baseline nhân quả từ cửa sổ đầu tiên để bảo toàn độ sụp mi lũy tiến qua các cửa sổ
        if kin_state is not None and "baseline_org_mean" in kin_state and "baseline_jn_mean" in kin_state:
            prev_org = kin_state["baseline_org_mean"].to(device=x.device, dtype=pos_org.dtype)
            prev_jn = kin_state["baseline_jn_mean"].to(device=x.device, dtype=pos_jn.dtype)
            if prev_org.shape[-1] == pos_org.shape[-1]:
                ref_org_mean = 0.65 * prev_org + 0.35 * cur_org_mean
                next_org_mean = torch.where(has_valid, 0.85 * prev_org + 0.15 * cur_org_mean, prev_org)
            else:
                ref_org_mean = cur_org_mean
                next_org_mean = cur_org_mean
            ref_jn_mean = 0.50 * prev_jn + 0.50 * cur_jn_mean
            next_jn_mean = torch.where(has_valid, 0.85 * prev_jn + 0.15 * cur_jn_mean, prev_jn)
        else:
            ref_org_mean = cur_org_mean
            ref_jn_mean = cur_jn_mean
            next_org_mean = cur_org_mean
            next_jn_mean = cur_jn_mean

        if self.static_shape_ratio < 0.999:
            axis_ratio = torch.tensor(
                [0.05, self.static_shape_ratio], device=x.device, dtype=pos_org.dtype
            ).view(1, 2, 1, 1)
            pos_feat[:, :, :, organ_indices] = (pos_org - ref_org_mean) + axis_ratio * ref_org_mean

        pos_feat[:, :, :, jaw_nose_indices] = (pos_jn - ref_jn_mean) + 0.05 * ref_jn_mean
        # Khuếch đại nhẹ vùng mắt (36..47) và miệng (48..67)
        pos_feat[:, :, :, 36:48] = pos_feat[:, :, :, 36:48] * 1.8
        pos_feat[:, :, :, 48:68] = pos_feat[:, :, :, 48:68] * 1.4
        pos_feat = torch.clamp(pos_feat, min=-8.0, max=8.0)

        # 4. Tính Vận tốc (dx, dy) và Gia tốc (d2x, d2y) chuẩn hóa giải tích trên tọa độ quy tâm cơ quan
        if T <= 1:
            vel_norm = torch.zeros_like(pos_norm)
            acc_norm = torch.zeros_like(pos_norm)
        else:
            # Quy tâm cục bộ cho lông mày/mắt (khóe mắt) & miệng (khóe miệng) để tách chớp mắt/ngáp khỏi xoay đầu
            pos_vel_base = pos_norm.clone()
            pos_vel_base[:, :, :, 17:22] = (x[:, :, :, 17:22] - c_left_eye) / iod_scale
            pos_vel_base[:, :, :, 22:27] = (x[:, :, :, 22:27] - c_right_eye) / iod_scale
            pos_vel_base[:, :, :, 36:42] = rel_le / iod_scale
            pos_vel_base[:, :, :, 42:48] = rel_re / iod_scale
            pos_vel_base[:, :, :, 48:68] = rel_mo / iod_scale
            vel_local = torch.diff(pos_vel_base, n=1, dim=2)  # (B, 2, T-1, V)

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

        # 5. Chuẩn hóa năng lượng động học bảo toàn biên độ sinh lý (Soft-Ceiling Kinematic Normalization)
        # Chỉ nén mềm khi RMS vượt ngưỡng trần sinh lý; giữ nguyên tỷ lệ năng lượng chớp mắt/gật đầu bình thường
        if mask is not None:
            m_valid = mask.clamp(min=0.0, max=1.0)
            n_valid = m_valid.sum(dim=(2, 3), keepdim=True).clamp(min=1.0) * float(V)
            vel_rms = torch.sqrt((vel_norm.pow(2) * m_valid).sum(dim=(2, 3), keepdim=True) / n_valid + 1e-6)
            acc_rms = torch.sqrt((acc_norm.pow(2) * m_valid).sum(dim=(2, 3), keepdim=True) / n_valid + 1e-6)
        else:
            vel_rms = torch.sqrt(vel_norm.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-6)
            acc_rms = torch.sqrt(acc_norm.pow(2).mean(dim=(2, 3), keepdim=True) + 1e-6)

        # Hybrid Instance-Soft Kinematic Normalization: đưa tín hiệu về thang O(1) >> bn.eps
        dt_scale = 10.0 / self.fps
        vel_ref = 0.012 * dt_scale
        acc_ref = 0.009 * dt_scale
        vel_ceil = 2.5 * vel_ref
        acc_ceil = 2.5 * acc_ref
        vel_denom = vel_ref * torch.pow(torch.clamp(vel_rms / vel_ceil, min=1.0), 0.7)
        acc_denom = acc_ref * torch.pow(torch.clamp(acc_rms / acc_ceil, min=1.0), 0.7)
        vel_norm = torch.clamp(vel_norm / vel_denom, min=-6.0, max=6.0)
        acc_norm = torch.clamp(acc_norm / acc_denom, min=-6.0, max=6.0)

        out = torch.cat([pos_feat, vel_norm, acc_norm], dim=1)  # (B, 6, T, V)
        out = torch.nan_to_num(out, nan=0.0, posinf=4.0, neginf=-4.0)

        # 6. Chuẩn hóa kênh MaskedBatchNorm2d(6) chỉ tính trên các khung hình hợp lệ
        out = self.channel_norm(out, mask=mask)
        if return_state:
            new_kin_state = {
                "baseline_iod": next_iod.detach(),
                "baseline_org_mean": next_org_mean.detach(),
                "baseline_jn_mean": next_jn_mean.detach(),
                "baseline_eye_open": next_eye_open.detach(),
            }
            return out, new_kin_state
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
            static_shape_ratio: float = 0.15,
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
            num_nodes=num_nodes,
        )

        # Giữ lại w (đóng băng) để tương thích backward compatibility với checkpoint cũ
        self.w = nn.Parameter(torch.ones(num_features, 1) / float(num_features), requires_grad=False)

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
            kin_state: Optional[Dict[str, torch.Tensor]] = None,
            return_kin_state: bool = False,
            return_mask: bool = False,
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        """
        Args:
            coords: Chuỗi tọa độ mốc [B, T, N, 2] (hoặc [B, T, N, 3] từ cache cũ).
            d: Vector ngữ cảnh thị giác [B, T, D] hoặc None (Landmark-Only).
            mask: Temporal float mask [B, 1, T, 1].
            kin_state: Trạng thái baseline nhân quả truyền giữa các cửa sổ trượt.
            return_kin_state: True để trả kèm next_kin_state cho streaming/mobile.
            return_mask: True để trả kèm mặt nạ frame hợp lệ [B, T] (kết hợp lengths & ~is_missing).
        Returns:
            X: Tensor đặc trưng đồ thị [B, D, N, T] (hoặc tuple (X, next_kin_state[, frame_valid_mask])).
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
            feat_kin, next_kin_state = self.kinematic(
                x_in,
                mask=temporal_mask,
                kin_state=kin_state,
                return_state=True,
            )  # [B, 6, T, N]
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

        out_x = X.to(orig_dtype)
        frame_valid_mask = temporal_mask[:, 0, :, 0] > 0.5
        if return_kin_state and return_mask:
            return out_x, next_kin_state, frame_valid_mask
        if return_kin_state:
            return out_x, next_kin_state
        if return_mask:
            return out_x, frame_valid_mask
        return out_x


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
            use_residual_out: bool = True,
    ):
        super().__init__()
        self.use_residual_out = bool(use_residual_out)
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
        self.bn = MaskedBatchNorm2d(residual_channels, momentum=0.01)
        self.res_scale = nn.Parameter(torch.ones(1))
        if not self.use_residual_out:
            self.bn.bn.weight.requires_grad_(False)
            self.bn.bn.bias.requires_grad_(False)
            self.res_scale.requires_grad_(False)

    def forward(
            self,
            x: torch.Tensor,
            A_adp: torch.Tensor,
            t_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        residual = x
        h_tcn = self.gated_tcn(x)
        if t_mask is not None:
            h_tcn = h_tcn * t_mask.to(dtype=h_tcn.dtype)
        h_gcn = self.spatial_gcn(h_tcn, A_adp)
        if t_mask is not None:
            h_gcn = h_gcn * t_mask.to(dtype=h_gcn.dtype)
        skip = self.skip_conv(h_gcn)
        if self.use_residual_out:
            x_res = self.bn(self.res_scale * h_gcn + residual, mask=t_mask)
        else:
            x_res = residual
        if t_mask is not None:
            skip = skip * t_mask.to(dtype=skip.dtype)
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
            prev_tail_lengths: Optional[torch.Tensor] = None,
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

            valid_ov = None
            if cur_lengths is not None:
                valid_ov = (
                    torch.arange(L_ov, device=dev).unsqueeze(0)
                    < cur_lengths.to(dev).unsqueeze(1)
                ).unsqueeze(-1).float()
            if prev_tail_lengths is not None:
                valid_prev = (
                    torch.arange(L_ov, device=dev).unsqueeze(0)
                    < prev_tail_lengths.to(dev).unsqueeze(1)
                ).unsqueeze(-1).float()
                valid_ov = valid_prev if valid_ov is None else (valid_ov * valid_prev)
            if valid_ov is not None:
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
            p_32 = prev_memory.float() if prev_memory is not None else torch.zeros_like(w_32)
            z = self.input_norm(w_32)
            h_prev = self.state_norm(p_32) if prev_memory is not None else p_32
            h_gru = self.gru_cell(z, h_prev)
            alpha = self.mem_gate(torch.cat([z, h_prev, h_gru], dim=-1))
            h_new = self.out_norm(alpha * h_gru + (1.0 - alpha) * w_32)
            if prev_memory is not None and active_mask is not None:
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

            valid_w_counts = valid_bool.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            w_idx = torch.arange(W_num, device=dev, dtype=ws_32.dtype).unsqueeze(0)
            recency_bias = torch.where(
                valid_w_counts > 1.0,
                -0.2 + 0.4 * (w_idx / (valid_w_counts - 1.0).clamp(min=1.0)),
                torch.zeros((B, W_num), device=dev, dtype=ws_32.dtype),
            )
            scores = self.window_attn_gate(ws_32).squeeze(-1) + recency_bias
            scores = scores.masked_fill(~valid_bool, -1e4)
            attn_w = F.softmax(scores, dim=1) * valid_bool.float()
            attn_w = attn_w / attn_w.sum(dim=1, keepdim=True).clamp(min=1e-6)
            attn_state = (ws_32 * attn_w.unsqueeze(-1)).sum(dim=1)

            final_state = 0.45 * last_state + 0.55 * attn_state

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

            offset = 0.75 * torch.tanh(self.clip_reg_head(clip_feat.float()).squeeze(-1))
            kss_reg = torch.clamp(kss_exp + offset, min=self.kss_min, max=self.kss_max)
            mix_w = torch.sigmoid(self.kss_mix_logit.float())
            clip_kss = torch.clamp(mix_w * kss_exp + (1.0 - mix_w) * kss_reg, min=self.kss_min, max=self.kss_max)

        return clip_kss.to(orig_dtype), clip_logits


class GradientReversalFn(torch.autograd.Function):
    """Lớp đảo ngược gradient (Gradient Reversal Layer) cho Domain-Adversarial Training."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Module bọc GradientReversalFn hỗ trợ điều chỉnh hệ số lambda/alpha."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = float(alpha)

    def forward(self, x: torch.Tensor, alpha: Optional[float] = None) -> torch.Tensor:
        eff_alpha = self.alpha if alpha is None else float(alpha)
        return GradientReversalFn.apply(x, eff_alpha)


class RegimeConditionedDomainDiscriminator(nn.Module):
    """
    Bộ phân biệt miền có điều kiện theo 3 vùng trạng thái KSS (Alert / Mid / Drowsy):
      - Chỉ kích hoạt trong lúc huấn luyện (zero overhead khi suy luận trên mobile).
      - Ghép đặc trưng cửa sổ/clip sau GRL với phân phối 3 vùng p_regime (detached)
        để tránh co cụm các mức KSS trung gian khi phân phối nhãn giữa các domain lệch nhau.
    """

    def __init__(
            self,
            in_dim: int = 256,
            num_regimes: int = 3,
            hidden_dim: int = 64,
            num_domains: int = 2,
            dropout: float = 0.2,
    ):
        super().__init__()
        self.in_dim = int(in_dim)
        self.num_regimes = int(num_regimes)
        self.num_domains = int(num_domains)
        self.grl = GradientReversalLayer(alpha=0.0)
        self.net = nn.Sequential(
            nn.LayerNorm(self.in_dim + self.num_regimes),
            nn.Linear(self.in_dim + self.num_regimes, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_domains),
        )

    @staticmethod
    def logits_to_regime_probs(logits: torch.Tensor) -> torch.Tensor:
        """Quy đổi logits 9 mức KSS [1..9] sang xác suất 3 vùng: Alert (1-4), Mid (5-6), Drowsy (7-9)."""
        probs = F.softmax(logits.detach().float(), dim=-1)
        p_alert = probs[..., :4].sum(dim=-1, keepdim=True)
        p_mid = probs[..., 4:6].sum(dim=-1, keepdim=True)
        p_drowsy = probs[..., 6:].sum(dim=-1, keepdim=True)
        return torch.cat([p_alert, p_mid, p_drowsy], dim=-1)

    def forward(
            self,
            feat: torch.Tensor,
            kss_logits: torch.Tensor,
            alpha: Optional[float] = None,
    ) -> torch.Tensor:
        dev = feat.device
        dev_type = dev.type if dev.type in ("cuda", "cpu") else "cpu"
        with torch.amp.autocast(dev_type, enabled=False):
            feat_rev = self.grl(feat.float(), alpha=alpha)
            regime_probs = self.logits_to_regime_probs(kss_logits).to(dev)
            cond_in = torch.cat([feat_rev, regime_probs], dim=-1)
            domain_logits = self.net(cond_in)
        return domain_logits


class LiteFatV1(nn.Module):
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
            stride_sec: float = 45.0,
            num_blocks: int = 3,
            layers_per_block: int = 4,
            kernel_size: int = 3,
            pretrained_backbone: bool = True,
            dropout: float = 0.3,
            freeze_backbone: bool = False,
            vertical_pairs: Optional[List[Tuple[int, int]]] = None,
            cross_region_pairs: Optional[List[Tuple[int, int]]] = None,
            static_shape_ratio: float = 0.15,
            use_dann: bool = False,
            num_domains: int = 2,
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
        self.use_dann = bool(use_dann)
        self.num_domains = int(max(2, num_domains))
        self.dann_alpha = 0.0

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
        total_stgl_layers = num_blocks * layers_per_block
        layer_global_idx = 0
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
                        use_residual_out=(layer_global_idx < total_stgl_layers - 1),
                    )
                )
                dilation *= 2
                layer_global_idx += 1

        # Chiếu đặc trưng đầu ra đồ thị
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))

        # Attentive Node Pooling: Tự học trọng số chú ý trên đặc trưng 256D của 68 đỉnh khuôn mặt
        self.node_gate = nn.Sequential(
            nn.Conv2d(end_channels, 64, kernel_size=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
        )
        # Prior giải phẫu ưu tiên vùng mắt (36..47: 12 đỉnh) hơn vùng miệng (48..67: 20 đỉnh)
        node_prior = torch.zeros(1, 1, num_nodes, 1)
        node_prior[:, :, 36:48, :] = 1.50  # Mắt
        node_prior[:, :, 48:68, :] = 0.75  # Miệng
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

        # 7. Nhánh Domain-Adversarial Training có điều kiện theo 3 vùng trạng thái (chỉ chạy khi train)
        self.domain_discriminator = RegimeConditionedDomainDiscriminator(
            in_dim=end_channels,
            num_regimes=3,
            hidden_dim=64,
            num_domains=self.num_domains,
            dropout=min(0.3, dropout),
        )
        self.set_dann_enabled(self.use_dann)

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
            min_tail_frames: int = 50,
    ) -> List[Tuple[int, int]]:
        """
        Tạo danh sách các khoảng cửa sổ (start, end) cố định kích thước `window_size`
        (tương thích cơ chế Ring Buffer trên điện thoại di động) với bước trượt `stride_size`.
        """
        if total_frames <= window_size:
            return [(0, total_frames)]

        starts = list(range(0, max(1, total_frames - window_size + 1), stride_size))
        if not starts:
            starts = [0]

        rem = total_frames - (starts[-1] + window_size)
        if rem > 0:
            last_start = total_frames - window_size
            if rem >= int(max(1, min_tail_frames)):
                if last_start > starts[-1]:
                    starts.append(last_start)
            else:
                if len(starts) == 1 or last_start > starts[-2]:
                    starts[-1] = last_start

        return [(s, s + window_size) for s in starts]

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
            frame_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Chạy đồ thị STGL và Attentive Spatial Node Pooling cho 1 cửa sổ."""
        A_adp = self.adaptive_adj()
        h = self.start_conv(X)

        B, _, _, T = X.shape
        t_mask = None
        if frame_mask is not None:
            t_mask = frame_mask.to(device=X.device).float().view(B, 1, 1, T)
            if lengths is not None:
                t_idx = torch.arange(T, device=X.device).unsqueeze(0)
                len_mask = (t_idx < lengths.to(X.device).unsqueeze(1)).float().view(B, 1, 1, T)
                t_mask = t_mask * len_mask
        elif lengths is not None:
            t_idx = torch.arange(T, device=X.device).unsqueeze(0)  # [1, T]
            t_mask = (t_idx < lengths.to(X.device).unsqueeze(1)).float().view(B, 1, 1, T)

        skip_sum = None
        for block in self.stgl_blocks:
            h, skip = block(h, A_adp, t_mask=t_mask)
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
            frame_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gom tụ đặc trưng thời gian đa nhánh (Attn + Top-K + Mean) cho 1 cửa sổ."""
        B, T, _ = frame_feat.shape
        feat_norm = self.temp_norm(frame_feat)
        temp_scores = self.temporal_gate(feat_norm) * self.temp_scale  # [B, T, 1]

        if lengths is not None or frame_mask is not None:
            if lengths is not None:
                len_bool = (
                    torch.arange(T, device=frame_logits.device).unsqueeze(0)
                    < lengths.to(frame_logits.device).unsqueeze(1)
                ).unsqueeze(-1)
            else:
                len_bool = torch.ones((B, T, 1), dtype=torch.bool, device=frame_logits.device)

            if frame_mask is not None:
                fm_bool = frame_mask.to(frame_logits.device).bool().view(B, T, 1)
                raw_valid_bool = len_bool & fm_bool
                has_any_valid = raw_valid_bool.any(dim=1, keepdim=True)
                valid_bool = torch.where(has_any_valid, raw_valid_bool, len_bool)
            else:
                valid_bool = len_bool

            valid_mask = valid_bool.type_as(frame_logits)
            denom = valid_mask.sum(dim=1).clamp(min=1.0)

            masked_scores = temp_scores.masked_fill(~valid_bool, -1e4)
            temp_weights = F.softmax(masked_scores, dim=1) * valid_mask
            temp_weights = temp_weights / temp_weights.sum(dim=1, keepdim=True).clamp(min=1e-6)

            feat_mean = (frame_feat * valid_mask).sum(dim=1) / denom
            feat_attn = (frame_feat * temp_weights).sum(dim=1)

            scores_2d = masked_scores.squeeze(-1)  # [B, T]
            valid_counts = valid_bool.squeeze(-1).long().sum(dim=1)
            k_per = (valid_counts.float() * 0.25).round().clamp(min=1.0).long()  # [B]
            ranks = scores_2d.argsort(dim=1, descending=True).argsort(dim=1)  # [B, T]
            topk_mask = (ranks < k_per.unsqueeze(1)) & valid_bool.squeeze(-1)  # [B, T]
            feat_topk = (
                (frame_feat * topk_mask.unsqueeze(-1).type_as(frame_feat)).sum(dim=1)
                / topk_mask.sum(dim=1, keepdim=True).clamp(min=1.0).type_as(frame_feat)
            )

            pooled_window_feat = 0.50 * feat_attn + 0.30 * feat_topk + 0.20 * feat_mean
            attn_frame_logits = (frame_logits * temp_weights).sum(dim=1)

            active_col = (
                (lengths > 0) if lengths is not None else valid_bool.squeeze(-1).any(dim=1)
            ).to(frame_logits.device).view(B, 1)
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
            stride_size: Union[int, torch.Tensor],
            overlap_size: Union[int, torch.Tensor],
            prev_tail: Optional[torch.Tensor] = None,
            prev_tail_len: Optional[torch.Tensor] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Trích xuất vùng đặc trưng khung hình chồng lấp cho cửa sổ kế tiếp (hỗ trợ per-sample stride/overlap)."""
        B, T_w, D = frame_feat.shape
        dev = frame_feat.device

        if isinstance(stride_size, torch.Tensor) or isinstance(overlap_size, torch.Tensor):
            st_vec = (
                stride_size.to(dev).long()
                if isinstance(stride_size, torch.Tensor)
                else torch.full((B,), int(stride_size), dtype=torch.long, device=dev)
            )
            ov_vec = (
                overlap_size.to(dev).long()
                if isinstance(overlap_size, torch.Tensor)
                else torch.full((B,), int(overlap_size), dtype=torch.long, device=dev)
            )
            max_ov = int(ov_vec.max().item()) if ov_vec.numel() > 0 else 0
            if max_ov <= 0:
                return None, None

            overlap_tail = frame_feat.new_zeros((B, max_ov, D))
            tail_lens = torch.zeros((B,), dtype=torch.long, device=dev)
            for b in range(B):
                st_b = int(st_vec[b].item())
                ov_b = int(ov_vec[b].item())
                len_b = int(lengths[b].item()) if lengths is not None else T_w
                if len_b > 0 and ov_b > 0 and T_w > st_b:
                    end_b = min(T_w, len_b, st_b + ov_b)
                    if end_b > st_b:
                        n_cp = end_b - st_b
                        overlap_tail[b, :n_cp, :] = frame_feat[b, st_b:end_b, :]
                        tail_lens[b] = n_cp
                elif len_b <= 0 and prev_tail is not None:
                    copy_len = min(max_ov, prev_tail.size(1))
                    overlap_tail[b, :copy_len, :] = prev_tail[b, :copy_len, :].to(overlap_tail.dtype)
                    if prev_tail_len is not None:
                        tail_lens[b] = min(copy_len, int(prev_tail_len[b].item()))
                    else:
                        tail_lens[b] = copy_len
            return overlap_tail, tail_lens

        ov_int = int(overlap_size)
        st_int = int(stride_size)
        if ov_int <= 0:
            return None, None

        if T_w > st_int:
            raw_tail = frame_feat[:, st_int:min(T_w, st_int + ov_int), :]
            cur_len = raw_tail.size(1)
            if cur_len < ov_int:
                overlap_tail = F.pad(raw_tail, (0, 0, 0, ov_int - cur_len))
            else:
                overlap_tail = raw_tail

            if lengths is not None:
                valid_tail_len = (lengths.to(dev) - st_int).clamp(min=0, max=ov_int)
                tail_mask = (
                    torch.arange(ov_int, device=dev).unsqueeze(0)
                    < valid_tail_len.unsqueeze(1)
                ).unsqueeze(-1).type_as(overlap_tail)
                overlap_tail = overlap_tail * tail_mask
                tail_lens = valid_tail_len
            else:
                tail_lens = torch.full((B,), min(cur_len, ov_int), dtype=torch.long, device=dev)
        else:
            overlap_tail = frame_feat.new_zeros((B, ov_int, D))
            tail_lens = torch.zeros((B,), dtype=torch.long, device=dev)

        if prev_tail is not None and lengths is not None and prev_tail.size(1) == ov_int:
            active_col = (lengths.to(dev) > 0).view(B, 1, 1)
            overlap_tail = torch.where(active_col, overlap_tail, prev_tail.to(overlap_tail.dtype))
            if prev_tail_len is not None:
                tail_lens = torch.where(lengths.to(dev) > 0, tail_lens, prev_tail_len.to(dev))

        return overlap_tail, tail_lens

    def forward_stgl_from_fused(
            self,
            X: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
            prev_state: Optional[Dict[str, torch.Tensor]] = None,
            overlap_size: Optional[Union[int, torch.Tensor]] = None,
            stride_size: Optional[Union[int, torch.Tensor]] = None,
            kin_state: Optional[Dict[str, torch.Tensor]] = None,
            frame_mask: Optional[torch.Tensor] = None,
            return_details: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, Any]]:
        """
        Chạy phần đồ thị STGL từ tensor đặc trưng X [B, D, N, T_w] cho 1 cửa sổ,
        hỗ trợ tích lũy kép từ prev_state (vùng chồng lấp khung hình + bộ nhớ Gated GRU).
        """
        if stride_size is None:
            eff_stride: Union[int, torch.Tensor] = self.stride_size
        elif isinstance(stride_size, torch.Tensor):
            eff_stride = stride_size.clamp(min=1)
        else:
            eff_stride = int(max(1, stride_size))

        if overlap_size is None:
            eff_overlap: Union[int, torch.Tensor] = self.overlap_size
        elif isinstance(overlap_size, torch.Tensor):
            eff_overlap = overlap_size.clamp(min=0)
        else:
            eff_overlap = int(max(0, overlap_size))

        pooled_node_feat, frame_feat = self._extract_window_frame_features(
            X, lengths=lengths, frame_mask=frame_mask
        )

        prev_tail = prev_state.get("overlap_tail_feat") if prev_state is not None else None
        prev_tail_len = prev_state.get("overlap_tail_len") if prev_state is not None else None
        if prev_tail is not None:
            frame_feat = self.accumulator.fuse_overlap_frames(
                frame_feat,
                prev_tail,
                cur_lengths=lengths,
                prev_tail_lengths=prev_tail_len,
            )
            pooled_node_feat = frame_feat.permute(0, 2, 1).contiguous()

        frame_kss, frame_logits = self.kss_head.forward_frames(pooled_node_feat, frame_feat)
        pooled_window_feat, attn_frame_logits, _ = self._pool_window_temporal_features(
            frame_feat,
            frame_logits,
            lengths=lengths,
            frame_mask=frame_mask,
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

        win_domain_logits = None
        if self.use_dann and self.training:
            win_domain_logits = self.domain_discriminator(
                memory_feat,
                clip_logits,
                alpha=self.dann_alpha,
            )

        if return_details:
            overlap_tail_feat, overlap_tail_len = self._extract_overlap_tail(
                frame_feat=frame_feat,
                lengths=lengths,
                stride_size=eff_stride,
                overlap_size=eff_overlap,
                prev_tail=prev_tail,
                prev_tail_len=prev_tail_len,
            )
            new_state: Dict[str, Any] = {
                "memory_feat": memory_feat,
                "overlap_tail_feat": overlap_tail_feat,
                "overlap_tail_len": overlap_tail_len,
                "window_feat": pooled_window_feat,
            }
            if kin_state is not None:
                new_state["kin_state"] = kin_state
            elif prev_state is not None and "kin_state" in prev_state:
                new_state["kin_state"] = prev_state["kin_state"]
            return {
                "frame_kss": frame_kss,
                "clip_kss": clip_kss,
                "frame_logits": frame_logits,
                "clip_logits": clip_logits,
                "attn_frame_logits": attn_frame_logits,
                "domain_logits": win_domain_logits,
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
        Chạy cơ chế cửa sổ trượt (mặc định 60s = 600 frames, stride 45s = 450 frames @ 10 FPS)
        riêng theo độ dài thực của từng mẫu trong batch (bất biến tuyệt đối với batch padding).
        """
        lm_norm = self._normalize_landmarks_shape(landmarks)
        B, T, N_lm, C_lm = lm_norm.shape
        dev = lm_norm.device

        win_size = self.window_size if window_size is None else int(max(1, window_size))
        str_size = self.stride_size if stride_size is None else int(max(1, stride_size))

        d_full = None
        if video_frames is not None and video_frames.numel() > 0:
            d_full = self._encode_visual_context(video_frames, target_length=T)

        if mask is None and lengths is not None:
            mask = KinematicFeatureExtractor.build_temporal_mask(
                lm_norm.permute(0, 3, 1, 2),
                lengths=lengths,
            )

        # Xây dựng danh sách cửa sổ trượt riêng cho từng mẫu theo độ dài thực L_b
        if lengths is not None:
            sample_lens = [max(0, min(T, int(lengths[b].item()))) for b in range(B)]
            per_sample_ranges: List[List[Tuple[int, int]]] = [
                self._build_sliding_window_ranges(L_b, win_size, str_size) if L_b > 0 else []
                for L_b in sample_lens
            ]
        else:
            shared_ranges = self._build_sliding_window_ranges(T, win_size, str_size)
            per_sample_ranges = [shared_ranges for _ in range(B)]

        max_windows = max(1, max((len(r) for r in per_sample_ranges), default=1))

        full_frame_kss = lm_norm.new_full((B, T), self.kss_min, dtype=torch.float32)
        full_frame_logits = lm_norm.new_zeros((B, T, self.num_classes), dtype=torch.float32)
        full_frame_valid_mask = torch.zeros((B, T), dtype=torch.bool, device=dev)
        written_mask = torch.zeros((B, T), dtype=torch.bool, device=dev)

        state = prev_state
        kin_state = prev_state.get("kin_state") if prev_state is not None else None
        window_memories: List[torch.Tensor] = []
        window_kss_list: List[torch.Tensor] = []
        window_logits_list: List[torch.Tensor] = []
        window_attn_logits_list: List[torch.Tensor] = []
        window_valid_list: List[torch.Tensor] = []
        window_domain_logits_list: List[torch.Tensor] = []

        for w_idx in range(max_windows):
            win_spans: List[Tuple[int, int, bool]] = []
            sub_len_span = 1
            for b in range(B):
                if w_idx < len(per_sample_ranges[b]):
                    s_b, e_b = per_sample_ranges[b][w_idx]
                    win_spans.append((s_b, e_b, True))
                    sub_len_span = max(sub_len_span, e_b - s_b)
                else:
                    win_spans.append((0, 0, False))

            active_mask = torch.tensor([sp[2] for sp in win_spans], dtype=torch.bool, device=dev)
            sub_len_list = [(sp[1] - sp[0]) if sp[2] else 0 for sp in win_spans]
            sub_lengths = torch.tensor(sub_len_list, dtype=torch.long, device=dev)

            uniform_slice = (
                all(sp[2] for sp in win_spans)
                and len({(sp[0], sp[1]) for sp in win_spans}) == 1
            )
            if uniform_slice:
                s0, e0, _ = win_spans[0]
                sub_landmarks = lm_norm[:, s0:e0]
                sub_d = d_full[:, s0:e0] if d_full is not None else None
                sub_mask = (
                    mask[:, :, s0:e0, :]
                    if mask is not None
                    else KinematicFeatureExtractor.build_temporal_mask(
                        sub_landmarks.permute(0, 3, 1, 2),
                        lengths=sub_lengths,
                    )
                )
            else:
                sub_landmarks = lm_norm.new_ones((B, sub_len_span, N_lm, C_lm))
                sub_d = d_full.new_zeros((B, sub_len_span, d_full.size(-1))) if d_full is not None else None
                sub_mask = lm_norm.new_zeros((B, 1, sub_len_span, 1), dtype=torch.float32)
                for b, (s_b, e_b, is_act) in enumerate(win_spans):
                    if not is_act:
                        continue
                    len_b = e_b - s_b
                    sub_landmarks[b, :len_b] = lm_norm[b, s_b:e_b]
                    if d_full is not None and sub_d is not None:
                        sub_d[b, :len_b] = d_full[b, s_b:e_b]
                    if mask is not None:
                        sub_mask[b, :, :len_b, :] = mask[b, :, s_b:e_b, :]
                    else:
                        sub_mask[b, :, :len_b, :] = 1.0

            prev_kin_state = kin_state
            X_sub, next_kin_state, sub_frame_mask = self.fusion(
                sub_landmarks,
                d=sub_d,
                mask=sub_mask,
                kin_state=prev_kin_state,
                return_kin_state=True,
                return_mask=True,
            )
            if prev_kin_state is not None and not bool(active_mask.all().item()):
                for k_kin, v_new in list(next_kin_state.items()):
                    if k_kin in prev_kin_state and prev_kin_state[k_kin].shape == v_new.shape:
                        act_view = active_mask.view(B, *([1] * (v_new.dim() - 1)))
                        next_kin_state[k_kin] = torch.where(act_view, v_new, prev_kin_state[k_kin])
            kin_state = next_kin_state

            tail_strides: List[int] = []
            tail_overlaps: List[int] = []
            for b, (s_b, e_b, is_act) in enumerate(win_spans):
                if is_act and (w_idx + 1 < len(per_sample_ranges[b])):
                    next_s_b = per_sample_ranges[b][w_idx + 1][0]
                    tail_strides.append(max(1, next_s_b - s_b))
                    tail_overlaps.append(max(0, e_b - next_s_b))
                else:
                    tail_strides.append(str_size)
                    tail_overlaps.append(0)

            stride_arg: Union[int, torch.Tensor] = (
                tail_strides[0]
                if len(set(tail_strides)) == 1
                else torch.tensor(tail_strides, dtype=torch.long, device=dev)
            )
            overlap_arg: Union[int, torch.Tensor] = (
                tail_overlaps[0]
                if len(set(tail_overlaps)) == 1
                else torch.tensor(tail_overlaps, dtype=torch.long, device=dev)
            )

            prev_state_cur = state
            win_out = self.forward_stgl_from_fused(
                X_sub,
                lengths=sub_lengths,
                prev_state=prev_state_cur,
                overlap_size=overlap_arg,
                stride_size=stride_arg,
                kin_state=kin_state,
                frame_mask=sub_frame_mask,
                return_details=True,
            )
            state = win_out["state"]
            if prev_state_cur is not None and not bool(active_mask.all().item()):
                if "window_feat" in prev_state_cur and prev_state_cur["window_feat"] is not None:
                    state["window_feat"] = torch.where(
                        active_mask.view(B, 1),
                        state["window_feat"],
                        prev_state_cur["window_feat"],
                    )

            w_frame_kss = win_out["frame_kss"].float()
            w_frame_logits = win_out["frame_logits"].float()

            next_frame_kss = full_frame_kss.clone()
            next_frame_logits = full_frame_logits.clone()
            next_valid_mask = full_frame_valid_mask.clone()

            for b, (s_b, e_b, is_act) in enumerate(win_spans):
                if not is_act:
                    continue
                len_b = e_b - s_b
                w_fkss_b = w_frame_kss[b, :len_b]
                w_flogits_b = w_frame_logits[b, :len_b, :]
                w_fmask_b = sub_frame_mask[b, :len_b]

                if w_idx == 0:
                    if s_b > 0:
                        next_frame_kss[b, :s_b] = w_fkss_b[:1]
                        next_frame_logits[b, :s_b, :] = w_flogits_b[:1, :]
                        next_valid_mask[b, :s_b] = w_fmask_b[:1]
                        written_mask[b, :s_b] = True
                    next_frame_kss[b, s_b:e_b] = w_fkss_b
                    next_frame_logits[b, s_b:e_b, :] = w_flogits_b
                    next_valid_mask[b, s_b:e_b] = w_fmask_b
                    written_mask[b, s_b:e_b] = True
                else:
                    ov_slice = written_mask[b, s_b:e_b].clone()
                    if ov_slice.any():
                        ov_idx = torch.nonzero(ov_slice, as_tuple=False).squeeze(-1)
                        blend_w = torch.ones((len_b,), dtype=torch.float32, device=dev)
                        blend_w[ov_idx] = torch.linspace(0.25, 1.0, ov_idx.numel(), device=dev, dtype=torch.float32)
                        old_kss_b = full_frame_kss[b, s_b:e_b]
                        old_logits_b = full_frame_logits[b, s_b:e_b, :]
                        next_frame_kss[b, s_b:e_b] = torch.where(
                            ov_slice,
                            (1.0 - blend_w) * old_kss_b + blend_w * w_fkss_b,
                            w_fkss_b,
                        )
                        blend_w_3d = blend_w.unsqueeze(-1)
                        next_frame_logits[b, s_b:e_b, :] = torch.where(
                            ov_slice.unsqueeze(-1),
                            (1.0 - blend_w_3d) * old_logits_b + blend_w_3d * w_flogits_b,
                            w_flogits_b,
                        )
                        next_valid_mask[b, s_b:e_b] = full_frame_valid_mask[b, s_b:e_b] | w_fmask_b
                    else:
                        next_frame_kss[b, s_b:e_b] = w_fkss_b
                        next_frame_logits[b, s_b:e_b, :] = w_flogits_b
                        next_valid_mask[b, s_b:e_b] = w_fmask_b
                    written_mask[b, s_b:e_b] = True

            full_frame_kss = next_frame_kss
            full_frame_logits = next_frame_logits
            full_frame_valid_mask = next_valid_mask

            window_memories.append(state["memory_feat"])
            window_kss_list.append(win_out["clip_kss"])
            window_logits_list.append(win_out["clip_logits"])
            window_attn_logits_list.append(win_out["attn_frame_logits"])
            window_valid_list.append(active_mask)
            if win_out.get("domain_logits") is not None:
                window_domain_logits_list.append(win_out["domain_logits"])

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

        clip_domain_logits = None
        window_domain_logits = None
        if self.use_dann and self.training:
            clip_domain_logits = self.domain_discriminator(
                final_feat,
                clip_logits,
                alpha=self.dann_alpha,
            )
            if window_domain_logits_list:
                window_domain_logits = torch.stack(window_domain_logits_list, dim=1)

        orig_dtype = lm_norm.dtype
        return {
            "frame_kss": full_frame_kss.to(orig_dtype),
            "clip_kss": clip_kss,
            "frame_logits": full_frame_logits.to(orig_dtype),
            "clip_logits": clip_logits,
            "window_kss": window_kss,
            "window_logits": window_logits,
            "window_valid_mask": window_valid_mask,
            "frame_valid_mask": full_frame_valid_mask,
            "domain_logits": clip_domain_logits,
            "window_domain_logits": window_domain_logits,
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

            in_kin_state = prev_state.get("kin_state") if prev_state is not None else None
            X, next_kin_state, frame_valid_mask = self.fusion(
                lm_norm,
                d=d,
                mask=mask,
                kin_state=in_kin_state,
                return_kin_state=True,
                return_mask=True,
            )
            win_out = self.forward_stgl_from_fused(
                X,
                lengths=lengths,
                prev_state=prev_state,
                overlap_size=ov_frames,
                stride_size=str_frames,
                kin_state=next_kin_state,
                frame_mask=frame_valid_mask,
                return_details=True,
            )
            out_dict = {
                "frame_kss": win_out["frame_kss"],
                "clip_kss": win_out["clip_kss"],
                "frame_logits": win_out["frame_logits"],
                "clip_logits": win_out["clip_logits"],
                "window_kss": win_out["clip_kss"].unsqueeze(1),
                "window_logits": win_out["clip_logits"].unsqueeze(1),
                "window_valid_mask": (
                    (lengths.to(lm_norm.device) > 0).unsqueeze(1)
                    if lengths is not None
                    else torch.ones((lm_norm.size(0), 1), dtype=torch.bool, device=lm_norm.device)
                ),
                "frame_valid_mask": frame_valid_mask,
                "domain_logits": win_out.get("domain_logits"),
                "window_domain_logits": (
                    win_out["domain_logits"].unsqueeze(1)
                    if win_out.get("domain_logits") is not None
                    else None
                ),
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

    def set_visual_context_enabled(self, enabled: bool) -> None:
        """Đóng băng hoặc mở khóa toàn bộ nhánh thị giác (MobileNetV3 + Projector + FiLM)."""
        flag = bool(enabled)
        if not flag:
            self.context_encoder.freeze_backbone(freeze=True, freeze_projector=True)
        else:
            for param in self.context_encoder.projector.parameters():
                param.requires_grad = True
        for param in self.fusion.film_scale.parameters():
            param.requires_grad = flag
        for param in self.fusion.film_shift.parameters():
            param.requires_grad = flag

    def set_dann_enabled(self, enabled: bool) -> None:
        """Bật/tắt nhánh phân biệt miền có điều kiện (Regime-Conditioned DANN)."""
        self.use_dann = bool(enabled)
        for param in self.domain_discriminator.parameters():
            param.requires_grad = self.use_dann

    def set_dann_alpha(self, alpha: float) -> None:
        """Cập nhật hệ số đảo dấu gradient GRL (lambda) theo lịch trình Ganin."""
        self.dann_alpha = float(max(0.0, alpha))

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


LiteFat_v1 = LiteFatV1
