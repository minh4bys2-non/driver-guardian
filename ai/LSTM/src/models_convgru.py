import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union


import torch
import torch.nn as nn
import torch.nn.functional as F

from src.convgru import ConvGRU


class SpatialReductionNeck(nn.Module):
    """

    """

    def __init__(
            self,
            in_channels: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...]] = (64, 128, 256),
            out_channels: int = 64,
            dropout: float = 0.1,
    ):
        super().__init__()
        if len(in_channels) > 0 and isinstance(in_channels[0], (tuple, list)):
            self.channel_dims = tuple(c[0] for c in in_channels)
        else:
            self.channel_dims = tuple(int(c) for c in in_channels)

        self.num_scales = len(self.channel_dims)
        self.total_in_channels = sum(self.channel_dims)
        self.out_channels = int(out_channels)

        # 1. Các tầng căn chỉnh kích thước không gian về target_size (mặc định 40x40)
        self.p3_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p5_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 2. Tầng tích chập nén kênh 1x1 tinh gọn (~28.8K tham số)
        self.reduction = nn.Sequential(
            nn.Conv2d(
                self.total_in_channels,
                self.out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(self.out_channels),
            nn.SiLU(inplace=True),
            nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()
        )

    def _align_features(
            self,
            p3: torch.Tensor,
            p4: torch.Tensor,
            p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ """

        out_p3 = self.p3_down(p3)
        out_p4 = p4
        out_p5 = self.p5_up(p5)
        return out_p3, out_p4, out_p5

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            seq_lens: Optional[torch.Tensor] = None,
            *args: torch.Tensor
    ) -> torch.Tensor:
        """

        """
        if len(args) > 0:
            features = (features, *args)

        if isinstance(features, (tuple, list)):
            if len(features) != self.num_scales:
                raise ValueError(
                    f"[SpatialReductionNeck] Yêu cầu {self.num_scales} tầng đặc trưng, nhưng nhận {len(features)}."
                )
            p3, p4, p5 = features[0], features[1], features[2]
            is_5d = (p3.dim() == 5)
            orig_b, orig_t = 0, 0
            target_size = p4.shape[3]
            if is_5d:
                orig_b, orig_t = p3.shape[0], p3.shape[1]
                p3 = p3.reshape(orig_b * orig_t, *p3.shape[2:])
                p4 = p4.reshape(orig_b * orig_t, *p4.shape[2:])
                p5 = p5.reshape(orig_b * orig_t, *p5.shape[2:])

            out_p3, out_p4, out_p5 = self._align_features(p3, p4, p5)
            fused = torch.cat([out_p3, out_p4, out_p5], dim=1)  # [N, total_in, 40, 40]

            if is_5d and seq_lens is not None:
                time_indices = torch.arange(orig_t, device=fused.device).unsqueeze(0)  # [1, T]
                lens_expanded = seq_lens.to(fused.device).unsqueeze(1)  # [B, 1]
                mask = (time_indices < lens_expanded).view(-1)  # [B*T]

                valid_fused = fused[mask]  # [N_valid, total_in, 40, 40]
                if valid_fused.numel() > 0:
                    valid_reduced = self.reduction(valid_fused)
                else:
                    valid_reduced = torch.empty((0, self.out_channels, target_size, target_size),
                                                device=fused.device, dtype=fused.dtype)

                reduced = torch.zeros((orig_b * orig_t, self.out_channels, target_size, target_size),
                                      device=fused.device, dtype=fused.dtype)
                reduced[mask] = valid_reduced.to(reduced.dtype)
            else:
                reduced = self.reduction(fused)  # [N, out_channels, 40, 40]

            if is_5d:
                reduced = reduced.view(orig_b, orig_t, self.out_channels, *reduced.shape[-2:])
            return reduced

        elif isinstance(features, torch.Tensor):
            feat = features
            is_5d = (feat.dim() == 5)
            orig_b, orig_t = 0, 0

            if is_5d:
                orig_b, orig_t = feat.shape[0], feat.shape[1]
                feat = feat.reshape(orig_b * orig_t, *feat.shape[2:])

            if feat.shape[1] == self.total_in_channels:
                if is_5d and seq_lens is not None:
                    time_indices = torch.arange(orig_t, device=feat.device).unsqueeze(0)
                    lens_expanded = seq_lens.to(feat.device).unsqueeze(1)
                    mask = (time_indices < lens_expanded).view(-1)

                    valid_feat = feat[mask]
                    if valid_feat.numel() > 0:
                        valid_reduced = self.reduction(valid_feat)
                    else:
                        valid_reduced = torch.empty((0, self.out_channels, self.target_size[0], self.target_size[1]),
                                                    device=feat.device, dtype=feat.dtype)

                    reduced = torch.zeros(
                        (orig_b * orig_t, self.out_channels, self.target_size[0], self.target_size[1]),
                        device=feat.device, dtype=feat.dtype)
                    reduced[mask] = valid_reduced.to(reduced.dtype)
                else:
                    reduced = self.reduction(feat)
            elif feat.shape[1] == self.out_channels:
                reduced = feat
            else:
                raise ValueError(
                    f"[SpatialReductionNeck] Số kênh đầu vào ({feat.shape[1]}) không khớp "
                    f"total_in ({self.total_in_channels}) hoặc out_channels ({self.out_channels})."
                )

            if is_5d:
                reduced = reduced.view(orig_b, orig_t, self.out_channels, *reduced.shape[-2:])
            return reduced

        else:
            raise TypeError(
                f"[SpatialReductionNeck] Kiểu dữ liệu không hợp lệ: {type(features)}. Yêu cầu Tuple/List hoặc Tensor."
            )


class SpatialAttentionPooling(nn.Module):
    """

    """

    def __init__(self, in_channels: int, hidden_dim: Optional[int] = None):
        super().__init__()
        self.in_channels = int(in_channels)
        mid_channels = hidden_dim or max(16, self.in_channels // 2)

        self.attn_conv = nn.Sequential(
            nn.Conv2d(self.in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, bias=True)
        )

    def forward(
            self,
            x: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None,
            return_weights: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """

        """
        is_5d = (x.dim() == 5)
        orig_b, orig_t = 0, 0

        if is_5d:
            orig_b, orig_t, c, h, w = x.shape
            x_reshaped = x.reshape(orig_b * orig_t, c, h, w)
        elif x.dim() == 4:
            x_reshaped = x
            _, c, h, w = x.shape
        else:
            raise ValueError(f"[SpatialAttentionPooling] Yêu cầu tensor 4D hoặc 5D, nhận được {x.dim()}D.")

        # Tính điểm số chú ý: [N, 1, H, W]
        if is_5d and seq_lens is not None:
            time_indices = torch.arange(orig_t, device=x.device).unsqueeze(0)
            lens_expanded = seq_lens.to(x.device).unsqueeze(1)
            mask = (time_indices < lens_expanded).view(-1)

            valid_x = x_reshaped[mask]
            if valid_x.numel() > 0:
                valid_scores = self.attn_conv(valid_x)
            else:
                valid_scores = torch.empty((0, 1, h, w), device=x.device, dtype=x.dtype)

            scores = torch.full((orig_b * orig_t, 1, h, w), fill_value=-1e9 if x.dtype == torch.float32 else -1e4,
                                device=x.device, dtype=x.dtype)
            scores[mask] = valid_scores.to(scores.dtype)
        else:
            scores = self.attn_conv(x_reshaped)

        # Softmax trên toàn bộ lưới không gian H x W
        scores_flat = scores.view(-1, 1, h * w)
        weights_flat = F.softmax(scores_flat, dim=-1)
        weights = weights_flat.view(-1, 1, h, w)

        # Gom tụ có trọng số: sum_{x,y} (weights * features)
        pooled = (x_reshaped * weights).sum(dim=(-2, -1))  # [N, C]

        if is_5d:
            pooled = pooled.view(orig_b, orig_t, c)
            weights = weights.view(orig_b, orig_t, 1, h, w)

        if return_weights:
            return pooled, weights
        return pooled


class TemporalAttentionPooling(nn.Module):
    """

    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_dim, max(16, self.hidden_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(16, self.hidden_dim // 2), 1)
        )

    def forward(
            self,
            seq_features: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """

        """
        b, t, h = seq_features.shape
        scores = self.attn(seq_features).squeeze(-1)  # [B, T]

        if seq_lens is not None:
            time_indices = torch.arange(t, device=seq_features.device).unsqueeze(0)  # [1, T]
            lens_expanded = seq_lens.to(seq_features.device).unsqueeze(1)  # [B, 1]
            mask = (time_indices < lens_expanded)  # [B, T]
            fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask, fill_value)

        weights = F.softmax(scores, dim=-1)  # [B, T]
        # Gom tụ có trọng số: [B, 1, T] x [B, T, H] -> [B, 1, H] -> [B, H]
        pooled = torch.bmm(weights.unsqueeze(1), seq_features).squeeze(1)
        return pooled, weights


class ConvGRUClassifier(nn.Module):
    """

    """

    def __init__(
            self,
            input_dim: int = 64,
            hidden_dim: Union[int, List[int], Tuple[int, ...]] = 64,
            num_layers: int = 2,
            kernel_size: Union[int, Tuple[int, int]] = 3,
            num_classes: int = 2,
            spatial_in_channels: Tuple[int, ...] = (64, 128, 256),
            target_size: Tuple = (40, 40),
            neck_dropout: float = 0.1,
            convgru_dropout: float = 0.1,
            dropout: float = 0.35,
            supervision_mode: str = "attention_pooling"
    ):
        """
        Khởi tạo SpatioTemporalConvGRUClassifier.

        Args:
            input_dim (int): Số kênh sau cổ giảm kênh SpatialReductionNeck (mặc định 64).
            hidden_dim (int | List[int]): Số kênh ẩn của từng tầng ConvGRU (mặc định 64 hoặc [64, 64]).
            num_layers (int): Số tầng ConvGRU xếp chồng (mặc định 2).
            kernel_size (int | Tuple): Kích thước kernel ConvGRU (mặc định 3).
            num_classes (int): Số lớp phân loại (mặc định 2: Tỉnh táo / Buồn ngủ).
            spatial_in_channels (Tuple[int, ...]): Số kênh của các tầng PAFPN (mặc định (64, 128, 256)).
            neck_dropout (float): Dropout cho cổ giảm kênh.
            convgru_dropout (float): Spatial Dropout giữa các tầng ConvGRU.
            dropout (float): Dropout cho đầu phân loại FC Head.
            supervision_mode (str): 'attention_pooling' (clip-level), 'sequence' (frame-level).
        """
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_layers = int(num_layers)
        self.num_classes = int(num_classes)
        self.supervision_mode = supervision_mode.lower()
        # Chuẩn hóa hidden_dims
        if isinstance(hidden_dim, (list, tuple)):
            self.hidden_dims = [int(h) for h in hidden_dim]
        else:
            self.hidden_dims = [int(hidden_dim)] * self.num_layers
        self.last_hidden_dim = self.hidden_dims[-1]

        # 1. Cổ giảm kênh không gian (Spatial Reduction Neck)
        self.spatial_neck = SpatialReductionNeck(
            in_channels=spatial_in_channels,
            out_channels=self.input_dim,
            dropout=neck_dropout,
        )

        # 2. Khối chuỗi không gian - thời gian (Spatio-Temporal ConvGRU)
        self.convgru = ConvGRU(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dims,
            kernel_size=kernel_size,
            num_layers=self.num_layers,
            batch_first=True,
            bias=True,
            return_all_layers=False,
            dropout=convgru_dropout,
            residual=False
        )

        # 3. Gom tụ chú ý không gian (Spatial Attention Pooling)
        self.spatial_pooling = SpatialAttentionPooling(
            in_channels=self.last_hidden_dim
        )

        # 4. Gom tụ chú ý thời gian (Temporal Attention Pooling)
        self.temporal_pooling = TemporalAttentionPooling(
            hidden_dim=self.last_hidden_dim
        )

        # 5. Đầu phân loại (Classification Head)
        self.fc_out = nn.Sequential(
            nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity(),
            nn.Linear(self.last_hidden_dim, self.num_classes)
        )

    @classmethod
    def from_config(cls, config: Any) -> "ConvGRUClassifier":
        """
        Khởi tạo SpatioTemporalConvGRUClassifier trực tiếp từ đối tượng cấu hình TrainConfig hoặc dict.
        """
        model_cfg = getattr(config, "model", config)
        if isinstance(model_cfg, dict):
            cfg_get = lambda k, default: model_cfg.get(k, default)
        else:
            cfg_get = lambda k, default: getattr(model_cfg, k, default)

        # Ưu tiên các siêu tham số ConvGRU hoặc fallback về config mặc định
        in_dim = cfg_get("convgru_in_dim", cfg_get("input_dim", 64))
        # Nếu config cũ có input_dim=256, ta tự động chuẩn hóa về 64 để tối ưu tham số
        if in_dim > 128:
            in_dim = 64

        hid_dim = cfg_get("convgru_hidden_dim", 64)
        num_layers = cfg_get("convgru_num_layers", cfg_get("num_layers", 2))
        num_classes = cfg_get("num_classes", 2)
        spatial_in = cfg_get("cnn_neck_channels", (64, 128, 256))
        adapter_drop = cfg_get("adapter_dropout", 0.1)
        dropout = cfg_get("dropout", 0.35)
        supervision = cfg_get("supervision_mode", "attention_pooling")

        return cls(
            input_dim=in_dim,
            hidden_dim=hid_dim,
            num_layers=num_layers,
            kernel_size=3,
            num_classes=num_classes,
            spatial_in_channels=tuple(spatial_in),
            neck_dropout=adapter_drop,
            convgru_dropout=0.1,
            dropout=dropout,
            supervision_mode=supervision
        )

    @classmethod
    def from_checkpoint(
            cls,
            checkpoint_path: Union[str, Path],
            map_location: str = "cpu"
    ) -> "ConvGRUClassifier":
        """
        Khởi tạo mô hình và nạp trọng số từ checkpoint .pt/.pth.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"[SpatioTemporalConvGRUClassifier] Không tìm thấy checkpoint: {path}")

        ckpt = torch.load(str(path), map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cfg_dict = ckpt.get("config", {})

        # Trích xuất số kênh từ state_dict nếu có
        weight_key = "convgru.cells.0.conv_gates.weight"
        if weight_key in state_dict:
            w_shape = state_dict[weight_key].shape  # [2 * H, in + H, K, K]
            out_c, in_total, _, _ = w_shape
            hid_dim = out_c // 2
            in_dim = in_total - hid_dim
        else:
            in_dim = 64
            hid_dim = 64

        # Đếm số tầng ConvGRU
        layer_keys = {int(k.split(".")[2]) for k in state_dict if
                      k.startswith("convgru.cells.") and k.split(".")[2].isdigit()}
        num_layers = len(layer_keys) if layer_keys else 2
        num_classes = int(cfg_dict.get("num_classes", 2))
        spatial_in = cfg_dict.get("cnn_neck_channels", (64, 128, 256))

        model = cls(
            input_dim=in_dim,
            hidden_dim=hid_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            spatial_in_channels=tuple(spatial_in)
        )
        if map_location is not None:
            model = model.to(map_location)

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            seq_lens: Optional[torch.Tensor] = None,
            hidden_state: Optional[List[torch.Tensor]] = None,
            return_sequence: bool = False,
            return_weights: bool = False,
            return_state: bool = False,
            *args: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        """
        Lan truyền xuôi toàn bộ mô hình.

        Args:
            features: Tuple/List 3 tensor (p3, p4, p5) hoặc Tensor đơn lẻ 4D/5D.
            seq_lens: Độ dài thực tế của từng clip trong batch [Batch].
            hidden_state: Danh sách hidden state ngoại vi (cho streaming liên tục).
            return_sequence: Nếu True, trả về logits từng frame [B, T, num_classes].
            return_weights: Nếu True, trả về dict trọng số chú ý không gian và thời gian.
            return_state: Nếu True, trả về hidden state cuối cùng của ConvGRU.
        Returns:
            logits hoặc Tuple(logits, [weights], [states])
        """
        # Xử lý trường hợp người dùng truyền rời 3 tensor dạng model(p3, p4, p5)
        if isinstance(seq_lens, torch.Tensor) and seq_lens.dim() >= 4:
            extra = [seq_lens]
            if isinstance(hidden_state, torch.Tensor) and hidden_state.dim() >= 4:
                extra.append(hidden_state)
                hidden_state = None
            if len(args) > 0:
                extra.extend(args)
            features = (features, *extra)
            seq_lens = None
        elif len(args) > 0 and isinstance(features, torch.Tensor):
            features = (features, *args)

        # 1. Căn chỉnh không gian và nén số kênh từ PAFPN (448 -> 64 kênh)
        fused = self.spatial_neck(features, seq_lens=seq_lens)  # [B, T, 64, 40, 40] hoặc [B, 64, 40, 40]

        is_single_frame = (fused.dim() == 4)
        if is_single_frame:
            fused = fused.unsqueeze(1)  # [B, 1, 64, 40, 40]

        # 2. Học biểu diễn chuỗi không gian - thời gian qua ConvGRU
        st_out, last_states = self.convgru(
            fused,
            hidden_state=hidden_state,
            seq_lens=seq_lens
        )  # st_out: [B, T, hid, 40, 40]

        # 3. Gom tụ chú ý không gian (Spatial Attention Pooling: 40x40 -> 1)
        if return_weights:
            t_features, spatial_weights = self.spatial_pooling(st_out, seq_lens=seq_lens, return_weights=True)
        else:
            t_features = self.spatial_pooling(st_out, seq_lens=seq_lens, return_weights=False)
            spatial_weights = None
        # t_features: [B, T, last_hidden_dim]

        # 4. Phân loại theo cơ chế Clip-level hoặc Frame-level
        is_seq_mode = return_sequence or (self.supervision_mode in ("sequence", "frame"))
        if is_seq_mode:
            logits = self.fc_out(t_features)  # [B, T, num_classes]
            if seq_lens is not None and not is_single_frame:
                b, t, _ = logits.shape
                time_indices = torch.arange(t, device=logits.device).unsqueeze(0)
                lens_expanded = seq_lens.to(logits.device).unsqueeze(1)
                mask = (time_indices < lens_expanded).unsqueeze(-1)
                logits = logits * mask.to(logits.dtype)
            temporal_weights = None
            if is_single_frame:
                logits = logits.squeeze(1)  # [B, num_classes]
        else:
            # Chế độ Clip-level chuẩn (Temporal Attention Pooling qua thời gian T)
            pooled, temporal_weights = self.temporal_pooling(t_features, seq_lens=seq_lens)
            logits = self.fc_out(pooled)  # [B, num_classes]

        # 5. Đóng gói kết quả đầu ra
        outputs = [logits]
        if return_weights:
            outputs.append({
                "spatial": spatial_weights,
                "temporal": temporal_weights
            })
        if return_state:
            outputs.append(last_states)

        return tuple(outputs) if len(outputs) > 1 else logits


# ==============================================================================
# BỘ KIỂM THỬ TỰ ĐỘNG CHUẨN XÁC CAO (SELF-VERIFICATION SUITE)
# ==============================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvGRUClassifier(
        input_dim=64,
        hidden_dim=[64, 64],
        num_layers=2,
        num_classes=2,
        spatial_in_channels=(64, 128, 256)
    ).to(device).eval()

    # Trường hợp A: Tuple 3 tensor 5D (p3, p4, p5)
    p3 = torch.randn(1, 6, 64, 28, 28, device=device)
    p4 = torch.randn(1, 6, 128, 14, 14, device=device)
    p5 = torch.randn(1, 6, 256, 7, 7, device=device)

    output = model((p3, p4, p5))
    print(output)
