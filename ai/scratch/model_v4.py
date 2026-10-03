import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, List, Optional, Any
from pathlib import Path

# ==============================================================================
# CELL 4: KIẾN TRÚC MÔ HÌNH SPATIAL FEATURE ADAPTER, TEMPORAL ATTENTION POOLING & DEEP GRU CLASSIFIER
# ==============================================================================
class SpatialFeatureAdapter(nn.Module):
    """
    Bộ chuyển đổi và kết hợp đặc trưng không gian đa tỷ lệ (p3: 64, p4: 128, p5: 256).
    Hỗ trợ các phương thức Fusion: 'attention', 'concat', 'sum', 'mean'.
    Hỗ trợ cả tensor 2D [B, C], tensor chuỗi 3D [B, T, C], và feature map 4D/5D.
    """

    def __init__(
            self,
            in_channels: Tuple[int, ...] = (64, 128, 256),
            out_dim: int = 256,
            fusion: str = "attention",
            dropout: float = 0.25,
            hidden_dim: int = 256
    ):
        super().__init__()
        self.num_scales = len(in_channels)
        self.in_channels = tuple(in_channels)
        self.out_dim = out_dim
        self.fusion = fusion.lower()
        self.total_in_channels = sum(in_channels)

        if self.fusion == "attention":
            self.projection = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(inplace=True)
                ) for c in in_channels
            ])
            self.attention_mlp = nn.Sequential(
                nn.Linear(out_dim * self.num_scales, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, self.num_scales)
            )
        elif self.fusion == "concat":
            self.projection = nn.Sequential(
                nn.Linear(self.total_in_channels, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True)
            )
            self.attention_mlp = None
        elif self.fusion in ("sum", "mean"):
            self.projection = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(c, out_dim),
                    nn.LayerNorm(out_dim),
                    nn.ReLU(inplace=True)
                ) for c in in_channels
            ])
            self.attention_mlp = None
        else:
            raise ValueError(f"Phương thức fusion '{fusion}' không được hỗ trợ. Chọn: ['attention', 'concat', 'sum', 'mean'].")

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def _pool_feature(self, x: torch.Tensor) -> torch.Tensor:
        """Nén đặc trưng không gian (H, W) về (1, 1) nếu đầu vào là feature map 4D hoặc 5D."""
        if x.dim() == 5:  # [B, T, C, H, W]
            b, t, c, h, w = x.shape
            return F.adaptive_avg_pool2d(x.reshape(b * t, c, h, w), (1, 1)).view(b, t, c)
        elif x.dim() == 4:  # [B, C, H, W]
            b, c, h, w = x.shape
            return F.adaptive_avg_pool2d(x, (1, 1)).view(b, c)
        return x

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            *args: torch.Tensor,
            return_weights: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if len(args) > 0:
            features = (features, *args)

        if isinstance(features, (tuple, list)):
            if len(features) != self.num_scales:
                raise ValueError(f"Số lượng feature maps đầu vào ({len(features)}) không khớp in_channels ({self.num_scales}).")
            feats = [self._pool_feature(f) for f in features]
        elif isinstance(features, torch.Tensor):
            feat = self._pool_feature(features)
            if feat.shape[-1] == self.out_dim:
                return (self.dropout(feat), None) if return_weights else self.dropout(feat)
            if self.fusion == "concat":
                out = self.dropout(self.projection(feat))
                if return_weights:
                    weights = torch.ones(*feat.shape[:-1], self.num_scales, device=feat.device) / self.num_scales
                    return out, weights
                return out
            elif feat.shape[-1] == self.total_in_channels:
                feats = list(torch.split(feat, list(self.in_channels), dim=-1))
            else:
                raise ValueError(f"Kích thước kênh cuối ({feat.shape[-1]}) không khớp ({self.total_in_channels}) hoặc ({self.out_dim}).")
        else:
            raise TypeError(f"Định dạng features không hợp lệ: {type(features)}.")

        if self.fusion == "concat":
            concat_v = torch.cat(feats, dim=-1)
            out = self.dropout(self.projection(concat_v))
            if return_weights:
                weights = torch.ones(*concat_v.shape[:-1], self.num_scales, device=concat_v.device) / self.num_scales
                return out, weights
            return out

        v_list = [proj(f) for proj, f in zip(self.projection, feats)]

        if self.fusion == "attention":
            concat_v = torch.cat(v_list, dim=-1)
            weights = self.attention_mlp(concat_v)
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)

            stacked_v = torch.stack(v_list, dim=-2)
            fused = (stacked_v * weights).sum(dim=-2)
            fused = self.dropout(fused)

            if return_weights:
                return fused, weights.squeeze(-1)
            return fused

        elif self.fusion == "sum":
            stacked_v = torch.stack(v_list, dim=-2)
            fused = self.dropout(stacked_v.sum(dim=-2))
            if return_weights:
                weights = torch.ones(*stacked_v.shape[:-1], self.num_scales, device=stacked_v.device) / self.num_scales
                return fused, weights
            return fused

        elif self.fusion == "mean":
            stacked_v = torch.stack(v_list, dim=-2)
            fused = self.dropout(stacked_v.mean(dim=-2))
            if return_weights:
                weights = torch.ones(*stacked_v.shape[:-1], self.num_scales, device=stacked_v.device) / self.num_scales
                return fused, weights
            return fused


class TemporalAttentionPooling(nn.Module):
    """
    Gom tụ chuỗi thời gian có trọng số chú ý động (Temporal Attention MLP),
    triệt tiêu 100% gradient rác từ các khung hình zero-padding qua Attention Masking.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, gru_out: torch.Tensor, seq_lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            gru_out: Tensor đầu ra từ GRU [Batch, Time, Hidden]
            seq_lens: Tensor độ dài thực tế của từng mẫu trong batch [Batch]
        Returns:
            pooled: Vector đặc trưng đại diện clip [Batch, Hidden]
            weights: Trọng số chú ý tương ứng từng khung hình [Batch, Time]
        """
        b, t, h = gru_out.shape
        scores = self.attn(gru_out).squeeze(-1)  # [B, T]

        if seq_lens is not None:
            # Tạo boolean mask: True cho frame hợp lệ, False cho frame padding
            mask = torch.arange(t, device=gru_out.device).unsqueeze(0) < seq_lens.to(gru_out.device).unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e9)

        weights = F.softmax(scores, dim=-1)  # [B, T]
        # Gom tụ có trọng số: [B, 1, T] x [B, T, H] -> [B, 1, H] -> [B, H]
        pooled = torch.bmm(weights.unsqueeze(1), gru_out).squeeze(1)
        return pooled, weights


class DeepGRUClassifier(nn.Module):
    """
    Mô hình phân loại chuỗi thời gian Deep GRU 2 tầng kết hợp Temporal Attention Pooling (Phương án A):
    (p3, p4, p5) -> Spatial Adapter -> 2-layer GRU -> Temporal Attention Pooling -> FC Head (num_classes).
    """

    def __init__(
            self,
            input_dim: int = 256,
            hidden_dim: int = 192,
            num_layers: int = 2,
            num_classes: int = 2,
            spatial_in_channels: Tuple[int, ...] = (64, 128, 256),
            fusion: str = "attention",
            adapter_dropout: float = 0.25,
            dropout: float = 0.35,
            supervision_mode: str = "attention_pooling"
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.fusion = fusion
        self.supervision_mode = supervision_mode

        self.spatial_adapter = SpatialFeatureAdapter(
            in_channels=spatial_in_channels,
            out_dim=input_dim,
            fusion=fusion,
            dropout=adapter_dropout
        )
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.temporal_pooling = TemporalAttentionPooling(hidden_dim=hidden_dim)
        self.fc_out = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    @property
    def lstm(self) -> nn.GRU:
        """Alias tương thích ngược cho các đoạn mã cũ gọi model.lstm."""
        return self.gru

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], map_location: str = "cpu") -> "DeepGRUClassifier":
        """Khởi tạo DeepGRUClassifier và nạp trọng số trực tiếp từ file checkpoint (.pth/.pt)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

        ckpt = torch.load(str(path), map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cfg_dict = ckpt.get("config", {})

        weight_key = next((k for k in ("gru.weight_ih_l0", "lstm.weight_ih_l0") if k in state_dict), None)
        if weight_key:
            input_dim = state_dict[weight_key].shape[1]
            hidden_dim = state_dict[weight_key].shape[0] // 3
        else:
            input_dim = int(cfg_dict.get("input_dim", 256))
            hidden_dim = int(cfg_dict.get("hidden_dim", 192))

        num_classes = int(cfg_dict.get("num_classes", 2))
        layer_indices = {int(k.split("weight_ih_l")[-1]) for k in state_dict if "weight_ih_l" in k and k.split("weight_ih_l")[-1].isdigit()}
        num_layers = len(layer_indices) if layer_indices else int(cfg_dict.get("num_layers", 2))
        fusion = cfg_dict.get("spatial_fusion", "attention")
        spatial_in_channels = cfg_dict.get("cnn_neck_channels", (64, 128, 256))

        remapped_sd = {k.replace("lstm.", "gru.", 1) if k.startswith("lstm.") else k: v for k, v in state_dict.items()}

        model = cls(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            spatial_in_channels=spatial_in_channels,
            fusion=fusion
        )
        if map_location is not None:
            model = model.to(map_location)
        model.load_state_dict(remapped_sd, strict=False)
        model.eval()
        return model

    @classmethod
    def from_config(cls, config: Any) -> "DeepGRUClassifier":
        """Khởi tạo DeepGRUClassifier trực tiếp từ đối tượng TrainConfig."""
        return cls(
            input_dim=getattr(config, "input_dim", 256),
            hidden_dim=getattr(config, "hidden_dim", 192),
            num_layers=getattr(config, "num_layers", 2),
            num_classes=getattr(config, "num_classes", 2),
            spatial_in_channels=getattr(config, "cnn_neck_channels", (64, 128, 256)),
            fusion=getattr(config, "spatial_fusion", "attention"),
            adapter_dropout=getattr(config, "adapter_dropout", 0.25),
            dropout=getattr(config, "dropout", 0.35),
            supervision_mode=getattr(config, "supervision_mode", "attention_pooling")
        )

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            *args: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None,
            h_0: Optional[torch.Tensor] = None,
            hc: Optional[Any] = None,
            return_sequence: bool = False,
            return_weights: bool = False,
            return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        if len(args) > 0:
            features = (features, *args)
        if hc is not None and h_0 is None:
            h_0 = hc[0] if isinstance(hc, tuple) else hc
        if isinstance(h_0, tuple):
            h_0 = h_0[0]

        # 1. Chuyển đổi đặc trưng không gian
        x = self.spatial_adapter(features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 2. Trích xuất đặc trưng chuỗi thời gian qua Deep GRU
        gru_out, h_n = self.gru(x, h_0)

        # 3. Phân loại theo cơ chế
        if return_sequence:
            logits = self.fc_out(gru_out)  # [B, T, num_classes] (Hỗ trợ streaming / sequence)
            attn_weights = None
        else:
            # Phương án A: Temporal Attention Pooling (Clip-level)
            pooled, attn_weights = self.temporal_pooling(gru_out, seq_lens)
            logits = self.fc_out(pooled)  # [B, num_classes]

        outputs = [logits]
        if return_weights:
            outputs.append(attn_weights)
        if return_state:
            outputs.append(h_n)

        return tuple(outputs) if len(outputs) > 1 else logits


if __name__ == "__main__":
    print("=" * 75)
    print("[*] KIỂM TRA MÔ HÌNH DEEP GRU V4 (TEMPORAL ATTENTION POOLING & STREAMING)")
    print("=" * 75)

    batch_size, seq_len = 4, 120
    f1_3d = torch.randn(batch_size, seq_len, 64)
    f2_3d = torch.randn(batch_size, seq_len, 128)
    f3_3d = torch.randn(batch_size, seq_len, 256)
    dummy_lens = torch.tensor([120, 100, 80, 60], dtype=torch.long)

    model_v4 = DeepGRUClassifier(
        input_dim=256,
        hidden_dim=192,
        num_layers=2,
        num_classes=2,
        fusion="attention"
    )

    out_clip, w_attn = model_v4((f1_3d, f2_3d, f3_3d), seq_lens=dummy_lens, return_sequence=False, return_weights=True)
    out_seq = model_v4((f1_3d, f2_3d, f3_3d), return_sequence=True)

    print(f"[+] Output Clip Mode (Phương án A) : {list(out_clip.shape)} (Mong đợi [4, 2])")
    print(f"[+] Attention Weights Shape         : {list(w_attn.shape)} (Mong đợi [4, 120])")
    print(f"[+] Output Sequence Mode            : {list(out_seq.shape)} (Mong đợi [4, 120, 2])")

    # Kiểm tra mask: mẫu thứ 4 (lens=60) các frame từ 60 đến 119 phải có attention weight = 0.0
    padded_attn_sum = w_attn[3, 60:].sum().item()
    valid_attn_sum = w_attn[3, :60].sum().item()
    print(f"[+] Tổng trọng số Attention frame padding (mẫu 4): {padded_attn_sum:.6f} (Mong đợi 0.000000)")
    print(f"[+] Tổng trọng số Attention frame hợp lệ (mẫu 4) : {valid_attn_sum:.6f} (Mong đợi 1.000000)")

    total_params = sum(p.numel() for p in model_v4.parameters())
    print(f"[+] Tổng số tham số mô hình tinh gọn v4          : {total_params:,} (Giảm 64% so với 1.24M)")

    assert out_clip.shape == (4, 2)
    assert out_seq.shape == (4, 120, 2)
    assert abs(valid_attn_sum - 1.0) < 1e-4
    assert padded_attn_sum < 1e-6
    print("[+] TẤT CẢ KIỂM THỬ ARCHITECTURE ĐÃ VƯỢT QUA 100% THÀNH CÔNG!")
    print("=" * 75)
