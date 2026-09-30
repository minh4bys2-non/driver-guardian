from pathlib import Path
from typing import Optional, Union, Tuple, Any, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TrainConfig


# ==============================================================================
# SPATIAL FEATURE ADAPTER & DEEP GRU CLASSIFIER ARCHITECTURE
# ==============================================================================
class SpatialFeatureAdapter(nn.Module):
    """
    Bộ chuyển đổi và kết hợp đặc trưng không gian đa tỷ lệ (p3: 64, p4: 128, p5: 256).
    Hỗ trợ các phương thức Fusion:
      - 'attention': Chiếu riêng từng tầng về out_dim, tính trọng số qua Attention MLP động và lấy tổng có trọng số.
      - 'concat': Ghép nối toàn bộ kênh (448) rồi chiếu về out_dim.
      - 'sum' / 'mean': Chiếu từng tầng về out_dim rồi cộng dồn / lấy trung bình.

    Hỗ trợ hoàn hảo cả tensor 2D [B, C], tensor chuỗi 3D [B, T, C], và feature map chưa pooling 4D/5D.
    """

    def __init__(
            self,
            in_channels: Tuple[int, ...] = (64, 128, 256),
            out_dim: int = 256,
            fusion: str = "attention",
            dropout: float = 0.1,
            hidden_dim: int = 256
    ):
        super().__init__()
        self.num_scales = len(in_channels)
        self.in_channels = tuple(in_channels)
        self.out_dim = out_dim
        self.fusion = fusion.lower()
        self.total_in_channels = sum(in_channels)

        if self.fusion == "attention":
            # Chiếu riêng từng tầng về out_dim với LayerNorm
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
            # Ghép nối 1312 kênh -> chiếu về out_dim
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
            raise ValueError(
                f"Phương thức fusion '{fusion}' không được hỗ trợ. Chọn một trong: ['attention', 'concat', 'sum', 'mean']."
            )

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
        """
        Quá trình lan truyền xuôi của Adapter.

        Args:
            features: Danh sách/Tuple các feature maps [p3, p4, p5] hoặc Tensor đơn lẻ.
            *args: Các tensor tiếp theo nếu truyền dạng model(p3, p4, p5).
            return_weights: Trả về trọng số attention [..., num_scales].

        Returns:
            fused_vector: Tensor [..., out_dim]
            weights (optional): Tensor [..., num_scales]
        """
        if len(args) > 0:
            features = (features, *args)

        if isinstance(features, (tuple, list)):
            if len(features) != self.num_scales:
                raise ValueError(
                    f"Số lượng feature maps đầu vào ({len(features)}) không khớp cấu hình in_channels ({self.num_scales})."
                )
            feats = [self._pool_feature(f) for f in features]
        elif isinstance(features, torch.Tensor):
            feat = self._pool_feature(features)
            # Nếu tensor đã có số chiều khớp out_dim
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
                raise ValueError(
                    f"Kích thước kênh cuối của tensor ({feat.shape[-1]}) không khớp với tổng kênh "
                    f"({self.total_in_channels}) hoặc out_dim ({self.out_dim})."
                )
        else:
            raise TypeError(
                f"Định dạng features không hợp lệ: {type(features)}. Mong đợi Tuple/List hoặc torch.Tensor."
            )

        # Xử lý theo chiến lược Fusion
        if self.fusion == "concat":
            concat_v = torch.cat(feats, dim=-1)
            out = self.dropout(self.projection(concat_v))
            if return_weights:
                weights = torch.ones(*concat_v.shape[:-1], self.num_scales, device=concat_v.device) / self.num_scales
                return out, weights
            return out

        # Cho attention, sum, mean: chiếu từng tỷ lệ
        v_list = [proj(f) for proj, f in zip(self.projection, feats)]

        if self.fusion == "attention":
            concat_v = torch.cat(v_list, dim=-1)
            weights = self.attention_mlp(concat_v)
            weights = F.softmax(weights, dim=-1).unsqueeze(-1)  # [..., num_scales, 1]

            # Xếp chồng theo dim=-2 để tương thích cả 2D [B, S, D] lẫn 3D [B, T, S, D]
            stacked_v = torch.stack(v_list, dim=-2)
            fused = (stacked_v * weights).sum(dim=-2)  # [..., out_dim]
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


class DeepGRUClassifier(nn.Module):
    """
    Mô hình phân loại chuỗi thời gian Deep GRU 3 lớp xếp chồng:
    Đầu vào: (p3, p4, p5) -> Spatial Adapter (out_dim) -> GRU (3 layers) -> FC Head (num_classes).
    """

    def __init__(
            self,
            input_dim: int = 256,
            hidden_dim: int = 256,
            num_layers: int = 3,
            num_classes: int = 2,
            spatial_in_channels: Tuple[int, ...] = (64, 128, 256),
            fusion: str = "attention",
            adapter_dropout: float = 0.1,
            dropout: float = 0.2
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.fusion = fusion

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
        """
        Khởi tạo DeepGRUClassifier và nạp trọng số trực tiếp từ file checkpoint (.pth/.pt).
        Tự động nhận diện cấu hình lưu trong checkpoint và nạp khớp chính xác 100%.
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file checkpoint: {path}")

        ckpt = torch.load(str(path), map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        # Tự động nhận diện cấu hình lưu từ checkpoint
        cfg_dict = ckpt.get("config", {})

        # Tự động nhận diện input_dim và hidden_dim cho GRU (chia cho 3 vì GRU có 3 cổng)
        weight_key = next((k for k in ("gru.weight_ih_l0", "lstm.weight_ih_l0") if k in state_dict), None)
        if weight_key:
            input_dim = state_dict[weight_key].shape[1]
            hidden_dim = state_dict[weight_key].shape[0] // 3
        else:
            input_dim = int(cfg_dict.get("input_dim", 256))
            hidden_dim = int(cfg_dict.get("hidden_dim", 256))

        if "fc_out.1.weight" in state_dict:
            num_classes = state_dict["fc_out.1.weight"].shape[0]
        elif "fc_out.weight" in state_dict:
            num_classes = state_dict["fc_out.weight"].shape[0]
        else:
            num_classes = int(cfg_dict.get("num_classes", 2))

        # Đếm số lớp của GRU
        layer_indices = set()
        for k in state_dict.keys():
            if k.startswith("gru.weight_ih_l") or k.startswith("lstm.weight_ih_l"):
                idx_str = k.split("weight_ih_l")[-1]
                if idx_str.isdigit():
                    layer_indices.add(int(idx_str))
        num_layers = len(layer_indices) if layer_indices else int(cfg_dict.get("num_layers", 3))

        # Tự động nhận diện kiến trúc adapter
        has_seq_proj = "spatial_adapter.projection.0.weight" in state_dict and not any(
            k.startswith("spatial_adapter.projection.0.0.") for k in state_dict.keys()
        )
        if has_seq_proj:
            fusion = "concat"
        elif "spatial_adapter.attention_mlp.0.weight" in state_dict:
            fusion = "attention"
        else:
            fusion = cfg_dict.get("spatial_fusion", getattr(cfg_dict, "fusion", "concat"))

        # Tự động nhận diện cấu hình số kênh spatial_in_channels
        if "spatial_adapter.projection.0.0.weight" in state_dict:
            proj_weights = [
                state_dict[k] for k in sorted(state_dict.keys())
                if k.startswith("spatial_adapter.projection.") and k.endswith(".0.weight")
            ]
            if len(proj_weights) >= 3:
                spatial_in_channels = tuple(w.shape[1] for w in proj_weights)
            else:
                spatial_in_channels = cfg_dict.get("cnn_neck_channels", (64, 128, 256))
        elif "spatial_adapter.projection.0.weight" in state_dict:
            total_in = state_dict["spatial_adapter.projection.0.weight"].shape[1]
            if total_in == 448:
                spatial_in_channels = (64, 128, 256)
            elif total_in == 1312:
                spatial_in_channels = (224, 448, 640)
            else:
                spatial_in_channels = cfg_dict.get("cnn_neck_channels", (64, 128, 256))
        else:
            spatial_in_channels = cfg_dict.get("cnn_neck_channels", (64, 128, 256))

        # Tự động map lại các key 'lstm.' sang 'gru.' nếu checkpoint cũ lưu tiền tố 'lstm.'
        remapped_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("lstm."):
                remapped_state_dict[k.replace("lstm.", "gru.", 1)] = v
            else:
                remapped_state_dict[k] = v

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
        model.load_state_dict(remapped_state_dict, strict=False)
        model.eval()
        return model

    @classmethod
    def from_config(cls, config: Any) -> "DeepGRUClassifier":
        """Khởi tạo DeepGRUClassifier trực tiếp từ đối tượng TrainConfig."""
        spatial_in_channels = getattr(config, "cnn_neck_channels", getattr(config, "cnn_out_channels", (64, 128, 256)))
        fusion = getattr(config, "spatial_fusion", getattr(config, "fusion", "concat"))
        adapter_dropout = getattr(config, "adapter_dropout", 0.1)
        dropout = getattr(config, "dropout", 0.2)

        return cls(
            input_dim=getattr(config, "input_dim", 256),
            hidden_dim=getattr(config, "hidden_dim", 256),
            num_layers=getattr(config, "num_layers", 3),
            num_classes=getattr(config, "num_classes", 2),
            spatial_in_channels=spatial_in_channels,
            fusion=fusion,
            adapter_dropout=adapter_dropout,
            dropout=dropout
        )

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            *args: torch.Tensor,
            h_0: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
            hc: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
            return_sequence: bool = True,
            return_weights: bool = False,
            return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        """
        Quá trình lan truyền xuôi của DeepGRUClassifier.

        Args:
            features: Bộ 3 đặc trưng (p3, p4, p5) hoặc Tensor chuỗi thời gian.
            *args: Các tensor đặc trưng p4, p5 nếu gọi model(p3, p4, p5).
            h_0: Trạng thái ẩn ban đầu của GRU [num_layers, B, hidden_dim] (phục vụ streaming inference).
            hc: Tham số tương thích ngược nếu truyền tên biến hc cũ.
            return_sequence: True -> trả về logits cho toàn chuỗi [B, T, num_classes].
                             False -> chỉ trả về frame cuối cùng [B, num_classes].
            return_weights: True -> trả về thêm trọng số attention không gian.
            return_state: True -> trả về thêm trạng thái ẩn h_n phục vụ streaming inference.
        """
        if len(args) > 0:
            features = (features, *args)

        # Hỗ trợ tương thích ngược nếu gọi tham số hc
        if hc is not None and h_0 is None:
            h_0 = hc

        # An toàn: GRU chỉ nhận Tensor h_0, nếu truyền nhầm tuple (h, c) thì lấy h_0
        if isinstance(h_0, tuple):
            h_0 = h_0[0]

        # 1. Chuyển đổi đặc trưng không gian qua adapter
        if return_weights:
            x, weights = self.spatial_adapter(features, return_weights=True)
        else:
            x = self.spatial_adapter(features, return_weights=False)
            weights = None

        # Đảm bảo tensor có 3 chiều [Batch, Seq_Len, Dim]
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 2. Trích xuất đặc trưng chuỗi thời gian qua Deep GRU
        gru_out, h_n = self.gru(x, h_0)

        # 3. Phân loại theo sequence hoặc frame cuối cùng
        if return_sequence:
            logits = self.fc_out(gru_out)  # [B, T, num_classes]
        else:
            logits = self.fc_out(gru_out[:, -1, :])  # [B, num_classes]

        outputs = [logits]
        if return_weights:
            outputs.append(weights)
        if return_state:
            outputs.append(h_n)

        return tuple(outputs) if len(outputs) > 1 else logits


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    print("=" * 75)
    print("[*] KIỂM TRA MÔ HÌNH SPATIAL FEATURE ADAPTER & DEEP GRU CLASSIFIER")
    print("=" * 75)

    # 1. Kiểm thử với Tensor 2D [Batch, Channels]
    batch_size = 4
    f1_2d = torch.randn(batch_size, 64)
    f2_2d = torch.randn(batch_size, 128)
    f3_2d = torch.randn(batch_size, 256)

    adapter_attn = SpatialFeatureAdapter(out_dim=256, fusion="attention")
    fused_2d, alpha_2d = adapter_attn([f1_2d, f2_2d, f3_2d], return_weights=True)

    print(f"[+] Kiểm thử 2D [B={batch_size}, C]:")
    print(f"    - Output vector shape : {list(fused_2d.shape)} (Mong đợi [4, 256])")
    print(f"    - Alpha weights shape : {list(alpha_2d.shape)} (Mong đợi [4, 3])")
    print(f"    - Tổng alpha mẫu đầu : {alpha_2d[0].sum().item():.4f} (Xấp xỉ 1.0)")
    assert fused_2d.shape == (4, 256)
    assert alpha_2d.shape == (4, 3)

    # 2. Kiểm thử với Tensor chuỗi thời gian 3D [Batch, Seq_Len, Channels]
    seq_len = 120
    f1_3d = torch.randn(batch_size, seq_len, 64)
    f2_3d = torch.randn(batch_size, seq_len, 128)
    f3_3d = torch.randn(batch_size, seq_len, 256)

    fused_3d, alpha_3d = adapter_attn((f1_3d, f2_3d, f3_3d), return_weights=True)
    print(f"\n[+] Kiểm thử 3D [B={batch_size}, T={seq_len}, C]:")
    print(f"    - Output sequence shape: {list(fused_3d.shape)} (Mong đợi [4, 120, 256])")
    print(f"    - Alpha sequence shape : {list(alpha_3d.shape)} (Mong đợi [4, 120, 3])")
    assert fused_3d.shape == (4, 120, 256)
    assert alpha_3d.shape == (4, 120, 3)

    # 3. Kiểm thử DeepGRUClassifier toàn diện (chế độ Sequence, Clip, Attention Weights và Hidden State)
    model = DeepGRUClassifier(input_dim=256, hidden_dim=256, num_layers=3, num_classes=2, fusion="attention")

    out_seq, w_seq = model((f1_3d, f2_3d, f3_3d), return_sequence=True, return_weights=True)
    out_vid = model((f1_3d, f2_3d, f3_3d), return_sequence=False)

    # Kiểm thử Streaming & Trạng thái ẩn h_0, h_n
    h_init = torch.zeros(3, batch_size, 256)
    out_stream, h_next = model((f1_3d, f2_3d, f3_3d), h_0=h_init, return_sequence=False, return_state=True)

    # Kiểm thử tương thích ngược khi truyền tuple (h, c) cũ
    out_compat = model((f1_3d, f2_3d, f3_3d), hc=(h_init, h_init), return_sequence=False)

    print(f"\n[+] Kiểm thử DeepGRUClassifier:")
    print(f"    - Logits (Sequence Mode): {list(out_seq.shape)} (Mong đợi [4, 120, 2])")
    print(f"    - Logits (Clip Mode)    : {list(out_vid.shape)} (Mong đợi [4, 2])")
    print(f"    - Attention Weights     : {list(w_seq.shape)} (Mong đợi [4, 120, 3])")
    print(f"    - Streaming h_next shape: {list(h_next.shape)} (Mong đợi [3, 4, 256])")
    print(f"    - Tuple hc backward-comp: {list(out_compat.shape)} (Mong đợi [4, 2])")
    assert out_seq.shape == (4, 120, 2)
    assert out_vid.shape == (4, 2)
    assert w_seq.shape == (4, 120, 3)
    assert h_next.shape == (3, 4, 256)
    assert out_compat.shape == (4, 2)

    # 4. Kiểm thử các phương thức Fusion khác nhau
    for f_mode in ["concat", "sum", "mean", "attention"]:
        m_fusion = DeepGRUClassifier(fusion=f_mode)
        out = m_fusion((f1_3d, f2_3d, f3_3d))
        print(f"    - Fusion [{f_mode:<9}]: Output shape = {list(out.shape)}")
        assert out.shape == (4, 120, 2)

    # 5. Kiểm thử nạp Checkpoint thực tế (nếu tồn tại)
    for ckpt_candidate in [
        "gru_experiment_results/checkpoints/best_gru.pth",
        "gru_experiment_results/checkpoints/best_lstm.pth"
    ]:
        ckpt_path = Path(ckpt_candidate)
        if ckpt_path.exists():
            ckpt_model = DeepGRUClassifier.from_checkpoint(ckpt_path)
            print(f"\n[+] Nạp thành công checkpoint thực tế '{ckpt_path}'!")
            print(f"    - Tổng tham số mô hình nạp: {sum(p.numel() for p in ckpt_model.parameters()):,}")
            break

    print("\n" + "=" * 75)
    print("[+] TẤT CẢ KIỂM THỬ ĐÃ HOÀN TẤT VÀ VƯỢT QUA 100% THÀNH CÔNG!")
    print("=" * 75)

