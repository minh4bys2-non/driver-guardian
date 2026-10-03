import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path
from typing import Tuple, Union, List, Any, Optional


class CNNAdapter(nn.Module):
    """
    Bộ chuyển đổi và kết hợp đặc trưng không gian đa tỷ lệ:
      - p3: [Bz, 64, 80, 80] hoặc [Bz, T, 64, 80, 80]
      - p4: [Bz, 128, 40, 40] hoặc [Bz, T, 128, 40, 40]
      - p5: [Bz, 256, 20, 20] hoặc [Bz, T, 256, 20, 20]

    Nguyên tắc cốt lõi:
      - TUYỆT ĐỐI KHÔNG DÙNG Global Average Pooling (GAP) để tránh triệt tiêu tương quan không gian.
      - Sử dụng phương thức phễu tích chập phân tầng `hierarchical_conv_pyramid` với bước trượt (stride=2)
        để học tương quan từ cục bộ -> vùng -> toàn thể khuôn mặt, tổng hợp về out_dim bằng kernel học được (5x5).
      - Hỗ trợ toàn diện cả tensor 4D (đơn frame), 5D (chuỗi video), và tensor vector 2D/3D tương thích ngược.
    """

    def __init__(
            self,
            in_channels: Union[Tuple[Tuple[int, ...], ...], Tuple[int, ...]] = (64, 128, 256),
            out_dim: int = 256,
            fusion: str = "concat",
            dropout: float = 0.1,
            hidden_dim: int = 512
    ):
        super().__init__()

        # Chuẩn hóa cấu hình kênh đầu vào: hỗ trợ cả tuple(int) và tuple(tuple(int))
        if len(in_channels) > 0 and isinstance(in_channels[0], (tuple, list)):
            self.channel_dims = tuple(c[0] for c in in_channels)
            self.spatial_shapes = tuple(c[1:] for c in in_channels)
        else:
            self.channel_dims = tuple(int(c) for c in in_channels)
            self.spatial_shapes = None

        self.num_scales = len(self.channel_dims)
        self.in_channels = tuple(in_channels)
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.fusion = fusion.lower()
        self.total_in_channels = sum(self.channel_dims)

        if self.fusion not in ("concat", "attention", "sum", "mean"):
            raise ValueError(
                f"Phương thức fusion '{fusion}' không được hỗ trợ. Chọn một trong: ['attention', 'concat', 'sum', 'mean']."
            )

        # 1. Các tầng căn chỉnh kích thước không gian về lưới trung gian 40x40
        self.p3_down = nn.MaxPool2d(kernel_size=2, stride=2)
        self.p5_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        # 2. Các tầng tích chập của Phễu phân tầng (Hierarchical Conv Pyramid Layers - KHÔNG DÙNG GAP)
        # Stage 1: [N, total_in, 40, 40] -> [N, hidden_dim, 20, 20] (Học tương quan cục bộ: mép mi mắt, vành môi)
        self.pyramid_stage1 = nn.Sequential(
            nn.Conv2d(self.total_in_channels, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Stage 2: [N, hidden_dim, 20, 20] -> [N, hidden_dim, 10, 10] (Học tương quan vùng: khoảng cách hai mắt, mắt-mũi-miệng)
        self.pyramid_stage2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Stage 3: [N, hidden_dim, 10, 10] -> [N, hidden_dim, 5, 5] (Học tương quan toàn khuôn mặt & góc nghiêng đầu)
        self.pyramid_stage3 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True)
        )

        # Stage 4: [N, hidden_dim, 5, 5] -> [N, out_dim, 1, 1] (Tổng hợp toàn bộ lưới 5x5 bằng ma trận trọng số 5x5 học được)
        self.pyramid_stage4 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, kernel_size=5, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.SiLU(inplace=True)
        )

        # 3. Nhánh hỗ trợ Attention và Tương thích ngược
        self.scale_attention = nn.Sequential(
            nn.Linear(out_dim, hidden_dim // 2),
            nn.SiLU(inplace=True),
            nn.Linear(hidden_dim // 2, self.num_scales)
        )

        # Nhánh Linear cho tensor vector 2D/3D đã pooling sẵn
        self.linear_project = nn.Sequential(
            nn.Linear(self.total_in_channels, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU(inplace=True)
        )

        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def hierarchical_conv_pyramid(self, x: torch.Tensor) -> torch.Tensor:
        """
        Phương thức phễu tích chập phân tầng nén không gian từ 40x40 về vector out_dim:
          - Stage 1: [N, 448, 40, 40] -> [N, hidden_dim, 20, 20]
          - Stage 2: [N, hidden_dim, 20, 20] -> [N, hidden_dim, 10, 10]
          - Stage 3: [N, hidden_dim, 10, 10] -> [N, hidden_dim, 5, 5]
          - Stage 4: [N, hidden_dim, 5, 5] -> [N, out_dim, 1, 1] (Kernel 5x5 học được)
          - Squeeze: [N, out_dim, 1, 1] -> [N, out_dim]
        Tuyệt đối không dùng GAP.
        """
        x = self.pyramid_stage1(x)
        x = self.pyramid_stage2(x)
        x = self.pyramid_stage3(x)
        x = self.pyramid_stage4(x)
        x = x.squeeze(-1).squeeze(-1)
        return self.dropout(x)

    def _align_spatial_features(
            self,
            p3: torch.Tensor,
            p4: torch.Tensor,
            p5: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Căn chỉnh kích thước không gian của 3 tầng về lưới trung gian 40x40."""
        # p3: 80x80 -> 40x40 qua MaxPool2d
        out_p3 = self.p3_down(p3) if p3.shape[-2:] != (40, 40) else p3
        # p4: 40x40 giữ nguyên
        out_p4 = p4
        # p5: 20x20 -> 40x40 qua Upsample
        out_p5 = self.p5_up(p5) if p5.shape[-2:] != (40, 40) else p5
        return out_p3, out_p4, out_p5

    def forward(
            self,
            features: Union[Tuple[torch.Tensor, ...], List[torch.Tensor], torch.Tensor],
            *args: torch.Tensor,
            return_weights: bool = False
    ) -> Union[Tuple[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]:
        """
        Quá trình lan truyền xuôi của CNNAdapter.

        Hỗ trợ:
          - Tuple/List 3 tensor (p3, p4, p5) ở định dạng 4D [B, C, H, W] hoặc 5D [B, T, C, H, W]
          - Tensor đơn lẻ 4D/5D [B, (T), 448, 40, 40]
          - Tensor vector 2D/3D [B, (T), 448] tương thích ngược
        """
        if len(args) > 0:
            features = (features, *args)

        # ======================================================================
        # TRƯỜNG HỢP 1: ĐẦU VÀO LÀ TUPLE/LIST 3 TẦNG ĐẶC TRƯNG (p3, p4, p5)
        # ======================================================================
        if isinstance(features, (tuple, list)):
            if len(features) != self.num_scales:
                raise ValueError(
                    f"Số lượng feature maps đầu vào ({len(features)}) không khớp cấu hình ({self.num_scales})."
                )
            p3, p4, p5 = features[0], features[1], features[2]

            # Kiểm tra xem có phải chuỗi video 5D [B, T, C, H, W] không
            is_5d = (p3.dim() == 5)
            orig_b, orig_t = 0, 0
            if is_5d:
                orig_b, orig_t = p3.shape[0], p3.shape[1]
                p3 = p3.reshape(orig_b * orig_t, *p3.shape[2:])
                p4 = p4.reshape(orig_b * orig_t, *p4.shape[2:])
                p5 = p5.reshape(orig_b * orig_t, *p5.shape[2:])

            # Nếu đầu vào là tensor không gian 4D [N, C, H, W]
            if p3.dim() == 4:
                N = p3.shape[0]
                chunk_size = 32  # Chia chunk nhỏ để giữ đỉnh VRAM dưới 200MB, an toàn tuyệt đối trên RTX 3050 Laptop
                if N > chunk_size:
                    projected_list = []
                    weights_list = [] if (return_weights or self.fusion == "attention") else None
                    for i in range(0, N, chunk_size):
                        p3_c = p3[i : i + chunk_size]
                        p4_c = p4[i : i + chunk_size]
                        p5_c = p5[i : i + chunk_size]
                        out_p3_c, out_p4_c, out_p5_c = self._align_spatial_features(p3_c, p4_c, p5_c)
                        fused_c = torch.cat([out_p3_c, out_p4_c, out_p5_c], dim=1)
                        proj_c = self.hierarchical_conv_pyramid(fused_c)
                        projected_list.append(proj_c)
                        if weights_list is not None:
                            raw_w = self.scale_attention(proj_c)
                            weights_list.append(F.softmax(raw_w, dim=-1))
                    projected = torch.cat(projected_list, dim=0)
                    weights = torch.cat(weights_list, dim=0) if weights_list is not None else None
                else:
                    out_p3, out_p4, out_p5 = self._align_spatial_features(p3, p4, p5)
                    fused_features = torch.cat([out_p3, out_p4, out_p5], dim=1)  # [N, 448, 40, 40]
                    projected = self.hierarchical_conv_pyramid(fused_features)  # [N, out_dim]

                    # Tính trọng số attention (nếu được yêu cầu hoặc chế độ attention)
                    weights = None
                    if return_weights or self.fusion == "attention":
                        raw_weights = self.scale_attention(projected)  # [N, num_scales]
                        weights = F.softmax(raw_weights, dim=-1)

                # Khôi phục lại chiều thời gian nếu là chuỗi video 5D
                if is_5d:
                    projected = projected.view(orig_b, orig_t, self.out_dim)
                    if weights is not None:
                        weights = weights.view(orig_b, orig_t, self.num_scales)

                return (projected, weights) if return_weights else projected

            # Nếu đầu vào là tensor vector 2D [B, C] hoặc 3D [B, T, C] (tương thích ngược)
            elif p3.dim() in (2, 3):
                feat_cat = torch.cat([p3, p4, p5], dim=-1)  # [..., 448]
                projected = self.dropout(self.linear_project(feat_cat))
                weights = None
                if return_weights or self.fusion == "attention":
                    weights = F.softmax(self.scale_attention(projected), dim=-1)
                return (projected, weights) if return_weights else projected

            else:
                raise ValueError(f"Số chiều của tensor đặc trưng không hợp lệ: {p3.dim()}D.")

        # ======================================================================
        # TRƯỜNG HỢP 2: ĐẦU VÀO LÀ TENSOR ĐƠN LẺ
        # ======================================================================
        elif isinstance(features, torch.Tensor):
            feat = features
            is_5d = (feat.dim() == 5)
            orig_b, orig_t = 0, 0
            if is_5d:
                orig_b, orig_t = feat.shape[0], feat.shape[1]
                feat = feat.reshape(orig_b * orig_t, *feat.shape[2:])

            if feat.dim() == 4:
                # Nếu đã là tensor 448 kênh tại 40x40
                if feat.shape[1] == self.total_in_channels and feat.shape[-2:] == (40, 40):
                    projected = self.hierarchical_conv_pyramid(feat)
                else:
                    raise ValueError(
                        f"Kích thước tensor 4D ({list(feat.shape)}) không khớp [N, {self.total_in_channels}, 40, 40]."
                    )
            elif feat.dim() in (2, 3):
                if feat.shape[-1] == self.out_dim:
                    projected = self.dropout(feat)
                elif feat.shape[-1] == self.total_in_channels:
                    projected = self.dropout(self.linear_project(feat))
                else:
                    raise ValueError(
                        f"Kích thước kênh cuối ({feat.shape[-1]}) không khớp tổng kênh ({self.total_in_channels}) "
                        f"hoặc out_dim ({self.out_dim})."
                    )
            else:
                raise ValueError(f"Số chiều tensor đơn không hợp lệ: {feat.dim()}D.")

            weights = None
            if return_weights or self.fusion == "attention":
                weights = F.softmax(self.scale_attention(projected), dim=-1)

            if is_5d:
                projected = projected.view(orig_b, orig_t, self.out_dim)
                if weights is not None:
                    weights = weights.view(orig_b, orig_t, self.num_scales)

            return (projected, weights) if return_weights else projected

        else:
            raise TypeError(
                f"Định dạng features không hợp lệ: {type(features)}. Mong đợi Tuple/List hoặc torch.Tensor."
            )


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

    def forward(self, gru_out: torch.Tensor, seq_lens: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, torch.Tensor]:
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
            fill_value = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = scores.masked_fill(~mask, fill_value)

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
            input_dim: int = 512,
            hidden_dim: int = 256,
            num_layers: int = 2,
            num_classes: int = 2,
            spatial_in_channels: Tuple[int, ...] = (64, 128, 256),
            fusion: str = "concat",
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

        self.spatial_adapter = CNNAdapter(
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

    @classmethod
    def from_checkpoint(cls, checkpoint_path: Union[str, Path], map_location: str = "cpu") -> "DeepGRUClassifier":
        """Khởi tạo DeepGRUClassifier và nạp trọng số trực tiếp từ file checkpoint (.pth/.pt)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {path}")

        ckpt = torch.load(str(path), map_location=map_location)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        cfg_dict = ckpt.get("config", {})

        weight_key = "gru.weight_ih_l0" if "gru.weight_ih_l0" in state_dict else None
        if weight_key:
            input_dim = state_dict[weight_key].shape[1]
            hidden_dim = state_dict[weight_key].shape[0] // 3
        else:
            input_dim = int(cfg_dict.get("input_dim", 256))
            hidden_dim = int(cfg_dict.get("hidden_dim", 192))

        num_classes = int(cfg_dict.get("num_classes", 2))
        layer_indices = {int(k.split("weight_ih_l")[-1]) for k in state_dict if
                         "weight_ih_l" in k and k.split("weight_ih_l")[-1].isdigit()}
        num_layers = len(layer_indices) if layer_indices else int(cfg_dict.get("num_layers", 2))
        fusion = cfg_dict.get("spatial_fusion", "attention")
        spatial_in_channels = cfg_dict.get("cnn_neck_channels", (64, 128, 256))

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
        model.load_state_dict(state_dict, strict=False)
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
            seq_lens: Optional[torch.Tensor] = None,
            h_0: Optional[torch.Tensor] = None,
            return_sequence: bool = False,
            return_weights: bool = False,
            return_state: bool = False,
            *args: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        # Hỗ trợ truyền rời 3 tensor không gian (ví dụ model(p3, p4, p5))
        if isinstance(seq_lens, torch.Tensor) and seq_lens.dim() >= 4:
            extra = [seq_lens]
            if isinstance(h_0, torch.Tensor) and h_0.dim() >= 4:
                extra.append(h_0)
                h_0 = None
            if len(args) > 0:
                extra.extend(args)
            features = (features, *extra)
            seq_lens = None
        elif len(args) > 0 and isinstance(features, torch.Tensor):
            features = (features, *args)

        # 1. Chuyển đổi đặc trưng không gian
        x = self.spatial_adapter(features)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 2. Trích xuất đặc trưng chuỗi thời gian qua Deep GRU
        gru_out, h_n = self.gru(x, h_0)

        # 3. Phân loại theo cơ chế
        is_seq_mode = return_sequence or (self.supervision_mode in ("sequence", "frame"))
        if is_seq_mode:
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
