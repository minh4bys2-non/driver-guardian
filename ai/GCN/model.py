# File: model.py
from typing import List, Optional, Any
import torch
import torch.nn as nn
from block import (
    FacialGraphBuilder,
    KinematicFeatureExtractor,
    STGCNBlock
)


class DriverSTGCN(nn.Module):
    """
    Model 1: ST-GCN Tiêu chuẩn siêu nhẹ (Tối ưu cho Edge AI / Suy luận thời gian thực).
    Ghép nối chuỗi STGCNBlock + Global Adaptive Pooling + MLP Classifier.
    Hỗ trợ khởi tạo qua tham số rời rạc hoặc truyền trực tiếp đối tượng config (hoặc **dict).
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        num_nodes: int = 68,
        hidden_dims: Optional[List[int]] = None,
        strides: Optional[List[int]] = None,
        dropout: float = 0.25,
        temporal_kernel: int = 5,
        gat_heads: int = 4,
        classifier_hidden_dim: Optional[int] = None,
        config: Optional[Any] = None,
    ):
        super().__init__()
        # Nếu truyền đối tượng config, tự động trích xuất các giá trị
        if config is not None:
            in_channels = getattr(config, "in_channels", in_channels)
            num_classes = getattr(config, "num_classes", num_classes)
            num_nodes = getattr(config, "num_nodes", num_nodes)
            hidden_dims = getattr(config, "hidden_dims", hidden_dims)
            strides = getattr(config, "strides", strides)
            dropout = getattr(config, "dropout", dropout)
            temporal_kernel = getattr(config, "temporal_kernel", temporal_kernel)
            gat_heads = getattr(config, "gat_heads", gat_heads)
            classifier_hidden_dim = getattr(config, "classifier_hidden_dim", classifier_hidden_dim)

        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        if strides is None:
            strides = [1, 2, 2]

        assert len(hidden_dims) == len(strides), "hidden_dims và strides phải có cùng độ dài."

        # 1. Khởi tạo ma trận kề tiên nghiệm từ block.py
        adj_prior = FacialGraphBuilder.get_dense_adj(num_nodes=num_nodes)

        # 2. Khối trích xuất động học & Chuẩn hóa BatchNorm đầu vào trên từng điểm mốc
        self.kinematics = KinematicFeatureExtractor(config=config)
        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 3. Chuỗi các khối ST-GCN (dùng nn.ModuleList)
        blocks = []
        curr_dim = in_channels
        for out_dim, stride in zip(hidden_dims, strides):
            blocks.append(
                STGCNBlock(
                    in_channels=curr_dim,
                    out_channels=out_dim,
                    adj_prior=adj_prior,
                    temporal_kernel=temporal_kernel,
                    stride=stride,
                    dropout=dropout,
                    gat_heads=gat_heads,
                )
            )
            curr_dim = out_dim
        self.backbone = nn.ModuleList(blocks)

        # 4. Đầu phân loại (Classification Head)
        cls_mid = classifier_hidden_dim if classifier_hidden_dim is not None else (curr_dim // 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(curr_dim),
            nn.Dropout(dropout),
            nn.Linear(curr_dim, cls_mid),
            nn.GELU(),
            nn.Linear(cls_mid, num_classes)
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Trích xuất đặc trưng không gian - thời gian trước khi Pooling."""
        x = self.kinematics(x)  # Đảm bảo đầu vào có đủ 6 kênh [x, y, dx, dy, d2x, d2y]
        B, C, T, V = x.shape

        # Chuẩn hóa đầu vào dọc theo (C * V)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.input_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()

        for block in self.backbone:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 2, T, V) hoặc (B, 6, T, V)
        Output : (B, num_classes)
        """
        feat = self.forward_features(x)         # (B, C_last, T_out, V)
        pooled = self.global_pool(feat)         # (B, C_last, 1, 1)
        return self.classifier(pooled)          # (B, num_classes)


class DriverSTGCNTransformer(nn.Module):
    """
    Model 2: Kiến trúc Lai ST-GCN + PyTorch TransformerEncoder.
    Dùng ST-GCN để học tương quan đồ thị cơ mặt cục bộ, sau đó đưa qua
    nn.TransformerEncoder để học phụ thuộc chuỗi thời gian dài hạn.
    Hỗ trợ khởi tạo qua tham số rời rạc hoặc truyền trực tiếp đối tượng config (hoặc **dict).
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 2,
        num_nodes: int = 68,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        max_len: int = 300,
        dropout: float = 0.2,
        stem_channels: Optional[List[int]] = None,
        stem_strides: Optional[List[int]] = None,
        dim_feedforward: Optional[int] = None,
        init_std: float = 0.02,
        config: Optional[Any] = None,
    ):
        super().__init__()
        # Nếu truyền đối tượng config, tự động trích xuất các giá trị
        if config is not None:
            in_channels = getattr(config, "in_channels", in_channels)
            num_classes = getattr(config, "num_classes", num_classes)
            num_nodes = getattr(config, "num_nodes", num_nodes)
            d_model = getattr(config, "d_model", d_model)
            nhead = getattr(config, "nhead", nhead)
            num_transformer_layers = getattr(config, "num_transformer_layers", num_transformer_layers)
            max_len = getattr(config, "max_len", max_len)
            dropout = getattr(config, "dropout", dropout)
            stem_channels = getattr(config, "stem_channels", stem_channels)
            stem_strides = getattr(config, "stem_strides", stem_strides)
            dim_feedforward = getattr(config, "dim_feedforward", dim_feedforward)
            init_std = getattr(config, "init_std", init_std)

        if stem_channels is None:
            stem_channels = [64, d_model]
        if stem_strides is None:
            stem_strides = [1, 2]
        if dim_feedforward is None:
            dim_feedforward = d_model * 2

        adj_prior = FacialGraphBuilder.get_dense_adj(num_nodes=num_nodes)

        self.kinematics = KinematicFeatureExtractor(config=config)
        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 1. Tầng ST-GCN Stem rút trích đặc trưng đồ thị khuôn mặt
        self.stgcn_stem = nn.Sequential(
            STGCNBlock(in_channels, stem_channels[0], adj_prior, stride=stem_strides[0], dropout=dropout),
            STGCNBlock(stem_channels[0], stem_channels[1], adj_prior, stride=stem_strides[1], dropout=dropout)
        )

        # 2. Trọng số chú ý không gian (Spatial Node Attention Pooling) gộp V=68 đỉnh về 1 vector/frame
        self.node_attention = nn.Sequential(
            nn.Conv1d(stem_channels[-1], 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        # 3. Khối TransformerEncoder chuẩn của PyTorch dọc trục thời gian T
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * init_std)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, d_model) * init_std)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # 4. Đầu dự đoán trạng thái buồn ngủ
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, 2, T, V) hoặc (B, 6, T, V)
        Output : (B, num_classes)
        """
        x = self.kinematics(x)
        B, C, T, V = x.shape

        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.input_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()

        # Qua ST-GCN: (B, d_model, T'=T/2, V=68)
        x = self.stgcn_stem(x)
        _, D, T_prime, _ = x.shape

        # Gộp 68 đỉnh bằng Node Attention cho mỗi bước thời gian
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * T_prime, D, V)
        attn_weights = self.node_attention(x_flat)                  # (B*T', 1, V)
        x_spatial_pooled = torch.sum(x_flat * attn_weights, dim=-1) # (B*T', D)
        seq = x_spatial_pooled.view(B, T_prime, D)                  # (B, T', D)

        # Ghép [CLS] token và Positional Embedding rồi đưa qua TransformerEncoder
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat([cls_tokens, seq], dim=1)                   # (B, T'+1, D)
        seq = seq + self.pos_embedding[:, : seq.size(1), :]

        encoded = self.transformer(seq)                             # (B, T'+1, D)
        cls_out = encoded[:, 0]                                     # Lấy đặc trưng tại [CLS] token

        return self.head(cls_out)


# =====================================================================
# KIỂM THỬ CẢ 2 MODEL KHI CHẠY TRỰC TIẾP FILE model.py
# =====================================================================
if __name__ == "__main__":
    from config import get_default_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Thử nghiệm với đầu vào chỉ có 2 kênh tọa độ (x, y): Shape = (Batch=4, C=2, T=60, V=68)
    raw_landmarks = torch.randn(4, 2, 60, 68).to(device)

    # 1. Khởi tạo bằng đối tượng config (num_classes=2)
    cfg = get_default_config(num_classes=2)

    model_light = DriverSTGCN(config=cfg.stgcn).to(device)
    out_light = model_light(raw_landmarks)
    params_light = sum(p.numel() for p in model_light.parameters() if p.requires_grad)

    model_trans = DriverSTGCNTransformer(config=cfg.transformer).to(device)
    out_trans = model_trans(raw_landmarks)
    params_trans = sum(p.numel() for p in model_trans.parameters() if p.requires_grad)

    print("\n--- Test khoi tao qua config=cfg.xxx ---")
    print(f"[DriverSTGCN]            Output shape: {out_light.shape} | Params: {params_light:,}")
    print(f"[DriverSTGCNTransformer] Output shape: {out_trans.shape} | Params: {params_trans:,}")

    # 2. Khởi tạo truyền thống (tương thích ngược hoàn toàn)
    legacy_model = DriverSTGCN(in_channels=6, num_classes=3).to(device)
    legacy_out = legacy_model(raw_landmarks)
    print("\n--- Test khoi tao truyen thong (kwargs truyen tay) ---")
    print(f"[DriverSTGCN Legacy]     Output shape: {legacy_out.shape}")
