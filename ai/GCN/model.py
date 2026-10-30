# File: model.py
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
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        num_nodes: int = 68,
        hidden_dims: list = [64, 128, 256],
        strides: list = [1, 2, 2],
        dropout: float = 0.25
    ):
        super().__init__()
        assert len(hidden_dims) == len(strides), "hidden_dims và strides phải có cùng độ dài."

        # 1. Khởi tạo ma trận kề tiên nghiệm từ block.py
        adj_prior = FacialGraphBuilder.get_dense_adj(num_nodes=num_nodes)

        # 2. Khối trích xuất động học & Chuẩn hóa BatchNorm đầu vào trên từng điểm mốc
        self.kinematics = KinematicFeatureExtractor()
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
                    stride=stride,
                    dropout=dropout
                )
            )
            curr_dim = out_dim
        self.backbone = nn.ModuleList(blocks)

        # 4. Đầu phân loại (Classification Head) dùng các khối có sẵn của PyTorch
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(curr_dim),
            nn.Dropout(dropout),
            nn.Linear(curr_dim, curr_dim // 2),
            nn.GELU(),
            nn.Linear(curr_dim // 2, num_classes)
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
    """
    def __init__(
        self,
        in_channels: int = 6,
        num_classes: int = 3,
        num_nodes: int = 68,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        max_len: int = 300,
        dropout: float = 0.2
    ):
        super().__init__()
        adj_prior = FacialGraphBuilder.get_dense_adj(num_nodes=num_nodes)

        self.kinematics = KinematicFeatureExtractor()
        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 1. Tầng ST-GCN rút trích đặc trưng đồ thị khuôn mặt
        self.stgcn_stem = nn.Sequential(
            STGCNBlock(in_channels, 64, adj_prior, stride=1, dropout=dropout),
            STGCNBlock(64, d_model, adj_prior, stride=2, dropout=dropout)
        )

        # 2. Trọng số chú ý không gian (Spatial Node Attention Pooling) gộp V=68 đỉnh về 1 vector/frame
        self.node_attention = nn.Sequential(
            nn.Conv1d(d_model, 1, kernel_size=1),
            nn.Softmax(dim=-1)
        )

        # 3. Khối TransformerEncoder chuẩn của PyTorch dọc trục thời gian T
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len + 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Thử nghiệm với đầu vào chỉ có 2 kênh tọa độ (x, y): Shape = (Batch=4, C=2, T=60, V=68)
    # KinematicFeatureExtractor sẽ tự động mở rộng lên 6 kênh [x, y, dx, dy, d2x, d2y]
    raw_landmarks = torch.randn(4, 2, 60, 68).to(device)

    # 1. Test DriverSTGCN
    model_light = DriverSTGCN(in_channels=6, num_classes=3).to(device)
    out_light = model_light(raw_landmarks)
    params_light = sum(p.numel() for p in model_light.parameters() if p.requires_grad)

    # 2. Test DriverSTGCNTransformer
    model_trans = DriverSTGCNTransformer(in_channels=6, num_classes=3).to(device)
    out_trans = model_trans(raw_landmarks)
    params_trans = sum(p.numel() for p in model_trans.parameters() if p.requires_grad)

    print(f"[DriverSTGCN]            Output shape: {out_light.shape} | Params: {params_light:,}")
    print(f"[DriverSTGCNTransformer] Output shape: {out_trans.shape} | Params: {params_trans:,}")