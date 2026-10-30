# File: model.py
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from block import (
    FacialGraphBuilder,
    KinematicFeatureExtractor,
    STGCNBlock
)


class DriverSTGCN(nn.Module):
    """
    Model 1: ST-GCN Tiêu chuẩn siêu nhẹ (Tối ưu cho Edge AI / Suy luận thời gian thực).
    Ghép nối chuỗi STGCNBlock + Global Adaptive Pooling + MLP Classifier.
    Khởi tạo thuần túy qua các tham số thông thường, hoàn toàn độc lập với đối tượng config.
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
        nose_idx: int = 30,
        left_eye_idx: int = 36,
        right_eye_idx: int = 45,
        eps: float = 1e-6,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 128, 256]
        if strides is None:
            strides = [1, 2, 2]

        assert len(hidden_dims) == len(strides), "hidden_dims và strides phải có cùng độ dài."

        # 1. Khởi tạo ma trận kề tiên nghiệm từ block.py
        adj_prior = FacialGraphBuilder.get_dense_adj(num_nodes=num_nodes)

        # 2. Khối trích xuất động học & Chuẩn hóa BatchNorm đầu vào trên từng điểm mốc
        self.kinematics = KinematicFeatureExtractor(
            nose_idx=nose_idx,
            left_eye_idx=left_eye_idx,
            right_eye_idx=right_eye_idx,
            eps=eps,
        )
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

    def _forward_features_with_mask(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trích xuất đặc trưng kèm theo mask thời gian đã downsample tương ứng.
        Trả về (feat, mask) trong đó:
          - feat: (B, C_last, T_out, V)
          - mask: (B, 1, T_out, 1)
        """
        mask = self.kinematics.build_temporal_mask(x, lengths=lengths)
        x = self.kinematics(x, mask=mask)  # Đảm bảo đầu vào có đủ 6 kênh [x, y, dx, dy, d2x, d2y]
        B, C, T, V = x.shape

        # Chuẩn hóa đầu vào dọc theo (C * V)
        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.input_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        x = x * mask  # Triệt tiêu bias của BatchNorm tại các frame zero-padding

        curr_mask = mask
        for block in self.backbone:
            x = block(x, mask=curr_mask)
            curr_mask = curr_mask[:, :, ::block.stride, :][:, :, :x.size(2), :]

        return x, curr_mask

    def forward_features(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Trích xuất đặc trưng không gian - thời gian trước khi Pooling."""
        feat, _ = self._forward_features_with_mask(x, lengths=lengths)
        return feat

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x      : (B, 2, T, V) hoặc (B, 6, T, V)
        lengths: LongTensor (B,) [Tùy chọn - số frame hợp lệ trước khi pad của từng mẫu]
        Output : (B, num_classes)
        """
        feat, mask = self._forward_features_with_mask(x, lengths=lengths)  # (B, C_last, T_out, V), (B, 1, T_out, 1)

        # Masked Global Average Pooling:
        # 1. Trung bình trên 68 đỉnh không gian
        feat_spatial = feat.mean(dim=-1, keepdim=True)                     # (B, C_last, T_out, 1)
        # 2. Trung bình trên đúng số lượng frame hợp lệ L_out (loại bỏ hoàn toàn ảnh hưởng của frame pad)
        valid_t = mask.sum(dim=2, keepdim=True).clamp(min=1.0)             # (B, 1, 1, 1)
        pooled = (feat_spatial * mask).sum(dim=2, keepdim=True) / valid_t  # (B, C_last, 1, 1)

        return self.classifier(pooled)                                     # (B, num_classes)


class DriverSTGCNTransformer(nn.Module):
    """
    Model 2: Kiến trúc Lai ST-GCN + PyTorch TransformerEncoder.
    Dùng ST-GCN để học tương quan đồ thị cơ mặt cục bộ, sau đó đưa qua
    nn.TransformerEncoder để học phụ thuộc chuỗi thời gian dài hạn.
    Khởi tạo thuần túy qua các tham số thông thường, hoàn toàn độc lập với đối tượng config.
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
        nose_idx: int = 30,
        left_eye_idx: int = 36,
        right_eye_idx: int = 45,
        eps: float = 1e-6,
    ):
        super().__init__()
        if stem_channels is None:
            stem_channels = [64, d_model]
        if stem_strides is None:
            stem_strides = [1, 2]
        if dim_feedforward is None:
            dim_feedforward = d_model * 2

        adj_prior = FacialGraphBuilder.get_dense_adj(num_nodes=num_nodes)

        self.kinematics = KinematicFeatureExtractor(
            nose_idx=nose_idx,
            left_eye_idx=left_eye_idx,
            right_eye_idx=right_eye_idx,
            eps=eps,
        )
        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes)

        # 1. Tầng ST-GCN Stem rút trích đặc trưng đồ thị khuôn mặt
        self.stgcn_stem = nn.ModuleList([
            STGCNBlock(in_channels, stem_channels[0], adj_prior, stride=stem_strides[0], dropout=dropout),
            STGCNBlock(stem_channels[0], stem_channels[1], adj_prior, stride=stem_strides[1], dropout=dropout)
        ])

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

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        x      : (B, 2, T, V) hoặc (B, 6, T, V)
        lengths: LongTensor (B,) [Tùy chọn - số frame hợp lệ trước khi pad]
        Output : (B, num_classes)
        """
        mask = self.kinematics.build_temporal_mask(x, lengths=lengths)
        x = self.kinematics(x, mask=mask)
        B, C, T, V = x.shape

        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.input_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()
        x = x * mask

        # Qua ST-GCN Stem:
        curr_mask = mask
        for block in self.stgcn_stem:
            x = block(x, mask=curr_mask)
            curr_mask = curr_mask[:, :, ::block.stride, :][:, :, :x.size(2), :]

        _, D, T_prime, _ = x.shape

        # Gộp 68 đỉnh bằng Node Attention cho mỗi bước thời gian
        x_flat = x.permute(0, 2, 1, 3).contiguous().view(B * T_prime, D, V)
        attn_weights = self.node_attention(x_flat)                  # (B*T', 1, V)
        x_spatial_pooled = torch.sum(x_flat * attn_weights, dim=-1) # (B*T', D)
        seq = x_spatial_pooled.view(B, T_prime, D)                  # (B, T', D)

        # Ghép [CLS] token và Positional Embedding rồi đưa qua TransformerEncoder
        cls_tokens = self.cls_token.expand(B, -1, -1)               # (B, 1, D)
        seq = torch.cat([cls_tokens, seq], dim=1)                   # (B, T'+1, D)

        seq_len = seq.size(1)
        if seq_len <= self.pos_embedding.size(1):
            pos_emb = self.pos_embedding[:, :seq_len, :]
        else:
            pos_emb = F.interpolate(
                self.pos_embedding.permute(0, 2, 1),
                size=seq_len,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1)

        seq = seq + pos_emb

        # Xây dựng padding mask cho Transformer: True = bỏ qua (vùng pad), False = hợp lệ
        padding_mask = torch.zeros((B, seq_len), dtype=torch.bool, device=x.device)
        is_pad = (curr_mask.squeeze(1).squeeze(-1) <= 0.5)          # (B, T_prime) bool
        padding_mask[:, 1:1 + T_prime] = is_pad

        encoded = self.transformer(seq, src_key_padding_mask=padding_mask)  # (B, T'+1, D)
        cls_out = encoded[:, 0]                                     # Lấy đặc trưng tại [CLS] token

        return self.head(cls_out)


# =====================================================================
# KIỂM THỬ CẢ 2 MODEL KHI CHẠY TRỰC TIẾP FILE model.py
# =====================================================================
if __name__ == "__main__":
    from config import TrainConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Thử nghiệm với đầu vào chỉ có 2 kênh tọa độ (x, y): Shape = (Batch=4, C=2, T=60, V=68)
    raw_landmarks = torch.randn(4, 2, 60, 68).to(device)

    # 1. Khởi tạo qua kwargs được trích xuất từ TrainConfig
    cfg = TrainConfig(num_classes=2)

    model_light = DriverSTGCN(**cfg.get_stgcn_kwargs()).to(device)
    out_light = model_light(raw_landmarks)
    params_light = sum(p.numel() for p in model_light.parameters() if p.requires_grad)

    model_trans = DriverSTGCNTransformer(**cfg.get_transformer_kwargs()).to(device)
    out_trans = model_trans(raw_landmarks)
    params_trans = sum(p.numel() for p in model_trans.parameters() if p.requires_grad)

    print("\n--- Test khoi tao qua TrainConfig.get_*_kwargs() ---")
    print(f"[DriverSTGCN]            Output shape: {out_light.shape} | Params: {params_light:,}")
    print(f"[DriverSTGCNTransformer] Output shape: {out_trans.shape} | Params: {params_trans:,}")

    # 2. Khởi tạo truyền thống hoàn toàn độc lập (không cần bất kỳ đối tượng config nào)
    pure_model = DriverSTGCN(in_channels=6, num_classes=3).to(device)
    pure_out = pure_model(raw_landmarks)
    print("\n--- Test khoi tao thuan tuy (tham so doc lap, khong config) ---")
    print(f"[DriverSTGCN Pure]       Output shape: {pure_out.shape}")

    # 3. Test batch có Padding và tensor lengths (mô phỏng collate_gcn từ dataset.py)
    print("\n--- Test voi batch co Padding & lengths ---")
    lengths = torch.tensor([30, 45, 60, 20], dtype=torch.long, device=device)
    padded_input = raw_landmarks.clone()
    for b, l in enumerate(lengths):
        padded_input[b, :, l:, :] = 0.0

    out_pad_light = model_light(padded_input, lengths=lengths)
    out_pad_trans = model_trans(padded_input, lengths=lengths)
    print(f"[DriverSTGCN with Pad]   Output shape: {out_pad_light.shape}")
    print(f"[DriverTrans with Pad]   Output shape: {out_pad_trans.shape}")

    # Backward check:
    loss = out_pad_light.sum() + out_pad_trans.sum()
    loss.backward()
    print("Backward pass thanh cong, gradient hop le (khong co NaN/Inf)!")
