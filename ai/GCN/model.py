# File: model.py
from typing import List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from block import (
    FacialGraphBuilder,
    KinematicFeatureExtractor,
    STGCNBlock,
    AttentiveDrowsinessOutputHead
)


# Cập nhật trong file: model.py
class DriverSTGCN(nn.Module):
    def __init__(
            self,
            in_channels: int = 6,
            num_classes: int = 3,
            num_nodes: int = 68,
            num_domains: int = 0,  # > 0 để bật Domain Generalization (MSTGCN)
            enable_regression: bool = False,  # True để bật dự đoán điểm buồn ngủ liên tục
            hidden_dims: list = [64, 128, 256],
            strides: list = [1, 2, 2],
            dropout: float = 0.25
    ):
        super().__init__()
        adj_prior = FacialGraphBuilder.get_dense_adj(num_nodes=num_nodes)
        self.kinematics = KinematicFeatureExtractor()
        self.input_bn = nn.BatchNorm1d(in_channels * num_nodes)

        blocks = []
        curr_dim = in_channels
        for out_dim, stride in zip(hidden_dims, strides):
            blocks.append(STGCNBlock(curr_dim, out_dim, adj_prior, stride=stride, dropout=dropout))
            curr_dim = out_dim
        self.backbone = nn.ModuleList(blocks)

        # Sử dụng Khối Dự đoán Đầu ra Tối ưu mới
        self.output_head = AttentiveDrowsinessOutputHead(
            in_channels=curr_dim,
            num_classes=num_classes,
            num_nodes=num_nodes,
            num_domains=num_domains,
            enable_regression=enable_regression,
            dropout=dropout
        )

    def forward(self, x: torch.Tensor, return_domain: bool = False, return_probs: bool = False):
        x = self.kinematics(x)
        B, C, T, V = x.shape

        x = x.permute(0, 1, 3, 2).contiguous().view(B, C * V, T)
        x = self.input_bn(x)
        x = x.view(B, C, V, T).permute(0, 1, 3, 2).contiguous()

        for block in self.backbone:
            x = block(x)

        return self.output_head(x, return_domain=return_domain, return_probs=return_probs)


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
        attn_weights = self.node_attention(x_flat)  # (B*T', 1, V)
        x_spatial_pooled = torch.sum(x_flat * attn_weights, dim=-1)  # (B*T', D)
        seq = x_spatial_pooled.view(B, T_prime, D)  # (B, T', D)

        # Ghép [CLS] token và Positional Embedding rồi đưa qua TransformerEncoder
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        seq = torch.cat([cls_tokens, seq], dim=1)  # (B, T'+1, D)

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
        is_pad = (curr_mask.squeeze(1).squeeze(-1) <= 0.5)  # (B, T_prime) bool
        padding_mask[:, 1:1 + T_prime] = is_pad

        encoded = self.transformer(seq, src_key_padding_mask=padding_mask)  # (B, T'+1, D)
        cls_out = encoded[:, 0]  # Lấy đặc trưng tại [CLS] token

        return self.head(cls_out)


# =====================================================================
# KIỂM THỬ CẢ 2 MODEL KHI CHẠY TRỰC TIẾP FILE model.py
# =====================================================================
if __name__ == "__main__":
    pass
