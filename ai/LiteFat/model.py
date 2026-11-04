import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights
from typing import Optional, Tuple


# =====================================================================
# 1. NHÁNH NGỮ CẢNH: MOBILENETV3-SMALL (Section III-C1)
# =====================================================================
class MobileNetV3ContextEncoder(nn.Module):
    """
    Trích xuất vector ngữ cảnh d in R^{D} cho mỗi khung hình khuôn mặt
    bằng xương sống MobileNetV3-Small.
    """

    def __init__(
            self,
            out_dim: int = 64,
            proj_hidden_dim: int = 192,
            pretrained: bool = True,
            dropout: float = 0.2
    ):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        base_model = mobilenet_v3_small(weights=weights)

        # Giữ lại khối trích xuất đặc trưng tích chập (927,008 tham số, đầu ra 576 kênh)
        self.features = base_model.features
        self.avgpool = base_model.avgpool

        # Tầng chiếu gọn nhẹ từ 576 -> proj_hidden_dim -> D
        self.projector = nn.Sequential(
            nn.Linear(576, proj_hidden_dim),
            nn.Hardswish(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(proj_hidden_dim, out_dim)
        )

    def forward(self, frame_tensor: torch.Tensor) -> torch.Tensor:
        """
        Args:
            frame_tensor: Ảnh khuôn mặt [B * T, 3, H, W] hoặc [B, 3, H, W]
        Returns:
            d: Vector ngữ cảnh [B * T, D] hoặc [B, D]
        """
        feat = self.features(frame_tensor)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        d = self.projector(feat)
        return d


# =====================================================================
# 2. MÔ-ĐUN DUNG HỢP ĐA PHƯƠNG THỨC (Section III-C2, Equation 1)
# =====================================================================
class MultimodalFusionModule(nn.Module):
    """
    Thực hiện phép dung hợp đa phương thức theo Phương trình (1):
        X = (C * w) * d^T
    Trong đó:
        - C in R^{N x 3}: Ma trận tọa độ điểm mốc (X, Y, Confidence)
        - w in R^{3 x 1}: Vector trọng số khả học
        - d in R^{D x 1}: Vector đặc trưng ngữ cảnh từ MobileNetV3-Small
        - X in R^{N x D}: Ma trận đặc trưng nút đồ thị tại mỗi khung hình
    """

    def __init__(self):
        super().__init__()
        # Vector trọng số khả học w in R^{3 x 1}
        self.w = nn.Parameter(torch.ones(3, 1) / 3.0)

    @staticmethod
    def handle_missing_landmarks(coords: torch.Tensor, conf_threshold: float = 1e-4) -> torch.Tensor:
        """
        Theo Section III-B: Nếu bộ dò điểm mốc thất bại ở một khung hình
        (khi tài xế cúi gập đầu hoặc quay mặt đi), gán toàn bộ phần tử
        của ma trận tọa độ C tại khung hình đó về 1.0.
        """
        # coords shape: [B, T, N, 3]
        conf_sum = coords[..., 2].abs().sum(dim=-1, keepdim=True).unsqueeze(-1)  # [B, T, 1, 1]
        invalid_mask = conf_sum < conf_threshold
        return torch.where(invalid_mask, torch.ones_like(coords), coords)

    def forward(self, coords: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Ma trận điểm mốc C có kích thước [B, T, N, 3]
            d: Vector ngữ cảnh MobileNetV3 có kích thước [B, T, D]
        Returns:
            X: Tensor đặc trưng đồ thị không gian - thời gian [B, D, N, T]
        """
        coords = self.handle_missing_landmarks(coords)

        # (C * w): [B, T, N, 3] @ [3, 1] -> [B, T, N, 1]
        node_scalar = torch.matmul(coords, self.w)

        # d^T: [B, T, 1, D]
        d_row = d.unsqueeze(2)

        # Ngoại tích X = (C * w) @ d^T -> [B, T, N, D]
        X = torch.matmul(node_scalar, d_row)

        # Chuyển trục về chuẩn Graph WaveNet: [B, D, N, T]
        return X.permute(0, 3, 2, 1).contiguous()


# =====================================================================
# 3. MA TRẬN KỀ TỰ THÍCH NGHI & TÍCH CHẬP ĐỒ THỊ (Section III-D1, Eq. 2 & 3)
# =====================================================================
class SelfAdaptiveAdjacency(nn.Module):
    """
    Học ma trận kề tự thích nghi đầu-cuối theo Phương trình (2):
        A_adp = softmax(ReLU(E1 @ E2^T))
    """

    def __init__(self, num_nodes: int = 68, embed_dim: int = 16):
        super().__init__()
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim) * 0.1)

    def forward(self) -> torch.Tensor:
        """
        Returns:
            A_adp: Ma trận kề tự thích nghi đã chuẩn hóa hàng [N, N]
        """
        spatial_weights = F.relu(torch.matmul(self.E1, self.E2.transpose(0, 1)))
        A_adp = F.softmax(spatial_weights, dim=-1)
        return A_adp


class SpatialGCNLayer(nn.Module):
    """
    Tầng Tích chập Đồ thị (GCN) xấp xỉ bậc nhất 2 lớp theo Phương trình (3):
        Z = F(X, A_adp) = softmax(A_adp * ReLU(A_adp * X * W^(0)) * W^(1))
    """

    def __init__(
            self,
            in_dim: int = 64,
            hidden_dim: int = 64,
            out_dim: int = 64,
            dropout: float = 0.2
    ):
        super().__init__()
        # Trọng số W^(0) in R^{D x H} và W^(1) in R^{H x M}
        self.W0 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.W1 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, A_adp: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor đầu vào [B, D, N, T]
            A_adp: Ma trận kề tự thích nghi [N, N]
        Returns:
            z: Tensor đầu ra sau tích chập đồ thị [B, M, N, T]
        """
        # Đưa về [B, T, N, D] để nhân ma trận trên chiều nút N và chiều đặc trưng D
        x_perm = x.permute(0, 3, 2, 1).contiguous()  # [B, T, N, D]

        # Lớp GCN thứ nhất: ReLU(A_adp @ X @ W^(0))
        h1 = self.W0(x_perm)  # [B, T, N, H]
        h1 = torch.einsum("ij,btnf->btif", A_adp, h1)  # Nhân A_adp [N, N]
        h1 = F.relu(h1)
        h1 = self.dropout(h1)

        # Lớp GCN thứ hai: softmax(A_adp @ h1 @ W^(1))
        h2 = self.W1(h1)  # [B, T, N, M]
        h2 = torch.einsum("ij,btnf->btif", A_adp, h2)
        z = F.softmax(h2, dim=-1)

        # Trả lại kích thước [B, M, N, T]
        return z.permute(0, 3, 2, 1).contiguous()


# =====================================================================
# 4. TÍCH CHẬP THỜI GIAN NHÂN QUẢ CÓ CỔNG (Section III-D2, Eq. 4 & 5)
# =====================================================================
class GatedDilatedCausalTCN(nn.Module):
    """
    Tầng Tích chập Thời gian Nhân quả Giãn nở có Cổng (Gated TCN)
    theo Phương trình (4) và (5):
        (x * f)(t) = sum_{s=0}^{k-1} f(s) x(t - d * s)
        h = tanh(Theta_1 * X + b) (x) sigmoid(Theta_2 * X + c)
    """

    def __init__(
            self,
            in_channels: int = 64,
            dilation_channels: int = 64,
            kernel_size: int = 2,
            dilation: int = 1
    ):
        super().__init__()
        # Đệm nhân quả (Causal Padding) ở phía bên trái trục thời gian T
        # để đảm bảo khung hình t chỉ phụ thuộc vào các khung hình <= t
        self.causal_padding = (kernel_size - 1) * dilation

        # Hai nhánh tích chập 1D dọc theo trục thời gian (TCN-a và TCN-b)
        self.theta_1 = nn.Conv2d(
            in_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            bias=True
        )
        self.theta_2 = nn.Conv2d(
            in_channels,
            dilation_channels,
            kernel_size=(1, kernel_size),
            dilation=(1, dilation),
            bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, D, N, T]
        Returns:
            h: Tensor [B, dilation_channels, N, T]
        """
        # Đệm bên trái trên chiều cuối cùng (chiều thời gian T)
        x_pad = F.pad(x, (self.causal_padding, 0, 0, 0))

        # Phương trình (5): h = tanh(Theta_1 * X + b) * sigmoid(Theta_2 * X + c)
        tcn_a = torch.tanh(self.theta_1(x_pad))
        tcn_b = torch.sigmoid(self.theta_2(x_pad))
        h = tcn_a * tcn_b
        return h


# =====================================================================
# 5. KHỐI KHÔNG GIAN - THỜI GIAN (STGL BLOCK / GRAPHWAVENET LAYER)
# =====================================================================
class STGLBlock(nn.Module):
    """
    Một tầng Không gian - Thời gian trong STGL (Section III-D3):
    Kết hợp Gated Dilated TCN (Eq. 5) + Spatial GCN 2 lớp (Eq. 3) +
    Kết nối tắt (Skip Connection) và Kết nối phần dư (Residual Connection).
    """

    def __init__(
            self,
            residual_channels: int = 64,
            dilation_channels: int = 64,
            skip_channels: int = 128,
            kernel_size: int = 2,
            dilation: int = 1,
            dropout: float = 0.2
    ):
        super().__init__()
        self.gated_tcn = GatedDilatedCausalTCN(
            in_channels=residual_channels,
            dilation_channels=dilation_channels,
            kernel_size=kernel_size,
            dilation=dilation
        )
        self.spatial_gcn = SpatialGCNLayer(
            in_dim=dilation_channels,
            hidden_dim=residual_channels,
            out_dim=residual_channels,
            dropout=dropout
        )
        self.skip_conv = nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1))
        self.bn = nn.BatchNorm2d(residual_channels)

    def forward(
            self,
            x: torch.Tensor,
            A_adp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [B, residual_channels, N, T]
            A_adp: [N, N]
        Returns:
            x_res: Đầu ra có cộng phần dư [B, residual_channels, N, T]
            skip: Đầu ra nhánh skip [B, skip_channels, N, T]
        """
        residual = x

        # 1. Mô hình hóa thời gian qua Gated TCN
        h_tcn = self.gated_tcn(x)

        # 2. Trích xuất kết nối skip cho tầng phân loại cuối
        skip = self.skip_conv(h_tcn)

        # 3. Mô hình hóa không gian qua Self-Adaptive Spatial GCN
        h_gcn = self.spatial_gcn(h_tcn, A_adp)

        # 4. Kết nối phần dư + Chuẩn hóa Batch
        x_res = self.bn(h_gcn + residual)
        return x_res, skip


# =====================================================================
# 6. KIẾN TRÚC TỔNG THỂ LITEFAT (Section III & IV)
# =====================================================================
class LiteFat(nn.Module):
    """
    Mô hình LiteFat hoàn chỉnh (IEEE/RSJ IROS 2025):
      - N = 68 điểm mốc khuôn mặt
      - T = 16 khung hình mỗi mẫu video
      - Tổng tham số mặc định: ~1.32M tham số (khớp Bảng II bài báo)
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
            num_classes: int = 2,
            num_blocks: int = 4,
            layers_per_block: int = 2,
            kernel_size: int = 2,
            pretrained_backbone: bool = True,
            dropout: float = 0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_classes = num_classes

        # Thành phần 1: MobileNetV3-Small trích xuất ngữ cảnh
        self.context_encoder = MobileNetV3ContextEncoder(
            out_dim=feature_dim,
            proj_hidden_dim=mobilenet_proj_hidden,
            pretrained=pretrained_backbone,
            dropout=dropout
        )

        # Thành phần 2: Dung hợp đa phương thức X = (C * w) * d^T
        self.fusion = MultimodalFusionModule()

        # Thành phần 3: Ma trận kề tự thích nghi A_adp
        self.adaptive_adj = SelfAdaptiveAdjacency(
            num_nodes=num_nodes,
            embed_dim=node_embed_dim
        )

        # Tầng chiếu đầu vào đồ thị
        self.start_conv = nn.Conv2d(feature_dim, residual_channels, kernel_size=(1, 1))

        # Thành phần 4: Các khối Không gian - Thời gian (STGL Blocks)
        # Hệ số giãn nở (dilation) tăng theo cấp số nhân (1, 2) lặp qua các khối
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
                        dropout=dropout
                    )
                )
                dilation *= 2

        # Đầu phân loại (Output Layer) dự đoán nhãn cho từng khung hình t in [1..T]
        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, num_classes, kernel_size=(1, 1))

    def forward_stgl_from_fused(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chạy riêng phần đồ thị STGL khi đã có tensor đặc trưng X [B, D, N, T].
        """
        A_adp = self.adaptive_adj()
        h = self.start_conv(X)

        skip_sum = None
        for block in self.stgl_blocks:
            h, skip = block(h, A_adp)
            skip_sum = skip if skip_sum is None else (skip_sum + skip)

        out = F.relu(skip_sum)
        out = F.relu(self.end_conv_1(out))
        out = self.end_conv_2(out)  # [B, num_classes, N, T]

        # Gộp trung bình trên toàn bộ N=68 nút để ra dự đoán cấp khung hình
        frame_logits = out.mean(dim=2).permute(0, 2, 1).contiguous()  # [B, T, num_classes]
        clip_logits = frame_logits.mean(dim=1)  # [B, num_classes]
        return frame_logits, clip_logits

    def forward(
            self,
            video_frames: torch.Tensor,
            landmarks: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chế độ chuẩn (Huấn luyện & Đánh giá đồng bộ):
        Args:
            video_frames: Chuỗi khung hình [B, T, 3, H, W] (ví dụ H=W=224)
            landmarks: Chuỗi điểm mốc khuôn mặt [B, T, N, 3] (N=68)
        Returns:
            frame_logits: Điểm dự đoán cho từng khung hình [B, T, num_classes]
            clip_logits: Điểm dự đoán cho cả đoạn video [B, num_classes]
        """
        B, T, C_img, H, W = video_frames.shape

        # 1. Trích xuất vector ngữ cảnh d cho toàn bộ B * T khung hình
        frames_flat = video_frames.view(B * T, C_img, H, W)
        d_flat = self.context_encoder(frames_flat)  # [B * T, D]
        d = d_flat.view(B, T, self.feature_dim)  # [B, T, D]

        # 2. Dung hợp đa phương thức: X = (C * w) * d^T -> [B, D, N, T]
        X = self.fusion(landmarks, d)

        # 3. Đưa qua mạng đồ thị không gian - thời gian STGL
        return self.forward_stgl_from_fused(X)

    @torch.no_grad()
    def forward_asymmetric_step(
            self,
            landmark_window: torch.Tensor,
            d_history: torch.Tensor,
            current_frame: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Chế độ Bất đối xứng (Asymmetric Inference) tối ưu cho Snapdragon 680:
          - Ở khung hình cần cập nhật ngữ cảnh (mỗi 3 frame): truyền `current_frame` [1, 3, H, W]
            để chạy MobileNetV3-Small và đẩy vector d_t mới vào `d_history`.
          - Ở các khung hình xen kẽ: đặt `current_frame=None`, tái sử dụng vector d gần nhất
            nhưng vẫn cập nhật `landmark_window` [1, T, 68, 3] ở tốc độ 30 FPS.
        """
        if current_frame is not None:
            d_new = self.context_encoder(current_frame).unsqueeze(1)  # [1, 1, D]
        else:
            d_new = d_history[:, -1:, :]  # Tái sử dụng d_{t-1}

        # Trượt cửa sổ lịch sử vector ngữ cảnh d_history [1, T, D]
        d_updated = torch.cat([d_history[:, 1:, :], d_new], dim=1)

        # Dung hợp với cửa sổ điểm mốc 30 FPS và chạy đồ thị STGL
        X = self.fusion(landmark_window, d_updated)
        _, clip_logits = self.forward_stgl_from_fused(X)
        probs = F.softmax(clip_logits, dim=-1)
        return probs, d_updated


# =====================================================================
# 8. KIỂM TRA SỐ THAM SỐ, LUỒNG DỮ LIỆU & VÒNG LẶP HUẤN LUYỆN (Section IV)
# =====================================================================
if __name__ == "__main__":
    from loss import LiteFatLoss

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Khởi tạo mô hình cho bài toán nhị phân trên YawDD (2 lớp: Bình thường vs. Ngáp)
    # Hoặc đặt num_classes=3 cho bài toán 3 lớp (Bình thường, Ngáp, Nói chuyện)
    model = LiteFat(
        num_nodes=68,
        feature_dim=64,
        mobilenet_proj_hidden=192,
        node_embed_dim=16,
        num_classes=2,
        pretrained_backbone=False
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Tổng số tham số của LiteFat: {total_params:,} (~{total_params / 1e6:.2f}M)")

    # Giả lập 1 batch gồm B=2 đoạn video, mỗi đoạn T=16 khung hình (Section IV-A)
    B, T, N = 2, 16, 68
    dummy_frames = torch.randn(B, T, 3, 224, 224, device=device)
    dummy_landmarks = torch.rand(B, T, N, 3, device=device)
    dummy_labels = torch.tensor([0, 1], dtype=torch.long, device=device)

    # Thiết lập bộ tối ưu hóa đúng theo Section IV-A: Adam, lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = LiteFatLoss()

    # 1 bước huấn luyện (Forward + Backward)
    model.train()
    optimizer.zero_grad()
    frame_logits, clip_logits = model(dummy_frames, dummy_landmarks)
    loss = criterion(frame_logits, dummy_labels)
    loss.backward()
    optimizer.step()

    print(f"Kích thước frame_logits: {frame_logits.shape} | clip_logits: {clip_logits.shape}")
    print(f"Giá trị Loss (Eq. 6): {loss.item():.4f}")

    # Kiểm thử chế độ Bất đối xứng (Asymmetric Step) cho Snapdragon 680
    model.eval()
    d_cache = torch.zeros(1, T, 64, device=device)
    lm_window = torch.rand(1, T, N, 3, device=device)
    single_frame = torch.randn(1, 3, 224, 224, device=device)

    # Frame 0: Cập nhật cả MobileNetV3 và Đồ thị STGL
    probs_f0, d_cache = model.forward_asymmetric_step(lm_window, d_cache, current_frame=single_frame)
    # Frame 1: Bỏ qua MobileNetV3 (current_frame=None), chỉ chạy Đồ thị STGL
    probs_f1, d_cache = model.forward_asymmetric_step(lm_window, d_cache, current_frame=None)
    print(f"Xác suất dự đoán thời gian thực (Frame 1 - Fast Path): {probs_f1.cpu().numpy()}")
