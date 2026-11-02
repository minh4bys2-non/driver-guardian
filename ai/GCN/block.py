# File: block.py
from typing import List, Tuple, Optional
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGATConv
from torch_geometric.utils import to_undirected, add_self_loops, to_dense_adj
from torch.autograd import Function


class FacialGraphBuilder:
    """
    Xây dựng cấu trúc đồ thị 68 điểm mốc khuôn mặt bằng torch_geometric.utils.
    Hỗ trợ xuất cả edge_index (Sparse) và Adjacency Matrix (Dense).
    """

    @staticmethod
    def get_edge_index(
            num_nodes: int = 68,
            vertical_pairs: Optional[List[Tuple[int, int]] | List[List[int]]] = None,
            cross_region_pairs: Optional[List[Tuple[int, int]] | List[List[int]]] = None,
    ) -> torch.Tensor:
        edges = []

        def add_chain(indices, closed=False):
            for i in range(len(indices) - 1):
                edges.append([indices[i], indices[i + 1]])
            if closed:
                edges.append([indices[-1], indices[0]])

        # 1. Cạnh giải phẫu cơ mặt (Anatomical Contours)
        add_chain(range(0, 17), closed=False)  # Viền hàm
        add_chain(range(17, 22), closed=False)  # Lông mày trái
        add_chain(range(22, 27), closed=False)  # Lông mày phải
        add_chain(range(27, 31), closed=False)  # Sống mũi
        add_chain(range(31, 36), closed=False)  # Cánh mũi
        add_chain(range(36, 42), closed=True)  # Mắt trái
        add_chain(range(42, 48), closed=True)  # Mắt phải
        add_chain(range(48, 60), closed=True)  # Môi ngoài
        add_chain(range(60, 68), closed=True)  # Môi trong

        # 2. Cạnh dọc đo độ mở Mắt (EAR) & Miệng (MAR)
        if vertical_pairs is None:
            vertical_pairs = [
                [37, 41], [38, 40], [43, 47], [44, 46],  # Mí mắt trên - dưới
                [50, 58], [51, 57], [52, 56], [61, 67], [62, 66], [63, 65]  # Môi trên - dưới
            ]

        # 3. Cạnh tiên nghiệm liên vùng (Mắt <-> Miệng, Mũi <-> Cằm)
        if cross_region_pairs is None:
            cross_region_pairs = [
                [37, 48], [41, 48], [38, 62], [40, 66],  # Mắt trái <-> Miệng
                [44, 54], [46, 54], [43, 62], [47, 66],  # Mắt phải <-> Miệng
                [30, 8], [30, 36], [30, 45], [27, 8]  # Chóp mũi <-> Cằm & Đuôi mắt
            ]

        all_edges = edges + [list(p) for p in vertical_pairs] + [list(p) for p in cross_region_pairs]
        edge_index = torch.tensor(all_edges, dtype=torch.long).t().contiguous()

        # Sử dụng hàm có sẵn của PyTorch Geometric để tạo đồ thị vô hướng và thêm self-loops
        edge_index = to_undirected(edge_index, num_nodes=num_nodes)
        edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        return edge_index

    @classmethod
    def get_dense_adj(
            cls,
            num_nodes: int = 68,
            vertical_pairs: Optional[List[Tuple[int, int]] | List[List[int]]] = None,
            cross_region_pairs: Optional[List[Tuple[int, int]] | List[List[int]]] = None,
    ) -> torch.Tensor:
        """Trả về ma trận kề Dense kích thước (V, V) từ edge_index."""
        edge_index = cls.get_edge_index(
            num_nodes=num_nodes,
            vertical_pairs=vertical_pairs,
            cross_region_pairs=cross_region_pairs,
        )
        return to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0)


class KinematicFeatureExtractor(nn.Module):
    """
    Khối tiền xử lý chuẩn hóa tọa độ 68 điểm mốc:
    1. Đưa tọa độ tĩnh (x, y) về Hệ tọa độ Cục bộ Tâm mũi (điểm 30) & chia cho khoảng cách 2 mắt (36 <-> 45).
    2. Tính Vận tốc (dx, dy) và Gia tốc (d2x, d2y) đã chuẩn hóa theo kích thước khuôn mặt
       (giữ nguyên được quỹ đạo rơi của đầu khi tài xế ngủ gật).
    Hỗ trợ temporal mask để triệt tiêu cú sốc số học tại biên zero-padding khi batch có độ dài thay đổi.
    """

    def __init__(
            self,
            nose_idx: int = 30,
            left_eye_idx: int = 36,
            right_eye_idx: int = 45,
            eps: float = 1e-6,
    ):
        super().__init__()
        self.nose_idx = nose_idx
        self.left_eye_idx = left_eye_idx
        self.right_eye_idx = right_eye_idx
        self.eps = eps

    @staticmethod
    def build_temporal_mask(
            x: torch.Tensor,
            lengths: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Tạo binary float mask kích thước (B, 1, T, 1) biểu diễn các frame hợp lệ (1.0) và frame đệm (0.0).
        - Nếu lengths được truyền vào: tạo mask chính xác từ lengths.
        - Nếu lengths is None: tự động nhận diện từ biên độ tọa độ (frame hợp lệ có tọa độ khác 0).
        """
        B, _, T, _ = x.shape
        device = x.device
        if lengths is not None:
            t_idx = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
            mask = (t_idx < lengths.to(device).unsqueeze(1)).float()  # (B, T)
            return mask.view(B, 1, T, 1)
        else:
            # Tự nhận diện frame hợp lệ (tổng tọa độ x, y khác 0)
            frame_activity = x[:, :2].abs().sum(dim=(1, 3), keepdim=True)  # (B, 1, T, 1)
            return (frame_activity > eps).float()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input : (B, 2, T, V=68) - Tọa độ pixel hoặc [0, 1] trong Hệ tọa độ Khung ảnh.
        mask  : (B, 1, T, 1)    - Float mask (1.0 = valid, 0.0 = padded)
        Output: (B, 6, T, V=68) - [x_norm, y_norm, dx_norm, dy_norm, d2x_norm, d2y_norm]
        """
        if x.size(1) == 6:
            return x * mask if mask is not None else x

        # 1. Tính khoảng cách giải phẫu chuẩn d_t (Inter-Ocular Distance giữa điểm 36 và 45)
        # eye_dist shape: (B, 1, T, 1)
        p_left_eye = x[:, :, :, self.left_eye_idx: self.left_eye_idx + 1]  # (B, 2, T, 1)
        p_right_eye = x[:, :, :, self.right_eye_idx: self.right_eye_idx + 1]  # (B, 2, T, 1)
        raw_iod = torch.norm(p_right_eye - p_left_eye, p=2, dim=1, keepdim=True)
        # Tránh chia cho eps=1e-6 tại các frame pad gây spike số học khổng lồ
        iod_scale = raw_iod.clamp(min=self.eps)

        # 2. Chuẩn hóa tọa độ hình thái tĩnh theo Tâm mũi (điểm 30) và tỷ lệ IOD
        nose_pos = x[:, :, :, self.nose_idx: self.nose_idx + 1]  # (B, 2, T, 1)
        pos_norm = (x - nose_pos) / iod_scale  # (B, 2, T, V)

        # 3. Tính Vận tốc (dx, dy) và Gia tốc (d2x, d2y) chuẩn hóa theo IOD
        # (Tính diff trên x trước khi trừ mũi để giữ được vận tốc gật đầu của cả khuôn mặt)
        vel_raw = torch.diff(x, n=1, dim=2)
        vel_raw = F.pad(vel_raw, (0, 0, 1, 0), mode='replicate')

        if mask is not None:
            # Triệt tiêu bước nhảy biên giữa frame hợp lệ cuối cùng và frame pad đầu tiên
            pair_valid_vel = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            pair_valid_vel = F.pad(pair_valid_vel, (0, 0, 1, 0), mode='constant', value=1.0)
            vel_raw = vel_raw * pair_valid_vel

        vel_norm = vel_raw / iod_scale  # (B, 2, T, V)

        acc_raw = torch.diff(vel_norm, n=1, dim=2)
        acc_raw = F.pad(acc_raw, (0, 0, 1, 0), mode='replicate')

        if mask is not None:
            pair_valid_acc = mask[:, :, 1:, :] * mask[:, :, :-1, :]
            pair_valid_acc = F.pad(pair_valid_acc, (0, 0, 1, 0), mode='constant', value=1.0)
            acc_raw = acc_raw * pair_valid_acc

        acc_norm = acc_raw  # (B, 2, T, V)

        # Ghép thành 6 kênh đặc trưng chuẩn hóa hoàn chỉnh
        out = torch.cat([pos_norm, vel_norm, acc_norm], dim=1)  # (B, 6, T, V)

        if mask is not None:
            out = out * mask

        return out


class PyGSpatialGraphConv(nn.Module):
    """
    Khối tích chập không gian tận dụng DenseGCNConv và DenseGATConv của PyTorch Geometric.
    - DenseGCNConv: Học trên ma trận tiên nghiệm A_prior + ma trận tham số tự học A_adaptive.
    - DenseGATConv: Tự động tính Attention động (A_dynamic) giữa các điểm mốc theo từng mẫu.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            adj_prior: torch.Tensor,
            gat_heads: int = 4,
            dropout: float = 0.1,
            adaptive_init_scale: float = 1e-5,
    ):
        super().__init__()
        num_nodes = adj_prior.size(0)
        self.register_buffer('adj_prior', adj_prior)

        # Ma trận kề tự học toàn cục A_adaptive
        self.adj_adaptive = nn.Parameter(torch.randn(num_nodes, num_nodes) * adaptive_init_scale)

        # 1. Nhánh GCN tĩnh + thích ứng (PyG DenseGCNConv tự động chuẩn hóa D^-1/2 A D^-1/2)
        self.gcn = DenseGCNConv(in_channels, out_channels, improved=False, bias=True)

        # 2. Nhánh Graph Attention động (PyG DenseGATConv)
        self.gat = DenseGATConv(
            in_channels,
            out_channels,
            heads=gat_heads,
            concat=False,
            dropout=dropout,
            bias=True
        )
        self.alpha = nn.Parameter(torch.zeros(1))  # Hệ số hòa trộn nhánh GAT động

        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, C_in, T, V) -> Output shape: (B, C_out, T, V)
        """
        B, C, T, V = x.shape

        # Chuyển chiều về dạng chuẩn của PyG Dense layers: (B * T, V, C_in)
        x_dense = x.permute(0, 2, 3, 1).contiguous().view(B * T, V, C)

        # Tổng hợp ma trận kề cho nhánh GCN
        adj_total = self.adj_prior + self.adj_adaptive

        # Thực thi DenseGCNConv và DenseGATConv (add_loop=False vì adj_prior đã có self-loops)
        out_gcn = self.gcn(x_dense, adj_total, add_loop=False)  # (B*T, V, C_out)
        out_gat = self.gat(x_dense, self.adj_prior, add_loop=False)  # (B*T, V, C_out)

        out = out_gcn + self.alpha * out_gat  # (B*T, V, C_out)

        # Khôi phục lại chiều (B, C_out, T, V)
        out = out.view(B, T, V, -1).permute(0, 3, 1, 2).contiguous()
        return self.act(self.bn(out))


class MultiScaleTemporalConv(nn.Module):
    """
    Khối Tích chập Thời gian Đa tỷ lệ (Multi-Scale TCN) sử dụng nn.Conv2d với Dilation khác nhau:
    - Nhánh 1 (Dilation = 1): Bắt chuyển động nhanh (chớp mắt).
    - Nhánh 2 (Dilation = 2): Bắt chuyển động kéo dài (ngáp, ngủ gật).
    """

    def __init__(
            self,
            channels: int,
            kernel_size: int = 5,
            stride: int = 1,
            dropout: float = 0.2,
            dilations: Tuple[int, int] = (1, 2),
    ):
        super().__init__()
        mid_channels = channels // 2
        d1, d2 = dilations
        pad_d1 = (kernel_size - 1) * d1 // 2
        pad_d2 = (kernel_size - 1) * d2 // 2

        self.branch1 = nn.Conv2d(
            channels, mid_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(pad_d1, 0),
            dilation=(d1, 1)
        )
        self.branch2 = nn.Conv2d(
            channels, channels - mid_channels,
            kernel_size=(kernel_size, 1),
            stride=(stride, 1),
            padding=(pad_d2, 0),
            dilation=(d2, 1)
        )

        self.bn = nn.BatchNorm2d(channels)
        self.drop = nn.Dropout(dropout, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self.drop(self.bn(out))


class STGCNBlock(nn.Module):
    """
    Khối ST-GCN hoàn chỉnh: PyGSpatialGraphConv -> MultiScaleTemporalConv -> Residual Connection.
    Hỗ trợ temporal mask để triệt tiêu bias rò rỉ tại các frame zero-padding.
    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            adj_prior: torch.Tensor,
            temporal_kernel: int = 5,
            stride: int = 1,
            dropout: float = 0.2,
            gat_heads: int = 4,
            adaptive_init_scale: float = 1e-5,
            dilations: Tuple[int, int] = (1, 2),
    ):
        super().__init__()
        self.stride = stride
        self.sgcn = PyGSpatialGraphConv(
            in_channels,
            out_channels,
            adj_prior,
            gat_heads=gat_heads,
            dropout=dropout,
            adaptive_init_scale=adaptive_init_scale,
        )
        self.tgcn = MultiScaleTemporalConv(
            out_channels,
            kernel_size=temporal_kernel,
            stride=stride,
            dropout=dropout,
            dilations=dilations,
        )

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x   : (B, C_in, T_in, V)
        mask: Optional (B, 1, T_in, 1) - Float mask (1.0 = valid, 0.0 = padded)
        """
        res = self.residual(x)
        x = self.sgcn(x)
        if mask is not None:
            # Triệt tiêu bias của GCN/GAT/BN ở vùng pad trước khi tích chập thời gian
            x = x * mask
        x = self.tgcn(x)
        out = self.act(x + res)
        if mask is not None:
            mask_out = mask[:, :, ::self.stride, :][:, :, :out.size(2), :]
            out = out * mask_out
        return out


class _GradientReversalFunction(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_grl: float):
        ctx.lambda_grl = lambda_grl
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.lambda_grl * grad_output, None


class GradientReversalLayer(nn.Module):
    """Lớp Đảo ngược Gradient (GRL) dùng cho Domain Generalization theo bài báo MSTGCN."""

    def __init__(self, lambda_grl: float = 1.0):
        super().__init__()
        self.lambda_grl = lambda_grl

    def set_lambda(self, progress: float, gamma: float = 10.0):
        """
        Tự động tăng dần lambda_grl từ 0.0 -> 1.0 theo tiến độ huấn luyện progress thuộc [0, 1]
        giúp mạng học ổn định ở những epoch đầu tiên.
        """
        self.lambda_grl = float(2.0 / (1.0 + math.exp(-gamma * progress)) - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.lambda_grl)


class AttentiveDrowsinessOutputHead(nn.Module):
    """
    Khối Dự đoán Đầu ra Tối ưu cho ST-GCN Phát hiện Buồn ngủ:
    1. Spatial Node Attention: Tự động tập trung vào các điểm mốc Mắt/Miệng, loại bỏ nhiễu viền hàm.
    2. Temporal Gated-Conv + Max & Attention Pooling: Kế thừa tầng Output Temporal Conv của Yu et al.
       kết hợp Max-Pooling để bắt khoảnh khắc chợp mắt/ngáp ngắn (Micro-sleep) trong chuỗi dài.
    3. Dual Heads (G_y & G_d): Hỗ trợ đồng thời Phân loại buồn ngủ (G_y), Phân loại miền đối kháng (G_d - Jia et al.)
       và Hồi quy mức độ mệt mỏi liên tục (KSS Score Regression).
    """

    def __init__(
            self,
            in_channels: int = 256,
            num_classes: int = 3,
            num_nodes: int = 68,
            num_domains: int = 0,  # Đặt > 0 nếu muốn bật nhánh Domain Classifier (G_d) của MSTGCN
            enable_regression: bool = False,  # Đặt True nếu muốn dự đoán thêm điểm buồn ngủ liên tục (KSS)
            dropout: float = 0.25
    ):
        super().__init__()
        self.num_domains = num_domains
        self.enable_regression = enable_regression

        # --- BƯỚC 1: GOM ĐẶC TRƯNG KHÔNG GIAN CÓ CHÚ Ý (Spatial Node Attention) ---
        # Học trọng số quan trọng cho từng điểm trong 68 điểm mốc khuôn mặt
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(in_channels // 4, 1, kernel_size=1),
            nn.Softmax(dim=-1)  # Chuẩn hóa trọng số dọc theo 68 đỉnh (dim V)
        )

        # --- BƯỚC 2: TINH CHỈNH & GOM ĐẶC TRƯNG THỜI GIAN (Theo Yu et al. + Attention) ---
        # Lớp Temporal Conv cuối cùng kèm cổng GLU lọc nhiễu chuỗi thời gian
        self.final_temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=3, padding=1),
            nn.GLU(dim=1),
            nn.BatchNorm1d(in_channels)
        )
        # Trọng số chú ý thời gian (Temporal Attention) bắt khung hình buồn ngủ
        self.temporal_attn = nn.Sequential(
            nn.Conv1d(in_channels, 1, kernel_size=1),
            nn.Softmax(dim=-1)  # Chuẩn hóa dọc theo trục thời gian T
        )

        # Kết hợp cả Temporal Attention Pooling và Temporal Max Pooling -> Nhân đôi chiều đặc trưng
        fused_dim = in_channels * 2

        # --- BƯỚC 3: CÁC NHÁNH DỰ ĐOÁN ĐẦU RA (Prediction Heads) ---
        # 3.1. Nhánh chính G_y: Phân loại trạng thái buồn ngủ (Công thức 11 - MSTGCN)
        self.label_predictor = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Dropout(dropout),
            nn.Linear(fused_dim, in_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(in_channels // 2, num_classes)
        )

        # 3.2. Nhánh phụ G_d (Tùy chọn): Phân loại miền/tài xế qua GRL (Công thức 12 - MSTGCN)
        if num_domains > 0:
            self.grl = GradientReversalLayer(lambda_grl=1.0)
            self.domain_classifier = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Dropout(dropout),
                nn.Linear(fused_dim, in_channels // 2),
                nn.GELU(),
                nn.Linear(in_channels // 2, num_domains)
            )

        # 3.3. Nhánh Hồi quy (Tùy chọn): Dự đoán chỉ số buồn ngủ liên tục (như KSS / PERCLOS)
        if enable_regression:
            self.regression_head = nn.Sequential(
                nn.LayerNorm(fused_dim),
                nn.Linear(fused_dim, 1),
                nn.Sigmoid()  # Quy về khoảng [0.0 -> 1.0] (Tỉnh táo -> Ngủ gật hoàn toàn)
            )

    def pool_spatio_temporal(self, feat: torch.Tensor) -> torch.Tensor:
        """
        Nén Tensor (B, C, T, V=68) -> Vector đặc trưng toàn cục (B, 2*C)
        mà không làm mất tín hiệu cục bộ ở Mắt/Miệng hay các cơn chợp mắt ngắn.
        """
        # 1. Gộp chiều Không gian (V=68 -> 1) bằng Spatial Attention
        s_weights = self.spatial_attn(feat)  # (B, 1, T, V)
        x_spatial = torch.sum(feat * s_weights, dim=-1)  # (B, C, T)

        # 2. Lọc qua lớp Temporal Conv cuối cùng (Yu et al.)
        x_temp = self.final_temporal_conv(x_spatial) + x_spatial  # (B, C, T)

        # 3. Gộp chiều Thời gian (T -> 1): Kết hợp Attention Pool (xu hướng chung) + Max Pool (đỉnh cơn ngáp/nhắm mắt)
        t_weights = self.temporal_attn(x_temp)  # (B, 1, T)
        t_attn_pool = torch.sum(x_temp * t_weights, dim=-1)  # (B, C)
        t_max_pool = torch.amax(x_temp, dim=-1)  # (B, C)

        return torch.cat([t_attn_pool, t_max_pool], dim=-1)  # (B, 2*C)

    def forward(
            self,
            feat: torch.Tensor,
            return_domain: bool = False,
            return_probs: bool = False
    ):
        """
        Input : feat shape (B, C, T, V) từ khối STGCNBlock cuối cùng.
        Output:
          - Mặc định (Inference): y_logits (hoặc xác suất Softmax nếu return_probs=True).
          - Khi huấn luyện đa tác vụ / Domain Generalization: dict chứa 'logits', 'domain_logits', 'fatigue_score'.
        """
        global_feat = self.pool_spatio_temporal(feat)  # (B, 2*C)

        # Dự đoán nhãn trạng thái buồn ngủ (G_y)
        y_logits = self.label_predictor(global_feat)  # (B, num_classes)
        y_out = F.softmax(y_logits, dim=-1) if return_probs else y_logits

        if not return_domain and not self.enable_regression:
            return y_out

        outputs = {"logits": y_out}

        if return_domain and self.num_domains > 0:
            rev_feat = self.grl(global_feat)
            outputs["domain_logits"] = self.domain_classifier(rev_feat)

        if self.enable_regression:
            outputs["fatigue_score"] = self.regression_head(global_feat).squeeze(-1)

        return outputs
