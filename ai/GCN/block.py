# File: block.py
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv, DenseGATConv
from torch_geometric.utils import to_undirected, add_self_loops, to_dense_adj


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input : (B, 2, T, V=68) - Tọa độ pixel hoặc [0, 1] trong Hệ tọa độ Khung ảnh.
        Output: (B, 6, T, V=68) - [x_norm, y_norm, dx_norm, dy_norm, d2x_norm, d2y_norm]
        """
        if x.size(1) == 6:
            return x

        # 1. Tính khoảng cách giải phẫu chuẩn d_t (Inter-Ocular Distance giữa điểm 36 và 45)
        # eye_dist shape: (B, 1, T, 1)
        p_left_eye = x[:, :, :, self.left_eye_idx : self.left_eye_idx + 1]   # (B, 2, T, 1)
        p_right_eye = x[:, :, :, self.right_eye_idx : self.right_eye_idx + 1] # (B, 2, T, 1)
        iod_scale = torch.norm(p_right_eye - p_left_eye, p=2, dim=1, keepdim=True) + self.eps

        # 2. Chuẩn hóa tọa độ hình thái tĩnh theo Tâm mũi (điểm 30) và tỷ lệ IOD
        nose_pos = x[:, :, :, self.nose_idx : self.nose_idx + 1]              # (B, 2, T, 1)
        pos_norm = (x - nose_pos) / iod_scale                                 # (B, 2, T, V)

        # 3. Tính Vận tốc (dx, dy) và Gia tốc (d2x, d2y) chuẩn hóa theo IOD
        # (Tính diff trên x trước khi trừ mũi để giữ được vận tốc gật đầu của cả khuôn mặt)
        vel_raw = torch.diff(x, n=1, dim=2)
        vel_raw = F.pad(vel_raw, (0, 0, 1, 0), mode='replicate')
        vel_norm = vel_raw / iod_scale                                        # (B, 2, T, V)

        acc_norm = torch.diff(vel_norm, n=1, dim=2)
        acc_norm = F.pad(acc_norm, (0, 0, 1, 0), mode='replicate')            # (B, 2, T, V)

        # Ghép thành 6 kênh đặc trưng chuẩn hóa hoàn chỉnh
        return torch.cat([pos_norm, vel_norm, acc_norm], dim=1)               # (B, 6, T, V)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        x = self.sgcn(x)
        x = self.tgcn(x)
        return self.act(x + res)
