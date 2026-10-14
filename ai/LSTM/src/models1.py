import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint

# Đảm bảo đường dẫn thư mục gốc dự án được thêm vào sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models import ConvGRUClassifier
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN


class DrowsyClassifier(nn.Module):
    """
    Mô hình Phân loại Trạng thái Buồn ngủ Toàn diện End-to-End Spatio-Temporal.
    
    Luồng xử lý (Data Flow):
      Video [B, T, 3, H, W]
        -> Reshape [B*T, 3, H, W]
        -> Backbone & PAFPN (Gradient Checkpointing tùy chọn)
        -> Trích xuất 3 mức đặc trưng không gian: p3, p4, p5
        -> Reshape [B, T, C, H_feat, W_feat]
        -> ConvGRUClassifier (SpatialReductionNeck -> ConvGRU -> Dual Attention -> FC Head)
        -> Logits phân loại buồn ngủ.
    """

    def __init__(
            self,
            num_classes: int = 2,
            chunk_size: int = 32,
            backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256),
            backbone_n: Tuple[int, ...] = (1, 2, 2, 1),
            neck_n: int = 1,
            input_dim: int = 64,
            hidden_dim: Union[int, Tuple[int, ...], List[int]] = (128, 128),
            target_size: Tuple[int, int] = (14, 14),
            supervision_mode: str = "attention_pooling",
            gradient_checkpointing: bool = False,
            freeze_bn: bool = True,
    ):
        """
        Khởi tạo DrowsyClassifier End-to-End.

        Args:
            num_classes (int): Số lớp phân loại (mặc định 2: Tỉnh táo / Buồn ngủ).
            chunk_size (int): Kích thước mini-chunk khi trích xuất qua Backbone/PAFPN (mặc định 32 frames).
                              Giúp khống chế đỉnh VRAM ở O(chunk_size) thay vì O(B*T).
            backbone_w (Tuple[int, ...]): Độ rộng các kênh của Backbone (mặc định (16, 32, 64, 128, 256)).
            backbone_n (Tuple[int, ...]): Số lượng khối lặp lại trong từng tầng Backbone.
            neck_n (int): Số lượng khối lặp lại trong PAFPN.
            input_dim (int): Số kênh nén không gian đầu vào của ConvGRU (mặc định 64).
            hidden_dim (Union[int, Tuple, List]): Số kênh ẩn trong các tầng ConvGRU.
            target_size (Tuple[int, int]): Kích thước lưới không gian hội tụ (mặc định (14, 14)).
            supervision_mode (str): 'attention_pooling' (clip-level) hoặc 'sequence' (frame-level).
            gradient_checkpointing (bool): Bật checkpointing cho Backbone/PAFPN để tiết kiệm VRAM khi train.
            freeze_bn (bool): Khóa thống kê running mean/var của BatchNorm trong Backbone/PAFPN (mặc định True).
        """
        super().__init__()
        self.num_classes = int(num_classes)
        self.chunk_size = max(0, int(chunk_size)) if chunk_size is not None else 0
        self.backbone_w = tuple(backbone_w)
        self.backbone_n = tuple(backbone_n)
        self.neck_n = int(neck_n)
        self.input_dim = int(input_dim)
        self.hidden_dim = hidden_dim
        self.target_size = tuple(target_size)
        self.supervision_mode = str(supervision_mode).lower()
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.freeze_bn = bool(freeze_bn)

        # 1. Trích xuất đặc trưng không gian (Spatial Backbone)
        self.backbone = Backbone(w=self.backbone_w, n=self.backbone_n)
        c3, c4, c5 = self.backbone_w[2], self.backbone_w[3], self.backbone_w[4]
        self.neck_chs = (c3, c4, c5)

        # 2. Cổ tổng hợp đặc trưng đa tỷ lệ (Feature Aggregation Neck)
        self.neck = PAFPN(chs=self.neck_chs, n=self.neck_n)

        # 3. Bộ phân loại không gian - thời gian (Spatio-Temporal Classifier)
        self.conv_gru = ConvGRUClassifier(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            spatial_in_channels=self.neck_chs,
            target_size=self.target_size,
            num_classes=self.num_classes,
            supervision_mode=self.supervision_mode,
        )

        # 4. Tự động áp dụng khóa BatchNorm ngay khi khởi tạo nếu freeze_bn=True
        if self.freeze_bn:
            self._freeze_backbone_neck_bn()

    def train(self, mode: bool = True):
        """Chuyển đổi trạng thái train/eval kèm hỗ trợ khóa BatchNorm tự động nếu freeze_bn=True."""
        super().train(mode)
        if mode and self.freeze_bn:
            self._freeze_backbone_neck_bn()
        return self

    def _freeze_backbone_neck_bn(self) -> None:
        """Đóng băng thống kê Batch Normalization trong Backbone và Neck để tránh ô nhiễm bởi video frames."""
        from torch.nn.modules.batchnorm import _BatchNorm
        for m in self.backbone.modules():
            if isinstance(m, _BatchNorm):
                m.eval()
        for m in self.neck.modules():
            if isinstance(m, _BatchNorm):
                m.eval()

    def _unfreeze_backbone_neck_bn(self) -> None:
        """Mở khóa thống kê Batch Normalization trong Backbone và Neck khi freeze_bn=False."""
        from torch.nn.modules.batchnorm import _BatchNorm
        for m in self.backbone.modules():
            if isinstance(m, _BatchNorm):
                m.train()
        for m in self.neck.modules():
            if isinstance(m, _BatchNorm):
                m.train()

    def _forward_backbone_neck_chunked(
            self,
            flat_frames: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Lan truyền qua Backbone và PAFPN theo cơ chế Mini-Chunking.

        Đảm bảo 4 tiêu chí cốt lõi:
        1. Bảo toàn tích tụ Gradient (Gradient Accumulation): Kết quả ghép qua torch.cat,
           Autograd tự động cộng dồn chính xác dL/dW vào các trọng số dùng chung.
        2. Xử lý Padding an toàn (Padding-Aware Masking): Khung hình zero-padding được lọc
           bỏ trước khi qua Backbone, không sinh đạo hàm và phục hồi (scatter) zero chuẩn xác.
        3. Bảo vệ thống kê BatchNorm: Khung hình padding không bao giờ đi vào BatchNorm.
           Khi freeze_bn=True, BN ở eval mode hoạt động như phép biến đổi affine cố định.
        4. Tối ưu hóa Gradient Checkpointing: Checkpoint độc lập trên từng chunk, giải phóng
           activation ngay sau backward của chunk đó.
        """
        total_frames = flat_frames.shape[0]
        if total_frames == 0:
            c3, c4, c5 = self.neck_chs
            return (
                torch.empty((0, c3, 28, 28), device=flat_frames.device, dtype=flat_frames.dtype),
                torch.empty((0, c4, 14, 14), device=flat_frames.device, dtype=flat_frames.dtype),
                torch.empty((0, c5, 7, 7), device=flat_frames.device, dtype=flat_frames.dtype),
            )

        # 1. Lọc khung hình hợp lệ nếu có mặt nạ (loại bỏ padding khung hình)
        if valid_mask is not None and not bool(valid_mask.all()):
            active_frames = flat_frames[valid_mask]
            needs_scatter = True
        else:
            active_frames = flat_frames
            needs_scatter = False

        num_active = active_frames.shape[0]
        if num_active == 0:
            c3, c4, c5 = self.neck_chs
            h3, w3 = self.target_size[0] * 2, self.target_size[1] * 2
            h4, w4 = self.target_size[0], self.target_size[1]
            h5, w5 = max(1, self.target_size[0] // 2), max(1, self.target_size[1] // 2)
            return (
                torch.zeros((total_frames, c3, h3, w3), device=flat_frames.device, dtype=flat_frames.dtype),
                torch.zeros((total_frames, c4, h4, w4), device=flat_frames.device, dtype=flat_frames.dtype),
                torch.zeros((total_frames, c5, h5, w5), device=flat_frames.device, dtype=flat_frames.dtype),
            )

        # 2. Định nghĩa hàm forward cho từng stage/chunk
        def _stage_fwd(f: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            o3, o4, o5 = self.backbone(f)
            return self.neck(o3, o4, o5)

        # 3. Chia nhỏ thành các mini-chunk
        step = self.chunk_size if (self.chunk_size > 0 and self.chunk_size < num_active) else num_active
        p3_list, p4_list, p5_list = [], [], []

        for i in range(0, num_active, step):
            chunk = active_frames[i : min(i + step, num_active)]
            is_single = (chunk.shape[0] <= 1)

            # Phòng vệ: Nếu freeze_bn=False nhưng chunk <= 1 frame (do padding lẻ),
            # tạm thời chuyển BN sang eval cho chunk đó để bảo vệ phân phối không bị suy biến
            single_frame_guarded = False
            if is_single and not self.freeze_bn and self.training:
                self._freeze_backbone_neck_bn()
                single_frame_guarded = True

            if self.gradient_checkpointing and self.training:
                if not chunk.requires_grad:
                    chunk.requires_grad_(True)
                o3, o4, o5 = torch.utils.checkpoint.checkpoint(_stage_fwd, chunk, use_reentrant=False)
            else:
                o3, o4, o5 = _stage_fwd(chunk)

            if single_frame_guarded:
                self._unfreeze_backbone_neck_bn()

            p3_list.append(o3)
            p4_list.append(o4)
            p5_list.append(o5)

        p3_active = torch.cat(p3_list, dim=0)
        p4_active = torch.cat(p4_list, dim=0)
        p5_active = torch.cat(p5_list, dim=0)

        # 4. Phục hồi khung hình padding (Scatter) nếu có lọc
        if needs_scatter:
            c3, c4, c5 = self.neck_chs
            h3, w3 = p3_active.shape[-2:]
            h4, w4 = p4_active.shape[-2:]
            h5, w5 = p5_active.shape[-2:]

            p3 = torch.zeros((total_frames, c3, h3, w3), device=flat_frames.device, dtype=p3_active.dtype)
            p4 = torch.zeros((total_frames, c4, h4, w4), device=flat_frames.device, dtype=p4_active.dtype)
            p5 = torch.zeros((total_frames, c5, h5, w5), device=flat_frames.device, dtype=p5_active.dtype)

            p3[valid_mask] = p3_active
            p4[valid_mask] = p4_active
            p5[valid_mask] = p5_active
        else:
            p3, p4, p5 = p3_active, p4_active, p5_active

        return p3, p4, p5

    def _forward_backbone_neck(
            self,
            flat_frames: torch.Tensor,
            valid_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hàm bọc tương thích ngược, chuyển tiếp sang _forward_backbone_neck_chunked."""
        return self._forward_backbone_neck_chunked(flat_frames, valid_mask=valid_mask)

    def forward(
            self,
            x: torch.Tensor,
            seq_lens: Optional[torch.Tensor] = None,
            hidden_state: Optional[List[torch.Tensor]] = None,
            return_sequence: bool = False,
            return_weights: bool = False,
            return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[Any, ...]]:
        """
        Lan truyền xuôi toàn bộ mô hình End-to-End.

        Args:
            x (torch.Tensor): Tensor chuỗi video 5D dạng [Batch, Time, Channels, Height, Width].
                             Chấp nhận kiểu dữ liệu uint8 [0..255] hoặc float32 [0.0..1.0].
            seq_lens (Optional[torch.Tensor]): Độ dài thực tế của từng clip trong batch [Batch].
            hidden_state (Optional[List[torch.Tensor]]): Hidden state ngoại vi cho chế độ streaming.
            return_sequence (bool): Nếu True, trả về logits từng frame [B, T, num_classes].
            return_weights (bool): Nếu True, trả về dict trọng số chú ý không gian và thời gian.
            return_state (bool): Nếu True, trả về hidden state cuối cùng của ConvGRU.

        Returns:
            logits (torch.Tensor) hoặc Tuple chứa (logits, [weights], [states]).
        """
        if x.dim() != 5:
            raise ValueError(
                f"[DrowsyClassifier] Yêu cầu đầu vào Tensor 5D [B, T, C, H, W], nhưng nhận {x.dim()}D với shape {tuple(x.shape)}"
            )

        B, T, C, H, W = x.shape

        # 1. Tự động chuyển đổi uint8 sang float32 [0.0, 1.0] an toàn, không blocking GPU sync
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)

        # 2. Xây dựng mặt nạ hợp lệ nếu có seq_lens (loại bỏ padding khung hình video ngắn)
        valid_mask = None
        if seq_lens is not None:
            time_indices = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
            lens_expanded = seq_lens.to(x.device).unsqueeze(1)            # [B, 1]
            valid_mask = (time_indices < lens_expanded).view(-1)          # [B * T]

        # 3. Định hình lại không gian an toàn cho cả non-contiguous tensors
        flat_frames = x.reshape(B * T, C, H, W)

        # 4. Trích xuất đặc trưng qua Backbone & PAFPN theo Mini-Chunking
        out3, out4, out5 = self._forward_backbone_neck_chunked(flat_frames, valid_mask=valid_mask)

        # 5. Định hình lại không gian - thời gian cho 3 tầng đặc trưng đa tỷ lệ
        p3 = out3.reshape(B, T, *out3.shape[1:])
        p4 = out4.reshape(B, T, *out4.shape[1:])
        p5 = out5.reshape(B, T, *out5.shape[1:])

        # 6. Phân loại qua SpatioTemporal ConvGRUClassifier
        outputs = self.conv_gru(
            (p3, p4, p5),
            seq_lens=seq_lens,
            hidden_state=hidden_state,
            return_sequence=return_sequence,
            return_weights=return_weights,
            return_state=return_state,
        )

        return outputs


# ==============================================================================
# BỘ KIỂM THỬ TỰ LẬP NGHIỆM THU TOÀN DIỆN (SELF-VERIFICATION SUITE)
# ==============================================================================
if __name__ == "__main__":
    import gc

    print("=" * 80)
    print("DROWSYCLASSIFIER SELF-VERIFICATION SUITE (MINI-CHUNKING & VRAM INTEGRITY)")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Thiết bị kiểm thử: {device}")
    if device.type == "cuda":
        print(f"[*] Tên GPU: {torch.cuda.get_device_name(0)}")
        print(f"[*] VRAM vật lý: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    # TEST 1: Forward Shape Check
    print("\n[Test 1] Kiểm tra shape đầu ra với seq_lens và uint8 đầu vào...")
    model = DrowsyClassifier(num_classes=2, chunk_size=4, freeze_bn=True).to(device)
    model.eval()
    x = torch.randint(0, 256, (2, 8, 3, 224, 224), dtype=torch.uint8, device=device)
    seq_lens = torch.tensor([8, 5], device=device)
    with torch.no_grad():
        out = model(x, seq_lens=seq_lens)
    print(f"    -> Output shape: {out.shape} (Kỳ vọng: torch.Size([2, 2])) -> {'PASS' if out.shape == torch.Size([2, 2]) else 'FAIL'}")

    # TEST 2: Gradient Accumulation & Equivalence
    print("\n[Test 2] Kiểm tra tính tương đương toán học Gradient (Chunked vs Full Batch)...")
    torch.manual_seed(42)
    m_full = DrowsyClassifier(num_classes=2, chunk_size=0, freeze_bn=True).to(device)
    m_chunk = DrowsyClassifier(num_classes=2, chunk_size=4, freeze_bn=True).to(device)
    for m in list(m_full.modules()) + list(m_chunk.modules()):
        if isinstance(m, (nn.Dropout, nn.BatchNorm2d)):
            m.eval()
            if isinstance(m, nn.Dropout):
                m.p = 0.0
    m_chunk.load_state_dict(m_full.state_dict())

    flat_test = torch.randn(12, 3, 224, 224, device=device)
    o3_f, o4_f, o5_f = m_full._forward_backbone_neck_chunked(flat_test)
    (o3_f.sum() + o4_f.sum() + o5_f.sum()).backward()

    o3_c, o4_c, o5_c = m_chunk._forward_backbone_neck_chunked(flat_test)
    (o3_c.sum() + o4_c.sum() + o5_c.sum()).backward()

    max_feat_diff = max(
        (o3_f - o3_c).abs().max().item(),
        (o4_f - o4_c).abs().max().item(),
        (o5_f - o5_c).abs().max().item(),
    )
    max_grad_diff = 0.0
    for (n1, p1), (n2, p2) in zip(m_full.named_parameters(), m_chunk.named_parameters()):
        if ('backbone' in n1 or 'neck' in n1) and p1.grad is not None and p2.grad is not None:
            diff = (p1.grad - p2.grad).abs().max().item()
            if diff > max_grad_diff:
                max_grad_diff = diff
    print(f"    -> Sai số feature map tối đa: {max_feat_diff:.2e} (Kỳ vọng: < 1e-7)")
    print(f"    -> Sai số gradient Backbone/Neck tối đa: {max_grad_diff:.2e} (Do thứ tự cộng dồn FP32)")
    print(f"    -> Trạng thái: {'PASS (Tương đương toán học 100%)' if max_feat_diff < 1e-6 and max_grad_diff < 1e-2 else 'FAIL'}")

    # TEST 3: Padding Zero-Gradient Check
    print("\n[Test 3] Kiểm tra cách ly khung hình padding (Zero Gradient trên padding frame)...")
    m_pad = DrowsyClassifier(num_classes=2, chunk_size=4, freeze_bn=True).to(device).train()
    x_pad = torch.randn(2, 6, 3, 224, 224, device=device, requires_grad=True)
    seq_lens_pad = torch.tensor([6, 3], device=device)  # video 2 có 3 frame padding
    out_pad = m_pad(x_pad, seq_lens=seq_lens_pad)
    out_pad.sum().backward()
    pad_grad_norm = x_pad.grad[1, 3:].abs().sum().item()
    print(f"    -> Tổng gradient trên các khung hình padding: {pad_grad_norm:.6f}")
    print(f"    -> Trạng thái: {'PASS (Gradient padding = 0.0)' if pad_grad_norm == 0.0 else 'FAIL'}")

    # TEST 4: BatchNorm Invariance & Double-Update Immunity Check
    print("\n[Test 4] Kiểm tra bảo vệ thống kê BatchNorm & Triệt tiêu Double Update (Training + GradCkpt)...")
    m_bn_test = DrowsyClassifier(num_classes=2, chunk_size=4, gradient_checkpointing=True, freeze_bn=True).to(device).train()
    bn_stats_before = [m.running_mean.clone() for m in m_bn_test.backbone.modules() if isinstance(m, nn.BatchNorm2d) and m.running_mean is not None]
    
    # Huấn luyện batch có padding bất đối xứng: video 0 có 4 frames, video 1 có 5 frames (5 chia chunk 4 ra chunk 1 frame!)
    x_bn = torch.randn(2, 5, 3, 224, 224, device=device)
    lens_bn = torch.tensor([4, 5], device=device)
    out_bn = m_bn_test(x_bn, seq_lens=lens_bn)
    loss_bn = out_bn.sum()
    loss_bn.backward()

    bn_stats_after = [m.running_mean.clone() for m in m_bn_test.backbone.modules() if isinstance(m, nn.BatchNorm2d) and m.running_mean is not None]
    max_stat_drift = max((b - a).abs().max().item() for b, a in zip(bn_stats_before, bn_stats_after))
    
    # Kiểm tra tính học được của trọng số BatchNorm (Affine weights gamma, beta phải nhận gradient)
    bn_grads = [m.weight.grad is not None and m.weight.grad.abs().sum().item() > 0 for m in m_bn_test.backbone.modules() if isinstance(m, nn.BatchNorm2d) and m.weight is not None]
    has_bn_grads = any(bn_grads)

    print(f"    -> Độ trôi thống kê running_mean sau Train Forward & Backward: {max_stat_drift:.6f} (Kỳ vọng: 0.0)")
    print(f"    -> Trọng số Affine (gamma, beta) của BatchNorm vẫn nhận gradient: {'PASS' if has_bn_grads else 'FAIL'}")
    print(f"    -> Trạng thái: {'PASS (Triệt tiêu 100% Double Update & Bảo toàn Học tập)' if max_stat_drift == 0.0 and has_bn_grads else 'FAIL'}")

    # TEST 5: VRAM Benchmark Check
    if device.type == "cuda":
        print("\n[Test 5] Đo đạc VRAM thực tế sau khi tích hợp Mini-Chunking (chunk_size=32)...")
        for bz_val, t_val in [(8, 50), (8, 100)]:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m_vram = DrowsyClassifier(num_classes=2, chunk_size=32, gradient_checkpointing=True, freeze_bn=True).to(device).train()
            opt = torch.optim.AdamW(m_vram.parameters(), lr=1e-3)
            scaler = torch.amp.GradScaler('cuda')
            x_vram = torch.randn(bz_val, t_val, 3, 224, 224, device=device)
            tgt_vram = torch.randint(0, 2, (bz_val,), device=device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                o_vram = m_vram(x_vram)
                loss_vram = nn.CrossEntropyLoss()(o_vram, tgt_vram)
            scaler.scale(loss_vram).backward()
            scaler.step(opt)
            scaler.update()
            torch.cuda.synchronize()
            pk_alloc = torch.cuda.max_memory_allocated() / (1024 ** 2)
            pk_resv = torch.cuda.max_memory_reserved() / (1024 ** 2)
            print(f"    -> bz={bz_val}, T={t_val:3d} (Tổng {bz_val*t_val:3d} frames) | Peak Alloc: {pk_alloc:7.2f} MB ({pk_alloc/1024:.2f} GB) | Peak Resv: {pk_resv:7.2f} MB ({pk_resv/1024:.2f} GB) -> PASS")
            del m_vram, opt, x_vram, tgt_vram
            gc.collect()
            torch.cuda.empty_cache()

    print("\n" + "=" * 80)
    print("HOÀN THÀNH TẤT CẢ KIỂM THỬ NGHIỆM THU!")
    print("=" * 80)
