import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import warnings

# Bỏ qua riêng cảnh báo LeafSpec này
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=".*LeafSpec.*"
)
# --- Tự động cấu hình sys.path tương thích đa nền tảng ---
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1] if SCRIPT_DIR.name == "LSTM" and SCRIPT_DIR.parent.name == "ai" else SCRIPT_DIR
for p in [SCRIPT_DIR, PROJECT_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
import torch.nn as nn
import torch.nn.functional as F

from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner
from executorch.exir import to_edge, EdgeCompileConfig

from ai.LSTM.src.models_convgru import ConvGRUClassifier
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN
from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector


# ==============================================================================
# KHỐI CẤU HÌNH TẬP TRUNG (CENTRALIZED EXPORT CONFIGURATION)
# ==============================================================================
@dataclass
class ExportConfig:
    """Tập trung toàn bộ tham số cấu hình cho quy trình xuất ExecuTorch .pte."""

    # 1. Đường dẫn file Checkpoints
    cnn_ckpt_path: Path = PROJECT_ROOT / "ai" / "checkpoints" / "model_cnn" / "best.pt"
    gru_ckpt_path: Path = PROJECT_ROOT / "ai" / "checkpoints" / "model_convgru" / "best.pt"

    # Fallback cho checkpoint ConvGRU nếu đường dẫn trên không tồn tại
    gru_fallback_ckpt: Path = SCRIPT_DIR / "checkpoints" / "1m9" / "best.pt"

    # 2. Thư mục và tệp đầu ra (.pte)
    output_dir: Path = SCRIPT_DIR / "checkpoints" / "export_pte"
    cnn_pte_name: str = "detector_backbone_neck_vulkan.pte"
    gru_pte_name: str = "convgru_classifier_vulkan.pte"

    # 3. Kích thước Tensor đầu vào cho mô hình CNN Detector
    batch_size: int = 1         # Kích thước batch (1 cho frame-by-frame streaming trên thiết bị biên)
    img_channels: int = 3       # Kênh màu RGB
    img_size: int = 640         # Kích thước khung hình chuẩn 640x640

    # 4. Kích thước Tensor đầu vào cho mô hình ConvGRU Classifier
    window_size: int = 50       # Độ dài chuỗi khung hình cửa sổ trượt (50 frames ~ 10s quan sát)

    # 5. Cấu hình Backend ExecuTorch
    use_vulkan: bool = True     # Bật hạ cấp sang VulkanPartitioner (GPU di động / nhúng)


# ==============================================================================
# CÁC LỚP MÔ HÌNH ĐÓNG GÓI CHO EXPORT (EXPORT WRAPPER MODULES)
# ==============================================================================
class ExportNMSFreeDetector(nn.Module):
    """
    Wrapper đóng gói mô hình CNN Backbone + PAFPN Neck của NMSFreeDetector
    phục vụ xuất sang ExecuTorch .pte (Vulkan backend).

    Đầu vào: Tensor hình ảnh [B, 3, H, W] (mặc định B=1, H=640, W=640).
    Đầu ra: Tuple gồm 3 bản đồ đặc trưng không gian đa tỷ lệ (P3, P4, P5):
      - P3: [B, 64, 80, 80]
      - P4: [B, 128, 40, 40]
      - P5: [B, 256, 20, 20]
    """

    def __init__(self, backbone: Backbone, neck: PAFPN):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out3, out4, out5 = self.backbone(x)
        p3, p4, p5 = self.neck(out3, out4, out5)
        return p3.float(), p4.float(), p5.float()


class ExportConvGRUClassifier(nn.Module):
    """
    Wrapper đóng gói mô hình Spatio-Temporal ConvGRUClassifier
    phục vụ xuất sang ExecuTorch .pte (Vulkan backend).

    Đầu vào: 3 tensor đặc trưng không gian - thời gian (P3, P4, P5):
      - P3: [1, T, 64, 80, 80]
      - P4: [1, T, 128, 40, 40]
      - P5: [1, T, 256, 20, 20]
    Đầu ra:
      - Tensor xác suất buồn ngủ [1] (không dùng .item(), hoàn toàn tương thích ExecuTorch).
    """

    def __init__(self, model: ConvGRUClassifier):
        super().__init__()
        self.conv_gru = model

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor
    ) -> torch.Tensor:
        # seq_lens=None loại bỏ hoàn toàn dynamic boolean masking trong SpatialReductionNeck
        logits = self.conv_gru((p3, p4, p5), seq_lens=None)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]


# ==============================================================================
# HÀM ĐIỀU PHỐI EXPORT & LOWERING SANG EXECUTORCH .PTE
# ==============================================================================
def export_to_pte(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_path: Path,
    use_vulkan: bool = True
) -> Path:
    """
    Biên dịch và hạ cấp một mô hình PyTorch sang file ExecuTorch .pte.

    Quy trình:
      1. Thiết lập model.eval() và đóng băng gradient.
      2. Chụp đồ thị tĩnh FX Graph bằng torch.export.export().
      3. Hạ cấp sang Edge Dialect (to_edge) kèm danh sách ngoại lệ Core ATen cho BatchNorm.
      4. Phân vùng đồ thị và tạo Vulkan Delegate bằng VulkanPartitioner.
      5. Đóng gói buffer và ghi ra file .pte.
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model_name = model.__class__.__name__
    print(f"\n{'='*70}")
    print(f"[*] BẮT ĐẦU EXPORT: {model_name}")
    print(f"[*] Đường dẫn đích: {output_path}")
    print(f"{'='*70}")

    # 1. Tracing đồ thị tĩnh bằng torch.export
    print(f"[*] Bước 1/4: Đang chụp đồ thị FX Graph (torch.export.export)...")
    with torch.no_grad():
        exported_program = torch.export.export(model, example_inputs)
    print(f"[✓] Đã tạo thành công ExportedProgram.")

    # 2. Chuyển đổi sang ExecuTorch Edge IR
    print(f"[*] Bước 2/4: Đang hạ cấp sang ExecuTorch Edge IR (to_edge)...")
    compile_config = EdgeCompileConfig(
        _core_aten_ops_exception_list=[
            torch.ops.aten._native_batch_norm_legit_no_training.default,
            torch.ops.aten._native_batch_norm_legit_functional.default,
        ]
    )
    edge_program = to_edge(exported_program, compile_config=compile_config)
    print(f"[✓] Đã tạo thành công EdgeProgram.")

    # 3. Phân vùng Vulkan Partitioner
    if use_vulkan:
        print(f"[*] Bước 3/4: Đang phân vùng toán tử sang Vulkan Backend (VulkanPartitioner)...")
        edge_program = edge_program.to_backend(VulkanPartitioner())
        print(f"[✓] Đã tối ưu và gán toán tử cho Vulkan Delegate.")
    else:
        print(f"[*] Bước 3/4: Bỏ qua Vulkan Backend, sử dụng Standard Edge Dialect.")

    # 4. Đóng gói và lưu file .pte
    print(f"[*] Bước 4/4: Đang đóng gói chương trình ExecuTorch (.to_executorch)...")
    executorch_program = edge_program.to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"[✓] XUẤT THÀNH CÔNG: {output_path} ({file_size_mb:.2f} MB)")
    return output_path


# ==============================================================================
# HÀM THỰC THI CHÍNH (MAIN PIPELINE)
# ==============================================================================
def main():
    cfg = ExportConfig()

    print("=" * 80)
    print(" HỆ THỐNG XUẤT MÔ HÌNH EXECUTORCH .PTE (VULKAN BACKEND) - DRIVER GUARDIAN AI")
    print("=" * 80)
    print(f"[*] Thư mục gốc dự án: {PROJECT_ROOT}")
    print(f"[*] Thư mục xuất .pte: {cfg.output_dir}")
    print(f"[*] Kích thước CNN input: [{cfg.batch_size}, {cfg.img_channels}, {cfg.img_size}, {cfg.img_size}]")
    print(f"[*] Kích thước GRU sequence: Window = {cfg.window_size} frames")
    print(f"[*] Sử dụng Vulkan Backend: {cfg.use_vulkan}")

    # 1. Xác thực và nạp Checkpoint mô hình CNN
    cnn_path = cfg.cnn_ckpt_path
    if not cnn_path.exists():
        raise FileNotFoundError(f"[!] Không tìm thấy Checkpoint CNN tại: {cnn_path}")
    print(f"\n[*] Đang nạp Checkpoint CNN từ: {cnn_path}...")
    cnn_detector = NMSFreeDetector.from_checkpoint(cnn_path, map_location="cpu", eval_mode=True)
    export_cnn = ExportNMSFreeDetector(backbone=cnn_detector.backbone, neck=cnn_detector.neck)

    # 2. Xác thực và nạp Checkpoint mô hình ConvGRU
    gru_path = cfg.gru_ckpt_path
    if not gru_path.exists():
        print(f"[!] Checkpoint ConvGRU chính không tồn tại ({gru_path}), chuyển sang fallback: {cfg.gru_fallback_ckpt}")
        gru_path = cfg.gru_fallback_ckpt
    if not gru_path.exists():
        raise FileNotFoundError(f"[!] Không tìm thấy Checkpoint ConvGRU tại cả 2 đường dẫn.")
    print(f"[*] Đang nạp Checkpoint ConvGRU từ: {gru_path}...")
    gru_classifier = ConvGRUClassifier.from_checkpoint(gru_path, map_location="cpu")
    export_gru = ExportConvGRUClassifier(model=gru_classifier)

    # 3. Chuẩn bị tensor giả lập (Sample Inputs)
    print(f"\n[*] Đang khởi tạo tensor giả lập cho quy trình kiểm thử biên dịch...")
    dummy_img = torch.randn(cfg.batch_size, cfg.img_channels, cfg.img_size, cfg.img_size, dtype=torch.float32)
    dummy_p3 = torch.randn(1, cfg.window_size, 64, 80, 80, dtype=torch.float32)
    dummy_p4 = torch.randn(1, cfg.window_size, 128, 40, 40, dtype=torch.float32)
    dummy_p5 = torch.randn(1, cfg.window_size, 256, 20, 20, dtype=torch.float32)

    # 4. Xuất mô hình CNN Detector sang .pte
    cnn_output_path = cfg.output_dir / cfg.cnn_pte_name
    export_to_pte(
        model=export_cnn,
        example_inputs=(dummy_img,),
        output_path=cnn_output_path,
        use_vulkan=cfg.use_vulkan
    )

    # 5. Xuất mô hình ConvGRU Classifier sang .pte
    gru_output_path = cfg.output_dir / cfg.gru_pte_name
    export_to_pte(
        model=export_gru,
        example_inputs=(dummy_p3, dummy_p4, dummy_p5),
        output_path=gru_output_path,
        use_vulkan=cfg.use_vulkan
    )

    print("\n" + "=" * 80)
    print(" HOÀN THÀNH TOÀN BỘ QUY TRÌNH EXPORT .PTE")
    print("=" * 80)
    print(f"[1] Detector Backbone+Neck PTE: {cnn_output_path}")
    print(f"    - Kích thước: {cnn_output_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"[2] ConvGRU Classifier PTE:     {gru_output_path}")
    print(f"    - Kích thước: {gru_output_path.stat().st_size / (1024*1024):.2f} MB")


if __name__ == "__main__":
    main()
