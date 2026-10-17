import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


from ai.LSTM.src.models_convgru import ConvGRUClassifier
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN
from ai.LSTM.src.checkpoint import load_model

logger = logging.getLogger(__name__)
class ExportNMSFreeDetector(nn.Module):
    """Wrapper đóng gói mô hình CNN Backbone + PAFPN Neck của NMSFreeDetector"""

    def __init__(self, backbone: Backbone, neck: PAFPN):
        super().__init__()
        self.backbone = backbone
        self.neck = neck

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out3, out4, out5 = self.backbone(x)
        p3, p4, p5 = self.neck(out3, out4, out5)
        return p3.float(), p4.float(), p5.float()


class ExportConvGRUClassifier(nn.Module):
    """Wrapper đóng gói mô hình Spatio-Temporal ConvGRUClassifier"""

    def __init__(self, model: ConvGRUClassifier):
        super().__init__()
        self.conv_gru = model

    def forward(
        self,
        p3: torch.Tensor,
        p4: torch.Tensor,
        p5: torch.Tensor
    ) -> torch.Tensor:
        logits = self.conv_gru((p3, p4, p5), seq_lens=None)
        probs = F.softmax(logits, dim=-1)
        return probs[:, 1]


def export_to_pte(
    model: nn.Module,
    example_inputs: Tuple[torch.Tensor, ...],
    output_path: Path,
    use_vulkan: bool = True
) -> Path:
    """Biên dịch và hạ cấp một mô hình PyTorch sang file ExecuTorch .pte."""
    from executorch.exir import to_edge, EdgeCompileConfig

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model_name = model.__class__.__name__
    logger.info(f"\n{'='*70}")
    logger.info(f"[*] BẮT ĐẦU EXPORT: {model_name}")
    logger.info(f"[*] Đường dẫn đích: {output_path}")
    logger.info(f"{'='*70}")

    logger.info(f"[*] Bước 1/4: Đang chụp đồ thị FX Graph (torch.export.export)...")
    with torch.no_grad():
        exported_program = torch.export.export(model, example_inputs)
    logger.info(f"[✓] Đã tạo thành công ExportedProgram.")

    logger.info(f"[*] Bước 2/4: Đang hạ cấp sang ExecuTorch Edge IR (to_edge)...")
    compile_config = EdgeCompileConfig(
        _core_aten_ops_exception_list=[
            torch.ops.aten._native_batch_norm_legit_no_training.default,
            torch.ops.aten._native_batch_norm_legit_functional.default,
        ]
    )
    edge_program = to_edge(exported_program, compile_config=compile_config)
    logger.info(f"[✓] Đã tạo thành công EdgeProgram.")

    if use_vulkan:
        from executorch.backends.vulkan.partitioner.vulkan_partitioner import VulkanPartitioner

        logger.info(f"[*] Bước 3/4: Đang phân vùng toán tử sang Vulkan Backend (VulkanPartitioner)...")
        edge_program = edge_program.to_backend(VulkanPartitioner())
        logger.info(f"[✓] Đã tối ưu và gán toán tử cho Vulkan Delegate.")
    else:
        logger.info(f"[*] Bước 3/4: Bỏ qua Vulkan Backend, sử dụng Standard Edge Dialect.")

    logger.info(f"[*] Bước 4/4: Đang đóng gói chương trình ExecuTorch (.to_executorch)...")
    executorch_program = edge_program.to_executorch()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(executorch_program.buffer)

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"[✓] XUẤT THÀNH CÔNG: {output_path} ({file_size_mb:.2f} MB)")
    return output_path


def main() -> None:
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Export the trained backbone and ConvGRU together")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    model, state = load_model(args.checkpoint)
    cfg = state["config"]
    output_dir = args.output_dir or args.checkpoint.parent / "export_pte"
    cnn = ExportNMSFreeDetector(model.backbone_neck.backbone, model.backbone_neck.neck).eval()
    gru = ExportConvGRUClassifier(model.conv_gru).eval()
    image = torch.zeros(1, 3, cfg["image_size"], cfg["image_size"])
    with torch.no_grad():
        features = cnn(image)
    frames = round(cfg["window_sec"] * cfg["target_fps"])
    sequences = tuple(feature[:, None].expand(-1, frames, -1, -1, -1).contiguous() for feature in features)
    export_to_pte(cnn, (image,), output_dir / "backbone_neck.pte", use_vulkan=not args.cpu)
    export_to_pte(gru, sequences, output_dir / "convgru.pte", use_vulkan=not args.cpu)
    logger.info("Exported checkpoint epoch=%s to %s", state["epoch"], output_dir)


if __name__ == "__main__":
    main()
