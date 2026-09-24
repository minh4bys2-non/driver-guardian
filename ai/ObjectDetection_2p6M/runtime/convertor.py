from pathlib import Path

import onnx
import torch
from torch import nn

from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector
from ai.ObjectDetection_2p6M.utils.artifacts import validate_metadata

class BackboneNeck(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone, self.neck = model.backbone, model.neck

    def forward(self, images):
        return self.neck(*self.backbone(images))

class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head

    def forward(self, p3, p4, p5):
        output = self.head([p3, p4, p5], o2o_only=True)["o2o"]
        return output["cls"], output["box"]


@torch.inference_mode()
def convert(checkpoint_path, output_dir=Path(__file__).parent / "runtime", img_size=640):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = validate_metadata(checkpoint_path, checkpoint)
    if img_size <= 0 or img_size % max(metadata["architecture"]["strides"]):
        raise ValueError("img_size phải dương và chia hết cho stride lớn nhất")
    model = NMSFreeDetector(**metadata["architecture"], img_size=img_size).float().eval()
    model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])
    backbone_neck, head = BackboneNeck(model).eval(), Head(model).eval()
    images = torch.zeros(1, 3, img_size, img_size, dtype=torch.float32)
    features = backbone_neck(images)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exports = (
        ("backbone_neck.onnx", backbone_neck, (images,), ["images"], ["p3", "p4", "p5"]),
        ("head.onnx", head, features, ["p3", "p4", "p5"], ["logits", "boxes"]),
    )
    for filename, module, inputs, input_names, output_names in exports:
        path = output_dir / filename
        torch.onnx.export(
            module, inputs, str(path), opset_version=17, dynamo=False,
            input_names=input_names, output_names=output_names,
            dynamic_axes={name: {0: "batch_size"} for name in input_names + output_names},
        )
        graph = onnx.load(str(path))
        # Exporter có thể để chiều output cố định thành symbolic sau reshape/transpose.
        for value, tensor in zip(graph.graph.output, module(*inputs)):
            for dim, size in zip(value.type.tensor_type.shape.dim[1:], tensor.shape[1:]):
                dim.dim_value = int(size)
        onnx.checker.check_model(graph, full_check=True)
        onnx.save(graph, str(path))
        print(f"Exported float32: {path}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    checkpoint_path = "/home/tranmanhduy/Workspace/ptithcm/TTTN/NewVersionObDetect/checkpoints/2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031/de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310/finetune/best.pt"
    output_dir = root / "runtime"
    img_size = 640
    convert(checkpoint_path, output_dir, img_size)
