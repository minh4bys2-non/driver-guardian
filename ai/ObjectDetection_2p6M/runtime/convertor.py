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

    def forward(self, x):
        return self.neck(*self.backbone(x))


class Head(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.head = model.head

    def forward(self, p3, p4, p5):
        out = self.head([p3, p4, p5], o2o_only=True)["o2o"]
        return out["cls"], out["box"]


@torch.inference_mode()
def convert(
    checkpoint_path,
    output_dir=Path(__file__).parent / "runtime",
    img_size=640,
    only_backbone=False,
    precision="float32",
    calibration_data=None,
):
    if precision not in {"float32", "float16", "int8"}:
        raise ValueError("precision must be float32, float16 or int8")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    metadata = validate_metadata(checkpoint_path, checkpoint)

    if img_size <= 0 or img_size % max(metadata["architecture"]["strides"]):
        raise ValueError("img_size must be divisible by max stride")

    model = NMSFreeDetector(
        **metadata["architecture"],
        img_size=img_size,
    ).float().eval()

    model.load_state_dict(checkpoint.get("ema") or checkpoint["model"])

    x = torch.zeros(1, 3, img_size, img_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backbone_neck = BackboneNeck(model).eval()

    exports = (
        [("backbone.onnx", model.backbone, (x,), ["images"], ["p3", "p4", "p5"])]
        if only_backbone
        else [
            ("backbone_neck.onnx", backbone_neck, (x,), ["images"], ["p3", "p4", "p5"]),
            ("head.onnx", Head(model).eval(), backbone_neck(x),
             ["p3", "p4", "p5"], ["logits", "boxes"]),
        ]
    )

    for name, module, inputs, input_names, output_names in exports:
        path = output_dir / name

        torch.onnx.export(
            module,
            inputs,
            path,
            opset_version=17,
            dynamo=False,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes={
                name: {0: "batch_size"}
                for name in input_names + output_names
            },
        )

        if precision == "float16":
            from onnxruntime.transformers.float16 import convert_float_to_float16

            graph = onnx.load(path)
            graph = convert_float_to_float16(graph, keep_io_types=False)
            onnx.save(graph, path)

        elif precision == "int8":
            if calibration_data is None:
                raise ValueError("INT8 requires calibration_data")

            from onnxruntime.quantization import (
                CalibrationDataReader,
                QuantFormat,
                QuantType,
                quantize_static,
            )

            class Reader(CalibrationDataReader):
                def __init__(self, batches):
                    self.batches = iter(batches)

                def get_next(self):
                    return next(self.batches, None)

            def batches():
                for batch in calibration_data:
                    batch = torch.as_tensor(batch).cpu().float()

                    values = (
                        backbone_neck(batch)
                        if name == "head.onnx"
                        else (batch,)
                    )

                    yield {
                        key: value.numpy()
                        for key, value in zip(input_names, values)
                    }

            fp32_path = path.with_suffix(".fp32.onnx")
            path.rename(fp32_path)

            quantize_static(
                str(fp32_path),
                str(path),
                Reader(batches()),
                quant_format=QuantFormat.QDQ,
                activation_type=QuantType.QInt8,
                weight_type=QuantType.QInt8,
                per_channel=True,
                op_types_to_quantize=["Conv", "MatMul", "Gemm"],
            )

            fp32_path.unlink()

        onnx.checker.check_model(onnx.load(path), full_check=True)
        print(f"Exported {precision}: {path}")

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    checkpoint_path = r"D:\Project\DATN\driver-guardian\ai\ObjectDetection_2p6M\checkpoints\2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031\de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310\finetune\best.pt"
    output_dir = root / "runtime"
    img_size = 640
    only_backbone = True
    precision = "float32"
    # Phần phục vụ chuyển đổi theo int8
    # calibration_data = (
    #     images
    #     for i, (images, _) in enumerate(loader)
    #     if i < 20
    # )
    convert(
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
        img_size=img_size,
        only_backbone=only_backbone,
        precision=precision,
    )
