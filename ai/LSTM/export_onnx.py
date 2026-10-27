import json
import logging
from pathlib import Path
import sys

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.LSTM.src.checkpoint import load_model
from ai.LSTM.src.models_convgru import ConvGRUCell

logger = logging.getLogger(__name__)

# ==============================================================================
# CONFIGURATIONS
# ==============================================================================
CHECKPOINT_PATH = Path(r"D:\Project\DATN\tempt\driver-guardian\ai\LSTM\checkpoints\head_landmarkv1\best.pt")
OUTPUT_DIR = Path(__file__).resolve().parent / "checkpoints" / "head_landmarkv1"
MODEL_BASE_NAME = "drowsy_classifier"
METADATA_JSON_NAME = "drowsy_classifier_metadata.json"

# Quantization and precision formats to export:
# Options can be any subset or list of: ["float32", "float16", "int8"]
# (aliases: "fp32" -> "float32", "fp16" -> "float16")
EXPORT_PRECISIONS = ["float32", "float16", "int8"]

OPSET_VERSION = 17
IMAGE_SIZE = 224
NECK_P4_CHANNELS = 128
VERIFY_ONNX = True

# Verification / Sanity test settings
TEST_VIDEO_FRAMES = 50
TEST_P4_OLD_FRAMES = 49


# ==============================================================================


class ScriptableConvGRU(nn.Module):
    """ConvGRU loop & attention pooling designed for TorchScript & dynamic ONNX Loop export."""

    def __init__(
            self,
            convgru: ConvGRUCell,
            spatial_attention: nn.Conv2d,
            fc_out: nn.Sequential,
            hidden_dim: int,
    ):
        super().__init__()
        self.convgru = convgru
        self.spatial_attention = spatial_attention
        self.fc_out = fc_out
        self.hidden_dim = hidden_dim

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # features shape: (1, T, input_dim, H, W)
        b = features.shape[0]
        t = features.shape[1]
        h_dim = features.shape[3]
        w_dim = features.shape[4]

        h = torch.zeros(
            b,
            self.hidden_dim,
            h_dim,
            w_dim,
            dtype=features.dtype,
            device=features.device,
        )
        vectors = torch.jit.annotate(list[torch.Tensor], [])

        for i in range(t):
            h = self.convgru(features[:, i], h)
            att = F.softmax(self.spatial_attention(h).flatten(2), dim=-1)
            att = att.reshape(b, 1, h_dim, w_dim)
            vectors.append((h * att).sum(dim=(-2, -1)))

        stacked = torch.stack(vectors, dim=1)
        mean_vec = stacked.mean(dim=1)
        logits = self.fc_out(mean_vec)
        return F.softmax(logits, dim=1)


class ExportDrowsyClassifier(nn.Module):
    """Wrapper encapsulating BackboneNeck, Adapter, and ConvGRU for ONNX export and inference.

    Inputs:
        x: Video frames tensor of shape (T, 3, H, W). Dtype can be uint8 [0..255] or matching float.
        p4_old: Previous P4 feature tensor of shape (T_old, C, H, W).
                If no old features are available, pass an empty tensor of shape (0, C, H, W)
                or a 1D indicator such as torch.tensor([0]).

    Outputs:
        prob: Softmax probabilities of shape (1, num_classes) [alert, drowsy].
        p4_features: Concatenated P4 features of shape (T_old + T, C, H, W).
    """

    def __init__(self, checkpoint: str | Path | None = CHECKPOINT_PATH):
        super().__init__()
        model, _ = load_model(checkpoint)
        self.backbone_neck = model.backbone_neck
        self.adapter = model.conv_gru.adapter
        self.gru_classifier = ScriptableConvGRU(
            model.conv_gru.convgru,
            model.conv_gru.spatial_attention,
            model.conv_gru.fc_out,
            model.conv_gru.hidden_dim,
        )

    def forward(
            self,
            x: torch.Tensor,
            p4_old: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Handle uint8 or float frame inputs matching weight dtype
        param_dtype = self.adapter[0].weight.dtype
        if x.dtype == torch.uint8:
            frames = x.to(param_dtype) / 255.0
        else:
            frames = x.to(param_dtype)

        # Extract P4 features: (T, C, H, W)
        p4 = self.backbone_neck(frames)[1]

        # Normalize p4_old: if 1D / dummy / empty tensor, convert to 4D empty tensor
        if p4_old.dim() != 4:
            p4_norm = torch.zeros(
                0,
                p4.shape[1],
                p4.shape[2],
                p4.shape[3],
                dtype=p4.dtype,
                device=x.device,
            )
        else:
            p4_norm = p4_old.to(p4.dtype)

        # Concatenate cached features and new features along sequence dimension
        features = torch.cat([p4_norm, p4], dim=0)

        # Adapt P4 channels to ConvGRU input_dim
        adapted = self.adapter(features)

        # Add batch dimension for ConvGRU: (1, total_T, input_dim, H, W)
        adapted_seq = adapted.unsqueeze(0)

        probs = self.gru_classifier(adapted_seq)
        return probs, features


def export_fp32(
        model: nn.Module,
        output_path: Path,
        opset_version: int = OPSET_VERSION,
        image_size: int = IMAGE_SIZE,
        neck_p4_channels: int = NECK_P4_CHANNELS,
) -> Path:
    """Exports model in FP32 (single precision)."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = model.eval().float()
    scripted_model = torch.jit.script(m)

    feat_spatial = image_size // 16
    dummy_video = torch.randint(
        0, 255, (25, 3, image_size, image_size), dtype=torch.uint8
    )
    dummy_p4_old = torch.zeros(
        0, neck_p4_channels, feat_spatial, feat_spatial, dtype=torch.float32
    )

    logger.info("Exporting FP32 ONNX model to %s...", output_path)
    torch.onnx.export(
        scripted_model,
        (dummy_video, dummy_p4_old),
        str(output_path),
        input_names=["video", "p4_old"],
        output_names=["prob", "p4_features"],
        dynamic_axes={
            "video": {0: "num_frames"},
            "p4_old": {0: "num_old_frames"},
            "p4_features": {0: "total_frames"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    import onnx
    onnx_proto = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_proto)
    return output_path


def export_fp16(
        model: nn.Module,
        output_path: Path,
        opset_version: int = OPSET_VERSION,
        image_size: int = IMAGE_SIZE,
        neck_p4_channels: int = NECK_P4_CHANNELS,
) -> Path:
    """Exports model in FP16 (half precision)."""
    import copy

    output_path.parent.mkdir(parents=True, exist_ok=True)
    m = copy.deepcopy(model).eval().half()
    scripted_model = torch.jit.script(m)

    feat_spatial = image_size // 16
    dummy_video = torch.randint(
        0, 255, (25, 3, image_size, image_size), dtype=torch.uint8
    )
    dummy_p4_old = torch.zeros(
        0, neck_p4_channels, feat_spatial, feat_spatial, dtype=torch.float16
    )

    logger.info("Exporting FP16 ONNX model to %s...", output_path)
    torch.onnx.export(
        scripted_model,
        (dummy_video, dummy_p4_old),
        str(output_path),
        input_names=["video", "p4_old"],
        output_names=["prob", "p4_features"],
        dynamic_axes={
            "video": {0: "num_frames"},
            "p4_old": {0: "num_old_frames"},
            "p4_features": {0: "total_frames"},
        },
        opset_version=opset_version,
        dynamo=False,
    )

    import onnx
    onnx_proto = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_proto)
    return output_path


def export_int8(
        fp32_onnx_path: Path,
        output_path: Path,
) -> Path:
    """Quantizes an FP32 ONNX model to dynamic INT8."""
    import onnxruntime.quantization as ort_quant

    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Quantizing to INT8 ONNX model: %s -> %s...", fp32_onnx_path.name, output_path.name)
    ort_quant.quantize_dynamic(
        model_input=str(fp32_onnx_path),
        model_output=str(output_path),
        weight_type=ort_quant.QuantType.QInt8,
        per_channel=True,
        reduce_range=False,
    )

    import onnx
    onnx_proto = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_proto)
    return output_path


def generate_metadata_json(
        output_json_path: Path,
        exported_files: dict[str, Path],
        checkpoint_path: Path,
        opset_version: int = OPSET_VERSION,
        image_size: int = IMAGE_SIZE,
        neck_p4_channels: int = NECK_P4_CHANNELS,
) -> Path:
    """Generates a detailed JSON metadata manifest documenting inputs, outputs, quantizations, and deployment guides."""
    feat_spatial = image_size // 16
    metadata = {
        "model_name": "DrowsyClassifier",
        "version": "1.0.0",
        "description": "Video RGB -> Backbone/PAFPN Neck (P4) -> ConvGRU -> Spatial Attention Pooling -> Drowsiness classification (Alert vs Drowsy)",
        "source_checkpoint": checkpoint_path.name,
        "opset_version": opset_version,
        "inputs": [
            {
                "name": "video",
                "description": "Input video frames in RGB format. Values can be uint8 in [0, 255] or matching float in [0.0, 1.0].",
                "shape": ["num_frames", 3, image_size, image_size],
                "dynamic_axes": {"0": "num_frames (T >= 1)"},
                "dtype": {
                    "float32": "uint8 or float32",
                    "float16": "uint8 or float16",
                    "int8": "uint8 or float32",
                },
            },
            {
                "name": "p4_old",
                "description": "Cached P4 feature map from previous inference window. Pass empty tensor of shape (0, 128, 14, 14) when starting or when no cache is available.",
                "shape": ["num_old_frames", neck_p4_channels, feat_spatial, feat_spatial],
                "dynamic_axes": {"0": "num_old_frames (T_old >= 0)"},
                "dtype": {
                    "float32": "float32",
                    "float16": "float16",
                    "int8": "float32",
                },
            },
        ],
        "outputs": [
            {
                "name": "prob",
                "description": "Softmax probability distribution over drowsiness classes.",
                "shape": [1, 2],
                "classes": {
                    "0": "alert",
                    "1": "drowsy",
                },
                "dtype": {
                    "float32": "float32",
                    "float16": "float16",
                    "int8": "float32",
                },
            },
            {
                "name": "p4_features",
                "description": "Concatenated P4 feature sequence (previous cached features + new extracted features). Maintain the trailing frames to pass as p4_old in the next step.",
                "shape": ["total_frames", neck_p4_channels, feat_spatial, feat_spatial],
                "dynamic_axes": {"0": "total_frames (T_old + T)"},
                "dtype": {
                    "float32": "float32",
                    "float16": "float16",
                    "int8": "float32",
                },
            },
        ],
        "exported_models": {
            prec: {
                "file_name": path.name,
                "file_size_mb": round(path.stat().st_size / (1024 * 1024), 2),
                "format_details": {
                    "float32": "Single precision FP32 baseline, highest accuracy.",
                    "float16": "Half precision FP16 weights & ops, 50% size reduction, optimal for NVIDIA GPU (TensorRT/CUDA) and mobile NPUs.",
                    "int8": "Dynamic INT8 quantized weights (QInt8), 51% size reduction, optimal for CPU edge devices with AVX/VNNI.",
                }.get(prec.split()[0], "Standard ONNX model."),
            }
            for prec, path in exported_files.items()
        },
        "deployment_guide": {
            "preprocessing": {
                "color_space": "RGB",
                "frame_size": [image_size, image_size],
                "resize_method": "Letterbox or standard resize to 224x224 maintaining aspect ratio",
                "normalization": "Division by 255.0 is handled inside the ONNX graph if uint8 tensor is provided.",
                "target_fps": 5,
                "recommended_window_sec": 10,
                "recommended_window_frames": 50,
            },
            "streaming_inference_steps": [
                "1. Initialize ONNX Runtime session: sess = ort.InferenceSession('drowsy_classifier.onnx', providers=['CPUExecutionProvider']).",
                "2. First step (No prior features): Provide p4_old as an empty array of shape (0, 128, 14, 14) and float32 (or float16 if using FP16 model).",
                "3. Subsequent steps (Sliding window): Receive new video chunk (e.g. 25 frames). Pass the latest frames as 'video' and pass the cached 'p4_features' from previous step (e.g. last 25 frames) as 'p4_old'.",
                "4. Postprocessing: Read prob[0, 1] as the drowsy score. If score > threshold (e.g. 0.5), trigger drowsiness alert.",
            ],
            "code_example_python": (
                "import numpy as np\n"
                "import onnxruntime as ort\n\n"
                "# 1. Load session\n"
                "sess = ort.InferenceSession('drowsy_classifier.onnx')\n\n"
                "# 2. Initial call (step 1)\n"
                "video_chunk = np.random.randint(0, 255, (25, 3, 224, 224), dtype=np.uint8)\n"
                "p4_cache = np.zeros((0, 128, 14, 14), dtype=np.float32)\n"
                "prob, p4_cache = sess.run(None, {'video': video_chunk, 'p4_old': p4_cache})\n"
                "print('Initial Drowsy prob:', prob[0, 1])\n\n"
                "# 3. Next step (step 2 - sliding window)\n"
                "next_video_chunk = np.random.randint(0, 255, (25, 3, 224, 224), dtype=np.uint8)\n"
                "# Keep at most 25 trailing frames from cache so total frames <= 50 (10 sec @ 5 fps)\n"
                "p4_cache = p4_cache[-25:]\n"
                "prob, p4_cache = sess.run(None, {'video': next_video_chunk, 'p4_old': p4_cache})\n"
                "print('Updated Drowsy prob:', prob[0, 1])"
            ),
        },
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Metadata JSON manifest generated: %s", output_json_path)
    return output_json_path


def verify_onnx(
        onnx_path: Path,
        precision: str,
        model_fp32: ExportDrowsyClassifier,
) -> None:
    """Verifies ONNX inference on empty p4_old, with p4_old, and dynamic sequence length."""
    import numpy as np
    import onnxruntime as ort

    logger.info("[%s] Verifying ONNX model with ONNX Runtime: %s...", precision.upper(), onnx_path.name)
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    feat_spatial = IMAGE_SIZE // 16

    np_dtype = np.float16 if precision in ("float16", "fp16") else np.float32
    pt_dtype = torch.float16 if precision in ("float16", "fp16") else torch.float32

    # Case 1: Without p4_old
    v1 = torch.randint(0, 255, (TEST_VIDEO_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    p0_np = np.zeros((0, NECK_P4_CHANNELS, feat_spatial, feat_spatial), dtype=np_dtype)

    ort_prob1, ort_feat1 = sess.run(None, {"video": v1.numpy(), "p4_old": p0_np})
    assert ort_prob1.shape == (1, 2), f"Unexpected prob shape: {ort_prob1.shape}"
    assert ort_feat1.shape == (TEST_VIDEO_FRAMES, NECK_P4_CHANNELS, feat_spatial, feat_spatial)

    # Parity check against PyTorch for float32 and float16
    import copy
    with torch.no_grad():
        m = copy.deepcopy(model_fp32).half() if precision in ("float16", "fp16") else model_fp32
        pt_prob1, pt_feat1 = m(v1, torch.zeros(0, NECK_P4_CHANNELS, feat_spatial, feat_spatial, dtype=pt_dtype))

    diff_prob1 = float(np.max(np.abs(pt_prob1.cpu().numpy() - ort_prob1)))
    diff_feat1 = float(np.max(np.abs(pt_feat1.cpu().numpy() - ort_feat1)))
    logger.info("[%s] Case 1 (No p4_old): prob diff = %.2e, feat diff = %.2e", precision.upper(), diff_prob1,
                diff_feat1)

    tol = 1e-2 if precision in ("float16", "fp16") else (0.15 if precision == "int8" else 1e-4)
    assert diff_prob1 < tol, f"Prob diff exceeded tolerance ({diff_prob1} >= {tol})"

    # Case 2: With p4_old
    p_old_np = np.random.randn(TEST_P4_OLD_FRAMES, NECK_P4_CHANNELS, feat_spatial, feat_spatial).astype(np_dtype)
    ort_prob2, ort_feat2 = sess.run(None, {"video": v1.numpy(), "p4_old": p_old_np})
    total_frames = TEST_P4_OLD_FRAMES + TEST_VIDEO_FRAMES
    assert ort_prob2.shape == (1, 2)
    assert ort_feat2.shape == (total_frames, NECK_P4_CHANNELS, feat_spatial, feat_spatial)

    with torch.no_grad():
        pt_prob2, pt_feat2 = m(v1, torch.as_tensor(p_old_np))
    diff_prob2 = float(np.max(np.abs(pt_prob2.cpu().numpy() - ort_prob2)))
    diff_feat2 = float(np.max(np.abs(pt_feat2.cpu().numpy() - ort_feat2)))
    logger.info("[%s] Case 2 (With p4_old): prob diff = %.2e, feat diff = %.2e", precision.upper(), diff_prob2,
                diff_feat2)
    assert diff_prob2 < tol, f"Prob diff exceeded tolerance ({diff_prob2} >= {tol})"

    # Case 3: Arbitrary dynamic sequence length
    v_short = np.random.randint(0, 255, (10, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)
    p35_np = np.random.randn(35, NECK_P4_CHANNELS, feat_spatial, feat_spatial).astype(np_dtype)
    ort_prob3, ort_feat3 = sess.run(None, {"video": v_short, "p4_old": p35_np})
    assert ort_prob3.shape == (1, 2)
    assert ort_feat3.shape == (45, NECK_P4_CHANNELS, feat_spatial, feat_spatial)
    logger.info("[%s] Case 3 (10 video + 35 p4_old = 45 total): verified!", precision.upper())


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found at: {CHECKPOINT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Loading base PyTorch model from %s...", CHECKPOINT_PATH)
    model = ExportDrowsyClassifier(checkpoint=CHECKPOINT_PATH).eval()

    # PyTorch sanity check
    logger.info("Running PyTorch forward pass sanity check...")
    video = torch.randint(0, 255, (TEST_VIDEO_FRAMES, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
    feat_spatial = IMAGE_SIZE // 16

    # 1. Backward-compatible call with 1D indicator torch.tensor([0])
    prob_p0, feat_p0 = model(video, torch.tensor([0]))
    logger.info("PyTorch call with tensor([0]) (no p4_old): prob=%s, feat=%s", tuple(prob_p0.shape),
                tuple(feat_p0.shape))

    # 2. Call with empty tensor
    prob_empty, feat_empty = model(video, torch.zeros(0, NECK_P4_CHANNELS, feat_spatial, feat_spatial))
    logger.info("PyTorch call with zeros(0, %d, %d, %d): prob=%s, feat=%s", NECK_P4_CHANNELS, feat_spatial,
                feat_spatial, tuple(prob_empty.shape), tuple(feat_empty.shape))

    # 3. Call with old P4 features
    p4_old = torch.randn(TEST_P4_OLD_FRAMES, NECK_P4_CHANNELS, feat_spatial, feat_spatial)
    prob_p_old, feat_p_old = model(video, p4_old)
    logger.info("PyTorch call with %d p4_old frames: prob=%s, feat=%s", TEST_P4_OLD_FRAMES, tuple(prob_p_old.shape),
                tuple(feat_p_old.shape))

    # Normalize precision list
    raw_precisions = [EXPORT_PRECISIONS] if isinstance(EXPORT_PRECISIONS, str) else EXPORT_PRECISIONS
    precisions = set()
    for p in raw_precisions:
        pl = p.strip().lower()
        if pl in ("float32", "fp32"):
            precisions.add("float32")
        elif pl in ("float16", "fp16"):
            precisions.add("float16")
        elif pl in ("int8", "qint8"):
            precisions.add("int8")
        else:
            raise ValueError(f"Unsupported precision: {p}. Choose from: float32, float16, int8")

    exported_files: dict[str, Path] = {}
    fp32_path = OUTPUT_DIR / f"{MODEL_BASE_NAME}_fp32.onnx"

    # Export FP32 (needed also if INT8 is requested)
    if "float32" in precisions or "int8" in precisions:
        export_fp32(
            model=model,
            output_path=fp32_path,
            opset_version=OPSET_VERSION,
            image_size=IMAGE_SIZE,
            neck_p4_channels=NECK_P4_CHANNELS,
        )
        if "float32" in precisions:
            exported_files["float32"] = fp32_path
            # Also create standard default name
            default_onnx = OUTPUT_DIR / f"{MODEL_BASE_NAME}.onnx"
            if default_onnx != fp32_path:
                import shutil
                shutil.copyfile(fp32_path, default_onnx)
                exported_files["default (fp32)"] = default_onnx

    # Export FP16
    if "float16" in precisions:
        fp16_path = OUTPUT_DIR / f"{MODEL_BASE_NAME}_fp16.onnx"
        export_fp16(
            model=model,
            output_path=fp16_path,
            opset_version=OPSET_VERSION,
            image_size=IMAGE_SIZE,
            neck_p4_channels=NECK_P4_CHANNELS,
        )
        exported_files["float16"] = fp16_path

    # Export INT8
    if "int8" in precisions:
        int8_path = OUTPUT_DIR / f"{MODEL_BASE_NAME}_int8.onnx"
        export_int8(fp32_onnx_path=fp32_path, output_path=int8_path)
        exported_files["int8"] = int8_path

    # Verify exported ONNX models
    if VERIFY_ONNX:
        for precision_name, file_path in exported_files.items():
            canonical_prec = "float32" if "fp32" in precision_name or "float32" in precision_name else precision_name
            verify_onnx(file_path, canonical_prec, model)

    # Generate metadata JSON manifest
    metadata_path = OUTPUT_DIR / METADATA_JSON_NAME
    generate_metadata_json(
        output_json_path=metadata_path,
        exported_files=exported_files,
        checkpoint_path=CHECKPOINT_PATH,
        opset_version=OPSET_VERSION,
        image_size=IMAGE_SIZE,
        neck_p4_channels=NECK_P4_CHANNELS,
    )

    # Print summary table
    logger.info("=" * 70)
    logger.info("ONNX EXPORT & QUANTIZATION SUMMARY:")
    for prec, path in exported_files.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(" - %-16s: %-50s (%6.2f MB)", prec, path.name, size_mb)
    logger.info(" - %-16s: %-50s", "metadata JSON", metadata_path.name)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
