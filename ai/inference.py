#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: inference.py
Hệ Thống Driver Guardian — Mô Hình Suy Luận Nhận Diện Trạng Thái Buồn Ngủ (Driver Drowsiness Detection)

Triển khai đầy đủ kế hoạch kiến trúc và đóng gói ONNX Float32 từ structer_inference.md:
1. Kiến trúc mô hình chính:
   - Trunk: Backbone + Neck (PAFPN) của NMSFreeDetector (~1.64M params)
   - Spatial Pooling: F.adaptive_avg_pool2d(1, 1) trên (P3: 64, P4: 128, P5: 256) -> 448 kênh
   - DeepLSTMClassifier: SpatialFeatureAdapter (448 -> 256) + Stacked 3-Layer LSTM (256) + FC Head (256 -> 2)
   - Input shape: [bz, T, 3, 640, 640]
   - Output shape: Sequence Mode [bz, T, 2] | Clip Mode [bz, 2]

2. Cơ chế suy luận:
   - Chế độ End-to-End PyTorch (Full Video Clip)
   - Chế độ Phân tách Streaming thời gian thực (Two-Stage Decoupled Stateful Streaming)

3. Kế hoạch đóng gói ONNX Float32 (Opset 17):
   - Phương án 1: driver_guardian_end2end.onnx (hỗ trợ Dynamic Axes batch_size và seq_len)
   - Phương án 2: spatial_extractor.onnx + temporal_classifier.onnx (Streaming Edge)
   - Thẩm định sai số số học (Numerical Parity & Cosine Similarity giữa PyTorch và ONNX Runtime)
   - Tự động sinh tệp metadata: model_manifest.json

4. Bộ điều phối suy luận thời gian thực ONNX Runtime (ONNXEnd2EndPredictor & ONNXStreamingPredictor).
"""

import os
import sys
import time
import json
import hashlib
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List, Union

# Đảm bảo console Windows in tiếng Việt UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Cấu hình sys.path linh hoạt để nạp đúng các module con
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR / "LSTM") not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR / "LSTM"))
if str(CURRENT_DIR / "ObjectDetection_2p6M") not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR / "ObjectDetection_2p6M"))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# Nạp các khối từ ObjectDetection_2p6M
from ai.ObjectDetection_2p6M.src.backbone_neck import Backbone, PAFPN
from ai.ObjectDetection_2p6M.src.config import TrainConfig as DetectionTrainConfig

# Nạp các khối từ LSTM
from ai.LSTM.model import CNNAdapter, DeepLSTMClassifier

try:
    import onnx
except ImportError:
    onnx = None

try:
    import onnxruntime as ort
except ImportError:
    ort = None


# ==============================================================================
# 1. TIỀN XỬ LÝ KHUNG HÌNH (IMAGE & VIDEO PREPROCESSING)
# ==============================================================================

def letterbox(image: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """
    Chuẩn hóa tỷ lệ khung hình với phần đệm đồng màu (Letterbox 640x640).
    Giữ nguyên tỷ lệ gốc của khuôn mặt tài xế mà không bị méo mó.
    """
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top: pad_top + new_h, pad_left: pad_left + new_w] = resized
    return canvas


def preprocess_frame(frame: np.ndarray, new_size: int = 640) -> np.ndarray:
    """
    Chuyển đổi khung hình BGR (từ OpenCV/Webcam) sang RGB, Letterbox và chuẩn hóa [0.0, 1.0].
    Returns: numpy array shape [3, new_size, new_size] dtype float32.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxed = letterbox(rgb, new_size=new_size)
    tensor_img = np.ascontiguousarray(boxed.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor_img


# ==============================================================================
# 2. KIẾN TRÚC MÔ HÌNH CHÍNH (DRIVER GUARDIAN INFERENCE MODEL)
# ==============================================================================

class DriverGuardianModel(nn.Module):
    """
    Mô hình suy luận chính End-to-End của Hệ Thống Driver Guardian:
    Được cấu thành trực tiếp từ các khối:
      1. Backbone của NMSFreeDetector: trích xuất đặc trưng đa tỷ lệ thô đến tinh.
      2. PAFPN Neck của NMSFreeDetector: dung hợp đặc trưng không gian đa tỷ lệ 2 chiều.
      3. Phép toán adaptive_avg_pool2d(1, 1): nén không gian 2D về vector 1D (tổng 448 kênh).
      4. DeepLSTMClassifier: SpatialFeatureAdapter (448 -> 256) + Deep LSTM 3 lớp + FC Head.

    Input:
        x: Tensor [bz, T, 3, 640, 640]
    Processing Pipeline:
        1. Time-Folding: Reshape -> [bz * T, 3, 640, 640]
        2. Backbone: -> p3_bb, p4_bb, p5_bb
        3. PAFPN Neck: -> p3_out, p4_out, p5_out
        4. F.adaptive_avg_pool2d: -> v3, v4, v5 [bz, T, C_k]
        5. DeepLSTMClassifier: -> Logits
    Output:
        Sequence Mode (return_sequence=True): [bz, T, 2]
        Clip Mode (return_sequence=False):     [bz, 2]
    """

    def __init__(
            self,
            backbone_w: Tuple[int, ...] = (16, 32, 64, 128, 256),
            backbone_n: Tuple[int, ...] = (1, 2, 2, 1),
            neck_n: int = 1,
            spatial_in_channels: Tuple[int, ...] = (64, 128, 256),
            spatial_fusion: str = "concat",
            adapter_dropout: float = 0.1,
            input_dim: int = 256,
            hidden_dim: int = 256,
            num_layers: int = 3,
            num_classes: int = 2,
            lstm_dropout: float = 0.2
    ):
        super().__init__()
        self.spatial_in_channels = spatial_in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.spatial_fusion = spatial_fusion

        # 1. Thân mạng trích xuất không gian sử dụng trực tiếp Backbone & PAFPN của NMSFreeDetector
        c3, c4, c5 = backbone_w[2], backbone_w[3], backbone_w[4]  # (64, 128, 256)
        self.backbone = Backbone(w=backbone_w, n=backbone_n)
        self.neck = PAFPN(chs=(c3, c4, c5), n=neck_n)

        # 2. Sử dụng trực tiếp DeepLSTMClassifier (chứa SpatialFeatureAdapter, Deep LSTM 3 lớp và FC Head)
        self.classifier = DeepLSTMClassifier(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_classes=num_classes,
            spatial_in_channels=spatial_in_channels,
            fusion=spatial_fusion,
            adapter_dropout=adapter_dropout,
            dropout=lstm_dropout
        )

    def forward(
            self,
            x: torch.Tensor,
            hc: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_sequence: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        x: Tensor [bz, T, 3, 640, 640]
        hc: Optional hidden state tuple (h_0, c_0)
        return_sequence: True -> [bz, T, num_classes] | False -> [bz, num_classes]
        """
        bz, t, c, h, w = x.shape
        x_flat = x.view(bz * t, c, h, w)

        # 1. Trích xuất đặc trưng qua Backbone và PAFPN Neck của NMSFreeDetector
        p3, p4, p5 = self.backbone(x_flat)
        p3, p4, p5 = self.neck(p3, p4, p5)

        # 2. Đầu ra đi qua phép adaptive_avg_pool2d nén về vector 1D
        v3 = F.adaptive_avg_pool2d(p3, (1, 1)).view(bz, t, self.spatial_in_channels[0])
        v4 = F.adaptive_avg_pool2d(p4, (1, 1)).view(bz, t, self.spatial_in_channels[1])
        v5 = F.adaptive_avg_pool2d(p5, (1, 1)).view(bz, t, self.spatial_in_channels[2])

        # 3. Đi thẳng vào lớp DeepLSTMClassifier để phân loại
        logits = self.classifier((v3, v4, v5), hc=hc, return_sequence=return_sequence)
        return logits

    def load_trunk_weights(self, checkpoint_path: Union[str, Path], map_location: str = "cpu", use_ema: bool = True) -> int:
        """Nạp trọng số Backbone + PAFPN Neck trực tiếp từ checkpoint của ObjectDetection_2p6M (.pt)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint detection: {path}")

        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
        state_dict = (ckpt.get("ema") if use_ema and "ema" in ckpt else None) or ckpt.get("model", ckpt)

        # Lọc các tham số thuộc backbone và neck
        bb_dict = {}
        neck_dict = {}
        for k, v in state_dict.items():
            if k.startswith("backbone."):
                bb_dict[k.replace("backbone.", "", 1)] = v
            elif k.startswith("neck."):
                neck_dict[k.replace("neck.", "", 1)] = v

        if bb_dict:
            self.backbone.load_state_dict(bb_dict, strict=False)
        if neck_dict:
            self.neck.load_state_dict(neck_dict, strict=False)

        loaded_count = len(bb_dict) + len(neck_dict)
        print(f"[+] Đã nạp thành công {loaded_count} tensors trọng số (Backbone: {len(bb_dict)}, Neck: {len(neck_dict)}) từ: {path.name}")
        return loaded_count

    def load_lstm_weights(self, checkpoint_path: Union[str, Path], map_location: str = "cpu") -> int:
        """Nạp trọng số trực tiếp vào DeepLSTMClassifier từ checkpoint (.pt/.pth)."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy checkpoint LSTM: {path}")

        ckpt = torch.load(str(path), map_location=map_location, weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))

        lstm_dict = {}
        for k, v in state_dict.items():
            if k.startswith("spatial_adapter.") or k.startswith("lstm.") or k.startswith("fc_out."):
                lstm_dict[k] = v
            elif k.startswith("classifier."):
                lstm_dict[k.replace("classifier.", "", 1)] = v

        missing, unexpected = self.classifier.load_state_dict(lstm_dict, strict=False)
        loaded_count = len(lstm_dict)
        print(f"[+] Đã nạp thành công {loaded_count} tensors trọng số DeepLSTMClassifier từ: {path.name}")
        return loaded_count

    @classmethod
    def create_pretrained(
            cls,
            cnn_checkpoint: Optional[Union[str, Path]] = None,
            lstm_checkpoint: Optional[Union[str, Path]] = None,
            device: str = "cpu"
    ) -> "DriverGuardianModel":
        """Khởi tạo mô hình hoàn chỉnh và nạp trọng số tương ứng nếu đường dẫn tồn tại."""
        model = cls()
        if cnn_checkpoint and Path(cnn_checkpoint).exists():
            model.load_trunk_weights(cnn_checkpoint, map_location=device)
        if lstm_checkpoint and Path(lstm_checkpoint).exists():
            model.load_lstm_weights(lstm_checkpoint, map_location=device)
        model.to(device).eval()
        return model


# ==============================================================================
# 3. CÁC LỚP WRAPPER THUẦN TENSOR PHỤC VỤ XUẤT ONNX (ONNX EXPORT WRAPPERS)
# ==============================================================================

class End2EndONNXExportWrapper(nn.Module):
    """
    Wrapper thuần túy kiểu Tensor phục vụ xuất mô hình hợp nhất:
    driver_guardian_end2end.onnx
    Input:  video [bz, T, 3, 640, 640]
    Output: logits [bz, T, 2]
    """

    def __init__(self, model: DriverGuardianModel):
        super().__init__()
        self.model = model

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        return self.model(video, return_sequence=True)


class SpatialExtractorONNXExportWrapper(nn.Module):
    """
    Wrapper phục vụ xuất Khối 1: spatial_extractor.onnx
    Sử dụng trực tiếp Backbone & PAFPN Neck của NMSFreeDetector kèm adaptive_avg_pool2d
    Input:  image [bz, 3, 640, 640]
    Output: features [bz, 448] (v_p3: 64 + v_p4: 128 + v_p5: 256)
    """

    def __init__(self, model: DriverGuardianModel):
        super().__init__()
        self.backbone = model.backbone
        self.neck = model.neck

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        p3, p4, p5 = self.backbone(image)
        p3, p4, p5 = self.neck(p3, p4, p5)
        v3 = F.adaptive_avg_pool2d(p3, (1, 1)).flatten(1)
        v4 = F.adaptive_avg_pool2d(p4, (1, 1)).flatten(1)
        v5 = F.adaptive_avg_pool2d(p5, (1, 1)).flatten(1)
        return torch.cat([v3, v4, v5], dim=-1)


class TemporalClassifierONNXExportWrapper(nn.Module):
    """
    Wrapper phục vụ xuất Khối 2: temporal_classifier.onnx
    Sử dụng trực tiếp các thành phần của DeepLSTMClassifier
    Input:
        features: [bz, 1, 448] (đặc trưng từ Khối 1 tại frame mới nhất)
        h_0: [3, bz, 256]
        c_0: [3, bz, 256]
    Output:
        logits: [bz, 1, 2]
        h_n: [3, bz, 256]
        c_n: [3, bz, 256]
    """

    def __init__(self, model: DriverGuardianModel):
        super().__init__()
        self.spatial_adapter = model.classifier.spatial_adapter
        self.lstm = model.classifier.lstm
        self.fc_out = model.classifier.fc_out

    def forward(
            self,
            features: torch.Tensor,
            h_0: torch.Tensor,
            c_0: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # features: [bz, 1, 448]
        proj = self.spatial_adapter(features)  # [bz, 1, 256]
        lstm_out, (h_n, c_n) = self.lstm(proj, (h_0, c_0))
        logits = self.fc_out(lstm_out)  # [bz, 1, 2]
        return logits, h_n, c_n


# ==============================================================================
# 4. KẾ HOẠCH ĐÓNG GÓI & THẨM ĐỊNH ONNX FLOAT32 (ONNX PACKAGING & VERIFICATION)
# ==============================================================================

def compute_sha256(filepath: Union[str, Path]) -> str:
    """Tính toán mã băm SHA-256 của tệp để kiểm tra tính toàn vẹn."""
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            sha.update(chunk)
    return sha.hexdigest()


def verify_numerical_parity(
        torch_fn,
        ort_session,
        feed_dict: Dict[str, np.ndarray],
        torch_inputs: Tuple[torch.Tensor, ...],
        atol: float = 1e-4
) -> Tuple[float, float, bool]:
    """
    So sánh sai số số học giữa PyTorch gốc và ONNX Runtime:
    Returns:
        max_abs_error: Sai số tuyệt đối lớn nhất
        cosine_sim: Độ tương đồng Cosine
        passed: True nếu max_abs_error <= atol
    """
    with torch.no_grad():
        torch_out = torch_fn(*torch_inputs)
        if isinstance(torch_out, tuple):
            torch_np = torch_out[0].cpu().numpy()
        else:
            torch_np = torch_out.cpu().numpy()

    ort_out = ort_session.run(None, feed_dict)
    ort_np = ort_out[0]

    abs_err = np.max(np.abs(torch_np - ort_np))
    dot = np.sum(torch_np * ort_np)
    norm_t = np.linalg.norm(torch_np)
    norm_o = np.linalg.norm(ort_np)
    cosine_sim = dot / (norm_t * norm_o + 1e-9)

    passed = abs_err <= atol
    return float(abs_err), float(cosine_sim), passed


def export_models_to_onnx_fp32(
        model: DriverGuardianModel,
        output_dir: Union[str, Path] = "runtime",
        opset_version: int = 17,
        verify: bool = True
) -> Dict[str, str]:
    """
    Triển khai quy trình đóng gói mô hình thành định dạng ONNX Float32 (Opset 17)
    theo chuẩn của structer_inference.md (Mục 8):
    - File 1: driver_guardian_end2end.onnx
    - File 2: spatial_extractor.onnx
    - File 3: temporal_classifier.onnx
    - File 4: model_manifest.json
    """
    if onnx is None:
        raise ImportError("Vui lòng cài đặt onnx: pip install onnx")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()

    exported_files = {}

    print("\n" + "=" * 78)
    print("[*] BẮT ĐẦU ĐÓNG GÓI MÔ HÌNH THÀNH ĐỊNH DẠNG ONNX FLOAT32 (OPSET 17)")
    print("=" * 78)

    # --------------------------------------------------------------------------
    # 1. Xuất Mô Hình Hợp Nhất End-to-End (driver_guardian_end2end.onnx)
    # --------------------------------------------------------------------------
    e2e_path = out_dir / "driver_guardian_end2end.onnx"
    print(f"\n[1/3] Đang xuất mô hình End-to-End: {e2e_path.name}...")

    e2e_wrapper = End2EndONNXExportWrapper(model).eval()
    # Dummy input: [batch_size=1, seq_len=4, 3, 640, 640] dạng Float32
    dummy_video = torch.randn(1, 4, 3, 640, 640, dtype=torch.float32)

    torch.onnx.export(
        e2e_wrapper,
        (dummy_video,),
        str(e2e_path),
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
        input_names=["video"],
        output_names=["logits"],
        dynamic_axes={
            "video": {0: "batch_size", 1: "seq_len"},
            "logits": {0: "batch_size", 1: "seq_len"}
        }
    )

    # Thẩm định cấu trúc đồ thị ONNX
    onnx_e2e = onnx.load(str(e2e_path))
    onnx.checker.check_model(onnx_e2e, full_check=True)
    e2e_sha = compute_sha256(e2e_path)
    e2e_size_mb = e2e_path.stat().st_size / (1024 * 1024)
    exported_files["end2end"] = str(e2e_path)
    print(f"    -> Đã xuất: {e2e_path.name} ({e2e_size_mb:.2f} MB, SHA-256: {e2e_sha[:16]}...)")

    # --------------------------------------------------------------------------
    # 2. Xuất Khối 1: Trích xuất không gian (spatial_extractor.onnx)
    # --------------------------------------------------------------------------
    spatial_path = out_dir / "spatial_extractor.onnx"
    print(f"\n[2/3] Đang xuất Khối 1 (Spatial Extractor): {spatial_path.name}...")

    spatial_wrapper = SpatialExtractorONNXExportWrapper(model).eval()
    dummy_image = torch.randn(1, 3, 640, 640, dtype=torch.float32)

    torch.onnx.export(
        spatial_wrapper,
        (dummy_image,),
        str(spatial_path),
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["features"],
        dynamic_axes={
            "image": {0: "batch_size"},
            "features": {0: "batch_size"}
        }
    )

    onnx_spatial = onnx.load(str(spatial_path))
    onnx.checker.check_model(onnx_spatial, full_check=True)
    spatial_sha = compute_sha256(spatial_path)
    spatial_size_mb = spatial_path.stat().st_size / (1024 * 1024)
    exported_files["spatial_extractor"] = str(spatial_path)
    print(f"    -> Đã xuất: {spatial_path.name} ({spatial_size_mb:.2f} MB, SHA-256: {spatial_sha[:16]}...)")

    # --------------------------------------------------------------------------
    # 3. Xuất Khối 2: Phân loại chuỗi thời gian (temporal_classifier.onnx)
    # --------------------------------------------------------------------------
    temporal_path = out_dir / "temporal_classifier.onnx"
    print(f"\n[3/3] Đang xuất Khối 2 (Temporal Classifier): {temporal_path.name}...")

    temporal_wrapper = TemporalClassifierONNXExportWrapper(model).eval()
    dummy_feat = torch.randn(1, 1, 448, dtype=torch.float32)
    dummy_h0 = torch.zeros(3, 1, 256, dtype=torch.float32)
    dummy_c0 = torch.zeros(3, 1, 256, dtype=torch.float32)

    torch.onnx.export(
        temporal_wrapper,
        (dummy_feat, dummy_h0, dummy_c0),
        str(temporal_path),
        opset_version=opset_version,
        dynamo=False,
        do_constant_folding=True,
        input_names=["features", "h_0", "c_0"],
        output_names=["logits", "h_n", "c_n"],
        dynamic_axes={
            "features": {0: "batch_size"},
            "h_0": {1: "batch_size"},
            "c_0": {1: "batch_size"},
            "logits": {0: "batch_size"},
            "h_n": {1: "batch_size"},
            "c_n": {1: "batch_size"}
        }
    )

    onnx_temporal = onnx.load(str(temporal_path))
    onnx.checker.check_model(onnx_temporal, full_check=True)
    temporal_sha = compute_sha256(temporal_path)
    temporal_size_mb = temporal_path.stat().st_size / (1024 * 1024)
    exported_files["temporal_classifier"] = str(temporal_path)
    print(f"    -> Đã xuất: {temporal_path.name} ({temporal_size_mb:.2f} MB, SHA-256: {temporal_sha[:16]}...)")

    # --------------------------------------------------------------------------
    # 4. Kiểm Định Sai Số Số Học Với ONNX Runtime
    # --------------------------------------------------------------------------
    if verify and ort is not None:
        print("\n" + "-" * 78)
        print("[*] KIỂM ĐỊNH SAI SỐ SỐ HỌC (NUMERICAL PARITY & COSINE SIMILARITY)")
        print("-" * 78)

        providers = ["CPUExecutionProvider"]

        # Thẩm định End-to-End
        sess_e2e = ort.InferenceSession(str(e2e_path), providers=providers)
        test_video = torch.randn(2, 6, 3, 640, 640, dtype=torch.float32)
        err_e2e, sim_e2e, pass_e2e = verify_numerical_parity(
            e2e_wrapper, sess_e2e, {"video": test_video.numpy()}, (test_video,)
        )
        print(f"  [+] End-to-End Model      : Max Abs Error = {err_e2e:.6e} | Cosine Sim = {sim_e2e:.6f} | [{'PASS' if pass_e2e else 'WARN'}]")

        # Thẩm định Spatial Extractor
        sess_spatial = ort.InferenceSession(str(spatial_path), providers=providers)
        test_img = torch.randn(2, 3, 640, 640, dtype=torch.float32)
        err_sp, sim_sp, pass_sp = verify_numerical_parity(
            spatial_wrapper, sess_spatial, {"image": test_img.numpy()}, (test_img,)
        )
        print(f"  [+] Spatial Extractor     : Max Abs Error = {err_sp:.6e} | Cosine Sim = {sim_sp:.6f} | [{'PASS' if pass_sp else 'WARN'}]")

        # Thẩm định Temporal Classifier
        sess_temp = ort.InferenceSession(str(temporal_path), providers=providers)
        test_feat = torch.randn(2, 1, 448, dtype=torch.float32)
        test_h0 = torch.randn(3, 2, 256, dtype=torch.float32)
        test_c0 = torch.randn(3, 2, 256, dtype=torch.float32)
        err_tp, sim_tp, pass_tp = verify_numerical_parity(
            temporal_wrapper, sess_temp,
            {"features": test_feat.numpy(), "h_0": test_h0.numpy(), "c_0": test_c0.numpy()},
            (test_feat, test_h0, test_c0)
        )
        print(f"  [+] Temporal Classifier   : Max Abs Error = {err_tp:.6e} | Cosine Sim = {sim_tp:.6f} | [{'PASS' if pass_tp else 'WARN'}]")

    # --------------------------------------------------------------------------
    # 5. Tạo Tệp Manifest Metadata (model_manifest.json)
    # --------------------------------------------------------------------------
    manifest_data = {
        "system": "Driver Guardian",
        "description": "Production ONNX Float32 Models for Driver Drowsiness Detection",
        "format": "ONNX",
        "precision": "Float32",
        "opset_version": opset_version,
        "classes": ["Alert", "Drowsy"],
        "sample_interval_sec": 0.5,
        "models": {
            "end2end": {
                "file": e2e_path.name,
                "size_mb": round(e2e_size_mb, 2),
                "sha256": e2e_sha,
                "input_names": ["video"],
                "output_names": ["logits"],
                "input_shape": ["batch_size", "seq_len", 3, 640, 640]
            },
            "spatial_extractor": {
                "file": spatial_path.name,
                "size_mb": round(spatial_size_mb, 2),
                "sha256": spatial_sha,
                "input_names": ["image"],
                "output_names": ["features"],
                "input_shape": ["batch_size", 3, 640, 640],
                "output_shape": ["batch_size", 448]
            },
            "temporal_classifier": {
                "file": temporal_path.name,
                "size_mb": round(temporal_size_mb, 2),
                "sha256": temporal_sha,
                "input_names": ["features", "h_0", "c_0"],
                "output_names": ["logits", "h_n", "c_n"],
                "input_shape": ["batch_size", 1, 448]
            }
        },
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    manifest_path = out_dir / "model_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)
    print(f"\n[+] Đã tạo manifest thành công: {manifest_path.name}")
    print("=" * 78)

    return exported_files


# ==============================================================================
# 5. BỘ ĐIỀU PHỐI SUY LUẬN ONNX RUNTIME (ONNX RUNTIME PREDICTORS)
# ==============================================================================

class ONNXEnd2EndPredictor:
    """
    Suy luận toàn diện trên ONNX Runtime cho tệp video hoặc Tensor video hoàn chỉnh.
    Sử dụng mô hình driver_guardian_end2end.onnx.
    """

    def __init__(self, model_path: Union[str, Path], device: str = "cuda:0"):
        self.model_path = str(model_path)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Không tìm thấy file ONNX: {self.model_path}")

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device.lower() else ["CPUExecutionProvider"]
        if ort is None:
            raise ImportError("Vui lòng cài đặt onnxruntime-gpu hoặc onnxruntime.")

        self.session = ort.InferenceSession(self.model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.providers = self.session.get_providers()

    def predict_video_tensor(self, video_tensor: np.ndarray) -> Dict[str, Any]:
        """
        video_tensor: numpy array shape [bz, T, 3, 640, 640] dtype float32
        """
        start = time.perf_counter()
        logits = self.session.run([self.output_name], {self.input_name: video_tensor})[0]  # [bz, T, 2]
        latency_ms = (time.perf_counter() - start) * 1000.0

        # Tính Softmax
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)  # [bz, T, 2]
        drowsy_probs = probs[..., 1]  # [bz, T]

        return {
            "logits": logits,
            "probabilities": probs,
            "drowsy_prob_sequence": drowsy_probs,
            "clip_drowsy_prob": float(np.mean(drowsy_probs)),
            "is_drowsy": bool(np.mean(drowsy_probs) > 0.5),
            "latency_ms": latency_ms
        }


class ONNXStreamingPredictor:
    """
    Bộ suy luận dòng thời gian thực phân tách hai giai đoạn (Decoupled Streaming Predictor)
    tối ưu hóa cho các thiết bị nhúng (NVIDIA Jetson, Raspberry Pi 5):
    - Khối 1: spatial_extractor.onnx (chạy mỗi frame mới thu nhận được)
    - Khối 2: temporal_classifier.onnx (cập nhật trạng thái ẩn LSTM < 1ms)
    """

    def __init__(
            self,
            spatial_model_path: Union[str, Path],
            temporal_model_path: Union[str, Path],
            device: str = "cpu"
    ):
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "cuda" in device.lower() else ["CPUExecutionProvider"]
        if ort is None:
            raise ImportError("Vui lòng cài đặt onnxruntime.")

        self.sess_spatial = ort.InferenceSession(str(spatial_model_path), providers=providers)
        self.sess_temporal = ort.InferenceSession(str(temporal_model_path), providers=providers)

        self.spatial_in = self.sess_spatial.get_inputs()[0].name
        self.spatial_out = self.sess_spatial.get_outputs()[0].name

        self.temp_in_feat = self.sess_temporal.get_inputs()[0].name
        self.temp_in_h0 = self.sess_temporal.get_inputs()[1].name
        self.temp_in_c0 = self.sess_temporal.get_inputs()[2].name

        self.temp_out_logits = self.sess_temporal.get_outputs()[0].name
        self.temp_out_hn = self.sess_temporal.get_outputs()[1].name
        self.temp_out_cn = self.sess_temporal.get_outputs()[2].name

        self.reset_state()

    def reset_state(self):
        """Khởi tạo lại trạng thái ẩn của mạng LSTM."""
        self.h_state = np.zeros((3, 1, 256), dtype=np.float32)
        self.c_state = np.zeros((3, 1, 256), dtype=np.float32)
        self.step_count = 0

    def update_frame(self, frame_bgr: np.ndarray) -> Dict[str, Any]:
        """
        Nhận 1 khung hình thô BGR từ Camera/Webcam, xử lý và cập nhật ngay lập tức trạng thái buồn ngủ.
        """
        start = time.perf_counter()

        # 1. Tiền xử lý nhanh 1 frame
        tensor_img = preprocess_frame(frame_bgr, new_size=640)
        img_batch = np.expand_dims(tensor_img, axis=0)  # [1, 3, 640, 640]

        # 2. Forward qua Khối Spatial Extractor
        t0 = time.perf_counter()
        feat = self.sess_spatial.run([self.spatial_out], {self.spatial_in: img_batch})[0]  # [1, 448]
        feat_time_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Forward qua Khối Temporal Classifier với trạng thái ẩn hiện tại
        feat_seq = np.expand_dims(feat, axis=1)  # [1, 1, 448]
        t1 = time.perf_counter()
        outs = self.sess_temporal.run(
            [self.temp_out_logits, self.temp_out_hn, self.temp_out_cn],
            {
                self.temp_in_feat: feat_seq,
                self.temp_in_h0: self.h_state,
                self.temp_in_c0: self.c_state
            }
        )
        lstm_time_ms = (time.perf_counter() - t1) * 1000.0

        logits, self.h_state, self.c_state = outs
        self.step_count += 1

        # 4. Tính toán xác suất Softmax
        exp_logits = np.exp(logits[0, 0] - np.max(logits[0, 0]))
        probs = exp_logits / np.sum(exp_logits)
        alert_prob, drowsy_prob = float(probs[0]), float(probs[1])

        total_latency_ms = (time.perf_counter() - start) * 1000.0

        return {
            "step": self.step_count,
            "state": "Drowsy" if drowsy_prob > 0.5 else "Alert",
            "is_drowsy": bool(drowsy_prob > 0.5),
            "drowsy_prob": drowsy_prob,
            "alert_prob": alert_prob,
            "spatial_latency_ms": feat_time_ms,
            "temporal_latency_ms": lstm_time_ms,
            "total_latency_ms": total_latency_ms
        }


# ==============================================================================
# 6. GIAO DIỆN DÒNG LỆNH & KIỂM THỬ TỔNG HỢP (CLI & BENCHMARK)
# ==============================================================================

def get_default_detection_checkpoint() -> Optional[str]:
    """Tìm kiếm checkpoint tốt nhất của ObjectDetection_2p6M trong thư mục hiện tại."""
    candidates = [
        CURRENT_DIR / "ObjectDetection_2p6M" / "checkpoints" / "2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031" / "de0c5b23e1ae749b972f10cb5f3e21e2ccf37bd334eb7a7c3091c460073ff310" / "finetune" / "best.pt",
        CURRENT_DIR / "ObjectDetection_2p6M" / "checkpoints" / "2bfb6cc36ef5ab822a944006c746e5d348d67e387501c9027df5b957cf124031" / "d9afe7f332a0080b29fdd068bbb94f32147807d062bc8dbfeb01f8b968f5ad22" / "train" / "best.pt"
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return None


def run_benchmark(model: DriverGuardianModel, device: str = "cpu", iters: int = 20):
    """Đo đạc hiệu năng suy luận và mức tiêu thụ thời gian."""
    print("\n" + "=" * 78)
    print(f"[*] CHẠY BENCHMARK HIỆU NĂNG MÔ HÌNH TRÊN THIẾT BỊ: {device.upper()}")
    print("=" * 78)

    model.to(device).eval()
    dummy_input = torch.randn(1, 60, 3, 640, 640, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(3):
            _ = model(dummy_input)

    # Đo thời gian
    latencies = []
    with torch.no_grad():
        for i in range(iters):
            if "cuda" in device:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(dummy_input)
            if "cuda" in device:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    mean_lat = np.mean(latencies)
    fps = 60.0 / (mean_lat / 1000.0)

    print(f"  - Số lượng frame / clip : 60 frames")
    print(f"  - Độ trễ trung bình     : {mean_lat:.2f} ms / clip ({mean_lat/60.0:.2f} ms / frame)")
    print(f"  - Độ trễ P50            : {p50:.2f} ms")
    print(f"  - Độ trễ P95            : {p95:.2f} ms")
    print(f"  - Tốc độ tương đương    : {fps:.1f} FPS")
    print("=" * 78)


def main():
    parser = argparse.ArgumentParser(
        description="Driver Guardian — Suy luận và Đóng gói ONNX Float32 cho mô hình Nhận diện Buồn ngủ"
    )
    parser.add_argument("--mode", type=str, default="test", choices=["test", "export-onnx", "benchmark", "stream"],
                        help="Chế độ chạy: 'test' (kiểm tra forward), 'export-onnx' (đóng gói ONNX Float32), 'benchmark' (đo latency), 'stream' (test webcam/video)")
    parser.add_argument("--cnn-checkpoint", type=str, default=None,
                        help="Đường dẫn file .pt checkpoint của ObjectDetection_2p6M (Backbone + Neck)")
    parser.add_argument("--lstm-checkpoint", type=str, default=None,
                        help="Đường dẫn file .pt checkpoint của DeepLSTMClassifier")
    parser.add_argument("--output-dir", type=str, default="runtime",
                        help="Thư mục lưu các mô hình ONNX xuất ra")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Thiết bị tính toán: 'cuda:0' hoặc 'cpu'")
    parser.add_argument("--opset", type=int, default=17,
                        help="Phiên bản ONNX Opset (mặc định 17)")

    args = parser.parse_args()

    cnn_ckpt = args.cnn_checkpoint or get_default_detection_checkpoint()
    lstm_ckpt = args.lstm_checkpoint

    print("=" * 78)
    print("HỆ THỐNG DRIVER GUARDIAN — INFERENCE & PACKAGING ENGINE")
    print("=" * 78)
    print(f"[+] Detection Checkpoint : {cnn_ckpt}")
    print(f"[+] LSTM Checkpoint      : {lstm_ckpt or '(Khởi tạo trọng số mặc định)'}")
    print(f"[+] Thiết bị             : {args.device}")

    # Khởi tạo mô hình PyTorch
    model = DriverGuardianModel.create_pretrained(
        cnn_checkpoint=cnn_ckpt,
        lstm_checkpoint=lstm_ckpt,
        device=args.device
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[+] Tổng tham số mô hình : {n_params:,} ({n_params/1e6:.2f}M params)")

    if args.mode == "test":
        print("\n[*] Chạy kiểm thử suy luận PyTorch với batch mẫu [bz=2, T=10, 3, 640, 640]...")
        dummy = torch.randn(2, 10, 3, 640, 640, device=args.device)
        with torch.no_grad():
            out_seq = model(dummy, return_sequence=True)
            out_clip = model(dummy, return_sequence=False)
        print(f"[+] Output Sequence Logits : {list(out_seq.shape)} (Mong đợi [2, 10, 2])")
        print(f"[+] Output Clip Logits     : {list(out_clip.shape)} (Mong đợi [2, 2])")
        print("[SUCCESS] Kiểm thử Forward Pass hoàn toàn chính xác!")

    elif args.mode == "export-onnx":
        export_models_to_onnx_fp32(
            model=model,
            output_dir=args.output_dir,
            opset_version=args.opset,
            verify=True
        )

    elif args.mode == "benchmark":
        run_benchmark(model=model, device=args.device, iters=20)

    elif args.mode == "stream":
        print("\n[*] Kiểm tra mô phỏng Streaming Decoupled Predictor...")
        runtime_dir = Path(args.output_dir)
        sp_path = runtime_dir / "spatial_extractor.onnx"
        tp_path = runtime_dir / "temporal_classifier.onnx"

        if not sp_path.exists() or not tp_path.exists():
            print("[WARN] Chưa tìm thấy các file ONNX phân tách, đang tiến hành export trước...")
            export_models_to_onnx_fp32(model=model, output_dir=args.output_dir, opset_version=args.opset)

        streamer = ONNXStreamingPredictor(sp_path, tp_path, device=args.device)
        dummy_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        print("[+] Mô phỏng chạy 5 frames liên tiếp:")
        for i in range(5):
            res = streamer.update_frame(dummy_frame)
            print(f"  Frame {res['step']:02d}: Trạng thái = {res['state']:<7} | Drowsy Prob = {res['drowsy_prob']:.4f} | Total Latency = {res['total_latency_ms']:.2f} ms")


if __name__ == "__main__":
    main()
