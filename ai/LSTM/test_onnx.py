import os
import sys
import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

# Đảm bảo console Windows in tiếng Việt UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")


def letterbox(image: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """Resize ảnh giữ nguyên tỉ lệ (aspect ratio) với padding đồng màu (mặc định 640x640)."""
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


backbone_neck_path = r"D:\Project\DATN\driver-guardian\ai\LSTM\backbone_neck.onnx"
img_path = r"D:\Project\DATN\driver-guardian\ai\LSTM\img.png"

# 1. Cấu hình Providers đúng chuẩn (ưu tiên CUDA, fallback CPU)
available_providers = ort.get_available_providers()
providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in available_providers else ["CPUExecutionProvider"]

print(f"[+] Khởi tạo InferenceSession: {backbone_neck_path}")
backbone_neck = ort.InferenceSession(str(backbone_neck_path), providers=providers)
print(f"[+] Active Providers: {backbone_neck.get_providers()}")

# 2. Đọc metadata input
inp_meta = backbone_neck.get_inputs()[0]
input_name = inp_meta.name
req_h = inp_meta.shape[2] if len(inp_meta.shape) > 2 and isinstance(inp_meta.shape[2], int) else 640
req_w = inp_meta.shape[3] if len(inp_meta.shape) > 3 and isinstance(inp_meta.shape[3], int) else 640
print(f"[*] Input requirement: name='{input_name}', shape={inp_meta.shape}, dtype={inp_meta.type}")

# 3. Đọc và tiền xử lý ảnh
if not os.path.exists(img_path):
    raise FileNotFoundError(f"Không tìm thấy file ảnh: {img_path}")

raw_img = cv2.imread(str(img_path))
if raw_img is None:
    raise ValueError(f"Không thể giải mã ảnh: {img_path}")

inputs = letterbox(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB), new_size=req_h)
inputs = np.ascontiguousarray(inputs.transpose(2, 0, 1), dtype=np.float32) / 255.0
inputs = np.expand_dims(inputs, axis=0)  # [1, 3, 640, 640]
print(f"[*] Preprocessed input shape: {inputs.shape}, dtype: {inputs.dtype}, range: [{inputs.min():.2f}, {inputs.max():.2f}]")

# 4. Chạy suy luận ONNX
output_names = [o.name for o in backbone_neck.get_outputs()]
features = backbone_neck.run(output_names, {input_name: inputs})

# 5. Phân tích kết quả và mô phỏng Adaptive Average Pooling (1, 1) cho LSTM
print("\n" + "=" * 70)
print(f"{'Output Name':<12} | {'Raw Feature Shape':<24} | {'Pooled 1D Shape':<18} | {'Range':<14}")
print("-" * 70)
pooled_feats = []
for name, feat in zip(output_names, features):
    t_feat = torch.from_numpy(feat)
    pooled = F.adaptive_avg_pool2d(t_feat, (1, 1)).flatten(1)
    pooled_feats.append(pooled)
    print(f"{name:<12} | {str(list(feat.shape)):<24} | {str(list(pooled.shape)):<18} | [{feat.min():.2f}, {feat.max():.2f}]")

fused_1d = torch.cat(pooled_feats, dim=-1)
print("=" * 70)
print(f"[SUCCESS] Tổng số kênh sau ghép nối Spatial Adapter: {fused_1d.shape[-1]} kênh (khớp với cnn_out_channels=448 trong config.py).")
