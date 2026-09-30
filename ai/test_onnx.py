#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File: test_onnx.py
Hệ Thống Driver Guardian — Script Chạy Thử Nghiệm Mô Hình ONNX End-to-End
Mô hình: runtime/driver_guardian_end2end.onnx

Mục đích:
1. Nạp và khởi tạo mô hình ONNX Runtime trên GPU CUDA (hoặc CPU fallback).
2. Kiểm tra thông tin metadata, cấu trúc đầu vào [bz, T, 3, 640, 640] và đầu ra [bz, T, 2].
3. Chạy suy luận thử nghiệm trên dữ liệu giả lập (Synthetic Dummy Tensor) hoặc tệp video thực tế (.mp4, .avi).
4. Phân tích kết quả đầu ra:
   - Tính toán phân phối xác suất Softmax cho từng bước thời gian t (Alert vs Drowsy).
   - Đánh giá trạng thái buồn ngủ của người lái theo cấp độ từng khung hình và toàn bộ clip.
5. Đo đạc hiệu năng suy luận (Độ trễ P50, P95, Mean latency và FPS).
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

# Đảm bảo console Windows in tiếng Việt UTF-8 không bị lỗi charmap
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Nạp thư viện CUDA DLL từ PyTorch nếu chạy trên môi trường Windows
try:
    import torch
    torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
    if os.path.exists(torch_lib_path) and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(torch_lib_path)
except Exception:
    pass

try:
    import onnxruntime as ort
except ImportError as e:
    raise ImportError("Vui lòng cài đặt onnxruntime-gpu hoặc onnxruntime: pip install onnxruntime-gpu") from e

import numpy as np
import cv2


# ==============================================================================
# 1. TIỀN XỬ LÝ ẢNH & VIDEO (LETTERBOX PREPROCESSING)
# ==============================================================================

def letterbox(image: np.ndarray, new_size: int = 640, color=(114, 114, 114)) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỷ lệ khung hình với đệm đồng màu 640x640.
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
    Chuyển đổi BGR -> RGB, Letterbox và chuẩn hóa dải màu [0.0, 1.0].
    Returns: Tensor dạng NumPy shape [3, 640, 640], float32.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxed = letterbox(rgb, new_size=new_size)
    tensor = np.ascontiguousarray(boxed.transpose(2, 0, 1), dtype=np.float32) / 255.0
    return tensor


def load_video_tensor(
        video_path: str,
        seq_len: int = 60,
        sample_interval: float = 0.5,
        img_size: int = 640
) -> Tuple[np.ndarray, List[float]]:
    """
    Đọc tệp video thực tế, lấy mẫu khung hình theo chu kỳ thời gian (sample_interval = 0.5s)
    và đóng gói thành tensor 5D: [1, T, 3, 640, 640].
    """
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Không thể mở file video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0
    frame_interval = max(1, int(round(fps * sample_interval)))

    frames = []
    timestamps = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frames.append(preprocess_frame(frame, new_size=img_size))
            timestamps.append(frame_idx / fps)
            if len(frames) >= seq_len:
                break
        frame_idx += 1

    cap.release()

    if not frames:
        raise ValueError(f"Không thể đọc được khung hình nào từ video: {video_path}")

    # Nếu video ngắn hơn seq_len, lặp lại khung hình cuối
    while len(frames) < seq_len:
        frames.append(frames[-1])
        timestamps.append(timestamps[-1] + sample_interval)

    # Tensor: [1, seq_len, 3, H, W]
    video_tensor = np.expand_dims(np.stack(frames, axis=0), axis=0)
    return video_tensor, timestamps


# ==============================================================================
# 2. KHỞI TẠO MÔ HÌNH VÀ SUY LUẬN ONNX RUNTIME
# ==============================================================================

class End2EndONNXRunner:
    """Bộ điều phối chạy mô hình driver_guardian_end2end.onnx trên ONNX Runtime."""

    def __init__(self, model_path: str, device: str = "cuda"):
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file mô hình ONNX tại: {self.model_path}")

        # Cấu hình Execution Providers
        avail_providers = ort.get_available_providers()
        if "cuda" in device.lower() and "CUDAExecutionProvider" in avail_providers:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        # Thiết lập Session Options tối ưu
        sess_opts = ort.SessionOptions()
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_opts.log_severity_level = 3  # Giảm bớt cảnh báo không cần thiết

        self.session = ort.InferenceSession(str(self.model_path), sess_options=sess_opts, providers=providers)
        self.active_providers = self.session.get_providers()

        # Đọc thông tin I/O metadata
        self.inp_meta = self.session.get_inputs()[0]
        self.out_meta = self.session.get_outputs()[0]
        self.input_name = self.inp_meta.name
        self.output_name = self.out_meta.name

    def print_metadata(self):
        """In thông tin chi tiết về mô hình ONNX."""
        file_size_mb = self.model_path.stat().st_size / (1024 * 1024)
        print("=" * 78)
        print("THÔNG TIN MÔ HÌNH ONNX RUNTIME (DRIVER GUARDIAN END-TO-END)")
        print("=" * 78)
        print(f"  - Đường dẫn mô hình : {self.model_path}")
        print(f"  - Dung lượng file   : {file_size_mb:.2f} MB")
        print(f"  - Active Providers  : {self.active_providers}")
        print(f"  - Đầu vào (Input)   : name='{self.input_name}', shape={self.inp_meta.shape}, dtype={self.inp_meta.type}")
        print(f"  - Đầu ra (Output)   : name='{self.output_name}', shape={self.out_meta.shape}, dtype={self.out_meta.type}")
        print("=" * 78)

    def forward(self, video_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        video_tensor: numpy array shape [bz, T, 3, 640, 640] dạng float32
        Returns:
            logits: [bz, T, 2]
            latency_ms: độ trễ suy luận tính bằng mili-giây
        """
        start = time.perf_counter()
        logits = self.session.run([self.output_name], {self.input_name: video_tensor})[0]
        latency_ms = (time.perf_counter() - start) * 1000.0
        return logits, latency_ms

    def benchmark(self, batch_size: int = 1, seq_len: int = 60, iters: int = 20):
        """Đo đạc độ trễ và thông lượng FPS."""
        print("\n" + "=" * 78)
        print(f"[*] CHẠY BENCHMARK ĐỘ TRỄ SUY LUẬN (BATCH={batch_size}, SEQ_LEN={seq_len}, ITERS={iters})")
        print("=" * 78)

        dummy = np.random.randn(batch_size, seq_len, 3, 640, 640).astype(np.float32)

        # 1. Warmup
        print("[+] Đang chạy khởi động (Warmup)...")
        for _ in range(3):
            _ = self.session.run([self.output_name], {self.input_name: dummy})

        # 2. Đo thời gian lặp
        latencies = []
        for _ in range(iters):
            t0 = time.perf_counter()
            _ = self.session.run([self.output_name], {self.input_name: dummy})
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0)

        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        mean_lat = float(np.mean(latencies))
        fps = (batch_size * seq_len) / (mean_lat / 1000.0)

        print(f"  - Độ trễ trung bình : {mean_lat:.2f} ms / clip ({mean_lat / seq_len:.2f} ms / frame)")
        print(f"  - Phân vị P50       : {p50:.2f} ms")
        print(f"  - Phân vị P95       : {p95:.2f} ms")
        print(f"  - Tốc độ tương đương: {fps:.1f} FPS")
        print("=" * 78)


# ==============================================================================
# 3. HÀM MAIN VÀ XỬ LÝ KẾT QUẢ DỰ ĐOÁN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chạy thử nghiệm mô hình ONNX runtime/driver_guardian_end2end.onnx trên ONNX Runtime"
    )
    parser.add_argument("--model", type=str, default=r"runtime\driver_guardian_end2end.onnx",
                        help="Đường dẫn tới file ONNX model")
    parser.add_argument("--video", type=str, default=None,
                        help="Đường dẫn tới tệp video thực tế cần kiểm thử (nếu không cung cấp, dùng tensor giả lập)")
    parser.add_argument("--seq-len", type=int, default=10,
                        help="Số lượng khung hình trong chuỗi thời gian (mặc định 10 cho test nhanh, 60 cho video thực tế)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (mặc định 1)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"],
                        help="Thiết bị tính toán: 'cuda' hoặc 'cpu'")
    parser.add_argument("--benchmark", action="store_true",
                        help="Kích hoạt đo đạc hiệu năng benchmark độ trễ")

    args = parser.parse_args()

    # Tìm kiếm đường dẫn file model nếu đường dẫn mặc định không thấy
    model_path = Path(args.model)
    if not model_path.exists():
        fallback_path = Path("runtime_onnx") / "driver_guardian_end2end.onnx"
        if fallback_path.exists():
            print(f"[WARN] Không tìm thấy '{model_path}', chuyển sang sử dụng: '{fallback_path}'")
            model_path = fallback_path
        else:
            raise FileNotFoundError(f"Không tìm thấy file ONNX: {model_path}")

    # 1. Khởi tạo ONNX Runner
    runner = End2EndONNXRunner(str(model_path), device=args.device)
    runner.print_metadata()

    # 2. Chuẩn bị dữ liệu đầu vào
    timestamps = None
    if args.video:
        print(f"\n[*] Đang nạp và tiền xử lý video thực tế: {args.video}...")
        video_tensor, timestamps = load_video_tensor(
            video_path=args.video,
            seq_len=args.seq_len,
            sample_interval=0.5,
            img_size=640
        )
        print(f"[+] Đã trích xuất chuỗi {video_tensor.shape[1]} khung hình (khoảng {timestamps[-1]:.1f}s quan sát).")
    else:
        print(f"\n[*] Tạo dữ liệu giả lập (Synthetic Dummy Tensor): shape=[{args.batch_size}, {args.seq_len}, 3, 640, 640]...")
        video_tensor = np.random.randn(args.batch_size, args.seq_len, 3, 640, 640).astype(np.float32)
        timestamps = [i * 0.5 for i in range(args.seq_len)]

    # 3. Chạy suy luận trên ONNX Runtime
    print("\n[*] Đang chạy suy luận qua mô hình ONNX...")
    logits, latency_ms = runner.forward(video_tensor)

    # 4. Phân tích kết quả dự đoán với hàm Softmax
    # logits shape: [bz, T, 2]
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    bz, t, num_classes = probs.shape
    print("\n" + "=" * 78)
    print(f"KẾT QUẢ SUY LUẬN THEO TỪNG BƯỚC THỜI GIAN (ĐỘ TRỄ TOÀN CHUỖI: {latency_ms:.2f} ms)")
    print("=" * 78)
    print(f"{'Frame':<8} | {'Thời Điểm':<12} | {'P(Tỉnh Táo)':<14} | {'P(Buồn Ngủ)':<14} | {'Trạng Thái':<12} | {'Mức Độ Tin Cậy'}")
    print("-" * 78)

    sample_probs = probs[0]  # Phân tích mẫu đầu tiên trong batch
    drowsy_sequence = []

    for idx in range(t):
        p_alert = sample_probs[idx, 0]
        p_drowsy = sample_probs[idx, 1]
        t_sec = timestamps[idx] if timestamps else idx * 0.5
        is_drowsy = p_drowsy > 0.5
        status_str = "BUỒN NGỦ" if is_drowsy else "Tỉnh táo"
        confidence = max(p_alert, p_drowsy)
        drowsy_sequence.append(p_drowsy)

        print(f"{idx + 1:02d}       | {t_sec:6.1f}s       | {p_alert * 100:6.2f}%       | {p_drowsy * 100:6.2f}%       | {status_str:<12} | {confidence * 100:5.1f}%")

    print("-" * 78)

    # Đánh giá mức độ buồn ngủ tổng thể cho toàn clip
    mean_drowsy = float(np.mean(drowsy_sequence))
    final_drowsy = float(drowsy_sequence[-1])
    overall_status = "CẢNH BÁO BUỒN NGỦ (DROWSY)" if mean_drowsy > 0.5 else "BÌNH THƯỜNG / TỈNH TÁO (ALERT)"

    print(f"  - Xác suất buồn ngủ trung bình : {mean_drowsy * 100:.2f}%")
    print(f"  - Xác suất buồn ngủ frame cuối  : {final_drowsy * 100:.2f}%")
    print(f"  => KẾT LUẬN TOÀN BỘ VIDEO CLIP  : [ {overall_status} ]")
    print("=" * 78)

    # 5. Tùy chọn chạy benchmark
    if args.benchmark:
        runner.benchmark(batch_size=args.batch_size, seq_len=args.seq_len, iters=20)


if __name__ == "__main__":
    main()
