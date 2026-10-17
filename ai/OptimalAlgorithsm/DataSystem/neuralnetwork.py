from pathlib import Path
import sys
from time import perf_counter
from typing import Tuple

import cv2
import numpy as np
import torch
from torch import nn

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from ai.LSTM.src.checkpoint import load_model
from ai.LSTM.src.model import DrowsyClassifier
from ai.OptimalAlgorithsm.DataSystem.config import (
    BENCHMARK_RUNS, BENCHMARK_WARMUP, DEFAULT_CONFIG, LETTERBOX_COLOR, MODEL_IMAGE_SIZE,
)


class NeuralNetwork(nn.Module):
    def __init__(self, checkpoint_path=None, device=None, chunk_size=None, *,
                 config=DEFAULT_CONFIG, model: DrowsyClassifier | None = None):
        super().__init__()
        device = config.device if device is None else device
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.image_size = config.neural_size
        if (not np.isfinite([config.neural_fps, config.neural_sec]).all()
                or min(config.neural_fps, config.neural_sec) <= 0
                or config.neural_count < 1
                or abs(config.neural_count / config.neural_fps - config.neural_sec) > 1e-6):
            raise ValueError("neural_sec must contain a positive integer number of neural frames")
        self.frame_count = config.neural_count
        self.fps = config.neural_fps
        chunk_size = config.chunk_size if chunk_size is None else chunk_size
        if type(chunk_size) is not int or chunk_size < 0:
            raise ValueError("chunk_size must be a nonnegative integer")
        if type(self.image_size) is not int or self.image_size < 1:
            raise ValueError("neural_size must be a positive integer")
        if model is None:
            path = Path(checkpoint_path or config.checkpoint_path).expanduser()
            model, state = load_model(path, self.device)
            for key, expected in (("image_size", self.image_size),
                                  ("target_fps", config.neural_fps),
                                  ("window_sec", config.neural_sec)):
                if state["config"].get(key) != expected:
                    raise ValueError(f"Checkpoint {key}={state['config'].get(key)!r}, expected {expected}")
        self.model = model.to(self.device).eval()
        self.model.chunk_size = chunk_size
        self.eval()

    @classmethod
    def from_checkpoint(cls, checkpoint_path=None, **kwargs):
        return cls(checkpoint_path=checkpoint_path, **kwargs)

    @torch.inference_mode()
    def forward(self, x) -> float:
        """RGB [50,3,H,W] đã lấy mẫu 5 FPS/10s -> P(drowsy).

        Engine lấy mỗi 6 frame từ read_video 30 FPS trước khi gọi mạng.
        DrowsyClassifier tự chia uint8 cho 255; giữ nguyên thứ tự kênh RGB.
        """
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if not isinstance(x, torch.Tensor):
            raise TypeError("Frames must be a numpy array or torch tensor")
        if x.ndim != 4 or x.shape[1] != 3 or any(size == 0 for size in x.shape):
            raise ValueError("Frames must have nonempty shape [T,3,H,W]")
        if x.shape[0] != self.frame_count:
            raise ValueError(f"Expected {self.frame_count} frames sampled at {self.fps:g} FPS, got {x.shape[0]}")
        if x.dtype != torch.uint8 and (
            not x.is_floating_point() or not torch.isfinite(x).all()
            or x.min() < 0 or x.max() > 1
        ):
            raise ValueError("Frames must be uint8 [0,255] or finite float [0,1]")
        if x.shape[-2:] != (self.image_size, self.image_size):
            frames = x.detach().cpu()
            frames = (frames if frames.dtype == torch.uint8 else frames.float()).numpy()
            x = torch.from_numpy(np.stack([letterbox(frame.transpose(1, 2, 0), self.image_size).transpose(2, 0, 1)
                                           for frame in frames]))
        if x.dtype != torch.uint8:
            x = x.float()
        x = x.unsqueeze(0).to(self.device)
        self.model.eval()
        logits = self.model(x, seq_lens=torch.tensor([x.shape[1]], device=self.device))
        if logits.shape != (1, 2) or not torch.isfinite(logits).all():
            raise ValueError("Expected finite binary clip logits [1,2]")
        return logits.float().softmax(dim=-1)[0, 1].item()


def letterbox(
    image: np.ndarray,
    new_size: int = MODEL_IMAGE_SIZE,
    color: Tuple[int, int, int] = LETTERBOX_COLOR,
) -> np.ndarray:
    """
    Căn chỉnh kích thước ảnh về new_size x new_size giữ nguyên tỷ lệ khung hình
    và chèn viền màu (mặc định 114, 114, 114).
    Hỗ trợ linh hoạt cả định dạng HWC [H, W, 3] và CHW [3, H, W].
    """
    is_chw = image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3
    if is_chw:
        image = np.transpose(image, (1, 2, 0))

    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    if np.issubdtype(image.dtype, np.floating):
        color = np.asarray(color) / 255.0
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    pad_left = (new_size - new_w) // 2
    pad_top = (new_size - new_h) // 2
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized

    if is_chw:
        return np.transpose(canvas, (2, 0, 1))
    return canvas


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark tốc độ suy luận NeuralNetwork")
    parser.add_argument("--device", default=DEFAULT_CONFIG.device)
    parser.add_argument("--frames", type=int, default=DEFAULT_CONFIG.neural_count)
    parser.add_argument("--size", type=int, default=MODEL_IMAGE_SIZE)
    parser.add_argument("--warmup", type=int, default=BENCHMARK_WARMUP)
    parser.add_argument("--runs", type=int, default=BENCHMARK_RUNS)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CONFIG.chunk_size)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CONFIG.checkpoint_path))
    args = parser.parse_args()
    if min(args.frames, args.size, args.runs) < 1 or args.chunk_size < 0 or args.warmup < 0:
        parser.error("frames, size, runs phải > 0; chunk-size, warmup phải >= 0")

    if args.frames != DEFAULT_CONFIG.neural_count:
        parser.error(f"frames phải bằng {DEFAULT_CONFIG.neural_count} theo cấu hình train")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[*] Khởi tạo NeuralNetwork trên device: {device}")
    model = NeuralNetwork(
        device=device, chunk_size=args.chunk_size,
        checkpoint_path=args.checkpoint,
    ).eval()
    print("[*] Nạp model thành công!")

    test_dummy = np.zeros((args.frames, 3, args.size, args.size), dtype=np.uint8)
    print(f"[*] Input giả lập: {test_dummy.shape}, uint8; chunk_size={args.chunk_size}")
    print(f"[*] Warm-up {args.warmup} lượt, đo {args.runs} lượt...")
    for _ in range(args.warmup):
        model(test_dummy)

    elapsed = []
    for _ in range(args.runs):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        start = perf_counter()
        score = model(test_dummy)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed.append(perf_counter() - start)

    latency_ms = np.asarray(elapsed) * 1000
    print(f"[*] Điểm dự đoán: {score:.4f}")
    print("[*] Thời gian gồm tiền xử lý, truyền dữ liệu và suy luận; không gồm nạp model.")
    print(f"[*] Độ trễ/chuỗi: mean={latency_ms.mean():.2f} ms, "
          f"P50={np.percentile(latency_ms, 50):.2f} ms, "
          f"P95={np.percentile(latency_ms, 95):.2f} ms")
    print(f"[*] Tốc độ: {args.runs / sum(elapsed):.2f} chuỗi/s, "
          f"{args.frames * args.runs / sum(elapsed):.2f} frame/s")
