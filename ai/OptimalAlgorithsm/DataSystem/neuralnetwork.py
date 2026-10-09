import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

# Đảm bảo đường dẫn gốc (thư mục cha chứa 'ai') và các gói con có mặt trong sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
for _p in _CURRENT_DIR.parents:
    if (_p / "ai").is_dir():
        if str(_p) not in sys.path:
            sys.path.insert(0, str(_p))
        _ai_dir = _p / "ai"
        for _sub in ("ObjectDetection_2p6M", "LSTM"):
            _sub_dir = str(_ai_dir / _sub)
            if _sub_dir not in sys.path:
                sys.path.insert(0, _sub_dir)
        break

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ai.LSTM.src import ConvGRUClassifier
from ai.ObjectDetection_2p6M.src.model import NMSFreeDetector


def _resolve_checkpoint(user_path: Optional[Union[str, Path]], relative_fallback: str) -> Path:
    """Tự động phân giải đường dẫn checkpoint hợp lệ trong dự án."""
    if user_path is not None:
        p = Path(user_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Checkpoint không tồn tại: {user_path}")

    # Tìm kiếm các vị trí mặc định trong cây thư mục project
    for parent in _CURRENT_DIR.parents:
        cand1 = parent / "ai" / "checkpoints" / relative_fallback
        if cand1.exists():
            return cand1
        cand2 = parent / "checkpoints" / relative_fallback
        if cand2.exists():
            return cand2
        cand3 = parent / relative_fallback
        if cand3.exists():
            return cand3

    cwd = Path.cwd()
    cand_cwd1 = cwd / "ai" / "checkpoints" / relative_fallback
    if cand_cwd1.exists():
        return cand_cwd1
    cand_cwd2 = cwd / "checkpoints" / relative_fallback
    if cand_cwd2.exists():
        return cand_cwd2

    raise FileNotFoundError(f"Không tìm thấy checkpoint mặc định: {relative_fallback}")


class NeuralNetwork(nn.Module):
    def __init__(
            self,
            cnn_model: Optional[NMSFreeDetector] = None,
            conv_gru_model: Optional[ConvGRUClassifier] = None,
            chunk_size: int = 32,
            cnn_path: Optional[Union[str, Path]] = None,
            conv_gru_path: Optional[Union[str, Path]] = None,
            device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Tự động nạp checkpoints tối ưu nếu chưa truyền model
        if cnn_model is None:
            resolved_cnn = _resolve_checkpoint(cnn_path, "model_cnn/best.pt")
            cnn_model = NMSFreeDetector.from_checkpoint(resolved_cnn, map_location=self.device, eval_mode=True)

        if conv_gru_model is None:
            resolved_gru = _resolve_checkpoint(conv_gru_path, "model_convgru/best.pt")
            conv_gru_model = ConvGRUClassifier.from_checkpoint(resolved_gru, map_location=str(self.device))

        self.cnn_model: NMSFreeDetector = cnn_model
        self.conv_gru: ConvGRUClassifier = conv_gru_model.eval()
        self.cnn_backbone = self.cnn_model.backbone.eval()
        self.cnn_neck = self.cnn_model.neck.eval()
        self.chunk_size: int = max(1, int(chunk_size))
        self.to(self.device)

    @classmethod
    def from_checkpoint(
            cls,
            cnn_path: Optional[Union[str, Path]] = None,
            conv_gru_path: Optional[Union[str, Path]] = None,
            device: Optional[Union[str, torch.device]] = None,
            chunk_size: int = 32,
    ):
        """Khởi tạo NeuralNetwork trực tiếp từ các file checkpoint .pt"""
        return cls(
            cnn_path=cnn_path,
            conv_gru_path=conv_gru_path,
            device=device,
            chunk_size=chunk_size,
        )

    def forward(self, x: Union[np.ndarray, torch.Tensor]) -> float:
        """
        Dự đoán mức độ buồn ngủ từ chuỗi khung hình video.

        Args:
            x: Chuỗi frame dạng np.ndarray hoặc torch.Tensor với shape [T, 3, H, W].
               Dtype có thể là uint8 [0, 255] hoặc float [0.0, 1.0].
               H và W có thể là 640 hoặc 224 (sẽ tự động điều chỉnh về 640x640).

        Returns:
            float: Điểm xác suất buồn ngủ trong khoảng [0.0, 1.0].
        """
        if self.cnn_backbone is None or self.cnn_neck is None or self.conv_gru is None:
            raise RuntimeError("Model checkpoints are not loaded.")

        device = next(self.cnn_backbone.parameters()).device

        # Vấn đề 2: Chuyển đổi an toàn từ numpy ndarray sang torch.Tensor
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        elif not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(x).__name__}")

        if x.ndim != 4:
            raise ValueError(f"Expected input with 4 dimensions [T, C, H, W], got shape {tuple(x.shape)}")

        T, C, H, W = x.shape
        if T == 0:
            raise ValueError("Empty frame sequence (T = 0)")
        if H == 0 or W == 0:
            raise ValueError("Frame dimensions must be nonzero")
        if C != 3:
            raise ValueError(f"Expected 3 color channels (RGB), got C = {C}")

        if x.dtype != torch.uint8 and (
                not x.is_floating_point() or not torch.isfinite(x).all()
                or x.min() < 0 or x.max() > 1):
            raise ValueError("Frames must be uint8 [0,255] or finite float [0,1]")

        # Vấn đề 3: Tự động điều chỉnh kích thước về 640x640 qua hàm letterbox nếu H != 640 hoặc W != 640
        if H != 640 or W != 640:
            x_cpu = x.detach().cpu()
            x_np = (x_cpu if x_cpu.dtype == torch.uint8 else x_cpu.float()).numpy()

            # Áp dụng hàm letterbox chuẩn hóa từng khung hình về [3, 640, 640]
            x_lb = np.stack([letterbox(np.moveaxis(frame, 0, -1), new_size=640).transpose(2, 0, 1) for frame in x_np])
            x = torch.from_numpy(x_lb)
            H, W = 640, 640

        p3_list, p4_list, p5_list = [], [], []

        with torch.inference_mode():
            for i in range(0, T, self.chunk_size):
                chunk = x[i : i + self.chunk_size].to(device, non_blocking=True)

                # Chuẩn hóa về [0.0, 1.0] float32
                if chunk.dtype == torch.uint8:
                    chunk = chunk.float() / 255.0
                elif chunk.dtype != torch.float32:
                    chunk = chunk.float()

                out3, out4, out5 = self.cnn_backbone(chunk)
                out3, out4, out5 = self.cnn_neck(out3, out4, out5)

                p3_list.append(out3.float())
                p4_list.append(out4.float())
                p5_list.append(out5.float())

            p3 = torch.cat(p3_list, dim=0).unsqueeze(0)
            p4 = torch.cat(p4_list, dim=0).unsqueeze(0)
            p5 = torch.cat(p5_list, dim=0).unsqueeze(0)

            seq_lens = torch.tensor([T], dtype=torch.long, device=device)
            logits = self.conv_gru((p3, p4, p5), seq_lens=seq_lens)

            if logits.shape not in ((1, 1), (1, 2)):
                raise ValueError(f"Expected binary clip logits [1,1] or [1,2], got {tuple(logits.shape)}")
            # Tính điểm xác suất
            if logits.shape[-1] == 1:
                score = torch.sigmoid(logits).item()
            else:
                probs = F.softmax(logits, dim=-1)
                score = probs[0, 1].item()

        return float(score)


def letterbox(
        image: np.ndarray,
        new_size: int = 640,
        color: Tuple[int, int, int] = (114, 114, 114)
) -> np.ndarray:
    """
    Căn chỉnh kích thước ảnh về new_size x new_size giữ nguyên tỷ lệ khung hình
    và chèn viền màu (mặc định 114, 114, 114).
    Hỗ trợ linh hoạt cả định dạng HWC [H, W, 3] và CHW [3, H, W].
    """
    is_chw = (image.ndim == 3 and image.shape[0] == 3 and image.shape[-1] != 3)
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[*] Khởi tạo NeuralNetwork trên device: {device}")
    model = NeuralNetwork(device=device)
    print("[*] Nạp model thành công!")

    # Test với dummy array 224x224 mô phỏng dữ liệu từ Engine
    test_dummy = np.zeros((10, 3, 224, 224), dtype=np.uint8)

    score = model(test_dummy)
    print(f"[*] Điểm dự đoán với mảng 224x224: {score:.4f}")
