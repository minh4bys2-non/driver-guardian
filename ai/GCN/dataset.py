# File: dataset.py
import csv
import logging
import os
import random
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Cấu hình nạp DLL PyTorch CUDA và môi trường cho ONNX Runtime trên Windows
torch_lib_path = os.path.join(os.path.dirname(torch.__file__), "lib")
if os.path.exists(torch_lib_path):
    try:
        os.add_dll_directory(torch_lib_path)
    except Exception:
        pass
    os.environ["PATH"] = torch_lib_path + os.pathsep + os.environ.get("PATH", "")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# =====================================================================
# 1. BẢNG CHỈ SỐ ÁNH XẠ 68 ĐIỂM MỐC (MediaPipe 478 -> Dlib 68)
# Khớp chính xác với cấu trúc đồ thị FacialGraphBuilder trong block.py
# =====================================================================
LANDMARK_68_INDICES = [
    # Viền hàm (0-16): 17 điểm
    162, 234, 93, 58, 172, 136, 149, 148, 152, 377, 378, 365, 397, 288, 323, 454, 389,
    # Lông mày trái (17-21) & phải (22-26): 10 điểm
    71, 63, 105, 66, 107, 336, 296, 334, 293, 301,
    # Sống mũi & chân mũi (27-35): 9 điểm
    168, 197, 5, 4, 75, 97, 2, 326, 305,
    # Mắt trái (36-41) & Mắt phải (42-47): 12 điểm
    33, 160, 158, 133, 153, 144, 362, 385, 387, 263, 373, 380,
    # Môi ngoài (48-59) & Môi trong (60-67): 20 điểm
    61, 39, 37, 0, 267, 269, 291, 405, 314, 17, 84, 181,
    78, 82, 13, 312, 308, 317, 14, 87,
]


def letterbox(image: np.ndarray, new_size: int = 480, color: tuple = (114, 114, 114)) -> np.ndarray:
    """Resize ảnh giữ nguyên tỷ lệ khung hình với đệm màu viền."""
    h, w = image.shape[:2]
    scale = min(new_size / h, new_size / w)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_size, new_size, 3), color, dtype=image.dtype)
    x, y = (new_size - nw) // 2, (new_size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


# =====================================================================
# 2. BỘ TRÍCH XUẤT 68 ĐIỂM MỐC KHUÔN MẶT BẰNG INSIGHTFACE (CUDA)
# =====================================================================
class FaceLandmarkExtractor:
    """Trích xuất 68 điểm mốc chuẩn iBUG/Dlib từ các khung hình ảnh bằng InsightFace (mặc định CUDA)."""

    def __init__(
            self,
            model_path: Optional[str | Path] = None,
            time_interval: float = 0.0,
            device: str = "cuda",
            det_size: Tuple[int, int] = (640, 640),
            model_name: str = "buffalo_l",
    ):
        self.model_path = Path(model_path) if model_path is not None else None
        self.time_interval = float(time_interval)
        self.device = device.lower().strip()
        self.det_size = det_size
        self.model_name = model_name
        self.app = None

    def _init_landmarker(self):
        """Khởi tạo trễ (lazy init) để tương thích đa tiến trình của DataLoader."""
        import onnxruntime as ort
        from insightface.app import FaceAnalysis

        available_providers = ort.get_available_providers()
        use_cuda = self.device.startswith("cuda") and ("CUDAExecutionProvider" in available_providers)

        if self.device.startswith("cuda") and not use_cuda:
            logging.warning(
                f"[WARNING] Không tìm thấy CUDAExecutionProvider (khả dụng: {available_providers})! "
                "Mô hình InsightFace sẽ chạy trên CPU."
            )

        if use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        init_kwargs = {
            "name": self.model_name,
            "allowed_modules": ["detection", "landmark_3d_68"],
            "providers": providers,
        }
        if self.model_path is not None and self.model_path.is_dir():
            init_kwargs["root"] = str(self.model_path)

        self.app = FaceAnalysis(**init_kwargs)
        self.app.prepare(ctx_id=ctx_id, det_size=self.det_size)

    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Nhận danh sách frames BGR và trích xuất chuỗi 68 điểm mốc chuẩn hóa trong [0, 1].
        Trả về numpy array kích thước: (T, 68, 2)
        """
        if self.app is None:
            self._init_landmarker()

        landmarks_seq = []
        last_valid = None

        for frame in frames:
            h, w = frame.shape[:2]
            faces = self.app.get(frame)

            # Lọc các khuôn mặt có landmark_3d_68 hợp lệ
            valid_faces = [f for f in faces if getattr(f, "landmark_3d_68", None) is not None]

            if valid_faces:
                # Nếu có nhiều khuôn mặt, chọn khuôn mặt có diện tích bounding box lớn nhất
                if len(valid_faces) > 1:
                    face = max(
                        valid_faces,
                        key=lambda f: (
                            (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
                            if getattr(f, "bbox", None) is not None else 0.0
                            if getattr(f, "bbox", None) is not None else 0.0
                        ),
                    )
                else:
                    face = valid_faces[0]

                lmk_3d = face.landmark_3d_68  # Shape: (68, 3) gồm (x, y, z)
                lm68 = np.empty((68, 2), dtype=np.float32)
                lm68[:, 0] = lmk_3d[:, 0] / max(1, w)
                lm68[:, 1] = lmk_3d[:, 1] / max(1, h)
                last_valid = lm68
            else:
                # Forward-fill: dùng lại landmark của frame hợp lệ gần nhất nếu bị mất mặt
                lm68 = last_valid.copy() if last_valid is not None else np.zeros((68, 2), dtype=np.float32)

            landmarks_seq.append(lm68)

        # Backward-fill: nếu các frame đầu chưa thấy mặt nhưng sau đó phát hiện được
        if last_valid is not None:
            for i in range(len(landmarks_seq)):
                if np.all(landmarks_seq[i] == 0):
                    landmarks_seq[i] = last_valid.copy()
                else:
                    break

        return np.array(landmarks_seq, dtype=np.float32)

    def extract_from_video(
            self,
            video_path: str | Path,
            time_interval: Optional[float] = None,
            window_sec: Optional[float] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """
        Đọc frame trực tiếp từ video file và trích xuất chuỗi 68 landmark.
        - time_interval: khoảng thời gian (giây) giữa 2 frame lấy mẫu.
                         Nếu == 0: đọc toàn bộ frame tuần tự thông thường từ video.
                         Nếu None: lấy giá trị self.time_interval.
        - window_sec: thời lượng cửa sổ (giây) cần lấy. None = toàn bộ video.
        """
        video_path = Path(video_path)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise IOError(f"Không thể mở video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / fps if fps > 0 else 0.0

        ti = self.time_interval if time_interval is None else float(time_interval)
        if ti < 0:
            raise ValueError(f"time_interval phải là số không âm (>= 0): {ti}")

        if ti == 0:
            if window_sec is None:
                indices = np.arange(total_frames, dtype=int)
            else:
                win_sec = min(window_sec, duration_sec)
                num_frames = max(1, min(int(round(win_sec * fps)), total_frames))
                indices = np.arange(num_frames, dtype=int)
        else:
            if window_sec is None:
                timestamps = np.arange(0, duration_sec, ti)
            else:
                win_sec = min(window_sec, duration_sec)
                timestamps = np.arange(0, win_sec, ti)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
            indices = np.rint(timestamps * fps).astype(int)
            indices = np.clip(indices, 0, total_frames - 1)

        frames = []
        curr_frame = indices[0] if len(indices) > 0 else 0
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

        for target_idx in indices:
            while curr_frame < target_idx:
                if not cap.grab():
                    break
                curr_frame += 1
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            curr_frame += 1

        cap.release()

        if len(frames) < len(indices):
            last = frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
            frames.extend([last.copy()] * (len(indices) - len(frames)))

        lm68 = self.extract(frames)
        meta = {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frames": len(frames),
            "duration_sec": duration_sec,
            "time_interval": ti,
        }
        return lm68, frames, meta


# =====================================================================
# 3. LỚP GCNDATASET
# =====================================================================
class GCNDataset(Dataset):
    """
    Dataset đọc video từ thư mục dữ liệu (vd: E:\\LSTM\\data_processed),
    hỗ trợ:
      - Từng video có FPS riêng (60, 30, 25 FPS, v.v.).
      - time_interval = 0 -> đọc frame tuần tự thông thường từ video (không bỏ cách frame).
      - time_interval > 0 -> cơ chế lấy mẫu sau mỗi time_interval giây (time-domain sampling).
      - window_sec = None -> lấy frame trên toàn bộ thời lượng video.
      - window_sec = số giây -> lấy frame trong phạm vi cửa sổ đó.
    """

    def __init__(
            self,
            dataset_dir: str | Path = r"E:\LSTM\data_processed",
            split: str = "train",
            time_interval: float = 0.2,
            window_sec: Optional[float] = 10.0,
            manifest_filename: str = "dataset_merged_split.csv",
            landmark_model_path: Optional[str | Path] = None,
            letterbox_size: int = 480,
            return_frames: bool = False,
            device: str = "cuda",
            det_size: Tuple[int, int] = (640, 640),
            model_name: str = "buffalo_l",
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split không hợp lệ: {split}. Chỉ chấp nhận 'train', 'val', 'test'.")

        self.manifest_filename = manifest_filename

        if time_interval < 0:
            raise ValueError(f"time_interval phải là số không âm (>= 0): {time_interval}")
        if window_sec is not None and window_sec <= 0:
            raise ValueError(f"window_sec phải là số dương hoặc None: {window_sec}")

        self.root = Path(dataset_dir)
        self.split = split
        self.time_interval = float(time_interval)
        self.window_sec = float(window_sec) if window_sec is not None else None
        self.letterbox_size = letterbox_size
        self.return_frames = return_frames
        self.device = device
        self.det_size = det_size
        self.model_name = model_name
        self.extractor = FaceLandmarkExtractor(
            landmark_model_path,
            time_interval=self.time_interval,
            device=self.device,
            det_size=self.det_size,
            model_name=self.model_name,
        )
        self.samples = self._load_samples()

        if len(self.samples) == 0:
            logging.warning(f"Không tìm thấy mẫu nào trong {self.root} cho split={split}")

    def _load_samples(self) -> List[Dict[str, Any]]:
        """Nạp danh sách video từ manifest CSV hoặc quét thư mục dự phòng."""
        manifest = self.root / self.manifest_filename
        samples = []

        if manifest.is_file():
            with manifest.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("split", "").strip().lower() == self.split:
                        rel_path = row.get("rel_path") or row.get("processed_path")
                        samples.append({
                            "path": self.root / rel_path.replace("\\", "/"),
                            "label": int(row["label"]),
                            "video_id": row.get("video_id") or row.get("clip_id") or Path(rel_path).stem,
                            "fps": float(row.get("fps", 0.0)),
                            "total_frames": int(float(row.get("frame_count", 0))),
                            "duration_sec": float(row.get("duration_sec", 0.0)),
                        })
        else:
            for label, folder in ((0, "0_alert"), (1, "1_drowsy")):
                split_dir = self.root / self.split / folder
                if split_dir.is_dir():
                    for p in sorted(split_dir.glob("*.mp4")):
                        samples.append({
                            "path": p,
                            "label": label,
                            "video_id": p.stem,
                            "fps": 0.0,
                            "total_frames": 0,
                            "duration_sec": 0.0,
                        })
        return samples

    def _get_frame_indices(self, total_frames: int, fps: float) -> np.ndarray:
        """Tính toán các chỉ số frame cần lấy dựa trên miền thời gian thực và FPS riêng của video."""
        if total_frames <= 0:
            return np.array([0], dtype=int)

        fps = fps if fps > 0 else 30.0
        duration_sec = total_frames / fps

        # TRƯỜNG HỢP 1: time_interval == 0 -> Đọc frame tuần tự thông thường từ video (không bỏ cách frame)
        if self.time_interval == 0:
            if self.window_sec is None:
                # Toàn bộ video tuần tự
                return np.arange(total_frames, dtype=int)
            else:
                # Chuỗi frame liên tiếp trong phạm vi cửa sổ window_sec
                win_sec = min(self.window_sec, duration_sec)
                num_frames = int(round(win_sec * fps))
                num_frames = max(1, min(num_frames, total_frames))
                max_start = max(0, total_frames - num_frames)
                start_idx = random.randint(0, max_start) if (
                            self.split == "train" and max_start > 0) else max_start // 2
                return np.arange(start_idx, start_idx + num_frames, dtype=int)

        # TRƯỜNG HỢP 2: time_interval > 0 -> Lấy mẫu cách đều theo thời gian thực (time-domain sampling)
        if self.window_sec is None:
            # Lấy mẫu trên toàn bộ thời lượng video
            timestamps = np.arange(0, duration_sec, self.time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
        else:
            # Lấy mẫu trong phạm vi cửa sổ window_sec
            win = min(self.window_sec, duration_sec)
            max_start = max(0.0, duration_sec - win)
            start_sec = random.uniform(0.0, max_start) if (self.split == "train" and max_start > 0) else max_start / 2.0
            timestamps = start_sec + np.arange(0, win, self.time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([start_sec])

        # Quy đổi từ mốc thời gian vật lý sang chỉ số frame theo FPS riêng của video
        indices = np.rint(timestamps * fps).astype(int)
        indices = np.clip(indices, 0, total_frames - 1)
        return indices

    def _read_frames(self, path: Path, indices: np.ndarray) -> List[np.ndarray]:
        """Đọc nhanh các frame tại các chỉ số được chỉ định bằng cv2.VideoCapture."""
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise IOError(f"Không thể mở video: {path}")

        frames = []
        curr_frame = indices[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)

        for target_idx in indices:
            while curr_frame < target_idx:
                if not cap.grab():
                    break
                curr_frame += 1
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
            curr_frame += 1

        cap.release()

        # Bù frame cuối nếu video kết thúc sớm hơn dự kiến
        if len(frames) < len(indices):
            last = frames[-1] if frames else np.zeros((480, 640, 3), dtype=np.uint8)
            frames.extend([last.copy()] * (len(indices) - len(frames)))

        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        sample = self.samples[index]
        fps = sample["fps"]
        total_frames = sample["total_frames"]

        # Nếu thiếu metadata, đọc trực tiếp từ video file
        if fps <= 0 or total_frames <= 0:
            cap = cv2.VideoCapture(str(sample["path"]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            sample["fps"] = fps
            sample["total_frames"] = total_frames

        indices = self._get_frame_indices(total_frames, fps)
        frames = self._read_frames(sample["path"], indices)

        if self.letterbox_size:
            frames = [letterbox(f, new_size=self.letterbox_size) for f in frames]

        # Trích xuất 68 điểm mốc: shape (T, 68, 2)
        lm68 = self.extractor.extract(frames)

        # Chuyển đổi sang tensor chuẩn cho GCN: shape (2, T, 68)
        landmarks_tensor = torch.from_numpy(lm68).permute(2, 0, 1)

        label_tensor = torch.tensor(sample["label"], dtype=torch.long)
        meta = {
            "video_id": sample["video_id"],
            "path": str(sample["path"]),
            "fps": fps,
            "num_sampled_frames": len(indices),
            "duration_sec": total_frames / fps if fps > 0 else 0.0,
            "time_interval": self.time_interval,
        }

        if self.return_frames:
            return landmarks_tensor, label_tensor, meta, frames
        return landmarks_tensor, label_tensor, meta


# Alias hỗ trợ nếu người dùng gọi FaceLandmarkExtractor / FaceLandmarkDataset dưới dạng Dataset
FaceLandmarkDataset = GCNDataset
FaceLandmarkExtractorDataset = GCNDataset


# =====================================================================
# 4. HÀM VẼ LANDMARK VÀ HIỂN THỊ VIDEO
# =====================================================================
def get_landmark_region_color(idx: int) -> Tuple[int, int, int]:
    """
    Trả về màu sắc đại diện (BGR) theo từng vùng giải phẫu của hệ 68 điểm mốc:
      - 00 - 16: Viền hàm (Jawline) -> Xám bạc (200, 200, 200)
      - 17 - 21: Lông mày trái (Left Eyebrow) -> Vàng cam (255, 190, 0)
      - 22 - 26: Lông mày phải (Right Eyebrow) -> Vàng cam (255, 190, 0)
      - 27 - 30: Sống mũi (Nose Bridge) -> Vàng chanh (0, 255, 255)
      - 31 - 35: Chân mũi / Cánh mũi (Lower Nose) -> Vàng chanh (0, 255, 255)
      - 36 - 41: Mắt trái (Left Eye) -> Xanh lá (0, 255, 0)
      - 42 - 47: Mắt phải (Right Eye) -> Xanh lá (0, 255, 0)
      - 48 - 59: Môi ngoài (Outer Lip) -> Đỏ (0, 0, 255)
      - 60 - 67: Môi trong (Inner Lip) -> Cam đậm (0, 140, 255)
    """
    if 0 <= idx <= 16:
        return (200, 200, 200)  # Viền hàm
    elif 17 <= idx <= 26:
        return (255, 190, 0)  # Lông mày trái & phải
    elif 27 <= idx <= 35:
        return (0, 255, 255)  # Sống mũi & cánh mũi
    elif 36 <= idx <= 47:
        return (0, 255, 0)  # Mắt trái & phải
    elif 48 <= idx <= 59:
        return (0, 0, 255)  # Môi ngoài
    elif 60 <= idx <= 67:
        return (0, 140, 255)  # Môi trong
    return (0, 255, 0)


def draw_landmarks_on_frame(
        frame: np.ndarray,
        landmarks_xy: np.ndarray,
        show_ids: bool | List[int] | str = False,
        font_scale: float = 0.28,
        id_color: Optional[Tuple[int, int, int] | str] = (0, 255, 255),
        point_radius: int = 2,
        point_color: Optional[Tuple[int, int, int]] = (0, 255, 0),
        draw_contours: bool = True,
) -> np.ndarray:
    """
    Vẽ 68 điểm mốc khuôn mặt và các đường viền giải phẫu lên frame ảnh,
    đồng thời hỗ trợ hiển thị ID của các điểm theo hệ 68 điểm mốc của dự án (0-67).

    Tham số:
        frame: Ảnh gốc định dạng numpy array (H, W, 3) BGR.
        landmarks_xy: Tọa độ điểm mốc shape (68, 2) đã chuẩn hóa trong đoạn [0, 1].
        show_ids:
            - True: Hiển thị ID của toàn bộ 68 điểm (0 đến 67).
            - False / None: Tắt hiển thị ID.
            - List[int]: Danh sách các ID cụ thể cần hiển thị (vd: list(range(36, 48)) cho vùng mắt).
            - str: Từ khóa vùng giải phẫu: "eyes", "mouth", "nose", "eyebrows", "jaw", "all".
        font_scale: Kích thước chữ số hiển thị ID (mặc định 0.28, nhỏ gọn, rõ nét).
        id_color: Màu sắc chữ số (BGR). Mặc định vàng (0, 255, 255).
                  Nếu đặt id_color="region" hoặc id_color=None: Chữ số sẽ mang màu đồng bộ
                  theo từng vùng giải phẫu tương ứng.
        point_radius: Bán kính chấm tròn của mốc (mặc định 2).
        point_color: Màu sắc chấm tròn mốc (mặc định xanh lá). Nếu None: lấy theo màu vùng.
        draw_contours: Vẽ các đường liên kết giải phẫu (viền hàm, mắt, mày, môi, mũi).

    Hệ 68 điểm mốc của dự án:
        - 00 - 16: Viền hàm (17 điểm)
        - 17 - 21: Lông mày trái (5 điểm)
        - 22 - 26: Lông mày phải (5 điểm)
        - 27 - 30: Sống mũi (4 điểm)
        - 31 - 35: Chân / Cánh mũi (5 điểm)
        - 36 - 41: Mắt trái (6 điểm)
        - 42 - 47: Mắt phải (6 điểm)
        - 48 - 59: Môi ngoài (12 điểm)
        - 60 - 67: Môi trong (8 điểm)
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    num_pts = min(68, len(landmarks_xy))
    pts = []
    for i in range(num_pts):
        x = int(landmarks_xy[i, 0] * w)
        y = int(landmarks_xy[i, 1] * h)
        pts.append((x, y))

    # 1. Vẽ các đường nối giải phẫu (Contours)
    if draw_contours and len(pts) >= 68:
        def draw_chain(indices, closed=False, color=(0, 255, 255), thickness=1):
            for i in range(len(indices) - 1):
                cv2.line(vis, pts[indices[i]], pts[indices[i + 1]], color, thickness)
            if closed:
                cv2.line(vis, pts[indices[-1]], pts[indices[0]], color, thickness)

        draw_chain(list(range(0, 17)), closed=False, color=(200, 200, 200), thickness=1)  # Viền hàm
        draw_chain(list(range(17, 22)), closed=False, color=(255, 190, 0), thickness=1)  # Lông mày trái
        draw_chain(list(range(22, 27)), closed=False, color=(255, 190, 0), thickness=1)  # Lông mày phải
        draw_chain(list(range(27, 31)), closed=False, color=(0, 255, 255), thickness=1)  # Sống mũi
        draw_chain(list(range(31, 36)), closed=False, color=(0, 255, 255), thickness=1)  # Cánh mũi
        draw_chain(list(range(36, 42)), closed=True, color=(0, 255, 0), thickness=1)  # Mắt trái
        draw_chain(list(range(42, 48)), closed=True, color=(0, 255, 0), thickness=1)  # Mắt phải
        draw_chain(list(range(48, 60)), closed=True, color=(0, 0, 255), thickness=1)  # Môi ngoài
        draw_chain(list(range(60, 68)), closed=True, color=(0, 140, 255), thickness=1)  # Môi trong

    # 2. Vẽ 68 điểm mốc dạng chấm tròn
    for i, (x, y) in enumerate(pts):
        c = point_color if point_color is not None else get_landmark_region_color(i)
        cv2.circle(vis, (x, y), point_radius, c, -1)

    # 3. Vẽ ID của các điểm theo hệ 68 điểm của dự án (0 đến 67)
    if show_ids:
        # Xác định tập hợp ID cần vẽ
        if isinstance(show_ids, bool):
            target_ids = list(range(len(pts)))
        elif isinstance(show_ids, str):
            kw = show_ids.lower().strip()
            regions = {
                "jaw": list(range(0, 17)),
                "eyebrows": list(range(17, 27)),
                "left_eyebrow": list(range(17, 22)),
                "right_eyebrow": list(range(22, 27)),
                "nose": list(range(27, 36)),
                "nose_bridge": list(range(27, 31)),
                "lower_nose": list(range(31, 36)),
                "eyes": list(range(36, 48)),
                "left_eye": list(range(36, 42)),
                "right_eye": list(range(42, 48)),
                "mouth": list(range(48, 68)),
                "outer_lip": list(range(48, 60)),
                "inner_lip": list(range(60, 68)),
                "all": list(range(len(pts))),
            }
            target_ids = regions.get(kw, list(range(len(pts))))
        elif hasattr(show_ids, "__iter__"):
            target_ids = list(show_ids)
        else:
            target_ids = []

        for i in target_ids:
            if 0 <= i < len(pts):
                x, y = pts[i]
                # Canh chỉnh tọa độ đặt text ID để không bị che khuất điểm và không tràn mép ảnh
                tx = max(0, min(w - 18, x + 2))
                ty = max(10, min(h - 2, y - 2))
                text = str(i)

                # Xác định màu sắc text
                if id_color is None or id_color == "region":
                    text_color = get_landmark_region_color(i)
                elif isinstance(id_color, (tuple, list)):
                    text_color = tuple(id_color)
                else:
                    text_color = (0, 255, 255)

                # Viền chữ đen dày 2px tạo độ tương phản tuyệt đối trên mọi nền ảnh
                cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
                # Chữ số chính dày 1px
                cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, 1, cv2.LINE_AA)

    return vis


def visualize_frame_landmarks(
        frame: np.ndarray,
        landmarks_xy: np.ndarray,
        window_name: str = "Driver Guardian - 68 Facial Landmarks",
        show_ids: bool | List[int] | str = True,
        font_scale: float = 0.28,
        id_color: Optional[Tuple[int, int, int] | str] = (0, 255, 255),
        wait_key: bool = True,
) -> np.ndarray:
    """
    Hiển thị 1 frame ảnh kèm 68 điểm mốc và ID của chúng theo hệ 68 điểm của dự án.
    Rất thuận tiện cho việc kiểm tra nhanh và debug từng frame đơn lẻ.
    """
    vis = draw_landmarks_on_frame(
        frame=frame,
        landmarks_xy=landmarks_xy,
        show_ids=show_ids,
        font_scale=font_scale,
        id_color=id_color,
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = vis.shape[:2]
    cv2.resizeWindow(window_name, min(w, 800), min(h, 600))
    cv2.imshow(window_name, vis)
    if wait_key:
        print(f"\n[Hiển thị Frame Landmark] Đang mở cửa sổ '{window_name}' (show_ids={show_ids}).")
        print(">>> Nhấn phím bất kỳ hoặc 'q' trên cửa sổ ảnh để đóng <<<")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return vis


def visualize_video_landmarks(
        landmarks_tensor: torch.Tensor | np.ndarray,
        frames: List[np.ndarray],
        meta: Dict[str, Any],
        label: torch.Tensor | int = 0,
        time_interval: float = 0.2,
        window_name: str = "Driver Guardian - 68 Facial Landmarks",
        show_ids: bool | List[int] | str = False,
        font_scale: float = 0.28,
        id_color: Optional[Tuple[int, int, int] | str] = (0, 255, 255),
):
    """
    Hiển thị video tuần tự từng frame kèm các điểm mốc landmark đã trích xuất.
    Hỗ trợ hiển thị và bật/tắt ID của 68 điểm mốc theo hệ chuẩn của dự án (0-67).

    Phím điều khiển tương tác trên cửa sổ video:
        - [i] hoặc [I]: Bật / Tắt hiển thị ID của 68 điểm mốc trong thời gian thực.
        - [Space]: Tạm dừng / Tiếp tục phát video (dùng khi muốn dừng lại soi kỹ từng ID).
        - [d] / [Mũi tên phải]: Tiến 1 frame (khi đang tạm dừng).
        - [a] / [Mũi tên trái]: Lùi 1 frame (khi đang tạm dừng).
        - [q] hoặc [ESC]: Thoát video sớm.
    """
    # Xử lý linh hoạt landmarks_tensor: nhận cả (2, T, 68), (T, 68, 2), (B, 2, T, 68) hoặc np.ndarray
    if isinstance(landmarks_tensor, torch.Tensor):
        if landmarks_tensor.ndim == 4:
            landmarks_tensor = landmarks_tensor[0]
        if landmarks_tensor.shape[0] == 2:
            lm_np = landmarks_tensor.permute(1, 2, 0).cpu().numpy()
        else:
            lm_np = landmarks_tensor.cpu().numpy()
    elif isinstance(landmarks_tensor, np.ndarray):
        if landmarks_tensor.ndim == 3 and landmarks_tensor.shape[0] == 2:
            lm_np = np.transpose(landmarks_tensor, (1, 2, 0))
        else:
            lm_np = landmarks_tensor
    else:
        raise TypeError(f"landmarks_tensor không đúng kiểu: {type(landmarks_tensor)}")

    T = min(lm_np.shape[0], len(frames))
    fps = meta.get("fps", 30.0) or 30.0
    if time_interval == 0:
        delay_ms = max(1, int(1000 / fps))
        total_time_str = f"~{T / fps:.1f}s"
    else:
        delay_ms = max(1, int(time_interval * 1000))
        total_time_str = f"~{T * time_interval:.1f}s"

    lbl_val = label.item() if isinstance(label, torch.Tensor) else int(label)
    label_text = "ALERT (0)" if lbl_val == 0 else "DROWSY (1)"
    label_color = (0, 255, 0) if lbl_val == 0 else (0, 0, 255)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = frames[0].shape[:2]
    cv2.resizeWindow(window_name, min(w, 800), min(h, 600))

    video_id = meta.get("video_id", "Sample")
    print(f"\n[Playing Video] {video_id} ({T} frames, {total_time_str})")
    print(
        ">>> Phím nóng: [i] Bật/tắt ID (0-67) | [Space] Tạm dừng/Soi chi tiết | [a/d] Frame lùi/tiến | [q/ESC] Thoát <<<")

    t = 0
    paused = False
    show_ids_state = show_ids

    while t < T:
        vis = draw_landmarks_on_frame(
            frame=frames[t],
            landmarks_xy=lm_np[t],
            show_ids=show_ids_state,
            font_scale=font_scale,
            id_color=id_color,
        )

        # Dòng 1: Video ID & Trạng thái Alert / Drowsy
        cv2.putText(vis, f"Video: {video_id} | Status: {label_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"Video: {video_id} | Status: {label_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)

        # Dòng 2: Chỉ số frame, mốc thời gian thực, FPS và trạng thái tạm dừng
        curr_time_sec = (t / fps) if time_interval == 0 else (t * time_interval)
        time_info = f"Frame: {t + 1}/{T} | Time: {curr_time_sec:.1f}s | FPS: {fps:.1f}"
        if paused:
            time_info += "  [PAUSED - Soi ID]"
        cv2.putText(vis, time_info, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, time_info, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Dòng 3: Hướng dẫn phím nóng tương tác
        id_str = "ON" if show_ids_state else "OFF"
        guide = f"[i]: IDs ({id_str}) | [Space]: {'Resume' if paused else 'Pause'} | [q/ESC]: Exit"
        cv2.putText(vis, guide, (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, guide, (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, vis)

        # Kiểm tra nếu cửa sổ bị tắt bởi nút X
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            print("Cửa sổ đã được đóng.")
            break

        wait_ms = 0 if paused else delay_ms
        key = cv2.waitKey(wait_ms) & 0xFF

        if key in (ord('q'), ord('Q'), 27):
            print("Đã nhận lệnh dừng từ người dùng.")
            break
        elif key in (ord('i'), ord('I')):
            show_ids_state = not show_ids_state
            print(f">>> Landmark IDs (0-67): {'BẬT' if show_ids_state else 'TẮT'}")
            continue
        elif key == 32:  # Phím Space
            paused = not paused
            print(f">>> {'TẠM DỪNG' if paused else 'TIẾP TỤC'}")
            continue
        elif paused and key in (ord('d'), ord('D'), 83):  # Frame kế tiếp
            t = min(T - 1, t + 1)
            continue
        elif paused and key in (ord('a'), ord('A'), 81):  # Frame trước đó
            t = max(0, t - 1)
            continue

        if not paused:
            t += 1

    cv2.destroyAllWindows()


# =====================================================================
# 5. HÀM COLLATE VÀ BUILD DATALOADER
# =====================================================================
def collate_gcn(batch: List[Tuple[Any, ...]]):
    """
    Gom các mẫu vào batch.
    Khi window_sec=None (số frame T giữa các video có thể khác nhau) hoặc các video có độ dài khác nhau,
    tự động pad thêm frame 0 ở chiều thời gian T về max_T của batch và trả về tensor lengths.

    Returns:
        - padded_landmarks: Tensor (B, 2, max_T, 68)
        - labels          : LongTensor (B,)
        - lengths         : LongTensor (B,) chứa số frame thực tế trước khi pad của từng mẫu
        - metas           : List[Dict[str, Any]]
        - frames (tùy chọn): List[List[np.ndarray]] nếu dataset bật return_frames=True
    """
    has_frames = len(batch[0]) == 4
    if has_frames:
        landmarks, labels, metas, frames_list = zip(*batch)
    else:
        landmarks, labels, metas = zip(*batch)

    lengths = [lm.shape[1] for lm in landmarks]
    max_len = max(lengths)

    padded_landmarks = []
    updated_metas = []
    for lm, meta, t in zip(landmarks, metas, lengths):
        if t < max_len:
            pad_tensor = torch.zeros((lm.shape[0], max_len, lm.shape[2]), dtype=lm.dtype)
            pad_tensor[:, :t, :] = lm
            padded_landmarks.append(pad_tensor)
        else:
            padded_landmarks.append(lm)

        meta_copy = dict(meta)
        meta_copy["valid_frames"] = t
        meta_copy["padded_frames"] = max_len
        updated_metas.append(meta_copy)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    if has_frames:
        return torch.stack(padded_landmarks), torch.stack(labels), lengths_tensor, updated_metas, list(frames_list)
    return torch.stack(padded_landmarks), torch.stack(labels), lengths_tensor, updated_metas


def build_gcn_dataloaders(
        dataset_dir: str | Path = r"E:\LSTM\data_processed",
        batch_size: int = 16,
        num_workers: int = 0,
        time_interval: float = 0.2,
        window_sec: Optional[float] = 10.0,
        manifest_filename: str = "dataset_merged_split.csv",
        pin_memory: bool = True,
        shuffle_train: bool = True,
        landmark_model_path: Optional[str | Path] = None,
        letterbox_size: int = 480,
        device: str = "cuda",
        det_size: Tuple[int, int] = (640, 640),
        model_name: str = "buffalo_l",
):
    """
    Xây dựng nhanh bộ DataLoader cho cả 3 tập train, val, test.
    - time_interval = 0: đọc toàn bộ frame tuần tự thông thường từ video.
    - time_interval > 0: lấy mẫu cách nhau time_interval giây (vd 0.2s tương đương 5 FPS).
    """
    train_ds = GCNDataset(
        dataset_dir=dataset_dir,
        split="train",
        time_interval=time_interval,
        window_sec=window_sec,
        manifest_filename=manifest_filename,
        landmark_model_path=landmark_model_path,
        letterbox_size=letterbox_size,
        device=device,
        det_size=det_size,
        model_name=model_name,
    )
    val_ds = GCNDataset(
        dataset_dir=dataset_dir,
        split="val",
        time_interval=time_interval,
        window_sec=window_sec,
        manifest_filename=manifest_filename,
        landmark_model_path=landmark_model_path,
        letterbox_size=letterbox_size,
        device=device,
        det_size=det_size,
        model_name=model_name,
    )
    test_ds = GCNDataset(
        dataset_dir=dataset_dir,
        split="test",
        time_interval=time_interval,
        window_sec=window_sec,
        manifest_filename=manifest_filename,
        landmark_model_path=landmark_model_path,
        letterbox_size=letterbox_size,
        device=device,
        det_size=det_size,
        model_name=model_name,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_gcn
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_gcn
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=collate_gcn
    )
    return train_loader, val_loader, test_loader


# =====================================================================
# 6. KHỐI LOAD 1 VIDEO VÀ HIỂN THỊ LANDMARK
# =====================================================================
if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    DATASET_DIR = r"E:\LSTM\data_processed"
    print("=" * 60)
    print("LOAD 1 VIDEO TU GCNDataset VA HIEN THI LANDMARK")
    print("=" * 60)

    # 1. Khởi tạo GCNDataset với return_frames=True để lấy cả ảnh hiển thị
    # Ghi chú:
    # - time_interval = 0  -> đọc chuỗi frame thông thường tuần tự (không bỏ cách frame)
    # - time_interval > 0  -> lấy mẫu cách nhau time_interval giây (vd: 0.2s = 5 FPS)
    ds = GCNDataset(
        dataset_dir=DATASET_DIR,
        split="train",
        time_interval=0,  # Đọc frame tuần tự thông thường từ video
        letterbox_size=640,
        window_sec=None,  # Lấy cửa sổ 5 giây (vd: 30 FPS -> 150 frames liên tiếp)
        return_frames=True,
    )
    print(f"Tổng số mẫu trong tập train: {len(ds)}")

    # 2. Lấy mẫu video đầu tiên
    landmarks, label, meta, frames = ds[1000]
    print(f"Video ID      : {meta['video_id']}")
    print(f"Đường dẫn file: {meta['path']}")
    print(f"FPS nguồn     : {meta['fps']}")
    print(f"Thời lượng    : {meta['duration_sec']:.1f}s")
    print(f"Số frame lấy  : {len(frames)} frames")
    print(f"Landmarks     : shape {landmarks.shape} (C=2, T={landmarks.shape[1]}, V=68)")
    print(f"Nhãn (Label)  : {label.item()} ({'Alert' if label.item() == 0 else 'Drowsy'})")

    # 3. Điền các landmark và hiển thị video (bật show_ids=True để hiển thị toàn bộ ID 0-67)
    # Ghi chú: Trong lúc video phát, có thể nhấn 'i' để bật/tắt ID hoặc 'Space' để tạm dừng soi từng điểm mốc.
    visualize_video_landmarks(landmarks, frames, meta, label, time_interval=ds.time_interval, show_ids=True)
