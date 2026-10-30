# File: dataset.py
import csv
import logging
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

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
# 2. BỘ TRÍCH XUẤT 68 ĐIỂM MỐC KHUÔN MẶT BẰNG MEDIAPIPE TASKS
# =====================================================================
class FaceLandmarkExtractor:
    """Trích xuất 68 điểm mốc chuẩn Dlib từ các khung hình ảnh bằng MediaPipe Tasks."""

    def __init__(self, model_path: Optional[str | Path] = None, time_interval: float = 0.0):
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "PhysicalBranch" / "models" / "face_landmarker.task"
        self.model_path = Path(model_path)
        self.time_interval = float(time_interval)
        self.landmarker = None

    def _init_landmarker(self):
        """Khởi tạo trễ (lazy init) để tương thích đa tiến trình của DataLoader."""
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy file mô hình MediaPipe tại: {self.model_path}")

        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Nhận danh sách frames BGR và trích xuất chuỗi 68 điểm mốc chuẩn hóa trong [0, 1].
        Trả về numpy array kích thước: (T, 68, 2)
        """
        if self.landmarker is None:
            self._init_landmarker()

        import mediapipe as mp

        landmarks_seq = []
        last_valid = None

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_image)

            if result.face_landmarks:
                pts = result.face_landmarks[0]
                lm68 = np.array([[pts[i].x, pts[i].y] for i in LANDMARK_68_INDICES], dtype=np.float32)
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
        dataset_dir: Optional[str | Path] = None,
        split: str = "train",
        time_interval: Optional[float] = None,
        window_sec: Optional[float] = 10.0,
        landmark_model_path: Optional[str | Path] = None,
        return_frames: bool = False,
        config: Optional[Any] = None,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"Split không hợp lệ: {split}. Chỉ chấp nhận 'train', 'val', 'test'.")

        # Nạp giá trị từ config nếu được truyền
        if config is not None:
            ds_cfg = getattr(config, "dataset", config)
            if dataset_dir is None:
                dataset_dir = getattr(ds_cfg, "dataset_dir", r"E:\LSTM\data_processed")
            if time_interval is None:
                time_interval = getattr(ds_cfg, "time_interval", 0.2)
            if window_sec is None and hasattr(ds_cfg, "window_sec"):
                window_sec = getattr(ds_cfg, "window_sec", 10.0)
            if landmark_model_path is None:
                landmark_model_path = getattr(ds_cfg, "landmark_model_path", None)
            self.manifest_filename = getattr(ds_cfg, "manifest_filename", "dataset_merged_split.csv")
        else:
            self.manifest_filename = "dataset_merged_split.csv"

        if dataset_dir is None:
            dataset_dir = r"E:\LSTM\data_processed"
        if time_interval is None:
            time_interval = 0.2

        if time_interval < 0:
            raise ValueError(f"time_interval phải là số không âm (>= 0): {time_interval}")
        if window_sec is not None and window_sec <= 0:
            raise ValueError(f"window_sec phải là số dương hoặc None: {window_sec}")

        self.root = Path(dataset_dir)
        self.split = split
        self.time_interval = float(time_interval)
        self.window_sec = float(window_sec) if window_sec is not None else None
        self.return_frames = return_frames
        self.extractor = FaceLandmarkExtractor(landmark_model_path, time_interval=self.time_interval)
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
                start_idx = random.randint(0, max_start) if (self.split == "train" and max_start > 0) else max_start // 2
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
def draw_landmarks_on_frame(frame: np.ndarray, landmarks_xy: np.ndarray) -> np.ndarray:
    """
    Vẽ 68 điểm mốc khuôn mặt và các đường viền giải phẫu lên frame ảnh.
    landmarks_xy: numpy array shape (68, 2) với tọa độ chuẩn hóa trong [0, 1].
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    pts = []
    for i in range(68):
        x = int(landmarks_xy[i, 0] * w)
        y = int(landmarks_xy[i, 1] * h)
        pts.append((x, y))

    # 1. Vẽ các đường nối giải phẫu (Contours)
    def draw_chain(indices, closed=False, color=(0, 255, 255), thickness=1):
        for i in range(len(indices) - 1):
            cv2.line(vis, pts[indices[i]], pts[indices[i + 1]], color, thickness)
        if closed:
            cv2.line(vis, pts[indices[-1]], pts[indices[0]], color, thickness)

    draw_chain(list(range(0, 17)), closed=False, color=(200, 200, 200), thickness=1)  # Viền hàm
    draw_chain(list(range(17, 22)), closed=False, color=(255, 190, 0), thickness=1)   # Lông mày trái
    draw_chain(list(range(22, 27)), closed=False, color=(255, 190, 0), thickness=1)   # Lông mày phải
    draw_chain(list(range(27, 31)), closed=False, color=(0, 255, 255), thickness=1)   # Sống mũi
    draw_chain(list(range(31, 36)), closed=False, color=(0, 255, 255), thickness=1)   # Cánh mũi
    draw_chain(list(range(36, 42)), closed=True, color=(0, 255, 0), thickness=1)      # Mắt trái
    draw_chain(list(range(42, 48)), closed=True, color=(0, 255, 0), thickness=1)      # Mắt phải
    draw_chain(list(range(48, 60)), closed=True, color=(0, 0, 255), thickness=1)      # Môi ngoài
    draw_chain(list(range(60, 68)), closed=True, color=(0, 140, 255), thickness=1)    # Môi trong

    # 2. Vẽ 68 điểm mốc dạng chấm tròn
    for x, y in pts:
        cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)

    return vis


def visualize_video_landmarks(
    landmarks_tensor: torch.Tensor,
    frames: List[np.ndarray],
    meta: Dict[str, Any],
    label: torch.Tensor,
    time_interval: float = 0.2,
    window_name: str = "Driver Guardian - 68 Facial Landmarks",
):
    """
    Hiển thị video tuần tự từng frame kèm các điểm mốc landmark đã trích xuất.
    Hỗ trợ cả time_interval > 0 (lấy mẫu giãn cách) và time_interval == 0 (video thông thường).
    Nhấn phím 'q' hoặc ESC trên cửa sổ để dừng hiển thị sớm.
    """
    T = landmarks_tensor.shape[1]
    # Chuyển về shape (T, 68, 2)
    lm_np = landmarks_tensor.permute(1, 2, 0).cpu().numpy()

    fps = meta.get("fps", 30.0) or 30.0
    if time_interval == 0:
        delay_ms = max(1, int(1000 / fps))
        total_time_str = f"~{T / fps:.1f}s"
    else:
        delay_ms = max(1, int(time_interval * 1000))
        total_time_str = f"~{T * time_interval:.1f}s"

    label_text = "ALERT (0)" if label.item() == 0 else "DROWSY (1)"
    label_color = (0, 255, 0) if label.item() == 0 else (0, 0, 255)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    h, w = frames[0].shape[:2]
    cv2.resizeWindow(window_name, min(w, 800), min(h, 600))

    print(f"\n[Playing Video] {meta['video_id']} ({T} frames, {total_time_str})")
    print(">>> Nhấn phím 'q' hoặc ESC trên cửa sổ video để dừng hiển thị sớm <<<")

    for t in range(T):
        vis = draw_landmarks_on_frame(frames[t], lm_np[t])

        # Vẽ thông tin trạng thái lên ảnh
        cv2.putText(vis, f"Video: {meta['video_id']} | Status: {label_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"Video: {meta['video_id']} | Status: {label_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, label_color, 2, cv2.LINE_AA)

        curr_time_sec = (t / fps) if time_interval == 0 else (t * time_interval)
        info = f"Frame: {t + 1}/{T} | Time: {curr_time_sec:.1f}s | FPS: {fps:.1f}"
        cv2.putText(vis, info, (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, info, (15, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, vis)
        key = cv2.waitKey(delay_ms) & 0xFF
        if key == ord('q') or key == 27:
            print("Đã nhận lệnh dừng từ người dùng.")
            break

    cv2.destroyAllWindows()


# =====================================================================
# 5. HÀM COLLATE VÀ BUILD DATALOADER
# =====================================================================
def collate_gcn(batch: List[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]]):
    """
    Gom các mẫu vào batch.
    Khi window_sec=None (số frame T giữa các video có thể khác nhau),
    tự động pad thêm frame 0 ở chiều thời gian T về max_T của batch.
    """
    landmarks, labels, metas = zip(*batch)
    lengths = [lm.shape[1] for lm in landmarks]
    max_len = max(lengths)

    padded_landmarks = []
    for lm in landmarks:
        t = lm.shape[1]
        if t < max_len:
            pad_tensor = torch.zeros((2, max_len, 68), dtype=lm.dtype)
            pad_tensor[:, :t, :] = lm
            padded_landmarks.append(pad_tensor)
        else:
            padded_landmarks.append(lm)

    return torch.stack(padded_landmarks), torch.stack(labels), list(metas)


def build_gcn_dataloaders(
    dataset_dir: Optional[str | Path] = None,
    batch_size: Optional[int] = None,
    num_workers: Optional[int] = None,
    time_interval: Optional[float] = None,
    window_sec: Optional[float] = None,
    pin_memory: Optional[bool] = None,
    config: Optional[Any] = None,
):
    """
    Xây dựng nhanh bộ DataLoader cho cả 3 tập train, val, test.
    - time_interval = 0: đọc toàn bộ frame tuần tự thông thường từ video.
    - time_interval > 0: lấy mẫu cách nhau time_interval giây (vd 0.2s tương đương 5 FPS).
    - Hỗ trợ truyền cấu hình thông qua config (MasterConfig hoặc DatasetConfig/DataLoaderConfig).
    """
    ds_cfg = getattr(config, "dataset", config) if config is not None else None
    dl_cfg = getattr(config, "dataloader", config) if config is not None else None

    if dataset_dir is None:
        dataset_dir = getattr(ds_cfg, "dataset_dir", r"E:\LSTM\data_processed")
    if time_interval is None:
        time_interval = getattr(ds_cfg, "time_interval", 0.2)
    if window_sec is None:
        window_sec = getattr(ds_cfg, "window_sec", 10.0) if ds_cfg is not None else 10.0
    if batch_size is None:
        batch_size = getattr(dl_cfg, "batch_size", 16)
    if num_workers is None:
        num_workers = getattr(dl_cfg, "num_workers", 0)
    if pin_memory is None:
        pin_memory = getattr(dl_cfg, "pin_memory", True)

    train_ds = GCNDataset(dataset_dir, split="train", time_interval=time_interval, window_sec=window_sec, config=ds_cfg)
    val_ds = GCNDataset(dataset_dir, split="val", time_interval=time_interval, window_sec=window_sec, config=ds_cfg)
    test_ds = GCNDataset(dataset_dir, split="test", time_interval=time_interval, window_sec=window_sec, config=ds_cfg)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
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
        time_interval=0,         # Đọc frame tuần tự thông thường từ video
        window_sec=None,          # Lấy cửa sổ 5 giây (vd: 30 FPS -> 150 frames liên tiếp)
        return_frames=True,
    )
    print(f"Tổng số mẫu trong tập train: {len(ds)}")

    # 2. Lấy mẫu video đầu tiên
    landmarks, label, meta, frames = ds[0]
    print(f"Video ID      : {meta['video_id']}")
    print(f"Đường dẫn file: {meta['path']}")
    print(f"FPS nguồn     : {meta['fps']}")
    print(f"Thời lượng    : {meta['duration_sec']:.1f}s")
    print(f"Số frame lấy  : {len(frames)} frames")
    print(f"Landmarks     : shape {landmarks.shape} (C=2, T={landmarks.shape[1]}, V=68)")
    print(f"Nhãn (Label)  : {label.item()} ({'Alert' if label.item() == 0 else 'Drowsy'})")

    # 3. Điền các landmark và hiển thị video
    visualize_video_landmarks(landmarks, frames, meta, label, time_interval=ds.time_interval)
