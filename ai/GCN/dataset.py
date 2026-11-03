# File: dataset.py
from __future__ import annotations
import csv
import logging
import math
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

logger = logging.getLogger("DriverGuardian.GCN")

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


def _parse_letterbox_size(new_size: Optional[int | Tuple[int, int] | List[int]]) -> Tuple[int, int]:
    """Chuẩn hóa tham số kích thước letterbox về bộ (target_w, target_h). Trả về (0, 0) nếu tắt."""
    if new_size is None:
        return (0, 0)
    if isinstance(new_size, int):
        return (new_size, new_size) if new_size > 0 else (0, 0)
    if isinstance(new_size, (tuple, list)) and len(new_size) == 2:
        w, h = int(new_size[0]), int(new_size[1])
        return (w, h) if (w > 0 and h > 0) else (0, 0)
    raise ValueError(f"new_size phải là int hoặc tuple/list (w, h), nhận được: {new_size}")


def letterbox(
        image: np.ndarray,
        new_size: Optional[int | Tuple[int, int] | List[int]] = (640, 480),
        color: Tuple[int, int, int] = (114, 114, 114),
) -> np.ndarray:
    """
    Resize ảnh giữ nguyên tỷ lệ khung hình gốc với đệm màu viền về kích thước đích (w, h).
    - image: Ảnh BGR numpy array shape (H, W, 3).
    - new_size: Kích thước đích theo (w, h) (vd: (640, 480)) hoặc số nguyên S tương ứng (S, S).
    - color: Màu đệm viền BGR, mặc định (114, 114, 114).
    Trả về ảnh numpy array có shape (target_h, target_w, 3).
    """
    target_w, target_h = _parse_letterbox_size(new_size)
    if target_w <= 0 or target_h <= 0:
        return image

    h, w = image.shape[:2]
    if w == target_w and h == target_h:
        return image

    scale = min(target_w / float(w), target_h / float(h))
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (nw, nh), interpolation=interp)

    if nw == target_w and nh == target_h:
        return resized

    canvas = np.full((target_h, target_w, 3), color, dtype=image.dtype)
    x, y = (target_w - nw) // 2, (target_h - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def read_video_frames(
        video_path: str | Path,
        time_interval: float = 0.1,
        window_sec: Optional[float] = None,
        letterbox_size: Optional[int | Tuple[int, int] | List[int]] = (640, 480),
        max_side: int = 0,
) -> Tuple[List[np.ndarray], float, int, float]:
    """
    Đọc frame từ video chỉ với MỘT lần mở cv2.VideoCapture.
    - time_interval > 0: lấy mẫu cách đều theo thời gian thực; == 0: đọc tuần tự mọi frame.
    - window_sec: None = toàn bộ video; số giây = chỉ lấy cửa sổ đầu tiên (bắt đầu từ 0s).
    - letterbox_size: (w, h) kích thước chuẩn hóa letterbox đồng nhất tỷ lệ cho mọi video.
    - max_side: tùy chọn tương thích ngược (nếu letterbox_size=(0, 0)).
    Trả về (frames, fps, total_frames, duration_sec).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Không thể mở video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not math.isfinite(fps) or fps <= 0:
            fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            raise IOError(f"Video không có frame hợp lệ: {video_path}")
        duration_sec = total_frames / fps
        win = duration_sec if window_sec is None else min(float(window_sec), duration_sec)

        if time_interval == 0:
            num = total_frames if window_sec is None else max(1, min(int(round(win * fps)), total_frames))
            indices = np.arange(num, dtype=int)
        else:
            timestamps = np.arange(0, win, time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
            indices = np.clip(np.rint(timestamps * fps).astype(int), 0, total_frames - 1)

        target_w, target_h = _parse_letterbox_size(letterbox_size)
        frames: List[np.ndarray] = []
        curr_frame = 0
        for target_idx in indices:
            if target_idx < curr_frame and frames:
                frames.append(frames[-1])  # Chỉ số trùng lặp (time_interval < 1/fps)
                continue
            while curr_frame < target_idx:
                if not cap.grab():
                    break
                curr_frame += 1
            ret, frame = cap.read()
            if not ret:
                break
            if target_w > 0 and target_h > 0:
                frame = letterbox(frame, new_size=(target_w, target_h))
            elif max_side and max(frame.shape[:2]) > max_side:
                scale = max_side / float(max(frame.shape[:2]))
                frame = cv2.resize(
                    frame,
                    (max(1, round(frame.shape[1] * scale)), max(1, round(frame.shape[0] * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            frames.append(frame)
            curr_frame += 1
    finally:
        cap.release()

    if not frames:
        raise IOError(f"Không đọc được frame nào từ video: {video_path}")
    # Bù frame cuối nếu video kết thúc sớm hơn dự kiến
    if len(frames) < len(indices):
        frames.extend([frames[-1]] * (len(indices) - len(frames)))
    return frames, fps, total_frames, duration_sec


# =====================================================================
# 2. BỘ TRÍCH XUẤT 68 ĐIỂM MỐC KHUÔN MẶT BẰNG INSIGHTFACE (CUDA, BATCHED)
# =====================================================================
class FaceLandmarkExtractor:
    """
    Trích xuất 68 điểm mốc 3D (lấy x, y) bằng InsightFace, tối ưu tốc độ:
      - Chỉ nạp det_*.onnx (SCRFD) + 1k3d68.onnx, không nạp recognition / genderage / 2d106det.
      - max_num=1 (metric="max"): chỉ giữ khuôn mặt lớn nhất trước khi chạy 1k3d68.
      - Chạy 1k3d68.onnx theo lô (batched) cho toàn bộ frame hợp lệ của video.
      - Chỉ lấy 68 điểm cuối, biến đổi ngược Affine vector hóa bằng np.einsum, bỏ qua 3D Pose (lstsq).
    Tọa độ chuẩn hóa theo kích thước (w, h) của frame, kết quả trong [0, 1].
    """

    def __init__(
            self,
            model_path: Optional[str | Path] = None,
            time_interval: float = 0.0,
            device: str = "cuda",
            det_size: Tuple[int, int] = (320, 320),
            model_name: str = "buffalo_l",
            landmark_batch_size: int = 64,
            det_thresh: float = 0.5,
            letterbox_size: Optional[int | Tuple[int, int] | List[int]] = (640, 480),
    ):
        self.model_path = Path(model_path) if model_path is not None else None
        self.time_interval = float(time_interval)
        self.device = device.lower().strip()
        self.det_size = tuple(det_size)
        self.model_name = model_name
        self.landmark_batch_size = max(1, int(landmark_batch_size))
        self.det_thresh = det_thresh
        self.letterbox_size = _parse_letterbox_size(letterbox_size)
        self.det = None
        self.lmk = None

    def _init_landmarker(self):
        """Khởi tạo trễ (lazy init) chỉ 2 mô hình ONNX cần thiết để tương thích đa tiến trình."""
        import glob
        import onnxruntime as ort
        from insightface.model_zoo import model_zoo
        from insightface.utils import ensure_available

        ort.set_default_logger_severity(3)

        available_providers = ort.get_available_providers()
        use_cuda = self.device.startswith("cuda") and ("CUDAExecutionProvider" in available_providers)

        if self.device.startswith("cuda") and not use_cuda:
            logger.warning(
                f"[WARNING] Không tìm thấy CUDAExecutionProvider (khả dụng: {available_providers})! "
                "Mô hình InsightFace sẽ chạy trên CPU."
            )

        if use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            ctx_id = 0
        else:
            providers = ["CPUExecutionProvider"]
            ctx_id = -1

        root = str(self.model_path) if (self.model_path is not None and self.model_path.is_dir()) else "~/.insightface"
        model_dir = ensure_available("models", self.model_name, root=root)

        det_files = sorted(glob.glob(os.path.join(model_dir, "det_*.onnx")))
        lmk_file = os.path.join(model_dir, "1k3d68.onnx")
        if not det_files or not os.path.isfile(lmk_file):
            raise FileNotFoundError(
                f"Thiếu det_*.onnx hoặc 1k3d68.onnx trong {model_dir} (model_name='{self.model_name}')."
            )

        self.det = model_zoo.get_model(det_files[0], providers=providers)
        self.lmk = model_zoo.get_model(lmk_file, providers=providers)
        self.det.prepare(ctx_id, input_size=self.det_size, det_thresh=self.det_thresh)
        self.lmk.prepare(ctx_id)
        if hasattr(self.lmk, "require_pose"):
            self.lmk.require_pose = False
        logger.info(
            f"InsightFace ({self.model_name}) khởi tạo thành công trên provider: {providers[0]} "
            f"(det={os.path.basename(det_files[0])}, det_size={self.det_size}, "
            f"landmark_batch={self.landmark_batch_size}, letterbox={self.letterbox_size})."
        )

    def release(self) -> None:
        """Giải phóng session ONNX và bộ nhớ GPU."""
        import gc
        self.det = None
        self.lmk = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def extract(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Nhận danh sách frames BGR và trích xuất chuỗi 68 điểm mốc chuẩn hóa trong [0, 1].
        Trả về numpy array kích thước: (T, 68, 2)
        """
        if self.det is None or self.lmk is None:
            self._init_landmarker()

        from insightface.utils import face_align

        T = len(frames)
        if T == 0:
            return np.zeros((0, 68, 2), dtype=np.float32)

        target_w, target_h = self.letterbox_size
        if target_w > 0 and target_h > 0:
            processed_frames = []
            for f in frames:
                if f.shape[1] != target_w or f.shape[0] != target_h:
                    processed_frames.append(letterbox(f, new_size=(target_w, target_h)))
                else:
                    processed_frames.append(f)
            frames = processed_frames

        # 1. Phát hiện khuôn mặt lớn nhất trên từng frame
        bboxes: List[Optional[np.ndarray]] = []
        for frame in frames:
            det, _ = self.det.detect(frame, max_num=1, metric="max")
            bboxes.append(det[0, :4] if (det is not None and det.shape[0] > 0) else None)

        valid_idx = [i for i, b in enumerate(bboxes) if b is not None]
        lm68 = np.zeros((T, 68, 2), dtype=np.float32)
        found = np.zeros(T, dtype=bool)

        # 2. Chạy 1k3d68.onnx theo lô trên các frame có khuôn mặt
        isz = int(self.lmk.input_size[0])
        mean = float(self.lmk.input_mean)
        std = float(self.lmk.input_std)
        for s in range(0, len(valid_idx), self.landmark_batch_size):
            chunk = valid_idx[s:s + self.landmark_batch_size]
            aimgs, ims = [], []
            for i in chunk:
                bbox = bboxes[i]
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                center = (bbox[2] + bbox[0]) / 2, (bbox[3] + bbox[1]) / 2
                scale = isz / (max(w, h) * 1.5)
                aimg, M = face_align.transform(frames[i], center, isz, scale, 0)
                aimgs.append(aimg)
                ims.append(cv2.invertAffineTransform(M))

            blob = cv2.dnn.blobFromImages(aimgs, 1.0 / std, (isz, isz), (mean, mean, mean), swapRB=True)
            pred = self.lmk.session.run(self.lmk.output_names, {self.lmk.input_name: blob})[0]
            pred = pred.reshape(len(chunk), -1, 3)[:, -68:, :2].astype(np.float32)
            pred = (pred + 1.0) * (isz // 2)

            homo = np.concatenate([pred, np.ones((len(chunk), 68, 1), dtype=np.float32)], axis=-1)  # (n, 68, 3)
            im_batch = np.stack(ims).astype(np.float32)  # (n, 2, 3)
            out = np.einsum("nkj,nij->nki", homo, im_batch)  # (n, 68, 2)

            for j, i in enumerate(chunk):
                fh, fw = frames[i].shape[:2]
                lm68[i, :, 0] = out[j, :, 0] / float(max(1, fw))
                lm68[i, :, 1] = out[j, :, 1] / float(max(1, fh))
                found[i] = True

        # 3. Điền các frame mất mặt: forward-fill, sau đó backward-fill các frame đầu
        if found.any():
            last = None
            for i in range(T):
                if found[i]:
                    last = lm68[i]
                elif last is not None:
                    lm68[i] = last
            first = int(np.argmax(found))
            if first > 0:
                lm68[:first] = lm68[first]

        return lm68

    def extract_from_video(
            self,
            video_path: str | Path,
            time_interval: Optional[float] = None,
            window_sec: Optional[float] = None,
            letterbox_size: Optional[int | Tuple[int, int] | List[int]] = None,
    ) -> Tuple[np.ndarray, List[np.ndarray], Dict[str, Any]]:
        """
        Đọc frame trực tiếp từ video file và trích xuất chuỗi 68 landmark.
        - time_interval: khoảng thời gian (giây) giữa 2 frame lấy mẫu (0: đọc mọi frame, None: dùng self.time_interval).
        - window_sec: thời lượng cửa sổ (giây) cần lấy. None = toàn bộ video.
        - letterbox_size: (w, h) kích thước chuẩn hóa letterbox (None = dùng self.letterbox_size).
        """
        ti = self.time_interval if time_interval is None else float(time_interval)
        if ti < 0:
            raise ValueError(f"time_interval phải là số không âm (>= 0): {ti}")

        lbs = self.letterbox_size if letterbox_size is None else _parse_letterbox_size(letterbox_size)
        frames, fps, total_frames, duration_sec = read_video_frames(
            video_path, ti, window_sec, letterbox_size=lbs
        )
        lm68 = self.extract(frames)
        meta = {
            "video_path": str(video_path),
            "fps": fps,
            "total_frames": total_frames,
            "sampled_frames": len(frames),
            "duration_sec": duration_sec,
            "time_interval": ti,
            "letterbox_size": lbs,
        }
        return lm68, frames, meta


# =====================================================================
# 3. LỚP GCNDATASET (HỖ TRỢ CACHE .PT & TRÍCH XUẤT TRỰC TIẾP TỪ VIDEO)
# =====================================================================
class GCNDataset(Dataset):
    """
    Dataset đọc chuỗi 68 điểm mốc khuôn mặt cho ST-GCN, hỗ trợ:
      - Nạp trực tiếp từ file cache .pt ({cache_dir}/{split}.pt) vào RAM và cắt cửa sổ thời gian (Random Crop cho Train, Center Crop cho Val/Test).
      - Fallback đọc video trực tiếp từ thư mục dữ liệu khi use_cache=False, return_frames=True, hoặc thiếu mẫu trong cache.
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
            time_interval: float = 0.1,
            window_sec: Optional[float] = 10.0,
            manifest_filename: str = "dataset_merged_split.csv",
            landmark_model_path: Optional[str | Path] = None,
            letterbox_size: int | Tuple[int, int] = (640, 480),
            return_frames: bool = False,
            device: str = "cuda",
            det_size: Tuple[int, int] = (320, 320),
            model_name: str = "buffalo_l",
            use_cache: bool = True,
            cache_dir: Optional[str | Path] = None,
            landmark_batch_size: int = 64,
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
        self.cache_time_interval: Optional[float] = None
        self.window_sec = float(window_sec) if window_sec is not None else None
        self.letterbox_size = _parse_letterbox_size(letterbox_size)
        self.return_frames = return_frames
        self.device = device
        self.det_size = tuple(det_size)
        self.model_name = model_name
        self.use_cache = bool(use_cache)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.cached: Optional[Dict[str, Dict[str, Any]]] = None

        self.extractor = FaceLandmarkExtractor(
            model_path=landmark_model_path,
            time_interval=self.time_interval,
            device=self.device,
            det_size=self.det_size,
            model_name=self.model_name,
            landmark_batch_size=landmark_batch_size,
            letterbox_size=self.letterbox_size,
        )
        self.samples = self._load_samples()
        self._load_cache()

        if len(self.samples) == 0:
            logger.warning(f"Không tìm thấy mẫu nào trong {self.root} cho split={split}")

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
                        if not rel_path:
                            continue
                        norm_key = Path(rel_path.replace("\\", "/")).as_posix()
                        samples.append({
                            "key": norm_key,
                            "path": self.root / norm_key,
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
                            "key": p.relative_to(self.root).as_posix(),
                            "path": p,
                            "label": label,
                            "video_id": p.stem,
                            "fps": 0.0,
                            "total_frames": 0,
                            "duration_sec": 0.0,
                        })
        return samples

    def _load_cache(self) -> None:
        """Nạp chuỗi landmark toàn thời lượng từ {cache_dir}/{split}.pt vào RAM."""
        if not self.use_cache or self.cache_dir is None:
            return

        cache_path = self.cache_dir / f"{self.split}.pt"
        if not cache_path.is_file():
            logger.warning(
                f"[CACHE] Không tìm thấy {cache_path}. Sẽ fallback trích xuất trực tiếp từ video."
            )
            return

        try:
            try:
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            except Exception:
                payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        except TypeError:
            payload = torch.load(cache_path, map_location="cpu")

        if isinstance(payload, dict) and "samples" in payload:
            cache_ti = float(payload.get("time_interval", self.time_interval))
            self.cache_time_interval = cache_ti if cache_ti >= 0 else self.time_interval
            if abs(cache_ti - self.time_interval) > 1e-6:
                logger.warning(
                    f"[CACHE] {cache_path.name} được tạo với time_interval={cache_ti}s (cấu hình hiện tại={self.time_interval}s). "
                    f"Tự động đồng bộ cắt cửa sổ theo time_interval={cache_ti}s của cache."
                )
            self.cached = payload["samples"]
        elif isinstance(payload, dict):
            self.cache_time_interval = self.time_interval
            self.cached = payload
        else:
            return

        hit = sum(1 for s in self.samples if s["key"] in self.cached)
        logger.info(
            f"[CACHE] Đã nạp {cache_path.name}: {len(self.cached)} video trong cache | "
            f"khớp {hit}/{len(self.samples)} mẫu của split '{self.split}'."
        )
        if hit < len(self.samples):
            logger.warning(
                f"[CACHE] Thiếu {len(self.samples) - hit} mẫu trong cache; các mẫu này sẽ được trích xuất trực tiếp từ video."
            )

    def _slice_temporal_window(self, full_lm: torch.Tensor, fps: float, duration_sec: float) -> torch.Tensor:
        """
        Cắt cửa sổ window_sec trên chuỗi landmark toàn thời lượng (2, T_full, 68).
        Train: vị trí bắt đầu ngẫu nhiên mỗi lần gọi (Random Temporal Crop); Val/Test: cắt chính giữa.
        """
        T = full_lm.shape[1]
        if self.window_sec is None or T <= 1:
            return full_lm.clone()

        eff_ti = self.cache_time_interval if self.cache_time_interval is not None else self.time_interval
        win = min(self.window_sec, duration_sec) if duration_sec > 0 else self.window_sec
        if eff_ti > 0:
            win_frames = len(np.arange(0, win, eff_ti))
        else:
            win_frames = int(round(win * (fps if fps > 0 else 30.0)))
        win_frames = max(1, min(win_frames, T))

        max_start = T - win_frames
        start = random.randint(0, max_start) if (self.split == "train" and max_start > 0) else max_start // 2
        return full_lm[:, start:start + win_frames, :].clone()

    def _get_frame_indices(self, total_frames: int, fps: float) -> np.ndarray:
        """Tính toán các chỉ số frame cần lấy dựa trên miền thời gian thực và FPS riêng của video."""
        if total_frames <= 0:
            return np.array([0], dtype=int)

        fps = fps if fps > 0 else 30.0
        duration_sec = total_frames / fps

        # TRƯỜNG HỢP 1: time_interval == 0 -> Đọc frame tuần tự thông thường từ video (không bỏ cách frame)
        if self.time_interval == 0:
            if self.window_sec is None:
                return np.arange(total_frames, dtype=int)
            else:
                win_sec = min(self.window_sec, duration_sec)
                num_frames = int(round(win_sec * fps))
                num_frames = max(1, min(num_frames, total_frames))
                max_start = max(0, total_frames - num_frames)
                start_idx = random.randint(0, max_start) if (self.split == "train" and max_start > 0) else max_start // 2
                return np.arange(start_idx, start_idx + num_frames, dtype=int)

        # TRƯỜNG HỢP 2: time_interval > 0 -> Lấy mẫu cách đều theo thời gian thực (time-domain sampling)
        if self.window_sec is None:
            timestamps = np.arange(0, duration_sec, self.time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([0.0])
        else:
            win = min(self.window_sec, duration_sec)
            max_start = max(0.0, duration_sec - win)
            start_sec = random.uniform(0.0, max_start) if (self.split == "train" and max_start > 0) else max_start / 2.0
            timestamps = start_sec + np.arange(0, win, self.time_interval)
            if len(timestamps) == 0:
                timestamps = np.array([start_sec])

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
            if target_idx < curr_frame and frames:
                frames.append(frames[-1].copy())
                continue
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

        return frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        sample = self.samples[index]

        # Đường nhanh: lấy chuỗi landmark từ cache RAM và cắt cửa sổ thời gian
        if self.cached is not None and not self.return_frames and sample["key"] in self.cached:
            item = self.cached[sample["key"]]
            c_fps = float(item.get("fps", sample.get("fps", 30.0)))
            c_dur = float(item.get("duration_sec", sample.get("duration_sec", 0.0)))
            landmarks_tensor = self._slice_temporal_window(item["landmarks"], c_fps, c_dur)
            label_tensor = torch.tensor(sample["label"], dtype=torch.long)
            meta = {
                "video_id": sample["video_id"],
                "path": str(sample["path"]),
                "fps": c_fps,
                "num_sampled_frames": landmarks_tensor.shape[1],
                "duration_sec": c_dur,
                "time_interval": self.cache_time_interval if self.cache_time_interval is not None else self.time_interval,
            }
            return landmarks_tensor, label_tensor, meta

        fps = sample["fps"]
        total_frames = sample["total_frames"]

        if fps <= 0 or total_frames <= 0:
            cap = cv2.VideoCapture(str(sample["path"]))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            sample["fps"] = fps
            sample["total_frames"] = total_frames

        indices = self._get_frame_indices(total_frames, fps)
        frames = self._read_frames(sample["path"], indices)

        target_w, target_h = self.letterbox_size
        if target_w > 0 and target_h > 0:
            frames = [letterbox(f, new_size=(target_w, target_h)) for f in frames]

        lm68 = self.extractor.extract(frames)
        landmarks_tensor = torch.from_numpy(lm68).permute(2, 0, 1).contiguous()

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


# Alias hỗ trợ nếu người dùng gọi FaceLandmarkDataset
FaceLandmarkDataset = GCNDataset
FaceLandmarkExtractorDataset = GCNDataset


# =====================================================================
# 4. HÀM VẼ LANDMARK VÀ HIỂN THỊ VIDEO
# =====================================================================
def get_landmark_region_color(idx: int) -> Tuple[int, int, int]:
    """Trả về màu sắc đại diện (BGR) theo từng vùng giải phẫu của hệ 68 điểm mốc."""
    if 0 <= idx <= 16:
        return (200, 200, 200)  # Viền hàm
    elif 17 <= idx <= 26:
        return (255, 190, 0)    # Lông mày trái & phải
    elif 27 <= idx <= 35:
        return (0, 255, 255)    # Sống mũi & cánh mũi
    elif 36 <= idx <= 47:
        return (0, 255, 0)      # Mắt trái & phải
    elif 48 <= idx <= 59:
        return (0, 0, 255)      # Môi ngoài
    elif 60 <= idx <= 67:
        return (0, 140, 255)    # Môi trong
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
    """
    vis = frame.copy()
    h, w = vis.shape[:2]

    num_pts = min(68, len(landmarks_xy))
    pts = []
    for i in range(num_pts):
        x = int(landmarks_xy[i, 0] * w)
        y = int(landmarks_xy[i, 1] * h)
        pts.append((x, y))

    if draw_contours and len(pts) >= 68:
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

    for i, (x, y) in enumerate(pts):
        c = point_color if point_color is not None else get_landmark_region_color(i)
        cv2.circle(vis, (x, y), point_radius, c, -1)

    if show_ids:
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
                tx = max(0, min(w - 18, x + 2))
                ty = max(10, min(h - 2, y - 2))
                text = str(i)

                if id_color is None or id_color == "region":
                    text_color = get_landmark_region_color(i)
                elif isinstance(id_color, (tuple, list)):
                    text_color = tuple(id_color)
                else:
                    text_color = (0, 255, 255)

                cv2.putText(vis, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2, cv2.LINE_AA)
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
    """Hiển thị 1 frame ảnh kèm 68 điểm mốc và ID của chúng theo hệ 68 điểm của dự án."""
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
    """Hiển thị video tuần tự từng frame kèm các điểm mốc landmark đã trích xuất."""
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
    print(">>> Phím nóng: [i] Bật/tắt ID (0-67) | [Space] Tạm dừng/Soi chi tiết | [a/d] Frame lùi/tiến | [q/ESC] Thoát <<<")

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

        cv2.putText(vis, f"Video: {video_id} | Status: {label_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, f"Video: {video_id} | Status: {label_text}", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, label_color, 2, cv2.LINE_AA)

        curr_time_sec = (t / fps) if time_interval == 0 else (t * time_interval)
        time_info = f"Frame: {t + 1}/{T} | Time: {curr_time_sec:.1f}s | FPS: {fps:.1f}"
        if paused:
            time_info += "  [PAUSED - Soi ID]"
        cv2.putText(vis, time_info, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, time_info, (15, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        id_str = "ON" if show_ids_state else "OFF"
        guide = f"[i]: IDs ({id_str}) | [Space]: {'Resume' if paused else 'Pause'} | [q/ESC]: Exit"
        cv2.putText(vis, guide, (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, guide, (15, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow(window_name, vis)

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
        elif key == 32:
            paused = not paused
            print(f">>> {'TẠM DỪNG' if paused else 'TIẾP TỤC'}")
            continue
        elif paused and key in (ord('d'), ord('D'), 83):
            t = min(T - 1, t + 1)
            continue
        elif paused and key in (ord('a'), ord('A'), 81):
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
        time_interval: float = 0.1,
        window_sec: Optional[float] = 10.0,
        manifest_filename: str = "dataset_merged_split.csv",
        pin_memory: bool = True,
        shuffle_train: bool = True,
        landmark_model_path: Optional[str | Path] = None,
        letterbox_size: int | Tuple[int, int] = (640, 480),
        device: str = "cuda",
        det_size: Tuple[int, int] = (320, 320),
        model_name: str = "buffalo_l",
        use_cache: bool = True,
        cache_dir: Optional[str | Path] = None,
        landmark_batch_size: int = 64,
):
    """
    Xây dựng nhanh bộ DataLoader cho cả 3 tập train, val, test.
    Ưu tiên nạp landmark từ cache .pt ({cache_dir}/{split}.pt) nếu use_cache=True.
    """
    common = dict(
        dataset_dir=dataset_dir,
        time_interval=time_interval,
        window_sec=window_sec,
        manifest_filename=manifest_filename,
        landmark_model_path=landmark_model_path,
        letterbox_size=letterbox_size,
        device=device,
        det_size=det_size,
        model_name=model_name,
        use_cache=use_cache,
        cache_dir=cache_dir,
        landmark_batch_size=landmark_batch_size,
    )
    train_ds = GCNDataset(split="train", **common)
    val_ds = GCNDataset(split="val", **common)
    test_ds = GCNDataset(split="test", **common)

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
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    DATASET_DIR = r"E:\LSTM\data_processed"
    print("=" * 60)
    print("LOAD 1 VIDEO TU GCNDataset VA HIEN THI LANDMARK")
    print("=" * 60)

    if Path(DATASET_DIR).is_dir():
        ds = GCNDataset(
            dataset_dir=DATASET_DIR,
            split="train",
            time_interval=0,
            letterbox_size=(640, 480),
            window_sec=None,
            return_frames=True,
            use_cache=False,
        )
        print(f"Tổng số mẫu trong tập train: {len(ds)}")
        if len(ds) > 0:
            idx = min(1000, len(ds) - 1)
            landmarks, label, meta, frames = ds[idx]
            print(f"Video ID      : {meta['video_id']}")
            print(f"Đường dẫn file: {meta['path']}")
            print(f"FPS nguồn     : {meta['fps']}")
            print(f"Thời lượng    : {meta['duration_sec']:.1f}s")
            print(f"Số frame lấy  : {len(frames)} frames")
            print(f"Landmarks     : shape {landmarks.shape} (C=2, T={landmarks.shape[1]}, V=68)")
            print(f"Nhãn (Label)  : {label.item()} ({'Alert' if label.item() == 0 else 'Drowsy'})")
            visualize_video_landmarks(landmarks, frames, meta, label, time_interval=ds.time_interval, show_ids=True)
    else:
        print(f"[INFO] Thư mục dataset '{DATASET_DIR}' không tồn tại trên máy hiện tại. Bỏ qua hiển thị video.")
