"""
test_model.py - Ứng Dụng Chạy Suy Luận Thời Gian Thực với ModelInferenceCache từ Camera OpenCV
Hệ thống phát hiện tài xế ngủ gật Driver Guardian AI.

Tính năng nổi bật:
  1. Đọc luồng video camera liên tục qua OpenCV (30 FPS), triệt tiêu hoàn toàn hiện tượng đọng trễ khung hình.
  2. Lấy mẫu định kỳ chính xác mỗi 0.2s (5 Hz) bằng high-resolution timer và nạp vào hàng đợi FIFO không chặn.
  3. Nếu mô hình suy luận xử lý không kịp (> 0.2s), luồng lấy mẫu vẫn tiếp tục đẩy frame vào hàng chờ.
  4. Worker suy luận tự động gom cụm toàn bộ k frame mới tích lũy trong hàng chờ và nạp vào ModelInferenceCache:
     - Chỉ trích xuất CNN cho đúng k frame mới (tiết kiệm đến 98% chi phí CNN).
     - Tái sử dụng (50 - k) đặc trưng cũ từ bộ đệm Tensor Ring Buffer.
     - Lập tức bắt kịp thời gian thực ngay chu kỳ kế tiếp.
  5. Giao diện đồ họa OpenCV HUD chuyên nghiệp: hiển thị ngưỡng (Threshold), xác suất ngủ gật [0.0 - 1.0],
     thanh đo trực quan, nhãn trạng thái tài xế (Alert / Drowsy / Warming Up) và bảng chỉ số Telemetry (FPS, Latency, Queue).
  6. Tương tác phím nóng: [Q] Thoát, [+/-] Thay đổi ngưỡng, [R] Reset bộ đệm cache, [SPACE] Tạm dừng/Tiếp tục.
"""

import argparse
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Đảm bảo đường dẫn tìm kiếm module chuẩn xác trong dự án Driver Guardian
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

import cv2
import numpy as np
import torch

try:
    from model_inference_cache import ModelInferenceCache
except ImportError:
    from ai.LSTM.model_inference_cache import ModelInferenceCache


# ==============================================================================
# 1. TRẠNG THÁI CHIA SẺ GIỮA CÁC LUỒNG (THREAD-SAFE SHARED STATE)
# ==============================================================================
@dataclass
class SharedInferenceState:
    """Quản lý trạng thái suy luận chia sẻ giữa luồng Worker và luồng Giao diện chính."""
    score: Optional[float] = None
    is_drowsy: bool = False
    threshold: float = 0.50
    k_new: int = 0
    cache_size: int = 0
    cache_max: int = 50
    is_ready: bool = False
    infer_latency_ms: float = 0.0
    infer_fps: float = 0.0
    queue_backlog: int = 0
    total_inferences: int = 0
    status_text: str = "INITIALIZING"
    lock: threading.Lock = threading.Lock()

    def update(
            self,
            score: Optional[float],
            k_new: int,
            cache_size: int,
            cache_max: int,
            is_ready: bool,
            latency_ms: float,
            queue_backlog: int
    ) -> None:
        """Cập nhật an toàn kết quả từ Worker thread."""
        with self.lock:
            self.score = score
            self.k_new = k_new
            self.cache_size = cache_size
            self.cache_max = cache_max
            self.is_ready = is_ready
            self.infer_latency_ms = latency_ms
            self.infer_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
            self.queue_backlog = queue_backlog
            self.total_inferences += 1

            if score is None:
                self.is_drowsy = False
                self.status_text = f"WARMING UP ({cache_size}/{cache_max})"
            else:
                self.is_drowsy = (score >= self.threshold)
                if self.is_drowsy:
                    self.status_text = "DROWSY - CẢNH BÁO!"
                else:
                    self.status_text = "ALERT - TỈNH TÁO"

    def adjust_threshold(self, delta: float) -> float:
        """Tăng hoặc giảm ngưỡng quyết định trong khoảng [0.05, 0.95]."""
        with self.lock:
            self.threshold = round(max(0.05, min(0.95, self.threshold + delta)), 2)
            if self.score is not None:
                self.is_drowsy = (self.score >= self.threshold)
                self.status_text = "DROWSY - CẢNH BÁO!" if self.is_drowsy else "ALERT - TỈNH TÁO"
            return self.threshold

    def get_snapshot(self) -> Dict[str, Any]:
        """Lấy bản sao dữ liệu an toàn để luồng hiển thị vẽ lên màn hình."""
        with self.lock:
            return {
                "score": self.score,
                "is_drowsy": self.is_drowsy,
                "threshold": self.threshold,
                "k_new": self.k_new,
                "cache_size": self.cache_size,
                "cache_max": self.cache_max,
                "is_ready": self.is_ready,
                "infer_latency_ms": self.infer_latency_ms,
                "infer_fps": self.infer_fps,
                "queue_backlog": self.queue_backlog,
                "total_inferences": self.total_inferences,
                "status_text": self.status_text,
            }


# ==============================================================================
# 2. LUỒNG ĐỌC CAMERA LIÊN TỤC (CAMERA CAPTURE WORKER)
# ==============================================================================
class CameraReader(threading.Thread):
    """
    Đọc camera liên tục ở tốc độ gốc (thông thường 30 FPS) trong một thread riêng biệt.
    Mục đích: Giải phóng triệt để hàng đợi phần cứng của OpenCV, đảm bảo frame hiển thị
    và frame lấy mẫu luôn là khung hình mới nhất, không bao giờ bị đọng trễ.
    """

    def __init__(
            self,
            source: Union[int, str] = 0,
            width: int = 1280,
            height: int = 720,
            stop_event: Optional[threading.Event] = None
    ):
        super().__init__(name="CameraReaderThread", daemon=True)
        self.source = source
        self.width = width
        self.height = height
        self.stop_event = stop_event or threading.Event()

        self.cap: Optional[cv2.VideoCapture] = None
        self.latest_frame: Optional[np.ndarray] = None
        self.frame_id: int = 0
        self.fps_cam: float = 0.0
        self.is_opened: bool = False
        self.lock = threading.Lock()
        self.is_paused: bool = False

        self._init_camera()

    def _init_camera(self) -> None:
        """Khởi tạo camera và tối ưu hóa backend trên Windows."""
        is_device_index = False
        dev_id = 0
        if isinstance(self.source, int):
            is_device_index = True
            dev_id = self.source
        elif isinstance(self.source, str) and self.source.isdigit():
            is_device_index = True
            dev_id = int(self.source)

        if is_device_index:
            # Trên Windows, thử DSHOW trước để khởi động camera nhanh hơn
            if sys.platform.startswith("win"):
                self.cap = cv2.VideoCapture(dev_id, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(dev_id)
            else:
                self.cap = cv2.VideoCapture(dev_id)

            if self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        else:
            # Đường dẫn tệp video hoặc luồng RTSP
            self.cap = cv2.VideoCapture(str(self.source))

        self.is_opened = bool(self.cap and self.cap.isOpened())

    def run(self) -> None:
        """Vòng lặp đọc camera liên tục."""
        if not self.is_opened or self.cap is None:
            return

        prev_time = time.perf_counter()
        fps_smoothing = 0.9

        while not self.stop_event.is_set():
            if self.is_paused:
                time.sleep(0.01)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Nếu là file video thì tự động tua lại đầu file để kiểm thử liên tục
                if not (isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit())):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.01)
                    continue
                else:
                    time.sleep(0.005)
                    continue

            now = time.perf_counter()
            dt = now - prev_time
            prev_time = now
            if dt > 0:
                inst_fps = 1.0 / dt
                self.fps_cam = self.fps_cam * fps_smoothing + inst_fps * (1.0 - fps_smoothing)

            with self.lock:
                self.latest_frame = frame
                self.frame_id += 1

        if self.cap is not None:
            self.cap.release()

    def get_latest_frame(self) -> Tuple[int, Optional[np.ndarray], float]:
        """Lấy khung hình mới nhất kèm mã frame và FPS đo được."""
        with self.lock:
            frame_copy = self.latest_frame.copy() if self.latest_frame is not None else None
            return self.frame_id, frame_copy, self.fps_cam

    def toggle_pause(self) -> bool:
        """Chuyển đổi trạng thái tạm dừng stream."""
        self.is_paused = not self.is_paused
        return self.is_paused

    def stop(self) -> None:
        """Dừng thread và giải phóng camera."""
        self.stop_event.set()


# ==============================================================================
# 3. LUỒNG LẤY MẪU ĐỊNH KỲ 0.2S (FRAME SAMPLER WORKER)
# ==============================================================================
class FrameSampler(threading.Thread):
    """
    Lấy mẫu định kỳ chính xác mỗi 0.2s (tần số 5 Hz) từ CameraReader.
    Đẩy khung hình vào hàng đợi FIFO không chặn (non-blocking queue).
    Nếu mô hình suy luận xử lý không kịp, luồng này VẪN TIẾP TỤC lấy mẫu và tích lũy vào hàng chờ.
    """

    def __init__(
            self,
            camera_reader: CameraReader,
            sample_queue: queue.Queue,
            sample_interval: float = 0.20,
            stop_event: Optional[threading.Event] = None
    ):
        super().__init__(name="FrameSamplerThread", daemon=True)
        self.camera_reader = camera_reader
        self.sample_queue = sample_queue
        self.sample_interval = max(0.01, float(sample_interval))
        self.stop_event = stop_event or threading.Event()
        self.sample_count: int = 0
        self.actual_sample_rate: float = 0.0

    def run(self) -> None:
        """Vòng lặp đo thời gian chính xác cao và lấy mẫu định kỳ."""
        last_sample_time = time.perf_counter()
        last_frame_id = -1
        prev_sample_ts = last_sample_time

        while not self.stop_event.is_set():
            current_time = time.perf_counter()
            elapsed = current_time - last_sample_time

            if elapsed >= self.sample_interval:
                last_sample_time = current_time

                frame_id, frame, _ = self.camera_reader.get_latest_frame()
                if frame is not None and frame_id != last_frame_id:
                    last_frame_id = frame_id

                    # Tính tốc độ lấy mẫu thực tế
                    dt = current_time - prev_sample_ts
                    prev_sample_ts = current_time
                    if dt > 0:
                        self.actual_sample_rate = 1.0 / dt

                    # Đẩy vào hàng đợi kèm timestamp (NON-BLOCKING)
                    self.sample_queue.put((current_time, frame))
                    self.sample_count += 1

            # Sleep siêu ngắn để nhường CPU mà không làm lệch nhịp timer
            sleep_time = max(0.001, (self.sample_interval - (time.perf_counter() - last_sample_time)) * 0.5)
            time.sleep(min(0.005, sleep_time))


# ==============================================================================
# 4. LUỒNG SUY LUẬN MÔ HÌNH THÍCH ỨNG (INFERENCE WORKER)
# ==============================================================================
class InferenceWorker(threading.Thread):
    """
    Luồng tiêu thụ hàng đợi và chạy suy luận ModelInferenceCache:
    - Khi sẵn sàng, rút TOÀN BỘ k frames mới đang tích lũy trong hàng chờ (k >= 1).
    - Gom cụm thành mảng numpy và gọi ModelInferenceCache.predict_step().
    - Mô hình chỉ tính toán CNN cho đúng k frame mới, tái sử dụng (50 - k) đặc trưng cũ.
    - Cập nhật kết quả vào SharedInferenceState.
    """

    def __init__(
            self,
            model: ModelInferenceCache,
            sample_queue: queue.Queue,
            shared_state: SharedInferenceState,
            stop_event: Optional[threading.Event] = None
    ):
        super().__init__(name="InferenceWorkerThread", daemon=True)
        self.model = model
        self.sample_queue = sample_queue
        self.shared_state = shared_state
        self.stop_event = stop_event or threading.Event()

    def run(self) -> None:
        """Vòng lặp tiêu thụ hàng đợi suy luận."""
        while not self.stop_event.is_set():
            try:
                # 1. Chờ lấy ít nhất 1 frame đầu tiên từ hàng đợi (timeout 0.1s)
                item = self.sample_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # 2. DRAIN TOÀN BỘ các frame mới còn đang dồn ứ trong hàng đợi (k >= 1)
            batch_frames = [item[1]]
            while not self.sample_queue.empty():
                try:
                    extra_item = self.sample_queue.get_nowait()
                    batch_frames.append(extra_item[1])
                except queue.Empty:
                    break

            k = len(batch_frames)
            queue_remaining = self.sample_queue.qsize()

            # 3. Chuẩn hóa màu từ BGR sang RGB cho mô hình deep learning
            batch_rgb_list = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch_frames]
            if k == 1:
                input_tensor_or_np = batch_rgb_list[0]  # [H, W, 3]
            else:
                input_tensor_or_np = np.stack(batch_rgb_list, axis=0)  # [K, H, W, 3]

            # 4. Đo đạc thời gian suy luận chính xác
            t0 = time.perf_counter()
            try:
                score, details = self.model.predict_step(input_tensor_or_np, return_details=True)
            except Exception as e:
                print(f"[InferenceWorker] Lỗi suy luận: {e}", file=sys.stderr)
                score, details = None, {"cache_size": 0, "is_ready": False}
            latency_ms = (time.perf_counter() - t0) * 1000.0

            # 5. Cập nhật kết quả an toàn sang SharedInferenceState
            self.shared_state.update(
                score=score,
                k_new=k,
                cache_size=details.get("cache_size", 0),
                cache_max=self.model.window_size,
                is_ready=details.get("is_ready", False),
                latency_ms=latency_ms,
                queue_backlog=queue_remaining
            )


# ==============================================================================
# 5. BỘ VẼ GIAO DIỆN ĐỒ HỌA OPENCV HUD (OPENCV HUD VISUALIZER)
# ==============================================================================
class HUDVisualizer:
    """
    Bộ kết xuất giao diện trực quan Heads-Up Display (HUD) chuyên nghiệp bằng OpenCV:
    - Bảng thông tin trạng thái tài xế (ALERT / DROWSY / WARMING UP).
    - Thanh đo xác suất buồn ngủ kèm vạch đánh dấu ngưỡng Threshold.
    - Bảng chỉ số kỹ thuật chi tiết (Camera FPS, Infer Latency, Cache Size, Queue Backlog k).
    - Hiệu ứng nhấp nháy viền đỏ cảnh báo khi phát hiện tài xế ngủ gật.
    """

    COLOR_BG_DARK = (20, 20, 20)
    COLOR_ALERT = (30, 215, 96)  # Xanh lá neon (Tỉnh táo)
    COLOR_DROWSY = (36, 36, 235)  # Đỏ tươi cảnh báo (Buồn ngủ)
    COLOR_WARNING = (0, 190, 255)  # Vàng cam (Đang warmup hoặc cảnh báo nhẹ)
    COLOR_WHITE = (245, 245, 245)
    COLOR_GRAY = (140, 140, 140)
    COLOR_DARK_BAR = (50, 50, 50)

    @classmethod
    def render(
            cls,
            frame: np.ndarray,
            state: Dict[str, Any],
            cam_fps: float,
            actual_sample_rate: float
    ) -> np.ndarray:
        """Vẽ toàn bộ các thành phần HUD lên khung hình hiển thị."""
        h, w = frame.shape[:2]
        canvas = frame.copy()

        score = state["score"]
        is_drowsy = state["is_drowsy"]
        threshold = state["threshold"]
        status_text = state["status_text"]
        k_new = state["k_new"]
        cache_size = state["cache_size"]
        cache_max = state["cache_max"]
        latency_ms = state["infer_latency_ms"]
        queue_backlog = state["queue_backlog"]

        # 1. Hiệu ứng viền nhấp nháy đỏ khi BUỒN NGỦ (Drowsy Alert)
        if is_drowsy and (int(time.time() * 4) % 2 == 0):
            border_thick = 8
            cv2.rectangle(canvas, (0, 0), (w, h), cls.COLOR_DROWSY, border_thick)

        # 2. Thanh tiêu đề phía trên (Top Header Bar)
        header_h = 50
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (w, header_h), cls.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

        cv2.putText(
            canvas,
            "DRIVER GUARDIAN AI | ModelInferenceCache (Sliding Window W=50, Stride=0.2s)",
            (20, 32),
            cv2.FONT_HERSHEY_DUPLEX,
            0.65,
            cls.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )

        # 3. Thẻ trạng thái tài xế (Driver Status Card) góc trên bên trái
        card_x, card_y = 20, 65
        card_w, card_h = 360, 105
        overlay = canvas.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), cls.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

        # Màu sắc chủ đạo thẻ theo trạng thái
        if score is None:
            theme_color = cls.COLOR_WARNING
        elif is_drowsy:
            theme_color = cls.COLOR_DROWSY
        else:
            theme_color = cls.COLOR_ALERT

        # Viền thẻ trạng thái
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), theme_color, 2)

        cv2.putText(
            canvas,
            "TRANG THAI TAI XE:",
            (card_x + 15, card_y + 28),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            cls.COLOR_GRAY,
            1,
            cv2.LINE_AA
        )
        cv2.putText(
            canvas,
            status_text,
            (card_x + 15, card_y + 65),
            cv2.FONT_HERSHEY_DUPLEX,
            0.80,
            theme_color,
            2,
            cv2.LINE_AA
        )

        sub_info = f"Score: {score:.3f}" if score is not None else "Buffer dang nap..."
        cv2.putText(
            canvas,
            f"{sub_info} | Nguong: {threshold:.2f}",
            (card_x + 15, card_y + 92),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cls.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )

        # 4. Thanh đo xác suất ngủ gật (Drowsiness Probability Bar)
        bar_x, bar_y = 20, 185
        bar_w, bar_h = 360, 26
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), cls.COLOR_DARK_BAR, -1)

        val = score if score is not None else 0.0
        val = max(0.0, min(1.0, val))
        fill_w = int(bar_w * val)

        # Màu sắc động của thanh tiến trình
        if val < 0.35:
            bar_color = cls.COLOR_ALERT
        elif val < threshold:
            bar_color = cls.COLOR_WARNING
        else:
            bar_color = cls.COLOR_DROWSY

        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), bar_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), cls.COLOR_WHITE, 1)

        # Vạch đánh dấu ngưỡng Threshold trên thanh đo
        thresh_x = int(bar_x + bar_w * threshold)
        cv2.line(canvas, (thresh_x, bar_y - 4), (thresh_x, bar_y + bar_h + 4), cls.COLOR_WHITE, 2)
        cv2.putText(
            canvas,
            f"T:{threshold:.2f}",
            (thresh_x - 18, bar_y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            cls.COLOR_WHITE,
            1,
            cv2.LINE_AA
        )

        # Nhãn phần trăm xác suất bên trong thanh
        percent_str = f"{val * 100:.1f}%"
        cv2.putText(
            canvas,
            percent_str,
            (bar_x + bar_w + 10, bar_y + 18),
            cv2.FONT_HERSHEY_DUPLEX,
            0.55,
            bar_color,
            1,
            cv2.LINE_AA
        )

        # 5. Bảng chỉ số kỹ thuật Telemetry góc trên bên phải
        panel_w, panel_h = 340, 175
        panel_x = w - panel_w - 20
        panel_y = 65

        overlay = canvas.copy()
        cv2.rectangle(overlay, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), cls.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)
        cv2.rectangle(canvas, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h), cls.COLOR_GRAY, 1)

        cv2.putText(
            canvas,
            "THONG SO HE THONG (TELEMETRY)",
            (panel_x + 15, panel_y + 24),
            cv2.FONT_HERSHEY_DUPLEX,
            0.50,
            cls.COLOR_WARNING,
            1,
            cv2.LINE_AA
        )

        lines = [
            f"Camera FPS: {cam_fps:.1f} fps",
            f"Toc do lay mau: 0.2s ({actual_sample_rate:.1f} Hz)",
            f"Do tre suy luan: {latency_ms:.1f} ms",
            f"Bo dem Cache: {cache_size}/{cache_max} frames " + ("[FULL]" if cache_size >= cache_max else "[FILLING]"),
            f"Hang doi ton (k moi): k = {k_new} frame(s)",
            f"Hang doi cho (queue): {queue_backlog} frame(s)",
        ]

        for i, line in enumerate(lines):
            line_color = cls.COLOR_WHITE
            if "k =" in line and k_new > 1:
                line_color = cls.COLOR_WARNING
            if "queue:" in line and queue_backlog > 2:
                line_color = cls.COLOR_DROWSY

            cv2.putText(
                canvas,
                line,
                (panel_x + 15, panel_y + 50 + i * 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                line_color,
                1,
                cv2.LINE_AA
            )

        # 6. Dòng hướng dẫn phím tắt ở đáy màn hình
        footer_h = 36
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, h - footer_h), (w, h), cls.COLOR_BG_DARK, -1)
        cv2.addWeighted(overlay, 0.85, canvas, 0.15, 0, canvas)

        help_text = "[Q/ESC] Thoat  |  [+/-] Chinh nguong (+/-0.05)  |  [R] Reset Cache  |  [SPACE] Tam dung"
        cv2.putText(
            canvas,
            help_text,
            (20, h - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            cls.COLOR_GRAY,
            1,
            cv2.LINE_AA
        )

        return canvas


# ==============================================================================
# 6. HÀM CHÍNH VÀ ĐIỀU PHỐI VÒNG LỜI (MAIN CONTROLLER)
# ==============================================================================
def parse_args() -> argparse.Namespace:
    """Phân tích các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(
        description="Driver Guardian AI - Chạy suy luận real-time với ModelInferenceCache từ camera OpenCV"
    )
    parser.add_argument(
        "--source", "-s",
        type=str,
        default=r"0",
        help="Nguồn video: camera index (0, 1, ...) hoặc đường dẫn file video (.mp4)"
    )
    parser.add_argument(
        "--cnn-path",
        type=str,
        default=r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_cnn\best.pt",
        help="Đường dẫn checkpoint NMSFreeDetector CNN"
    )
    parser.add_argument(
        "--convgru-path",
        type=str,
        default=r"D:\Project\DATN\driver-guardian\ai\checkpoints\model_convgru\best1.pt",
        help="Đường dẫn checkpoint ConvGRUClassifier"
    )
    parser.add_argument(
        "--window-size", "-w",
        type=int,
        default=50,
        help="Kích thước cửa sổ trượt (mặc định 50 frames)"
    )
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=0.20,
        help="Tốc độ lấy mẫu định kỳ vào hàng chờ (giây, mặc định 0.2s)"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.50,
        help="Ngưỡng kích hoạt cảnh báo ngủ gật (mặc định 0.50)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Thiết bị tính toán: 'cuda' hoặc 'cpu' (mặc định tự động phát hiện)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=32,
        help="Chunk size tối đa khi trích xuất CNN để tối ưu VRAM"
    )
    parser.add_argument(
        "--warmup-mode",
        type=str,
        default="dynamic",
        choices=["dynamic", "wait"],
        help="Chế độ khởi động: 'dynamic' (dự đoán ngay) hoặc 'wait' (đợi đủ 50 frame)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Độ phân giải chiều rộng camera mong muốn"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Độ phân giải chiều cao camera mong muốn"
    )
    return parser.parse_args()


def resolve_checkpoint_paths(args: argparse.Namespace) -> Tuple[str, str]:
    """Tự động kiểm tra và giải quyết đường dẫn checkpoint hợp lệ."""
    cnn_path = Path(args.cnn_path)
    conv_gru_path = Path(args.convgru_path)

    # Dự phòng tìm kiếm đường dẫn nếu không tồn tại ở đường dẫn chỉ định
    if not cnn_path.exists():
        fallback_cnn = PROJECT_ROOT / "checkpoints" / "model_cnn" / "best.pt"
        if fallback_cnn.exists():
            cnn_path = fallback_cnn

    if not conv_gru_path.exists():
        fallback_conv = PROJECT_ROOT / "checkpoints" / "model_convgru" / "best.pt"
        if fallback_conv.exists():
            conv_gru_path = fallback_conv
        else:
            fallback_1m9 = CURRENT_DIR / "checkpoints" / "1m9" / "best.pt"
            if fallback_1m9.exists():
                conv_gru_path = fallback_1m9

    return str(cnn_path), str(conv_gru_path)


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print(" DRIVER GUARDIAN AI - TEST MODEL INFERENCE CACHE (OPENCV CAMERA)")
    print("=" * 80)

    # 1. Thiết bị tính toán
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "CPU"
    print(f"[*] Thiet bi tinh toan: {device} ({device_name})")

    # 2. Kiểm tra đường dẫn Checkpoint
    cnn_path, conv_gru_path = resolve_checkpoint_paths(args)
    print(f"[*] Checkpoint CNN    : {cnn_path}")
    print(f"[*] Checkpoint ConvGRU: {conv_gru_path}")

    if not Path(cnn_path).exists() or not Path(conv_gru_path).exists():
        print(f"[!] Canh bao: Khong tim thay tep checkpoint tai {cnn_path} hoac {conv_gru_path}!", file=sys.stderr)

    # 3. Nạp ModelInferenceCache
    print(f"[*] Dang nap ModelInferenceCache (Window = {args.window_size}, Chunk = {args.chunk_size})...")
    model = ModelInferenceCache.from_checkpoint(
        cnn_path=cnn_path,
        conv_gru_path=conv_gru_path,
        device=device,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        enable_cache=True,
        warmup_mode=args.warmup_mode
    )
    print(" -> ModelInferenceCache da khoi tao thanh cong!")

    # Warm-up CUDA kernels để loại bỏ độ trễ JIT ở bước đầu tiên
    print("[*] Dang khoi dong GPU/CPU kernels (Warm-up)...")
    try:
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model.predict_step(dummy_frame)
        model.reset_cache()
        print(" -> Khoi dong kernels thanh cong! Do tre suy luan dat muc toi uu (~25ms).")
    except Exception as e:
        print(f"[!] Warm-up canh bao: {e}")

    # 4. Kiểm tra nguồn camera / video
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)

    stop_event = threading.Event()
    camera_reader = CameraReader(
        source=source_val,
        width=args.width,
        height=args.height,
        stop_event=stop_event
    )

    if not camera_reader.is_opened:
        # Nếu mở camera 0 thất bại, kiểm tra video kiểm thử dự phòng trong docs/video
        fallback_video = CURRENT_DIR / "docs" / "video" / "video.mp4"
        if fallback_video.exists():
            print(f"[!] Khong the mo camera {args.source}. Tu dong chuyen sang video kiem thu: {fallback_video}")
            camera_reader = CameraReader(
                source=str(fallback_video),
                width=args.width,
                height=args.height,
                stop_event=stop_event
            )
        else:
            print(f"[X] Lỗi: Không thể kết nối với nguồn video {args.source} và không tìm thấy video fallback!",
                  file=sys.stderr)
            return

    camera_reader.start()
    print(f"[*] Luong CameraReader da khoi chay (nguon: {args.source})")

    # 5. Khởi tạo hàng đợi và các luồng làm việc
    sample_queue: queue.Queue = queue.Queue()
    shared_state = SharedInferenceState(threshold=args.threshold, cache_max=args.window_size)

    # Luồng lấy mẫu 0.2s
    frame_sampler = FrameSampler(
        camera_reader=camera_reader,
        sample_queue=sample_queue,
        sample_interval=args.sample_interval,
        stop_event=stop_event
    )
    frame_sampler.start()
    print(f"[*] Luong FrameSampler da khoi chay (chu ky lay mau: {args.sample_interval}s)")

    # Luồng suy luận Worker
    inference_worker = InferenceWorker(
        model=model,
        sample_queue=sample_queue,
        shared_state=shared_state,
        stop_event=stop_event
    )
    inference_worker.start()
    print("[*] Luong InferenceWorker da khoi chay (ho tro tieu thu k frames dồn)")

    # 6. Vòng lặp hiển thị giao diện OpenCV (Luồng chính - Main Thread)
    window_name = "Driver Guardian AI - ModelInferenceCache Real-time Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, args.width, args.height)

    print("\n" + "=" * 80)
    print(" GIAO DIEN OPENCV DANG HOAT DONG - PHIM NONG:")
    print("   [Q] hoặc [ESC] : Thoát chương trình")
    print("   [+] hoặc [=]   : Tăng ngưỡng cảnh báo (+0.05)")
    print("   [-] hoặc [_]   : Giảm ngưỡng cảnh báo (-0.05)")
    print("   [R]            : Reset toàn bộ bộ đệm cache")
    print("   [SPACE]        : Tạm dừng / Tiếp tục luồng camera")
    print("=" * 80 + "\n")

    try:
        while not stop_event.is_set():
            _, frame, cam_fps = camera_reader.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            # Lấy bản sao snapshot kết quả suy luận
            state_snapshot = shared_state.get_snapshot()

            # Vẽ HUD giao diện chuyên nghiệp
            rendered_frame = HUDVisualizer.render(
                frame=frame,
                state=state_snapshot,
                cam_fps=cam_fps,
                actual_sample_rate=frame_sampler.actual_sample_rate
            )

            cv2.imshow(window_name, rendered_frame)

            # Xử lý phím bấm (delay 1ms)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord('Q'), 27):  # 27 là ESC
                print("[*] Nguoi dung yeu cau thoat chuong trinh...")
                break
            elif key in (ord('+'), ord('=')):
                new_thresh = shared_state.adjust_threshold(0.05)
                print(f" -> Da tang nguong canh bao len: {new_thresh:.2f}")
            elif key in (ord('-'), ord('_')):
                new_thresh = shared_state.adjust_threshold(-0.05)
                print(f" -> Da giam nguong canh bao xuong: {new_thresh:.2f}")
            elif key in (ord('r'), ord('R')):
                model.reset_cache()
                with shared_state.lock:
                    shared_state.score = None
                    shared_state.cache_size = 0
                    shared_state.is_ready = False
                    shared_state.status_text = "CACHE RESET - DANG NAP LAI"
                print(" -> Da reset toan bo bo dem cache cua ModelInferenceCache!")
            elif key == ord(' '):  # Phím Space
                is_paused = camera_reader.toggle_pause()
                print(f" -> Luong camera: {'TAM DUNG' if is_paused else 'TIEP TUC'}")

    except KeyboardInterrupt:
        print("\n[*] Nhan tin hieu ngat tu ban phim (Ctrl+C)...")
    finally:
        # Giải phóng tài nguyên mượt mà
        print("[*] Dang dung cac luong lam viec va giai phong tai nguyen...")
        stop_event.set()

        camera_reader.stop()
        camera_reader.join(timeout=1.0)
        frame_sampler.join(timeout=1.0)
        inference_worker.join(timeout=1.0)

        cv2.destroyAllWindows()
        print("[*] Da giai phong hoan tat. Chuong trinh ket thuc an toan!")


if __name__ == "__main__":
    main()
