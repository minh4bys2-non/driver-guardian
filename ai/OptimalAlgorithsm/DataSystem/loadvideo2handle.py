import sys
from pathlib import Path

import cv2
import numpy as np


if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from ai.OptimalAlgorithsm.DataSystem.config import DEFAULT_CONFIG, VIDEO_ROOT


def balance_brightness(frame, target_luma=DEFAULT_CONFIG.target_luma):
    """Nâng sáng BGR uint8 trên kênh Y; giữ chroma và giới hạn khuếch đại vùng tối."""
    if not np.isfinite(target_luma) or not 0 <= target_luma <= 255:
        raise ValueError("target_luma must be finite and in [0,255]")
    if target_luma == 0:
        return frame
    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    mean = float(ycrcb[:, :, 0].mean())
    if mean == 0 or mean >= target_luma:
        return frame
    gamma = max(0.5, np.log(target_luma / 255) / np.log(mean / 255))
    lut = np.rint(255 * (np.arange(256) / 255) ** gamma).astype(np.uint8)
    ycrcb[:, :, 0] = cv2.LUT(ycrcb[:, :, 0], lut)
    return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)


# [Frame: RGB, shape: [3, H, W], dtype: uint8, range: [0, 255], timeline = 30FPS]
# (frame: np.ndarray, timestamp: float)
def read_video(path, target_fps=DEFAULT_CONFIG.fps, *, target_luma=DEFAULT_CONFIG.target_luma):
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
    if not np.isfinite(target_luma) or not 0 <= target_luma <= 255:
        raise ValueError("target_luma must be finite and in [0,255]")
    cap = cv2.VideoCapture(path)
    previous = None
    previous_t = origin = None
    index = output_index = 0
    use_frame_index = False
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        source_fps = cap.get(cv2.CAP_PROP_FPS)
        if not np.isfinite(source_fps) or source_fps <= 0:
            raise ValueError(f"Invalid source FPS: {path}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if origin is None:
                origin = t
            # Chỉ dùng FPS thay thế khi backend không có timestamp ngay từ đầu.
            if index == 1:
                use_frame_index = origin == 0 and t == 0
            t = index / source_fps if use_frame_index else t - origin
            if (
                not np.isfinite(t)
                or t < 0
                or (previous_t is not None and t <= previous_t)
            ):
                raise ValueError(f"Invalid video timestamp: {path}")
            frame = balance_brightness(frame, target_luma)
            frame = np.transpose(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (2, 0, 1))
            if previous is not None:
                while output_index / target_fps < t - 1e-9:
                    yield previous, output_index / target_fps
                    output_index += 1
            previous, previous_t = frame, t
            index += 1
        if previous is None:
            raise ValueError(f"Video has no decodable frames: {path}")
        # Khung cuối vẫn chiếm một khoảng thời gian, kể cả khi tăng FPS.
        end = previous_t + 1 / source_fps
        while output_index / target_fps < end - 1e-9:
            yield previous, output_index / target_fps
            output_index += 1
    finally:
        cap.release()


if __name__ == "__main__":
    path = str(VIDEO_ROOT / "Fold1_part1/01/10.MOV")
    for frame, timestamp in read_video(path):
        print(frame.shape, timestamp)
        image = np.transpose(frame, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
