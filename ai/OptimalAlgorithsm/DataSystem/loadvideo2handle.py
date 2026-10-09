import cv2
import numpy as np

# [Frame: RGB, shape: [3, H, W], dtype: uint8, range: [0, 255], timeline = 30FPS]
# (frame: np.ndarray, timestamp: float)
def read_video(path, target_fps=30):
    if not np.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("target_fps must be finite and positive")
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
            if not np.isfinite(t) or t < 0 or (previous_t is not None and t <= previous_t):
                raise ValueError(f"Invalid video timestamp: {path}")
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
    path = "/home/tranmanhduy/Workspace/ptithcm/driver-guardian/ai/OptimalAlgorithsm/UTA-RLDD/Fold1_part1/01/10.MOV"
    for frame, timestamp in read_video(path):
        print(frame.shape, timestamp)
        image = np.transpose(frame, (1, 2, 0))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", image)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()