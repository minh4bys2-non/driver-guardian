import cv2
import numpy as np

# [Frame: RGB, shape: [3, H, W], dtype: unit8, range: [0, 255], timeline = 30FPS]
# (frame: np.ndarray, timestamp: float)
def read_video(path, target_fps=30):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    dt = 1.0 / target_fps
    next_t = 0.0
    prev_frame = None
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = np.transpose(frame, (2, 0, 1))  # [3, H, W]
            if prev_frame is None:
                prev_frame = frame
            while next_t <= t:
                yield prev_frame, next_t
                next_t += dt
            prev_frame = frame
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