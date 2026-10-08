import time
import cv2
import numpy as np
from ai.PhysicalBranch.camera_metrics import CameraMetrics

class VideoMetrics:
    def __init__(self, fps=30, window_sec=60, **options):
        self.fps = fps
        self.camera = CameraMetrics(fps=fps, window_sec=window_sec, **options)
    def process(self, frames):
        """frames: iterable of (RGB uint8 [3,H,W], timestamp_sec)."""
        self.camera.reset()
        result = None
        try:
            for frame, timestamp in frames:
                if (not isinstance(frame, np.ndarray) or frame.dtype != np.uint8
                        or frame.ndim != 3 or frame.shape[0] != 3
                        or min(frame.shape[1:]) == 0):
                    raise ValueError("Expected RGB uint8 frame with shape [3,H,W]")

                start = time.perf_counter()
                bgr = cv2.cvtColor(np.moveaxis(frame, 0, -1), cv2.COLOR_RGB2BGR)
                result = self.camera.process(bgr, timestamp)
                result["processing_ms"] = (time.perf_counter() - start) * 1000
                result["fps"] = self.fps
        finally:
            self.camera.close()

        if result is None:
            raise ValueError("No frames supplied")
        return result