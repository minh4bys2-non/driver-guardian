"""Đổi đơn vị theo miền bộ đo, không ánh xạ mức nguy hiểm."""

import hashlib
import json
from dataclasses import asdict
from math import ceil, isfinite
from numbers import Real

from ai.PhysicalBranch.camera_metrics import CameraMetrics

FEATURE_ENCODING = "physical_v1"


class PhysicalNormalizer:
    def __init__(self, config):
        for percentile in (config.blink_duration_percentile, config.nod_duration_percentile):
            if not isfinite(percentile) or not 0 <= percentile <= 100:
                raise ValueError("Duration percentiles must be in [0, 100]")
        # Dùng đúng cấu hình đã gộp của detector; chưa mở camera/FaceMesh.
        camera = CameraMetrics(config=config)

        def rate_bound(fsm):
            duration = max((fsm.minimum - 1e-6) / 1000, 1 / config.fps)
            return (ceil(fsm.window / duration) + 1) * 60 / fsm.window

        # FFT làm tròn số mẫu nội suy; cửa sổ lẻ frame có thể tăng Fs hiệu dụng.
        motion_window = camera.head_motion.window
        nyquist = max(3, ceil(motion_window * config.fps)) / (2 * motion_window)
        self.scales = {
            "blink_frequency": rate_bound(camera.eye.fsm),
            "blink_duration": camera.eye.fsm.maximum,
            "perclos": 1.0,
            "yawn_frequency": rate_bound(camera.mouth.fsm),
            "nod_duration": camera.nod.maximum / 1000,
            "nod_frequency": rate_bound(camera.nod),
            "dominant_head_motion_frequency": nyquist,
            "neuralnetwork_score": 1.0,
        }
        payload = json.dumps({"encoding": FEATURE_ENCODING, "config": asdict(config)},
                             sort_keys=True, allow_nan=False)
        self.id = hashlib.sha256(payload.encode()).hexdigest()

    def transform(self, raw):
        result = {}
        for key, upper in self.scales.items():
            value = raw[key]
            if isinstance(value, bool) or not isinstance(value, Real) or not isfinite(value) or value < 0:
                raise ValueError(f"Invalid raw feature: {key}")
            # Chỉ dung sai số học tại biên; không che số đo lỗi bằng clipping.
            if value > upper + 1e-6:
                raise ValueError(f"Raw feature exceeds detector bound: {key}")
            result[key] = min(float(value / upper), 1.0)
        return result
