from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite

from ai.PhysicalBranch.interface import AIResult


# Các mốc tham khảo từ Metric_revised.md; cần hiệu chỉnh theo người/camera/cửa sổ.
DEFAULT_SCALES = dict.fromkeys((
    "ear", "blink_rate_per_min", "mar", "mouth_open_duration_ms", "yawn_duration_ms",
    "yawning_frequency_per_min", "pom_pct", "nod_duration_ms", "nod_event_duration_ms",
    "nodding_frequency_per_min", "pitch_mean_amplitude_deg", "head_motion_frequency_hz",
    "over_angle_pct",
))
DEFAULT_SCALES.update(
    eye_closure_duration_ms=((400, 0), (800, 1)),
    blink_duration_ms=((400, 0), (800, 1)),
    perclos_pct=((0, 0), (15, 1)),
    pitch_amplitude_deg=((8, 0), (20, 1)),
)


def normalize_metrics(
    raw: AIResult | Mapping,
    *,
    scales: Mapping[str, tuple[tuple[float, float], ...] | None] | None = None,
) -> dict[str, float | None]:
    """Nội suy điểm quy ước 0–1, không kết luận buồn ngủ.

    Thiếu số đo/thang hoặc NaN/Inf trả None. Các chỉ số tần suất, MAR và
    thời lượng ngáp/gật cần thang do người dùng hiệu chỉnh. Không chấm điểm
    góc pitch có dấu bằng một dải tuyệt đối; dùng pitch_amplitude_deg.
    """
    data = asdict(raw) if isinstance(raw, AIResult) else raw
    configured = dict(DEFAULT_SCALES)
    if scales is not None:
        unknown = scales.keys() - configured.keys()
        if unknown:
            raise ValueError(f"Chỉ số không hỗ trợ cấu hình thang: {sorted(unknown)}")
        configured.update(scales)

    def read(name):
        value = data.get(name)
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} phải là số đo, không phải bool")
        value = float(value)
        if not isfinite(value):
            return None
        if value < 0 or name.endswith("_pct") and value > 100:
            raise ValueError(f"{name} nằm ngoài miền giá trị hợp lệ")
        return value

    result = {}
    for name, points in configured.items():
        if points is not None:
            if (len(points) < 2
                    or any(not isfinite(x) or not isfinite(y) or not 0 <= y <= 1 for x, y in points)
                    or any(a[0] >= b[0] for a, b in zip(points, points[1:]))):
                raise ValueError(f"Thang {name} cần >=2 mốc hữu hạn, x tăng dần và điểm trong [0, 1]")
        value = read(name)
        if value is None or points is None:
            result[name] = None
            continue
        score = points[0][1] if value <= points[0][0] else points[-1][1]
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            if x0 <= value <= x1:
                score = y0 + (value - x0) / (x1 - x0) * (y1 - y0)
                break
        result[name] = max(0.0, min(1.0, float(score)))
    return result
