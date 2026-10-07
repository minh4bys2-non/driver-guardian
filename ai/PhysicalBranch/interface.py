import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields


@dataclass(frozen=True)
class AIResult:
    """Chỉ số thô: giây, ms, độ, %, lần/phút và Hz theo tên trường."""

    timestamp_sec: float
    window_sec: float = 60.0
    windows_sec: dict[str, float] = field(default_factory=dict)
    face_detected: bool = False
    eye_ready: bool = False
    mouth_ready: bool = False
    head_ready: bool = False
    head_calibrated: bool = False
    pose_valid: bool = False
    pose_error: str | None = None

    ear: float | None = None
    eye_state: int | None = None
    blink_count: int = 0
    blink_rate_per_min: float = 0.0
    blink_detected: bool = False
    blink_duration_ms: float | None = None
    eye_closure_duration_ms: float | None = None
    last_eye_closure_duration_ms: float = 0.0
    perclos_pct: float | None = None
    p80_ear_threshold: float | None = None
    eye_observed_sec: float = 0.0

    mar: float | None = None
    mouth_state: int | None = None
    yawn_count: int = 0
    yawning_frequency_per_min: float = 0.0
    yawn_detected: bool = False
    yawn_duration_ms: float | None = None
    mouth_open_duration_ms: float | None = None
    last_mouth_open_duration_ms: float = 0.0
    pom_pct: float | None = None
    mouth_observed_sec: float = 0.0

    pitch_deg: float | None = None
    yaw_deg: float | None = None
    roll_deg: float | None = None
    pitch_amplitude_deg: float | None = None
    nodding: bool | None = None
    nod_count: int = 0
    nod_detected: bool = False
    nodding_frequency_per_min: float = 0.0
    nod_duration_ms: float | None = None
    last_nod_duration_ms: float = 0.0
    nod_event_duration_ms: float | None = None
    over_angle: bool | None = None
    over_angle_pct: float | None = None
    head_observed_sec: float = 0.0
    pitch_mean_deg: float | None = None
    pitch_mean_amplitude_deg: float | None = None
    head_motion_frequency_hz: float | None = None
    head_motion_resolution_hz: float | None = None
    head_motion_observed_sec: float = 0.0

    lstm_drowsiness_probability: float | None = None
    warning_score: float | None = None

    @classmethod
    def from_metrics(cls, metrics: Mapping):
        return cls(**{f.name: metrics[f.name] for f in fields(cls) if f.name in metrics})

    def to_json(self) -> str:
        return json.dumps(asdict(self), allow_nan=False)
