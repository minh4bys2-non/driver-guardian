from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DrowsinessEventCreate(BaseModel):
    session_id: int = Field(gt=0)

    # AI trả ALARM thì Android đổi thành DANGER trước khi gửi API.
    driver_state: Literal["WARNING", "DANGER"]

    alert_level: int = Field(ge=1, le=2)

    # Có thể để null nếu model chưa cung cấp.
    confidence: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )

    duration_ms: int | None = Field(
        default=None,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_state_and_level(self):
        expected_level = {
            "WARNING": 1,
            "DANGER": 2,
        }[self.driver_state]

        if self.alert_level != expected_level:
            raise ValueError(
                f"{self.driver_state} must use "
                f"alert_level={expected_level}"
            )

        return self


class DrowsinessEventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    event_id: int
    session_id: int
    event_time: datetime
    driver_state: str
    alert_level: int
    confidence: float | None = None
    drowsiness_score: float | None = None
    ear_value: float | None = None
    mar_value: float | None = None
    head_pose: str | None = None
    duration_ms: int | None = None
    acknowledged: str
    sync_status: str
