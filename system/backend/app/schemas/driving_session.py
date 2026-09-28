from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class DrivingSessionCreate(BaseModel):
    driver_id: int = Field(gt=0)
    vehicle_id: int = Field(gt=0)
    model_version_id: int = Field(gt=0)


class DrivingSessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    session_id: int
    driver_id: int
    vehicle_id: int
    model_version_id: int | None
    start_time: datetime
    end_time: datetime | None = None
    duration_seconds: int
    total_alerts: int
    safety_score: float | None = None
    status: str
    sync_status: str


class DrivingSessionDetailResponse(DrivingSessionResponse):
    driver_name: str
    vehicle_name: str | None = None
    plate_number: str
    version_name: str | None = None
