from datetime import datetime

from pydantic import BaseModel, ConfigDict


class VehicleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vehicle_id: int
    plate_number: str
    vehicle_name: str | None = None
    vehicle_type: str | None = None
    device_code: str | None = None
    status: str
    created_at: datetime
