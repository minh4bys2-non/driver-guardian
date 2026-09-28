from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DriverResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    driver_id: int
    driver_code: str
    full_name: str
    phone_number: str | None = None
    license_number: str | None = None
    status: str
    created_at: datetime
