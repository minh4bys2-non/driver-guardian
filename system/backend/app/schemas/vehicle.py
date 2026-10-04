from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class VehicleResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    vehicle_id: int
    plate_number: str
    vehicle_name: str | None = None
    vehicle_type: str | None = None
    device_code: str | None = None
    status: str
    created_at: datetime
    driver_id: int | None = None


class VehicleCreate(BaseModel):
    plate_number: str
    vehicle_name: str
    vehicle_type: str
    device_code: str | None = None

    @field_validator("plate_number")
    @classmethod
    def validate_plate(cls, v: str) -> str:
        s = v.strip().upper()
        if len(s) < 3 or len(s) > 20:
            raise ValueError("Plate number must be between 3 and 20 characters")
        return s

    @field_validator("vehicle_name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 2 or len(s) > 100:
            raise ValueError("Vehicle name must be between 2 and 100 characters")
        return s

    @field_validator("vehicle_type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 2 or len(s) > 50:
            raise ValueError("Vehicle type must be between 2 and 50 characters")
        return s

    @field_validator("device_code")
    @classmethod
    def validate_device(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None


class VehicleUpdate(BaseModel):
    plate_number: str | None = None
    vehicle_name: str | None = None
    vehicle_type: str | None = None
    device_code: str | None = None

    @field_validator("plate_number")
    @classmethod
    def validate_plate(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip().upper()
        if len(s) < 3 or len(s) > 20:
            raise ValueError("Plate number must be between 3 and 20 characters")
        return s

    @field_validator("vehicle_name")
    @classmethod
    def validate_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if len(s) < 2 or len(s) > 100:
            raise ValueError("Vehicle name must be between 2 and 100 characters")
        return s

    @field_validator("vehicle_type")
    @classmethod
    def validate_type(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if len(s) < 2 or len(s) > 50:
            raise ValueError("Vehicle type must be between 2 and 50 characters")
        return s

    @field_validator("device_code")
    @classmethod
    def validate_device(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        return s if s else None

