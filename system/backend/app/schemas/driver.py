from datetime import datetime

from pydantic import BaseModel, ConfigDict, field_validator


class DriverResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    driver_id: int
    driver_code: str
    full_name: str
    phone_number: str | None = None
    license_number: str | None = None
    status: str
    created_at: datetime


class DriverCreateMe(BaseModel):
    full_name: str
    phone_number: str | None = None
    license_number: str

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 2 or len(s) > 100:
            raise ValueError("Full name must be between 2 and 100 characters")
        return s

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            return None
        if len(s) < 7 or len(s) > 20:
            raise ValueError("Phone number must be between 7 and 20 characters")
        return s

    @field_validator("license_number")
    @classmethod
    def validate_license_number(cls, v: str) -> str:
        s = v.strip()
        if len(s) < 3 or len(s) > 50:
            raise ValueError("License number must be between 3 and 50 characters")
        return s


class DriverUpdateMe(BaseModel):
    full_name: str | None = None
    phone_number: str | None = None
    license_number: str | None = None

    @field_validator("full_name")
    @classmethod
    def validate_full_name(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if len(s) < 2 or len(s) > 100:
            raise ValueError("Full name must be between 2 and 100 characters")
        return s

    @field_validator("phone_number")
    @classmethod
    def validate_phone_number(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if not s:
            return None
        if len(s) < 7 or len(s) > 20:
            raise ValueError("Phone number must be between 7 and 20 characters")
        return s

    @field_validator("license_number")
    @classmethod
    def validate_license_number(cls, v: str | None) -> str | None:
        if v is None:
            return None
        s = v.strip()
        if len(s) < 3 or len(s) > 50:
            raise ValueError("License number must be between 3 and 50 characters")
        return s

