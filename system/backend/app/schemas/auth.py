from pydantic import BaseModel, ConfigDict, Field


class GoogleAuthRequest(BaseModel):
    id_token: str = Field(..., min_length=10, description="Google ID Token from Credential Manager")
    device_info: str | None = Field(None, max_length=255)


class RefreshTokenRequest(BaseModel):
    refresh_token: str = Field(..., min_length=10)
    device_info: str | None = Field(None, max_length=255)


class LogoutRequest(BaseModel):
    refresh_token: str | None = Field(None, description="Optional refresh token to revoke")


class DriverSummary(BaseModel):
    driver_id: int
    driver_code: str
    full_name: str

    model_config = ConfigDict(from_attributes=True)


class UserProfileResponse(BaseModel):
    user_id: int
    email: str
    display_name: str | None = None
    avatar_url: str | None = None
    role: str
    is_active: bool
    driver: DriverSummary | None = None

    model_config = ConfigDict(from_attributes=True)


class AuthTokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user: UserProfileResponse
