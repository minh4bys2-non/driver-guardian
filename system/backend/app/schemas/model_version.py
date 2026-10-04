from datetime import datetime

from pydantic import BaseModel, ConfigDict


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    model_version_id: int
    version_name: str
    model_type: str | None = None
    file_name: str | None = None
    description: str | None = None
    is_active: str
    deployed_at: datetime
