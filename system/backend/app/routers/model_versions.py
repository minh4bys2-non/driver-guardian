import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.schemas.model_version import ModelVersionResponse


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/model-versions",
    tags=["Model Versions"],
)


DatabaseSession = Annotated[
    Session,
    Depends(get_db),
]

CurrentUser = Annotated[
    dict,
    Depends(get_current_user),
]


@router.get(
    "/active",
    response_model=ModelVersionResponse,
)
def get_active_model_version(
    database: DatabaseSession,
    current_user: CurrentUser,
):
    query = text(
        """
        SELECT
            MODEL_VERSION_ID AS "model_version_id",
            VERSION_NAME AS "version_name",
            MODEL_TYPE AS "model_type",
            FILE_NAME AS "file_name",
            DESCRIPTION AS "description",
            IS_ACTIVE AS "is_active",
            DEPLOYED_AT AS "deployed_at"
        FROM MODEL_VERSIONS
        WHERE IS_ACTIVE = 'Y'
        ORDER BY DEPLOYED_AT DESC
        FETCH FIRST 1 ROW ONLY
        """
    )

    try:
        row = database.execute(query).mappings().first()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail="No active model version found",
            )

        return dict(row)

    except HTTPException:
        raise

    except Exception as error:
        logger.error(
            "Active model query failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve active model",
        ) from error
