import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.schemas.driver import DriverResponse


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/drivers",
    tags=["Drivers"],
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
    "",
    response_model=list[DriverResponse],
)
def get_drivers(
    database: DatabaseSession,
    current_user: CurrentUser,
):
    query = text(
        """
        SELECT
            DRIVER_ID AS "driver_id",
            DRIVER_CODE AS "driver_code",
            FULL_NAME AS "full_name",
            PHONE_NUMBER AS "phone_number",
            LICENSE_NUMBER AS "license_number",
            STATUS AS "status",
            CREATED_AT AS "created_at"
        FROM DRIVERS
        ORDER BY DRIVER_ID
        """
    )

    try:
        rows = database.execute(query).mappings().all()
        return [dict(row) for row in rows]
    except Exception as error:
        logger.error(
            "Driver query failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve drivers",
        ) from error
