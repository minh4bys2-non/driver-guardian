import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.vehicle import VehicleResponse


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/vehicles",
    tags=["Vehicles"],
)


DatabaseSession = Annotated[
    Session,
    Depends(get_db),
]


@router.get(
    "",
    response_model=list[VehicleResponse],
)
def get_vehicles(database: DatabaseSession):
    query = text(
        """
        SELECT
            VEHICLE_ID AS "vehicle_id",
            PLATE_NUMBER AS "plate_number",
            VEHICLE_NAME AS "vehicle_name",
            VEHICLE_TYPE AS "vehicle_type",
            DEVICE_CODE AS "device_code",
            STATUS AS "status",
            CREATED_AT AS "created_at"
        FROM VEHICLES
        ORDER BY VEHICLE_ID
        """
    )

    try:
        rows = database.execute(query).mappings().all()

        return [
            dict(row)
            for row in rows
        ]

    except Exception as error:
        logger.error(
            "Vehicle query failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve vehicles",
        ) from error
