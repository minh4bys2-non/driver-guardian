import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    insert,
    text,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_db
from app.schemas.driving_session import (
    DrivingSessionCreate,
    DrivingSessionResponse,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/sessions",
    tags=["Driving Sessions"],
)


DatabaseSession = Annotated[
    Session,
    Depends(get_db),
]


metadata = MetaData()


driving_sessions_table = Table(
    "DRIVING_SESSIONS",
    metadata,
    Column("SESSION_ID", Integer, primary_key=True),
    Column("DRIVER_ID", Integer, nullable=False),
    Column("VEHICLE_ID", Integer, nullable=False),
    Column("MODEL_VERSION_ID", Integer),
    Column("START_TIME", DateTime),
    Column("DURATION_SECONDS", Integer),
    Column("TOTAL_ALERTS", Integer),
    Column("STATUS", String(20)),
    Column("SYNC_STATUS", String(20)),
)


@router.post(
    "",
    response_model=DrivingSessionResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_driving_session(
    payload: DrivingSessionCreate,
    database: DatabaseSession,
):
    validation_query = text(
        """
        SELECT
            (
                SELECT COUNT(*)
                FROM DRIVERS
                WHERE DRIVER_ID = :driver_id
                  AND STATUS = 'ACTIVE'
            ) AS "driver_count",

            (
                SELECT COUNT(*)
                FROM VEHICLES
                WHERE VEHICLE_ID = :vehicle_id
                  AND STATUS = 'ACTIVE'
            ) AS "vehicle_count",

            (
                SELECT COUNT(*)
                FROM MODEL_VERSIONS
                WHERE MODEL_VERSION_ID = :model_version_id
                  AND IS_ACTIVE = 'Y'
            ) AS "model_count"

        FROM DUAL
        """
    )

    parameters = {
        "driver_id": payload.driver_id,
        "vehicle_id": payload.vehicle_id,
        "model_version_id": payload.model_version_id,
    }

    try:
        validation_result = database.execute(
            validation_query,
            parameters,
        ).mappings().one()

        if validation_result["driver_count"] == 0:
            raise HTTPException(
                status_code=404,
                detail="Active driver not found",
            )

        if validation_result["vehicle_count"] == 0:
            raise HTTPException(
                status_code=404,
                detail="Active vehicle not found",
            )

        if validation_result["model_count"] == 0:
            raise HTTPException(
                status_code=404,
                detail="Active model version not found",
            )

        insert_statement = (
            insert(driving_sessions_table)
            .values(
                DRIVER_ID=payload.driver_id,
                VEHICLE_ID=payload.vehicle_id,
                MODEL_VERSION_ID=payload.model_version_id,
            )
            .returning(
                driving_sessions_table.c.SESSION_ID.label(
                    "session_id"
                ),
                driving_sessions_table.c.DRIVER_ID.label(
                    "driver_id"
                ),
                driving_sessions_table.c.VEHICLE_ID.label(
                    "vehicle_id"
                ),
                driving_sessions_table.c.MODEL_VERSION_ID.label(
                    "model_version_id"
                ),
                driving_sessions_table.c.START_TIME.label(
                    "start_time"
                ),
                driving_sessions_table.c.DURATION_SECONDS.label(
                    "duration_seconds"
                ),
                driving_sessions_table.c.TOTAL_ALERTS.label(
                    "total_alerts"
                ),
                driving_sessions_table.c.STATUS.label(
                    "status"
                ),
                driving_sessions_table.c.SYNC_STATUS.label(
                    "sync_status"
                ),
            )
        )

        result = database.execute(
            insert_statement
        ).mappings().one()

        database.commit()

        return dict(result)

    except HTTPException:
        database.rollback()
        raise

    except IntegrityError as error:
        database.rollback()

        raise HTTPException(
            status_code=409,
            detail="Session could not be created due to invalid data",
        ) from error

    except SQLAlchemyError as error:
        database.rollback()
        logger.error(
            "Driving session creation failed (%s)",
            type(error).__name__,
        )

        raise HTTPException(
            status_code=500,
            detail="Failed to create driving session",
        ) from error
