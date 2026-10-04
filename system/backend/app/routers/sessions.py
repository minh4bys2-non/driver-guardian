import logging
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import (
    Column,
    DateTime,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    insert,
    text,
)
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.schemas.driving_session import (
    DrivingSessionCreate,
    DrivingSessionDetailResponse,
    DrivingSessionResponse,
)
from app.schemas.drowsiness_event import DrowsinessEventResponse


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/sessions",
    tags=["Driving Sessions"],
)


DatabaseSession = Annotated[
    Session,
    Depends(get_db),
]

CurrentUser = Annotated[
    dict,
    Depends(get_current_user),
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
    Column("END_TIME", DateTime),
    Column("DURATION_SECONDS", Integer),
    Column("TOTAL_ALERTS", Integer),
    Column("SAFETY_SCORE", Numeric(5, 2)),
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
    current_user: CurrentUser,
):
    if current_user.get("role") == "DRIVER":
        effective_driver_id = current_user.get("driver_id")
        if effective_driver_id is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is not linked to any driver profile",
            )
    else:
        effective_driver_id = payload.driver_id

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
                FROM VEHICLES
                WHERE VEHICLE_ID = :vehicle_id
                  AND DRIVER_ID = :driver_id
                  AND STATUS = 'ACTIVE'
            ) AS "driver_vehicle_count",

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
        "driver_id": effective_driver_id,
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

        if (
            current_user.get("role") == "DRIVER"
            and validation_result.get("driver_vehicle_count") is not None
            and validation_result.get("driver_vehicle_count") == 0
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Vehicle does not belong to the authenticated driver",
            )

        if validation_result["model_count"] == 0:
            raise HTTPException(
                status_code=404,
                detail="Active model version not found",
            )

        insert_statement = (
            insert(driving_sessions_table)
            .values(
                DRIVER_ID=effective_driver_id,
                VEHICLE_ID=payload.vehicle_id,
                MODEL_VERSION_ID=payload.model_version_id,
            )
            .returning(
                driving_sessions_table.c.SESSION_ID.label("session_id"),
                driving_sessions_table.c.DRIVER_ID.label("driver_id"),
                driving_sessions_table.c.VEHICLE_ID.label("vehicle_id"),
                driving_sessions_table.c.MODEL_VERSION_ID.label("model_version_id"),
                driving_sessions_table.c.START_TIME.label("start_time"),
                driving_sessions_table.c.END_TIME.label("end_time"),
                driving_sessions_table.c.DURATION_SECONDS.label("duration_seconds"),
                driving_sessions_table.c.TOTAL_ALERTS.label("total_alerts"),
                driving_sessions_table.c.SAFETY_SCORE.label("safety_score"),
                driving_sessions_table.c.STATUS.label("status"),
                driving_sessions_table.c.SYNC_STATUS.label("sync_status"),
            )
        )

        result = database.execute(insert_statement).mappings().one()
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


SESSION_DETAIL_SELECT = """
    SELECT
        S.SESSION_ID AS "session_id",
        S.DRIVER_ID AS "driver_id",
        D.FULL_NAME AS "driver_name",
        S.VEHICLE_ID AS "vehicle_id",
        V.VEHICLE_NAME AS "vehicle_name",
        V.PLATE_NUMBER AS "plate_number",
        S.MODEL_VERSION_ID AS "model_version_id",
        M.VERSION_NAME AS "version_name",
        S.START_TIME AS "start_time",
        S.END_TIME AS "end_time",
        S.DURATION_SECONDS AS "duration_seconds",
        S.TOTAL_ALERTS AS "total_alerts",
        S.SAFETY_SCORE AS "safety_score",
        S.STATUS AS "status",
        S.SYNC_STATUS AS "sync_status"
    FROM DRIVING_SESSIONS S
    JOIN DRIVERS D ON D.DRIVER_ID = S.DRIVER_ID
    JOIN VEHICLES V ON V.VEHICLE_ID = S.VEHICLE_ID
    LEFT JOIN MODEL_VERSIONS M
        ON M.MODEL_VERSION_ID = S.MODEL_VERSION_ID
"""


@router.get(
    "",
    response_model=list[DrivingSessionDetailResponse],
)
def get_driving_sessions(
    database: DatabaseSession,
    current_user: CurrentUser,
    driver_id: Annotated[int | None, Query(gt=0)] = None,
    session_status: Annotated[
        Literal["ACTIVE", "COMPLETED", "CANCELLED"] | None,
        Query(alias="status"),
    ] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 50,
):
    filters = []
    parameters = {"limit": limit}

    if current_user.get("role") == "DRIVER":
        effective_driver_id = current_user.get("driver_id")
        filters.append("S.DRIVER_ID = :driver_id")
        parameters["driver_id"] = effective_driver_id
    elif driver_id is not None:
        filters.append("S.DRIVER_ID = :driver_id")
        parameters["driver_id"] = driver_id

    if session_status is not None:
        filters.append("S.STATUS = :status")
        parameters["status"] = session_status

    where_clause = ""
    if filters:
        where_clause = " WHERE " + " AND ".join(filters)

    query = text(
        f"""
        SELECT *
        FROM (
            {SESSION_DETAIL_SELECT}
            {where_clause}
            ORDER BY S.START_TIME DESC
        )
        WHERE ROWNUM <= :limit
        """
    )

    try:
        rows = database.execute(query, parameters).mappings().all()
        return [dict(row) for row in rows]
    except SQLAlchemyError as error:
        logger.error(
            "Driving session history query failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve sessions",
        ) from error


@router.post(
    "/{session_id}/complete",
    response_model=DrivingSessionResponse,
)
def complete_driving_session(
    session_id: int,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    lock_query = text(
        """
        SELECT
            SESSION_ID AS "session_id",
            DRIVER_ID AS "driver_id",
            STATUS AS "status"
        FROM DRIVING_SESSIONS
        WHERE SESSION_ID = :session_id
        FOR UPDATE
        """
    )
    completion_time_query = text(
        """
        SELECT SYSTIMESTAMP
        FROM DUAL
        """
    )
    update_query = text(
        """
        UPDATE DRIVING_SESSIONS
        SET
            END_TIME = :completion_time,
            DURATION_SECONDS = GREATEST(
                0,
                FLOOR(
                    (
                        CAST(:completion_time AS DATE)
                        - CAST(START_TIME AS DATE)
                    ) * 86400
                )
            ),
            STATUS = 'COMPLETED'
        WHERE SESSION_ID = :session_id
        """
    )
    detail_query = text(
        SESSION_DETAIL_SELECT
        + " WHERE S.SESSION_ID = :session_id"
    )

    try:
        locked_session = database.execute(
            lock_query,
            {"session_id": session_id},
        ).mappings().first()

        if locked_session is None:
            raise HTTPException(
                status_code=404,
                detail="Driving session not found",
            )

        session_driver_id = locked_session.get("driver_id")
        if current_user.get("role") == "DRIVER" and session_driver_id is not None and session_driver_id != current_user.get("driver_id"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this driving session",
            )

        if locked_session["status"] != "ACTIVE":
            raise HTTPException(
                status_code=409,
                detail="Driving session is not active",
            )

        completion_time = database.execute(completion_time_query).scalar_one()
        database.execute(
            update_query,
            {
                "session_id": session_id,
                "completion_time": completion_time,
            },
        )
        completed_session = database.execute(
            detail_query,
            {"session_id": session_id},
        ).mappings().first()
        database.commit()
        return dict(completed_session)

    except HTTPException:
        database.rollback()
        raise
    except SQLAlchemyError as error:
        database.rollback()
        logger.error(
            "Driving session completion failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to complete driving session",
        ) from error


@router.get(
    "/{session_id}/events",
    response_model=list[DrowsinessEventResponse],
)
def get_session_events(
    session_id: int,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    session_query = text(
        """
        SELECT
            SESSION_ID AS "session_id",
            DRIVER_ID AS "driver_id"
        FROM DRIVING_SESSIONS
        WHERE SESSION_ID = :session_id
        """
    )
    event_query = text(
        """
        SELECT
            EVENT_ID AS "event_id",
            SESSION_ID AS "session_id",
            EVENT_TIME AS "event_time",
            DRIVER_STATE AS "driver_state",
            ALERT_LEVEL AS "alert_level",
            CONFIDENCE AS "confidence",
            DROWSINESS_SCORE AS "drowsiness_score",
            EAR_VALUE AS "ear_value",
            MAR_VALUE AS "mar_value",
            HEAD_POSE AS "head_pose",
            DURATION_MS AS "duration_ms",
            ACKNOWLEDGED AS "acknowledged",
            SYNC_STATUS AS "sync_status"
        FROM DROWSINESS_EVENTS
        WHERE SESSION_ID = :session_id
        ORDER BY EVENT_TIME ASC
        """
    )

    try:
        session_result = database.execute(
            session_query,
            {"session_id": session_id},
        )
        session_row = session_result.mappings().first() if hasattr(session_result, "mappings") else None
        if session_row is None and hasattr(session_result, "scalar_one"):
            try:
                scalar_val = session_result.scalar_one()
                if scalar_val == 0:
                    raise HTTPException(
                        status_code=404,
                        detail="Driving session not found",
                    )
            except Exception:
                raise HTTPException(
                    status_code=404,
                    detail="Driving session not found",
                )
        elif session_row is None:
            raise HTTPException(
                status_code=404,
                detail="Driving session not found",
            )
        else:
            session_driver_id = session_row.get("driver_id")
            if current_user.get("role") == "DRIVER" and session_driver_id is not None and session_driver_id != current_user.get("driver_id"):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Access denied to this driving session",
                )

        rows = database.execute(
            event_query,
            {"session_id": session_id},
        ).mappings().all()
        return [dict(row) for row in rows]
    except HTTPException:
        raise
    except SQLAlchemyError as error:
        logger.error(
            "Session event query failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve session events",
        ) from error


@router.get(
    "/{session_id}",
    response_model=DrivingSessionDetailResponse,
)
def get_driving_session(
    session_id: int,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    query = text(
        SESSION_DETAIL_SELECT
        + " WHERE S.SESSION_ID = :session_id"
    )
    try:
        row = database.execute(
            query,
            {"session_id": session_id},
        ).mappings().first()
        if row is None:
            raise HTTPException(
                status_code=404,
                detail="Driving session not found",
            )
        session_driver_id = row.get("driver_id")
        if current_user.get("role") == "DRIVER" and session_driver_id is not None and session_driver_id != current_user.get("driver_id"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this driving session",
            )
        return dict(row)
    except HTTPException:
        raise
    except SQLAlchemyError as error:
        logger.error(
            "Driving session detail query failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve driving session",
        ) from error
