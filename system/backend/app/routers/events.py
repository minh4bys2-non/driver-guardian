import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
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
from app.schemas.drowsiness_event import (
    DrowsinessEventCreate,
    DrowsinessEventResponse,
)


logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/events",
    tags=["Drowsiness Events"],
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


drowsiness_events_table = Table(
    "DROWSINESS_EVENTS",
    metadata,
    Column("EVENT_ID", Integer, primary_key=True),
    Column("SESSION_ID", Integer, nullable=False),
    Column("EVENT_TIME", DateTime),
    Column("DRIVER_STATE", String(20), nullable=False),
    Column("ALERT_LEVEL", Integer, nullable=False),
    Column("CONFIDENCE", Numeric(6, 5)),
    Column("DROWSINESS_SCORE", Numeric(5, 2)),
    Column("EAR_VALUE", Numeric(7, 5)),
    Column("MAR_VALUE", Numeric(7, 5)),
    Column("HEAD_POSE", String(50)),
    Column("DURATION_MS", Integer),
    Column("ACKNOWLEDGED", String(1)),
    Column("SYNC_STATUS", String(20)),
)


@router.post(
    "",
    response_model=DrowsinessEventResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_drowsiness_event(
    payload: DrowsinessEventCreate,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    active_session_query = text(
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

    try:
        session = database.execute(
            active_session_query,
            {"session_id": payload.session_id},
        ).mappings().first()

        if session is None or session["status"] != "ACTIVE":
            raise HTTPException(
                status_code=404,
                detail="Active driving session not found",
            )

        session_driver_id = session.get("driver_id")
        if current_user.get("role") == "DRIVER" and session_driver_id is not None and session_driver_id != current_user.get("driver_id"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this driving session",
            )

        insert_statement = (
            insert(drowsiness_events_table)
            .values(
                SESSION_ID=payload.session_id,
                DRIVER_STATE=payload.driver_state,
                ALERT_LEVEL=payload.alert_level,
                CONFIDENCE=payload.confidence,
                DURATION_MS=payload.duration_ms,
            )
            .returning(
                drowsiness_events_table.c.EVENT_ID.label("event_id"),
                drowsiness_events_table.c.SESSION_ID.label("session_id"),
                drowsiness_events_table.c.EVENT_TIME.label("event_time"),
                drowsiness_events_table.c.DRIVER_STATE.label("driver_state"),
                drowsiness_events_table.c.ALERT_LEVEL.label("alert_level"),
                drowsiness_events_table.c.CONFIDENCE.label("confidence"),
                drowsiness_events_table.c.DROWSINESS_SCORE.label("drowsiness_score"),
                drowsiness_events_table.c.EAR_VALUE.label("ear_value"),
                drowsiness_events_table.c.MAR_VALUE.label("mar_value"),
                drowsiness_events_table.c.HEAD_POSE.label("head_pose"),
                drowsiness_events_table.c.DURATION_MS.label("duration_ms"),
                drowsiness_events_table.c.ACKNOWLEDGED.label("acknowledged"),
                drowsiness_events_table.c.SYNC_STATUS.label("sync_status"),
            )
        )

        event = database.execute(insert_statement).mappings().one()

        database.execute(
            text(
                """
                UPDATE DRIVING_SESSIONS
                SET TOTAL_ALERTS = TOTAL_ALERTS + 1
                WHERE SESSION_ID = :session_id
                """
            ),
            {"session_id": payload.session_id},
        )

        database.commit()
        return dict(event)

    except HTTPException:
        database.rollback()
        raise
    except IntegrityError as error:
        database.rollback()
        raise HTTPException(
            status_code=409,
            detail="Event violates database constraints",
        ) from error
    except SQLAlchemyError as error:
        database.rollback()
        logger.error(
            "Drowsiness event creation failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to create event",
        ) from error


@router.post(
    "/{event_id}/acknowledge",
    response_model=DrowsinessEventResponse,
)
def acknowledge_drowsiness_event(
    event_id: int,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    lock_query = text(
        """
        SELECT
            E.EVENT_ID AS "event_id",
            E.SESSION_ID AS "session_id",
            E.ACKNOWLEDGED AS "acknowledged",
            S.DRIVER_ID AS "driver_id"
        FROM DROWSINESS_EVENTS E
        JOIN DRIVING_SESSIONS S ON S.SESSION_ID = E.SESSION_ID
        WHERE E.EVENT_ID = :event_id
        FOR UPDATE
        """
    )
    existing_action_query = text(
        """
        SELECT COUNT(*)
        FROM ALERT_ACTIONS
        WHERE EVENT_ID = :event_id
          AND ACTION_TYPE = 'DRIVER_CONFIRMATION'
        """
    )
    acknowledge_query = text(
        """
        UPDATE DROWSINESS_EVENTS
        SET ACKNOWLEDGED = 'Y'
        WHERE EVENT_ID = :event_id
        """
    )
    action_query = text(
        """
        INSERT INTO ALERT_ACTIONS (
            EVENT_ID,
            ACTION_TYPE
        ) VALUES (
            :event_id,
            'DRIVER_CONFIRMATION'
        )
        """
    )
    response_query = text(
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
        WHERE EVENT_ID = :event_id
        """
    )

    try:
        event = database.execute(
            lock_query,
            {"event_id": event_id},
        ).mappings().first()
        if event is None:
            raise HTTPException(
                status_code=404,
                detail="Drowsiness event not found",
            )

        event_driver_id = event.get("driver_id")
        if current_user.get("role") == "DRIVER" and event_driver_id is not None and event_driver_id != current_user.get("driver_id"):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied to this drowsiness event",
            )

        confirmation_count = database.execute(
            existing_action_query,
            {"event_id": event_id},
        ).scalar_one()

        if event["acknowledged"] != "Y":
            database.execute(
                acknowledge_query,
                {"event_id": event_id},
            )

        if confirmation_count == 0:
            database.execute(
                action_query,
                {"event_id": event_id},
            )

        acknowledged_event = database.execute(
            response_query,
            {"event_id": event_id},
        ).mappings().first()
        database.commit()
        return dict(acknowledged_event)

    except HTTPException:
        database.rollback()
        raise
    except (IntegrityError, SQLAlchemyError) as error:
        database.rollback()
        logger.error(
            "Event acknowledgement failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to acknowledge event",
        ) from error
