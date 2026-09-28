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
):
    active_session_query = text(
        """
        SELECT COUNT(*)
        FROM DRIVING_SESSIONS
        WHERE SESSION_ID = :session_id
          AND STATUS = 'ACTIVE'
        """
    )

    try:
        session_count = database.execute(
            active_session_query,
            {"session_id": payload.session_id},
        ).scalar_one()

        if session_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Active driving session not found",
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
                drowsiness_events_table.c.DRIVER_STATE.label(
                    "driver_state"
                ),
                drowsiness_events_table.c.ALERT_LEVEL.label(
                    "alert_level"
                ),
                drowsiness_events_table.c.CONFIDENCE.label("confidence"),
                drowsiness_events_table.c.DURATION_MS.label(
                    "duration_ms"
                ),
                drowsiness_events_table.c.ACKNOWLEDGED.label(
                    "acknowledged"
                ),
                drowsiness_events_table.c.SYNC_STATUS.label(
                    "sync_status"
                ),
            )
        )

        event = database.execute(
            insert_statement
        ).mappings().one()

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
