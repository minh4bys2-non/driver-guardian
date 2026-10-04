import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.routers.auth import USER_SELECT_SQL, build_user_profile
from app.schemas.auth import UserProfileResponse
from app.schemas.driver import DriverCreateMe, DriverResponse, DriverUpdateMe

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


@router.post(
    "/me",
    response_model=UserProfileResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_my_driver_profile(
    payload: DriverCreateMe,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    user_id = current_user["user_id"]

    try:
        # 1. Lock the user row to prevent race conditions from concurrent onboarding requests
        lock_user_query = text(
            """
            SELECT USER_ID, DRIVER_ID, ROLE
            FROM USERS
            WHERE USER_ID = :user_id
            FOR UPDATE
            """
        )
        locked_user = database.execute(lock_user_query, {"user_id": user_id}).mappings().first()
        if locked_user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        if locked_user["driver_id"] is not None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="User account is already linked to a driver profile",
            )

        # 2. Check license number uniqueness
        license_check_query = text(
            "SELECT COUNT(*) FROM DRIVERS WHERE LICENSE_NUMBER = :license_number"
        )
        dup_count = database.execute(
            license_check_query, {"license_number": payload.license_number}
        ).scalar_one()
        if dup_count > 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="License number is already registered",
            )

        # 3. Generate safe unique DRIVER_CODE from Oracle sequence
        driver_code = None
        for _ in range(20):
            seq_val = database.execute(text("SELECT DRIVER_CODE_SEQ.NEXTVAL FROM DUAL")).scalar_one()
            candidate = f"DRV{seq_val:04d}"
            code_exists = database.execute(
                text("SELECT COUNT(*) FROM DRIVERS WHERE DRIVER_CODE = :code"),
                {"code": candidate},
            ).scalar_one()
            if code_exists == 0:
                driver_code = candidate
                break

        if driver_code is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not generate unique driver code",
            )

        # 4. Insert new DRIVER
        insert_driver_query = text(
            """
            INSERT INTO DRIVERS (
                DRIVER_CODE,
                FULL_NAME,
                PHONE_NUMBER,
                LICENSE_NUMBER,
                STATUS
            )
            VALUES (
                :driver_code,
                :full_name,
                :phone_number,
                :license_number,
                'ACTIVE'
            )
            """
        )
        database.execute(
            insert_driver_query,
            {
                "driver_code": driver_code,
                "full_name": payload.full_name,
                "phone_number": payload.phone_number,
                "license_number": payload.license_number,
            },
        )

        new_driver_id = database.execute(
            text("SELECT DRIVER_ID FROM DRIVERS WHERE DRIVER_CODE = :driver_code"),
            {"driver_code": driver_code},
        ).scalar_one()

        # 5. Link USERS.DRIVER_ID atomically in the same transaction
        update_user_query = text(
            """
            UPDATE USERS
            SET DRIVER_ID = :driver_id,
                DISPLAY_NAME = COALESCE(DISPLAY_NAME, :full_name),
                UPDATED_AT = SYSTIMESTAMP
            WHERE USER_ID = :user_id
            """
        )
        database.execute(
            update_user_query,
            {
                "driver_id": new_driver_id,
                "full_name": payload.full_name,
                "user_id": user_id,
            },
        )

        database.commit()

        # 6. Fetch updated user profile
        updated_row = database.execute(
            text(USER_SELECT_SQL + " WHERE U.USER_ID = :user_id"),
            {"user_id": user_id},
        ).mappings().first()

        return build_user_profile(dict(updated_row))

    except HTTPException:
        database.rollback()
        raise
    except IntegrityError as error:
        database.rollback()
        logger.warning("Integrity error during driver onboarding: %s", error)
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Driver information violates database constraints",
        ) from error
    except SQLAlchemyError as error:
        database.rollback()
        logger.error("Driver onboarding failed (%s)", type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create driver profile",
        ) from error


@router.get(
    "/me",
    response_model=DriverResponse,
    status_code=status.HTTP_200_OK,
)
def get_my_driver_profile(
    database: DatabaseSession,
    current_user: CurrentUser,
):
    driver_id = current_user.get("driver_id")
    if driver_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current user does not have a linked driver profile",
        )

    driver_query = text(
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
        WHERE DRIVER_ID = :driver_id
        """
    )
    row = database.execute(driver_query, {"driver_id": driver_id}).mappings().first()
    if row is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Driver profile not found",
        )
    return dict(row)


@router.patch(
    "/me",
    response_model=DriverResponse,
    status_code=status.HTTP_200_OK,
)
def update_my_driver_profile(
    payload: DriverUpdateMe,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    driver_id = current_user.get("driver_id")
    if driver_id is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Current user does not have a linked driver profile",
        )

    try:
        # Check license uniqueness if license_number is updated
        if payload.license_number is not None:
            dup_query = text(
                """
                SELECT COUNT(*)
                FROM DRIVERS
                WHERE LICENSE_NUMBER = :license_number
                  AND DRIVER_ID != :driver_id
                """
            )
            dup_count = database.execute(
                dup_query,
                {"license_number": payload.license_number, "driver_id": driver_id},
            ).scalar_one()
            if dup_count > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="License number already in use by another driver",
                )

        updates = []
        params = {"driver_id": driver_id}
        if payload.full_name is not None:
            updates.append("FULL_NAME = :full_name")
            params["full_name"] = payload.full_name
        if payload.phone_number is not None:
            updates.append("PHONE_NUMBER = :phone_number")
            params["phone_number"] = payload.phone_number
        if payload.license_number is not None:
            updates.append("LICENSE_NUMBER = :license_number")
            params["license_number"] = payload.license_number

        if updates:
            update_sql = f"UPDATE DRIVERS SET {', '.join(updates)} WHERE DRIVER_ID = :driver_id"
            database.execute(text(update_sql), params)

            if payload.full_name is not None:
                database.execute(
                    text("UPDATE USERS SET DISPLAY_NAME = :name, UPDATED_AT = SYSTIMESTAMP WHERE USER_ID = :user_id"),
                    {"name": payload.full_name, "user_id": current_user["user_id"]},
                )
            database.commit()

        driver_query = text(
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
            WHERE DRIVER_ID = :driver_id
            """
        )
        row = database.execute(driver_query, {"driver_id": driver_id}).mappings().first()
        return dict(row)

    except HTTPException:
        database.rollback()
        raise
    except IntegrityError as error:
        database.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Driver update violates database constraints",
        ) from error
    except SQLAlchemyError as error:
        database.rollback()
        logger.error("Driver update failed (%s)", type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update driver profile",
        ) from error
