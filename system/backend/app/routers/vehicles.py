import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.auth.dependencies import get_current_user
from app.database import get_db
from app.schemas.vehicle import VehicleCreate, VehicleResponse, VehicleUpdate

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/vehicles",
    tags=["Vehicles"],
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
    response_model=list[VehicleResponse],
)
def get_vehicles(
    database: DatabaseSession,
    current_user: CurrentUser,
    driver_id: Annotated[int | None, Query(gt=0)] = None,
):
    role = current_user.get("role")
    effective_driver_id = current_user.get("driver_id")

    if role == "DRIVER":
        if effective_driver_id is None:
            return []
        query = text(
            """
            SELECT
                VEHICLE_ID AS "vehicle_id",
                PLATE_NUMBER AS "plate_number",
                VEHICLE_NAME AS "vehicle_name",
                VEHICLE_TYPE AS "vehicle_type",
                DEVICE_CODE AS "device_code",
                STATUS AS "status",
                CREATED_AT AS "created_at",
                DRIVER_ID AS "driver_id"
            FROM VEHICLES
            WHERE DRIVER_ID = :driver_id
              AND STATUS = 'ACTIVE'
            ORDER BY VEHICLE_ID
            """
        )
        params = {"driver_id": effective_driver_id}
    else:
        # ADMIN: can view all or filter by driver_id
        if driver_id is not None:
            query = text(
                """
                SELECT
                    VEHICLE_ID AS "vehicle_id",
                    PLATE_NUMBER AS "plate_number",
                    VEHICLE_NAME AS "vehicle_name",
                    VEHICLE_TYPE AS "vehicle_type",
                    DEVICE_CODE AS "device_code",
                    STATUS AS "status",
                    CREATED_AT AS "created_at",
                    DRIVER_ID AS "driver_id"
                FROM VEHICLES
                WHERE DRIVER_ID = :driver_id
                  AND STATUS = 'ACTIVE'
                ORDER BY VEHICLE_ID
                """
            )
            params = {"driver_id": driver_id}
        else:
            query = text(
                """
                SELECT
                    VEHICLE_ID AS "vehicle_id",
                    PLATE_NUMBER AS "plate_number",
                    VEHICLE_NAME AS "vehicle_name",
                    VEHICLE_TYPE AS "vehicle_type",
                    DEVICE_CODE AS "device_code",
                    STATUS AS "status",
                    CREATED_AT AS "created_at",
                    DRIVER_ID AS "driver_id"
                FROM VEHICLES
                WHERE STATUS = 'ACTIVE'
                ORDER BY VEHICLE_ID
                """
            )
            params = {}

    try:
        rows = database.execute(query, params).mappings().all()
        return [dict(row) for row in rows]
    except Exception as error:
        logger.error(
            "Vehicle query failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="Failed to retrieve vehicles",
        ) from error


@router.post(
    "",
    response_model=VehicleResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_vehicle(
    payload: VehicleCreate,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    role = current_user.get("role")
    effective_driver_id = current_user.get("driver_id")

    if role == "DRIVER" and effective_driver_id is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is not linked to any driver profile",
        )

    try:
        # Check duplicate plate number
        dup_plate_query = text(
            "SELECT COUNT(*) FROM VEHICLES WHERE PLATE_NUMBER = :plate_number"
        )
        if database.execute(dup_plate_query, {"plate_number": payload.plate_number}).scalar_one() > 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Plate number already exists",
            )

        # Check duplicate device code if provided
        if payload.device_code:
            dup_device_query = text(
                "SELECT COUNT(*) FROM VEHICLES WHERE DEVICE_CODE = :device_code"
            )
            if database.execute(dup_device_query, {"device_code": payload.device_code}).scalar_one() > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Device code already exists",
                )

        insert_query = text(
            """
            INSERT INTO VEHICLES (
                PLATE_NUMBER,
                VEHICLE_NAME,
                VEHICLE_TYPE,
                DEVICE_CODE,
                DRIVER_ID,
                STATUS
            )
            VALUES (
                :plate_number,
                :vehicle_name,
                :vehicle_type,
                :device_code,
                :driver_id,
                'ACTIVE'
            )
            """
        )
        database.execute(
            insert_query,
            {
                "plate_number": payload.plate_number,
                "vehicle_name": payload.vehicle_name,
                "vehicle_type": payload.vehicle_type,
                "device_code": payload.device_code,
                "driver_id": effective_driver_id,
            },
        )
        database.commit()

        # Fetch created vehicle
        fetch_query = text(
            """
            SELECT
                VEHICLE_ID AS "vehicle_id",
                PLATE_NUMBER AS "plate_number",
                VEHICLE_NAME AS "vehicle_name",
                VEHICLE_TYPE AS "vehicle_type",
                DEVICE_CODE AS "device_code",
                STATUS AS "status",
                CREATED_AT AS "created_at",
                DRIVER_ID AS "driver_id"
            FROM VEHICLES
            WHERE PLATE_NUMBER = :plate_number
            """
        )
        row = database.execute(fetch_query, {"plate_number": payload.plate_number}).mappings().first()
        return dict(row)

    except HTTPException:
        database.rollback()
        raise
    except IntegrityError as error:
        database.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Vehicle information violates unique constraints",
        ) from error
    except SQLAlchemyError as error:
        database.rollback()
        logger.error("Vehicle creation failed (%s)", type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create vehicle",
        ) from error


@router.patch(
    "/{vehicle_id}",
    response_model=VehicleResponse,
    status_code=status.HTTP_200_OK,
)
def update_vehicle(
    vehicle_id: int,
    payload: VehicleUpdate,
    database: DatabaseSession,
    current_user: CurrentUser,
):
    role = current_user.get("role")
    effective_driver_id = current_user.get("driver_id")

    # Fetch vehicle
    find_query = text(
        """
        SELECT
            VEHICLE_ID AS "vehicle_id",
            PLATE_NUMBER AS "plate_number",
            VEHICLE_NAME AS "vehicle_name",
            VEHICLE_TYPE AS "vehicle_type",
            DEVICE_CODE AS "device_code",
            STATUS AS "status",
            DRIVER_ID AS "driver_id"
        FROM VEHICLES
        WHERE VEHICLE_ID = :vehicle_id
        """
    )
    existing = database.execute(find_query, {"vehicle_id": vehicle_id}).mappings().first()
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Vehicle not found",
        )

    # Ownership check
    if role == "DRIVER" and (effective_driver_id is None or existing["driver_id"] != effective_driver_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied to this vehicle",
        )

    try:
        # Check duplicate plate number if updating
        if payload.plate_number is not None:
            dup_plate_query = text(
                """
                SELECT COUNT(*)
                FROM VEHICLES
                WHERE PLATE_NUMBER = :plate_number
                  AND VEHICLE_ID != :vehicle_id
                """
            )
            if database.execute(dup_plate_query, {"plate_number": payload.plate_number, "vehicle_id": vehicle_id}).scalar_one() > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Plate number already exists",
                )

        # Check duplicate device code if updating
        if payload.device_code is not None:
            dup_device_query = text(
                """
                SELECT COUNT(*)
                FROM VEHICLES
                WHERE DEVICE_CODE = :device_code
                  AND VEHICLE_ID != :vehicle_id
                """
            )
            if database.execute(dup_device_query, {"device_code": payload.device_code, "vehicle_id": vehicle_id}).scalar_one() > 0:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Device code already exists",
                )

        updates = []
        params = {"vehicle_id": vehicle_id}
        if payload.plate_number is not None:
            updates.append("PLATE_NUMBER = :plate_number")
            params["plate_number"] = payload.plate_number
        if payload.vehicle_name is not None:
            updates.append("VEHICLE_NAME = :vehicle_name")
            params["vehicle_name"] = payload.vehicle_name
        if payload.vehicle_type is not None:
            updates.append("VEHICLE_TYPE = :vehicle_type")
            params["vehicle_type"] = payload.vehicle_type
        if payload.device_code is not None:
            updates.append("DEVICE_CODE = :device_code")
            params["device_code"] = payload.device_code

        if updates:
            update_sql = f"UPDATE VEHICLES SET {', '.join(updates)} WHERE VEHICLE_ID = :vehicle_id"
            database.execute(text(update_sql), params)
            database.commit()

        fetch_query = text(
            """
            SELECT
                VEHICLE_ID AS "vehicle_id",
                PLATE_NUMBER AS "plate_number",
                VEHICLE_NAME AS "vehicle_name",
                VEHICLE_TYPE AS "vehicle_type",
                DEVICE_CODE AS "device_code",
                STATUS AS "status",
                CREATED_AT AS "created_at",
                DRIVER_ID AS "driver_id"
            FROM VEHICLES
            WHERE VEHICLE_ID = :vehicle_id
            """
        )
        row = database.execute(fetch_query, {"vehicle_id": vehicle_id}).mappings().first()
        return dict(row)

    except HTTPException:
        database.rollback()
        raise
    except IntegrityError as error:
        database.rollback()
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Vehicle update violates unique constraints",
        ) from error
    except SQLAlchemyError as error:
        database.rollback()
        logger.error("Vehicle update failed (%s)", type(error).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update vehicle",
        ) from error
