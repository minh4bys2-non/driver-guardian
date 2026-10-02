from typing import Annotated

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.google_verifier import DefaultGoogleIdTokenVerifier, GoogleTokenVerifier
from app.auth.security import decode_access_token
from app.database import get_db


security_scheme = HTTPBearer(auto_error=False)


def get_google_verifier() -> GoogleTokenVerifier:
    return DefaultGoogleIdTokenVerifier()


def get_current_user(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(security_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> dict:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = decode_access_token(token)
    except ValueError as error:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
            headers={"WWW-Authenticate": "Bearer"},
        ) from error

    user_id_str = payload.get("sub")
    if not user_id_str or not user_id_str.isdigit():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token subject",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(user_id_str)
    query = text(
        """
        SELECT
            U.USER_ID AS "user_id",
            U.GOOGLE_SUB AS "google_sub",
            U.EMAIL AS "email",
            U.DISPLAY_NAME AS "display_name",
            U.AVATAR_URL AS "avatar_url",
            U.ROLE AS "role",
            U.DRIVER_ID AS "driver_id",
            U.IS_ACTIVE AS "is_active",
            D.DRIVER_CODE AS "driver_code",
            D.FULL_NAME AS "driver_full_name"
        FROM USERS U
        LEFT JOIN DRIVERS D ON D.DRIVER_ID = U.DRIVER_ID
        WHERE U.USER_ID = :user_id
        """
    )
    user_row = db.execute(query, {"user_id": user_id}).mappings().first()
    if user_row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user_row["is_active"] != "Y":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    return dict(user_row)


def get_current_driver(
    current_user: Annotated[dict, Depends(get_current_user)],
) -> dict:
    if current_user["role"] == "DRIVER":
        if current_user["driver_id"] is None:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="User account is not linked to any driver profile",
            )
    return current_user
