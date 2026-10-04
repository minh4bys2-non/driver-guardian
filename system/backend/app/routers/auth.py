import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.auth.dependencies import (
    get_current_user,
    get_google_verifier,
)
from app.auth.google_verifier import GoogleTokenVerifier
from app.auth.security import (
    REFRESH_TOKEN_EXPIRE_DAYS,
    create_access_token,
    generate_refresh_token,
    hash_token,
)
from app.database import get_db
from app.schemas.auth import (
    AuthTokenResponse,
    DriverSummary,
    GoogleAuthRequest,
    LogoutRequest,
    RefreshTokenRequest,
    UserProfileResponse,
)


logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/auth",
    tags=["Authentication"],
)

DatabaseSession = Annotated[Session, Depends(get_db)]


def build_user_profile(user_dict: dict) -> UserProfileResponse:
    driver_summary = None
    if user_dict.get("driver_id") is not None and user_dict.get("driver_code") is not None:
        driver_summary = DriverSummary(
            driver_id=user_dict["driver_id"],
            driver_code=user_dict["driver_code"],
            full_name=user_dict.get("driver_full_name") or "",
        )

    return UserProfileResponse(
        user_id=user_dict["user_id"],
        email=user_dict["email"],
        display_name=user_dict.get("display_name"),
        avatar_url=user_dict.get("avatar_url"),
        role=user_dict["role"],
        is_active=(user_dict.get("is_active") == "Y"),
        driver=driver_summary,
    )


USER_SELECT_SQL = """
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
"""


@router.post(
    "/google",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_200_OK,
)
def login_with_google(
    payload: GoogleAuthRequest,
    database: DatabaseSession,
    verifier: Annotated[GoogleTokenVerifier, Depends(get_google_verifier)],
):
    try:
        google_user = verifier.verify(payload.id_token)
    except ValueError as error:
        logger.warning("Google token verification failed: %s", error)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(error),
        ) from error

    # 1. Look for existing user by Google sub
    sub_query = text(USER_SELECT_SQL + " WHERE U.GOOGLE_SUB = :google_sub")
    user_row = database.execute(sub_query, {"google_sub": google_user.sub}).mappings().first()

    if user_row is None:
        # 2. Check if a pre-provisioned user exists with matching verified email
        email_query = text(USER_SELECT_SQL + " WHERE U.EMAIL = :email")
        existing_email_user = database.execute(
            email_query, {"email": google_user.email}
        ).mappings().first()

        if existing_email_user is not None:
            # Bind google_sub to existing account
            bind_query = text(
                """
                UPDATE USERS
                SET GOOGLE_SUB = :google_sub,
                    DISPLAY_NAME = COALESCE(DISPLAY_NAME, :display_name),
                    AVATAR_URL = COALESCE(AVATAR_URL, :avatar_url),
                    LAST_LOGIN_AT = SYSTIMESTAMP,
                    UPDATED_AT = SYSTIMESTAMP
                WHERE USER_ID = :user_id
                """
            )
            database.execute(
                bind_query,
                {
                    "google_sub": google_user.sub,
                    "display_name": google_user.name,
                    "avatar_url": google_user.picture,
                    "user_id": existing_email_user["user_id"],
                },
            )
            user_row = database.execute(
                sub_query, {"google_sub": google_user.sub}
            ).mappings().first()
        else:
            # 3. Create a new unlinked DRIVER user
            insert_query = text(
                """
                INSERT INTO USERS (
                    GOOGLE_SUB,
                    EMAIL,
                    DISPLAY_NAME,
                    AVATAR_URL,
                    ROLE,
                    DRIVER_ID,
                    IS_ACTIVE,
                    LAST_LOGIN_AT
                )
                VALUES (
                    :google_sub,
                    :email,
                    :display_name,
                    :avatar_url,
                    'DRIVER',
                    NULL,
                    'Y',
                    SYSTIMESTAMP
                )
                """
            )
            database.execute(
                insert_query,
                {
                    "google_sub": google_user.sub,
                    "email": google_user.email,
                    "display_name": google_user.name,
                    "avatar_url": google_user.picture,
                },
            )
            user_row = database.execute(
                sub_query, {"google_sub": google_user.sub}
            ).mappings().first()
    else:
        # Update last login time
        update_login_query = text(
            """
            UPDATE USERS
            SET LAST_LOGIN_AT = SYSTIMESTAMP,
                UPDATED_AT = SYSTIMESTAMP
            WHERE USER_ID = :user_id
            """
        )
        database.execute(update_login_query, {"user_id": user_row["user_id"]})

    if user_row["is_active"] != "Y":
        database.rollback()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    user_dict = dict(user_row)
    access_token = create_access_token(user_dict["user_id"], user_dict["role"])
    raw_refresh = generate_refresh_token()
    token_hash = hash_token(raw_refresh)

    insert_refresh_query = text(
        f"""
        INSERT INTO AUTH_REFRESH_TOKENS (
            USER_ID,
            TOKEN_HASH,
            DEVICE_INFO,
            EXPIRES_AT
        )
        VALUES (
            :user_id,
            :token_hash,
            :device_info,
            SYSTIMESTAMP + NUMTODSINTERVAL({REFRESH_TOKEN_EXPIRE_DAYS}, 'DAY')
        )
        """
    )
    database.execute(
        insert_refresh_query,
        {
            "user_id": user_dict["user_id"],
            "token_hash": token_hash,
            "device_info": payload.device_info,
        },
    )
    database.commit()

    return AuthTokenResponse(
        access_token=access_token,
        refresh_token=raw_refresh,
        token_type="bearer",
        user=build_user_profile(user_dict),
    )


@router.post(
    "/refresh",
    response_model=AuthTokenResponse,
    status_code=status.HTTP_200_OK,
)
def refresh_access_token(
    payload: RefreshTokenRequest,
    database: DatabaseSession,
):
    token_hash = hash_token(payload.refresh_token)
    find_token_query = text(
        """
        SELECT
            R.TOKEN_ID AS "token_id",
            R.USER_ID AS "user_id",
            R.REVOKED_AT AS "revoked_at",
            R.EXPIRES_AT AS "expires_at",
            U.ROLE AS "role",
            U.IS_ACTIVE AS "is_active",
            U.EMAIL AS "email",
            U.DISPLAY_NAME AS "display_name",
            U.AVATAR_URL AS "avatar_url",
            U.DRIVER_ID AS "driver_id",
            D.DRIVER_CODE AS "driver_code",
            D.FULL_NAME AS "driver_full_name"
        FROM AUTH_REFRESH_TOKENS R
        JOIN USERS U ON U.USER_ID = R.USER_ID
        LEFT JOIN DRIVERS D ON D.DRIVER_ID = U.DRIVER_ID
        WHERE R.TOKEN_HASH = :token_hash
        """
    )
    row = database.execute(find_token_query, {"token_hash": token_hash}).mappings().first()

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token",
        )

    if row["revoked_at"] is not None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has been revoked",
        )

    # Check expiration via query or DB comparison
    check_expired_query = text(
        """
        SELECT CASE WHEN :expires_at < SYSTIMESTAMP THEN 1 ELSE 0 END AS "is_expired"
        FROM DUAL
        """
    )
    is_expired = database.execute(
        check_expired_query, {"expires_at": row["expires_at"]}
    ).scalar_one()

    if is_expired == 1:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token has expired",
        )

    if row["is_active"] != "Y":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive",
        )

    # Revoke old refresh token (rotation)
    revoke_query = text(
        """
        UPDATE AUTH_REFRESH_TOKENS
        SET REVOKED_AT = SYSTIMESTAMP
        WHERE TOKEN_ID = :token_id
        """
    )
    database.execute(revoke_query, {"token_id": row["token_id"]})

    # Issue new pair
    user_dict = dict(row)
    new_access_token = create_access_token(user_dict["user_id"], user_dict["role"])
    new_raw_refresh = generate_refresh_token()
    new_token_hash = hash_token(new_raw_refresh)

    insert_refresh_query = text(
        f"""
        INSERT INTO AUTH_REFRESH_TOKENS (
            USER_ID,
            TOKEN_HASH,
            DEVICE_INFO,
            EXPIRES_AT
        )
        VALUES (
            :user_id,
            :token_hash,
            :device_info,
            SYSTIMESTAMP + NUMTODSINTERVAL({REFRESH_TOKEN_EXPIRE_DAYS}, 'DAY')
        )
        """
    )
    database.execute(
        insert_refresh_query,
        {
            "user_id": user_dict["user_id"],
            "token_hash": new_token_hash,
            "device_info": payload.device_info,
        },
    )
    database.commit()

    return AuthTokenResponse(
        access_token=new_access_token,
        refresh_token=new_raw_refresh,
        token_type="bearer",
        user=build_user_profile(user_dict),
    )


@router.post(
    "/logout",
    status_code=status.HTTP_200_OK,
)
def logout(
    payload: LogoutRequest,
    database: DatabaseSession,
    current_user: Annotated[dict, Depends(get_current_user)],
):
    if payload.refresh_token:
        token_hash = hash_token(payload.refresh_token)
        revoke_query = text(
            """
            UPDATE AUTH_REFRESH_TOKENS
            SET REVOKED_AT = SYSTIMESTAMP
            WHERE TOKEN_HASH = :token_hash
              AND USER_ID = :user_id
              AND REVOKED_AT IS NULL
            """
        )
        database.execute(
            revoke_query,
            {"token_hash": token_hash, "user_id": current_user["user_id"]},
        )
    else:
        # Revoke all active refresh sessions for this user
        revoke_all_query = text(
            """
            UPDATE AUTH_REFRESH_TOKENS
            SET REVOKED_AT = SYSTIMESTAMP
            WHERE USER_ID = :user_id
              AND REVOKED_AT IS NULL
            """
        )
        database.execute(
            revoke_all_query,
            {"user_id": current_user["user_id"]},
        )

    database.commit()
    return {"message": "Logged out successfully"}


@router.get(
    "/me",
    response_model=UserProfileResponse,
    status_code=status.HTTP_200_OK,
)
def get_current_user_profile(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    return build_user_profile(current_user)
