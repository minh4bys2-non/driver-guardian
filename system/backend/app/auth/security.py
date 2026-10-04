import hashlib
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any

import jwt


JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "driver-guardian-insecure-secret-key-change-in-prod")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "14"))


def create_access_token(
    user_id: int,
    role: str,
    expires_delta: timedelta | None = None,
) -> str:
    now = datetime.now(timezone.utc)
    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    claims: dict[str, Any] = {
        "sub": str(user_id),
        "role": role,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
        "iss": "driver-guardian",
    }
    return jwt.encode(claims, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET_KEY,
            algorithms=[JWT_ALGORITHM],
            issuer="driver-guardian",
        )
        return payload
    except jwt.ExpiredSignatureError as error:
        raise ValueError("Token has expired") from error
    except jwt.InvalidTokenError as error:
        raise ValueError("Invalid token") from error


def generate_refresh_token() -> str:
    """Generates a secure random 256-bit URL-safe token."""
    return secrets.token_urlsafe(48)


def hash_token(token: str) -> str:
    """Computes a SHA-256 hash of the token for secure storage in the database."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
