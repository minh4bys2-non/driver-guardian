# Driver Guardian FastAPI Backend

## Purpose

This service exposes the Driver Guardian system APIs and persists
driver, vehicle, model-version, user authentication, driving-session,
and drowsiness-event data in Oracle Database.

The backend is intentionally independent from Android and AI code. It exposes
the persisted driving-session lifecycle without inventing AI-derived scores.

## Architecture

- `app/main.py`: FastAPI application, router registration, and health routes.
- `app/database.py`: environment loading, SQLAlchemy Oracle engine, session
  factory, and database connectivity check.
- `app/auth/`: Google ID token verification, JWT issuance, passwordless session
  handling, and authentication dependencies (`get_current_user`, `get_current_driver`).
- `app/routers/`: HTTP routes (Auth, Drivers, Vehicles, Model Versions, Sessions, Events).
- `app/schemas/`: Pydantic request and response models.
- `tests/`: Oracle-independent unit tests (Trip lifecycle, Auth, Error handling, Smoke).

## Authentication & Security

- **Google ID Token Verification**: Validates issuer, audience, signature, and expiration
  using official Google cryptography tools (`google-auth`).
- **Driver Guardian Tokens**:
  - Access Token: Short-lived JWT (default 30 mins) with minimal claims (`sub`, `role`, `exp`).
  - Refresh Token: Long-lived cryptographically secure random token (default 30 days).
  - Storage: Only the SHA-256 hash (`TOKEN_HASH`) is stored in `AUTH_REFRESH_TOKENS`.
- **USER ≠ DRIVER Boundary**:
  - Authenticated accounts exist in `USERS`.
  - Business profiles exist in `DRIVERS`.
  - For `ROLE=DRIVER`, sessions are strictly bound to `USERS.DRIVER_ID`.
  - Users without a linked `DRIVER_ID` cannot create driving sessions.
  - Cross-driver session/event access returns `403 Forbidden`.

## Current API

| Method | Path | Description | Protected |
|---|---|---|---|
| GET | `/` | Service identity | No |
| GET | `/health` | Process-level health | No |
| GET | `/health/database` | Oracle connectivity and context | No |
| POST | `/auth/google` | Exchange verified Google ID token for Driver Guardian tokens | No |
| POST | `/auth/refresh` | Rotate access and refresh tokens | No |
| POST | `/auth/logout` | Revoke refresh token and invalidate session | Yes |
| GET | `/auth/me` | Return current authenticated user profile and linked driver | Yes |
| GET | `/drivers` | List drivers | Yes |
| GET | `/vehicles` | List vehicles | Yes |
| GET | `/model-versions/active` | Get the active model version | Yes |
| POST | `/sessions` | Create an active driving session (enforces driver ownership) | Yes |
| POST | `/sessions/{session_id}/complete` | Complete an active session | Yes |
| GET | `/sessions` | List session history filtered by authenticated driver | Yes |
| GET | `/sessions/{session_id}` | Get session detail (enforces driver ownership) | Yes |
| GET | `/sessions/{session_id}/events` | List session events in chronological order | Yes |
| POST | `/events` | Create a drowsiness event and increment session alerts | Yes |
| POST | `/events/{event_id}/acknowledge` | Idempotently record driver confirmation | Yes |

## Tests

Run the test suite:

```powershell
python -m unittest discover -s tests -v
```

All 27 backend tests run without needing live Google servers or a live Oracle database.
