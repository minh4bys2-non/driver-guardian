# Driver Guardian FastAPI Backend

## Purpose

This service exposes the current Driver Guardian system APIs and persists
driver, vehicle, model-version, driving-session, and drowsiness-event data in
Oracle Database.

The backend is intentionally independent from Android and AI code. This phase
preserves the recovered API behavior and does not add session-end, history,
alert-action, or analytics endpoints.

## Architecture

- `app/main.py`: FastAPI application, router registration, and health routes.
- `app/database.py`: environment loading, SQLAlchemy Oracle engine, session
  factory, and database connectivity check.
- `app/routers/`: HTTP routes and SQL operations.
- `app/schemas/`: Pydantic request and response models.
- `tests/`: Oracle-independent smoke and error-handling regression tests.

The implementation uses SQLAlchemy with the `oracle+oracledb` driver. A real
Oracle connection is opened only by database-backed requests.

## Requirements

- Python 3.12. The recovered environment used Python 3.12.10; repository
  verification was performed with Python 3.12.14.
- Oracle Database with the schema documented in `../database/` for
  database-backed endpoints.

## Setup

From `system/backend`:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Environment configuration

Copy `.env.example` to `.env` and supply values for:

```dotenv
DB_USER=
DB_PASSWORD=
DB_HOST=
DB_PORT=
DB_SERVICE=
```

The names match `app/database.py`. `.env` is ignored by Git; never commit real
credentials.

## Run

```powershell
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

- API root: `http://127.0.0.1:8000/`
- Swagger UI: `http://127.0.0.1:8000/docs`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

## Current API

| Method | Path | Description |
|---|---|---|
| GET | `/` | Service identity |
| GET | `/health` | Process-level health |
| GET | `/health/database` | Oracle connectivity and context |
| GET | `/drivers` | List drivers |
| GET | `/vehicles` | List vehicles |
| GET | `/model-versions/active` | Get the active model version |
| POST | `/sessions` | Create an active driving session |
| POST | `/events` | Create a drowsiness event and increment session alerts |

Database failures return generic client messages. Logs record only the
exception type, not the exception text, to avoid exposing connection details or
credentials.

## Tests

The unit smoke tests do not need Oracle. They set non-secret placeholder
environment variables and replace only the database access boundary needed by
each error-path test.

```powershell
python -m unittest discover -s tests -v
```

The suite verifies application import, `/`, `/health`, OpenAPI generation,
preservation of the eight current routes, and redaction of raw database errors.

## Oracle schema

See [`../database/README.md`](../database/README.md). Apply the recovered schema
only to an appropriate Oracle user. No `CREATE USER` artifact was recovered,
and the reference-only tablespace script must not be used as a default setup
script.

## Known limitations

- Oracle was not reachable during repository verification. Unit smoke tests
  remain valid without it; database integration is not verified.
- No portable Oracle container or user-provisioning script is included.
- The API currently has no session-end, trip-history, trip-detail,
  alert-action, or analytics routes.
- Android Retrofit integration is outside this phase.
