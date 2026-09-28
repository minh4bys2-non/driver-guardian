import logging

from fastapi import FastAPI, HTTPException

from app.database import check_database_connection
from app.routers.drivers import router as drivers_router
from app.routers.vehicles import router as vehicles_router
from app.routers.model_versions import router as model_versions_router
from app.routers.sessions import router as sessions_router
from app.routers.events import router as events_router


logger = logging.getLogger(__name__)


app = FastAPI(
    title="Driver Monitoring API",
    description=(
        "Backend cho hệ thống giám sát "
        "buồn ngủ tài xế"
    ),
    version="1.0.0",
)


app.include_router(drivers_router)
app.include_router(vehicles_router)
app.include_router(model_versions_router)
app.include_router(sessions_router)
app.include_router(events_router)


@app.get("/")
def root():
    return {
        "message": "Driver Monitoring API is running"
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "driver-monitoring-api",
    }


@app.get("/health/database")
def database_health_check():
    try:
        database_info = check_database_connection()

        return {
            "status": "ok",
            **database_info,
        }

    except Exception as error:
        logger.error(
            "Database health check failed (%s)",
            type(error).__name__,
        )
        raise HTTPException(
            status_code=503,
            detail="Database unavailable",
        ) from error
