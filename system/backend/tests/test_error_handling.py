import os
import unittest
from unittest.mock import patch

from httpx import ASGITransport, AsyncClient
from sqlalchemy.exc import SQLAlchemyError


os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("DB_HOST", "127.0.0.1")
os.environ.setdefault("DB_PORT", "1521")
os.environ.setdefault("DB_SERVICE", "FREEPDB1")

from app.database import get_db
from app.main import app


SENSITIVE_DATABASE_ERROR = (
    "ORA-12514: connection failed; password=local-secret"
)


class FailingDatabaseSession:
    def execute(self, *_args, **_kwargs):
        raise SQLAlchemyError(SENSITIVE_DATABASE_ERROR)

    def rollback(self):
        pass


def failing_database_dependency():
    yield FailingDatabaseSession()


class DatabaseErrorHandlingTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        app.dependency_overrides.clear()
        await self.client.aclose()

    async def test_database_health_hides_internal_exception(self):
        with patch(
            "app.main.check_database_connection",
            side_effect=RuntimeError(SENSITIVE_DATABASE_ERROR),
        ):
            response = await self.client.get("/health/database")

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json(), {"detail": "Database unavailable"})
        self.assertNotIn("local-secret", response.text)
        self.assertNotIn("ORA-12514", response.text)

    async def test_database_routes_hide_internal_exceptions(self):
        app.dependency_overrides[get_db] = failing_database_dependency
        cases = (
            ("GET", "/drivers", None, 500, "Failed to retrieve drivers"),
            ("GET", "/vehicles", None, 500, "Failed to retrieve vehicles"),
            (
                "GET",
                "/model-versions/active",
                None,
                500,
                "Failed to retrieve active model",
            ),
            (
                "POST",
                "/sessions",
                {
                    "driver_id": 1,
                    "vehicle_id": 1,
                    "model_version_id": 1,
                },
                500,
                "Failed to create driving session",
            ),
            (
                "POST",
                "/events",
                {
                    "session_id": 1,
                    "driver_state": "WARNING",
                    "alert_level": 1,
                },
                500,
                "Failed to create event",
            ),
        )

        for method, path, payload, status, detail in cases:
            with self.subTest(path=path):
                response = await self.client.request(
                    method,
                    path,
                    json=payload,
                )

                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json(), {"detail": detail})
                self.assertNotIn("local-secret", response.text)
                self.assertNotIn("ORA-12514", response.text)


if __name__ == "__main__":
    unittest.main()
