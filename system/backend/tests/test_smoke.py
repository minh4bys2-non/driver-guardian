import os
import unittest

from httpx import ASGITransport, AsyncClient


os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("DB_HOST", "127.0.0.1")
os.environ.setdefault("DB_PORT", "1521")
os.environ.setdefault("DB_SERVICE", "FREEPDB1")

from app.main import app


class BackendSmokeTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        await self.client.aclose()

    async def test_application_imports(self):
        self.assertEqual(app.title, "Driver Monitoring API")

    async def test_root_returns_service_message(self):
        response = await self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"message": "Driver Monitoring API is running"},
        )

    async def test_health_returns_service_status(self):
        response = await self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "service": "driver-monitoring-api",
            },
        )

    async def test_openapi_generation_includes_preserved_routes(self):
        schema = app.openapi()

        self.assertEqual(schema["info"]["title"], "Driver Monitoring API")
        self.assertEqual(
            set(schema["paths"]),
            {
                "/",
                "/health",
                "/health/database",
                "/drivers",
                "/vehicles",
                "/model-versions/active",
                "/sessions",
                "/sessions/{session_id}",
                "/sessions/{session_id}/complete",
                "/sessions/{session_id}/events",
                "/events",
                "/events/{event_id}/acknowledge",
                "/auth/google",
                "/auth/refresh",
                "/auth/logout",
                "/auth/me",
            },
        )


if __name__ == "__main__":
    unittest.main()
