import os
import unittest
from datetime import datetime

from httpx import ASGITransport, AsyncClient


os.environ.setdefault("DB_USER", "test_user")
os.environ.setdefault("DB_PASSWORD", "test_password")
os.environ.setdefault("DB_HOST", "127.0.0.1")
os.environ.setdefault("DB_PORT", "1521")
os.environ.setdefault("DB_SERVICE", "FREEPDB1")

from app.database import get_db
from app.main import app


START_TIME = datetime(2026, 9, 28, 8, 0, 0)
END_TIME = datetime(2026, 9, 28, 9, 2, 3)


def session_row(**overrides):
    row = {
        "session_id": 42,
        "driver_id": 1,
        "driver_name": "Lan",
        "vehicle_id": 2,
        "vehicle_name": None,
        "plate_number": "51A-123.45",
        "model_version_id": 3,
        "version_name": "drowsy-v1",
        "start_time": START_TIME,
        "end_time": END_TIME,
        "duration_seconds": 3723,
        "total_alerts": 2,
        "safety_score": None,
        "status": "COMPLETED",
        "sync_status": "SYNCED",
    }
    row.update(overrides)
    return row


def event_row(**overrides):
    row = {
        "event_id": 9,
        "session_id": 42,
        "event_time": datetime(2026, 9, 28, 8, 30, 0),
        "driver_state": "DANGER",
        "alert_level": 2,
        "confidence": None,
        "drowsiness_score": None,
        "ear_value": None,
        "mar_value": None,
        "head_pose": None,
        "duration_ms": None,
        "acknowledged": "N",
        "sync_status": "SYNCED",
    }
    row.update(overrides)
    return row


class FakeResult:
    def __init__(self, *, one=None, first=None, all_rows=None, scalar=None, rowcount=1):
        self._one = one
        self._first = first
        self._all = all_rows
        self._scalar = scalar
        self.rowcount = rowcount

    def mappings(self):
        return self

    def one(self):
        return self._one

    def first(self):
        return self._first

    def all(self):
        return self._all if self._all is not None else []

    def scalar_one(self):
        return self._scalar


class QueueDatabase:
    def __init__(self, results):
        self.results = list(results)
        self.calls = []
        self.commits = 0
        self.rollbacks = 0

    def execute(self, statement, parameters=None):
        self.calls.append((str(statement), parameters or {}))
        return self.results.pop(0)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


class TripLifecycleTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client = AsyncClient(
            transport=ASGITransport(app=app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self):
        app.dependency_overrides.clear()
        await self.client.aclose()

    def use_database(self, database):
        def dependency():
            yield database

        app.dependency_overrides[get_db] = dependency

    async def test_complete_session_uses_one_database_timestamp(self):
        database = QueueDatabase([
            FakeResult(first={"session_id": 42, "status": "ACTIVE"}),
            FakeResult(scalar=END_TIME),
            FakeResult(rowcount=1),
            FakeResult(first=session_row()),
        ])
        self.use_database(database)

        response = await self.client.post("/sessions/42/complete")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "COMPLETED")
        self.assertEqual(response.json()["end_time"], "2026-09-28T09:02:03")
        self.assertIsNone(response.json()["safety_score"])
        update_sql, update_parameters = database.calls[2]
        self.assertGreaterEqual(update_sql.lower().count(":completion_time"), 2)
        self.assertEqual(update_parameters["completion_time"], END_TIME)
        self.assertEqual(database.commits, 1)

    async def test_complete_missing_session_returns_404(self):
        database = QueueDatabase([FakeResult(first=None)])
        self.use_database(database)
        response = await self.client.post("/sessions/404/complete")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Driving session not found"})
        self.assertEqual(database.rollbacks, 1)

    async def test_complete_non_active_session_returns_409(self):
        database = QueueDatabase([FakeResult(first={"session_id": 42, "status": "COMPLETED"})])
        self.use_database(database)
        response = await self.client.post("/sessions/42/complete")
        self.assertEqual(response.status_code, 409)
        self.assertEqual(response.json(), {"detail": "Driving session is not active"})

    async def test_history_maps_joined_rows_and_filters(self):
        database = QueueDatabase([FakeResult(all_rows=[session_row()])])
        self.use_database(database)
        response = await self.client.get("/sessions?driver_id=1&status=COMPLETED&limit=25")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body[0]["driver_name"], "Lan")
        self.assertIsNone(body[0]["vehicle_name"])
        self.assertIsNone(body[0]["safety_score"])
        self.assertEqual(database.calls[0][1], {"driver_id": 1, "status": "COMPLETED", "limit": 25})

    async def test_session_detail_missing_returns_404(self):
        database = QueueDatabase([FakeResult(first=None)])
        self.use_database(database)
        response = await self.client.get("/sessions/777")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Driving session not found"})

    async def test_session_events_preserve_null_values(self):
        database = QueueDatabase([
            FakeResult(scalar=1),
            FakeResult(all_rows=[event_row()]),
        ])
        self.use_database(database)
        response = await self.client.get("/sessions/42/events")
        self.assertEqual(response.status_code, 200)
        body = response.json()[0]
        for field in ("confidence", "drowsiness_score", "ear_value", "mar_value", "head_pose", "duration_ms"):
            self.assertIsNone(body[field])
        self.assertIn("ORDER BY", database.calls[1][0].upper())

    async def test_events_for_missing_session_returns_404(self):
        database = QueueDatabase([FakeResult(scalar=0)])
        self.use_database(database)
        response = await self.client.get("/sessions/404/events")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json(), {"detail": "Driving session not found"})

    async def test_acknowledge_is_idempotent_and_inserts_one_confirmation(self):
        database = QueueDatabase([
            FakeResult(first={"event_id": 9, "acknowledged": "N"}),
            FakeResult(scalar=0),
            FakeResult(rowcount=1),
            FakeResult(rowcount=1),
            FakeResult(first=event_row(acknowledged="Y")),
            FakeResult(first={"event_id": 9, "acknowledged": "Y"}),
            FakeResult(scalar=1),
            FakeResult(first=event_row(acknowledged="Y")),
        ])
        self.use_database(database)

        first = await self.client.post("/events/9/acknowledge")
        second = await self.client.post("/events/9/acknowledge")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 200)
        self.assertEqual(first.json()["acknowledged"], "Y")
        inserts = [sql for sql, _ in database.calls if "INSERT INTO ALERT_ACTIONS" in sql.upper()]
        self.assertEqual(len(inserts), 1)
        self.assertEqual(database.commits, 2)

    async def test_create_event_locks_session_until_event_transaction_commits(self):
        database = QueueDatabase([
            FakeResult(first={"session_id": 42, "status": "ACTIVE"}),
            FakeResult(one=event_row()),
            FakeResult(rowcount=1),
        ])
        self.use_database(database)

        response = await self.client.post(
            "/events",
            json={
                "session_id": 42,
                "driver_state": "DANGER",
                "alert_level": 2,
                "confidence": None,
                "duration_ms": None,
            },
        )

        self.assertEqual(response.status_code, 201)
        lock_sql = database.calls[0][0].upper()
        self.assertIn("FOR UPDATE", lock_sql)
        self.assertIn("STATUS", lock_sql)
        self.assertEqual(database.commits, 1)


if __name__ == "__main__":
    unittest.main()
