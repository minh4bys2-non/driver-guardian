import unittest
from datetime import datetime, timedelta, timezone

from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user, get_google_verifier
from app.auth.google_verifier import GoogleTokenVerifier, GoogleUserPayload
from app.auth.security import create_access_token, generate_refresh_token, hash_token
from app.database import get_db
from app.main import app
from tests.test_trip_lifecycle import FakeResult, QueueDatabase


class MockGoogleVerifier:
    def __init__(self, payload: GoogleUserPayload | None = None, raise_error: Exception | None = None):
        self.payload = payload
        self.raise_error = raise_error

    def verify(self, id_token_string: str) -> GoogleUserPayload:
        if self.raise_error is not None:
            raise self.raise_error
        if self.payload is not None:
            return self.payload
        raise ValueError("Mock verifier not configured")


def user_row(**overrides):
    row = {
        "user_id": 1,
        "google_sub": "google-sub-12345",
        "email": "driver.an@driverguardian.com",
        "display_name": "Nguyen Van An",
        "avatar_url": "https://example.com/avatar.jpg",
        "role": "DRIVER",
        "driver_id": 1,
        "is_active": "Y",
        "driver_code": "DRV001",
        "driver_full_name": "Nguyen Van An",
    }
    row.update(overrides)
    return row


class BackendAuthTest(unittest.IsolatedAsyncioTestCase):
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

    def use_google_verifier(self, verifier: GoogleTokenVerifier):
        app.dependency_overrides[get_google_verifier] = lambda: verifier

    # 1. Google token verification failures
    async def test_google_login_invalid_token_returns_401(self):
        self.use_google_verifier(MockGoogleVerifier(raise_error=ValueError("Token expired or invalid")))
        response = await self.client.post("/auth/google", json={"id_token": "invalid-token-string"})
        self.assertEqual(response.status_code, 401)
        self.assertIn("Token expired or invalid", response.json()["detail"])

    # 2. Existing user by Google sub logs in successfully
    async def test_google_login_existing_user_by_sub(self):
        mock_payload = GoogleUserPayload(
            sub="sub-existing-123",
            email="driver.an@driverguardian.com",
            email_verified=True,
            name="Nguyen Van An",
            picture="https://example.com/an.jpg",
        )
        self.use_google_verifier(MockGoogleVerifier(payload=mock_payload))
        database = QueueDatabase([
            FakeResult(first=user_row(google_sub="sub-existing-123")),
            FakeResult(rowcount=1),  # update last login
            FakeResult(rowcount=1),  # insert refresh token
        ])
        self.use_database(database)

        response = await self.client.post("/auth/google", json={"id_token": "valid-token-xyz"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("access_token", body)
        self.assertIn("refresh_token", body)
        self.assertEqual(body["user"]["email"], "driver.an@driverguardian.com")
        self.assertEqual(body["user"]["role"], "DRIVER")
        self.assertEqual(body["user"]["driver"]["driver_id"], 1)
        self.assertEqual(database.commits, 1)

    # 3. Pre-provisioned user binds Google sub on first login
    async def test_google_login_binds_sub_to_pre_provisioned_email(self):
        mock_payload = GoogleUserPayload(
            sub="new-google-sub-789",
            email="driver.an@driverguardian.com",
            email_verified=True,
            name="Nguyen Van An",
            picture="https://example.com/an.jpg",
        )
        self.use_google_verifier(MockGoogleVerifier(payload=mock_payload))
        database = QueueDatabase([
            FakeResult(first=None),  # not found by google_sub
            FakeResult(first=user_row(google_sub=None)),  # found by email
            FakeResult(rowcount=1),  # update to bind sub
            FakeResult(first=user_row(google_sub="new-google-sub-789")),  # re-fetch bound user
            FakeResult(rowcount=1),  # insert refresh token
        ])
        self.use_database(database)

        response = await self.client.post("/auth/google", json={"id_token": "valid-token-bind"})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user"]["email"], "driver.an@driverguardian.com")
        # Check bind update query was called
        bind_call = database.calls[2]
        self.assertIn("UPDATE USERS", bind_call[0].upper())
        self.assertEqual(bind_call[1]["google_sub"], "new-google-sub-789")

    # 4. Inactive user is rejected with 403
    async def test_google_login_inactive_user_rejected(self):
        mock_payload = GoogleUserPayload(
            sub="sub-inactive-999",
            email="inactive@driverguardian.com",
            email_verified=True,
        )
        self.use_google_verifier(MockGoogleVerifier(payload=mock_payload))
        database = QueueDatabase([
            FakeResult(first=user_row(is_active="N")),
            FakeResult(rowcount=1),
        ])
        self.use_database(database)

        response = await self.client.post("/auth/google", json={"id_token": "valid-token-inactive"})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["detail"], "Account is inactive")
        self.assertEqual(database.rollbacks, 1)

    # 5. Protected route without token returns 401
    async def test_protected_route_without_token_returns_401(self):
        response = await self.client.get("/auth/me")
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Authentication required")

    # 6. /auth/me returns profile for authenticated user
    async def test_auth_me_returns_profile(self):
        valid_token = create_access_token(user_id=1, role="DRIVER")
        database = QueueDatabase([
            FakeResult(first=user_row()),
        ])
        self.use_database(database)

        response = await self.client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {valid_token}"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["user_id"], 1)
        self.assertEqual(body["email"], "driver.an@driverguardian.com")
        self.assertEqual(body["driver"]["driver_id"], 1)

    # 7. Token refresh rotation
    async def test_refresh_token_rotation_succeeds(self):
        raw_refresh = "valid-refresh-token-sample"
        future_time = datetime.now(timezone.utc) + timedelta(days=7)
        database = QueueDatabase([
            FakeResult(first={
                "token_id": 101,
                "user_id": 1,
                "revoked_at": None,
                "expires_at": future_time,
                "role": "DRIVER",
                "is_active": "Y",
                "email": "driver.an@driverguardian.com",
                "display_name": "Nguyen Van An",
                "avatar_url": None,
                "driver_id": 1,
                "driver_code": "DRV001",
                "driver_full_name": "Nguyen Van An",
            }),
            FakeResult(scalar=0),  # is_expired == 0
            FakeResult(rowcount=1),  # revoke old token
            FakeResult(rowcount=1),  # insert new refresh token
        ])
        self.use_database(database)

        response = await self.client.post("/auth/refresh", json={"refresh_token": raw_refresh})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("access_token", body)
        self.assertIn("refresh_token", body)
        self.assertNotEqual(body["refresh_token"], raw_refresh)
        self.assertEqual(database.commits, 1)

    # 8. Refresh revoked token fails with 401
    async def test_refresh_revoked_token_fails(self):
        database = QueueDatabase([
            FakeResult(first={
                "token_id": 101,
                "user_id": 1,
                "revoked_at": datetime.now(timezone.utc),
                "expires_at": datetime.now(timezone.utc) + timedelta(days=7),
                "role": "DRIVER",
                "is_active": "Y",
                "email": "driver.an@driverguardian.com",
                "display_name": "Nguyen Van An",
                "avatar_url": None,
                "driver_id": 1,
                "driver_code": "DRV001",
                "driver_full_name": "Nguyen Van An",
            }),
        ])
        self.use_database(database)

        response = await self.client.post("/auth/refresh", json={"refresh_token": "revoked-token"})
        self.assertEqual(response.status_code, 401)
        self.assertEqual(response.json()["detail"], "Refresh token has been revoked")

    # 9. Logout revokes token
    async def test_logout_revokes_token(self):
        valid_token = create_access_token(user_id=1, role="DRIVER")
        database = QueueDatabase([
            FakeResult(first=user_row()),  # get_current_user
            FakeResult(rowcount=1),  # revoke refresh token
        ])
        self.use_database(database)

        response = await self.client.post(
            "/auth/logout",
            headers={"Authorization": f"Bearer {valid_token}"},
            json={"refresh_token": "some-token-to-logout"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"message": "Logged out successfully"})
        self.assertEqual(database.commits, 1)

    # 10. Session creation enforces authenticated DRIVER_ID
    async def test_session_creation_uses_authenticated_driver_id(self):
        app.dependency_overrides[get_current_user] = lambda: user_row(driver_id=5)
        database = QueueDatabase([
            FakeResult(one={"driver_count": 1, "vehicle_count": 1, "model_count": 1}),
            FakeResult(one={
                "session_id": 55,
                "driver_id": 5,
                "vehicle_id": 2,
                "model_version_id": 1,
                "start_time": datetime.now(),
                "end_time": None,
                "duration_seconds": 0,
                "total_alerts": 0,
                "safety_score": None,
                "status": "ACTIVE",
                "sync_status": "SYNCED",
            }),
        ])
        self.use_database(database)

        # Client tries to send driver_id=999, but authenticated user has driver_id=5
        response = await self.client.post(
            "/sessions",
            headers={"Authorization": "Bearer mock-token"},
            json={
                "driver_id": 999,
                "vehicle_id": 2,
                "model_version_id": 1,
            },
        )
        self.assertEqual(response.status_code, 201)
        # Validation query must have used driver_id=5
        validation_params = database.calls[0][1]
        self.assertEqual(validation_params["driver_id"], 5)

    # 11. Unlinked driver account cannot create session
    async def test_unlinked_driver_cannot_create_session(self):
        app.dependency_overrides[get_current_user] = lambda: user_row(driver_id=None)
        response = await self.client.post(
            "/sessions",
            headers={"Authorization": "Bearer mock-token"},
            json={"driver_id": 1, "vehicle_id": 2, "model_version_id": 1},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("not linked to any driver profile", response.json()["detail"])

    # 12. Driver cannot access another driver's session detail
    async def test_driver_cannot_access_other_driver_session(self):
        app.dependency_overrides[get_current_user] = lambda: user_row(driver_id=1)
        database = QueueDatabase([
            FakeResult(first={
                "session_id": 88,
                "driver_id": 2,  # Belongs to driver 2
                "driver_name": "Other Driver",
                "vehicle_id": 1,
                "vehicle_name": None,
                "plate_number": "51A-999.99",
                "model_version_id": 1,
                "version_name": "v1",
                "start_time": datetime.now(),
                "end_time": None,
                "duration_seconds": 0,
                "total_alerts": 0,
                "safety_score": None,
                "status": "ACTIVE",
                "sync_status": "SYNCED",
            }),
        ])
        self.use_database(database)

        response = await self.client.get(
            "/sessions/88",
            headers={"Authorization": "Bearer mock-token"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
