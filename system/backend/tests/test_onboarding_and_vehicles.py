import unittest
from datetime import datetime

from httpx import ASGITransport, AsyncClient

from app.auth.dependencies import get_current_user
from app.auth.security import create_access_token
from app.database import get_db
from app.main import app
from tests.test_trip_lifecycle import FakeResult, QueueDatabase


def user_row(**overrides):
    row = {
        "user_id": 1,
        "google_sub": "google-sub-12345",
        "email": "driver.test@driverguardian.com",
        "display_name": "Test Driver",
        "avatar_url": None,
        "role": "DRIVER",
        "driver_id": 10,
        "is_active": "Y",
        "driver_code": "DRV0010",
        "driver_full_name": "Test Driver",
    }
    row.update(overrides)
    return row


class OnboardingAndVehiclesTest(unittest.IsolatedAsyncioTestCase):
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

    def authenticate_as(self, user_dict):
        def current_user_override():
            return user_dict

        app.dependency_overrides[get_current_user] = current_user_override

    async def test_unlinked_user_can_create_driver_profile_atomically(self):
        unlinked_user = user_row(driver_id=None, driver_code=None, driver_full_name=None)
        self.authenticate_as(unlinked_user)

        # Expected DB calls:
        # 1. Lock user row: returns user with driver_id=None
        # 2. Check duplicate license: returns count=0
        # 3. Next sequence val: returns 1002
        # 4. Check code exists: returns count=0
        # 5. Insert driver
        # 6. Get new driver_id: returns 55
        # 7. Update user
        # 8. Fetch updated user profile
        db = QueueDatabase([
            FakeResult(first={"user_id": 1, "driver_id": None, "role": "DRIVER"}),
            FakeResult(scalar=0),
            FakeResult(scalar=1002),
            FakeResult(scalar=0),
            FakeResult(),
            FakeResult(scalar=55),
            FakeResult(),
            FakeResult(first=user_row(driver_id=55, driver_code="DRV1002", driver_full_name="Tran Van B")),
        ])
        self.use_database(db)

        response = await self.client.post(
            "/drivers/me",
            json={
                "full_name": "Tran Van B",
                "phone_number": "0987654321",
                "license_number": "GPLX-998877",
            },
        )

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["driver"]["driver_id"], 55)
        self.assertEqual(data["driver"]["driver_code"], "DRV1002")
        self.assertEqual(data["driver"]["full_name"], "Tran Van B")
        self.assertEqual(db.commits, 1)

    async def test_cannot_create_second_driver_profile_for_same_user(self):
        linked_user = user_row(driver_id=10)
        self.authenticate_as(linked_user)

        db = QueueDatabase([
            FakeResult(first={"user_id": 1, "driver_id": 10, "role": "DRIVER"}),
        ])
        self.use_database(db)

        response = await self.client.post(
            "/drivers/me",
            json={
                "full_name": "Nguyen Van C",
                "phone_number": "0912345678",
                "license_number": "GPLX-112233",
            },
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("already linked", response.json()["detail"])
        self.assertEqual(db.rollbacks, 1)

    async def test_driver_profile_validation_rejects_empty_and_invalid(self):
        unlinked_user = user_row(driver_id=None)
        self.authenticate_as(unlinked_user)

        # Full name too short
        res1 = await self.client.post(
            "/drivers/me",
            json={"full_name": "A", "license_number": "GPLX-123"},
        )
        self.assertEqual(res1.status_code, 422)

        # License number missing
        res2 = await self.client.post(
            "/drivers/me",
            json={"full_name": "Nguyen Van A"},
        )
        self.assertEqual(res2.status_code, 422)

    async def test_duplicate_license_number_returns_409(self):
        unlinked_user = user_row(driver_id=None)
        self.authenticate_as(unlinked_user)

        db = QueueDatabase([
            FakeResult(first={"user_id": 1, "driver_id": None, "role": "DRIVER"}),
            FakeResult(scalar=1),  # Duplicate license exists
        ])
        self.use_database(db)

        response = await self.client.post(
            "/drivers/me",
            json={
                "full_name": "Tran Van D",
                "phone_number": "0981112233",
                "license_number": "GPLX-EXISTS",
            },
        )

        self.assertEqual(response.status_code, 409)
        self.assertIn("already registered", response.json()["detail"])

    async def test_get_my_driver_profile_returns_linked_details(self):
        linked_user = user_row(driver_id=15)
        self.authenticate_as(linked_user)

        db = QueueDatabase([
            FakeResult(first={
                "driver_id": 15,
                "driver_code": "DRV0015",
                "full_name": "Le Thi E",
                "phone_number": "0909090909",
                "license_number": "GPLX-556677",
                "status": "ACTIVE",
                "created_at": datetime(2026, 10, 1, 10, 0, 0),
            })
        ])
        self.use_database(db)

        response = await self.client.get("/drivers/me")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["driver_id"], 15)
        self.assertEqual(data["driver_code"], "DRV0015")
        self.assertEqual(data["full_name"], "Le Thi E")

    async def test_patch_my_driver_profile_updates_allowed_fields(self):
        linked_user = user_row(driver_id=15)
        self.authenticate_as(linked_user)

        db = QueueDatabase([
            FakeResult(scalar=0),  # License duplicate check -> 0
            FakeResult(),          # Update query
            FakeResult(),          # Update USERS display name query
            FakeResult(first={
                "driver_id": 15,
                "driver_code": "DRV0015",
                "full_name": "Le Thi E Updated",
                "phone_number": "0909090999",
                "license_number": "GPLX-NEW123",
                "status": "ACTIVE",
                "created_at": datetime(2026, 10, 1, 10, 0, 0),
            }),
        ])
        self.use_database(db)

        response = await self.client.patch(
            "/drivers/me",
            json={
                "full_name": "Le Thi E Updated",
                "phone_number": "0909090999",
                "license_number": "GPLX-NEW123",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["full_name"], "Le Thi E Updated")
        self.assertEqual(db.commits, 1)

    async def test_user_can_create_own_vehicle(self):
        linked_user = user_row(driver_id=10)
        self.authenticate_as(linked_user)

        db = QueueDatabase([
            FakeResult(scalar=0),  # Check plate duplicate -> 0
            FakeResult(),          # Insert vehicle
            FakeResult(first={
                "vehicle_id": 101,
                "plate_number": "29A-99999",
                "vehicle_name": "Hyundai Ioniq 5",
                "vehicle_type": "Electric SUV",
                "device_code": None,
                "status": "ACTIVE",
                "created_at": datetime(2026, 10, 2, 8, 0, 0),
                "driver_id": 10,
            }),
        ])
        self.use_database(db)

        response = await self.client.post(
            "/vehicles",
            json={
                "plate_number": "29A-99999",
                "vehicle_name": "Hyundai Ioniq 5",
                "vehicle_type": "Electric SUV",
            },
        )

        self.assertEqual(response.status_code, 201)
        data = response.json()
        self.assertEqual(data["plate_number"], "29A-99999")
        self.assertEqual(data["driver_id"], 10)
        self.assertEqual(db.commits, 1)

    async def test_unlinked_driver_cannot_create_vehicle(self):
        unlinked_user = user_row(driver_id=None)
        self.authenticate_as(unlinked_user)

        response = await self.client.post(
            "/vehicles",
            json={
                "plate_number": "30E-12345",
                "vehicle_name": "Toyota Camry",
                "vehicle_type": "Sedan",
            },
        )
        self.assertEqual(response.status_code, 403)

    async def test_get_vehicles_scopes_to_authenticated_driver(self):
        linked_user = user_row(driver_id=10)
        self.authenticate_as(linked_user)

        db = QueueDatabase([
            FakeResult(all_rows=[
                {
                    "vehicle_id": 101,
                    "plate_number": "29A-99999",
                    "vehicle_name": "Car A",
                    "vehicle_type": "SUV",
                    "device_code": None,
                    "status": "ACTIVE",
                    "created_at": datetime(2026, 10, 2, 8, 0, 0),
                    "driver_id": 10,
                },
                {
                    "vehicle_id": 102,
                    "plate_number": "29B-88888",
                    "vehicle_name": "Car B",
                    "vehicle_type": "Sedan",
                    "device_code": None,
                    "status": "ACTIVE",
                    "created_at": datetime(2026, 10, 2, 9, 0, 0),
                    "driver_id": 10,
                },
            ])
        ])
        self.use_database(db)

        response = await self.client.get("/vehicles")
        self.assertEqual(response.status_code, 200)
        vehicles = response.json()
        self.assertEqual(len(vehicles), 2)
        self.assertEqual(vehicles[0]["driver_id"], 10)
        self.assertEqual(vehicles[1]["driver_id"], 10)
        # Check query SQL filtered by DRIVER_ID
        self.assertIn("DRIVER_ID = :driver_id", db.calls[0][0])
        self.assertEqual(db.calls[0][1]["driver_id"], 10)

    async def test_driver_cannot_edit_other_driver_vehicle(self):
        # Authenticated as driver 10
        linked_user = user_row(driver_id=10)
        self.authenticate_as(linked_user)

        # Vehicle 200 belongs to driver 99
        db = QueueDatabase([
            FakeResult(first={
                "vehicle_id": 200,
                "plate_number": "51G-77777",
                "vehicle_name": "Other Driver Car",
                "vehicle_type": "Truck",
                "device_code": None,
                "status": "ACTIVE",
                "driver_id": 99,
            })
        ])
        self.use_database(db)

        response = await self.client.patch(
            "/vehicles/200",
            json={"vehicle_name": "Hacked Name"},
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("Access denied", response.json()["detail"])

    async def test_driver_cannot_use_other_driver_vehicle_for_session(self):
        # Driver 10 trying to create session with vehicle 200 which belongs to someone else
        linked_user = user_row(driver_id=10)
        self.authenticate_as(linked_user)

        db = QueueDatabase([
            FakeResult(one={
                "driver_count": 1,
                "vehicle_count": 1,
                "driver_vehicle_count": 0,  # Belongs to another driver!
                "model_count": 1,
            })
        ])
        self.use_database(db)

        response = await self.client.post(
            "/sessions",
            json={
                "driver_id": 10,
                "vehicle_id": 200,
                "model_version_id": 1,
            },
        )
        self.assertEqual(response.status_code, 403)
        self.assertIn("does not belong to the authenticated driver", response.json()["detail"])

    async def test_admin_can_view_all_vehicles(self):
        admin_user = user_row(user_id=2, role="ADMIN", driver_id=None)
        self.authenticate_as(admin_user)

        db = QueueDatabase([
            FakeResult(all_rows=[
                {
                    "vehicle_id": 1,
                    "plate_number": "51A-12345",
                    "vehicle_name": "VinFast VF 8",
                    "vehicle_type": "SUV",
                    "device_code": "DEVICE001",
                    "status": "ACTIVE",
                    "created_at": datetime(2026, 9, 28, 0, 0, 0),
                    "driver_id": 1,
                },
                {
                    "vehicle_id": 2,
                    "plate_number": "29A-99999",
                    "vehicle_name": "Hyundai Ioniq 5",
                    "vehicle_type": "SUV",
                    "device_code": None,
                    "status": "ACTIVE",
                    "created_at": datetime(2026, 10, 2, 8, 0, 0),
                    "driver_id": 10,
                },
            ])
        ])
        self.use_database(db)

        response = await self.client.get("/vehicles")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()), 2)
        # Verify query SQL does NOT filter by DRIVER_ID
        self.assertNotIn("DRIVER_ID = :driver_id", db.calls[0][0])
