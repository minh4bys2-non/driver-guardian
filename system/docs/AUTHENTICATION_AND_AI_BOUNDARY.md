# Driver Guardian System: Authentication Architecture & AI Boundary

## 1. Authentication Architecture

Driver Guardian uses the modern Android Credential Manager architecture integrated with a FastAPI backend and Oracle database.

```
[ Android Client ]
       │
       ▼ (Android Credential Manager)
[ Google Sign-In ]
       │ (Google ID Token)
       ▼
[ POST /auth/google ] ───► [ FastAPI Backend ]
                                  │
                                  ├─► Verify Google Token (Issuer, Audience, Signature, Exp)
                                  ├─► Lookup / Bind USERS table in Oracle
                                  ├─► Enforce Active Status & Driver Link
                                  └─► Generate Driver Guardian Tokens:
                                        ├── Access Token (Short-lived JWT: 30 mins)
                                        └── Refresh Token (Hashed in AUTH_REFRESH_TOKENS: 30 days)
       │
       ▼ (EncryptedSharedPreferences / Android Keystore)
[ Local Session Store ]
       │
       ▼ (AuthInterceptor: Authorization: Bearer <token>)
[ Protected Endpoints ]
       ├── GET  /auth/me
       ├── POST /sessions (enforces USERS.DRIVER_ID for DRIVER role)
       ├── POST /events (enforces session driver ownership)
       ├── POST /events/{id}/acknowledge
       └── POST /sessions/{id}/complete
```

### Key Security Principles

1. **Google Token is Only for Identity Proof**: Google ID Token is never used as the permanent bearer token for Driver Guardian APIs. The backend verifies the token and issues its own authenticated session.
2. **Access Token & Refresh Token**:
   - Access token: short-lived JWT containing only essential claims (`sub` = `user_id`, `role`, `exp`, `iat`).
   - Refresh token: cryptographically secure random token stored securely as a SHA-256 hash in Oracle (`AUTH_REFRESH_TOKENS`). Raw refresh tokens are never stored in the database.
   - Refresh tokens support rotation, server-side revocation on `/auth/logout`, and expiration.
3. **Android Keystore-backed Storage**:
   - Android client stores session credentials using `EncryptedSharedPreferences` backed by the Android Keystore with AES-256-GCM encryption.
4. **Automatic Token Refresh with Recursion Prevention**:
   - `AuthInterceptor` intercepts HTTP 401 on protected routes, uses a synchronized lock to prevent parallel refresh storms, calls `/auth/refresh` once, updates local tokens, and retries the original request.
   - If refresh fails, the local session is cleared and the user is redirected to `LoginScreen`.

---

## 2. Boundary: USER ≠ DRIVER

Driver Guardian strictly decouples user identity from domain driver profiles:

| Entity | Purpose | Storage |
|---|---|---|
| **USER** | Authenticated account, authentication identity, system role | `USERS` table |
| **DRIVER** | Business profile, driver license, vehicle assignment, trip history | `DRIVERS` table |

### Mapping Rules:
- `USERS.DRIVER_ID` is a nullable foreign key pointing to `DRIVERS.DRIVER_ID`.
- For `ROLE = 'DRIVER'`, the authenticated account determines the driver profile. Client-provided `driver_id` in request payloads is ignored or validated to match the user's bound `DRIVER_ID`.
- **Pre-provisioned Accounts**: An administrator creates a user with `EMAIL` and `DRIVER_ID`. On first Google login, the verified `GOOGLE_SUB` is bound to the record.
- **Unlinked Driver State**: If a user logs in with a Google account that has no linked `DRIVER_ID`, the user profile returns `driver: null`. The Android application detects this and displays a safe blocked state ("Tài khoản chưa được liên kết với hồ sơ tài xế"), preventing invalid trip creation until linked by an administrator.
- **Cross-Driver Access Denial**: Endpoints (`/sessions`, `/sessions/{id}`, `/sessions/{id}/complete`, `/events`, `/events/{id}/acknowledge`) verify driver ownership and return HTTP 403 Forbidden if a driver attempts to access another driver's sessions or events.

---

## 3. Boundary: MediaPipe Landmark Model ≠ Drowsiness ONNX Model

It is critical to distinguish between the facial landmark detection asset and the future neural drowsiness classifier:

| Aspect | MediaPipe Face Landmarker | Drowsiness ONNX Model (Future) |
|---|---|---|
| **Asset** | `face_landmarker.task` | `spatial_extractor.onnx` + `temporal_classifier.onnx` |
| **Function** | Detects 478 3D facial landmarks from camera stream | Classifies driver drowsiness state (Normal vs. Drowsy) |
| **Architecture** | Google MediaPipe Tasks Vision (BlazeFace + Mesh) | Spatial CNN/PAFPN (448 features) + 3-layer LSTM |
| **Current Status** | **VERIFIED & OPERATIONAL** in current build | **FUTURE INTEGRATION** (No trained weights artifact yet) |
| **Classification** | Does NOT classify drowsiness | Will output drowsiness class probabilities |

### Current Operational Monitoring Architecture:
```
CameraX
  └─► MediaPipe Face Landmarker (face_landmarker.task)
        └─► PhysicalBranchProcessor
              └─► PhysicalMetrics (EAR, MAR, PERCLOS, POM, Head Pose, Nodding)

Rotation Vector Sensor
  └─► DeviceMotionProcessor
        └─► DeviceMotionMetrics (Pitch, Roll, Yaw, Device Tilt)

PhysicalMetrics + DeviceMotionMetrics
  └─► MonitoringSnapshot (Diagnostics display only)
```

### DEMO Danger Alert Workflow:
- The danger alert flow in the current application is triggered **exclusively** via the explicit DEMO button:
  `DEMO: Cảnh báo nguy hiểm`.
- Physical metrics and sensor values do **not** automatically trigger WARNING or DANGER states.
- There is **no fake AI confidence** or simulated classification.

---

## 4. Database Auth Schema (Oracle DDL)

Auth tables are added via safe migrations without dropping existing data:
- `system/database/schema/002_auth_schema.sql`: defines `USERS` and `AUTH_REFRESH_TOKENS` tables, indexes, and constraints.
- `system/database/seed/003_auth_users.sql`: seeds pre-provisioned demo accounts (`driver.an@driverguardian.com` linked to `DRV001`).
