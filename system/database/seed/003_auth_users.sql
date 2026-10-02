-- Seed: 003_auth_users.sql
-- Description: Pre-provisions sample authenticated users linked to existing reference drivers.

-- Pre-provisioned user linked to driver DRV001 (Nguyen Van An).
-- GOOGLE_SUB is null initially; it will automatically bind upon first verified Google login with this email.
INSERT INTO USERS (
    EMAIL,
    DISPLAY_NAME,
    ROLE,
    DRIVER_ID,
    IS_ACTIVE
)
SELECT
    'driver.an@driverguardian.com',
    'Nguyen Van An',
    'DRIVER',
    D.DRIVER_ID,
    'Y'
FROM DRIVERS D
WHERE D.DRIVER_CODE = 'DRV001';

-- Pre-provisioned administrator user (unlinked to any driver)
INSERT INTO USERS (
    EMAIL,
    DISPLAY_NAME,
    ROLE,
    DRIVER_ID,
    IS_ACTIVE
)
VALUES (
    'admin@driverguardian.com',
    'System Administrator',
    'ADMIN',
    NULL,
    'Y'
);

COMMIT;
