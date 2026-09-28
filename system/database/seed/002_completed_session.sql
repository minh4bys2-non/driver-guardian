-- Recovered verbatim from local SQL Developer history.
-- History source: 4326643575054547632history.xml
-- SQL Developer recorded executed=1 on connection "Driver Monitoring DB".
-- Prerequisite: run 001_reference_data.sql first. This script resolves
-- identity values by stable recovered business keys; it does not hard-code IDs.

INSERT INTO DRIVING_SESSIONS (
    DRIVER_ID,
    VEHICLE_ID,
    MODEL_VERSION_ID,
    START_TIME,
    END_TIME,
    DURATION_SECONDS,
    TOTAL_ALERTS,
    SAFETY_SCORE,
    STATUS,
    SYNC_STATUS
)
SELECT
    d.DRIVER_ID,
    v.VEHICLE_ID,
    m.MODEL_VERSION_ID,
    SYSTIMESTAMP - INTERVAL '45' MINUTE,
    SYSTIMESTAMP,
    2700,
    2,
    88.5,
    'COMPLETED',
    'SYNCED'
FROM DRIVERS d
CROSS JOIN VEHICLES v
CROSS JOIN MODEL_VERSIONS m
WHERE d.DRIVER_CODE = 'DRV001'
  AND v.DEVICE_CODE = 'DEVICE001'
  AND m.VERSION_NAME = 'v1.0.0';

COMMIT;
