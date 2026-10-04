-- Recovered verbatim from local SQL Developer history.
-- History source: 5074896135868636069history.xml
-- SQL Developer recorded executed=1 on connection "Driver Monitoring DB".

INSERT INTO DRIVERS (
    DRIVER_CODE,
    FULL_NAME,
    PHONE_NUMBER,
    LICENSE_NUMBER
)
VALUES (
    'DRV001',
    'Nguyen Van An',
    '0901234567',
    'GPLX001'
);

INSERT INTO VEHICLES (
    PLATE_NUMBER,
    VEHICLE_NAME,
    VEHICLE_TYPE,
    DEVICE_CODE
)
VALUES (
    '51A-12345',
    'VinFast VF 8',
    'SUV',
    'DEVICE001'
);

INSERT INTO MODEL_VERSIONS (
    VERSION_NAME,
    MODEL_TYPE,
    FILE_NAME,
    DESCRIPTION
)
VALUES (
    'v1.0.0',
    'DROWSINESS_DETECTION',
    'drowsiness_model.onnx',
    'Mo hinh phat hien buon ngu phien ban dau tien'
);

COMMIT;
