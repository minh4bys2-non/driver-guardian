-- Migration: 003_driver_onboarding_and_vehicles.sql
-- Description: Adds DRIVER_ID foreign key to VEHICLES, links existing reference vehicle 51A-12345 to DRV001,
-- and creates DRIVER_CODE_SEQ sequence initialized safely based on existing DRIVER_CODE records.
-- Preserves all historical DRIVING_SESSIONS, VEHICLES, DRIVERS, and USERS data.

-- 1. Add DRIVER_ID column to VEHICLES if it does not already exist
DECLARE
    v_col_count NUMBER := 0;
BEGIN
    SELECT COUNT(*) INTO v_col_count
    FROM USER_TAB_COLS
    WHERE TABLE_NAME = 'VEHICLES' AND COLUMN_NAME = 'DRIVER_ID';

    IF v_col_count = 0 THEN
        EXECUTE IMMEDIATE 'ALTER TABLE VEHICLES ADD (DRIVER_ID NUMBER)';
    END IF;
END;
/

-- 2. Add foreign key constraint FK_VEHICLES_DRIVER if not already present
DECLARE
    v_fk_count NUMBER := 0;
BEGIN
    SELECT COUNT(*) INTO v_fk_count
    FROM USER_CONSTRAINTS
    WHERE CONSTRAINT_NAME = 'FK_VEHICLES_DRIVER';

    IF v_fk_count = 0 THEN
        EXECUTE IMMEDIATE 'ALTER TABLE VEHICLES ADD CONSTRAINT FK_VEHICLES_DRIVER FOREIGN KEY (DRIVER_ID) REFERENCES DRIVERS (DRIVER_ID)';
    END IF;
END;
/

-- 3. Add index on VEHICLES(DRIVER_ID) if not already present
DECLARE
    v_idx_count NUMBER := 0;
BEGIN
    SELECT COUNT(*) INTO v_idx_count
    FROM USER_INDEXES
    WHERE INDEX_NAME = 'IDX_VEHICLES_DRIVER_ID';

    IF v_idx_count = 0 THEN
        EXECUTE IMMEDIATE 'CREATE INDEX IDX_VEHICLES_DRIVER_ID ON VEHICLES (DRIVER_ID)';
    END IF;
END;
/

-- 4. Safely migrate existing reference vehicle (51A-12345) to reference driver (DRV001)
UPDATE VEHICLES V
SET V.DRIVER_ID = (
    SELECT D.DRIVER_ID
    FROM DRIVERS D
    WHERE D.DRIVER_CODE = 'DRV001'
)
WHERE V.PLATE_NUMBER = '51A-12345'
  AND V.DRIVER_ID IS NULL
/

-- 5. Create DRIVER_CODE_SEQ sequence safely relative to existing DRIVER_CODE values
DECLARE
    v_seq_count NUMBER := 0;
    v_max_code NUMBER := 0;
    v_start_val NUMBER := 2;
    v_sql VARCHAR2(500);
BEGIN
    SELECT COUNT(*) INTO v_seq_count
    FROM USER_SEQUENCES
    WHERE SEQUENCE_NAME = 'DRIVER_CODE_SEQ';

    IF v_seq_count = 0 THEN
        SELECT NVL(MAX(TO_NUMBER(REGEXP_SUBSTR(DRIVER_CODE, '\d+'))), 0)
        INTO v_max_code
        FROM DRIVERS;

        v_start_val := v_max_code + 1;
        IF v_start_val < 2 THEN
            v_start_val := 2;
        END IF;

        v_sql := 'CREATE SEQUENCE DRIVER_CODE_SEQ START WITH ' || v_start_val || ' INCREMENT BY 1 NOCACHE';
        EXECUTE IMMEDIATE v_sql;
    END IF;
END;
/

COMMIT
/
