-- NON-PORTABLE / REFERENCE ONLY
-- Recovered verbatim from local SQL Developer history.
-- History source: 546370717546548924history.xml
-- SQL Developer recorded executed=1 on connection "Oracle Local".
-- The datafile path is specific to the original Oracle Free container.
-- Do not use this as the default installation script.

CREATE TABLESPACE DRIVER_DATA
DATAFILE '/opt/oracle/oradata/FREE/FREEPDB1/driver_data01.dbf'
SIZE 200M
AUTOEXTEND ON NEXT 50M
MAXSIZE 2G;

ALTER USER DRIVER_APP
DEFAULT TABLESPACE DRIVER_DATA;

ALTER USER DRIVER_APP
QUOTA UNLIMITED ON DRIVER_DATA;
