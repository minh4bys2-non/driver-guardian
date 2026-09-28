# Driver Guardian Oracle Database Artifacts

This directory preserves the Oracle schema and seed data recovered from local
SQL Developer history. It is intentionally separate from the backend runtime
so the repository can be installed and unit-tested without a running Oracle
instance.

## Provenance

The SQL below was recovered from local SQL Developer history entries marked
`executed=1`. That flag proves the statement was executed in the historical
local environment; it does not prove that the same database is currently
running or reachable.

| Repository file | Recovered SQL Developer history | Historical connection |
| --- | --- | --- |
| `schema/001_driver_guardian_schema.sql` | `5670336839559678953history.xml` | `Driver Monitoring DB` |
| `seed/001_reference_data.sql` | `5074896135868636069history.xml` | `Driver Monitoring DB` |
| `seed/002_completed_session.sql` | `4326643575054547632history.xml` | `Driver Monitoring DB` |
| `recovered/driver_data_tablespace.sql` | `546370717546548924history.xml` | `Oracle Local` |

No original `CREATE USER` artifact was found, so none has been invented or
added here.

## Schema

`schema/001_driver_guardian_schema.sql` defines the tables used by the
recovered backend:

- `DRIVERS`
- `VEHICLES`
- `MODEL_VERSIONS`
- `DRIVING_SESSIONS`
- `DROWSINESS_EVENTS`
- `ALERT_ACTIONS`

It also preserves the recovered constraints and indexes. The application
expects an Oracle user that can access these objects, but provisioning that
user is outside the recovered evidence.

## Seed data

Apply seed files in numeric order after the schema:

1. `seed/001_reference_data.sql` preserves the recovered driver, vehicle, and
   model-version records exactly; no new reference records were created.
2. `seed/002_completed_session.sql` creates the recovered completed-session
   example. It does not hard-code identity column values. Instead, it resolves
   foreign keys through the recovered natural keys `DRV001`, `DEVICE001`, and
   `v1.0.0`, so `001_reference_data.sql` is an exact prerequisite.

The seed scripts are recovered examples, not an idempotent migration system.
Reapplying them can conflict with existing unique data.

## Recovered non-portable artifact

`recovered/driver_data_tablespace.sql` is **NON-PORTABLE / REFERENCE ONLY**.
It contains the machine/container-specific datafile path:

`/opt/oracle/oradata/FREE/FREEPDB1/driver_data01.dbf`

Do not use that script as the default installation path. Confirm the target
Oracle storage layout and create an environment-specific provisioning script
outside this recovered reference before using it.

## Suggested application order

For an already provisioned compatible Oracle user:

1. Review `schema/001_driver_guardian_schema.sql` for the target Oracle version.
2. Apply the schema script.
3. Optionally apply `seed/001_reference_data.sql`.
4. Optionally apply `seed/002_completed_session.sql` after its prerequisite.

The tablespace reference is deliberately excluded from this sequence.

## Not verified in this phase

- No Oracle user or privilege provisioning was recovered.
- No portable tablespace provisioning was recovered.
- No Docker, Podman, WSL, or Oracle container was started.
- The SQL was not applied to a live Oracle instance in this phase.
- The project does not yet include a database migration runner.
