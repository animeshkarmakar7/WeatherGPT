#!/bin/sh
set -eu

pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" >/dev/null

psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -v ON_ERROR_STOP=1 -tAc '
SELECT 1
WHERE EXISTS (
    SELECT 1 FROM pg_extension WHERE extname = ''timescaledb''
)
AND EXISTS (
    SELECT 1 FROM pg_extension WHERE extname = ''postgis''
)
AND EXISTS (
    SELECT 1
    FROM information_schema.tables
    WHERE table_schema = ''public''
      AND table_name = ''weather_observations''
)
AND EXISTS (
    SELECT 1
    FROM timescaledb_information.hypertables
    WHERE hypertable_name = ''weather_observations''
);
' | grep -qx '1'
