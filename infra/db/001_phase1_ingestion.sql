\set ON_ERROR_STOP on

CREATE EXTENSION IF NOT EXISTS timescaledb;
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    connector TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,
    records_fetched INTEGER NOT NULL DEFAULT 0,
    records_published INTEGER NOT NULL DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS weather_observations (
    observed_at TIMESTAMPTZ NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    geom GEOGRAPHY(POINT, 4326) GENERATED ALWAYS AS (
        ST_SetSRID(ST_MakePoint(longitude, latitude), 4326)::geography
    ) STORED,
    temp_c DOUBLE PRECISION,
    wind_speed_kph DOUBLE PRECISION,
    wind_direction_deg DOUBLE PRECISION,
    humidity_pct DOUBLE PRECISION,
    precipitation_mm DOUBLE PRECISION,
    pressure_hpa DOUBLE PRECISION,
    weather_code TEXT,
    quality_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
    provenance JSONB NOT NULL,
    raw_payload JSONB NOT NULL,
    PRIMARY KEY (observed_at, source, external_id)
);

SELECT create_hypertable('weather_observations', 'observed_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_weather_observations_location_time
    ON weather_observations (location_name, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_weather_observations_geom
    ON weather_observations USING GIST (geom);

-- Source-partitioned index to accelerate per-source deduplication lookups.
CREATE INDEX IF NOT EXISTS idx_weather_observations_source_external
    ON weather_observations (source, external_id, observed_at DESC);

CREATE TABLE IF NOT EXISTS ingestion_dead_letters (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    topic TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------------
-- Spark Structured Streaming staging table
--
-- Spark foreachBatch writes here first (as a hypertable for efficient
-- time-range scans), then promotes rows into weather_observations via
-- upsert_weather_observation() called within the same batch transaction.
-- Columns mirror weather_observations (minus the generated geom column) so
-- the Spark JDBC schema mapping is 1-to-1.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS weather_observations_stream_staging (
    observed_at          TIMESTAMPTZ NOT NULL,
    ingested_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    source               TEXT NOT NULL,
    external_id          TEXT NOT NULL,
    location_name        TEXT NOT NULL,
    latitude             DOUBLE PRECISION NOT NULL,
    longitude            DOUBLE PRECISION NOT NULL,
    temp_c               DOUBLE PRECISION,
    wind_speed_kph       DOUBLE PRECISION,
    wind_direction_deg   DOUBLE PRECISION,
    humidity_pct         DOUBLE PRECISION,
    precipitation_mm     DOUBLE PRECISION,
    pressure_hpa         DOUBLE PRECISION,
    weather_code         TEXT,
    quality_flags        JSONB NOT NULL DEFAULT '[]'::jsonb,
    provenance           JSONB NOT NULL DEFAULT '{}'::jsonb,
    raw_payload          JSONB NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (observed_at, source, external_id)
);

SELECT create_hypertable('weather_observations_stream_staging', 'observed_at', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_stream_staging_source_time
    ON weather_observations_stream_staging (source, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_stream_staging_location_time
    ON weather_observations_stream_staging (location_name, observed_at DESC);

-- ---------------------------------------------------------------------------
-- upsert_weather_observation
--
-- Called by the Spark foreachBatch write path via JDBC executeUpdate.
-- Guarantees the same ON CONFLICT idempotent upsert semantics as the Python
-- observation-writer consumer, so Kafka replay is safe regardless of which
-- path last wrote a given (observed_at, source, external_id) triple.
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION upsert_weather_observation(
    p_observed_at        TIMESTAMPTZ,
    p_source             TEXT,
    p_external_id        TEXT,
    p_location_name      TEXT,
    p_latitude           DOUBLE PRECISION,
    p_longitude          DOUBLE PRECISION,
    p_temp_c             DOUBLE PRECISION,
    p_wind_speed_kph     DOUBLE PRECISION,
    p_wind_direction_deg DOUBLE PRECISION,
    p_humidity_pct       DOUBLE PRECISION,
    p_precipitation_mm   DOUBLE PRECISION,
    p_pressure_hpa       DOUBLE PRECISION,
    p_weather_code       TEXT,
    p_quality_flags      JSONB,
    p_provenance         JSONB,
    p_raw_payload        JSONB
) RETURNS VOID LANGUAGE plpgsql AS $$
BEGIN
    INSERT INTO weather_observations (
        observed_at, source, external_id, location_name, latitude, longitude,
        temp_c, wind_speed_kph, wind_direction_deg, humidity_pct,
        precipitation_mm, pressure_hpa, weather_code, quality_flags,
        provenance, raw_payload
    )
    VALUES (
        p_observed_at, p_source, p_external_id, p_location_name,
        p_latitude, p_longitude, p_temp_c, p_wind_speed_kph,
        p_wind_direction_deg, p_humidity_pct, p_precipitation_mm,
        p_pressure_hpa, p_weather_code, p_quality_flags,
        p_provenance, p_raw_payload
    )
    ON CONFLICT (observed_at, source, external_id) DO UPDATE SET
        ingested_at          = now(),
        temp_c               = EXCLUDED.temp_c,
        wind_speed_kph       = EXCLUDED.wind_speed_kph,
        wind_direction_deg   = EXCLUDED.wind_direction_deg,
        humidity_pct         = EXCLUDED.humidity_pct,
        precipitation_mm     = EXCLUDED.precipitation_mm,
        pressure_hpa         = EXCLUDED.pressure_hpa,
        weather_code         = EXCLUDED.weather_code,
        quality_flags        = EXCLUDED.quality_flags,
        provenance           = EXCLUDED.provenance,
        raw_payload          = EXCLUDED.raw_payload;
END;
$$;

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at
    ON ingestion_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_ingestion_dead_letters_created_at
    ON ingestion_dead_letters (created_at DESC);

-- Source-filtered DLQ lookup (used by /ingestion/dead-letters?source=imd).
CREATE INDEX IF NOT EXISTS idx_ingestion_dead_letters_source_created
    ON ingestion_dead_letters (source, created_at DESC);
