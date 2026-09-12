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

CREATE TABLE IF NOT EXISTS ingestion_dead_letters (
    id UUID PRIMARY KEY,
    source TEXT NOT NULL,
    topic TEXT NOT NULL,
    error_type TEXT NOT NULL,
    error_message TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS weather_observations_stream_staging (
    source TEXT NOT NULL,
    external_id TEXT NOT NULL,
    location_name TEXT NOT NULL,
    latitude DOUBLE PRECISION NOT NULL,
    longitude DOUBLE PRECISION NOT NULL,
    observed_at TIMESTAMPTZ NOT NULL,
    payload TEXT,
    quality_flags TEXT[]
);

CREATE INDEX IF NOT EXISTS idx_ingestion_runs_started_at
    ON ingestion_runs (started_at DESC);

CREATE INDEX IF NOT EXISTS idx_ingestion_dead_letters_created_at
    ON ingestion_dead_letters (created_at DESC);
