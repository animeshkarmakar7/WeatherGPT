# ---------------------------------------------------------------------------
# Kafka topic name constants
# ---------------------------------------------------------------------------
# Raw source topics (one per connector)
OPEN_METEO_CURRENT   = "weather.raw.open_meteo.current.v1"
NOAA_FORECAST        = "weather.raw.noaa.forecast.v1"
IMD_BULLETIN         = "weather.raw.imd.bulletin.v1"
IMD_CURRENT          = "weather.raw.imd.current.v1"
WIS2_NOTIFICATION    = "weather.raw.wis2.notification.v1"

# Normalized observation topic (consumed by both Python writer and Spark)
NORMALIZED_OBSERVATION = "weather.normalized.observation.v1"

# ---------------------------------------------------------------------------
# Dead-letter topics  (one per connector so malformed IMD / ECMWF / NOAA
# payloads are quarantined and inspectable independently — per spec)
# ---------------------------------------------------------------------------
DLQ_OPEN_METEO   = "weather.dlq.open_meteo.v1"
DLQ_NOAA         = "weather.dlq.noaa.v1"
DLQ_IMD          = "weather.dlq.imd.v1"
DLQ_WIS2         = "weather.dlq.wis2.v1"
# DLQ for the observation-writer consumer (malformed *normalized* events)
DLQ_NORMALIZED   = "weather.dlq.normalized.v1"
# Generic / catch-all fallback (kept for backward compat; prefer source-specific)
INGESTION_DLQ    = "weather.dlq.ingestion.v1"
