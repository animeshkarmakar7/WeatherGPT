# Phase 1 - Data Ingestion Foundation

## Goal

Deliver real weather data flowing through a production-shaped ingestion pipeline before any chatbot or LLM features are added.

## Architecture

1. Connectors fetch source-specific payloads from Open-Meteo, NOAA, and IMD.
2. Each connector runs behind retry, timeout, circuit breaker, and last-known-good cache controls.
3. Raw events are published to source-specific Kafka topics.
4. Normalized observations are published to `weather.normalized.observation.v1`.
5. Invalid payloads or processing failures are sent to `weather.dlq.ingestion.v1`.
6. Spark Structured Streaming can read raw topics for scalable stream/batch processing and staging.
7. Celery schedules recurring ingestion jobs and handles backfill/retry workloads that do not belong on the synchronous API path.

## Production Decisions From The Blueprint

- External sources are isolated by connector so NOAA failures cannot stop Open-Meteo ingestion.
- Dead-letter events preserve malformed payloads for inspection instead of poisoning downstream processors.
- Numeric weather values are stored as sourced facts with provenance. Later LLM agents should phrase these facts, not invent them.
- Redis is used for last-known-good upstream fallback and Celery task coordination.
- TimescaleDB hypertables are used for time-series observations; PostGIS geography enables later spatial queries and alert zones.

## Exit Criteria

- `docker compose up --build` starts the ingestion stack.
- `GET /ready` confirms Redis, Kafka producer initialization, and TimescaleDB access.
- `POST /ingest/current?city=mumbai` publishes a normalized current-weather event.
- `GET /observations/current?city=mumbai` returns the latest stored observation after the streaming writer is enabled.
- Dead-letter topic receives connector or validation failures with enough context to debug the source payload.

## Local Commands

```bash
docker compose up --build
```

```bash
curl -X POST "http://localhost:8081/ingest/current?city=mumbai"
curl "http://localhost:8081/observations/current?city=mumbai"
```

Spark streaming is optional in local development because the API path also persists a normalized demo observation:

```bash
docker compose --profile streaming up spark-streaming
```

## Source Connector Status

| Source | Status | Notes |
| --- | --- | --- |
| Open-Meteo | Runnable | Current conditions for configured cities. |
| NOAA/NWS | Runnable for supported coordinates | Mainly US coverage; useful for connector hardening and public-domain forecasts. |
| IMD | Adapter scaffold | Requires an approved feed URL through `WEATHERGPT_IMD_BASE_URL`; do not hardcode unstable scraped endpoints. |
| WIS2.0/MQTT | Adapter scaffold | Configure broker details and run `weathergpt_ingestion.run_wis2_mqtt_subscriber` as a long-lived worker when a WIS2 broker is available. |
