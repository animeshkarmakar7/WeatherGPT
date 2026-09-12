# Phase 1 - Data Ingestion Foundation

## Goal

Deliver real weather data flowing through a production-shaped event-driven ingestion pipeline before chatbot or LLM features are added.

## Authoritative persistence flow

1. Connectors fetch source-specific payloads from Open-Meteo, NOAA/NWS, and the IMD/WIS2 adapters.
2. Connector calls run behind timeout, circuit-breaker, and last-known-good fallback controls.
3. Raw events are published to source-specific Kafka topics.
4. The connector process normalizes the payload and publishes `weather.normalized.observation.v1`.
5. A dedicated Kafka observation-writer consumer owns the authoritative TimescaleDB write.
6. The writer disables Kafka auto-commit, validates the normalized event, writes an idempotent PostgreSQL/TimescaleDB upsert, and commits the Kafka offset only after the DB write succeeds.
7. Malformed normalized events are published to `weather.dlq.ingestion.v1` and then acknowledged so a poison message cannot block its partition.
8. Transient database/Kafka failures are not acknowledged; the writer exits and the orchestrator restarts it from the last committed offset.
9. Spark Structured Streaming consumes the normalized topic independently for staging/analytics. It is not the authoritative observation writer.
10. Celery schedules source polling and backfill orchestration; it does not directly persist normalized observations.

## CQRS boundary

### Command/write side

```text
External source
    -> connector
    -> raw Kafka topic
    -> normalized Kafka topic
    -> observation-writer consumer
    -> TimescaleDB/PostGIS
```

### Query/read side

The future Weather Query service should read from optimized read models and Redis rather than from ingestion workers. This keeps external API traffic and user query traffic independent from ingestion throughput.

## Reliability semantics

- Kafka producer idempotence is enabled.
- Kafka consumer auto-commit is disabled.
- DB persistence is committed before the Kafka offset is committed.
- PostgreSQL/TimescaleDB uses an idempotent `ON CONFLICT` upsert keyed by source observation identity.
- A crash after DB commit but before Kafka commit causes a safe replay/update rather than a duplicate logical observation.
- A malformed event is isolated in the DLQ and its source offset is committed.

## Infrastructure

Local Docker Compose provides Kafka, Redis, TimescaleDB/PostGIS, MinIO, the ingestion API, Celery worker/beat, the Kafka observation writer, and optional Spark streaming.

The local Kafka configuration intentionally uses one broker and replication factor 1. This is a development topology; a production deployment must use a multi-broker Kafka cluster with replicated topics and authenticated/encrypted listeners.

## Source connector status

| Source | Status | Notes |
| --- | --- | --- |
| Open-Meteo | Runnable | Current conditions for configured cities. |
| NOAA/NWS | Runnable for supported coordinates | Point forecast API; this is not NOAA GFS/NOMADS ingestion. |
| IMD | Adapter scaffold | Requires an approved IMD API/feed endpoint. |
| WIS2.0/MQTT | Adapter scaffold | Receives WIS2 notifications and forwards them to Kafka; full referenced-resource ingestion remains a follow-up. |

## Exit criteria

- `docker compose up --build` starts the ingestion API, Kafka observation writer, and dependencies.
- `GET /ready` checks Redis, TimescaleDB, and Kafka topic metadata.
- `POST /ingest/current?city=mumbai` publishes raw and normalized Kafka events and returns `status=queued`.
- The observation writer consumes `weather.normalized.observation.v1` and persists it to `weather_observations`.
- `GET /observations/current?city=mumbai` returns the persisted observation after the writer processes the event.
- Duplicate Kafka delivery does not create a duplicate logical observation.
- Invalid normalized events land in the DLQ and do not stall their Kafka partition.
- Spark can consume the normalized topic as a separate analytics/staging consumer without becoming the authoritative database writer.
