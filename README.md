# WeatherGPT

Production-oriented conversational weather intelligence platform.

The repository starts with Phase 1 from the build blueprint: real weather data ingestion through source connectors, Kafka events, a dedicated Kafka-to-TimescaleDB writer, Spark/Celery processing, and TimescaleDB-ready storage contracts.

## Phase 1 Scope

- Source connectors for Open-Meteo, NOAA/NWS, and IMD/WIS2 adapter scaffolding.
- Timeout, circuit breaker, and last-known-good behavior for upstream resilience.
- Idempotent Kafka producer configuration and versioned topic contracts with DLQ routing.
- Kafka consumer groups with manual offset commits for authoritative database persistence.
- PostgreSQL/TimescaleDB connection pooling and idempotent observation upserts.
- Celery worker tasks for scheduled ingestion and backfill orchestration.
- Spark Structured Streaming as a separate normalized-event analytics/staging consumer.
- Database schema for weather observations, ingestion runs, and DLQ records.
- Local Docker Compose topology for Kafka, Redis, TimescaleDB/PostGIS, MinIO, Spark, ingestion API, and the observation writer.
- CI gate for ingestion tests, package compilation, and Compose validation.

## Authoritative Phase 1 Flow

```text
External Weather Source
        -> Connector
        -> Raw Kafka Topic
        -> Normalization
        -> weather.normalized.observation.v1
        -> Observation Writer Consumer
        -> TimescaleDB/PostGIS
        -> Future CQRS Read Model / Redis
        -> Query API
```

The ingestion API no longer writes observations directly to PostgreSQL. It returns `status=queued` after the normalized event is durably handed to Kafka. The observation writer persists the event and commits the Kafka offset only after the database transaction succeeds.

## Quick Start

```bash
docker compose up --build
```

Health check:

```bash
curl http://localhost:8081/health
curl http://localhost:8081/ready
```

Run a one-shot ingestion for a configured city:

```bash
curl -X POST "http://localhost:8081/ingest/current?city=mumbai"
```

Then query the database-backed observation after the Kafka writer processes the event:

```bash
curl "http://localhost:8081/observations/current?city=mumbai"
```

Run the optional Spark staging/analytics consumer:

```bash
docker compose --profile streaming up spark-streaming
```

## Project Layout

```text
services/ingestion/        Phase 1 ingestion microservice
infra/db/                  TimescaleDB/PostGIS schema
infra/kafka/               Topic definitions
docs/phase-1-ingestion.md  Architecture and operating notes
```
