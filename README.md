# WeatherGPT

Production-oriented conversational weather intelligence platform.

This repository starts with Phase 1 from the build blueprint: real weather data ingestion through source connectors, Kafka events, Spark/Celery processing, and TimescaleDB-ready storage contracts.

## Phase 1 Scope

- Source connectors for Open-Meteo, NOAA, and IMD adapter scaffolding.
- Circuit breaker, retry, and last-known-good behavior for upstream resilience.
- Kafka topic contracts with dead-letter routing.
- Celery worker tasks for scheduled ingestion and backfill orchestration.
- Spark Structured Streaming entrypoint for Kafka-to-Timescale processing.
- Database schema for weather observations, ingestion runs, and DLQ records.
- Local Docker Compose topology for Kafka, Redis, TimescaleDB, MinIO, Spark, and the ingestion API.
- CI gate for ingestion tests, package compilation, and Compose validation.

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
curl -X POST http://localhost:8081/ingest/current?city=mumbai
```

## Project Layout

```text
services/ingestion/        Phase 1 ingestion microservice
infra/db/                  TimescaleDB schema
infra/kafka/               Topic definitions
docs/phase-1-ingestion.md  Architecture and operating notes
```
