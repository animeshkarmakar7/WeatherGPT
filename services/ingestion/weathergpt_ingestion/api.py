from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException
from redis.asyncio import Redis

from .config import get_settings
from .kafka import WeatherEventProducer
from .models import SourceName
from .repository import WeatherRepository
from .service import IngestionService, UnknownLocationError, UnsupportedSourceError


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    producer = WeatherEventProducer(settings.kafka_bootstrap_servers)
    repository = WeatherRepository(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    await producer.start()
    await repository.start()
    app.state.redis = redis
    app.state.producer = producer
    app.state.repository = repository
    yield
    await producer.stop()
    await repository.close()
    await redis.aclose()


app = FastAPI(title="WeatherGPT Ingestion API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "weathergpt-ingestion"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    checks: dict[str, bool] = {}
    checks["redis"] = bool(await app.state.redis.ping())
    checks["database"] = await app.state.repository.ping()
    checks["kafka_producer"] = await app.state.producer.ready()
    ready_state = all(checks.values())
    if not ready_state:
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.post("/ingest/current", status_code=202)
async def ingest_current(city: str, source: SourceName = SourceName.OPEN_METEO) -> dict[str, object]:
    settings = get_settings()
    service = IngestionService(settings, app.state.redis, app.state.producer, app.state.repository)
    try:
        return await service.ingest_current(city, source)
    except UnknownLocationError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except UnsupportedSourceError as exc:
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(status_code=502, detail="weather ingestion failed") from None


@app.get("/observations/current")
async def latest_observation(city: str) -> dict[str, object]:
    observation = await app.state.repository.latest_observation(city)
    if observation is None:
        raise HTTPException(status_code=404, detail=f"no observation found for {city}")
    return observation


@app.get("/ingestion/runs")
async def list_runs(limit: int = 25) -> list[dict[str, object]]:
    return await app.state.repository.list_runs(limit=min(limit, 100))


@app.get("/ingestion/dead-letters")
async def list_dead_letters(limit: int = 25) -> list[dict[str, object]]:
    return await app.state.repository.list_dead_letters(limit=min(limit, 100))


@app.post("/ingestion/backfill/default-cities")
async def enqueue_default_city_backfill() -> dict[str, str]:
    from .workers.celery_app import ingest_default_cities

    task = ingest_default_cities.delay()
    return {"task_id": task.id, "status": "queued"}
