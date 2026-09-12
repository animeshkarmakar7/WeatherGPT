import asyncio

from celery import Celery
from redis.asyncio import Redis

from ..config import get_settings
from ..connectors import OpenMeteoConnector, Wis2MqttSubscriber
from ..kafka import WeatherEventProducer
from ..models import DeadLetterEvent, IngestionRun, IngestionStatus, SourceName
from ..normalizer import normalize_event
from ..repository import WeatherRepository
from ..topics import WIS2_NOTIFICATION

settings = get_settings()

celery_app = Celery(
    "weathergpt_ingestion",
    broker=settings.redis_url,
    backend=settings.redis_url,
)

celery_app.conf.beat_schedule = {
    "ingest-default-cities-every-10-minutes": {
        "task": "weathergpt_ingestion.ingest_default_cities",
        "schedule": 600.0,
    }
}


@celery_app.task(name="weathergpt_ingestion.ingest_default_cities")
def ingest_default_cities() -> list[dict[str, str]]:
    return asyncio.run(_ingest_default_cities())


@celery_app.task(name="weathergpt_ingestion.run_wis2_mqtt_subscriber")
def run_wis2_mqtt_subscriber() -> str:
    asyncio.run(_run_wis2_mqtt_subscriber())
    return "stopped"


async def _ingest_default_cities() -> list[dict[str, str]]:
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    producer = WeatherEventProducer(settings.kafka_bootstrap_servers)
    repository = WeatherRepository(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    connector = OpenMeteoConnector(settings, redis)
    await producer.start()
    await repository.start()
    results: list[dict[str, str]] = []
    try:
        for city, coordinates in settings.default_cities.items():
            run = IngestionRun(source=SourceName.OPEN_METEO, connector=connector.name)
            await repository.start_run(run)
            try:
                event = await connector.fetch_current(city, coordinates[0], coordinates[1])
                await producer.publish_raw(event)
                observation = normalize_event(event)
                await producer.publish_normalized(observation)
                await repository.finish_run(run, IngestionStatus.SUCCEEDED, 1, 2)
                results.append({"city": city, "status": "queued"})
            except Exception as exc:
                dlq_event = DeadLetterEvent(
                    source=SourceName.OPEN_METEO.value,
                    topic=connector.topic,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    payload={"city": city},
                )
                await producer.publish_dead_letter(dlq_event)
                await repository.save_dead_letter(dlq_event)
                await repository.finish_run(run, IngestionStatus.FAILED, 0, 0, str(exc))
                results.append({"city": city, "status": "failed"})
        return results
    finally:
        await connector.close()
        await repository.close()
        await producer.stop()
        await redis.aclose()


async def _run_wis2_mqtt_subscriber() -> None:
    loop = asyncio.get_running_loop()
    producer = WeatherEventProducer(settings.kafka_bootstrap_servers)
    repository = WeatherRepository(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    await producer.start()
    await repository.start()

    def on_message(payload: dict) -> None:
        asyncio.run_coroutine_threadsafe(
            producer.publish_message(WIS2_NOTIFICATION, payload["mqtt_topic"], payload),
            loop,
        )

    def on_dead_letter(event: DeadLetterEvent) -> None:
        asyncio.run_coroutine_threadsafe(producer.publish_dead_letter(event), loop)
        asyncio.run_coroutine_threadsafe(repository.save_dead_letter(event), loop)

    try:
        Wis2MqttSubscriber(settings, on_message=on_message, on_dead_letter=on_dead_letter).run_forever()
    finally:
        await repository.close()
        await producer.stop()
