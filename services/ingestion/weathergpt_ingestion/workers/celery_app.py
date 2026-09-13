import asyncio
import logging

from celery import Celery
from redis.asyncio import Redis

from ..config import get_settings
from ..connectors import Wis2MqttSubscriber
from ..kafka import WeatherEventProducer
from ..models import DeadLetterEvent, SourceName
from ..repository import WeatherRepository
from ..service import IngestionService
from ..topics import DLQ_WIS2, WIS2_NOTIFICATION

logger = logging.getLogger(__name__)

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
    """Poll Open-Meteo for all default cities and publish events to Kafka.

    Delegates to IngestionService so run accounting, DLQ routing, and
    normalization use the same code path as the HTTP API. This avoids the
    previous bug where records_published was hardcoded to 2 regardless of
    what was actually published.
    """
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    producer = WeatherEventProducer(settings.kafka_bootstrap_servers)
    repository = WeatherRepository(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    await producer.start()
    await repository.start()
    results: list[dict[str, str]] = []
    try:
        service = IngestionService(settings, redis, producer, repository)
        for city in settings.default_cities:
            try:
                result = await service.ingest_current(city, SourceName.OPEN_METEO)
                results.append({"city": city, "status": result["status"]})
            except Exception as exc:
                logger.exception("ingest failed city=%s exc=%s", city, exc)
                results.append({"city": city, "status": "failed"})
        return results
    finally:
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
        asyncio.run_coroutine_threadsafe(
            producer.publish_dead_letter(event, topic=DLQ_WIS2), loop
        )
        asyncio.run_coroutine_threadsafe(repository.save_dead_letter(event), loop)

    try:
        Wis2MqttSubscriber(settings, on_message=on_message, on_dead_letter=on_dead_letter).run_forever()
    finally:
        await repository.close()
        await producer.stop()
