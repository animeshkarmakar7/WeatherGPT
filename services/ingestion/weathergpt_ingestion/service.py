from redis.asyncio import Redis

from .config import Settings
from .connectors import ImdCurrentWeatherConnector, NoaaForecastConnector, OpenMeteoConnector
from .kafka import WeatherEventProducer
from .models import DeadLetterEvent, IngestionRun, IngestionStatus, SourceName
from .normalizer import normalize_event
from .repository import WeatherRepository
from .topics import DLQ_IMD, DLQ_NOAA, DLQ_OPEN_METEO, DLQ_WIS2, INGESTION_DLQ


def _dlq_topic_for(source: SourceName) -> str:
    """Return the per-connector dead-letter topic for *source*.

    Per-connector DLQ topics quarantine malformed payloads independently so
    that a flood of bad IMD messages, for example, does not bury NOAA errors.
    """
    return {
        SourceName.OPEN_METEO: DLQ_OPEN_METEO,
        SourceName.NOAA: DLQ_NOAA,
        SourceName.IMD: DLQ_IMD,
        SourceName.WIS2: DLQ_WIS2,
    }.get(source, INGESTION_DLQ)


class IngestionService:
    """Fetch, normalize, and publish weather events.

    Kafka is the authoritative hand-off for persistence. This service does not
    write normalized observations directly to TimescaleDB; the dedicated Kafka
    observation writer owns that side effect after successful offset handling.
    """

    def __init__(
        self,
        settings: Settings,
        redis: Redis,
        producer: WeatherEventProducer,
        repository: WeatherRepository,
    ) -> None:
        self.settings = settings
        self.redis = redis
        self.producer = producer
        self.repository = repository

    async def ingest_current(self, city: str, source: SourceName = SourceName.OPEN_METEO) -> dict[str, object]:
        city = city.strip().lower()
        coordinates = self.settings.default_cities.get(city)
        if not coordinates:
            raise UnknownLocationError(f"unknown configured city: {city}")

        connector = self._connector_for(source)
        run = IngestionRun(source=source, connector=connector.name)
        await self.repository.start_run(run)
        try:
            event = await connector.fetch_current(city, coordinates[0], coordinates[1])
            await self.producer.publish_raw(event)
            observation = normalize_event(event)
            await self.producer.publish_normalized(observation)
            await self.repository.finish_run(run, IngestionStatus.SUCCEEDED, 1, 2)
            return {
                "run_id": str(run.id),
                "status": "queued",
                "persistence": "kafka",
                "published_topic": event.topic,
                "normalized_topic": "weather.normalized.observation.v1",
                "location": event.location_name,
                "source": event.source.value,
                "quality_flags": [flag.value for flag in event.quality_flags],
            }
        except Exception as exc:
            try:
                dlq_event = DeadLetterEvent(
                    source=source.value,
                    topic=getattr(connector, "topic", "unknown"),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    payload={"city": city, "source": source.value},
                )
                dlq_topic = _dlq_topic_for(source)
                await self.producer.publish_dead_letter(dlq_event, topic=dlq_topic)
                await self.repository.save_dead_letter(dlq_event)
            except Exception:
                # Preserve the original ingestion failure while logging DLQ
                # publication/audit failures through the caller's error logs.
                pass
            await self.repository.finish_run(run, IngestionStatus.FAILED, 0, 0, str(exc))
            raise
        finally:
            await connector.close()

    def _connector_for(self, source: SourceName):
        if source == SourceName.OPEN_METEO:
            return OpenMeteoConnector(self.settings, self.redis)
        if source == SourceName.NOAA:
            return NoaaForecastConnector(self.settings, self.redis)
        if source == SourceName.IMD:
            return ImdCurrentWeatherConnector(self.settings, self.redis)
        raise UnsupportedSourceError(f"unsupported source: {source.value}")


class UnknownLocationError(ValueError):
    pass


class UnsupportedSourceError(ValueError):
    pass
