from redis.asyncio import Redis

from .config import Settings
from .connectors import NoaaForecastConnector, OpenMeteoConnector
from .kafka import WeatherEventProducer
from .models import DeadLetterEvent, IngestionRun, IngestionStatus, SourceName
from .normalizer import normalize_event
from .repository import WeatherRepository


class IngestionService:
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
        coordinates = self.settings.default_cities.get(city.lower())
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
            await self.repository.save_observation(observation)
            await self.repository.finish_run(run, IngestionStatus.SUCCEEDED, 1, 2)
            return {
                "run_id": str(run.id),
                "published_topic": event.topic,
                "normalized_topic": "weather.normalized.observation.v1",
                "location": event.location_name,
                "source": event.source.value,
                "quality_flags": [flag.value for flag in event.quality_flags],
            }
        except Exception as exc:
            dlq_event = DeadLetterEvent(
                source=source,
                topic=getattr(connector, "topic", "unknown"),
                error_type=type(exc).__name__,
                error_message=str(exc),
                payload={"city": city, "source": source.value},
            )
            await self.producer.publish_dead_letter(dlq_event)
            await self.repository.save_dead_letter(dlq_event)
            await self.repository.finish_run(run, IngestionStatus.FAILED, 0, 0, str(exc))
            raise
        finally:
            await connector.close()

    def _connector_for(self, source: SourceName):
        if source == SourceName.OPEN_METEO:
            return OpenMeteoConnector(self.settings, self.redis)
        if source == SourceName.NOAA:
            return NoaaForecastConnector(self.settings, self.redis)
        raise UnsupportedSourceError(f"{source.value} connector is configured for batch adapter use only")


class UnknownLocationError(ValueError):
    pass


class UnsupportedSourceError(ValueError):
    pass
