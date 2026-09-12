from aiokafka import AIOKafkaProducer

from .models import DeadLetterEvent, NormalizedObservation, RawWeatherEvent
from .resilience import stable_json
from .topics import INGESTION_DLQ, NORMALIZED_OBSERVATION


class WeatherEventProducer:
    def __init__(self, bootstrap_servers: str) -> None:
        self.producer = AIOKafkaProducer(
            bootstrap_servers=bootstrap_servers,
            value_serializer=stable_json,
            key_serializer=lambda value: value.encode("utf-8"),
        )

    async def start(self) -> None:
        await self.producer.start()

    async def stop(self) -> None:
        await self.producer.stop()

    async def publish_raw(self, event: RawWeatherEvent) -> None:
        await self.producer.send_and_wait(
            event.topic,
            key=f"{event.source}:{event.location_name}",
            value=event.model_dump(mode="json"),
        )

    async def publish_dead_letter(self, event: DeadLetterEvent) -> None:
        await self.producer.send_and_wait(
            INGESTION_DLQ,
            key=f"{event.source}:{event.error_type}",
            value=event.model_dump(mode="json"),
        )

    async def publish_normalized(self, observation: NormalizedObservation) -> None:
        await self.producer.send_and_wait(
            NORMALIZED_OBSERVATION,
            key=f"{observation.source}:{observation.location_name}",
            value=observation.model_dump(mode="json"),
        )

    async def publish_message(self, topic: str, key: str, value: dict) -> None:
        await self.producer.send_and_wait(topic, key=key, value=value)
