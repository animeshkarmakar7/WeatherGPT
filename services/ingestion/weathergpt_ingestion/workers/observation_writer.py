import asyncio
import json
import logging
from uuid import NAMESPACE_URL, uuid5

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition
from pydantic import ValidationError

from ..config import Settings, get_settings
from ..kafka import WeatherEventProducer
from ..models import DeadLetterEvent, NormalizedObservation
from ..repository import WeatherRepository
from ..topics import DLQ_NORMALIZED, NORMALIZED_OBSERVATION

logger = logging.getLogger(__name__)


class ObservationWriter:
    """Persist normalized Kafka events with at-least-once delivery semantics."""

    def __init__(
        self,
        consumer: AIOKafkaConsumer,
        dlq_producer: WeatherEventProducer,
        repository: WeatherRepository,
    ) -> None:
        self.consumer = consumer
        self.dlq_producer = dlq_producer
        self.repository = repository

    async def handle(self, record) -> str:
        raw_value = record.value.decode("utf-8", errors="replace")
        try:
            payload = json.loads(raw_value)
            observation = NormalizedObservation.model_validate(payload)
        except (json.JSONDecodeError, ValidationError, TypeError) as exc:
            dlq_event = DeadLetterEvent(
                id=uuid5(NAMESPACE_URL, f"{record.topic}:{record.partition}:{record.offset}"),
                source="kafka-normalized",
                topic=record.topic,
                error_type=type(exc).__name__,
                error_message=str(exc),
                payload={
                    "partition": record.partition,
                    "offset": record.offset,
                    "raw_value": raw_value,
                },
            )
            await self.dlq_producer.publish_dead_letter(dlq_event, topic=DLQ_NORMALIZED)
            await self.repository.save_dead_letter(dlq_event)
            await self.consumer.commit(
                {TopicPartition(record.topic, record.partition): record.offset + 1}
            )
            return "dead_lettered"

        await self.repository.save_observation(observation)
        await self.consumer.commit(
            {TopicPartition(record.topic, record.partition): record.offset + 1}
        )
        return "persisted"


async def run(settings: Settings | None = None) -> None:
    settings = settings or get_settings()

    consumer = AIOKafkaConsumer(
        NORMALIZED_OBSERVATION,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_observation_writer_group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_records=settings.kafka_consumer_max_poll_records,
        max_poll_interval_ms=settings.kafka_consumer_max_poll_interval_ms,
        isolation_level="read_committed",
    )
    dlq_producer = WeatherEventProducer(settings.kafka_bootstrap_servers)
    repository = WeatherRepository(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    writer = ObservationWriter(consumer, dlq_producer, repository)

    repository_started = False
    producer_started = False
    consumer_started = False

    try:
        await repository.start()
        repository_started = True
        await dlq_producer.start()
        producer_started = True
        await consumer.start()
        consumer_started = True

        logger.info(
            "observation writer started group_id=%s topic=%s",
            settings.kafka_observation_writer_group_id,
            NORMALIZED_OBSERVATION,
        )

        async for record in consumer:
            try:
                result = await writer.handle(record)
            except Exception:
                logger.exception(
                    "fatal processing error topic=%s partition=%s offset=%s",
                    record.topic,
                    record.partition,
                    record.offset,
                )
                raise
            logger.info(
                "processed topic=%s partition=%s offset=%s result=%s",
                record.topic,
                record.partition,
                record.offset,
                result,
            )
    finally:
        if consumer_started:
            await consumer.stop()
        if producer_started:
            await dlq_producer.stop()
        if repository_started:
            await repository.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
