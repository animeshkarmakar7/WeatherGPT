import json
import logging

from aiokafka import AIOKafkaConsumer
from aiokafka.structs import TopicPartition
from pydantic import ValidationError
from redis.asyncio import Redis

from ..config import Settings
from ..kafka import WeatherEventProducer
from ..models import DeadLetterEvent, NormalizedObservation
from ..repository import WeatherRepository
from ..topics import INGESTION_DLQ, NORMALIZED_OBSERVATION

logger = logging.getLogger(__name__)


class ObservationWriter:
    """Persist normalized Kafka events using at-least-once delivery semantics.

    The database write is completed before the Kafka offset is committed. The
    repository uses an idempotent upsert, so a crash between those operations
    results in a safe replay rather than a duplicate logical observation.
    """

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
        try:
            raw_value = record.value.decode("utf-8")
            payload = json.loads(raw_value)
            observation = NormalizedObservation.model_validate(payload)
        except (UnicodeDecodeError, json.JSONDecodeError, ValidationError, TypeError) as exc:
            # A malformed event is not retryable. Preserve the original bytes
            # in the DLQ, then commit only this record so one poison message
            # cannot block the partition indefinitely.
            dlq_event = DeadLetterEvent(
                source="kafka-normalized",
                topic=record.topic,
                error_type=type(exc).__name__,
                error_message=str(exc),
                payload={
                    "partition": record.partition,
                    "offset": record.offset,
                    "raw_value": raw_value if "raw_value" in locals() else record.value.decode("utf-8", errors="replace"),
                },
            )
            await self.dlq_producer.publish_dead_letter(dlq_event)
            await self.consumer.commit(
                {TopicPartition(record.topic, record.partition): record.offset + 1}
            )
            return "dead_lettered"

        # Any database failure is deliberately allowed to propagate. The
        # offset is not committed, so the message is replayed after recovery.
        await self.repository.save_observation(observation)
        await self.consumer.commit(
            {TopicPartition(record.topic, record.partition): record.offset + 1}
        )
        return "persisted"


async def run(settings: Settings | None = None) -> None:
    settings = settings or __import__(
        "weathergpt_ingestion.config", fromlist=["get_settings"]
    ).get_settings()

    consumer = AIOKafkaConsumer(
        NORMALIZED_OBSERVATION,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_observation_writer_group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
        max_poll_records=settings.kafka_consumer_max_poll_records,
        max_poll_interval_ms=settings.kafka_consumer_max_poll_interval_ms,
        enable_partition_eof=False,
        isolation_level="read_committed",
    )
    dlq_producer = WeatherEventProducer(settings.kafka_bootstrap_servers)
    repository = WeatherRepository(
        settings.database_url,
        min_size=settings.database_pool_min_size,
        max_size=settings.database_pool_max_size,
    )
    writer = ObservationWriter(consumer, dlq_producer, repository)

    await repository.start()
    await dlq_producer.start()
    await consumer.start()

    logger.info(
        "observation writer started group_id=%s topic=%s",
        settings.kafka_observation_writer_group_id,
        NORMALIZED_OBSERVATION,
    )

    try:
        async for record in consumer:
            result = await writer.handle(record)
            logger.debug(
                "processed topic=%s partition=%s offset=%s result=%s",
                record.topic,
                record.partition,
                record.offset,
                result,
            )
    finally:
        await consumer.stop()
        await dlq_producer.stop()
        await repository.close()


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
