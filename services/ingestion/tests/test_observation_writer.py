from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from weathergpt_ingestion.models import NormalizedObservation, Provenance, SourceName
from weathergpt_ingestion.workers.observation_writer import ObservationWriter


class FakeConsumer:
    def __init__(self) -> None:
        self.commits = []

    async def commit(self, offsets):
        self.commits.append(offsets)


class FakeProducer:
    def __init__(self) -> None:
        self.dead_letters = []

    async def publish_dead_letter(self, event, topic=None):
        self.dead_letters.append((event, topic))


class FakeRepository:
    def __init__(self) -> None:
        self.observations = []
        self.dead_letters = []

    async def save_observation(self, observation):
        self.observations.append(observation)

    async def save_dead_letter(self, event):
        self.dead_letters.append(event)


@pytest.fixture
def valid_record():
    payload = {
        "observed_at": "2026-09-12T12:00:00Z",
        "source": "open_meteo",
        "external_id": "event-1",
        "location_name": "mumbai",
        "latitude": 19.076,
        "longitude": 72.8777,
        "temp_c": 28.4,
        "wind_speed_kph": 18.0,
        "wind_direction_deg": 240.0,
        "humidity_pct": 81.0,
        "precipitation_mm": 0.2,
        "pressure_hpa": 1006.2,
        "weather_code": "3",
        "quality_flags": [],
        "provenance": {
            "source": "open_meteo",
            "connector": "open_meteo_current",
            "fetched_at": "2026-09-12T12:00:00Z",
        },
        "raw_payload": {"current": {"temperature_2m": 28.4}},
    }
    import json

    return SimpleNamespace(topic="weather.normalized.observation.v1", partition=2, offset=41, value=json.dumps(payload).encode())


@pytest.mark.asyncio
async def test_successful_record_is_persisted_before_offset_commit(valid_record):
    consumer = FakeConsumer()
    producer = FakeProducer()
    repository = FakeRepository()
    writer = ObservationWriter(consumer, producer, repository)

    result = await writer.handle(valid_record)

    assert result == "persisted"
    assert len(repository.observations) == 1
    assert isinstance(repository.observations[0], NormalizedObservation)
    committed = consumer.commits[0]
    assert next(iter(committed.values())) == 42
    assert producer.dead_letters == []


@pytest.mark.asyncio
async def test_malformed_record_goes_to_dlq_and_is_committed(valid_record):
    valid_record.value = b"not-json"
    consumer = FakeConsumer()
    producer = FakeProducer()
    repository = FakeRepository()
    writer = ObservationWriter(consumer, producer, repository)

    result = await writer.handle(valid_record)

    assert result == "dead_lettered"
    assert len(producer.dead_letters) == 1
    event, topic = producer.dead_letters[0]
    assert event.source == "kafka-normalized"
    assert topic == "weather.dlq.normalized.v1"
    assert len(repository.dead_letters) == 1
    assert repository.observations == []
    committed = consumer.commits[0]
    assert next(iter(committed.values())) == 42
