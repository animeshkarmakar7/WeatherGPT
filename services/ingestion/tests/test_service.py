from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from weathergpt_ingestion.models import Provenance, RawWeatherEvent, SourceName
from weathergpt_ingestion.service import IngestionService, UnknownLocationError
from weathergpt_ingestion.topics import OPEN_METEO_CURRENT


class FakeConnector:
    name = "fake_open_meteo"
    topic = OPEN_METEO_CURRENT

    def __init__(self, event: RawWeatherEvent) -> None:
        self.event = event
        self.closed = False

    async def fetch_current(self, city: str, latitude: float, longitude: float) -> RawWeatherEvent:
        return self.event

    async def close(self) -> None:
        self.closed = True


class FakeProducer:
    def __init__(self) -> None:
        self.raw = []
        self.normalized = []
        self.dead_letters = []

    async def publish_raw(self, event):
        self.raw.append(event)

    async def publish_normalized(self, observation):
        self.normalized.append(observation)

    async def publish_dead_letter(self, event, topic=None):
        self.dead_letters.append((event, topic))


class FakeRepository:
    def __init__(self) -> None:
        self.runs = []
        self.finished = []
        self.observations = []
        self.dead_letters = []

    async def start_run(self, run):
        self.runs.append(run)

    async def finish_run(self, run, status, records_fetched, records_published, error_message=None):
        self.finished.append((run, status, records_fetched, records_published, error_message))

    async def save_observation(self, observation):
        self.observations.append(observation)

    async def save_dead_letter(self, event):
        self.dead_letters.append(event)


@pytest.mark.asyncio
async def test_ingestion_service_publishes_events_without_direct_db_persistence():
    event = RawWeatherEvent(
        source=SourceName.OPEN_METEO,
        topic=OPEN_METEO_CURRENT,
        external_id="event-1",
        location_name="Mumbai",
        latitude=19.076,
        longitude=72.8777,
        observed_at=datetime(2026, 9, 12, 12, 0, tzinfo=UTC),
        payload={"current": {"temperature_2m": 28}},
        provenance=Provenance(source=SourceName.OPEN_METEO, connector="fake"),
    )
    producer = FakeProducer()
    repository = FakeRepository()
    service = IngestionService(
        settings=SimpleNamespace(default_cities={"mumbai": (19.076, 72.8777)}),
        redis=SimpleNamespace(),
        producer=producer,
        repository=repository,
    )
    connector = FakeConnector(event)
    service._connector_for = lambda source: connector

    result = await service.ingest_current("mumbai")

    assert result["source"] == "open_meteo"
    assert result["status"] == "queued"
    assert result["persistence"] == "kafka"
    assert len(producer.raw) == 1
    assert len(producer.normalized) == 1
    assert repository.observations == []
    assert repository.finished[0][2:4] == (1, 2)
    assert connector.closed is True


@pytest.mark.asyncio
async def test_ingestion_service_rejects_unknown_city_before_starting_run():
    repository = FakeRepository()
    service = IngestionService(
        settings=SimpleNamespace(default_cities={}),
        redis=SimpleNamespace(),
        producer=FakeProducer(),
        repository=repository,
    )

    with pytest.raises(UnknownLocationError):
        await service.ingest_current("unknown")

    assert repository.runs == []
