import pytest
from fastapi import HTTPException

from weathergpt_ingestion import api


class FakeRedis:
    async def ping(self):
        return True


class FakeRepository:
    async def ping(self):
        return True


class FakeProducer:
    producer = object()


@pytest.mark.asyncio
async def test_ready_returns_dependency_checks(monkeypatch):
    monkeypatch.setattr(
        api.app,
        "state",
        type("State", (), {"redis": FakeRedis(), "repository": FakeRepository(), "producer": FakeProducer()})(),
    )

    result = await api.ready()

    assert result["status"] == "ready"
    assert result["checks"] == {"redis": True, "database": True, "kafka_producer": True}


@pytest.mark.asyncio
async def test_ready_raises_when_dependency_check_fails(monkeypatch):
    class BrokenRepository:
        async def ping(self):
            return False

    monkeypatch.setattr(
        api.app,
        "state",
        type("State", (), {"redis": FakeRedis(), "repository": BrokenRepository(), "producer": FakeProducer()})(),
    )

    with pytest.raises(HTTPException) as exc:
        await api.ready()

    assert exc.value.status_code == 503
