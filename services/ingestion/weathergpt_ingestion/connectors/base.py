from abc import ABC, abstractmethod

import httpx
from redis.asyncio import Redis

from ..config import Settings
from ..models import RawWeatherEvent
from ..resilience import CircuitBreaker, LastKnownGoodCache


class WeatherConnector(ABC):
    source_url: str | None = None

    def __init__(self, settings: Settings, redis: Redis) -> None:
        self.settings = settings
        self.redis = redis
        self.cache = LastKnownGoodCache(redis, settings.last_known_good_ttl_seconds)
        self.breaker = CircuitBreaker(
            name=self.name,
            failure_threshold=settings.circuit_breaker_failure_threshold,
            recovery_seconds=settings.circuit_breaker_recovery_seconds,
        )
        self.client = httpx.AsyncClient(timeout=settings.request_timeout_seconds)

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def topic(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def fetch_current(self, location_name: str, latitude: float, longitude: float) -> RawWeatherEvent:
        raise NotImplementedError

    async def close(self) -> None:
        await self.client.aclose()

    def cache_key(self, location_name: str, latitude: float, longitude: float) -> str:
        return f"lkg:{self.name}:{location_name.lower()}:{latitude:.4f}:{longitude:.4f}"
