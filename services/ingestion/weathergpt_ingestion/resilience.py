import json
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from redis.asyncio import Redis

from .models import QualityFlag, RawWeatherEvent

T = TypeVar("T")


class CircuitOpenError(RuntimeError):
    pass


@dataclass
class CircuitBreaker:
    name: str
    failure_threshold: int
    recovery_seconds: int
    failures: int = 0
    opened_at: float | None = None

    def allow_request(self) -> bool:
        if self.opened_at is None:
            return True
        if time.time() - self.opened_at >= self.recovery_seconds:
            self.failures = 0
            self.opened_at = None
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()

    async def call(self, operation: Callable[[], Awaitable[T]]) -> T:
        if not self.allow_request():
            raise CircuitOpenError(f"circuit breaker open for {self.name}")
        try:
            result = await operation()
        except Exception:
            self.record_failure()
            raise
        self.record_success()
        return result


class LastKnownGoodCache:
    def __init__(self, redis: Redis, ttl_seconds: int) -> None:
        self.redis = redis
        self.ttl_seconds = ttl_seconds

    async def get(self, key: str) -> RawWeatherEvent | None:
        raw = await self.redis.get(key)
        if raw is None:
            return None
        event = RawWeatherEvent.model_validate_json(raw)
        if QualityFlag.LAST_KNOWN_GOOD not in event.quality_flags:
            event.quality_flags.append(QualityFlag.LAST_KNOWN_GOOD)
        return event

    async def set(self, key: str, event: RawWeatherEvent) -> None:
        await self.redis.setex(key, self.ttl_seconds, event.model_dump_json())


def stable_json(payload: object) -> bytes:
    return json.dumps(payload, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")
