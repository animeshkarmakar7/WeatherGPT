import pytest

from weathergpt_ingestion.resilience import CircuitBreaker, CircuitOpenError


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_threshold():
    breaker = CircuitBreaker("upstream", failure_threshold=2, recovery_seconds=60)

    async def failing_call():
        raise RuntimeError("upstream unavailable")

    with pytest.raises(RuntimeError):
        await breaker.call(failing_call)
    with pytest.raises(RuntimeError):
        await breaker.call(failing_call)
    with pytest.raises(CircuitOpenError):
        await breaker.call(failing_call)
