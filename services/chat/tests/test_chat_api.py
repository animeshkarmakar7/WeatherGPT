import pytest

from weathergpt_chat.api import health


@pytest.mark.asyncio
async def test_health_endpoint():
    result = await health()
    assert result == {"status": "ok", "service": "weathergpt-chat-rag"}
