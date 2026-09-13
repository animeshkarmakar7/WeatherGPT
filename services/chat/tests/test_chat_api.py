import pytest
from httpx import ASGITransport, AsyncClient
from weathergpt_chat.api import app
from weathergpt_chat.config import ChatSettings
from weathergpt_chat.query_service import WeatherDataQueryService
from weathergpt_chat.llm_client import LLMClient
from weathergpt_chat.session_store import SessionStore
from weathergpt_chat.orchestrator import create_weather_orchestrator


@pytest.mark.asyncio
async def test_chat_api_endpoints():
    settings = ChatSettings()
    query_service = WeatherDataQueryService(settings)
    llm_client = LLMClient(settings)
    session_store = SessionStore(None)
    orchestrator = create_weather_orchestrator(query_service, llm_client)

    app.state.settings = settings
    app.state.query_service = query_service
    app.state.llm_client = llm_client
    app.state.session_store = session_store
    app.state.orchestrator = orchestrator
    app.state.redis = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Health check
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "weathergpt-chat"

        # Ready check
        resp = await client.get("/ready")
        assert resp.status_code == 200

        # Chat message query (Exit criteria: "will it rain in Pune tomorrow")
        resp = await client.post(
            "/api/v1/chat/message",
            json={"message": "will it rain in Pune tomorrow"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] is not None
        assert "pune" in data["response_text"].lower()
        assert data["structured_data"] is not None
        assert data["structured_data"]["location"] == "Pune"
        assert data["structured_data"]["target_date"] == "tomorrow"
        assert isinstance(data["structured_data"]["will_rain"], bool)

        # History check
        session_id = data["session_id"]
        resp_hist = await client.get(f"/api/v1/chat/history/{session_id}")
        assert resp_hist.status_code == 200
        history = resp_hist.json()
        assert len(history) == 2  # user + assistant
