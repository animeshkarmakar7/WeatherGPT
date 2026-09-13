import pytest
from httpx import ASGITransport, AsyncClient
from weathergpt_chat.api import app
from weathergpt_chat.config import ChatSettings
from weathergpt_chat.query_service import WeatherDataQueryService
from weathergpt_chat.llm_client import LLMClient
from weathergpt_chat.session_store import SessionStore
from weathergpt_chat.orchestrator import create_weather_orchestrator
from weathergpt_chat.rag import RAGEvaluator


@pytest.mark.asyncio
async def test_chat_and_rag_api_endpoints():
    settings = ChatSettings()
    query_service = WeatherDataQueryService(settings)
    llm_client = LLMClient(settings)
    session_store = SessionStore(None)
    evaluator = RAGEvaluator()
    evaluator.setup_benchmark_corpus()
    orchestrator = create_weather_orchestrator(
        query_service, llm_client, rag_synthesizer=evaluator.synthesizer
    )

    app.state.settings = settings
    app.state.query_service = query_service
    app.state.llm_client = llm_client
    app.state.session_store = session_store
    app.state.evaluator = evaluator
    app.state.orchestrator = orchestrator
    app.state.redis = None

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["service"] == "weathergpt-chat-rag"

        resp = await client.get("/ready")
        assert resp.status_code == 200

        resp = await client.post(
            "/api/v1/chat/message",
            json={"message": "will it rain in Pune tomorrow"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "pune" in data["response_text"].lower()

        resp_rag = await client.post(
            "/api/v1/chat/message",
            json={"message": "What is the cyclone alert stage 2 guideline for fishing?"},
        )
        assert resp_rag.status_code == 200
        data_rag = resp_rag.json()
        assert "fishing" in data_rag["response_text"].lower()

        resp_eval = await client.get("/api/v1/rag/eval")
        assert resp_eval.status_code == 200
        eval_data = resp_eval.json()
        assert eval_data["passed"] is True
        assert eval_data["state_scores"]["context_relevance"] >= 0.80
        assert eval_data["state_scores"]["faithfulness"] >= 0.85
