import pytest
from weathergpt_chat.models import IntentType, WeatherAspect
from weathergpt_chat.orchestrator import create_weather_orchestrator
from weathergpt_chat.query_service import WeatherDataQueryService
from weathergpt_chat.llm_client import LLMClient
from weathergpt_chat.config import ChatSettings


@pytest.mark.asyncio
async def test_end_to_end_pune_rain_query():
    settings = ChatSettings()
    query_service = WeatherDataQueryService(settings)
    llm_client = LLMClient(settings)
    orchestrator = create_weather_orchestrator(query_service, llm_client)

    result = await orchestrator.ainvoke(
        {
            "user_message": "will it rain in Pune tomorrow",
            "session_id": "test-session-1",
            "classification": None,
            "weather_fact": None,
            "structured_response": None,
            "final_text": None,
        }
    )

    classification = result["classification"]
    assert classification.location == "pune"
    assert classification.target_date == "tomorrow"
    assert classification.aspect == WeatherAspect.RAIN
    assert classification.intent == IntentType.WEATHER_FORECAST

    structured = result["structured_response"]
    assert structured is not None
    assert structured.location == "Pune"
    assert structured.target_date == "tomorrow"
    assert isinstance(structured.will_rain, bool)
    assert structured.precipitation_probability_pct is not None
    assert len(structured.data_sources) > 0
    assert "Pune" in result["final_text"]
