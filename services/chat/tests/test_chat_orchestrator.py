import pytest

from weathergpt_chat.models import IntentType, WeatherDataFact, WeatherNarrativeResponse
from weathergpt_chat.orchestrator import create_weather_orchestrator


class FakeQueryService:
    async def get_weather_data(self, location: str, target_date: str):
        return WeatherDataFact(
            location=location,
            latitude=18.5204,
            longitude=73.8567,
            target_date=target_date,
            temp_c=28.0,
            temp_min_c=24.0,
            temp_max_c=30.0,
            humidity_pct=70.0,
            precipitation_mm=0.0,
            precipitation_probability_pct=10.0,
            wind_speed_kph=12.0,
            weather_code="1",
            condition_description="Mainly clear",
            will_rain=False,
            source="test",
            source_url="https://example.test/weather",
            freshness="fresh",
        )


class FakeLLM:
    async def generate_structured(self, prompt, system_prompt, response_model):
        assert response_model is WeatherNarrativeResponse
        return response_model(summary="Current temperature in Pune is 28.0°C.")


@pytest.mark.asyncio
async def test_weather_query_returns_source_derived_structured_data():
    orchestrator = create_weather_orchestrator(FakeQueryService(), FakeLLM())
    result = await orchestrator.ainvoke(
        {
            "user_message": "What is the temperature in Pune right now?",
            "session_id": "test",
            "classification": None,
            "weather_fact": None,
            "rag_response": None,
            "structured_response": None,
            "final_text": None,
        }
    )
    assert result["classification"].intent == IntentType.WEATHER_CURRENT
    assert result["structured_response"].temp_c == 28.0
    assert result["structured_response"].temp_min_c == 24.0
    assert result["structured_response"].temp_max_c == 30.0
    assert result["structured_response"].will_rain is False
    assert result["structured_response"].data_sources == ["test"]
    assert result["structured_response"].freshness == "fresh"
    assert result["final_text"] == "Current temperature in Pune is 28.0°C."
