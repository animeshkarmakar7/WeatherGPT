import pytest

from weathergpt_chat.models import IntentType, WeatherDataFact
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
        )


class FakeLLM:
    async def generate_structured(self, prompt, system_prompt, response_model):
        return response_model(
            location="Pune",
            target_date="current",
            summary="Current temperature in Pune is 28.0°C.",
            will_rain=False,
            precipitation_probability_pct=10.0,
            temp_c=28.0,
            temp_min_c=28.0,
            temp_max_c=28.0,
            humidity_pct=70.0,
            wind_speed_kph=12.0,
            conditions="Mainly clear",
            confidence=0.95,
            data_sources=["test"],
        )


@pytest.mark.asyncio
async def test_weather_query_returns_structured_fact():
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
    assert result["structured_response"].location == "Pune"
