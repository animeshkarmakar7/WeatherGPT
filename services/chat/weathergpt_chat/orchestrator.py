import logging
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, END

from .models import (
    IntentType,
    QueryClassification,
    StructuredWeatherResponse,
    WeatherDataFact,
    WeatherAspect,
)
from .query_service import WeatherDataQueryService
from .llm_client import LLMClient
from .rag import HybridRetriever, RAGResponse, RAGSynthesizer

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    user_message: str
    session_id: str
    classification: QueryClassification | None
    weather_fact: WeatherDataFact | None
    rag_response: RAGResponse | None
    structured_response: StructuredWeatherResponse | None
    final_text: str | None


def create_weather_orchestrator(
    query_service: WeatherDataQueryService,
    llm_client: LLMClient,
    rag_synthesizer: RAGSynthesizer | None = None,
):
    async def query_classifier_node(state: AgentState) -> dict:
        msg = state["user_message"].lower()

        advisory_keywords = [
            "sop",
            "evacuat",
            "cyclone alert",
            "heatwave",
            "guideline",
            "regulation",
            "policy",
            "danger level",
            "warning stage",
            "warning level",
            "shelter",
            "protocol",
        ]
        if any(k in msg for k in advisory_keywords):
            return {
                "classification": QueryClassification(
                    intent=IntentType.ALERT,
                    location=None,
                    target_date="current",
                    aspect=WeatherAspect.GENERAL,
                    confidence=0.98,
                )
            }

        location = "pune"
        for city in ["pune", "mumbai", "delhi", "kolkata", "chennai", "bengaluru"]:
            if city in msg:
                location = city
                break

        target_date = "tomorrow" if "tomorrow" in msg else "today"
        aspect = WeatherAspect.GENERAL
        if "rain" in msg or "precipitation" in msg or "shower" in msg:
            aspect = WeatherAspect.RAIN
        elif "temp" in msg or "hot" in msg or "cold" in msg:
            aspect = WeatherAspect.TEMPERATURE
        elif "wind" in msg:
            aspect = WeatherAspect.WIND

        classification = QueryClassification(
            intent=IntentType.WEATHER_FORECAST if target_date == "tomorrow" else IntentType.WEATHER_CURRENT,
            location=location,
            target_date=target_date,
            aspect=aspect,
            confidence=0.95,
        )
        return {"classification": classification}

    def route_by_intent(state: AgentState) -> str:
        classification = state.get("classification")
        if classification and classification.intent in (IntentType.ALERT, IntentType.GENERAL):
            return "rag_agent"
        return "weather_agent"

    async def weather_agent_node(state: AgentState) -> dict:
        classification = state["classification"]
        location = classification.location or "pune"
        target_date = classification.target_date or "tomorrow"
        fact = await query_service.get_weather_data(location=location, target_date=target_date)
        return {"weather_fact": fact}

    async def rag_agent_node(state: AgentState) -> dict:
        if rag_synthesizer:
            resp = rag_synthesizer.answer_query(state["user_message"], top_k=3)
            return {"rag_response": resp}
        return {"rag_response": None}

    async def response_synthesis_node(state: AgentState) -> dict:
        rag_resp = state.get("rag_response")
        if rag_resp and rag_resp.retrieval_success:
            return {
                "final_text": rag_resp.answer,
                "structured_response": None,
            }

        fact = state.get("weather_fact")
        if not fact:
            return {
                "final_text": "I was unable to retrieve that information from the data layer.",
                "structured_response": None,
            }

        rain_phrase = (
            f"Yes, it is expected to rain in {fact.location.capitalize()} {fact.target_date}."
            if fact.will_rain
            else f"No significant rain is expected in {fact.location.capitalize()} {fact.target_date}."
        )

        details = []
        if fact.temp_min_c is not None and fact.temp_max_c is not None:
            details.append(f"temperatures between {fact.temp_min_c}°C and {fact.temp_max_c}°C")
        elif fact.temp_c is not None:
            details.append(f"temperature around {fact.temp_c}°C")

        if fact.precipitation_probability_pct is not None:
            details.append(f"precipitation chance of {fact.precipitation_probability_pct:.0f}%")

        if fact.wind_speed_kph is not None:
            details.append(f"wind speed at {fact.wind_speed_kph} km/h")

        detail_text = ", with ".join(details)
        summary = f"{rain_phrase} Expect {fact.condition_description.lower()}, with {detail_text}."

        structured = StructuredWeatherResponse(
            location=fact.location.capitalize(),
            target_date=fact.target_date,
            summary=summary,
            will_rain=fact.will_rain,
            precipitation_probability_pct=fact.precipitation_probability_pct,
            temp_c=fact.temp_c,
            temp_min_c=fact.temp_min_c,
            temp_max_c=fact.temp_max_c,
            humidity_pct=fact.humidity_pct,
            wind_speed_kph=fact.wind_speed_kph,
            conditions=fact.condition_description,
            confidence=0.92,
            data_sources=[fact.source],
        )

        return {
            "structured_response": structured,
            "final_text": summary,
        }

    workflow = StateGraph(AgentState)
    workflow.add_node("query_classifier", query_classifier_node)
    workflow.add_node("weather_agent", weather_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("response_synthesis", response_synthesis_node)

    workflow.set_entry_point("query_classifier")
    workflow.add_conditional_edges(
        "query_classifier",
        route_by_intent,
        {
            "weather_agent": "weather_agent",
            "rag_agent": "rag_agent",
        },
    )
    workflow.add_edge("weather_agent", "response_synthesis")
    workflow.add_edge("rag_agent", "response_synthesis")
    workflow.add_edge("response_synthesis", END)

    return workflow.compile()
