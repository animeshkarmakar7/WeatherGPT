import logging
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .llm_client import LLMClient
from .models import IntentType, QueryClassification, StructuredWeatherResponse, WeatherAspect
from .query_service import WeatherDataQueryService
from .rag import RAGResponse, RAGSynthesizer

logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    user_message: str
    session_id: str
    classification: QueryClassification | None
    weather_fact: object | None
    rag_response: RAGResponse | None
    structured_response: StructuredWeatherResponse | None
    final_text: str | None


def create_weather_orchestrator(query_service: WeatherDataQueryService, llm_client: LLMClient, rag_synthesizer: RAGSynthesizer | None = None):
    async def query_classifier_node(state: AgentState) -> dict:
        raw_msg = state["user_message"]
        msg = raw_msg.lower()
        advisory_keywords = ["sop", "evacuat", "cyclone alert", "heatwave", "guideline", "regulation", "policy", "danger level", "warning stage", "warning level", "shelter", "protocol"]
        if any(k in msg for k in advisory_keywords):
            return {"classification": QueryClassification(intent=IntentType.ALERT, location=None, target_date="current", aspect=WeatherAspect.GENERAL, confidence=0.98)}
        known_cities = ["pune", "mumbai", "delhi", "kolkata", "chennai", "bengaluru", "nagpur", "hyderabad", "ahmedabad", "jaipur", "lucknow", "surat", "thane", "navi mumbai", "nashik", "kochi"]
        detected_location = next((city for city in known_cities if city in msg), None)
        if not detected_location:
            match = re.search(r"(?:in|of|at|for)\s+([a-zA-Z\s]{3,40})", raw_msg, re.IGNORECASE)
            if match:
                detected_location = re.split(r"\s+(?:tomorrow|today|now|right now|yesterday|this|next)\b", match.group(1).strip(), flags=re.IGNORECASE)[0].strip()
        location = detected_location or "pune"
        target_date = "tomorrow" if "tomorrow" in msg else "current"
        aspect = WeatherAspect.GENERAL
        if any(k in msg for k in ["rain", "precipitation", "shower"]):
            aspect = WeatherAspect.RAIN
        elif any(k in msg for k in ["temp", "temperature", "hot", "cold"]):
            aspect = WeatherAspect.TEMPERATURE
        elif "wind" in msg:
            aspect = WeatherAspect.WIND
        return {"classification": QueryClassification(intent=IntentType.WEATHER_FORECAST if target_date == "tomorrow" else IntentType.WEATHER_CURRENT, location=location, target_date=target_date, aspect=aspect, confidence=0.95)}

    def route_by_intent(state: AgentState) -> str:
        classification = state.get("classification")
        return "rag_agent" if classification and classification.intent in (IntentType.ALERT, IntentType.GENERAL) else "weather_agent"

    async def weather_agent_node(state: AgentState) -> dict:
        classification = state["classification"]
        fact = await query_service.get_weather_data(location=classification.location or "pune", target_date=classification.target_date or "current")
        return {"weather_fact": fact}

    async def rag_agent_node(state: AgentState) -> dict:
        if rag_synthesizer is None:
            return {"rag_response": RAGResponse(query=state["user_message"], answer="The verified knowledge service is unavailable.", citations=[], confidence=0.0, retrieval_success=False)}
        return {"rag_response": await rag_synthesizer.answer_query(state["user_message"], top_k=5)}

    async def response_synthesis_node(state: AgentState) -> dict:
        rag_resp = state.get("rag_response")
        if rag_resp is not None:
            return {"final_text": rag_resp.answer if rag_resp.retrieval_success else rag_resp.answer, "structured_response": None}
        fact = state.get("weather_fact")
        if fact is None:
            return {"final_text": "Verified weather data is unavailable for this request.", "structured_response": None}
        structured = StructuredWeatherResponse(
            location=fact.location.capitalize(),
            target_date=fact.target_date,
            summary=(f"Current temperature in {fact.location.capitalize()} is {fact.temp_c}°C." if fact.target_date in ("current", "today", "now") else f"For {fact.location.capitalize()} {fact.target_date}, temperatures are expected between {fact.temp_min_c}°C and {fact.temp_max_c}°C."),
            will_rain=fact.will_rain,
            precipitation_probability_pct=fact.precipitation_probability_pct,
            temp_c=fact.temp_c,
            temp_min_c=fact.temp_min_c,
            temp_max_c=fact.temp_max_c,
            humidity_pct=fact.humidity_pct,
            wind_speed_kph=fact.wind_speed_kph,
            conditions=fact.condition_description,
            confidence=0.0 if fact.source == "unavailable" else 0.95,
            data_sources=[fact.source],
        )
        return {"structured_response": structured, "final_text": structured.summary}

    workflow = StateGraph(AgentState)
    workflow.add_node("query_classifier", query_classifier_node)
    workflow.add_node("weather_agent", weather_agent_node)
    workflow.add_node("rag_agent", rag_agent_node)
    workflow.add_node("response_synthesis", response_synthesis_node)
    workflow.set_entry_point("query_classifier")
    workflow.add_conditional_edges("query_classifier", route_by_intent, {"weather_agent": "weather_agent", "rag_agent": "rag_agent"})
    workflow.add_edge("weather_agent", "response_synthesis")
    workflow.add_edge("rag_agent", "response_synthesis")
    workflow.add_edge("response_synthesis", END)
    return workflow.compile()
