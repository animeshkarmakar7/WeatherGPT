import json
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
        location = detected_location
        if not location:
            raise ValueError("A weather location is required")
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
        fact = await query_service.get_weather_data(location=classification.location or "", target_date=classification.target_date or "current")
        return {"weather_fact": fact}

    async def rag_agent_node(state: AgentState) -> dict:
        if rag_synthesizer is None:
            return {"rag_response": RAGResponse(query=state["user_message"], answer="The verified knowledge service is unavailable.", citations=[], confidence=0.0, retrieval_success=False)}
        return {"rag_response": await rag_synthesizer.answer_query(state["user_message"], top_k=5)}

    async def response_synthesis_node(state: AgentState) -> dict:
        rag_resp = state.get("rag_response")
        if rag_resp is not None:
            return {"final_text": rag_resp.answer, "structured_response": None}

        fact = state.get("weather_fact")
        if fact is None or getattr(fact, "source", "") == "unavailable":
            return {"final_text": "Verified weather data is unavailable for this request.", "structured_response": None}

        fact_payload = {
            "location": fact.location,
            "target_date": fact.target_date,
            "latitude": fact.latitude,
            "longitude": fact.longitude,
            "temperature_c": fact.temp_c,
            "temperature_min_c": fact.temp_min_c,
            "temperature_max_c": fact.temp_max_c,
            "humidity_pct": fact.humidity_pct,
            "precipitation_mm": fact.precipitation_mm,
            "precipitation_probability_pct": fact.precipitation_probability_pct,
            "wind_speed_kph": fact.wind_speed_kph,
            "weather_code": fact.weather_code,
            "condition": fact.condition_description,
            "will_rain": fact.will_rain,
            "source": fact.source,
        }
        prompt = (
            "Generate a concise weather response using only the supplied weather facts. "
            "Do not change or invent any numeric value. Preserve null values as null. "
            "State clearly whether the data is current or a forecast. "
            "Set confidence to a value between 0 and 1 based only on the completeness of the supplied facts. "
            "Never claim a source that is not in the supplied facts.\n\n"
            f"FACTS:\n{json.dumps(fact_payload, ensure_ascii=False)}"
        )
        system_prompt = (
            "You are WeatherGPT. You are a grounded weather answer generator. "
            "The supplied facts are authoritative for this response. "
            "You must not fabricate, extrapolate, or estimate measurements. "
            "Return a valid StructuredWeatherResponse JSON object."
        )
        try:
            structured = await llm_client.generate_structured(prompt, system_prompt, StructuredWeatherResponse)
        except RuntimeError:
            return {"final_text": "The language model is unavailable. Verified weather data was retrieved but no generated answer can be returned.", "structured_response": None}
        if structured.location.lower() != fact.location.lower():
            return {"final_text": "The generated weather response failed validation and was withheld.", "structured_response": None}
        structured.data_sources = [fact.source]
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
