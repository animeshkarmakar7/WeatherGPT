import json
import re
from typing import TypedDict

from langgraph.graph import END, StateGraph

from .llm_client import LLMClient
from .models import IntentType, QueryClassification, StructuredWeatherResponse, WeatherAspect, WeatherDataFact, WeatherNarrativeResponse
from .query_service import WeatherDataQueryService
from .rag import RAGResponse, RAGSynthesizer


class AgentState(TypedDict):
    user_message: str
    session_id: str
    classification: QueryClassification | None
    weather_fact: WeatherDataFact | None
    rag_response: RAGResponse | None
    structured_response: StructuredWeatherResponse | None
    final_text: str | None


def _confidence_for_fact(fact: WeatherDataFact) -> float:
    required = [fact.location, fact.source, fact.target_date]
    completeness = sum(value is not None and value != "" for value in required) / len(required)
    freshness = {"fresh": 1.0, "stale": 0.75, "unknown": 0.5}.get(fact.freshness, 0.5)
    source_factor = 1.0 if fact.source in {"open_meteo", "timescaledb", "imd", "noaa"} else 0.0
    return round(min(1.0, completeness * freshness * source_factor), 2)


def create_weather_orchestrator(query_service: WeatherDataQueryService, llm_client: LLMClient, rag_synthesizer: RAGSynthesizer | None = None):
    async def query_classifier_node(state: AgentState) -> dict:
        raw_msg = state["user_message"]
        msg = raw_msg.lower()
        advisory_keywords = ["sop", "evacuat", "cyclone alert", "heatwave", "guideline", "regulation", "policy", "danger level", "warning stage", "warning level", "shelter", "protocol"]
        if any(k in msg for k in advisory_keywords):
            return {"classification": QueryClassification(intent=IntentType.ALERT, location=None, target_date="current", aspect=WeatherAspect.GENERAL, confidence=0.98)}
        known_cities = ["pune", "mumbai", "delhi", "kolkata", "chennai", "bengaluru", "nagpur", "hyderabad", "ahmedabad", "jaipur", "lucknow", "surat", "thane", "navi mumbai", "nashik", "kochi", "manali"]
        detected_location = next((city for city in known_cities if city in msg), None)
        if not detected_location:
            match = re.search(r"(?:in|of|at|for)\s+([a-zA-Z][a-zA-Z\s-]{2,60})", raw_msg, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                detected_location = re.split(r"\s+(?:tomorrow|today|now|right now|yesterday|this|next)\b", candidate, flags=re.IGNORECASE)[0].strip()
        if not detected_location:
            raise ValueError("A verified weather location is required")
        target_date = "tomorrow" if "tomorrow" in msg else "current"
        aspect = WeatherAspect.GENERAL
        if any(k in msg for k in ["rain", "precipitation", "shower"]):
            aspect = WeatherAspect.RAIN
        elif any(k in msg for k in ["temp", "temperature", "hot", "cold"]):
            aspect = WeatherAspect.TEMPERATURE
        elif "wind" in msg:
            aspect = WeatherAspect.WIND
        return {"classification": QueryClassification(intent=IntentType.WEATHER_FORECAST if target_date == "tomorrow" else IntentType.WEATHER_CURRENT, location=detected_location, target_date=target_date, aspect=aspect, confidence=0.95)}

    def route_by_intent(state: AgentState) -> str:
        classification = state.get("classification")
        return "rag_agent" if classification and classification.intent in (IntentType.ALERT, IntentType.GENERAL) else "weather_agent"

    async def weather_agent_node(state: AgentState) -> dict:
        classification = state["classification"]
        fact = await query_service.get_weather_data(location=classification.location or "", target_date=classification.target_date or "current")
        return {"weather_fact": fact}

    async def rag_agent_node(state: AgentState) -> dict:
        if rag_synthesizer is None:
            raise RuntimeError("Verified knowledge service is unavailable")
        try:
            return {"rag_response": await rag_synthesizer.answer_query(state["user_message"], top_k=5)}
        except Exception as exc:
            raise RuntimeError("Verified knowledge generation failed") from exc

    async def response_synthesis_node(state: AgentState) -> dict:
        rag_resp = state.get("rag_response")
        if rag_resp is not None:
            if not rag_resp.retrieval_success:
                return {"final_text": "No verified official evidence was sufficient to answer this request.", "structured_response": None}
            return {"final_text": rag_resp.answer, "structured_response": None}

        fact = state.get("weather_fact")
        if fact is None or fact.source == "unavailable":
            return {"final_text": "Verified weather data is unavailable for this request.", "structured_response": None}

        fact_payload = fact.model_dump(mode="json")
        prompt = "Generate only a concise natural-language summary of the supplied verified weather facts. Use exactly the supplied values and do not add, calculate, infer, estimate, round, or invent any weather fact. State whether the data is current or a forecast and include the source when relevant. Return only JSON matching the requested schema.\n\nFACTS:\n" + json.dumps(fact_payload, ensure_ascii=False)
        system_prompt = "You are WeatherGPT's grounded weather answer generator. You produce language from verified structured facts. Never fabricate, modify, estimate, or derive weather measurements."
        narrative = await llm_client.generate_structured(prompt, system_prompt, WeatherNarrativeResponse)
        structured = StructuredWeatherResponse(
            location=fact.location,
            target_date=fact.target_date,
            summary=narrative.summary,
            will_rain=fact.will_rain,
            precipitation_probability_pct=fact.precipitation_probability_pct,
            temp_c=fact.temp_c,
            temp_min_c=fact.temp_min_c,
            temp_max_c=fact.temp_max_c,
            humidity_pct=fact.humidity_pct,
            wind_speed_kph=fact.wind_speed_kph,
            conditions=fact.condition_description,
            confidence=_confidence_for_fact(fact),
            data_sources=[fact.source],
            quality_flags=[],
            observed_at=fact.observed_at,
            fetched_at=fact.fetched_at,
            freshness=fact.freshness,
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
