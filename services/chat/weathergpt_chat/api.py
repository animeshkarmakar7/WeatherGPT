from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from .config import get_chat_settings
from .llm_client import LLMClient
from .models import ChatMessageRequest, ChatMessageResponse, IntentType
from .orchestrator import create_weather_orchestrator
from .query_service import WeatherDataQueryService
from .session_store import SessionStore


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_chat_settings()

    # Redis
    redis = None
    try:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
    except Exception:
        redis = None

    # Query service & LLM client
    query_service = WeatherDataQueryService(settings, redis)
    await query_service.start()

    llm_client = LLMClient(settings)
    session_store = SessionStore(redis, settings.session_ttl_seconds)

    orchestrator = create_weather_orchestrator(query_service, llm_client)

    app.state.settings = settings
    app.state.redis = redis
    app.state.query_service = query_service
    app.state.llm_client = llm_client
    app.state.session_store = session_store
    app.state.orchestrator = orchestrator

    yield

    await query_service.close()
    await llm_client.close()
    if redis:
        await redis.aclose()


app = FastAPI(title="WeatherGPT Chat Service", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "weathergpt-chat"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    checks = {
        "database": await app.state.query_service.ping(),
        "redis": bool(await app.state.redis.ping()) if app.state.redis else False,
    }
    return {"status": "ready" if any(checks.values()) else "degraded", "checks": checks}


@app.post("/api/v1/chat/message", response_model=ChatMessageResponse)
async def handle_message(req: ChatMessageRequest) -> ChatMessageResponse:
    session_id = req.session_id or str(uuid4())
    orchestrator = app.state.orchestrator
    session_store = app.state.session_store

    # 1. Log incoming user query to session store
    await session_store.add_message(session_id, "user", req.message)

    # 2. Invoke LangGraph Orchestrator
    result = await orchestrator.ainvoke(
        {
            "user_message": req.message,
            "session_id": session_id,
            "classification": None,
            "weather_fact": None,
            "structured_response": None,
            "final_text": None,
        }
    )

    response_text = result.get("final_text") or "I was unable to retrieve that weather information."
    structured = result.get("structured_response")
    intent = (
        result["classification"].intent
        if result.get("classification")
        else IntentType.WEATHER_FORECAST
    )

    # 3. Store response
    await session_store.add_message(
        session_id,
        "assistant",
        response_text,
        extra=structured.model_dump(mode="json") if structured else None,
    )

    return ChatMessageResponse(
        session_id=session_id,
        response_text=response_text,
        structured_data=structured,
        intent=intent,
    )


@app.get("/api/v1/chat/history/{session_id}")
async def get_history(session_id: str) -> list[dict]:
    return await app.state.session_store.get_history(session_id)
