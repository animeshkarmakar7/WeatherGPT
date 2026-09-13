from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from .config import get_chat_settings
from .llm_client import LLMClient
from .models import ChatMessageRequest, ChatMessageResponse, IntentType
from .orchestrator import create_weather_orchestrator
from .query_service import WeatherDataQueryService
from .session_store import SessionStore
from .rag import (
    BGEM3Embedder,
    DocumentChunker,
    DocumentType,
    EvalMetricResult,
    HybridRetriever,
    LLMJudgeFaithfulness,
    MinioDocumentStore,
    QdrantVectorStoreClient,
    RAGEvaluator,
    RAGResponse,
    RAGSynthesizer,
    VectorStoreClient,
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_chat_settings()

    # Propagate feature flags to env so sub-modules pick them up
    if settings.use_real_embedder:
        os.environ.setdefault("WEATHERGPT_USE_REAL_EMBEDDER", "true")

    redis = None
    try:
        redis = Redis.from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
    except Exception:
        redis = None

    query_service = WeatherDataQueryService(settings, redis)
    await query_service.start()

    llm_client = LLMClient(settings)
    session_store = SessionStore(redis, settings.session_ttl_seconds)

    # ------------------------------------------------------------------
    # RAG knowledge-base components
    # ------------------------------------------------------------------
    embedder = BGEM3Embedder()  # respects WEATHERGPT_USE_REAL_EMBEDDER env

    if settings.use_qdrant:
        vector_store = QdrantVectorStoreClient(
            url=settings.qdrant_url,
            collection_name="weathergpt_knowledge_base",
        )
    else:
        vector_store = VectorStoreClient()

    retriever = HybridRetriever(vector_store, embedder)
    synthesizer = RAGSynthesizer(retriever)

    minio_store = MinioDocumentStore(
        endpoint=settings.minio_endpoint,
        access_key=settings.minio_access_key,
        secret_key=settings.minio_secret_key,
        secure=settings.minio_secure,
        bucket_name=settings.minio_bucket,
    )
    chunker = DocumentChunker()

    # Evaluator reuses the already-built retriever / synthesizer
    evaluator = RAGEvaluator(llm_base_url=settings.llm_base_url)
    evaluator.vector_store = vector_store
    evaluator.retriever = retriever
    evaluator.synthesizer = synthesizer
    evaluator.setup_benchmark_corpus()

    orchestrator = create_weather_orchestrator(
        query_service, llm_client, rag_synthesizer=synthesizer
    )

    app.state.settings = settings
    app.state.redis = redis
    app.state.query_service = query_service
    app.state.llm_client = llm_client
    app.state.session_store = session_store
    app.state.evaluator = evaluator
    app.state.orchestrator = orchestrator
    app.state.vector_store = vector_store
    app.state.retriever = retriever
    app.state.synthesizer = synthesizer
    app.state.minio_store = minio_store
    app.state.chunker = chunker
    app.state.embedder = embedder

    yield

    await query_service.close()
    await llm_client.close()
    if redis:
        await redis.aclose()


app = FastAPI(title="WeatherGPT Chat & RAG Service", version="0.4.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "weathergpt-chat-rag"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    v_store = getattr(app.state, "vector_store", None)
    if not v_store and hasattr(app.state, "evaluator"):
        v_store = getattr(app.state.evaluator, "vector_store", None)
    kb_count = len(v_store.get_all_chunks()) if v_store else 0

    checks = {
        "database": await app.state.query_service.ping(),
        "redis": bool(await app.state.redis.ping()) if getattr(app.state, "redis", None) else False,
        "knowledge_base": kb_count > 0,
    }
    return {"status": "ready" if any(checks.values()) else "degraded", "checks": checks}


@app.post("/api/v1/chat/message", response_model=ChatMessageResponse)
async def handle_message(req: ChatMessageRequest) -> ChatMessageResponse:
    session_id = req.session_id or str(uuid4())
    orchestrator = app.state.orchestrator
    session_store = app.state.session_store

    await session_store.add_message(session_id, "user", req.message)

    result = await orchestrator.ainvoke(
        {
            "user_message": req.message,
            "session_id": session_id,
            "classification": None,
            "weather_fact": None,
            "rag_response": None,
            "structured_response": None,
            "final_text": None,
        }
    )

    response_text = result.get("final_text") or "I was unable to retrieve that information."
    structured = result.get("structured_response")
    intent = (
        result["classification"].intent
        if result.get("classification")
        else IntentType.WEATHER_FORECAST
    )

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


@app.get("/api/v1/rag/eval", response_model=EvalMetricResult)
async def run_ragas_evaluation() -> EvalMetricResult:
    evaluator: RAGEvaluator = app.state.evaluator
    return evaluator.evaluate_production_baseline()


@app.post("/api/v1/rag/query", response_model=RAGResponse)
async def query_knowledge_base(query: str, top_k: int = 3) -> RAGResponse:
    synthesizer: RAGSynthesizer = app.state.synthesizer
    return synthesizer.answer_query(query, top_k=top_k)


@app.post("/api/v1/rag/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    doc_id: str = Form(...),
    doc_name: str = Form(...),
    doc_type: str = Form(default="government_sop"),
    region: str = Form(default="all"),
    language: str = Form(default="en"),
    doc_date: str = Form(default="2026-01-01"),
) -> dict[str, object]:
    """Upload a document to MinIO, chunk it, embed it, and upsert into Qdrant.

    Form fields
    -----------
    file      : The document file (plain text, markdown, or PDF extracted text).
    doc_id    : Unique document identifier (e.g. 'ndma-cyclone-sop-2026').
    doc_name  : Human-readable document name.
    doc_type  : One of government_sop | weather_bulletin | faq_advisory | climate_report.
    region    : Geographical region tag (e.g. 'coastal', 'all').
    language  : ISO 639-1 language code (default 'en').
    doc_date  : Document date in YYYY-MM-DD format.
    """
    try:
        doc_type_enum = DocumentType(doc_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid doc_type '{doc_type}'. Must be one of: "
                   + ", ".join(t.value for t in DocumentType),
        )

    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # 1. Store raw bytes in MinIO
    object_name = f"{doc_type}/{doc_id}/{file.filename or 'document.txt'}"
    minio_store: MinioDocumentStore = app.state.minio_store
    storage_uri = minio_store.upload_document(
        object_name,
        raw_bytes,
        content_type=file.content_type or "text/plain",
    )

    # 2. Decode text (UTF-8 with replacement for non-decodable bytes)
    text = raw_bytes.decode("utf-8", errors="replace")

    # 3. Chunk
    chunker: DocumentChunker = app.state.chunker
    chunks = chunker.chunk_document(
        doc_id=doc_id,
        doc_name=doc_name,
        doc_type=doc_type_enum,
        text=text,
        region=region,
        language=language,
        doc_date=doc_date,
    )

    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced zero chunks — too short?")

    # 4. Embed
    embedder = app.state.embedder
    vectors = embedder.embed_batch([c.content for c in chunks])

    # 5. Upsert into vector store (Qdrant or in-memory)
    vector_store = app.state.vector_store
    vector_store.insert_chunks(chunks, vectors)

    # 6. Rebuild BM25 index so retrieval is immediately consistent
    retriever: HybridRetriever = app.state.retriever
    retriever.build_bm25_index()

    return {
        "status": "ingested",
        "doc_id": doc_id,
        "doc_name": doc_name,
        "doc_type": doc_type,
        "chunks_created": len(chunks),
        "storage_uri": storage_uri,
        "knowledge_base_size": len(vector_store.get_all_chunks()),
    }
