from contextlib import asynccontextmanager
from hashlib import sha256
from typing import AsyncIterator
from uuid import uuid4

import fitz
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from redis.asyncio import Redis

from .config import get_chat_settings
from .llm_client import LLMClient
from .models import ChatMessageRequest, ChatMessageResponse, IntentType
from .orchestrator import create_weather_orchestrator
from .query_service import WeatherDataQueryService
from .session_store import SessionStore
from .rag import BGEM3Embedder, DocumentChunker, DocumentType, HybridRetriever, MinioDocumentStore, QdrantVectorStoreClient, RAGEvaluator, RAGResponse, RAGSynthesizer
from .rag.chunker import ParsedPage


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_chat_settings()
    redis = Redis.from_url(settings.redis_url, decode_responses=True)
    await redis.ping()
    query_service = WeatherDataQueryService(settings, redis)
    await query_service.start()
    llm_client = LLMClient(settings)
    if not await llm_client.health():
        await query_service.close()
        await llm_client.close()
        await redis.aclose()
        raise RuntimeError("Configured LLM service is unavailable")
    embedder = BGEM3Embedder(settings.bge_model_path, settings.bge_fp16, 1024)
    vector_store = QdrantVectorStoreClient(settings.qdrant_url, settings.qdrant_collection, 1024)
    vector_store.get_all_chunks()
    retriever = HybridRetriever(vector_store, embedder, rrf_k=60)
    retriever.build_bm25_index()
    synthesizer = RAGSynthesizer(retriever, settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_seconds, settings.rag_min_score)
    evaluator = RAGEvaluator(embedder, settings.llm_base_url, settings.llm_api_key, settings.llm_model, settings.llm_timeout_seconds)
    evaluator.setup_benchmark_corpus()
    session_store = SessionStore(redis, settings.session_ttl_seconds)
    minio_store = MinioDocumentStore(settings.minio_endpoint, settings.minio_access_key, settings.minio_secret_key, settings.minio_secure, settings.minio_bucket)
    chunker = DocumentChunker()
    orchestrator = create_weather_orchestrator(query_service, llm_client, rag_synthesizer=synthesizer)
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
    await redis.aclose()


app = FastAPI(title="WeatherGPT Chat & RAG Service", version="0.5.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "weathergpt-chat-rag"}


@app.get("/ready")
async def ready() -> dict[str, object]:
    checks = {"database": await app.state.query_service.ping(), "redis": bool(await app.state.redis.ping()), "qdrant": False, "llm": await app.state.llm_client.health(), "knowledge_base": False}
    try:
        chunks = app.state.vector_store.get_all_chunks()
        checks["qdrant"] = True
        checks["knowledge_base"] = len(chunks) > 0
    except Exception:
        pass
    required = all(bool(checks[key]) for key in ("database", "redis", "qdrant", "llm"))
    return {"status": "ready" if required else "degraded", "checks": checks}


@app.post("/api/v1/chat/message", response_model=ChatMessageResponse)
async def handle_message(req: ChatMessageRequest) -> ChatMessageResponse:
    session_id = req.session_id or str(uuid4())
    await app.state.session_store.add_message(session_id, "user", req.message)
    try:
        result = await app.state.orchestrator.ainvoke({"user_message": req.message, "session_id": session_id, "classification": None, "weather_fact": None, "rag_response": None, "structured_response": None, "final_text": None})
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    response_text = result.get("final_text") or "No verified answer is available for this request."
    structured = result.get("structured_response")
    intent = result["classification"].intent if result.get("classification") else IntentType.WEATHER_CURRENT
    await app.state.session_store.add_message(session_id, "assistant", response_text, extra=structured.model_dump(mode="json") if structured else None)
    return ChatMessageResponse(session_id=session_id, response_text=response_text, structured_data=structured, intent=intent)


@app.get("/api/v1/chat/history/{session_id}")
async def get_history(session_id: str) -> list[dict]:
    return await app.state.session_store.get_history(session_id)


@app.get("/api/v1/rag/eval")
async def run_rag_evaluation() -> object:
    return await app.state.evaluator.evaluate_production_baseline()


@app.post("/api/v1/rag/query", response_model=RAGResponse)
async def query_knowledge_base(query: str, top_k: int = 5) -> RAGResponse:
    if top_k < 1 or top_k > 20:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 20")
    return await app.state.synthesizer.answer_query(query, top_k=top_k)


@app.post("/api/v1/rag/ingest")
async def ingest_document(file: UploadFile = File(...), doc_id: str = Form(...), doc_name: str = Form(...), doc_type: str = Form(default="government_sop"), region: str = Form(default="all"), language: str = Form(default="en"), doc_date: str = Form(default="2026-01-01")) -> dict[str, object]:
    try:
        doc_type_enum = DocumentType(doc_type)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid doc_type '{doc_type}'") from exc
    raw_bytes = await file.read()
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    file_hash = sha256(raw_bytes).hexdigest()
    object_name = f"{doc_type}/{doc_id}/{file_hash}/{file.filename or 'document'}"
    storage_uri = app.state.minio_store.upload_document(object_name, raw_bytes, content_type=file.content_type or "application/octet-stream")
    if file.content_type == "application/pdf" or (file.filename or "").lower().endswith(".pdf"):
        try:
            with fitz.open(stream=raw_bytes, filetype="pdf") as pdf:
                parsed_pages = [ParsedPage(page_number=index, text=page.get_text("text").strip()) for index, page in enumerate(pdf, start=1)]
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"PDF parsing failed: {exc}") from exc
    else:
        try:
            parsed_pages = [ParsedPage(page_number=1, text=raw_bytes.decode("utf-8"))]
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Only UTF-8 text or PDF documents are supported") from exc
    chunks = app.state.chunker.chunk_document(doc_id=doc_id, doc_name=doc_name, doc_type=doc_type_enum, text="\n".join(page.text for page in parsed_pages), region=region, language=language, doc_date=doc_date, pages=parsed_pages)
    if not chunks:
        raise HTTPException(status_code=422, detail="Document produced zero chunks")
    vectors = app.state.embedder.embed_batch([chunk.content for chunk in chunks])
    app.state.vector_store.insert_chunks(chunks, vectors)
    app.state.retriever.build_bm25_index()
    return {"status": "ingested", "doc_id": doc_id, "doc_name": doc_name, "doc_type": doc_type, "sha256": file_hash, "chunks_created": len(chunks), "storage_uri": storage_uri, "knowledge_base_size": len(app.state.vector_store.get_all_chunks())}
