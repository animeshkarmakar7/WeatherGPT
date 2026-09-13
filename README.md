# WeatherGPT

Production-oriented conversational weather intelligence platform.

## Trust Model

Weather facts come only from verified ingestion or live upstream providers. The system does not generate synthetic weather values. Government guidance comes only from ingested official documents. The LLM generates language from supplied structured weather facts or retrieved document evidence and does not act as a source of truth.

## Phase 1

- Open-Meteo, NOAA/NWS and IMD ingestion connectors
- Kafka raw and normalized event contracts
- At-least-once Kafka consumption with idempotent TimescaleDB writes
- TimescaleDB and PostGIS weather storage
- Spark Structured Streaming staging path
- Circuit breaking and last-known-good handling without synthetic values

## Phase 2

- FastAPI chat service
- LangGraph query orchestration
- TimescaleDB-backed weather queries
- Strict OpenAI-compatible LLM contract through vLLM
- React frontend

## Phase 3

- Government document upload to MinIO
- PDF page-aware extraction with PyMuPDF
- Section-aware chunking
- BGE-M3 embeddings
- Qdrant vector storage
- BM25 plus dense hybrid retrieval
- Optional cross-encoder reranking
- LLM grounded RAG synthesis with citation IDs
- RAGAS retrieval and generation evaluation

## Local Services

```text
8081  ingestion API
8082  chat API
3000  frontend
5432  TimescaleDB/PostGIS
6333  Qdrant
6379  Redis
8000  vLLM when running locally
9000  MinIO API
9001  MinIO console
9092  Kafka
```

## Start Infrastructure

```bash
docker compose up --build
```

Run a local vLLM server for the chat service:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --host 0.0.0.0 --port 8000 --api-key dummy-vllm-key --generation-config vllm
```

Or use the included vLLM compose file on a GPU host:

```bash
docker compose -f docker-compose.yml -f docker-compose.llm.yml up --build
```

The chat service intentionally fails startup when the configured LLM is unavailable. It does not silently substitute a fake model.

## Phase 1 Check

```bash
curl -X POST "http://localhost:8081/ingest/current?city=mumbai"
curl "http://localhost:8081/observations/current?city=mumbai"
```

## RAG Document Ingestion

```bash
curl -X POST "http://localhost:8082/api/v1/rag/ingest" \
  -F "file=@official-cyclone-sop.pdf" \
  -F "doc_id=ndma-cyclone-sop-2026" \
  -F "doc_name=NDMA Cyclone SOP" \
  -F "doc_type=government_sop" \
  -F "region=coastal" \
  -F "language=en" \
  -F "doc_date=2026-03-01"
```

Query the knowledge base:

```bash
curl -X POST "http://localhost:8082/api/v1/rag/query?query=What%20does%20a%20Stage%202%20cyclone%20alert%20require%3F"
```

Run the RAG evaluation:

```bash
curl "http://localhost:8082/api/v1/rag/eval"
```
