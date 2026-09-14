from datetime import datetime
from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field


class DocumentType(StrEnum):
    GOVERNMENT_SOP = "government_sop"
    WEATHER_BULLETIN = "weather_bulletin"
    FAQ_ADVISORY = "faq_advisory"
    CLIMATE_REPORT = "climate_report"


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    doc_name: str
    doc_type: DocumentType
    content: str
    page_number: int = 1
    section_title: str = "General"
    region: str = "all"
    language: str = "en"
    confidence_tier: str = "authoritative"
    doc_date: str = "2026-01-01"
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    chunk: DocumentChunk
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


class Citation(BaseModel):
    doc_name: str
    page: int
    section: str
    excerpt: str


class RAGResponse(BaseModel):
    query: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    confidence: float
    retrieval_success: bool


class EvalMetricResult(BaseModel):
    context_relevance: float
    faithfulness: float
    answer_relevance: float
    citation_groundedness: float
    umbrela_score: float
    retrieval_recall_at_5: float = 0.0
    retrieval_mrr: float = 0.0
    retrieval_hit_rate: float = 0.0
    answer_correctness: float = 0.0
    passed: bool
    state_scores: dict[str, float]
