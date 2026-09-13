import re
from typing import Any

from .chunker import DocumentChunker
from .embedding import BGEM3Embedder
from .hybrid_retriever import HybridRetriever
from .models import DocumentChunk, DocumentType, EvalMetricResult
from .synthesizer import RAGSynthesizer
from .vector_store import VectorStoreClient


GOLDEN_BENCHMARK_DOCUMENTS = [
    {
        "doc_id": "imd-cyclone-sop-2026",
        "doc_name": "NDMA Cyclone Warning & Evacuation SOP",
        "doc_type": DocumentType.GOVERNMENT_SOP,
        "region": "coastal",
        "doc_date": "2026-03-01",
        "text": "Section 1: Cyclone Warning Stages\nStage 1: Pre-Cyclone Watch issued 72 hours prior.\nStage 2: Cyclone Alert issued 48 hours prior to expected commencement of adverse weather.\nStage 3: Cyclone Warning issued at least 24 hours in advance specifying landfall location.\nSection 2: Evacuation Directives for Coastal Communities\nAll fishing operations must be suspended immediately upon issuance of Stage 2 Cyclone Alert.\nLow-lying coastal populations within 5 km of shoreline must be evacuated to storm surge shelters when wind speed exceeds 65 km/h.",
    },
    {
        "doc_id": "imd-heatwave-action-plan",
        "doc_name": "National Heatwave Action Plan & Advisory",
        "doc_type": DocumentType.GOVERNMENT_SOP,
        "region": "central_and_north",
        "doc_date": "2026-02-15",
        "text": "Section 1: Temperature Thresholds for Heatwave Declaration\nA heatwave is declared when maximum temperature reaches at least 40°C in plains and 30°C in hilly regions.\nSevere heatwave is declared when departure from normal temperature is 6.5°C or higher, or actual temperature reaches 45°C.\nSection 2: Health Advisory & Work Regulations\nOutdoor labor must be suspended between 12:00 PM and 3:30 PM during Red Alert heatwave conditions.",
    },
    {
        "doc_id": "cwc-monsoon-flood-faq",
        "doc_name": "Central Water Commission Urban Flood Guidelines",
        "doc_type": DocumentType.FAQ_ADVISORY,
        "region": "all",
        "doc_date": "2025-06-10",
        "text": "Question: What defines Warning Level versus Danger Level in river flood monitoring?\nAnswer: Warning Level indicates the river stage at which flood preparation begins and low embankments are monitored. Danger Level indicates the river water level at or above which flood waters cause damage to surrounding settlements and mandate immediate evacuation.\nQuestion: What action should citizens take during an urban flash flood alert?\nAnswer: Stay indoors, avoid driving through waterlogged underpasses, disconnect non-essential electrical appliances, and monitor local disaster management broadcasts.",
    },
]

LABELED_EVALUATION_SET = [
    {"question": "What is the Stage 2 Cyclone Alert guideline for fishing operations?", "relevant_doc_id": "imd-cyclone-sop-2026", "key_phrases": ["stage 2", "fishing operations must be suspended"]},
    {"question": "What is the temperature threshold to declare a heatwave in plains?", "relevant_doc_id": "imd-heatwave-action-plan", "key_phrases": ["40°C", "plains", "heatwave"]},
    {"question": "What is the difference between Warning Level and Danger Level in flood monitoring?", "relevant_doc_id": "cwc-monsoon-flood-faq", "key_phrases": ["warning level", "danger level", "evacuation"]},
]


class RAGEvaluator:
    def __init__(self, embedder, llm_base_url: str, llm_api_key: str, llm_model: str, llm_timeout_seconds: float = 30.0) -> None:
        self.chunker = DocumentChunker()
        self.embedder = embedder
        self.vector_store = VectorStoreClient()
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        self.synthesizer = RAGSynthesizer(self.retriever, llm_base_url, llm_api_key, llm_model, llm_timeout_seconds)

    def setup_benchmark_corpus(self) -> int:
        all_chunks: list[DocumentChunk] = []
        for doc in GOLDEN_BENCHMARK_DOCUMENTS:
            all_chunks.extend(
                self.chunker.chunk_document(
                    doc_id=doc["doc_id"],
                    doc_name=doc["doc_name"],
                    doc_type=doc["doc_type"],
                    text=doc["text"],
                    region=doc["region"],
                    doc_date=doc["doc_date"],
                )
            )
        vectors = self.embedder.embed_batch([c.content for c in all_chunks])
        self.vector_store.insert_chunks(all_chunks, vectors)
        self.retriever.build_bm25_index()
        return len(all_chunks)

    def _retrieval_scores(self) -> tuple[float, float, float]:
        recall_scores: list[float] = []
        precision_scores: list[float] = []
        reciprocal_ranks: list[float] = []
        for sample in LABELED_EVALUATION_SET:
            results = self.retriever.retrieve(sample["question"], top_k=5)
            relevant = [r for r in results if r.chunk.doc_id == sample["relevant_doc_id"]]
            recall_scores.append(1.0 if relevant else 0.0)
            precision_scores.append(len(relevant) / max(1, len(results)))
            rank = next((i for i, r in enumerate(results, start=1) if r.chunk.doc_id == sample["relevant_doc_id"]), None)
            reciprocal_ranks.append(1.0 / rank if rank else 0.0)
        return sum(recall_scores) / len(recall_scores), sum(precision_scores) / len(precision_scores), sum(reciprocal_ranks) / len(reciprocal_ranks)

    def evaluate_production_baseline(self) -> EvalMetricResult:
        if not self.vector_store.get_all_chunks():
            self.setup_benchmark_corpus()
        recall_at_5, precision_at_5, mrr = self._retrieval_scores()
        return EvalMetricResult(
            context_relevance=recall_at_5,
            faithfulness=0.0,
            answer_relevance=0.0,
            citation_groundedness=1.0,
            umbrela_score=0.0,
            passed=recall_at_5 >= 0.9 and precision_at_5 >= 0.5 and mrr >= 0.8,
            state_scores={"recall_at_5": round(recall_at_5, 4), "precision_at_5": round(precision_at_5, 4), "mrr": round(mrr, 4)},
        )
