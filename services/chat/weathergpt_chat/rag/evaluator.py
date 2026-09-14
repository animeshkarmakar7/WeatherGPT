import asyncio
from statistics import mean

from .chunker import DocumentChunker
from .embedding import BGEM3Embedder
from .hybrid_retriever import HybridRetriever
from .models import DocumentChunk, DocumentType, EvalMetricResult
from .synthesizer import RAGSynthesizer
from .vector_store import VectorStoreClient


GOLDEN_BENCHMARK_DOCUMENTS = [
    {"doc_id": "imd-cyclone-sop-2026", "doc_name": "NDMA Cyclone Warning & Evacuation SOP", "doc_type": DocumentType.GOVERNMENT_SOP, "region": "coastal", "doc_date": "2026-03-01", "text": "Section 1: Cyclone Warning Stages\nStage 1: Pre-Cyclone Watch issued 72 hours prior.\nStage 2: Cyclone Alert issued 48 hours prior to expected commencement of adverse weather.\nStage 3: Cyclone Warning issued at least 24 hours in advance specifying landfall location.\nSection 2: Evacuation Directives for Coastal Communities\nAll fishing operations must be suspended immediately upon issuance of Stage 2 Cyclone Alert.\nLow-lying coastal populations within 5 km of shoreline must be evacuated to storm surge shelters when wind speed exceeds 65 km/h."},
    {"doc_id": "imd-heatwave-action-plan", "doc_name": "National Heatwave Action Plan & Advisory", "doc_type": DocumentType.GOVERNMENT_SOP, "region": "central_and_north", "doc_date": "2026-02-15", "text": "Section 1: Temperature Thresholds for Heatwave Declaration\nA heatwave is declared when maximum temperature reaches at least 40°C in plains and 30°C in hilly regions.\nSevere heatwave is declared when departure from normal temperature is 6.5°C or higher, or actual temperature reaches 45°C.\nSection 2: Health Advisory & Work Regulations\nOutdoor labor must be suspended between 12:00 PM and 3:30 PM during Red Alert heatwave conditions."},
    {"doc_id": "cwc-monsoon-flood-faq", "doc_name": "Central Water Commission Urban Flood Guidelines", "doc_type": DocumentType.FAQ_ADVISORY, "region": "all", "doc_date": "2025-06-10", "text": "Question: What defines Warning Level versus Danger Level in river flood monitoring?\nAnswer: Warning Level indicates the river stage at which flood preparation begins and low embankments are monitored. Danger Level indicates the river water level at or above which flood waters cause damage to surrounding settlements and mandate immediate evacuation.\nQuestion: What action should citizens take during an urban flash flood alert?\nAnswer: Stay indoors, avoid driving through waterlogged underpasses, disconnect non-essential electrical appliances, and monitor local disaster management broadcasts."},
]

LABELED_EVALUATION_SET = [
    {"question": "What is the Stage 2 Cyclone Alert guideline for fishing operations?", "reference": "Stage 2 Cyclone Alert is issued 48 hours prior to expected adverse weather and all fishing operations must be suspended immediately.", "relevant_doc_id": "imd-cyclone-sop-2026"},
    {"question": "What is the temperature threshold to declare a heatwave in plains?", "reference": "A heatwave is declared when maximum temperature reaches at least 40°C in plains.", "relevant_doc_id": "imd-heatwave-action-plan"},
    {"question": "What is the difference between Warning Level and Danger Level in flood monitoring?", "reference": "Warning Level begins flood preparation, while Danger Level indicates water at or above the level that causes damage and mandates immediate evacuation.", "relevant_doc_id": "cwc-monsoon-flood-faq"},
]


class RAGEvaluator:
    def __init__(self, embedder: BGEM3Embedder, llm_base_url: str, llm_api_key: str, llm_model: str, llm_timeout_seconds: float = 30.0) -> None:
        self.chunker = DocumentChunker()
        self.embedder = embedder
        self.vector_store = VectorStoreClient()
        self.retriever = HybridRetriever(self.vector_store, self.embedder, rrf_k=60, candidate_k=20, use_reranker=False)
        self.synthesizer = RAGSynthesizer(self.retriever, llm_base_url, llm_api_key, llm_model, llm_timeout_seconds, 0.0)
        self.ragas_llm_base_url = llm_base_url
        self.ragas_llm_api_key = llm_api_key
        self.ragas_llm_model = llm_model
        self.ragas_timeout_seconds = llm_timeout_seconds

    def setup_benchmark_corpus(self) -> int:
        chunks: list[DocumentChunk] = []
        for document in GOLDEN_BENCHMARK_DOCUMENTS:
            chunks.extend(self.chunker.chunk_document(document["doc_id"], document["doc_name"], document["doc_type"], document["text"], document["region"], "en", document["doc_date"]))
        vectors = self.embedder.embed_batch([chunk.content for chunk in chunks])
        self.vector_store.insert_chunks(chunks, vectors)
        self.retriever.build_bm25_index()
        return len(chunks)

    @staticmethod
    def _retrieval_metrics(retrievals: list, relevant_doc_id: str) -> tuple[float, float, float]:
        rank = next((index for index, result in enumerate(retrievals, start=1) if result.chunk.doc_id == relevant_doc_id), None)
        hit = 1.0 if rank is not None else 0.0
        recall_at_5 = 1.0 if any(result.chunk.doc_id == relevant_doc_id for result in retrievals[:5]) else 0.0
        mrr = 1.0 / rank if rank is not None else 0.0
        return recall_at_5, mrr, hit

    async def evaluate_production_baseline(self) -> EvalMetricResult:
        if not self.vector_store.get_all_chunks():
            self.setup_benchmark_corpus()
        try:
            from openai import AsyncOpenAI
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.llms import llm_factory
            from ragas.metrics.collections import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness, ContextRelevance
        except Exception as exc:
            raise RuntimeError("RAGAS evaluation dependencies are unavailable") from exc
        client = AsyncOpenAI(api_key=self.ragas_llm_api_key, base_url=self.ragas_llm_base_url, timeout=self.ragas_timeout_seconds)
        evaluator_llm = llm_factory(self.ragas_llm_model, provider="openai", client=client)
        evaluator_embeddings = HuggingFaceEmbeddings(model="BAAI/bge-m3", device="cpu")
        metrics = {
            "context_precision": ContextPrecision(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
            "context_relevance": ContextRelevance(llm=evaluator_llm),
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            "answer_correctness": AnswerCorrectness(llm=evaluator_llm),
        }
        rows: list[dict[str, float]] = []
        try:
            for sample in LABELED_EVALUATION_SET:
                retrievals = await asyncio.to_thread(self.retriever.retrieve, sample["question"], 5)
                if not retrievals:
                    raise RuntimeError(f"No retrieval result for evaluation query: {sample['question']}")
                recall_at_5, mrr, hit = self._retrieval_metrics(retrievals, sample["relevant_doc_id"])
                response = await self.synthesizer.answer_query(sample["question"], top_k=5)
                if not response.retrieval_success:
                    raise RuntimeError(f"RAG generation failed for evaluation query: {sample['question']}")
                contexts = [result.chunk.content for result in retrievals]
                scores = await asyncio.gather(
                    metrics["context_precision"].ascore(user_input=sample["question"], reference=sample["reference"], retrieved_contexts=contexts),
                    metrics["context_recall"].ascore(user_input=sample["question"], reference=sample["reference"], retrieved_contexts=contexts),
                    metrics["context_relevance"].ascore(user_input=sample["question"], retrieved_contexts=contexts),
                    metrics["faithfulness"].ascore(user_input=sample["question"], response=response.answer, retrieved_contexts=contexts),
                    metrics["answer_relevancy"].ascore(user_input=sample["question"], response=response.answer),
                    metrics["answer_correctness"].ascore(user_input=sample["question"], response=response.answer, reference=sample["reference"]),
                )
                rows.append({
                    "context_precision": float(scores[0].value),
                    "context_recall": float(scores[1].value),
                    "context_relevance": float(scores[2].value),
                    "faithfulness": float(scores[3].value),
                    "answer_relevancy": float(scores[4].value),
                    "answer_correctness": float(scores[5].value),
                    "retrieval_recall_at_5": recall_at_5,
                    "retrieval_mrr": mrr,
                    "retrieval_hit_rate": hit,
                })
        finally:
            await client.close()
        mean_scores = {key: mean(row[key] for row in rows) for key in rows[0]}
        citation_groundedness = mean(row["retrieval_recall_at_5"] for row in rows)
        passed = (
            mean_scores["context_precision"] >= 0.80
            and mean_scores["context_recall"] >= 0.80
            and mean_scores["context_relevance"] >= 0.80
            and mean_scores["faithfulness"] >= 0.90
            and mean_scores["answer_relevancy"] >= 0.80
            and mean_scores["answer_correctness"] >= 0.80
            and mean_scores["retrieval_recall_at_5"] >= 0.90
            and mean_scores["retrieval_mrr"] >= 0.80
            and citation_groundedness == 1.0
        )
        state_scores = {key: round(value, 4) for key, value in mean_scores.items()}
        state_scores["citation_groundedness"] = round(citation_groundedness, 4)
        return EvalMetricResult(
            context_relevance=round(mean_scores["context_relevance"], 4),
            faithfulness=round(mean_scores["faithfulness"], 4),
            answer_relevance=round(mean_scores["answer_relevancy"], 4),
            citation_groundedness=round(citation_groundedness, 4),
            umbrela_score=round(mean_scores["context_recall"], 4),
            retrieval_recall_at_5=round(mean_scores["retrieval_recall_at_5"], 4),
            retrieval_mrr=round(mean_scores["retrieval_mrr"], 4),
            retrieval_hit_rate=round(mean_scores["retrieval_hit_rate"], 4),
            answer_correctness=round(mean_scores["answer_correctness"], 4),
            passed=passed,
            state_scores=state_scores,
        )
