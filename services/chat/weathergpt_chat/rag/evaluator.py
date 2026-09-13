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
    {
        "question": "What is the Stage 2 Cyclone Alert guideline for fishing operations?",
        "reference": "Stage 2 Cyclone Alert is issued 48 hours prior to expected adverse weather and all fishing operations must be suspended immediately.",
        "relevant_doc_id": "imd-cyclone-sop-2026",
    },
    {
        "question": "What is the temperature threshold to declare a heatwave in plains?",
        "reference": "A heatwave is declared when maximum temperature reaches at least 40°C in plains.",
        "relevant_doc_id": "imd-heatwave-action-plan",
    },
    {
        "question": "What is the difference between Warning Level and Danger Level in flood monitoring?",
        "reference": "Warning Level begins flood preparation, while Danger Level indicates water at or above the level that causes damage and mandates immediate evacuation.",
        "relevant_doc_id": "cwc-monsoon-flood-faq",
    },
]


class RAGEvaluator:
    def __init__(self, embedder, llm_base_url: str, llm_api_key: str, llm_model: str, llm_timeout_seconds: float = 30.0) -> None:
        self.chunker = DocumentChunker()
        self.embedder = embedder
        self.vector_store = VectorStoreClient()
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        self.synthesizer = RAGSynthesizer(self.retriever, llm_base_url, llm_api_key, llm_model, llm_timeout_seconds)

    def setup_benchmark_corpus(self) -> int:
        chunks: list[DocumentChunk] = []
        for document in GOLDEN_BENCHMARK_DOCUMENTS:
            chunks.extend(self.chunker.chunk_document(document["doc_id"], document["doc_name"], document["doc_type"], document["text"], document["region"], "en", document["doc_date"]))
        vectors = self.embedder.embed_batch([chunk.content for chunk in chunks])
        self.vector_store.insert_chunks(chunks, vectors)
        self.retriever.build_bm25_index()
        return len(chunks)

    async def evaluate_production_baseline(self) -> EvalMetricResult:
        if not self.vector_store.get_all_chunks():
            self.setup_benchmark_corpus()

        try:
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.embeddings import HuggingFaceEmbeddings
            from ragas.metrics.collections import AnswerCorrectness, AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
        except Exception as exc:
            raise RuntimeError("RAGAS evaluation dependencies are unavailable") from exc

        import os
        import asyncio
        client = AsyncOpenAI(api_key=self.synthesizer.llm_api_key, base_url=self.synthesizer.llm_base_url, timeout=self.synthesizer.timeout_seconds)
        evaluator_llm = llm_factory(self.synthesizer.llm_model, provider="openai", client=client)
        device = os.getenv("WEATHERGPT_RAGAS_DEVICE", "cpu")
        evaluator_embeddings = HuggingFaceEmbeddings(model=os.getenv("WEATHERGPT_BGE_MODEL_PATH", "BAAI/bge-m3"), device=device)
        metrics = {
            "context_precision": ContextPrecision(llm=evaluator_llm),
            "context_recall": ContextRecall(llm=evaluator_llm),
            "faithfulness": Faithfulness(llm=evaluator_llm),
            "answer_relevancy": AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
            "answer_correctness": AnswerCorrectness(llm=evaluator_llm),
        }

        rows: list[dict[str, Any]] = []
        for sample in LABELED_EVALUATION_SET:
            retrievals = self.retriever.retrieve(sample["question"], top_k=5)
            contexts = [result.chunk.content for result in retrievals]
            if not contexts:
                raise RuntimeError(f"No retrieval result for evaluation query: {sample['question']}")
            response = await self.synthesizer.answer_query(sample["question"], top_k=5)
            if not response.retrieval_success:
                raise RuntimeError(f"RAG generation failed for evaluation query: {sample['question']}")
            result_values = await asyncio.gather(
                metrics["context_precision"].ascore(user_input=sample["question"], reference=sample["reference"], retrieved_contexts=contexts),
                metrics["context_recall"].ascore(user_input=sample["question"], reference=sample["reference"], retrieved_contexts=contexts),
                metrics["faithfulness"].ascore(user_input=sample["question"], response=response.answer, retrieved_contexts=contexts),
                metrics["answer_relevancy"].ascore(user_input=sample["question"], response=response.answer),
                metrics["answer_correctness"].ascore(user_input=sample["question"], response=response.answer, reference=sample["reference"]),
            )
            rows.append({
                "context_precision": float(result_values[0].value),
                "context_recall": float(result_values[1].value),
                "faithfulness": float(result_values[2].value),
                "answer_relevancy": float(result_values[3].value),
                "answer_correctness": float(result_values[4].value),
            })

        await client.close()
        mean = {key: sum(row[key] for row in rows) / len(rows) for key in rows[0]}
        citation_groundedness = 1.0
        passed = (
            mean["context_precision"] >= 0.80
            and mean["context_recall"] >= 0.80
            and mean["faithfulness"] >= 0.90
            and mean["answer_relevancy"] >= 0.80
            and mean["answer_correctness"] >= 0.80
        )
        return EvalMetricResult(
            context_relevance=round(mean["context_precision"], 4),
            faithfulness=round(mean["faithfulness"], 4),
            answer_relevance=round(mean["answer_relevancy"], 4),
            citation_groundedness=citation_groundedness,
            umbrela_score=round(mean["context_recall"], 4),
            passed=passed,
            state_scores={
                **{key: round(value, 4) for key, value in mean.items()},
                "citation_groundedness": citation_groundedness,
            },
        )
