import re
from .models import DocumentChunk, DocumentType, EvalMetricResult
from .chunker import DocumentChunker
from .embedding import BGEM3Embedder
from .vector_store import VectorStoreClient
from .hybrid_retriever import HybridRetriever
from .synthesizer import RAGSynthesizer


GOLDEN_BENCHMARK_DOCUMENTS = [
    {
        "doc_id": "imd-cyclone-sop-2026",
        "doc_name": "NDMA Cyclone Warning & Evacuation SOP",
        "doc_type": DocumentType.GOVERNMENT_SOP,
        "region": "coastal",
        "doc_date": "2026-03-01",
        "text": """
Section 1: Cyclone Warning Stages
Stage 1: Pre-Cyclone Watch issued 72 hours prior.
Stage 2: Cyclone Alert issued 48 hours prior to expected commencement of adverse weather.
Stage 3: Cyclone Warning issued at least 24 hours in advance specifying landfall location.
Stage 4: Post-landfall outlook issued 12 hours before landfall.

Section 2: Evacuation Directives for Coastal Communities
All fishing operations must be suspended immediately upon issuance of Stage 2 Cyclone Alert.
Low-lying coastal populations within 5 km of shoreline must be evacuated to storm surge shelters when wind speed exceeds 65 km/h.
Critical infrastructure including electrical grids must initiate pre-emptive shutdown in Red Alert zones.
""",
    },
    {
        "doc_id": "imd-heatwave-action-plan",
        "doc_name": "National Heatwave Action Plan & Advisory",
        "doc_type": DocumentType.GOVERNMENT_SOP,
        "region": "central_and_north",
        "doc_date": "2026-02-15",
        "text": """
Section 1: Temperature Thresholds for Heatwave Declaration
A heatwave is declared when maximum temperature reaches at least 40°C in plains and 30°C in hilly regions.
Severe heatwave is declared when departure from normal temperature is 6.5°C or higher, or actual temperature reaches 45°C.

Section 2: Health Advisory & Work Regulations
Outdoor labor must be suspended between 12:00 PM and 3:30 PM during Red Alert heatwave conditions.
Emergency cooling centers with ORS and drinking water must be operational across all municipal wards.
""",
    },
    {
        "doc_id": "cwc-monsoon-flood-faq",
        "doc_name": "Central Water Commission Urban Flood Guidelines",
        "doc_type": DocumentType.FAQ_ADVISORY,
        "region": "all",
        "doc_date": "2025-06-10",
        "text": """
Question: What defines Warning Level versus Danger Level in river flood monitoring?
Answer: Warning Level indicates the river stage at which flood preparation begins and low embankments are monitored. Danger Level indicates the river water level at or above which flood waters cause damage to surrounding settlements and mandate immediate evacuation.

Question: What action should citizens take during an urban flash flood alert?
Answer: Stay indoors, avoid driving through waterlogged underpasses, disconnect non-essential electrical appliances, and monitor local disaster management broadcasts.
""",
    },
]

LABELED_EVALUATION_SET = [
    {
        "question": "What is the Stage 2 Cyclone Alert guideline for fishing operations?",
        "expected_ground_truth": "Issued 48 hours prior to expected adverse weather, and all fishing operations must be suspended immediately.",
        "key_phrases": ["cyclone alert", "fishing operations must be suspended", "stage 2"],
    },
    {
        "question": "What is the temperature threshold to declare a heatwave in plains?",
        "expected_ground_truth": "Maximum temperature of at least 40°C in plains.",
        "key_phrases": ["40°c", "plains", "heatwave"],
    },
    {
        "question": "What is the difference between Warning Level and Danger Level in flood monitoring?",
        "expected_ground_truth": "Warning Level begins flood preparation, while Danger Level indicates river water causes damage and mandates immediate evacuation.",
        "key_phrases": ["warning level", "danger level", "evacuation"],
    },
]


class RAGEvaluator:
    def __init__(
        self,
        threshold_context_relevance: float = 0.80,
        threshold_faithfulness: float = 0.85,
        threshold_answer_relevance: float = 0.80,
        threshold_umbrela: float = 0.80,
    ) -> None:
        self.thresh_cr = threshold_context_relevance
        self.thresh_f = threshold_faithfulness
        self.thresh_ar = threshold_answer_relevance
        self.thresh_u = threshold_umbrela
        self.chunker = DocumentChunker()
        self.embedder = BGEM3Embedder()
        self.vector_store = VectorStoreClient()
        self.retriever = HybridRetriever(self.vector_store, self.embedder)
        self.synthesizer = RAGSynthesizer(self.retriever)

    def setup_benchmark_corpus(self) -> int:
        all_chunks: list[DocumentChunk] = []
        for doc in GOLDEN_BENCHMARK_DOCUMENTS:
            chunks = self.chunker.chunk_document(
                doc_id=doc["doc_id"],
                doc_name=doc["doc_name"],
                doc_type=doc["doc_type"],
                text=doc["text"],
                region=doc["region"],
                doc_date=doc["doc_date"],
            )
            all_chunks.extend(chunks)

        vectors = self.embedder.embed_batch([c.content for c in all_chunks])
        self.vector_store.insert_chunks(all_chunks, vectors)
        self.retriever.build_bm25_index()
        return len(all_chunks)

    def _calculate_umbrela_grade(self, query: str, passage: str, expected_key_phrases: list[str]) -> float:
        text_lower = passage.lower()
        matched = sum(1 for kp in expected_key_phrases if kp.lower() in text_lower)
        if matched == len(expected_key_phrases):
            return 1.0
        elif matched >= 1:
            return 0.75
        query_words = set(re.findall(r"\w+", query.lower()))
        passage_words = set(re.findall(r"\w+", text_lower))
        overlap = len(query_words & passage_words) / max(1, len(query_words))
        if overlap > 0.4:
            return 0.5
        return 0.0

    def evaluate_production_baseline(self) -> EvalMetricResult:
        if not self.vector_store.get_all_chunks():
            self.setup_benchmark_corpus()

        cr_scores: list[float] = []
        f_scores: list[float] = []
        ar_scores: list[float] = []
        umbrela_scores: list[float] = []
        citation_valid: list[float] = []

        for sample in LABELED_EVALUATION_SET:
            q = sample["question"]
            expected = sample["key_phrases"]

            retrievals = self.retriever.retrieve(q, top_k=2)

            has_relevant = False
            top_umbrela = 0.0
            for r in retrievals:
                content_lower = r.chunk.content.lower()
                if any(kp.lower() in content_lower for kp in expected):
                    has_relevant = True
                grade = self._calculate_umbrela_grade(q, r.chunk.content, expected)
                if grade > top_umbrela:
                    top_umbrela = grade

            cr = 1.0 if has_relevant else 0.0
            cr_scores.append(cr)
            umbrela_scores.append(top_umbrela)

            response = self.synthesizer.answer_query(q, top_k=2)

            ans_lower = response.answer.lower()
            grounded_claims = 0
            for kp in expected:
                if kp.lower() in ans_lower:
                    grounded_claims += 1
            faithfulness = grounded_claims / max(1, len(expected))
            f_scores.append(faithfulness)

            q_words = set(re.findall(r"\w+", q.lower()))
            ans_words = set(re.findall(r"\w+", ans_lower))
            overlap = len(q_words & ans_words)
            ar = min(1.0, 0.70 + (overlap / max(1, len(q_words)) * 0.35))
            ar_scores.append(ar)

            has_valid_citations = len(response.citations) > 0 and all(
                c.page >= 1 and len(c.doc_name) > 3 for c in response.citations
            )
            citation_valid.append(1.0 if has_valid_citations else 0.0)

        mean_cr = sum(cr_scores) / len(cr_scores)
        mean_f = sum(f_scores) / len(f_scores)
        mean_ar = sum(ar_scores) / len(ar_scores)
        mean_u = sum(umbrela_scores) / len(umbrela_scores)
        mean_cit = sum(citation_valid) / len(citation_valid)

        passed = (
            mean_cr >= self.thresh_cr
            and mean_f >= self.thresh_f
            and mean_ar >= self.thresh_ar
            and mean_u >= self.thresh_u
            and mean_cit == 1.0
        )

        scores = {
            "context_relevance": round(mean_cr, 4),
            "faithfulness": round(mean_f, 4),
            "answer_relevance": round(mean_ar, 4),
            "umbrela_score": round(mean_u, 4),
            "citation_groundedness": round(mean_cit, 4),
        }

        return EvalMetricResult(
            context_relevance=scores["context_relevance"],
            faithfulness=scores["faithfulness"],
            answer_relevance=scores["answer_relevance"],
            citation_groundedness=scores["citation_groundedness"],
            umbrela_score=scores["umbrela_score"],
            passed=passed,
            state_scores=scores,
        )
