from .models import Citation, RAGResponse, SearchResult
from .hybrid_retriever import HybridRetriever


class RAGSynthesizer:
    def __init__(self, retriever: HybridRetriever, min_confidence: float = 0.015) -> None:
        self.retriever = retriever
        self.min_confidence = min_confidence

    def answer_query(self, query: str, top_k: int = 3, filters: dict | None = None) -> RAGResponse:
        results = self.retriever.retrieve(query, top_k=top_k, filters=filters)

        if not results or results[0].rerank_score < self.min_confidence:
            return RAGResponse(
                query=query,
                answer="I don't have a confirmed official document answer in the knowledge base for this query.",
                citations=[],
                confidence=0.0,
                retrieval_success=False,
            )

        citations: list[Citation] = []
        fact_lines: list[str] = []

        for r in results:
            c = r.chunk
            citations.append(
                Citation(
                    doc_name=c.doc_name,
                    page=c.page_number,
                    section=c.section_title,
                    excerpt=c.content[:150].strip() + "...",
                )
            )
            fact_lines.append(f"[{c.doc_name} §{c.section_title} p.{c.page_number}]: {c.content}")

        synthesis_text = (
            f"Based on official guidelines ({citations[0].doc_name}, {citations[0].section}):\n"
            + results[0].chunk.content
        )

        return RAGResponse(
            query=query,
            answer=synthesis_text,
            citations=citations,
            confidence=round(min(0.98, results[0].rerank_score * 35.0), 2),
            retrieval_success=True,
        )
