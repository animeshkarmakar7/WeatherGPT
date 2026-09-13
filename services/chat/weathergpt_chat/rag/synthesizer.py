import json
import logging
from typing import Any

import httpx

from .hybrid_retriever import HybridRetriever
from .models import Citation, RAGResponse

logger = logging.getLogger(__name__)


class RAGSynthesizer:
    def __init__(
        self,
        retriever: HybridRetriever,
        llm_base_url: str,
        llm_api_key: str,
        llm_model: str,
        timeout_seconds: float = 30.0,
        min_confidence: float = 0.015,
    ) -> None:
        self.retriever = retriever
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.timeout_seconds = timeout_seconds
        self.min_confidence = min_confidence

    async def answer_query(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> RAGResponse:
        results = self.retriever.retrieve(query, top_k=top_k, filters=filters)
        if not results or results[0].rerank_score < self.min_confidence:
            return RAGResponse(query=query, answer="I do not have enough verified official evidence to answer that question.", citations=[], confidence=0.0, retrieval_success=False)

        citations: list[Citation] = []
        evidence: list[str] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            citations.append(Citation(doc_name=chunk.doc_name, page=chunk.page_number, section=chunk.section_title, excerpt=chunk.content[:300].strip()))
            evidence.append(json.dumps({"evidence_id": index, "doc_name": chunk.doc_name, "page": chunk.page_number, "section": chunk.section_title, "content": chunk.content}, ensure_ascii=False))

        system_prompt = (
            "You are WeatherGPT's grounded official-information assistant. "
            "Answer only from the supplied evidence. Do not add unsupported facts. "
            "Do not invent dates, thresholds, procedures, locations, or policy claims. "
            "When evidence is insufficient, say that the available official evidence is insufficient. "
            "Return JSON with fields answer, evidence_ids, confidence. evidence_ids must contain only supplied evidence_id values."
        )
        user_prompt = f"QUESTION:\n{query}\n\nEVIDENCE:\n" + "\n".join(evidence)
        payload = {
            "model": self.llm_model,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(
                    f"{self.llm_base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.llm_api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                content = body["choices"][0]["message"]["content"]
                generated = json.loads(content)
                answer = str(generated.get("answer", "")).strip()
                evidence_ids = generated.get("evidence_ids") or []
                confidence = float(generated.get("confidence", 0.0))
                valid_ids = {index for index in range(1, len(results) + 1)}
                if not answer or not evidence_ids or not set(evidence_ids).issubset(valid_ids):
                    raise ValueError("LLM returned invalid grounded response")
                selected = [citations[index - 1] for index in evidence_ids if isinstance(index, int) and 1 <= index <= len(citations)]
                if not selected:
                    raise ValueError("LLM returned no valid citations")
                return RAGResponse(query=query, answer=answer, citations=selected, confidence=max(0.0, min(1.0, confidence)), retrieval_success=True)
        except Exception as exc:
            logger.error("Grounded RAG generation failed: %s", exc)
            return RAGResponse(query=query, answer="The verified knowledge service is temporarily unavailable, so I cannot provide a grounded answer.", citations=[], confidence=0.0, retrieval_success=False)
