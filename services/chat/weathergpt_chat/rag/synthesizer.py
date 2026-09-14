import asyncio
import json
import re
from typing import Any

import httpx

from .hybrid_retriever import HybridRetriever
from .models import Citation, RAGResponse


class RAGSynthesizer:
    def __init__(self, retriever: HybridRetriever, llm_base_url: str, llm_api_key: str, llm_model: str, timeout_seconds: float = 30.0, min_confidence: float = 0.0) -> None:
        self.retriever = retriever
        self.llm_base_url = llm_base_url.rstrip("/")
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        self.timeout_seconds = timeout_seconds
        self.min_confidence = min_confidence

    @staticmethod
    def _numbers(text: str) -> set[str]:
        return {match.group(0) for match in re.finditer(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", text)}

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {token for token in re.findall(r"\b\w+\b", text.lower()) if len(token) > 2}

    def _validate_grounding(self, answer: str, evidence: list[dict[str, Any]], evidence_ids: list[int]) -> bool:
        if not answer.strip() or not evidence_ids:
            return False
        selected = [evidence[index - 1]["content"] for index in evidence_ids]
        context = "\n".join(selected)
        answer_numbers = self._numbers(answer)
        context_numbers = self._numbers(context)
        if not answer_numbers.issubset(context_numbers):
            return False
        context_tokens = self._tokens(context)
        sentences = [part.strip() for part in re.split(r"[.!?]\s+", answer) if part.strip()]
        for sentence in sentences:
            tokens = self._tokens(sentence)
            if len(tokens) >= 4 and len(tokens & context_tokens) / len(tokens) < 0.30:
                return False
        return True

    async def answer_query(self, query: str, top_k: int = 5, filters: dict[str, Any] | None = None) -> RAGResponse:
        results = await asyncio.to_thread(self.retriever.retrieve, query, top_k, filters)
        if not results:
            return RAGResponse(query=query, answer="I do not have enough verified official evidence to answer that question.", citations=[], confidence=0.0, retrieval_success=False)
        if results[0].rerank_score < self.min_confidence:
            return RAGResponse(query=query, answer="I do not have enough verified official evidence to answer that question.", citations=[], confidence=0.0, retrieval_success=False)

        citations: list[Citation] = []
        evidence: list[dict[str, Any]] = []
        for index, result in enumerate(results, start=1):
            chunk = result.chunk
            citations.append(Citation(doc_name=chunk.doc_name, page=chunk.page_number, section=chunk.section_title, excerpt=chunk.content[:300].strip()))
            evidence.append({"evidence_id": index, "doc_name": chunk.doc_name, "page": chunk.page_number, "section": chunk.section_title, "content": chunk.content})

        system_prompt = "You are WeatherGPT's grounded official-information assistant. Answer only from supplied evidence. Do not add unsupported facts. Do not invent dates, thresholds, procedures, locations, or policy claims. If the evidence does not support the answer, say that the evidence is insufficient. Return only the requested JSON object."
        schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "integer"}},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["answer", "evidence_ids", "confidence"],
            "additionalProperties": False,
        }
        payload = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"QUESTION:\n{query}\n\nEVIDENCE:\n{json.dumps(evidence, ensure_ascii=False)}"},
            ],
            "temperature": 0.0,
            "response_format": {"type": "json_schema", "json_schema": {"name": "grounded_rag_answer", "schema": schema}},
        }

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
        answer = str(generated["answer"]).strip()
        evidence_ids = [int(value) for value in generated["evidence_ids"]]
        confidence = float(generated["confidence"])
        valid_ids = set(range(1, len(results) + 1))
        if not set(evidence_ids).issubset(valid_ids):
            raise ValueError("LLM returned an invalid evidence reference")
        if not self._validate_grounding(answer, evidence, evidence_ids):
            raise ValueError("Generated answer failed grounding validation")
        selected = [citations[index - 1] for index in evidence_ids]
        confidence = max(0.0, min(1.0, confidence))
        return RAGResponse(query=query, answer=answer, citations=selected, confidence=confidence, retrieval_success=True)
