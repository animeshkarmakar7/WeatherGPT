"""Hybrid BM25 + Dense retriever with Reciprocal Rank Fusion (RRF) and an
optional cross-encoder reranker.

Retrieval flow
--------------
1. Dense ANN search via the vector store (top-20).
2. BM25 keyword search over all known chunks (top-20).
3. RRF fusion of both ranked lists → combined candidate set.
4. Recency decay and exact-token overlap bonus applied.
5. Optional cross-encoder reranking of the top-K RRF candidates.

Cross-encoder
-------------
Enabled when ``WEATHERGPT_USE_RERANKER=true``.  Requires ``FlagEmbedding``
(``FlagReranker`` class, model ``BAAI/bge-reranker-v2-m3``).  Falls back to
RRF scoring silently when the model or library is unavailable.
"""

import logging
import os
import re
import math
from rank_bm25 import BM25Okapi
from .models import DocumentChunk, SearchResult
from .embedding import BGEM3Embedder, MockBGEM3Embedder, _RealBGEM3Embedder
from .vector_store import VectorStoreClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Optional cross-encoder reranker
# ---------------------------------------------------------------------------

class _CrossEncoderReranker:
    """Thin wrapper around FlagReranker (BGE-reranker-v2-m3).

    Loads lazily so the process starts even without the model cached.
    Falls back to identity (no-op) if loading fails.
    """

    def __init__(self) -> None:
        self._model = None
        self._failed = False

    def _get_model(self):
        if self._model is None and not self._failed:
            try:
                from FlagEmbedding import FlagReranker  # type: ignore[import]
                model_name = os.getenv(
                    "WEATHERGPT_RERANKER_MODEL_PATH", "BAAI/bge-reranker-v2-m3"
                )
                use_fp16 = os.getenv("WEATHERGPT_BGE_FP16", "true").lower() == "true"
                logger.info("Loading cross-encoder reranker '%s'…", model_name)
                self._model = FlagReranker(model_name, use_fp16=use_fp16)
                logger.info("Cross-encoder reranker loaded successfully.")
            except Exception as exc:
                logger.warning(
                    "Cross-encoder unavailable (%s). RRF scores will be used as-is.", exc
                )
                self._failed = True
        return self._model

    def rerank(
        self, query: str, results: list[SearchResult]
    ) -> list[SearchResult]:
        model = self._get_model()
        if model is None:
            return results
        pairs = [[query, r.chunk.content[:512]] for r in results]
        try:
            scores = model.compute_score(pairs, normalize=True)
            for r, s in zip(results, scores):
                r.rerank_score = float(s)
            results.sort(key=lambda x: x.rerank_score, reverse=True)
        except Exception as exc:
            logger.warning("Cross-encoder scoring failed (%s); using RRF order.", exc)
        return results


_reranker: _CrossEncoderReranker | None = None


def _get_reranker() -> _CrossEncoderReranker | None:
    global _reranker
    use_reranker = os.getenv("WEATHERGPT_USE_RERANKER", "false").lower() == "true"
    if not use_reranker:
        return None
    if _reranker is None:
        _reranker = _CrossEncoderReranker()
    return _reranker


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedder,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[DocumentChunk] = []

    def _tokenize(self, text: str) -> list[str]:
        return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1]

    def build_bm25_index(self) -> None:
        chunks = self.vector_store.get_all_chunks()
        self._bm25_chunks = chunks
        if chunks:
            corpus = [self._tokenize(c.content) for c in chunks]
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict | None = None,
        recency_boost_years: float = 0.05,
    ) -> list[SearchResult]:
        if not self._bm25_chunks:
            self.build_bm25_index()

        q_vec = self.embedder.embed_text(query)
        dense_results = self.vector_store.search_dense(q_vec, top_k=20, filters=filters)

        bm25_results: list[tuple[DocumentChunk, float]] = []
        if self._bm25 and self._bm25_chunks:
            q_tokens = self._tokenize(query)
            bm25_scores = self._bm25.get_scores(q_tokens)
            indexed_scores = list(zip(self._bm25_chunks, bm25_scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            bm25_results = indexed_scores[:20]

        dense_ranks = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(dense_results)}
        bm25_ranks = {chunk.chunk_id: rank for rank, (chunk, _) in enumerate(bm25_results)}

        all_candidate_ids = set(dense_ranks.keys()) | set(bm25_ranks.keys())
        combined: list[SearchResult] = []

        q_tokens_set = set(self._tokenize(query))

        for cid in all_candidate_ids:
            # Use the public interface instead of accessing _chunks directly
            chunk = self.vector_store.get_chunk_by_id(cid)
            if chunk is None:
                # Fallback: try to retrieve from dense/bm25 result lists
                chunk = next(
                    (c for c, _ in dense_results if c.chunk_id == cid),
                    next((c for c, _ in bm25_results if c.chunk_id == cid), None),
                )
            if chunk is None:
                continue

            d_rank = dense_ranks.get(cid, 100)
            b_rank = bm25_ranks.get(cid, 100)

            rrf = (1.0 / (self.rrf_k + d_rank)) + (2.0 / (self.rrf_k + b_rank))

            chunk_tokens = set(self._tokenize(chunk.content))
            exact_overlap = len(q_tokens_set & chunk_tokens) / max(1, len(q_tokens_set))

            doc_year = 2026
            try:
                doc_year = int(chunk.doc_date[:4])
            except Exception:
                pass
            age = max(0, 2026 - doc_year)
            decay = 1.0 - (age * recency_boost_years)

            final_score = (rrf + (exact_overlap * 0.05)) * max(0.5, decay)

            d_score = next((s for c, s in dense_results if c.chunk_id == cid), 0.0)
            b_score = next((s for c, s in bm25_results if c.chunk_id == cid), 0.0)

            combined.append(
                SearchResult(
                    chunk=chunk,
                    dense_score=d_score,
                    bm25_score=b_score,
                    rrf_score=rrf,
                    rerank_score=final_score,
                )
            )

        combined.sort(key=lambda x: x.rerank_score, reverse=True)
        top_candidates = combined[:top_k]

        # Optional cross-encoder reranking
        reranker = _get_reranker()
        if reranker and top_candidates:
            top_candidates = reranker.rerank(query, top_candidates)

        return top_candidates
