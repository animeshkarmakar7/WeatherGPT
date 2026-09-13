import re
import math
from rank_bm25 import BM25Okapi
from .models import DocumentChunk, SearchResult
from .embedding import BGEM3Embedder
from .vector_store import VectorStoreClient


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStoreClient,
        embedder: BGEM3Embedder,
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
            chunk = self.vector_store._chunks[cid]
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
        return combined[:top_k]
