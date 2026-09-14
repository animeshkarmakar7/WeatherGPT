import logging
import re

from rank_bm25 import BM25Okapi

from .embedding import BGEM3Embedder
from .models import DocumentChunk, SearchResult

logger = logging.getLogger(__name__)


class _CrossEncoderReranker:
    def __init__(self, model_path: str, use_fp16: bool) -> None:
        try:
            from FlagEmbedding import FlagReranker
        except Exception as exc:
            raise RuntimeError("FlagEmbedding is required for the production reranker") from exc
        self.model = FlagReranker(model_path, use_fp16=use_fp16)

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if not results:
            return results
        scores = self.model.compute_score([[query, result.chunk.content[:2048]] for result in results], normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        for result, score in zip(results, scores):
            result.rerank_score = float(score)
        results.sort(key=lambda item: item.rerank_score, reverse=True)
        return results


class HybridRetriever:
    def __init__(self, vector_store, embedder: BGEM3Embedder, rrf_k: int = 60, candidate_k: int = 20, reranker_model_path: str | None = None, reranker_fp16: bool = False, use_reranker: bool = True) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidate_k = max(candidate_k, 5)
        self._bm25: BM25Okapi | None = None
        self._bm25_chunks: list[DocumentChunk] = []
        self._reranker = _CrossEncoderReranker(reranker_model_path, reranker_fp16) if use_reranker and reranker_model_path else None

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return [token for token in re.findall(r"\w+", text.lower()) if len(token) > 1]

    def build_bm25_index(self) -> None:
        chunks = self.vector_store.get_all_chunks()
        self._bm25_chunks = chunks
        self._bm25 = BM25Okapi([self._tokenize(chunk.content) for chunk in chunks]) if chunks else None

    def retrieve(self, query: str, top_k: int = 5, filters: dict | None = None, recency_boost_years: float = 0.02) -> list[SearchResult]:
        if not self._bm25_chunks:
            self.build_bm25_index()
        if not self._bm25_chunks:
            return []

        candidate_limit = max(self.candidate_k, top_k)
        query_vector = self.embedder.embed_text(query)
        dense_results = self.vector_store.search_dense(query_vector, top_k=candidate_limit, filters=filters)
        bm25_results: list[tuple[DocumentChunk, float]] = []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(self._tokenize(query))
            ranked = sorted(zip(self._bm25_chunks, scores), key=lambda item: item[1], reverse=True)
            if filters:
                ranked = [item for item in ranked if all(getattr(item[0], key, None) == value or item[0].metadata.get(key) == value for key, value in filters.items())]
            bm25_results = ranked[:candidate_limit]

        dense_rank = {chunk.chunk_id: rank + 1 for rank, (chunk, _) in enumerate(dense_results)}
        bm25_rank = {chunk.chunk_id: rank + 1 for rank, (chunk, _) in enumerate(bm25_results)}
        dense_score = {chunk.chunk_id: score for chunk, score in dense_results}
        bm25_score = {chunk.chunk_id: score for chunk, score in bm25_results}
        chunks_by_id = {chunk.chunk_id: chunk for chunk in self._bm25_chunks}
        combined: list[SearchResult] = []
        query_tokens = set(self._tokenize(query))

        for chunk_id in set(dense_rank) | set(bm25_rank):
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                continue
            d_rank = dense_rank.get(chunk_id, candidate_limit + 1)
            b_rank = bm25_rank.get(chunk_id, candidate_limit + 1)
            rrf = 1.0 / (self.rrf_k + d_rank) + 1.0 / (self.rrf_k + b_rank)
            overlap = len(query_tokens & set(self._tokenize(chunk.content))) / max(1, len(query_tokens))
            try:
                doc_year = int(chunk.doc_date[:4])
            except (TypeError, ValueError):
                doc_year = 0
            age = max(0, 2026 - doc_year)
            score = (rrf + 0.05 * overlap) * max(0.5, 1.0 - age * recency_boost_years)
            combined.append(SearchResult(chunk=chunk, dense_score=dense_score.get(chunk_id, 0.0), bm25_score=bm25_score.get(chunk_id, 0.0), rrf_score=rrf, rerank_score=score))

        combined.sort(key=lambda item: item.rrf_score, reverse=True)
        candidates = combined[:candidate_limit]
        if self._reranker:
            candidates = self._reranker.rerank(query, candidates)
        else:
            candidates.sort(key=lambda item: item.rrf_score, reverse=True)
        return candidates[:top_k]
