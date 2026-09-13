import math
from typing import Any
from .models import DocumentChunk


class VectorStoreClient:
    def __init__(self, collection_name: str = "weathergpt_knowledge_base") -> None:
        self.collection_name = collection_name
        self._chunks: dict[str, DocumentChunk] = {}
        self._vectors: dict[str, list[float]] = {}

    def insert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors):
            self._chunks[chunk.chunk_id] = chunk
            self._vectors[chunk.chunk_id] = vec

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        scores: list[tuple[DocumentChunk, float]] = []
        for chunk_id, chunk in self._chunks.items():
            if filters:
                match = True
                for k, v in filters.items():
                    if getattr(chunk, k, None) != v and chunk.metadata.get(k) != v:
                        match = False
                        break
                if not match:
                    continue
            vec = self._vectors[chunk_id]
            sim = sum(a * b for a, b in zip(query_vector, vec))
            scores.append((chunk, float(sim)))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_all_chunks(self) -> list[DocumentChunk]:
        return list(self._chunks.values())

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        return self._chunks.get(chunk_id)
