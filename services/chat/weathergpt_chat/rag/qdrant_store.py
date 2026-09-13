"""Qdrant-backed vector store for the WeatherGPT RAG knowledge base.

Implements the same interface as ``VectorStoreClient`` so it can be swapped
in transparently.  Set ``WEATHERGPT_USE_QDRANT=true`` in the environment
to activate (see ``config.py``).

Collection schema
-----------------
Vector name : ``dense``  (1024-dim, Cosine)
Payload     : all fields from ``DocumentChunk`` serialised as JSON

The collection is created automatically on first use if it does not exist.
"""

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from .models import DocumentChunk, DocumentType

logger = logging.getLogger(__name__)

COLLECTION_NAME = "weathergpt_knowledge_base"
VECTOR_NAME = "dense"
VECTOR_DIM = 1024


class QdrantVectorStoreClient:
    """Qdrant-backed vector store implementing the same public interface as
    the in-memory ``VectorStoreClient``."""

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection_name: str = COLLECTION_NAME,
        vector_dim: int = VECTOR_DIM,
    ) -> None:
        self.url = url
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self._client: QdrantClient | None = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.url, timeout=10)
            self._ensure_collection()
        return self._client

    def _ensure_collection(self) -> None:
        client = self._client
        assert client is not None
        existing = {c.name for c in client.get_collections().collections}
        if self.collection_name not in existing:
            client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    VECTOR_NAME: qmodels.VectorParams(
                        size=self.vector_dim,
                        distance=qmodels.Distance.COSINE,
                    )
                },
            )
            logger.info(
                "Created Qdrant collection '%s' (dim=%d, Cosine)",
                self.collection_name,
                self.vector_dim,
            )

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def insert_chunks(
        self, chunks: list[DocumentChunk], vectors: list[list[float]]
    ) -> None:
        client = self._get_client()
        points: list[qmodels.PointStruct] = []
        for chunk, vec in zip(chunks, vectors):
            payload = chunk.model_dump(mode="json")
            points.append(
                qmodels.PointStruct(
                    id=_chunk_id_to_uuid(chunk.chunk_id),
                    vector={VECTOR_NAME: vec},
                    payload=payload,
                )
            )
        if points:
            client.upsert(collection_name=self.collection_name, points=points)
            logger.debug("Upserted %d points into '%s'", len(points), self.collection_name)

    # ------------------------------------------------------------------
    # Read — dense ANN search
    # ------------------------------------------------------------------

    def search_dense(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[DocumentChunk, float]]:
        client = self._get_client()
        qdrant_filter = _build_filter(filters) if filters else None
        hits = client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            using=VECTOR_NAME,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        ).points
        return [(_payload_to_chunk(h.payload), h.score) for h in hits]

    # ------------------------------------------------------------------
    # Read — scroll (needed by BM25 index rebuild)
    # ------------------------------------------------------------------

    def get_all_chunks(self) -> list[DocumentChunk]:
        client = self._get_client()
        chunks: list[DocumentChunk] = []
        offset = None
        while True:
            records, offset = client.scroll(
                collection_name=self.collection_name,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for r in records:
                chunks.append(_payload_to_chunk(r.payload))
            if offset is None:
                break
        return chunks

    def get_chunk_by_id(self, chunk_id: str) -> DocumentChunk | None:
        client = self._get_client()
        results = client.retrieve(
            collection_name=self.collection_name,
            ids=[_chunk_id_to_uuid(chunk_id)],
            with_payload=True,
        )
        if results:
            return _payload_to_chunk(results[0].payload)
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_id_to_uuid(chunk_id: str) -> str:
    """Derive a deterministic UUID v5 from a chunk_id string."""
    import uuid
    return str(uuid.uuid5(uuid.NAMESPACE_OID, chunk_id))


def _payload_to_chunk(payload: dict[str, Any] | None) -> DocumentChunk:
    if not payload:
        raise ValueError("Qdrant returned a point with no payload")
    return DocumentChunk.model_validate(payload)


def _build_filter(filters: dict[str, Any]) -> qmodels.Filter:
    """Convert a simple equality-filter dict to a Qdrant ``Filter``."""
    must: list[qmodels.FieldCondition] = []
    for key, value in filters.items():
        must.append(
            qmodels.FieldCondition(
                key=key,
                match=qmodels.MatchValue(value=value),
            )
        )
    return qmodels.Filter(must=must)
