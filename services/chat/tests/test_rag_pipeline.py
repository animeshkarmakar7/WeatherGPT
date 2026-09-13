import hashlib

import pytest

from weathergpt_chat.rag.chunker import DocumentChunker, ParsedPage
from weathergpt_chat.rag.evaluator import GOLDEN_BENCHMARK_DOCUMENTS, RAGEvaluator
from weathergpt_chat.rag.hybrid_retriever import HybridRetriever
from weathergpt_chat.rag.models import DocumentType
from weathergpt_chat.rag.vector_store import VectorStoreClient


class TestEmbedder:
    def __init__(self, vector_dim: int = 1024) -> None:
        self.vector_dim = vector_dim

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.vector_dim
        for token in text.lower().split():
            index = int(hashlib.sha256(token.encode()).hexdigest(), 16) % self.vector_dim
            vector[index] += 1.0
        norm = sum(value * value for value in vector) ** 0.5
        return [value / norm for value in vector] if norm else vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


def test_document_chunker_types():
    chunker = DocumentChunker()
    document = GOLDEN_BENCHMARK_DOCUMENTS[0]
    chunks = chunker.chunk_document(document["doc_id"], document["doc_name"], DocumentType.GOVERNMENT_SOP, document["text"])
    assert len(chunks) >= 2
    assert all(chunk.doc_type == DocumentType.GOVERNMENT_SOP for chunk in chunks)
    assert any("Section" in chunk.section_title for chunk in chunks)


def test_page_aware_chunking():
    chunker = DocumentChunker()
    pages = [ParsedPage(1, "Section 1: Alert\nStage 2 cyclone alert."), ParsedPage(2, "Section 2: Action\nSuspend fishing operations.")]
    chunks = chunker.chunk_document("doc", "doc.pdf", DocumentType.GOVERNMENT_SOP, "", pages=pages)
    assert {chunk.page_number for chunk in chunks} == {1, 2}


def test_hybrid_search():
    chunker = DocumentChunker()
    embedder = TestEmbedder()
    store = VectorStoreClient()
    chunks = chunker.chunk_document("test-sop", "NDMA Evacuation SOP", DocumentType.GOVERNMENT_SOP, "Section 1: Evacuation Guidelines\nAll fishing vessels must return to port immediately.")
    store.insert_chunks(chunks, embedder.embed_batch([chunk.content for chunk in chunks]))
    retriever = HybridRetriever(store, embedder)
    results = retriever.retrieve("fishing vessels regulation", top_k=3)
    assert results
    assert results[0].chunk.doc_name == "NDMA Evacuation SOP"


def test_retriever_empty_store_returns_empty():
    store = VectorStoreClient()
    retriever = HybridRetriever(store, TestEmbedder())
    assert retriever.retrieve("How do I make chocolate cake?", top_k=3) == []


def test_evaluator_benchmark_corpus_isolated():
    evaluator = RAGEvaluator(TestEmbedder(), "http://127.0.0.1:9/v1", "test", "test")
    assert evaluator.setup_benchmark_corpus() > 0
    assert evaluator.vector_store.get_all_chunks()
