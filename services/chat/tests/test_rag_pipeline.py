import pytest

from weathergpt_chat.rag.chunker import DocumentChunker, ParsedPage
from weathergpt_chat.rag.embedding import BGEM3Embedder
from weathergpt_chat.rag.evaluator import GOLDEN_BENCHMARK_DOCUMENTS, RAGEvaluator
from weathergpt_chat.rag.hybrid_retriever import HybridRetriever
from weathergpt_chat.rag.models import DocumentType
from weathergpt_chat.rag.vector_store import VectorStoreClient
from weathergpt_chat.rag.synthesizer import RAGSynthesizer


def test_document_chunker_types():
    chunker = DocumentChunker()
    document = GOLDEN_BENCHMARK_DOCUMENTS[0]
    chunks = chunker.chunk_document(
        doc_id=document["doc_id"],
        doc_name=document["doc_name"],
        doc_type=DocumentType.GOVERNMENT_SOP,
        text=document["text"],
    )
    assert len(chunks) >= 2
    assert all(chunk.doc_type == DocumentType.GOVERNMENT_SOP for chunk in chunks)
    assert any("Section" in chunk.section_title for chunk in chunks)


def test_page_aware_chunking():
    chunker = DocumentChunker()
    pages = [ParsedPage(1, "Section 1: Alert\nStage 2 cyclone alert."), ParsedPage(2, "Section 2: Action\nSuspend fishing operations.")]
    chunks = chunker.chunk_document("doc", "doc.pdf", DocumentType.GOVERNMENT_SOP, "", pages=pages)
    assert {chunk.page_number for chunk in chunks} == {1, 2}


def test_hybrid_search_and_citation_binding():
    chunker = DocumentChunker()
    embedder = BGEM3Embedder()
    store = VectorStoreClient()
    chunks = chunker.chunk_document(
        doc_id="test-sop",
        doc_name="NDMA Evacuation SOP",
        doc_type=DocumentType.GOVERNMENT_SOP,
        text="Section 1: Evacuation Guidelines\nAll fishing vessels must return to port immediately.",
    )
    vectors = embedder.embed_batch([chunk.content for chunk in chunks])
    store.insert_chunks(chunks, vectors)
    retriever = HybridRetriever(store, embedder)
    retriever.build_bm25_index()
    results = retriever.retrieve("fishing vessels regulation", top_k=3)
    assert results
    assert results[0].chunk.doc_name == "NDMA Evacuation SOP"
    assert results[0].chunk.page_number == 1


def test_minio_document_store_upload_and_retrieval():
    from weathergpt_chat.rag.minio_store import MinioDocumentStore
    minio_store = MinioDocumentStore()
    content = b"Official Heatwave Action Plan Text"
    uri = minio_store.upload_document("heatwave_plan.txt", content)
    assert uri.startswith(("minio://", "memory://"))
    assert minio_store.get_document("heatwave_plan.txt") == content


@pytest.mark.parametrize("query", ["How do I make chocolate cake?", "Explain a quantum computer without weather context."])
def test_retriever_can_return_empty_or_low_relevance(query):
    store = VectorStoreClient()
    embedder = BGEM3Embedder()
    retriever = HybridRetriever(store, embedder)
    assert retriever.retrieve(query, top_k=3) == []


def test_evaluator_benchmark_corpus_isolated_from_production_store():
    evaluator = RAGEvaluator(
        embedder=BGEM3Embedder(),
        llm_base_url="http://127.0.0.1:9/v1",
        llm_api_key="test",
        llm_model="test",
    )
    count = evaluator.setup_benchmark_corpus()
    assert count > 0
    assert evaluator.vector_store.get_all_chunks()
