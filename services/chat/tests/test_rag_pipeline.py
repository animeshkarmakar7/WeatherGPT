import pytest
from weathergpt_chat.rag.models import DocumentType
from weathergpt_chat.rag.chunker import DocumentChunker
from weathergpt_chat.rag.embedding import BGEM3Embedder
from weathergpt_chat.rag.vector_store import VectorStoreClient
from weathergpt_chat.rag.hybrid_retriever import HybridRetriever
from weathergpt_chat.rag.synthesizer import RAGSynthesizer
from weathergpt_chat.rag.evaluator import RAGEvaluator, GOLDEN_BENCHMARK_DOCUMENTS
from weathergpt_chat.rag.minio_store import MinioDocumentStore


def test_document_chunker_types():
    chunker = DocumentChunker()
    sop_doc = GOLDEN_BENCHMARK_DOCUMENTS[0]
    chunks = chunker.chunk_document(
        doc_id=sop_doc["doc_id"],
        doc_name=sop_doc["doc_name"],
        doc_type=DocumentType.GOVERNMENT_SOP,
        text=sop_doc["text"],
    )
    assert len(chunks) >= 2
    assert all(c.doc_type == DocumentType.GOVERNMENT_SOP for c in chunks)
    assert any("Section" in c.section_title for c in chunks)


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
    vectors = embedder.embed_batch([c.content for c in chunks])
    store.insert_chunks(chunks, vectors)

    retriever = HybridRetriever(store, embedder)
    retriever.build_bm25_index()

    synthesizer = RAGSynthesizer(retriever)
    resp = synthesizer.answer_query("fishing vessels regulation")

    assert resp.retrieval_success is True
    assert len(resp.citations) > 0
    assert resp.citations[0].doc_name == "NDMA Evacuation SOP"
    assert resp.citations[0].page == 1
    assert "fishing vessels" in resp.answer.lower()


def test_out_of_domain_query_triggers_safe_fallback():
    embedder = BGEM3Embedder()
    store = VectorStoreClient()
    retriever = HybridRetriever(store, embedder)
    synthesizer = RAGSynthesizer(retriever)

    resp = synthesizer.answer_query("How do I make chocolate cake?")
    assert resp.retrieval_success is False
    assert "don't have a confirmed" in resp.answer.lower()


def test_minio_document_store_upload_and_retrieval():
    minio_store = MinioDocumentStore()
    content = b"Official Heatwave Action Plan Text"
    uri = minio_store.upload_document("heatwave_plan.txt", content)
    assert uri.startswith("minio://") or uri.startswith("memory://")
    fetched = minio_store.get_document("heatwave_plan.txt")
    assert fetched == content


def test_ragas_production_evaluation_benchmark():
    evaluator = RAGEvaluator(
        threshold_context_relevance=0.80,
        threshold_faithfulness=0.85,
        threshold_answer_relevance=0.80,
        threshold_umbrela=0.80,
    )
    res = evaluator.evaluate_production_baseline()

    assert res.passed is True
    assert res.context_relevance >= 0.80
    assert res.faithfulness >= 0.85
    assert res.answer_relevance >= 0.80
    assert res.umbrela_score >= 0.80
    assert res.citation_groundedness == 1.0
    assert "context_relevance" in res.state_scores
    assert "umbrela_score" in res.state_scores
