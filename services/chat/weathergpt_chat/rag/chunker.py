import re
from uuid import uuid4
from .models import DocumentChunk, DocumentType


class DocumentChunker:
    def chunk_document(
        self,
        doc_id: str,
        doc_name: str,
        doc_type: DocumentType,
        text: str,
        region: str = "all",
        language: str = "en",
        doc_date: str = "2026-01-01",
    ) -> list[DocumentChunk]:
        if doc_type == DocumentType.GOVERNMENT_SOP:
            return self._chunk_sop(doc_id, doc_name, text, region, language, doc_date)
        elif doc_type == DocumentType.WEATHER_BULLETIN:
            return self._chunk_bulletin_table(doc_id, doc_name, text, region, language, doc_date)
        elif doc_type == DocumentType.FAQ_ADVISORY:
            return self._chunk_faq(doc_id, doc_name, text, region, language, doc_date)
        elif doc_type == DocumentType.CLIMATE_REPORT:
            return self._chunk_climate_report(doc_id, doc_name, text, region, language, doc_date)
        return self._chunk_generic(doc_id, doc_name, doc_type, text, region, language, doc_date)

    def _chunk_sop(
        self, doc_id: str, doc_name: str, text: str, region: str, language: str, doc_date: str
    ) -> list[DocumentChunk]:
        sections = re.split(r"(?=(?:Section\s+\d+|Article\s+\d+|[A-Z0-9\.\s]{4,}:))", text)
        chunks: list[DocumentChunk] = []
        page = 1
        for idx, sec in enumerate(sections):
            clean = sec.strip()
            if not clean:
                continue
            lines = clean.split("\n", 1)
            title = lines[0][:80].strip()
            paragraphs = self._sliding_window_tokens(clean, max_tokens=600, overlap_tokens=90)
            for p_idx, p in enumerate(paragraphs):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{doc_id}-sop-{idx}-{p_idx}",
                        doc_id=doc_id,
                        doc_name=doc_name,
                        doc_type=DocumentType.GOVERNMENT_SOP,
                        content=p,
                        page_number=page,
                        section_title=title,
                        region=region,
                        language=language,
                        doc_date=doc_date,
                    )
                )
            page += 1
        return chunks

    def _chunk_bulletin_table(
        self, doc_id: str, doc_name: str, text: str, region: str, language: str, doc_date: str
    ) -> list[DocumentChunk]:
        blocks = text.split("\n\n")
        chunks: list[DocumentChunk] = []
        for idx, blk in enumerate(blocks):
            clean = blk.strip()
            if len(clean) < 30:
                continue
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}-tbl-{idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_type=DocumentType.WEATHER_BULLETIN,
                    content=clean,
                    page_number=1,
                    section_title="Observation & Warning Table",
                    region=region,
                    language=language,
                    doc_date=doc_date,
                )
            )
        return chunks

    def _chunk_faq(
        self, doc_id: str, doc_name: str, text: str, region: str, language: str, doc_date: str
    ) -> list[DocumentChunk]:
        qa_pairs = re.split(r"(?=(?:Q\d*:|Question:|FAQ:|\n\?))", text)
        chunks: list[DocumentChunk] = []
        for idx, qa in enumerate(qa_pairs):
            clean = qa.strip()
            if len(clean) < 20:
                continue
            first_line = clean.split("\n", 1)[0][:60]
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}-faq-{idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_type=DocumentType.FAQ_ADVISORY,
                    content=clean,
                    page_number=1,
                    section_title=first_line,
                    region=region,
                    language=language,
                    doc_date=doc_date,
                )
            )
        return chunks

    def _chunk_climate_report(
        self, doc_id: str, doc_name: str, text: str, region: str, language: str, doc_date: str
    ) -> list[DocumentChunk]:
        paragraphs = self._sliding_window_tokens(text, max_tokens=900, overlap_tokens=90)
        chunks: list[DocumentChunk] = []
        for idx, p in enumerate(paragraphs):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}-clim-{idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_type=DocumentType.CLIMATE_REPORT,
                    content=p,
                    page_number=(idx // 2) + 1,
                    section_title=f"Climate Trend Analysis Part {idx + 1}",
                    region=region,
                    language=language,
                    doc_date=doc_date,
                )
            )
        return chunks

    def _chunk_generic(
        self, doc_id: str, doc_name: str, doc_type: DocumentType, text: str, region: str, language: str, doc_date: str
    ) -> list[DocumentChunk]:
        paragraphs = self._sliding_window_tokens(text, max_tokens=500, overlap_tokens=50)
        chunks: list[DocumentChunk] = []
        for idx, p in enumerate(paragraphs):
            chunks.append(
                DocumentChunk(
                    chunk_id=f"{doc_id}-gen-{idx}",
                    doc_id=doc_id,
                    doc_name=doc_name,
                    doc_type=doc_type,
                    content=p,
                    page_number=1,
                    section_title="General Guidance",
                    region=region,
                    language=language,
                    doc_date=doc_date,
                )
            )
        return chunks

    def _sliding_window_tokens(self, text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
        words = text.split()
        if len(words) <= max_tokens:
            return [text.strip()]
        result: list[str] = []
        step = max_tokens - overlap_tokens
        for i in range(0, len(words), step):
            sub = " ".join(words[i : i + max_tokens])
            if sub.strip():
                result.append(sub.strip())
        return result
