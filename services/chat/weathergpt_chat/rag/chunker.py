import hashlib
import re
from dataclasses import dataclass

from .models import DocumentChunk, DocumentType


@dataclass(frozen=True)
class ParsedPage:
    page_number: int
    text: str


class DocumentChunker:
    def __init__(self, max_chars: int = 1800, overlap_chars: int = 250) -> None:
        self.max_chars = max_chars
        self.overlap_chars = overlap_chars

    def chunk_document(
        self,
        doc_id: str,
        doc_name: str,
        doc_type: DocumentType,
        text: str,
        region: str = "all",
        language: str = "en",
        doc_date: str = "2026-01-01",
        pages: list[ParsedPage] | None = None,
    ) -> list[DocumentChunk]:
        source_pages = pages or [ParsedPage(1, text)]
        chunks: list[DocumentChunk] = []
        for page in source_pages:
            page_text = page.text.strip()
            if not page_text:
                continue
            sections = re.split(r"(?=^(?:Section|SECTION|Question|QUESTION)\s+[^:\n]+:?)", page_text, flags=re.MULTILINE)
            for section_index, section in enumerate(sections):
                content = section.strip()
                if not content:
                    continue
                section_match = re.match(r"^(?:Section|SECTION|Question|QUESTION)\s+([^:\n]+):?", content)
                section_title = section_match.group(1).strip() if section_match else "Document Content"
                start = 0
                while start < len(content):
                    end = min(len(content), start + self.max_chars)
                    chunk_text = content[start:end].strip()
                    if chunk_text:
                        chunk_id = hashlib.sha256(f"{doc_id}:{page.page_number}:{section_index}:{start}:{chunk_text}".encode("utf-8")).hexdigest()[:32]
                        chunks.append(
                            DocumentChunk(
                                chunk_id=chunk_id,
                                doc_id=doc_id,
                                doc_name=doc_name,
                                doc_type=doc_type,
                                section_title=section_title,
                                page_number=page.page_number,
                                content=chunk_text,
                                region=region,
                                language=language,
                                doc_date=doc_date,
                                metadata={"page_start": page.page_number, "page_end": page.page_number},
                            )
                        )
                    if end >= len(content):
                        break
                    start = max(start + 1, end - self.overlap_chars)
        return chunks
