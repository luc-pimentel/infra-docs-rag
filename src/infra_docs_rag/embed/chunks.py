"""The units that get embedded: one chunk per ingestion section, cut from `cleaned_text`.

Chunking proper (size limits, overlap, heading context) is the next stage. Until then a chunk is
exactly one section, heading line included, so every search hit keeps the section's provenance.
"""

from pydantic import BaseModel

from ..ingest.models import Document, ParseStatus

USABLE = (ParseStatus.OK, ParseStatus.OCR_FALLBACK)


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    document_id: str
    source_uri: str
    section_path: list[str]  # headings from the top of the document down to this section
    start: int  # span of the document's `cleaned_text`, end-exclusive
    end: int
    pages: list[int] = []  # PDF pages the span touches
    text: str

    @property
    def section(self) -> str:
        return self.section_path[-1]

    def citation(self) -> str:
        """Where the chunk comes from, e.g. `k8s-service › Service › type: ExternalName`."""
        where = " › ".join([self.source_id, *self.section_path])
        if self.pages:
            first, last = self.pages[0], self.pages[-1]
            where += f", page {first}" if first == last else f", pages {first}-{last}"
        return where


def section_chunks(docs: list[Document]) -> list[Chunk]:
    """One chunk per section with text under its heading; duplicates and failed parses are skipped."""
    chunks: list[Chunk] = []
    for doc in docs:
        if doc.duplicate or doc.parse_status not in USABLE:
            continue
        for section in doc.sections:
            text = doc.cleaned_text[section.start : section.end]
            if not text.removeprefix(section.title).strip():
                continue  # a parent heading followed directly by its first child
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_id}:{section.start}",
                    source_id=doc.source_id,
                    document_id=doc.document_id,
                    source_uri=doc.source_uri,
                    section_path=section.path,
                    start=section.start,
                    end=section.end,
                    pages=[p.number for p in doc.pages if p.start < section.end and p.end > section.start],
                    text=text,
                )
            )
    return chunks
