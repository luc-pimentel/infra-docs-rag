"""The normalized record every source document becomes."""

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel


class SourceType(StrEnum):
    HTML = "html"
    MARKDOWN = "markdown"
    TEXT = "text"
    PDF = "pdf"


class ParseStatus(StrEnum):
    OK = "ok"  # text came straight from the document
    OCR_FALLBACK = "ocr_fallback"  # at least one page had no text layer and was read with OCR
    EMPTY = "empty"  # parsed without errors, but nothing usable came out
    FAILED = "failed"  # unreadable: damaged, encrypted, or the parser raised


class Span(BaseModel):
    """A character range of `cleaned_text`, end-exclusive."""

    start: int
    end: int


class Section(Span):
    title: str
    level: int
    path: list[str]  # headings from the top of the document down to this one


class Page(Span):
    number: int  # 1-based
    method: Literal["text", "ocr"]


class Provenance(BaseModel):
    local_path: str
    source_sha256: str
    fetched_at: str | None
    parser: str
    ingest_version: str
    ingested_at: str


class Duplicate(BaseModel):
    of: str  # document_id of the copy that is kept
    kind: Literal["exact", "near"]
    similarity: float


class Document(BaseModel):
    document_id: str
    source_id: str
    source_uri: str
    source_type: SourceType
    title: str | None
    raw_text: str
    cleaned_text: str
    content_hash: str | None
    sections: list[Section] = []
    pages: list[Page] = []
    parse_status: ParseStatus
    parse_error: str | None = None
    provenance: Provenance
    duplicate: Duplicate | None = None

    def locate(self, offset: int) -> tuple[Page | None, Section | None]:
        """Return the page and section that contain a `cleaned_text` offset."""
        page = next((p for p in self.pages if p.start <= offset < p.end), None)
        section = next((s for s in self.sections if s.start <= offset < s.end), None)
        return page, section
