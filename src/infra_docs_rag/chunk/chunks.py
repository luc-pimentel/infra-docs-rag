"""Chunks: the pieces of each document that get embedded and retrieved, with where they came from.

Four strategies cut the same cleaned text:

- `whole-sections`: one chunk per section, however long. The embedding stage's baseline.
- `fixed`: windows of whole words, a set number of tokens each, blind to sentences and headings.
- `sentences`: whole sentences, and whole lines for lists, code and config, packed up to the size.
- `section-aware`: one chunk per section when it fits. A longer section is cut at paragraphs, then
  lines, sentences and words, and every piece keeps the section's path.

A chunk's text is always an exact slice of the document's `cleaned_text`, so its offsets, section
path and pages trace it back to the source. With `context` on, the project and section path are
written in front of the text when it is embedded: a piece cut from the middle of a section still
says what it is about.
"""

import bisect
from typing import Literal, Protocol

from pydantic import BaseModel, model_validator

from ..ingest.models import Document, ParseStatus, Section
from .split import LINE, SENTENCE, WORD, Span, Tokens, fit, pack, pieces, split, trim

USABLE = (ParseStatus.OK, ParseStatus.OCR_FALLBACK)

Strategy = Literal["whole-sections", "fixed", "sentences", "section-aware"]
STRATEGIES: tuple[Strategy, ...] = ("whole-sections", "fixed", "sentences", "section-aware")


class Chunk(BaseModel):
    chunk_id: str
    source_id: str
    document_id: str
    source_uri: str
    project: str | None = None  # the documentation set the source belongs to, from sources.yaml
    section_path: list[str]  # headings from the top of the document down to this chunk's section
    start: int  # span of the document's `cleaned_text`, end-exclusive
    end: int
    pages: list[int] = []  # PDF pages the span touches
    text: str
    context: str = ""  # written in front of the text when it is embedded; empty when context is off

    @property
    def section(self) -> str:
        return self.section_path[-1]

    @property
    def embedded(self) -> str:
        """What the model reads: the context line, then the text."""
        return f"{self.context}\n\n{self.text}" if self.context else self.text

    def citation(self) -> str:
        """Where the chunk comes from, e.g. `k8s-service › Service › type: ExternalName`."""
        return locate([self.source_id, *self.section_path], self.pages)


def locate(path: list[str], pages: list[int]) -> str:
    """A path of names joined with ›, then the pages, e.g. `Whitepaper › Layers, pages 9-10`."""
    where = " › ".join(path)
    if pages:
        first, last = pages[0], pages[-1]
        where += f", page {first}" if first == last else f", pages {first}-{last}"
    return where


class Chunking(BaseModel):
    """How an index's chunks were cut. The default is the embedding stage's: whole sections."""

    strategy: Strategy = "whole-sections"
    size: int | None = None  # most tokens the model reads per chunk, context and special tokens included
    overlap: int = 0  # tokens a chunk repeats from the end of the one before it
    context: bool = False  # embed the project and section path in front of the text

    @model_validator(mode="after")
    def consistent(self) -> "Chunking":
        if (self.strategy == "whole-sections") != (self.size is None):
            raise ValueError("whole-sections takes no size; every other strategy needs one")
        if self.size is None and self.overlap:
            raise ValueError("whole-sections do not overlap")
        if self.size is not None and not 0 <= 2 * self.overlap < self.size:
            raise ValueError("overlap must be under half the size")
        return self

    @property
    def label(self) -> str:
        label = self.strategy
        if self.size is not None:
            label += f" {self.size}" + (f"/{self.overlap}" if self.overlap else "")
        return label + (" + context" if self.context else "")


# What `embed` builds its index with: sections kept whole up to everything bge reads, longer ones cut
# with an eighth of overlap, and the context line in front. reports/chunking.md shows how it was chosen.
DEFAULT_CHUNKING = Chunking(strategy="section-aware", size=512, overlap=64, context=True)


class Tokenizer(Protocol):
    """The part of an embedding model that chunking needs: its tokens and its limit."""

    max_tokens: int
    special_tokens: int  # added around every input, such as [CLS] and [SEP]

    def token_spans(self, text: str) -> list[Span]: ...


def usable(docs: list[Document]) -> list[Document]:
    """Documents worth chunking: parsed, and not a copy of one that is kept."""
    return [d for d in docs if not d.duplicate and d.parse_status in USABLE]


def has_text(doc: Document, section: Section) -> bool:
    """False for a parent heading followed directly by its first child."""
    return bool(doc.cleaned_text[section.start : section.end].removeprefix(section.title).strip())


def chunk_documents(
    docs: list[Document],
    chunking: Chunking,
    tokenizer: Tokenizer | None = None,
    projects: dict[str, str] | None = None,
) -> list[Chunk]:
    """Cut every usable document with one strategy. `projects` maps source ids to the documentation
    set each belongs to; `tokenizer` measures sizes, and every strategy but whole sections needs it."""
    projects = projects or {}
    chunks: list[Chunk] = []
    for doc in usable(docs):
        project = projects.get(doc.source_id)
        if chunking.strategy == "whole-sections":
            # The embedding stage's chunks, untouched: documents without headings are left out
            cuts = [((s.start, s.end), s.path) for s in doc.sections if has_text(doc, s)]
        else:
            if tokenizer is None:
                raise ValueError(f"{chunking.strategy} needs the model's tokenizer to measure chunks")
            cuts = cut(doc, chunking, tokenizer, project)
        for (start, end), path in cuts:
            pages = [p.number for p in doc.pages if p.start < end and p.end > start]
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source_id}:{start}",
                    source_id=doc.source_id,
                    document_id=doc.document_id,
                    source_uri=doc.source_uri,
                    project=project,
                    section_path=path,
                    start=start,
                    end=end,
                    pages=pages,
                    text=doc.cleaned_text[start:end],
                    context=context_line(project, path, pages) if chunking.context else "",
                )
            )
    return chunks


def context_line(project: str | None, path: list[str], pages: list[int]) -> str:
    """The line embedded in front of a chunk, e.g. `Argo CD › Getting Started › Requirements`."""
    return locate([project, *path] if project else path, pages)


def sections_of(doc: Document) -> list[tuple[Span, list[str]]]:
    """Each section with text of its own, with its path. Text outside every section, such as a whole
    document without headings, is filed under the document's title."""
    title = [doc.title or doc.source_id]
    text = doc.cleaned_text
    spans: list[tuple[Span, list[str]]] = []
    covered = 0
    for section in doc.sections:
        if text[covered : section.start].strip():
            spans.append(((covered, section.start), title))
        if has_text(doc, section):
            spans.append(((section.start, section.end), section.path))
        covered = max(covered, section.end)
    if text[covered:].strip():
        spans.append(((covered, len(text)), title))
    return spans


def cut(
    doc: Document, chunking: Chunking, tokenizer: Tokenizer, project: str | None
) -> list[tuple[Span, list[str]]]:
    """Spans and section paths for one document, each small enough for the model to read whole."""
    assert chunking.size is not None
    text = doc.cleaned_text
    tokens = Tokens(tokenizer.token_spans(text))
    room = min(chunking.size, tokenizer.max_tokens) - tokenizer.special_tokens
    sections = sections_of(doc)

    def header(span: Span, path: list[str]) -> int:
        """Tokens the context line takes, with every page the section touches."""
        if not chunking.context:
            return 0
        pages = [p.number for p in doc.pages if p.start < span[1] and p.end > span[0]]
        return len(tokenizer.token_spans(context_line(project, path, pages)))

    def budget(reserved: int) -> int:
        if 2 * reserved > room:
            raise ValueError(
                f"size {chunking.size} leaves too little room for text after {doc.source_id}'s context line"
            )
        return room - reserved

    if chunking.strategy == "section-aware":
        cuts: list[tuple[Span, list[str]]] = []
        for span, path in sections:
            fitted = fit(text, span, tokens, budget(header(span, path)), chunking.overlap)
            cuts += [(piece, path) for piece in trim(text, fitted)]
        return cuts

    # Fixed windows and sentence packing run over the whole document and ignore headings. The context
    # line is budgeted at its longest, plus a few tokens for a window that touches more pages than the
    # section it starts in, and each chunk is filed under the section it starts in.
    longest = max((header(span, path) for span, path in sections), default=0)
    limit = budget(longest + (4 if chunking.context else 0))
    everything = (0, len(text))
    if chunking.strategy == "fixed":
        units = split(text, everything, WORD)
    else:
        units = pieces(text, everything, (LINE, SENTENCE))
    units = [piece for unit in units for piece in fit(text, unit, tokens, limit, 0, (WORD,))]
    starts = [s.start for s in doc.sections]
    title = [doc.title or doc.source_id]

    def path_at(offset: int) -> list[str]:
        n = bisect.bisect_right(starts, offset)
        return doc.sections[n - 1].path if n else title

    return [(span, path_at(span[0])) for span in trim(text, pack(units, tokens, limit, chunking.overlap))]
