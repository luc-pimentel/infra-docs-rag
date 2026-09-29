"""What every parser returns, and how parts become `cleaned_text` with offsets."""

import re
from dataclasses import dataclass, field
from typing import Literal

from ..clean import normalize
from ..models import Page, ParseStatus, Section

SEPARATOR = "\n\n"


@dataclass
class Part:
    """A heading or a block of text, in reading order."""

    kind: Literal["heading", "text"]
    text: str
    level: int = 0
    page: int | None = None
    method: Literal["text", "ocr"] = "text"


@dataclass
class Parsed:
    parser: str
    raw_text: str = ""
    title: str | None = None
    parts: list[Part] = field(default_factory=list)
    status: ParseStatus = ParseStatus.OK
    error: str | None = None


def drop_trailing_headings(parts: list[Part]) -> list[Part]:
    """Drop headings with no text after them, like a "What's next" whose links were removed.

    Empty headings mid-document stay: in PDFs a parent heading is often set in the same
    size as its children, so "no text before the next heading" does not mean empty.
    """
    last_text = max((i for i, p in enumerate(parts) if p.kind == "text"), default=-1)
    return parts[: last_text + 1]


def assemble(parts: list[Part]) -> tuple[str, list[Section], list[Page]]:
    """Join parts into `cleaned_text` and record where each section and page lands in it."""
    parts = drop_trailing_headings([p for p in parts if p.text.strip()])
    spans: list[tuple[int, int]] = []
    position = 0
    for i, part in enumerate(parts):
        if i:
            position += len(SEPARATOR)
        spans.append((position, position + len(part.text)))
        position += len(part.text)
    text = SEPARATOR.join(p.text for p in parts)

    sections: list[Section] = []
    stack: list[tuple[int, str]] = []
    headings = [i for i, p in enumerate(parts) if p.kind == "heading"]
    for n, i in enumerate(headings):
        part = parts[i]
        while stack and stack[-1][0] >= part.level:
            stack.pop()
        stack.append((part.level, part.text))
        end = spans[headings[n + 1]][0] - len(SEPARATOR) if n + 1 < len(headings) else len(text)
        sections.append(
            Section(
                title=part.text,
                level=part.level,
                path=[title for _, title in stack],
                start=spans[i][0],
                end=end,
            )
        )

    pages: list[Page] = []
    for i, part in enumerate(parts):
        if part.page is None:
            continue
        if pages and pages[-1].number == part.page:
            pages[-1].end = spans[i][1]
        else:
            pages.append(Page(number=part.page, method=part.method, start=spans[i][0], end=spans[i][1]))
    return text, sections, pages


HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
HEADING_ID = re.compile(r"\s*\{#[^}]*\}\s*$")
CALLOUT = re.compile(r"^(note|caution|warning|tip|info|important|danger|attention)\s*:?$", re.IGNORECASE)


def markdown_parts(markdown: str, max_level: int = 4) -> list[Part]:
    """Split Markdown into heading and text parts, ignoring `#` lines inside code blocks."""
    parts: list[Part] = []
    buffer: list[str] = []
    in_code = False

    def flush() -> None:
        text = normalize("\n".join(buffer))
        buffer.clear()
        if text:
            parts.append(Part("text", text))

    for line in markdown.split("\n"):
        if FENCE.match(line):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        title = HEADING_ID.sub("", match[2]).rstrip("¶").strip() if match else ""
        if match and len(match[1]) <= max_level and title and not CALLOUT.match(title):
            flush()
            parts.append(Part("heading", normalize(title), level=len(match[1])))
        else:
            buffer.append(title if match else line)
    flush()
    return parts
