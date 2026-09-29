"""PDFs: layout-aware text, OCR for pages without a text layer, headings from font size."""

import re
from collections import Counter
from dataclasses import dataclass

import pymupdf

from ..clean import PAGE_NUMBER, join_wrapped_lines, line_key, normalize, repeated_edge_lines
from ..models import ParseStatus
from .base import Parsed, Part

PARSER = f"pymupdf {pymupdf.VersionBind}"
SUPERSCRIPT = 1  # PyMuPDF span flag; footnote markers like "concerns¹"
HEADING_RATIO = 1.3  # a line this much larger than body text is a heading
HEADING_MAX_WORDS = 12
OCR_DPI = 300
CONTENTS = re.compile(r"^(table of )?contents$", re.IGNORECASE)


@dataclass
class Line:
    text: str
    size: float
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass
class PageText:
    number: int
    method: str
    raw: str
    lines: list[Line]


def join_spans(spans: list[dict]) -> str:
    """Join a line's spans, adding the space OCR leaves out between word boxes."""
    text, right = "", None
    for span in spans:
        gap = span["bbox"][0] - right if right is not None else 0
        if text and not text.endswith(" ") and not span["text"].startswith(" ") and gap > span["size"] * 0.15:
            text += " "
        text += span["text"]
        right = span["bbox"][2]
    return text


def layout_lines(layout: dict) -> list[Line]:
    """Visual lines in reading order, footnote markers dropped, split fragments rejoined."""
    lines: list[Line] = []
    for block in layout["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line["spans"] if s["text"].strip() and not s["flags"] & SUPERSCRIPT]
            if not spans:
                continue
            size = round(max(s["size"] for s in spans), 1)
            text = join_spans(spans)
            previous = lines[-1] if lines else None
            # Layout tools often cut one visual line into pieces ("@mag" + "nologan").
            if (
                previous
                and abs(line["bbox"][1] - previous.y0) < 2
                and abs(size - previous.size) < 0.6
                and -size * 0.5 <= line["bbox"][0] - previous.x1 < size * 2
            ):
                gap = line["bbox"][0] - previous.x1
                joiner = "" if gap < size * 0.15 or previous.text.endswith(" ") else " "
                previous.text += joiner + text
                previous.x1 = line["bbox"][2]
                continue
            lines.append(Line(text, size, *line["bbox"]))
    return lines


def read_page(page: pymupdf.Page) -> PageText:
    """Use the text layer; fall back to OCR when the page has none (a scan)."""
    raw = page.get_text()
    if raw.strip():
        return PageText(page.number + 1, "text", raw, layout_lines(page.get_text("dict")))
    ocr = page.get_textpage_ocr(language="eng", dpi=OCR_DPI, full=True)
    raw = page.get_text(textpage=ocr)
    return PageText(page.number + 1, "ocr", raw, layout_lines(page.get_text("dict", textpage=ocr)))


def cover_title(pages: list[PageText]) -> str | None:
    """The largest text on the first page, if it stands out, which is how a reader names the file."""
    lines = pages[0].lines if pages else []
    if not lines:
        return None
    biggest = max(line.size for line in lines)
    body = Counter({line.size: len(line.text) for line in lines}).most_common(1)[0][0]
    if biggest < body * HEADING_RATIO:
        return None
    title = normalize(" ".join(" ".join(line.text.split()) for line in lines if line.size == biggest))
    return title if 0 < len(title.split()) <= 15 else None


def parse_pdf(data: bytes, fallback_title: str) -> Parsed:
    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
    except Exception as exc:  # MuPDF raises several error types for broken files
        return Parsed(PARSER, status=ParseStatus.FAILED, error=f"cannot open: {exc}")
    if doc.needs_pass:
        return Parsed(PARSER, status=ParseStatus.FAILED, error="encrypted: the file needs a password")
    if doc.page_count == 0:
        return Parsed(
            PARSER,
            status=ParseStatus.FAILED,
            error="damaged: no readable pages" + (" after MuPDF's repair attempt" if doc.is_repaired else ""),
        )

    pages: list[PageText] = []
    errors: list[str] = []
    for page in doc:
        try:
            pages.append(read_page(page))
        except Exception as exc:  # one bad page (or a missing OCR engine) must not sink the file
            errors.append(f"page {page.number + 1}: {exc}")
    raw_text = "\n\n".join(p.raw for p in pages)
    ocr_used = any(p.method == "ocr" for p in pages)
    parser = PARSER + (" + tesseract OCR" if ocr_used else "")
    if not pages:
        return Parsed(parser, status=ParseStatus.FAILED, error="; ".join(errors))

    parts = page_parts(pages)
    title = (doc.metadata or {}).get("title", "").strip() or cover_title(pages) or fallback_title
    error = "; ".join(errors) or None
    if not any(p.kind == "text" for p in parts):
        reason = "no text on any page, OCR included" if ocr_used else "no text on any page"
        return Parsed(parser, raw_text=raw_text, title=title, status=ParseStatus.EMPTY, error=error or reason)
    status = ParseStatus.OCR_FALLBACK if ocr_used else ParseStatus.OK
    return Parsed(parser, raw_text=raw_text, title=title, parts=parts, status=status, error=error)


def same_paragraph(previous: Line, line: Line) -> bool:
    """Next line of the same paragraph: same size, just below, roughly the same left edge.

    Boxes of large, tightly set type overlap a little, so a small negative gap still counts;
    a big one means the text jumped to a new column.
    """
    height = previous.y1 - previous.y0
    gap = line.y0 - previous.y1
    return (
        abs(line.size - previous.size) < 0.6
        and -height * 0.5 <= gap <= height * 0.6
        and abs(line.x0 - previous.x0) < 40
    )


def page_parts(pages: list[PageText]) -> list[Part]:
    """Turn laid-out lines into heading and paragraph parts, minus headers, footers and TOCs."""
    sizes: Counter[float] = Counter()
    for page in pages:
        for line in page.lines:
            sizes[line.size] += len(line.text)
    if not sizes:
        return []
    body_size = sizes.most_common(1)[0][0]
    heading_sizes = sorted((s for s in sizes if s >= body_size * HEADING_RATIO), reverse=True)
    level = {size: min(rank + 1, 3) for rank, size in enumerate(heading_sizes)}
    # Running headers are set in small type; headings like "Chapter 3" must not count.
    repeated = repeated_edge_lines(
        [[line.text for line in page.lines if line.size not in level] for page in pages]
    )

    def is_heading(line: Line) -> bool:
        return line.size in level and len(line.text.split()) <= HEADING_MAX_WORDS

    parts: list[Part] = []
    in_contents = False
    for page in pages:
        edge = {0, 1, len(page.lines) - 2, len(page.lines) - 1}
        lines = [
            line
            for i, line in enumerate(page.lines)
            if not (i in edge and PAGE_NUMBER.match(line.text.strip()))
            and line_key(line.text) not in repeated
        ]
        groups: list[list[Line]] = []
        for line in lines:
            previous = groups[-1][-1] if groups else None
            if previous and is_heading(line) == is_heading(previous) and same_paragraph(previous, line):
                groups[-1].append(line)
            else:
                groups.append([line])

        for group in groups:
            if is_heading(group[0]):
                text = normalize(" ".join(" ".join(line.text.split()) for line in group))
                in_contents = bool(CONTENTS.match(text))
                if in_contents:
                    continue
                part = Part("heading", text, level=level[max(line.size for line in group)])
            elif in_contents:  # table-of-contents entries only echo the headings
                continue
            else:
                part = Part("text", normalize(join_wrapped_lines([line.text for line in group])))
            part.page, part.method = page.number, page.method
            parts.append(part)
    return parts
