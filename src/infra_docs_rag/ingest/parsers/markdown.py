"""Markdown sources (docs repos): front matter for the title, headings for sections."""

import re

import yaml

from ..clean import FENCE_BLOCK, split_front_matter, strip_markdown_noise
from ..models import ParseStatus
from .base import Parsed, Part, markdown_parts

PARSER = "markdown (built-in)"


def outside_code(text: str, rule) -> str:
    """Apply a cleaning rule everywhere except inside fenced code blocks."""
    pieces = FENCE_BLOCK.split(text)
    return "".join(piece if FENCE_BLOCK.fullmatch(piece) else rule(piece) for piece in pieces)


def parse_markdown(data: bytes, fallback_title: str) -> Parsed:
    raw_text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    front_matter, body = split_front_matter(raw_text)
    try:
        meta = yaml.safe_load(front_matter) if front_matter else {}
    except yaml.YAMLError:
        meta = {}
    meta = meta if isinstance(meta, dict) else {}

    parts = markdown_parts(outside_code(body, strip_markdown_noise))
    h1 = next((p.text for p in parts if p.kind == "heading" and p.level == 1), None)
    title = str(meta.get("title") or h1 or fallback_title)
    if h1 is None:  # docs sites render the front-matter title as the page's h1
        parts.insert(0, Part("heading", re.sub(r"\s+", " ", title).strip(), level=1))

    if not any(p.kind == "text" for p in parts):
        return Parsed(PARSER, raw_text=raw_text, title=title, status=ParseStatus.EMPTY, error="no text")
    return Parsed(PARSER, raw_text=raw_text, title=title, parts=parts)
