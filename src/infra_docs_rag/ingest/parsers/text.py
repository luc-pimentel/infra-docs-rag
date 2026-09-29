"""Plain-text sources, such as commented config references. Indentation is kept."""

from ..clean import normalize
from ..models import ParseStatus
from .base import Parsed, Part

PARSER = "text (built-in)"


def parse_text(data: bytes, fallback_title: str) -> Parsed:
    raw_text = data.decode("utf-8", errors="replace")
    cleaned = normalize(raw_text)
    if not cleaned:
        return Parsed(
            PARSER, raw_text=raw_text, title=fallback_title, status=ParseStatus.EMPTY, error="no text"
        )
    return Parsed(PARSER, raw_text=raw_text, title=fallback_title, parts=[Part("text", cleaned)])
