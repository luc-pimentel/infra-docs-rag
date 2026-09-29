from ..models import SourceType
from .base import Parsed
from .html import parse_html
from .markdown import parse_markdown
from .pdf import parse_pdf
from .text import parse_text

PARSERS = {
    SourceType.HTML: parse_html,
    SourceType.MARKDOWN: parse_markdown,
    SourceType.TEXT: parse_text,
    SourceType.PDF: parse_pdf,
}


def parse(source_type: SourceType, data: bytes, fallback_title: str) -> Parsed:
    return PARSERS[source_type](data, fallback_title)
