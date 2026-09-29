"""HTML pages: boilerplate removal with trafilatura, headings kept as sections."""

from copy import deepcopy

import trafilatura
from lxml import etree
from lxml import html as lxml_html

from ..models import ParseStatus
from .base import CALLOUT, Parsed, markdown_parts

PARSER = f"trafilatura {trafilatura.__version__}"
MIN_WORDS = 25  # less "main content" than this is navigation that slipped through

BLOCK_TAGS = {
    "address", "article", "aside", "blockquote", "br", "dd", "details", "div", "dl", "dt",
    "figcaption", "figure", "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header",
    "hr", "li", "main", "nav", "ol", "p", "pre", "section", "summary", "table", "td", "th",
    "tr", "ul",
}  # fmt: skip
PERMALINKS = (
    "//*[self::h1 or self::h2 or self::h3 or self::h4]"
    "//a[not(normalize-space()) or normalize-space()='¶' or normalize-space()='#']"
)


def visible_text(tree: lxml_html.HtmlElement) -> str:
    """Everything a naive scraper would index: all visible text, one block per line."""
    tree = deepcopy(tree)
    for element in tree.xpath("//head|//script|//style|//noscript|//template|//svg"):
        element.drop_tree()
    for element in tree.iter():
        if isinstance(element.tag, str) and element.tag in BLOCK_TAGS:
            element.text = "\n" + (element.text or "")
            element.tail = "\n" + (element.tail or "")
    lines = (" ".join(line.split()) for line in tree.text_content().split("\n"))
    return "\n".join(line for line in lines if line)


def keep_headings(tree: lxml_html.HtmlElement) -> None:
    """Rewrite h1-h4 as Markdown-prefixed paragraphs.

    trafilatura drops some sites' headings (kubernetes.io's, for one), which would lose
    the section structure. As paragraphs, they survive extraction and come back as
    `## Heading` lines.
    """
    for anchor in tree.xpath(PERMALINKS):
        anchor.drop_tree()
    for heading in tree.xpath("//h1|//h2|//h3|//h4"):
        text = " ".join(heading.text_content().split())
        paragraph = etree.Element("p")
        paragraph.text = text if CALLOUT.match(text) else f"{'#' * int(heading.tag[1])} {text}"
        paragraph.tail = heading.tail
        heading.getparent().replace(heading, paragraph)


def parse_html(data: bytes, fallback_title: str) -> Parsed:
    html = data.decode("utf-8", errors="replace")
    try:
        tree = lxml_html.fromstring(html)
    except (etree.ParserError, ValueError) as exc:
        return Parsed(PARSER, raw_text=html, status=ParseStatus.FAILED, error=f"not parseable as HTML: {exc}")

    raw_text = visible_text(tree)
    keep_headings(tree)
    markdown = trafilatura.extract(
        lxml_html.tostring(tree, encoding="unicode"),
        output_format="markdown",
        include_formatting=True,
        include_tables=True,
        include_links=False,
        include_images=False,
        include_comments=False,
    )
    parts = markdown_parts(markdown) if markdown else []
    h1 = next((p.text for p in parts if p.kind == "heading" and p.level == 1), None)
    page_title = tree.findtext(".//title")
    title = h1 or (page_title.split(" | ")[0].strip() if page_title else None) or fallback_title

    words = sum(len(p.text.split()) for p in parts if p.kind == "text")
    if words < MIN_WORDS:
        return Parsed(
            PARSER,
            raw_text=raw_text,
            title=title,
            status=ParseStatus.EMPTY,
            error=f"no main content: {words} words survived boilerplate removal (navigation only)",
        )
    return Parsed(PARSER, raw_text=raw_text, title=title, parts=parts)
