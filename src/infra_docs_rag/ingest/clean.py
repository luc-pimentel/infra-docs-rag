"""Text cleaning rules shared by the parsers.

Every rule here removes noise that would otherwise end up in chunks and embeddings:
invisible characters, doc-site template syntax, wrapped PDF lines, and headers or
footers repeated on every page.
"""

import re
import unicodedata
from collections import Counter

INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"))


def normalize(text: str) -> str:
    """Unicode-normalize and tidy whitespace without touching indentation."""
    text = unicodedata.normalize("NFKC", text).translate(INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- Markdown and doc-site templates -------------------------------------------------

FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
FENCE_BLOCK = re.compile(
    r"(^[ \t]*(?:```|~~~)[^\n]*\n.*?^[ \t]*(?:```|~~~)[ \t]*$)", re.MULTILINE | re.DOTALL
)
SHORTCODE_WITH_TEXT = re.compile(r"\{\{[<%]\s*[\w-]+\s[^}]*?\btext=\"([^\"]*)\"[^}]*?[%>]\}\}")
SHORTCODE = re.compile(r"\{\{[<%].*?[%>]\}\}", re.DOTALL)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Known tags only: docs are full of placeholders like <pod-name> that must survive.
HTML_TAG = re.compile(
    r"</?(?:a|abbr|b|blockquote|br|center|code|dd|details|div|dl|dt|em|figcaption|figure|font|h[1-6]"
    r"|hr|i|iframe|img|kbd|li|ol|p|pre|section|small|source|span|strong|sub|summary|sup|table|tbody"
    r"|td|th|thead|tr|u|ul|video)\b[^>]*>",
    re.IGNORECASE,
)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
REF_DEFINITION = re.compile(r"^\s*\[[^\]]+\]:\s+\S+.*$", re.MULTILINE)
ADMONITION = re.compile(r'^(\s*)[!?]{3}\+?\s+(\w+)(?:\s+"([^"]*)")?\s*$', re.MULTILINE)


def split_front_matter(text: str) -> tuple[str | None, str]:
    """Return (front matter YAML, body) for a Markdown file."""
    match = FRONT_MATTER.match(text)
    if not match:
        return None, text
    return match.group(1), text[match.end() :]


def strip_markdown_noise(text: str) -> str:
    """Drop template tags, comments, images and link targets; keep the words."""
    text = SHORTCODE_WITH_TEXT.sub(r"\1", text)  # {{< glossary_tooltip text="Pods" ... >}} -> Pods
    text = SHORTCODE.sub("", text)
    text = HTML_COMMENT.sub("", text)
    text = IMAGE.sub("", text)
    text = LINK.sub(r"\1", text)
    text = REF_LINK.sub(r"\1", text)
    text = REF_DEFINITION.sub("", text)
    text = ADMONITION.sub(lambda m: f"{m[1]}{m[2].capitalize()}: {m[3] or ''}".rstrip(), text)
    return HTML_TAG.sub("", text)


# --- PDF ------------------------------------------------------------------------------

PAGE_NUMBER = re.compile(r"^(page\s+)?\d{1,4}(\s*(/|of)\s*\d{1,4})?$", re.IGNORECASE)


def join_wrapped_lines(lines: list[str]) -> str:
    """Rejoin a paragraph that the PDF layout wrapped across lines."""
    out = ""
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if out.endswith("­"):  # soft hyphen: the word was split for layout only
            out = out[:-1] + line
        elif out.endswith("-") and line[:1].islower():  # keep the hyphen, it may be real
            out += line
        else:
            out = f"{out} {line}" if out else line
    return re.sub(r"[ \t]{2,}", " ", out.replace("­", ""))


def line_key(line: str) -> str:
    return re.sub(r"\d+", "#", " ".join(line.lower().split()))


def repeated_edge_lines(pages: list[list[str]], edge: int = 3, min_share: float = 0.5) -> set[str]:
    """Find header/footer lines: text that repeats near the top or bottom of most pages.

    Digits are masked before comparing, so "Page 3" and "Page 4" count as the same line.
    """
    if len(pages) < 3:
        return set()
    seen: Counter[str] = Counter()
    for lines in pages:
        edges = lines[:edge] + lines[-edge:]
        seen.update({line_key(line) for line in edges if line.strip()})
    threshold = max(3, min_share * len(pages))
    return {key for key, count in seen.items() if count >= threshold}
