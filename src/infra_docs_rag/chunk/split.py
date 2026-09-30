"""Cutting text into spans that fit a token budget, on the most natural boundary that works.

A span is a `(start, end)` pair of character offsets into a document's `cleaned_text`,
end-exclusive, so every chunk is an exact slice of the text it came from. Sizes are counted in the
embedding model's own tokens: one pass of its tokenizer over the document locates every token, and
any span is then measured by counting the tokens that start inside it.
"""

import bisect
import re
from itertools import pairwise

Span = tuple[int, int]

# Boundaries from coarse to fine. A cut lands right after the separator, so the separator stays with
# the text before it and consecutive pieces tile the span with no gaps.
PARAGRAPH = re.compile(r"\n[ \t]*\n\s*")
LINE = re.compile(r"\n\s*")
SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'`(\[*-])")
WORD = re.compile(r"\s+")
BOUNDARIES = (PARAGRAPH, LINE, SENTENCE, WORD)


class Tokens:
    """Where a document's tokens start, from one pass of the model's tokenizer."""

    def __init__(self, spans: list[Span]) -> None:
        self.starts = [start for start, _ in spans]

    def count(self, start: int, end: int) -> int:
        """Tokens that start inside [start, end). Cuts only fall on whitespace or token edges, so the
        counts of neighbouring spans add up to the count of the two together."""
        return bisect.bisect_left(self.starts, end) - bisect.bisect_left(self.starts, start)

    def cut(self, start: int, end: int, budget: int) -> list[Span]:
        """Cut every `budget` tokens, for a run of text with no whitespace left to break on."""
        first, last = bisect.bisect_left(self.starts, start), bisect.bisect_left(self.starts, end)
        bounds = [start, *(self.starts[i] for i in range(first + budget, last, budget)), end]
        return list(pairwise(bounds))


def split(text: str, span: Span, boundary: re.Pattern[str]) -> list[Span]:
    """Tile a span into pieces that end right after each match of `boundary`."""
    start, end = span
    cuts = [m.end() for m in boundary.finditer(text, start, end) if start < m.end() < end]
    bounds = [start, *cuts, end]
    return list(pairwise(bounds))


def pieces(text: str, span: Span, boundaries: tuple[re.Pattern[str], ...]) -> list[Span]:
    """Tile a span at every match of every boundary."""
    spans = [span]
    for boundary in boundaries:
        spans = [piece for s in spans for piece in split(text, s, boundary)]
    return spans


def pack(units: list[Span], tokens: Tokens, budget: int, overlap: int = 0) -> list[Span]:
    """Join consecutive units into spans of at most `budget` tokens. Each span after the first starts
    with the last units of the one before, as many as fit in `overlap` tokens. A unit bigger than
    the budget becomes a span of its own; callers cut those first."""
    sizes = [tokens.count(*unit) for unit in units]
    packed: list[Span] = []
    i = 0
    while i < len(units):
        j, total = i + 1, sizes[i]
        while j < len(units) and total + sizes[j] <= budget:
            total += sizes[j]
            j += 1
        packed.append((units[i][0], units[j - 1][1]))
        if j == len(units):
            break
        k, carried = j, 0
        while k - 1 > i and carried + sizes[k - 1] <= overlap:
            k -= 1
            carried += sizes[k]
        i = k
    return packed


def fit(
    text: str,
    span: Span,
    tokens: Tokens,
    budget: int,
    overlap: int = 0,
    boundaries: tuple[re.Pattern[str], ...] = BOUNDARIES,
) -> list[Span]:
    """Cut a span into pieces of at most `budget` tokens at the coarsest boundary that divides it:
    paragraphs, then lines, sentences and words. Neighbouring pieces that fit are joined back up to
    the budget; a piece still too big is cut again at the next boundary down, and text with no
    boundary left is cut at token edges."""
    if tokens.count(*span) <= budget:
        return [span]
    for n, boundary in enumerate(boundaries):
        units = split(text, span, boundary)
        if len(units) > 1:
            finer = boundaries[n + 1 :]
            break
    else:
        return tokens.cut(*span, budget)
    out: list[Span] = []
    run: list[Span] = []
    for unit in units:
        if tokens.count(*unit) <= budget:
            run.append(unit)
            continue
        out += pack(run, tokens, budget, overlap)
        run = []
        out += fit(text, unit, tokens, budget, overlap, finer)
    return out + pack(run, tokens, budget, overlap)


def trim(text: str, spans: list[Span]) -> list[Span]:
    """Move each span's ends past the whitespace around it, and drop spans with nothing else."""
    trimmed: list[Span] = []
    for start, end in spans:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if start < end:
            trimmed.append((start, end))
    return trimmed
