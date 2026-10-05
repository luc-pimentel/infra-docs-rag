"""Keyword retrieval: BM25 over the same chunks the vector index holds.

Dense retrieval maps meaning, so a question and its answer need not share a word. The price is
that it blurs exact terms: `maxSurge` and `maxUnavailable` embed close together, a metric name is
just a long word, and an acronym is whatever its letters happen to resemble. BM25 scores a chunk
by the query terms it actually contains, weighted by how rare each term is across the corpus and
saturating with repetition, so a chunk that holds the one exact key the question names wins outright.

The index is built in memory from the vector index's chunks every time, which takes milliseconds
at this size; there is nothing to save. Scores are BM25 scores: non-negative, unbounded, and only
comparable within one query, which is why fusion normalizes them (see `fusion.py`).
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np

from ..chunk.chunks import Chunk
from ..embed.index import Hit

# A token is a run of letters, digits and the punctuation identifiers are made of, so
# `progressDeadlineSeconds`, `keep_firing_for`, `argocd.argoproj.io/skip-reconcile`, `--to-revision`,
# `$labels` and `.spec.os.name` each survive whole. Trailing sentence punctuation is dropped.
TOKEN = re.compile(r"[A-Za-z0-9_$.][A-Za-z0-9_.\-/$:]*")
# The pieces an identifier is made of: camelCase words, snake_case parts, dotted and dashed segments.
PART = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
# Function words that carry no retrieval signal. BM25's idf already discounts them, but dropping
# them keeps a question's "how do I" from matching every chunk a little.
STOPWORDS = frozenset(
    "a an and are as at be by can do does for from how i if in is it its my of on or that the this to "
    "what when where which who why will with you your".split()
)


def tokenize(text: str) -> list[str]:
    """Lowercased tokens for BM25. An identifier goes in whole and, when it has several parts, in parts
    too: `maxSurge` yields `maxsurge`, `max`, `surge`, so the query "max surge" still reaches it while the
    whole token is what tells `maxSurge` and `maxUnavailable` apart."""
    tokens: list[str] = []
    for raw in TOKEN.findall(text):
        whole = raw.strip(".:/-").casefold()
        if not whole or whole in STOPWORDS:
            continue
        tokens.append(whole)
        parts = [p.casefold() for p in PART.findall(raw)]
        if len(parts) > 1:
            tokens.extend(p for p in parts if p not in STOPWORDS)
    return tokens


@dataclass
class Lexical:
    """A BM25 index over chunks. `build` it from the chunks of a vector `Index`; `rank` a query against it."""

    chunks: list[Chunk]
    postings: dict[str, tuple[np.ndarray, np.ndarray]]  # term -> (chunk indices, term counts in each)
    idf: dict[str, float]
    lengths: np.ndarray  # tokens per chunk
    k1: float = 1.5  # how fast repeated occurrences of a term stop adding to the score
    b: float = 0.75  # how much a long chunk is penalized for being long (0 none, 1 fully)

    @classmethod
    def build(cls, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75) -> "Lexical":
        """Index what the embedding model read (`Chunk.embedded`): the context line, then the text, so
        a section's heading counts for keyword search the way it does for dense search."""
        counts = [Counter(tokenize(c.embedded)) for c in chunks]
        lengths = np.fromiter((sum(c.values()) for c in counts), dtype=np.float64, count=len(chunks))
        by_term: dict[str, list[tuple[int, int]]] = {}
        for i, count in enumerate(counts):
            for term, n in count.items():
                by_term.setdefault(term, []).append((i, n))
        postings = {
            term: (
                np.fromiter((i for i, _ in hits), dtype=np.int64, count=len(hits)),
                np.fromiter((n for _, n in hits), dtype=np.float64, count=len(hits)),
            )
            for term, hits in by_term.items()
        }
        total = len(chunks)
        idf = {
            term: math.log(1 + (total - len(hits) + 0.5) / (len(hits) + 0.5))
            for term, hits in by_term.items()
        }
        return cls(chunks, postings, idf, lengths, k1, b)

    @property
    def vocabulary(self) -> int:
        return len(self.postings)

    def scores(self, query: str) -> np.ndarray:
        """One BM25 score per chunk; 0 for a chunk that shares no term with the query."""
        scores = np.zeros(len(self.chunks), dtype=np.float64)
        if not self.chunks:
            return scores
        average = float(self.lengths.mean()) or 1.0
        for term, repeats in Counter(tokenize(query)).items():
            posting = self.postings.get(term)
            if posting is None:
                continue
            indices, tf = posting
            norm = self.k1 * (1 - self.b + self.b * self.lengths[indices] / average)
            scores[indices] += repeats * self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return scores

    def rank(self, query: str, k: int, within: np.ndarray | None = None) -> list[Hit]:
        """The k chunks that share the most weighted terms with the query, among the ones `within` marks
        (every chunk when None). A chunk that shares no term is left out: BM25 has nothing to say about
        it, and a zero would only be a tie."""
        scores = self.scores(query)
        if within is None:
            candidates = np.arange(len(self.chunks))
        elif within.shape != (len(self.chunks),):
            raise ValueError(
                f"the mask marks {within.shape[0]} chunks but the index holds {len(self.chunks)}"
            )
        else:
            candidates = np.flatnonzero(within)
        candidates = candidates[scores[candidates] > 0]
        order = candidates[np.argsort(-scores[candidates], kind="stable")[:k]]
        return [Hit(rank=n + 1, score=float(scores[i]), chunk=self.chunks[i]) for n, i in enumerate(order)]

    def explain(self, query: str) -> list[tuple[str, int, float]]:
        """Each query term the index knows, with how many chunks hold it and its idf: why a term matters."""
        return [
            (term, len(self.postings[term][0]), self.idf[term])
            for term in dict.fromkeys(tokenize(query))
            if term in self.postings
        ]
