"""Duplicate detection: exact copies by content hash, near copies by shingle overlap.

Near duplicates are found by comparing every pair's Jaccard similarity over 5-word
shingles (Manning et al., Introduction to Information Retrieval, section 19.6). That is
O(n^2) and fine for a corpus this size; MinHash with LSH is the same idea at scale.
"""

import hashlib
import re
from dataclasses import dataclass

from .models import Document, Duplicate

SHINGLE_WORDS = 5
# Copies of one page in two formats score 0.69-0.88 here; unrelated pages stay below 0.01.
NEAR_DUPLICATE = 0.6


def content_hash(text: str) -> str:
    """Hash of the text with case and whitespace normalized, so reflowed copies match."""
    return hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()


def shingles(text: str, k: int = SHINGLE_WORDS) -> set[str]:
    words = re.findall(r"\w+", text.lower())
    if len(words) <= k:
        return {" ".join(words)}
    return {" ".join(words[i : i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a or b else 0.0


@dataclass
class Pair:
    kept: str
    other: str
    similarity: float


def mark_duplicates(docs: list[Document], threshold: float = NEAR_DUPLICATE) -> list[Pair]:
    """Flag later copies in place (the first one seen is kept) and return all pair scores."""
    kept: list[Document] = []
    first_by_hash: dict[str, Document] = {}
    shingle_sets: dict[str, set[str]] = {}
    pairs: list[Pair] = []

    for doc in (d for d in docs if d.content_hash):
        original = first_by_hash.get(doc.content_hash)
        if original:
            doc.duplicate = Duplicate(of=original.document_id, kind="exact", similarity=1.0)
            continue
        shingle_sets[doc.document_id] = shingles(doc.cleaned_text)
        best: tuple[Document, float] | None = None
        for other in kept:
            similarity = jaccard(shingle_sets[doc.document_id], shingle_sets[other.document_id])
            pairs.append(Pair(other.document_id, doc.document_id, similarity))
            if similarity >= threshold and (best is None or similarity > best[1]):
                best = (other, similarity)
        if best:
            doc.duplicate = Duplicate(of=best[0].document_id, kind="near", similarity=round(best[1], 3))
        else:
            kept.append(doc)
        first_by_hash[doc.content_hash] = best[0] if best else doc
    return pairs
