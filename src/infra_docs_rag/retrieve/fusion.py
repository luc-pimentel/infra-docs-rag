"""Fusion: one ranked list out of a dense ranking and a lexical one, each hit keeping both scores.

Cosine similarity lives in [-1, 1] and BM25 in [0, ∞), so the two cannot be added as they are. Two
ways around it, both standard:

- Reciprocal rank fusion (`rrf`) ignores the scores and adds `1 / (rrf_k + rank)` for each list a chunk
  appears in, within the top `depth`. It is parameter-light and robust, and a chunk both rankings like
  wins even if neither put it first.
- Weighted score fusion (`weighted`) rescales each list's top-`depth` scores to [0, 1] by min-max and
  adds them with `alpha` on the dense side and `1 - alpha` on the lexical one. It keeps the margins the
  rankings expressed, so a chunk one side is sure about can carry the fused rank; it also inherits each
  side's scale quirks, which is what the experiment in `evaluate-hybrid` is about.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from ..embed.index import Hit

Strategy = Literal["rrf", "weighted"]
STRATEGIES: tuple[Strategy, ...] = ("rrf", "weighted")


class Fusion(BaseModel):
    """How the two rankings are merged."""

    strategy: Strategy = "rrf"
    depth: int = Field(default=50, ge=1)  # candidates taken from the top of each ranking
    rrf_k: int = Field(default=60, ge=0)  # rrf: the constant that softens the gap between rank 1 and rank 2
    alpha: float = Field(default=0.5, ge=0, le=1)  # weighted: share of the fused score the dense side gets

    @property
    def label(self) -> str:
        if self.strategy == "rrf":
            return f"rrf k={self.rrf_k} · depth {self.depth}"
        return f"weighted α={self.alpha:.1f} · depth {self.depth}"


@dataclass
class HybridHit(Hit):
    """A fused hit: its fused `score`, and what each side said about it."""

    dense: float  # cosine similarity
    dense_rank: int
    lexical: float  # BM25; 0 when the chunk shares no term with the query
    lexical_rank: int | None  # None when BM25 had nothing to say


def normalized(hits: list[Hit], depth: int) -> dict[str, float]:
    """Each chunk's score rescaled to [0, 1] over the top `depth`: the best hit is 1, the hit at `depth`
    is 0, anything deeper is clamped to 0. A list of one score, or none, gives every hit 1 or nothing."""
    top = hits[:depth]
    if not top:
        return {}
    high, low = top[0].score, top[-1].score
    if high == low:
        return {h.chunk.chunk_id: 1.0 for h in top}
    return {h.chunk.chunk_id: max(0.0, (h.score - low) / (high - low)) for h in hits}


def fuse(dense: list[Hit], lexical: list[Hit], fusion: Fusion, k: int) -> list[HybridHit]:
    """Merge two rankings of the same candidates into the top k. `dense` is a full ranking (every eligible
    chunk, best first) so every hit has a dense score; `lexical` holds only the chunks BM25 scored."""
    by_dense = {h.chunk.chunk_id: h for h in dense}
    by_lexical = {h.chunk.chunk_id: h for h in lexical}
    candidates = dict.fromkeys(
        [h.chunk.chunk_id for h in dense[: fusion.depth]]
        + [h.chunk.chunk_id for h in lexical[: fusion.depth]]
    )
    if fusion.strategy == "weighted":
        dense_norm, lexical_norm = normalized(dense, fusion.depth), normalized(lexical, fusion.depth)

    fused: list[HybridHit] = []
    for chunk_id in candidates:
        d, lex = by_dense[chunk_id], by_lexical.get(chunk_id)
        if fusion.strategy == "rrf":
            score = (1 / (fusion.rrf_k + d.rank) if d.rank <= fusion.depth else 0.0) + (
                1 / (fusion.rrf_k + lex.rank) if lex is not None and lex.rank <= fusion.depth else 0.0
            )
        else:
            score = fusion.alpha * dense_norm.get(chunk_id, 0.0) + (1 - fusion.alpha) * lexical_norm.get(
                chunk_id, 0.0
            )
        fused.append(
            HybridHit(
                rank=0,
                score=score,
                chunk=d.chunk,
                dense=d.score,
                dense_rank=d.rank,
                lexical=lex.score if lex is not None else 0.0,
                lexical_rank=lex.rank if lex is not None else None,
            )
        )
    fused.sort(key=lambda h: (-h.score, h.dense_rank))
    for n, hit in enumerate(fused[:k]):
        hit.rank = n + 1
    return fused[:k]
