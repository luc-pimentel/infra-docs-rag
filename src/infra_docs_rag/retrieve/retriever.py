"""The retriever: a question in, the passages a generator should read out, each with where it came from.

Retrieval is three decisions on top of the index's ranking. Which chunks are eligible: a `Filter` on
the metadata every chunk carries (project, source, section, page), applied before ranking so the top k
is the top k of what qualifies, not what is left of the top k after dropping the rest. How many: `k`,
the chunks a generator reads. How similar is similar enough: `min_score`, under which a hit is dropped
rather than shown, so a question the documentation does not cover gets no passages instead of the
least wrong ones. What survives is packed into a `Context`: the passages in rank order, each under a
numbered heading that says where it comes from, so an answer can point back at [2].

Which ranking the decisions sit on is the `mode`: `dense` is the index's cosine ranking, `lexical` is
BM25 over the same chunks (`lexical.py`), and `hybrid` fuses the two (`fusion.py`), each hit keeping
both component scores. The threshold is a cosine floor wherever there is a cosine score: the hit's own
score in dense mode, its dense component in hybrid mode; lexical mode has none.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, model_validator

from ..chunk.chunks import Chunk, locate
from ..embed.embedders import Embedder
from ..embed.index import Hit, Index
from .fusion import Fusion, HybridHit, fuse
from .lexical import Lexical

Mode = Literal["dense", "lexical", "hybrid"]
MODES: tuple[Mode, ...] = ("dense", "lexical", "hybrid")


class Filter(BaseModel):
    """Metadata a chunk has to match. Every field that is set is required; a field left unset matches
    anything, so the empty filter matches every chunk."""

    project: str | None = None  # the documentation set, as in sources.yaml; case does not matter
    source: str | None = None  # a source id from sources.yaml
    section: str | None = None  # a heading, or the end of a heading path written "Parent › Child"
    page: int | None = None  # a PDF page the chunk touches

    @property
    def empty(self) -> bool:
        return self == Filter()

    def matches(self, chunk: Chunk) -> bool:
        if self.project is not None and (chunk.project or "").casefold() != self.project.casefold():
            return False
        if self.source is not None and chunk.source_id != self.source:
            return False
        if self.section is not None:
            wanted = [part.strip().casefold() for part in self.section.split("›")]
            if [part.casefold() for part in chunk.section_path[-len(wanted) :]] != wanted:
                return False
        if self.page is not None and self.page not in chunk.pages:
            return False
        return True

    @property
    def label(self) -> str:
        """`project = Argo CD · page = 9`."""
        return " · ".join(
            f"{name} = {value}" for name, value in self.model_dump().items() if value is not None
        )


def mask(chunks: list[Chunk], criteria: Filter) -> np.ndarray:
    """One boolean per chunk: whether it matches."""
    return np.fromiter((criteria.matches(c) for c in chunks), dtype=bool, count=len(chunks))


class Retrieval(BaseModel):
    """How a question is answered from the index: how many chunks, how similar they have to be, which
    ones are eligible, and how much a generator gets to read."""

    k: int = Field(default=5, ge=1)  # chunks ranked; what a generator reads when every one qualifies
    min_score: float | None = Field(default=None, ge=-1, le=1)  # cosine similarity a hit needs to be shown
    filter: Filter = Filter()
    prefilter: bool = True  # filter before ranking; False ranks everything, then drops what does not match
    budget: int | None = Field(default=None, ge=1)  # most tokens the passages may add up to
    mode: Mode = "dense"
    fusion: Fusion = Fusion()  # how hybrid mode merges the two rankings; ignored by the other modes

    @model_validator(mode="after")
    def thresholdable(self) -> "Retrieval":
        if self.mode == "lexical" and self.min_score is not None:
            raise ValueError("BM25 scores have no fixed scale, so lexical mode takes no min_score")
        return self

    @property
    def label(self) -> str:
        parts = [f"top {self.k}"]
        if self.mode != "dense":
            parts.append(self.mode + (f" ({self.fusion.label})" if self.mode == "hybrid" else ""))
        if self.min_score is not None:
            parts.append(f"min {self.min_score:.2f}")
        if not self.filter.empty:
            parts.append(self.filter.label + ("" if self.prefilter else ", after ranking"))
        if self.budget is not None:
            parts.append(f"{self.budget} tokens")
        return " · ".join(parts)


# What `search` and `retrieve` do unless told otherwise: the top 5, with a floor under the cosine score
# that only catches questions unrelated to the corpus. reports/retrieval.md shows how both were chosen.
DEFAULT_RETRIEVAL = Retrieval(k=5, min_score=0.5)


@dataclass
class Source:
    """One passage in a context, with the number the context cites it by."""

    n: int
    hit: Hit
    tokens: int  # of the passage's text, in the embedding model's tokens

    @property
    def chunk(self) -> Chunk:
        return self.hit.chunk

    @property
    def heading(self) -> str:
        """`[2] Kubernetes › Deployments › Rolling Back a Deployment`, with the pages for a PDF."""
        c = self.chunk
        return f"[{self.n}] " + locate([c.project, *c.section_path] if c.project else c.section_path, c.pages)


@dataclass
class Context:
    """What a generator reads for one question: the passages that qualified, in rank order, each under
    a numbered heading that says where it comes from."""

    query: str
    retrieval: Retrieval
    sources: list[Source]
    candidates: int  # chunks eligible under the filter
    top_score: float | None  # the best hit's score before the threshold; None when nothing was eligible
    below: int  # hits dropped for scoring under min_score
    over_budget: int  # hits left out because the budget was spent

    @property
    def text(self) -> str:
        return "\n\n".join(f"{s.heading}\n{s.chunk.text}" for s in self.sources)

    @property
    def tokens(self) -> int:
        return sum(s.tokens for s in self.sources)

    @property
    def empty(self) -> bool:
        return not self.sources


def gate(hit: Hit) -> float | None:
    """The score the threshold is checked against: the cosine similarity, wherever the hit has one."""
    if isinstance(hit, HybridHit):
        return hit.dense
    return hit.score


class Retriever:
    """One index and the model it was built with, plus a BM25 index over the same chunks, answering
    questions with `retrieve`."""

    def __init__(self, index: Index, embedder: Embedder) -> None:
        index.check(embedder)
        self.index = index
        self.embedder = embedder
        self.lexical = Lexical.build(index.chunks)

    def retrieve(self, query: str, retrieval: Retrieval = DEFAULT_RETRIEVAL) -> Context:
        return self.pack(query, self.embedder.embed_queries([query])[0], retrieval)

    def ranking(
        self, query: str, vector: np.ndarray, retrieval: Retrieval, k: int, within: np.ndarray | None = None
    ) -> list[Hit]:
        """The top k under the retrieval's mode, among the chunks `within` marks (every chunk when None)."""
        if retrieval.mode == "dense":
            return self.index.rank(vector, k, within)
        if retrieval.mode == "lexical":
            return self.lexical.rank(query, k, within)
        everything = len(self.index.chunks)
        dense = self.index.rank(vector, everything, within)
        lexical = self.lexical.rank(query, everything, within)
        return fuse(dense, lexical, retrieval.fusion, k)

    def hits(self, query: str, vector: np.ndarray, retrieval: Retrieval) -> tuple[list[Hit], int]:
        """The ranked hits that pass the filter, at most k of them, and how many chunks were eligible."""
        chunks = self.index.chunks
        if retrieval.filter.empty:
            return self.ranking(query, vector, retrieval, retrieval.k), len(chunks)
        eligible = mask(chunks, retrieval.filter)
        if retrieval.prefilter:
            return self.ranking(query, vector, retrieval, retrieval.k, eligible), int(eligible.sum())
        ranked = self.ranking(query, vector, retrieval, retrieval.k)
        return [h for h in ranked if retrieval.filter.matches(h.chunk)], int(eligible.sum())

    def pack(self, query: str, vector: np.ndarray, retrieval: Retrieval) -> Context:
        """Rank, drop what scores under the threshold, and pack what is left up to the budget. The first
        passage always goes in: a generator with nothing to read has nothing to cite."""
        hits, candidates = self.hits(query, vector, retrieval)
        floor = retrieval.min_score
        kept = [h for h in hits if floor is None or gate(h) >= floor]
        tokens = self.embedder.token_counts([h.chunk.text for h in kept]) if kept else []
        sources: list[Source] = []
        spent = 0
        for hit, count in zip(kept, tokens, strict=True):
            if retrieval.budget is not None and sources and spent + count > retrieval.budget:
                break
            sources.append(Source(len(sources) + 1, hit, count))
            spent += count
        return Context(
            query=query,
            retrieval=retrieval,
            sources=sources,
            candidates=candidates,
            top_score=gate(hits[0]) if hits else None,
            below=len(hits) - len(kept),
            over_budget=len(kept) - len(sources),
        )
