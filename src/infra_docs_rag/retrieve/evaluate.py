"""The retrieval benchmark: the labelled questions in `eval/retrieval.yaml` against the index `embed` built.

Every in-scope query names the passages that answer it, the way the chunking benchmark's do, and the
out-of-scope ones have no label: the right result for them is no passage at all. One full ranking per
query gives every cutoff k at once; the scores of right hits against the best hits of unanswerable
questions give the threshold; the metadata queries carry the filter their wording implies, applied
before and after ranking; and timing exact search over synthetic vectors shows where a flat index
stops being enough.
"""

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel, model_validator

from ..chunk.chunks import DEFAULT_CHUNKING, usable
from ..chunk.evaluate import Answer, K, resolve_answer
from ..embed.embedders import Embedder
from ..embed.evaluate import Target, machine
from ..embed.index import Hit, Index, Manifest, unit
from ..ingest.models import Document
from .retriever import DEFAULT_RETRIEVAL, Context, Filter, Retrieval, Retriever, mask

Kind = Literal["paraphrase", "identifier", "metadata", "out-of-scope"]
KINDS: tuple[Kind, ...] = ("paraphrase", "identifier", "metadata", "out-of-scope")
K_GRID = (1, 3, K, 8, 10)
THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
SCALE = (1_000, 10_000, 100_000, 1_000_000)  # synthetic index sizes for timing exact search


class BenchmarkQuery(BaseModel):
    id: str
    kind: Kind
    text: str
    expect: Target | None = None  # the passages that answer it; None only when nothing in the corpus does
    filter: Filter | None = (
        None  # what the wording implies about where the answer is; metadata queries carry one
    )
    hypothesis: str  # written before any scores were computed

    @model_validator(mode="after")
    def labelled(self) -> "BenchmarkQuery":
        if (self.kind == "out-of-scope") != (self.expect is None):
            raise ValueError(f"{self.id}: an out-of-scope query has no `expect`; every other kind needs one")
        if self.kind == "metadata" and (self.filter is None or self.filter.empty):
            raise ValueError(f"{self.id}: a metadata query names the filter its wording implies")
        return self


class BenchmarkSet(BaseModel):
    queries: list[BenchmarkQuery]

    @property
    def in_scope(self) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.kind != "out-of-scope"]

    @property
    def out_of_scope(self) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.kind == "out-of-scope"]


def load_benchmark(path: Path) -> BenchmarkSet:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    bench = BenchmarkSet.model_validate(yaml.safe_load(path.read_text()))
    ids = [q.id for q in bench.queries]
    if len(ids) != len(set(ids)):
        raise ValueError("query ids must be unique")
    return bench


@dataclass
class Ranked:
    """One in-scope query against every chunk, best first."""

    query: BenchmarkQuery
    answer: Answer
    hits: list[Hit]
    right: list[bool]  # per hit, whether it answers the query
    tokens: list[int]  # per hit, tokens in its text

    @property
    def rank(self) -> int | None:
        """Where the first right chunk lands; None when no chunk answers at all."""
        return next((i + 1 for i, r in enumerate(self.right) if r), None)

    @property
    def right_score(self) -> float | None:
        return None if self.rank is None else self.hits[self.rank - 1].score

    @property
    def top_score(self) -> float:
        return self.hits[0].score

    def read(self, k: int) -> int:
        """Tokens in the top k."""
        return sum(self.tokens[:k])

    def shown(self, k: int, min_score: float | None) -> int:
        """Hits in the top k that pass the threshold."""
        return sum(min_score is None or h.score >= min_score for h in self.hits[:k])


@dataclass
class Unanswerable:
    """An out-of-scope query's closest chunks: every one of them is wrong."""

    query: BenchmarkQuery
    hits: list[Hit]

    @property
    def top_score(self) -> float:
        return self.hits[0].score

    def shown(self, k: int, min_score: float | None) -> int:
        return sum(min_score is None or h.score >= min_score for h in self.hits[:k])


@dataclass
class ThresholdRow:
    """What one threshold does at k = K."""

    min_score: float | None
    answered: int  # in-scope queries whose first right chunk is in the top K and passes
    silent: int  # in-scope queries left with no passage at all
    rejected: int  # out-of-scope queries left with no passage, as they should be
    shown: float  # median passages shown per query, over every query


def threshold_row(
    ranked: list[Ranked], unanswerable: list[Unanswerable], min_score: float | None
) -> ThresholdRow:
    def passes(score: float | None) -> bool:
        return score is not None and (min_score is None or score >= min_score)

    answered = sum(r.rank is not None and r.rank <= K and passes(r.right_score) for r in ranked)
    silent = sum(not passes(r.top_score) for r in ranked)
    rejected = sum(not passes(u.top_score) for u in unanswerable)
    shown = statistics.median([x.shown(K, min_score) for x in [*ranked, *unanswerable]])
    return ThresholdRow(min_score, answered, silent, rejected, shown)


def pick_threshold(rows: list[ThresholdRow]) -> ThresholdRow:
    """Most answers kept plus unanswerable questions rejected; ties go to fewer real questions left
    silent, then to fewer passages shown."""
    return max(rows, key=lambda r: (r.answered + r.rejected, -r.silent, -r.shown))


@dataclass
class Filtered:
    """A metadata query with the filter its wording implies, before and after ranking."""

    query: BenchmarkQuery
    candidates: int  # chunks the filter lets through
    unfiltered: int | None  # rank of the first right chunk among every chunk
    prefiltered: int | None  # rank among the eligible chunks
    postfiltered: int | None  # position among the top K that survive the filter; None when none does
    top: list[Hit]  # the pre-filtered top 3


@dataclass
class ScalePoint:
    chunks: int
    score_ms: float  # scoring every vector and sorting, median of a few runs
    megabytes: float  # the vectors alone


def measure_scale(
    dimensions: int, k: int, sizes: tuple[int, ...] = SCALE, repeats: int = 5
) -> list[ScalePoint]:
    """Time exact search over random unit vectors: what the flat index costs as the corpus grows."""
    rng = np.random.default_rng(0)
    points: list[ScalePoint] = []
    for n in sizes:
        vectors = rng.standard_normal((n, dimensions), dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        query = unit(rng.standard_normal(dimensions, dtype=np.float32))
        times = []
        for _ in range(repeats):
            started = time.perf_counter()
            scores = vectors @ query
            np.argsort(-scores, kind="stable")[:k]
            times.append((time.perf_counter() - started) * 1000)
        points.append(ScalePoint(n, statistics.median(times), vectors.nbytes / 1e6))
        del vectors
    return points


@dataclass
class Evaluation:
    benchmark: BenchmarkSet
    earlier: list[str]  # ids of the queries carried over from the embedding stage's set
    manifest: Manifest
    machine: str
    answers: dict[str, Answer]  # per in-scope query id
    ranked: list[Ranked]
    unanswerable: list[Unanswerable]
    thresholds: list[ThresholdRow]  # no threshold first, then THRESHOLDS
    pick: Retrieval  # top K with the threshold the benchmark picks
    filtered: list[Filtered]
    scale: list[ScalePoint]
    embed_ms: float  # embedding one query, median
    default: Retrieval  # DEFAULT_RETRIEVAL, what `retrieve` does
    plain: list[Context]  # the default's answer to every query, in file order, no filter
    with_filters: list[Context]  # the default's answer to each query that carries a filter, filter on

    def hits(self, k: int, only: list[Ranked] | None = None) -> int:
        """In-scope queries whose first right chunk is in the top k."""
        return sum(r.rank is not None and r.rank <= k for r in (self.ranked if only is None else only))

    @property
    def mrr(self) -> float:
        return statistics.fmean(1 / r.rank if r.rank else 0.0 for r in self.ranked)

    def read(self, k: int) -> float:
        """Median tokens in the top k."""
        return statistics.median(r.read(k) for r in self.ranked)

    def of_kind(self, kind: str) -> list[Ranked]:
        return [r for r in self.ranked if r.query.kind == kind]

    def answered(self, query: BenchmarkQuery, context: Context) -> bool:
        """Whether a context for an in-scope query holds a chunk that answers it."""
        answer = self.answers.get(query.id)
        return answer is not None and any(answer.answered_by(s.chunk) for s in context.sources)


def evaluate_retrieval(
    index: Index,
    embedder: Embedder,
    docs: list[Document],
    bench: BenchmarkSet,
    earlier_ids: set[str] = frozenset(),
    sizes: tuple[int, ...] = SCALE,
) -> Evaluation:
    if index.manifest.chunking != DEFAULT_CHUNKING:
        raise ValueError(
            f"the index holds `{index.manifest.chunking.label}` chunks, not the default "
            f"`{DEFAULT_CHUNKING.label}`; run `infra-docs-rag embed` first"
        )
    by_id = {d.source_id: d for d in usable(docs)}
    answers = {q.id: resolve_answer(q.expect, by_id, f"query {q.id}") for q in bench.in_scope if q.expect}
    chunks = index.chunks
    tokens = dict(
        zip((c.chunk_id for c in chunks), embedder.token_counts([c.text for c in chunks]), strict=True)
    )
    vectors = dict(
        zip(
            (q.id for q in bench.queries),
            embedder.embed_queries([q.text for q in bench.queries]),
            strict=True,
        )
    )
    everything = len(chunks)

    ranked: list[Ranked] = []
    for query in bench.in_scope:
        answer = answers[query.id]
        hits = index.rank(vectors[query.id], everything)
        ranked.append(
            Ranked(
                query,
                answer,
                hits,
                [answer.answered_by(h.chunk) for h in hits],
                [tokens[h.chunk.chunk_id] for h in hits],
            )
        )
    unanswerable = [Unanswerable(q, index.rank(vectors[q.id], max(K_GRID))) for q in bench.out_of_scope]
    print(
        f"{sum(r.rank == 1 for r in ranked)}/{len(ranked)} right chunk first, {sum(r.rank is not None and r.rank <= K for r in ranked)} in the top {K}"
    )

    rows = [
        threshold_row(ranked, unanswerable, None),
        *(threshold_row(ranked, unanswerable, t) for t in THRESHOLDS),
    ]
    picked = pick_threshold(rows)
    pick = Retrieval(k=K, min_score=picked.min_score)

    filtered: list[Filtered] = []
    for r in ranked:
        if r.query.filter is None:
            continue
        eligible = mask(chunks, r.query.filter)
        pre = index.rank(vectors[r.query.id], everything, eligible)
        survivors = [h for h in r.hits[:K] if r.query.filter.matches(h.chunk)]
        filtered.append(
            Filtered(
                query=r.query,
                candidates=int(eligible.sum()),
                unfiltered=r.rank,
                prefiltered=next((h.rank for h in pre if r.answer.answered_by(h.chunk)), None),
                postfiltered=next(
                    (i + 1 for i, h in enumerate(survivors) if r.answer.answered_by(h.chunk)), None
                ),
                top=pre[:3],
            )
        )

    embedder.embed_queries([bench.queries[0].text])  # warm-up
    timings = []
    for query in bench.in_scope:
        started = time.perf_counter()
        embedder.embed_queries([query.text])
        timings.append((time.perf_counter() - started) * 1000)

    retriever = Retriever(index, embedder)
    # This stage's benchmark is about the first stage: what `retrieve` does before any reranking.
    default = DEFAULT_RETRIEVAL.model_copy(update={"rerank": None})
    plain = [retriever.pack(q.text, vectors[q.id], default) for q in bench.queries]
    with_filters = [
        retriever.pack(q.text, vectors[q.id], default.model_copy(update={"filter": q.filter}))
        for q in bench.queries
        if q.filter is not None
    ]
    return Evaluation(
        benchmark=bench,
        earlier=[q.id for q in bench.queries if q.id in earlier_ids],
        manifest=index.manifest,
        machine=machine(),
        answers=answers,
        ranked=ranked,
        unanswerable=unanswerable,
        thresholds=rows,
        pick=pick,
        filtered=filtered,
        scale=measure_scale(index.manifest.dimensions, K, sizes),
        embed_ms=statistics.median(timings),
        default=default,
        plain=plain,
        with_filters=with_filters,
    )
