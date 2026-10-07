"""The reranking benchmark: the first stage's ranking against the same ranking reordered by each
cross-encoder over 20 and 50 candidates, for every labelled question, with what each step costs."""

import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from ..chunk.chunks import DEFAULT_CHUNKING
from ..chunk.evaluate import Answer, K, resolve_answer, usable
from ..embed.embedders import Embedder
from ..embed.evaluate import machine
from ..embed.index import Hit, Index, Manifest
from ..ingest.models import Document
from .evaluate import BenchmarkQuery, BenchmarkSet
from .rerank import RERANKERS, Reranker, RerankerSpec, rerank
from .retriever import DEFAULT_RETRIEVAL, Rerank, Retrieval, Retriever, gate

# The first stage every configuration starts from: what `retrieve` does before any reranking.
FIRST_STAGE = DEFAULT_RETRIEVAL.model_copy(update={"rerank": None, "min_score": None, "budget": None})
DEPTHS = (20, 50)  # candidates handed to the reranker
# The most a question may take, first stage and rerank together, for a configuration to be the default:
# a docs assistant answers while someone waits, and the generator's own time comes on top.
LATENCY_BUDGET_MS = 2000


def configurations(first: Retrieval = FIRST_STAGE, depths: tuple[int, ...] = DEPTHS) -> tuple[Retrieval, ...]:
    """The first stage alone, then each reranker over each depth."""
    return (
        first,
        *(
            first.model_copy(update={"rerank": Rerank(model=name, candidates=depth)})
            for name in RERANKERS
            for depth in depths
        ),
    )


def name(retrieval: Retrieval) -> str:
    """`first stage`, `minilm-l6 over 20`."""
    if retrieval.rerank is None:
        return "first stage"
    return f"{retrieval.rerank.model} over {retrieval.rerank.candidates}"


class Expectation(BaseModel):
    id: str
    moves: bool  # whether the right chunk is expected to rank higher after reranking
    hypothesis: str  # written before any reranker was run


class Expectations(BaseModel):
    expectations: list[Expectation] = []


def load_expectations(path: Path, queries: list[BenchmarkQuery]) -> list[Expectation]:
    if not path.exists():
        return []
    try:
        loaded = Expectations.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    known = {q.id for q in queries}
    for e in loaded.expectations:
        if e.id not in known:
            raise ValueError(f"{path}: {e.id} is not a benchmark query")
    return loaded.expectations


@dataclass
class Outcome:
    """One configuration's answer to one query."""

    rank: int | None  # of the first right chunk; None when none answers, or none within the candidates
    top: list[Hit]  # the top K, what a generator would read
    right: Hit | None  # the first right hit
    first_ms: float  # the first stage's time
    rerank_ms: float  # the reranker's time; 0 without one
    scored: int  # pairs the reranker read; 0 without one

    @property
    def ms(self) -> float:
        return self.first_ms + self.rerank_ms


@dataclass
class Run:
    """One configuration against every query."""

    retrieval: Retrieval
    reranker: RerankerSpec | None
    outcomes: dict[str, Outcome]

    def rank(self, query_id: str) -> int | None:
        return self.outcomes[query_id].rank

    def hits(self, k: int, queries: list[BenchmarkQuery]) -> int:
        return sum(self.rank(q.id) is not None and self.rank(q.id) <= k for q in queries)

    def mrr(self, queries: list[BenchmarkQuery]) -> float:
        ranks = [self.rank(q.id) for q in queries]
        return statistics.fmean(1 / r if r else 0.0 for r in ranks) if ranks else 0.0

    def median(self, field: str, queries: list[BenchmarkQuery]) -> float:
        return statistics.median(getattr(self.outcomes[q.id], field) for q in queries) if queries else 0.0

    def p95(self, field: str, queries: list[BenchmarkQuery]) -> float:
        values = sorted(getattr(self.outcomes[q.id], field) for q in queries)
        return values[min(len(values) - 1, int(round(0.95 * (len(values) - 1))))] if values else 0.0


@dataclass
class RerankComparison:
    manifest: Manifest
    machine: str
    queries: list[BenchmarkQuery]
    answers: dict[str, Answer]
    runs: list[Run]  # the first stage first, then the reranked configurations
    expectations: list[Expectation]
    parameters: dict[str, int]  # of each reranker loaded, by name

    @property
    def in_scope(self) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.kind != "out-of-scope"]

    @property
    def out_of_scope(self) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.kind == "out-of-scope"]

    def of_kind(self, kind: str) -> list[BenchmarkQuery]:
        return [q for q in self.in_scope if q.kind == kind]

    @property
    def first(self) -> Run:
        return self.runs[0]

    @property
    def reranked(self) -> list[Run]:
        return self.runs[1:]

    def quality(self, run: Run) -> tuple[int, float, int]:
        """What a configuration is ranked by: right chunks first, then MRR, then the top K."""
        return (run.hits(1, self.in_scope), run.mrr(self.in_scope), run.hits(K, self.in_scope))

    def best(self) -> Run:
        """The configuration with the most right chunks first, whatever it costs."""
        return max(self.runs, key=self.quality)

    def pick(self, budget_ms: float = LATENCY_BUDGET_MS) -> Run:
        """The best configuration whose median time per question fits the budget; the first stage always does."""
        within = [r for r in self.runs if r.median("ms", self.in_scope) <= budget_ms] or [self.first]
        return max(within, key=self.quality)

    def changed_set(self, run: Run) -> list[BenchmarkQuery]:
        """In-scope queries whose top K holds different chunks than the first stage's, order aside."""
        return [
            q
            for q in self.in_scope
            if {h.chunk.chunk_id for h in run.outcomes[q.id].top}
            != {h.chunk.chunk_id for h in self.first.outcomes[q.id].top}
        ]

    def changed(self, run: Run) -> list[BenchmarkQuery]:
        """In-scope queries whose top K differs from the first stage's, in content or order."""
        return [
            q
            for q in self.in_scope
            if [h.chunk.chunk_id for h in run.outcomes[q.id].top]
            != [h.chunk.chunk_id for h in self.first.outcomes[q.id].top]
        ]


def compare_rerank(
    index: Index,
    embedder: Embedder,
    docs: list[Document],
    bench: BenchmarkSet,
    extra: list[BenchmarkQuery],
    rerankers: dict[str, Reranker],
    expectations: list[Expectation] = (),
    configs: tuple[Retrieval, ...] | None = None,
) -> RerankComparison:
    if index.manifest.chunking != DEFAULT_CHUNKING:
        raise ValueError(
            f"the index holds `{index.manifest.chunking.label}` chunks, not the default "
            f"`{DEFAULT_CHUNKING.label}`; run `infra-docs-rag embed` first"
        )
    queries = [*bench.queries, *extra]
    by_id = {d.source_id: d for d in usable(docs)}
    answers = {q.id: resolve_answer(q.expect, by_id, f"query {q.id}") for q in queries if q.expect}
    vectors = dict(
        zip((q.id for q in queries), embedder.embed_queries([q.text for q in queries]), strict=True)
    )
    retriever = Retriever(index, embedder)
    everything = len(index.chunks)
    configs = configs or configurations()

    # The first stage once per query: every configuration starts from the same ranking.
    first_hits: dict[str, list[Hit]] = {}
    first_ms: dict[str, float] = {}
    retriever.first_stage(queries[0].text, vectors[queries[0].id], configs[0], everything)  # warm-up
    for q in queries:
        started = time.perf_counter()
        first_hits[q.id] = retriever.first_stage(q.text, vectors[q.id], configs[0], everything)
        first_ms[q.id] = (time.perf_counter() - started) * 1000

    def right_in(hits: list[Hit], q: BenchmarkQuery) -> Hit | None:
        answer = answers.get(q.id)
        return next((h for h in hits if answer is not None and answer.answered_by(h.chunk)), None)

    runs: list[Run] = []
    for retrieval in configs:
        outcomes: dict[str, Outcome] = {}
        if retrieval.rerank is None:
            for q in queries:
                hits = first_hits[q.id]
                right = right_in(hits, q)
                outcomes[q.id] = Outcome(
                    rank=right.rank if right else None,
                    top=hits[:K],
                    right=right,
                    first_ms=first_ms[q.id],
                    rerank_ms=0.0,
                    scored=0,
                )
            runs.append(Run(retrieval, None, outcomes))
        else:
            reranker = rerankers[retrieval.rerank.model]
            depth = retrieval.rerank.candidates
            rerank(queries[0].text, first_hits[queries[0].id][:depth], reranker, depth)  # warm-up
            for q in queries:
                result = rerank(q.text, first_hits[q.id][:depth], reranker, depth)
                right = right_in(result.hits, q)
                outcomes[q.id] = Outcome(
                    rank=right.rank if right else None,
                    top=result.hits[:K],
                    right=right,
                    first_ms=first_ms[q.id],
                    rerank_ms=result.ms,
                    scored=result.scored,
                )
            runs.append(Run(retrieval, RERANKERS[retrieval.rerank.model], outcomes))
        found = sum(o.rank == 1 for o in outcomes.values())
        print(f"{name(retrieval)}: right chunk first for {found}/{len(answers)}")
    return RerankComparison(
        manifest=index.manifest,
        machine=machine(),
        queries=queries,
        answers=answers,
        runs=runs,
        expectations=list(expectations),
        parameters={name: r.parameters for name, r in rerankers.items()},
    )


def top_gate(outcome: Outcome) -> float | None:
    """The top hit's cosine score, what the threshold sees; None when nothing ranked."""
    return gate(outcome.top[0]) if outcome.top else None
