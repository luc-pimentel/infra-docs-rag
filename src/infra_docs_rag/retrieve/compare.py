"""The hybrid search experiment: the same labelled questions under dense, lexical and fused rankings.

One full ranking per question per configuration, as in the retrieval benchmark, gives the rank of the
first right chunk for every cutoff at once. Side by side, the configurations answer the three questions
this stage asks: where does keyword search beat embeddings, does fusing the two keep the best of each,
and which fusion does it best. The out-of-scope questions ride along: BM25 scores a question that
shares no term with the corpus at 0, which is a cleaner silence than the cosine floor can give, and
the report shows whether the corpus' own near-miss questions get it.
"""

import statistics
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from ..chunk.chunks import DEFAULT_CHUNKING, usable
from ..chunk.evaluate import Answer, K, resolve_answer
from ..embed.embedders import Embedder
from ..embed.evaluate import machine
from ..embed.index import Hit, Index, Manifest
from ..ingest.models import Document
from .evaluate import BenchmarkQuery, BenchmarkSet, load_benchmark
from .fusion import Fusion
from .retriever import Retrieval, Retriever, gate

# The configurations compared, in report order. Dense is the retrieval stage's ranking and the baseline;
# lexical is BM25 alone; the fusions are what hybrid mode can run with.
CONFIGURATIONS: tuple[Retrieval, ...] = (
    Retrieval(mode="dense"),
    Retrieval(mode="lexical"),
    Retrieval(mode="hybrid", fusion=Fusion(strategy="rrf")),
    Retrieval(mode="hybrid", fusion=Fusion(strategy="weighted", alpha=0.3)),
    Retrieval(mode="hybrid", fusion=Fusion(strategy="weighted", alpha=0.5)),
    Retrieval(mode="hybrid", fusion=Fusion(strategy="weighted", alpha=0.7)),
)


def name(retrieval: Retrieval) -> str:
    """`dense`, `lexical`, `hybrid rrf k=60`, `hybrid weighted α=0.3`."""
    if retrieval.mode != "hybrid":
        return retrieval.mode
    f = retrieval.fusion
    return "hybrid " + (f"rrf k={f.rrf_k}" if f.strategy == "rrf" else f"weighted α={f.alpha:.1f}")


def load_extra(path: Path, base: BenchmarkSet) -> list[BenchmarkQuery]:
    """The stage's own queries, appended to the retrieval benchmark; none when the file has no entries."""
    if not path.exists():
        return []
    try:
        extra = BenchmarkSet.model_validate(yaml.safe_load(path.read_text()) or {"queries": []})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    taken = {q.id for q in base.queries}
    for q in extra.queries:
        if q.id in taken:
            raise ValueError(f"{path}: query id {q.id} is already in the retrieval benchmark")
        taken.add(q.id)
    return extra.queries


@dataclass
class Outcome:
    """One configuration's answer to one query: where the first right chunk landed, or the top hit of
    an unanswerable question."""

    rank: int | None  # of the first right chunk; None when none answers (always None out of scope)
    top: Hit | None  # the first hit, components included when the mode is hybrid; None when nothing ranked
    right: Hit | None  # the first right hit

    @property
    def top_gate(self) -> float | None:
        """The top hit's cosine score, what the threshold would see; None when nothing ranked."""
        return None if self.top is None else gate(self.top)


@dataclass
class Run:
    """One configuration against every query."""

    retrieval: Retrieval
    outcomes: dict[str, Outcome]  # by query id

    def rank(self, query_id: str) -> int | None:
        return self.outcomes[query_id].rank

    def hits(self, k: int, queries: list[BenchmarkQuery]) -> int:
        return sum(self.rank(q.id) is not None and self.rank(q.id) <= k for q in queries)

    def mrr(self, queries: list[BenchmarkQuery]) -> float:
        ranks = [self.rank(q.id) for q in queries]
        return statistics.fmean(1 / r if r else 0.0 for r in ranks) if ranks else 0.0


@dataclass
class Comparison:
    manifest: Manifest
    machine: str
    queries: list[BenchmarkQuery]  # in file order: the retrieval benchmark, then the stage's own
    extra: list[str]  # ids of the stage's own queries
    answers: dict[str, Answer]
    runs: list[Run]  # in CONFIGURATIONS order
    lexical_terms: int  # vocabulary size of the BM25 index

    @property
    def in_scope(self) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.kind != "out-of-scope"]

    @property
    def out_of_scope(self) -> list[BenchmarkQuery]:
        return [q for q in self.queries if q.kind == "out-of-scope"]

    def of_kind(self, kind: str) -> list[BenchmarkQuery]:
        return [q for q in self.in_scope if q.kind == kind]

    def run(self, mode: str) -> Run:
        """The first run of a mode: `dense`, `lexical`, or the first hybrid."""
        return next(r for r in self.runs if r.retrieval.mode == mode)

    @property
    def dense(self) -> Run:
        return self.run("dense")

    @property
    def lexical(self) -> Run:
        return self.run("lexical")

    @property
    def hybrids(self) -> list[Run]:
        return [r for r in self.runs if r.retrieval.mode == "hybrid"]

    def best_hybrid(self) -> Run:
        """The fusion with the most answers in the top K over the in-scope queries, MRR as tiebreak."""
        return max(self.hybrids, key=lambda r: (r.hits(K, self.in_scope), r.mrr(self.in_scope)))

    def dense_loses(self) -> list[BenchmarkQuery]:
        """In-scope queries where lexical ranks the right chunk above dense, or dense finds none."""
        out = []
        for q in self.in_scope:
            d, lex = self.dense.rank(q.id), self.lexical.rank(q.id)
            if lex is not None and (d is None or lex < d):
                out.append(q)
        return out


def compare(
    index: Index,
    embedder: Embedder,
    docs: list[Document],
    bench: BenchmarkSet,
    extra: list[BenchmarkQuery],
    configurations: tuple[Retrieval, ...] = CONFIGURATIONS,
) -> Comparison:
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

    runs: list[Run] = []
    for retrieval in configurations:
        outcomes: dict[str, Outcome] = {}
        for q in queries:
            hits = retriever.ranking(q.text, vectors[q.id], retrieval, everything)
            answer = answers.get(q.id)
            right = next((h for h in hits if answer is not None and answer.answered_by(h.chunk)), None)
            outcomes[q.id] = Outcome(
                rank=right.rank if right else None, top=hits[0] if hits else None, right=right
            )
        runs.append(Run(retrieval, outcomes))
        found = sum(o.rank == 1 for o in outcomes.values())
        print(f"{name(retrieval)}: right chunk first for {found}/{len(answers)}")
    return Comparison(
        manifest=index.manifest,
        machine=machine(),
        queries=queries,
        extra=[q.id for q in extra],
        answers=answers,
        runs=runs,
        lexical_terms=retriever.lexical.vocabulary,
    )


def load_all(benchmark: Path, extra: Path) -> tuple[BenchmarkSet, list[BenchmarkQuery]]:
    bench = load_benchmark(benchmark)
    return bench, load_extra(extra, bench)
