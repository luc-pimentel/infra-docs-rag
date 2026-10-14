"""The benchmark workload under a configuration, with every stage timed and the retrieval quality read
off the labels: no model call, so a configuration can be measured in two minutes on a laptop, and the
generation side is estimated from the passages the configuration would send and what the saved runs say
a call costs."""

import statistics
import time
from dataclasses import dataclass, field

from ..chunk.chunks import usable
from ..chunk.evaluate import Answer as Labelled
from ..chunk.evaluate import resolve_answer
from ..embed.embedders import Embedder
from ..embed.index import Index
from ..generate.evaluate import GroundingQuery
from ..generate.generator import MODELS
from ..generate.prompt import SYSTEM
from ..ingest.models import Document
from ..retrieve.evaluate import BenchmarkQuery, BenchmarkSet
from ..retrieve.rerank import Reranker
from ..retrieve.retriever import Retrieval, Retriever

STAGES = ("embed", "dense", "lexical", "fuse", "rerank", "pack")
QUESTION_TOKENS = 25  # about what a benchmark question costs in input tokens
SYSTEM_TOKENS = (
    len(SYSTEM.split()) * 4 // 3
)  # the frozen system prompt, read from the cache after the first call


def p50(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * 0.95
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class Question:
    id: str
    kind: str  # paraphrase, identifier, metadata, out-of-scope, adversarial
    text: str
    labelled: Labelled | None  # where the right answer sits, for a labelled question
    expect: str  # answer, abstain, partial
    retrieval_filter: Retrieval | None = None  # a per-question filter, when the benchmark gives one


@dataclass
class Observation:
    question: Question
    timings: dict[str, float]
    retrieve_ms: float
    passages: int
    passage_tokens: int
    right_rank: int | None  # among the passages read; None when no right one, or unlabelled
    no_passage: bool
    degraded: str | None


@dataclass
class Profile:
    name: str
    retrieval: Retrieval
    observations: list[Observation]
    note: str = ""

    @property
    def labelled(self) -> list[Observation]:
        return [o for o in self.observations if o.question.labelled is not None]

    @property
    def in_scope(self) -> list[Observation]:
        return [o for o in self.observations if o.question.kind not in ("out-of-scope", "adversarial")]

    @property
    def out_of_scope(self) -> list[Observation]:
        return [o for o in self.observations if o.question.kind == "out-of-scope"]

    def stage_p50(self, stage: str) -> float:
        return p50([o.timings[stage] for o in self.observations if stage in o.timings])

    def stage_p95(self, stage: str) -> float:
        return p95([o.timings[stage] for o in self.observations if stage in o.timings])

    @property
    def retrieve_p50(self) -> float:
        return p50([o.retrieve_ms for o in self.observations])

    @property
    def retrieve_p95(self) -> float:
        return p95([o.retrieve_ms for o in self.observations])

    @property
    def right_retrieved(self) -> int:
        return sum(o.right_rank is not None for o in self.labelled)

    @property
    def hit_at_1(self) -> int:
        return sum(o.right_rank == 1 for o in self.labelled)

    @property
    def mrr(self) -> float:
        ranks = [1 / o.right_rank if o.right_rank else 0.0 for o in self.labelled]
        return statistics.mean(ranks) if ranks else 0.0

    @property
    def false_abstentions(self) -> int:
        """In-scope questions the floor answered with nothing."""
        return sum(o.no_passage for o in self.in_scope)

    @property
    def oos_abstentions(self) -> int:
        return sum(o.no_passage for o in self.out_of_scope)

    @property
    def passages_p50(self) -> float:
        return p50([o.passages for o in self.observations if not o.no_passage])

    @property
    def passage_tokens_p50(self) -> float:
        return p50([o.passage_tokens for o in self.observations if not o.no_passage])

    @property
    def calls(self) -> int:
        """Questions that would reach the model: those with a passage."""
        return sum(not o.no_passage for o in self.observations)

    def generation_cost(
        self, output_tokens: float, prices: dict[str, float] = MODELS["claude-opus-5-5"]
    ) -> float:
        """Estimated cost per question at API list price: the passages and question as fresh input, the
        system prompt as a cache read, `output_tokens` out; a question with no passage costs nothing."""
        per_call = [
            (o.passage_tokens + QUESTION_TOKENS) * prices["input"] / 1e6
            + SYSTEM_TOKENS * prices["cache_read"] / 1e6
            + output_tokens * prices["output"] / 1e6
            for o in self.observations
            if not o.no_passage
        ]
        return sum(per_call) / len(self.observations) if self.observations else 0.0


def questions_of(
    docs: list[Document], bench: BenchmarkSet, extra: list[BenchmarkQuery], adversarial: list[GroundingQuery]
) -> list[Question]:
    by_id = {d.source_id: d for d in usable(docs)}
    out = []
    for q in [*bench.queries, *extra]:
        labelled = resolve_answer(q.expect, by_id, f"query {q.id}") if q.expect else None
        out.append(
            Question(
                q.id,
                q.kind,
                q.text,
                labelled,
                "abstain" if q.kind == "out-of-scope" else "answer",
                Retrieval(filter=q.filter) if q.filter else None,
            )
        )
    out += [Question(q.id, q.kind, q.text, None, q.expect) for q in adversarial]
    return out


def observe(retriever: Retriever, q: Question, retrieval: Retrieval) -> Observation:
    this = (
        retrieval.model_copy(update={"filter": q.retrieval_filter.filter})
        if q.retrieval_filter
        else retrieval
    )
    started = time.perf_counter()
    ctx = retriever.retrieve(q.text, this)
    ms = (time.perf_counter() - started) * 1000
    right = None
    if q.labelled is not None:
        right = next((s.n for s in ctx.sources if q.labelled.answered_by(s.chunk)), None)
    return Observation(
        question=q,
        timings=dict(ctx.timings),
        retrieve_ms=ms,
        passages=len(ctx.sources),
        passage_tokens=ctx.tokens,
        right_rank=right,
        no_passage=ctx.empty,
        degraded=ctx.degraded,
    )


def profile(
    name: str,
    retrieval: Retrieval,
    index: Index,
    embedder: Embedder,
    questions: list[Question],
    reranker: Reranker | None,
    note: str = "",
) -> Profile:
    retriever = Retriever(index, embedder, reranker)
    retriever.retrieve(questions[0].text, retrieval)  # warm-up: the first call pays for lazy model setup
    observations = [observe(retriever, q, retrieval) for q in questions]
    return Profile(name, retrieval, observations, note)


@dataclass
class BatchTiming:
    batch_size: int
    seconds: float
    chunks: int


def time_batches(embedder, texts: list[str], sizes: tuple[int, ...] = (8, 16, 64)) -> list[BatchTiming]:
    """How long the corpus takes to embed at each batch size, same model, same texts."""
    out = []
    for size in sizes:
        embedder.batch_size = size
        started = time.perf_counter()
        embedder.embed_documents(texts)
        out.append(BatchTiming(size, time.perf_counter() - started, len(texts)))
    embedder.batch_size = 16
    return out


@dataclass
class Workload:
    """What the latency report measures: the profiles, the batch sweep, and the saved runs it reads the
    generation side from."""

    profiles: list[Profile]
    batches: list[BatchTiming] = field(default_factory=list)
    machine: str = ""
