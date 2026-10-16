"""A trace is one question through the pipeline, as a record: what was asked, under which pipeline
version (the lineage), what the first stage ranked and with which scores, what the reranker made of
it, what was packed and what was dropped on the way, what the model wrote and cited, what the judge and
the reader read, and how long each stage took. One JSON line per question in `data/traces/traces.jsonl`;
`answer --trace` prints the one it made, and a run writes one per case, tagged with the example it
answered, the run and the dataset version, and keeps them in `traces.jsonl` beside the run (stage 17).
`spans` reads the same trace as spans: retrieve, prompt, generate, verify, judge, read, each with its
time and the attributes a diagnosis needs."""

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ..generate.answer import Answer, marked
from ..generate.judge import Judgement
from ..pipeline.manifest import PipelineManifest
from ..retrieve.fusion import HybridHit
from ..retrieve.rerank import RerankedHit
from ..retrieve.retriever import gate, relevance

TRACES = Path("data/traces/traces.jsonl")


class TracedHit(BaseModel):
    rank: int
    chunk_id: str
    citation: str
    score: float  # the ranking's score at that stage
    cosine: float | None = None
    lexical: float | None = None
    relevance: float | None = None  # the reranker's, on a reranked hit
    first_rank: int | None = None  # where the first stage had it, on a reranked hit


class TracedCitation(BaseModel):
    n: int
    chunk_id: str
    span: str
    cited_text: str


class TracedDrop(BaseModel):
    """A ranked passage the pack step left out, and why."""

    rank: int
    chunk_id: str
    citation: str
    reason: str  # threshold | relevance | budget


class TracedReading(BaseModel):
    relevant: float
    reference_agrees: float
    deprecation_named: float
    readings_named: float
    model: str
    ms: float
    cost_usd: float


class Span(BaseModel):
    """One stage of the request as a span: a name, its time, and what a diagnosis reads off it."""

    name: str
    ms: float
    attributes: dict[str, str | int | float | bool | None] = {}


class TracedJudgement(BaseModel):
    grounded: float
    fabricated: float
    gap_named: float
    model: str
    ms: float
    cost_usd: float


class Trace(BaseModel):
    trace_id: str
    at: str
    question: str
    version: str
    retrieval: str  # the retrieval's label
    lineage: PipelineManifest | None = None
    prompt_version: str = ""
    stages: dict[str, float] = {}  # ms: embed, dense, lexical, fuse, rerank, pack, generate, judge
    total_ms: float = 0.0
    candidates: int = 0  # chunks eligible under the filter
    first_stage: list[TracedHit] = []  # what the first stage ranked, in its order
    context: list[TracedHit] = []  # what the model read, in rank order (reranked when there was one)
    top_score: float | None = None
    below: int = 0
    degraded: str | None = None
    outcome: str
    reason: str = ""
    answer: str = ""  # with [n] after each cited span
    citations: list[TracedCitation] = []
    cited_share: float = 0.0
    claim_share: float = 0.0
    uncited: list[str] = []
    generator: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    judgement: TracedJudgement | None = None
    error: str | None = None
    # Stage 17: the eval identifiers, the rest of the retrieval's decisions, and the reader.
    example_id: str | None = None  # the record or benchmark question the answer is for
    run: str | None = None  # the saved run under eval/runs/ this trace belongs to
    dataset_version: str | None = None  # `<records>+<hash>`, on a dataset run
    reranked: list[TracedHit] = []  # every candidate in the reranker's order, when there was one
    ranked: list[TracedHit] = []  # the top k the pack step read, before the floor and the budget
    dropped: list[TracedDrop] = []  # ranked passages the pack step left out, with the reason
    tokens: int | None = None  # of the passages packed, in the embedding model's tokens
    budget: int | None = None  # the retrieval's token budget
    over_budget: int = 0
    reading: TracedReading | None = None

    @property
    def answered(self) -> bool:
        return self.outcome == "answered"

    @property
    def retrieve_ms(self) -> float:
        return sum(v for k, v in self.stages.items() if k in RETRIEVAL_STAGES)

    @property
    def spans(self) -> list[Span]:
        """The trace as spans, in request order."""
        out = [
            Span(
                name="retrieve",
                ms=round(self.retrieve_ms, 1),
                attributes={
                    "candidates": self.candidates,
                    "first_stage": len(self.first_stage),
                    "reranked": len(self.reranked) or None,
                    "ranked": len(self.ranked) or None,
                    "packed": len(self.context),
                    "dropped_threshold": sum(d.reason in ("threshold", "relevance") for d in self.dropped),
                    "dropped_budget": sum(d.reason == "budget" for d in self.dropped),
                    "below": self.below,
                    "over_budget": self.over_budget,
                    "top_score": self.top_score,
                    "degraded": self.degraded,
                    "retrieval": self.retrieval,
                },
            ),
            Span(
                name="prompt",
                ms=0.0,
                attributes={
                    "passages": len(self.context),
                    "tokens": self.tokens,
                    "budget": self.budget,
                    "prompt_version": self.prompt_version or None,
                },
            ),
        ]
        if "generate" in self.stages:
            out.append(
                Span(
                    name="generate",
                    ms=self.stages["generate"],
                    attributes={
                        "model": self.generator or None,
                        "input_tokens": self.input_tokens,
                        "output_tokens": self.output_tokens,
                        "cost_usd": self.cost_usd,
                        "outcome": self.outcome,
                        "error": self.error,
                    },
                )
            )
        if self.answered:
            out.append(
                Span(
                    name="verify",
                    ms=0.0,
                    attributes={
                        "citations": len(self.citations),
                        "cited_chunks": len({c.chunk_id for c in self.citations}),
                        "cited_share": self.cited_share,
                        "claim_share": self.claim_share,
                        "uncited": len(self.uncited),
                    },
                )
            )
        if self.judgement is not None:
            j = self.judgement
            out.append(
                Span(
                    name="judge",
                    ms=j.ms,
                    attributes={
                        "model": j.model,
                        "grounded": j.grounded,
                        "fabricated": j.fabricated,
                        "gap_named": j.gap_named,
                    },
                )
            )
        if self.reading is not None:
            r = self.reading
            out.append(
                Span(
                    name="read",
                    ms=r.ms,
                    attributes={
                        "model": r.model,
                        "relevant": r.relevant,
                        "reference_agrees": r.reference_agrees,
                        "deprecation_named": r.deprecation_named,
                        "readings_named": r.readings_named,
                    },
                )
            )
        return out

    @property
    def tags(self) -> dict[str, str | None]:
        """The identifiers that link the trace to its eval example and experiment."""
        L = self.lineage
        return {
            "example_id": self.example_id,
            "run": self.run,
            "dataset_version": self.dataset_version,
            "version": self.version,
            "prompt_version": self.prompt_version or (L.prompt_version if L else None),
            "model_version": self.generator or (L.generator if L else None) or None,
            "retriever_version": L.retrieval_version if L else None,
            "index_version": L.index_version if L else None,
        }


RETRIEVAL_STAGES = ("embed", "dense", "lexical", "fuse", "rerank", "pack")


def traced(hit) -> TracedHit:
    first = hit.first if isinstance(hit, RerankedHit) else hit
    return TracedHit(
        rank=hit.rank,
        chunk_id=hit.chunk.chunk_id,
        citation=hit.chunk.citation(),
        score=float(hit.score),
        cosine=gate(hit),
        lexical=float(first.lexical) if isinstance(first, HybridHit) else None,
        relevance=relevance(hit),
        first_rank=first.rank if isinstance(hit, RerankedHit) else None,
    )


def drops(ctx) -> list[TracedDrop]:
    """The ranked passages the pack step left out: under the cosine floor, under the reranker's floor,
    or past the budget."""
    packed = {s.chunk.chunk_id for s in ctx.sources}
    out: list[TracedDrop] = []
    r = ctx.retrieval
    for h in ctx.ranked:
        if h.chunk.chunk_id in packed:
            continue
        cosine = gate(h)
        if r.min_score is not None and cosine is not None and cosine < r.min_score:
            reason = "threshold"
        elif (
            r.rerank is not None
            and r.rerank.min_relevance is not None
            and (relevance(h) or 0.0) < r.rerank.min_relevance
        ):
            reason = "relevance"
        else:
            reason = "budget"
        out.append(
            TracedDrop(rank=h.rank, chunk_id=h.chunk.chunk_id, citation=h.chunk.citation(), reason=reason)
        )
    return out


def trace_answer(
    a: Answer,
    version: str = "default",
    lineage: PipelineManifest | None = None,
    prompt_version: str = "",
    judgement: Judgement | None = None,
    prices: dict[str, float] | None = None,
    example_id: str | None = None,
    run: str | None = None,
    dataset_version: str | None = None,
    reading=None,
) -> Trace:
    ctx, g = a.context, a.generated
    stages = dict(ctx.timings)
    if g is not None:
        stages["generate"] = g.ms
    if judgement is not None:
        stages["judge"] = judgement.ms
    if reading is not None:
        stages["read"] = reading.ms
    return Trace(
        trace_id=uuid.uuid4().hex[:12],
        at=datetime.now(UTC).isoformat(timespec="seconds"),
        question=a.question,
        version=version,
        retrieval=ctx.retrieval.label,
        lineage=lineage,
        prompt_version=prompt_version,
        stages={k: round(v, 1) for k, v in stages.items()},
        total_ms=round(a.retrieve_ms + (g.ms if g else 0.0) + (judgement.ms if judgement else 0.0), 1),
        candidates=ctx.candidates,
        first_stage=[traced(h) for h in ctx.first_stage],
        context=[traced(s.hit) for s in ctx.sources],
        top_score=ctx.top_score,
        below=ctx.below,
        degraded=ctx.degraded,
        outcome=a.outcome,
        reason=a.reason,
        answer=marked(a),
        citations=[
            TracedCitation(n=c.n, chunk_id=c.chunk.chunk_id, span=c.span, cited_text=c.cited_text)
            for c in a.citations
        ],
        cited_share=round(a.cited_share, 3),
        claim_share=round(a.claim_share, 3),
        uncited=list(a.uncited),
        generator=g.model if g else "",
        input_tokens=g.usage.input_tokens if g else 0,
        output_tokens=g.usage.output_tokens if g else 0,
        cost_usd=round(g.usage.cost(prices), 6) if g and prices else 0.0,
        judgement=TracedJudgement(
            grounded=judgement.grounded,
            fabricated=judgement.fabricated,
            gap_named=judgement.gap_named,
            model=judgement.model,
            ms=judgement.ms,
            cost_usd=judgement.cost_usd,
        )
        if judgement
        else None,
        error=a.error,
        example_id=example_id,
        run=run,
        dataset_version=dataset_version,
        reranked=[traced(h) for h in ctx.reranked],
        ranked=[traced(h) for h in ctx.ranked],
        dropped=drops(ctx),
        tokens=ctx.tokens if ctx.sources else 0,
        budget=ctx.retrieval.budget,
        over_budget=ctx.over_budget,
        reading=TracedReading(
            relevant=reading.relevant,
            reference_agrees=reading.reference_agrees,
            deprecation_named=reading.deprecation_named,
            readings_named=reading.readings_named,
            model=reading.model,
            ms=reading.ms,
            cost_usd=reading.cost_usd,
        )
        if reading is not None
        else None,
    )


class TraceLog:
    """Append-only JSON lines."""

    def __init__(self, path: Path = TRACES) -> None:
        self.path = path

    def append(self, trace: Trace) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            out.write(trace.model_dump_json() + "\n")

    def read(self, last: int | None = None) -> list[Trace]:
        if not self.path.exists():
            return []
        lines = [line for line in self.path.read_text().splitlines() if line]
        if last is not None:
            lines = lines[-last:]
        return [Trace.model_validate_json(line) for line in lines]

    def find(self, trace_id: str) -> Trace | None:
        return next((t for t in self.read() if t.trace_id == trace_id), None)


def render(t: Trace) -> str:
    """A trace as a person reads it: the question, the stages with their times, both orders, the answer."""
    lines = [
        f"trace {t.trace_id} · {t.at} · version {t.version} · {t.retrieval}",
        f"question: {t.question}",
    ]
    if t.example_id or t.run:
        lines.append(
            f"example: {t.example_id or '–'} · run {t.run or '–'}"
            + (f" · dataset {t.dataset_version}" if t.dataset_version else "")
        )
    if t.lineage:
        lines.append(
            f"lineage: index {t.lineage.index_version} · prompt {t.lineage.prompt_version} · "
            f"retrieval {t.lineage.retrieval_version} · embedding {t.lineage.embedding_version}"
        )
    elif t.prompt_version:
        lines.append(f"prompt: {t.prompt_version}")
    order = ["embed", "dense", "lexical", "fuse", "rerank", "pack", "generate", "judge", "read"]
    shown = [f"{k} {t.stages[k]:,.0f} ms" for k in order if k in t.stages]
    lines.append(f"stages: {' · '.join(shown)} · total {t.total_ms:,.0f} ms")
    if t.degraded:
        lines.append(f"degraded: {t.degraded}")
    lines.append(f"first stage ({t.candidates} eligible, top {len(t.first_stage)}):")
    for h in t.first_stage[:10]:
        parts = [f"{h.score:.3f}"]
        if h.cosine is not None and h.cosine != h.score:
            parts.append(f"cosine {h.cosine:.2f}")
        if h.lexical is not None:
            parts.append(f"bm25 {h.lexical:.1f}")
        lines.append(f"  {h.rank:2}. {' · '.join(parts)}  {h.citation}")
    if len(t.first_stage) > 10:
        lines.append(f"  … {len(t.first_stage) - 10} more")
    label = "context (reranked)" if any(h.relevance is not None for h in t.context) else "context"
    lines.append(f"{label}, {t.below} dropped under the floor:")
    for h in t.context:
        parts = [f"{h.score:.3f}"]
        if h.relevance is not None and h.first_rank is not None:
            parts.append(f"was {h.first_rank}.")
        if h.cosine is not None:
            parts.append(f"cosine {h.cosine:.2f}")
        lines.append(f"  [{h.rank}] {' · '.join(parts)}  {h.citation}")
    if t.dropped:
        lines.append("dropped by the pack step:")
        for d in t.dropped:
            lines.append(f"  {d.rank:2}. {d.reason:9} {d.citation}")
    if t.tokens is not None and t.context:
        lines.append(f"packed: {t.tokens:,} tokens" + (f" of a {t.budget:,} budget" if t.budget else ""))
    lines.append(f"outcome: {t.outcome}" + (f" ({t.reason})" if t.reason else ""))
    if t.answer:
        lines.append("answer:")
        lines += [f"  {line}" for line in t.answer.splitlines()]
        lines.append(
            f"citations: {len(t.citations)} · {t.cited_share:.0%} of the text cited, {t.claim_share:.0%} of the claims"
        )
    if t.generator:
        lines.append(
            f"model: {t.generator} · {t.input_tokens:,} in, {t.output_tokens:,} out · ${t.cost_usd:.4f} at list price"
        )
    if t.judgement:
        j = t.judgement
        lines.append(
            f"judge: {j.model} · grounded {j.grounded:.2f} · fabricated {j.fabricated:.2f} · gap named {j.gap_named:.2f}"
        )
    if t.reading:
        r = t.reading
        lines.append(
            f"reader: {r.model} · relevant {r.relevant:.2f} · agrees {r.reference_agrees:.2f} · "
            f"deprecation named {r.deprecation_named:.2f} · readings named {r.readings_named:.2f}"
        )
    return "\n".join(lines)


def as_json(t: Trace) -> str:
    return json.dumps(t.model_dump(), indent=2, ensure_ascii=False)
