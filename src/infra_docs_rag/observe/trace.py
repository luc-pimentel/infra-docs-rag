"""A trace is one question through the pipeline, as a record: what was asked, under which pipeline
version (the lineage), what the first stage ranked and with which scores, what the reranker made of
it, what was packed, what the model wrote and cited, what the judge read, and how long each stage took.
One JSON line per question in `data/traces/traces.jsonl`; `answer --trace` prints the one it made, and
a benchmark run writes one per case."""

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

    @property
    def answered(self) -> bool:
        return self.outcome == "answered"


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


def trace_answer(
    a: Answer,
    version: str = "default",
    lineage: PipelineManifest | None = None,
    prompt_version: str = "",
    judgement: Judgement | None = None,
    prices: dict[str, float] | None = None,
) -> Trace:
    ctx, g = a.context, a.generated
    stages = dict(ctx.timings)
    if g is not None:
        stages["generate"] = g.ms
    if judgement is not None:
        stages["judge"] = judgement.ms
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
    if t.lineage:
        lines.append(
            f"lineage: index {t.lineage.index_version} · prompt {t.lineage.prompt_version} · "
            f"retrieval {t.lineage.retrieval_version} · embedding {t.lineage.embedding_version}"
        )
    elif t.prompt_version:
        lines.append(f"prompt: {t.prompt_version}")
    order = ["embed", "dense", "lexical", "fuse", "rerank", "pack", "generate", "judge"]
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
    return "\n".join(lines)


def as_json(t: Trace) -> str:
    return json.dumps(t.model_dump(), indent=2, ensure_ascii=False)
