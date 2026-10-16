"""A benchmark run on disk: `eval/runs/<name>/manifest.json` says what pipeline answered, and
`cases.jsonl` holds every question's answer, passages, citations, checks and judge readings. Labels are
written against a saved run and two saved runs are compared, so nothing that costs a model call has to
be repeated to read a number off it."""

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..embed.index import Manifest
from ..generate.answer import marked
from ..generate.evaluate import Case, Grounding
from ..generate.prompt import SYSTEM, SYSTEM_TYPED
from ..pipeline.manifest import PipelineManifest
from ..retrieve.retriever import Retrieval, gate, relevance

RUNS_DIR = Path("eval/runs")
Expect = Literal["answer", "abstain", "partial"]


class CitedRecord(BaseModel):
    document: int  # the passage's number in the context
    cited_text: str


class SegmentRecord(BaseModel):
    text: str
    citations: list[CitedRecord] = []


class SourceRecord(BaseModel):
    """One passage the model read, in rank order."""

    n: int
    chunk_id: str
    citation: str  # `source › Section › Subsection, pages`
    heading: str = (
        ""  # `[n] Project › Section › Subsection, pages`: what the model and the judge read it under
    )
    score: float  # the ranking's score: cosine, BM25, fused or reranker relevance
    cosine: float | None  # the cosine similarity underneath, what the threshold reads
    relevance: float | None  # the reranker's score, when there was one
    right: bool | None  # answers the question, per the label; None when the question is unlabelled
    text: str


class UsageRecord(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def cost(self, prices: dict[str, float]) -> float:
        return (
            (self.input_tokens - self.cache_read_tokens) * prices["input"]
            + self.cache_read_tokens * prices["cache_read"]
            + self.output_tokens * prices["output"]
        ) / 1e6


class GeneratedRecord(BaseModel):
    model: str
    stop_reason: str
    usage: UsageRecord
    ms: float
    unverified: int = 0
    refusal: str | None = None


class ReadingRecord(BaseModel):
    """The dataset reader's four probabilities (evaluation/dataset_judge.py); only a dataset run has them."""

    relevant: float
    reference_agrees: float
    deprecation_named: float
    readings_named: float
    model: str
    cost_usd: float
    ms: float


class JudgementRecord(BaseModel):
    grounded: float
    fabricated: float
    gap_named: float
    model: str
    cost_usd: float
    ms: float


class CaseRecord(BaseModel):
    """One question through the pipeline, as the run saved it."""

    id: str
    kind: str
    expect: Expect
    hypothesis: str
    question: str
    outcome: str  # no-passage, abstained, answered, refused, cut-off, error
    verdict: str  # held, or what went wrong, as the grounding benchmark judged it
    reason: str = ""  # why there is no answer, when there is none
    text: str = ""
    marked: str = ""  # the answer with [n] after each cited span
    segments: list[SegmentRecord] = []
    sources: list[SourceRecord] = []
    candidates: int = 0
    top_score: float | None = None
    below: int = 0
    right_retrieved: bool | None = None
    right_cited: bool | None = None
    cited_share: float = 0.0  # of the answer text
    claim_share: float = 0.0  # of the answer text with the gap sentences left out; what the verdict reads
    gap_sentences: list[str] = []  # uncited sentences that name what the passages do not cover
    cited_chunk_ids: list[str] = []
    uncited: list[str] = []
    retrieve_ms: float = 0.0
    generated: GeneratedRecord | None = None
    judgement: JudgementRecord | None = None
    reading: ReadingRecord | None = None  # the dataset reader's probabilities, on a dataset run
    trace_id: str | None = None  # the trace behind the answer, in `traces.jsonl` beside the run (stage 17)

    @property
    def held(self) -> bool:
        return self.verdict == "held"

    @property
    def abstained(self) -> bool:
        return self.outcome in ("no-passage", "abstained")

    @property
    def answered(self) -> bool:
        return self.outcome == "answered"

    @property
    def called(self) -> bool:
        return self.generated is not None

    @property
    def in_scope(self) -> bool:
        return self.kind not in ("out-of-scope", "adversarial")

    @property
    def labelled(self) -> bool:
        return self.right_retrieved is not None

    @property
    def right_rank(self) -> int | None:
        """Where the first right passage sits among the ones the model read; None when none does."""
        return next((s.n for s in self.sources if s.right), None)

    @property
    def end_to_end_ms(self) -> float:
        return self.retrieve_ms + (self.generated.ms if self.generated else 0.0)


class RunManifest(BaseModel):
    name: str
    version: str  # the pipeline version's name, from eval/versions.yaml, or `default`
    description: str = ""
    created_at: str
    git_sha: str
    machine: str
    index: Manifest
    retrieval: Retrieval
    generator: str  # model id
    backend: str  # how it was called
    effort: str
    prompt_sha256: str  # of the system prompt the backend uses
    prices: dict[str, float]
    judge: str | None = None
    questions: int
    lineage: PipelineManifest | None = None  # which version of every stage produced the answers
    dataset: str | None = None  # `<records>+<hash>`: the dataset version a dataset run answered (dataset.py)


class Run(BaseModel):
    manifest: RunManifest
    cases: list[CaseRecord]

    @property
    def name(self) -> str:
        return self.manifest.name

    def by_id(self) -> dict[str, CaseRecord]:
        return {c.id: c for c in self.cases}

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(self.manifest.model_dump_json(indent=2) + "\n")
        (directory / "cases.jsonl").write_text("".join(c.model_dump_json() + "\n" for c in self.cases))
        return directory

    @classmethod
    def load(cls, directory: Path) -> "Run":
        manifest, cases = directory / "manifest.json", directory / "cases.jsonl"
        if not manifest.exists() or not cases.exists():
            raise FileNotFoundError(
                f"{directory} holds no run; `evaluate-grounding --save {directory.name}` writes one"
            )
        return cls(
            manifest=RunManifest.model_validate_json(manifest.read_text()),
            cases=[CaseRecord.model_validate_json(line) for line in cases.read_text().splitlines() if line],
        )


def git_sha() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:
        return "unknown"
    return done.stdout.strip() or "unknown"


def prompt_hash(backend: str) -> str:
    return hashlib.sha256((SYSTEM_TYPED if backend == "claude-code" else SYSTEM).encode()).hexdigest()[:12]


def record(c: Case) -> CaseRecord:
    a = c.answer
    g = a.generated
    flags = c.right_sources
    return CaseRecord(
        id=c.id,
        kind=c.kind,
        expect=c.expect,
        hypothesis=c.hypothesis,
        question=a.question,
        outcome=a.outcome,
        verdict=c.verdict,
        reason=a.reason,
        text=a.text,
        marked=marked(a),
        segments=[
            SegmentRecord(
                text=s.text,
                citations=[CitedRecord(document=x.document, cited_text=x.cited_text) for x in s.citations],
            )
            for s in (g.segments if g else [])
        ],
        sources=[
            SourceRecord(
                n=s.n,
                chunk_id=s.chunk.chunk_id,
                citation=s.chunk.citation(),
                heading=s.heading,
                score=s.hit.score,
                cosine=gate(s.hit),
                relevance=relevance(s.hit),
                right=flags[i] if flags is not None else None,
                text=s.chunk.text,
            )
            for i, s in enumerate(a.context.sources)
        ],
        candidates=a.context.candidates,
        top_score=a.context.top_score,
        below=a.context.below,
        right_retrieved=c.right_retrieved,
        right_cited=c.right_cited,
        cited_share=a.cited_share,
        claim_share=a.claim_share,
        gap_sentences=a.gap_sentences,
        cited_chunk_ids=[x.chunk_id for x in a.cited_chunks],
        uncited=list(a.uncited),
        retrieve_ms=a.retrieve_ms,
        generated=GeneratedRecord(
            model=g.model,
            stop_reason=g.stop_reason,
            usage=UsageRecord(
                input_tokens=g.usage.input_tokens,
                output_tokens=g.usage.output_tokens,
                cache_read_tokens=g.usage.cache_read_tokens,
            ),
            ms=g.ms,
            unverified=g.unverified,
            refusal=g.refusal,
        )
        if g
        else None,
        judgement=JudgementRecord(
            grounded=c.judgement.grounded,
            fabricated=c.judgement.fabricated,
            gap_named=c.judgement.gap_named,
            model=c.judgement.model,
            cost_usd=c.judgement.cost_usd,
            ms=c.judgement.ms,
        )
        if c.judgement
        else None,
        trace_id=c.trace_id,
    )


def from_grounding(
    ev: Grounding,
    name: str,
    version: str = "default",
    description: str = "",
    lineage: PipelineManifest | None = None,
) -> Run:
    """A grounding benchmark's result as a run to save."""
    return Run(
        manifest=RunManifest(
            name=name,
            version=version,
            description=description,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=git_sha(),
            machine=ev.machine,
            index=ev.manifest,
            retrieval=ev.retrieval,
            generator=ev.generator,
            backend=ev.backend,
            effort=ev.effort,
            prompt_sha256=prompt_hash(ev.backend),
            prices=dict(ev.prices),
            judge=ev.judge,
            questions=len(ev.cases),
            lineage=lineage.model_copy(update={"eval_run_id": name}) if lineage else None,
        ),
        cases=[record(c) for c in ev.cases],
    )
