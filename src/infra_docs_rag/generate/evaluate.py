"""The grounding benchmark: every labelled question answered end to end, with whether the answer was
grounded in the right passage, abstained, or neither, and what each call cost."""

import statistics
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from ..chunk.chunks import usable
from ..chunk.evaluate import Answer as Labelled
from ..chunk.evaluate import resolve_answer
from ..embed.embedders import Embedder
from ..embed.evaluate import machine
from ..embed.index import Index, Manifest
from ..ingest.models import Document
from ..retrieve.evaluate import BenchmarkQuery, BenchmarkSet
from ..retrieve.rerank import Reranker
from ..retrieve.retriever import DEFAULT_RETRIEVAL, Retrieval, Retriever
from .answer import Answer, answer, marked
from .generator import Generator
from .judge import Judge, Judgement

Expect = Literal["answer", "abstain", "partial"]
CLAIM_FLOOR = 0.8  # share of an answer's claims (gap sentences aside) that has to carry a citation to hold
RETRIES = 1  # how many more times a failed model call is tried before the case is recorded as an error


class GroundingQuery(BaseModel):
    id: str
    kind: Literal["adversarial"]
    text: str
    expect: Expect
    hypothesis: str  # written before any model was run


class GroundingSet(BaseModel):
    questions: list[GroundingQuery] = []


def load_grounding(path: Path, taken: set[str]) -> list[GroundingQuery]:
    if not path.exists():
        return []
    try:
        loaded = GroundingSet.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    for q in loaded.questions:
        if q.id in taken:
            raise ValueError(f"{path}: question id {q.id} is already in the retrieval benchmark")
        taken.add(q.id)
    return loaded.questions


@dataclass
class Case:
    """One question through the whole pipeline."""

    id: str
    kind: str  # paraphrase, identifier, metadata, out-of-scope, adversarial
    expect: Expect
    hypothesis: str
    answer: Answer
    right_retrieved: bool | None  # a labelled right chunk was among the passages; None when unlabelled
    right_cited: bool | None  # a labelled right chunk was cited; None when unlabelled or no answer
    judgement: Judgement | None = None  # the judge's reading of an answer; None when there was none
    right_sources: list[bool] | None = None  # per passage in rank order, whether it answers; None unlabelled

    @property
    def outcome(self) -> str:
        return self.answer.outcome

    @property
    def verdict(self) -> str:
        """`held` when the outcome is what the expectation asked for, else what went wrong."""
        o = self.outcome
        if self.expect == "abstain":
            return "held" if o in ("no-passage", "abstained") else "answered anyway"
        if self.expect == "answer":
            if o == "answered":
                if self.right_cited is False:
                    return "cited the wrong passage"
                return "held" if self.answer.claim_share >= CLAIM_FLOOR else "partly uncited"
            return "no-passage" if o == "no-passage" else o
        # partial: an answer that cites something and leaves something uncited or says so; judged by hand
        return "held" if o == "answered" else o

    @property
    def held(self) -> bool:
        return self.verdict == "held"


@dataclass
class Grounding:
    manifest: Manifest
    machine: str
    retrieval: Retrieval
    generator: str  # model id
    cases: list[Case]
    prices: dict[str, float]
    backend: str = ""  # how the model was called: the generator's name
    judge: str | None = None  # the judge model that read the answers, when one did
    effort: str = ""  # how hard the generator was asked to think

    def of_kind(self, *kinds: str) -> list[Case]:
        return [c for c in self.cases if c.kind in kinds]

    @property
    def in_scope(self) -> list[Case]:
        return [c for c in self.cases if c.kind not in ("out-of-scope", "adversarial")]

    @property
    def out_of_scope(self) -> list[Case]:
        return self.of_kind("out-of-scope")

    @property
    def adversarial(self) -> list[Case]:
        return self.of_kind("adversarial")

    @property
    def called(self) -> list[Case]:
        return [c for c in self.cases if c.answer.generated is not None]

    @property
    def judged(self) -> list[Case]:
        return [c for c in self.cases if c.judgement is not None]

    @property
    def unverified(self) -> int:
        """Typed citations that did not check out against a passage, over the run."""
        return sum(c.answer.generated.unverified for c in self.called)

    def judge_cost(self) -> float:
        return sum(c.judgement.cost_usd for c in self.judged)

    def count(self, cases: list[Case], outcome: str) -> int:
        return sum(c.outcome == outcome for c in cases)

    def cost(self) -> float:
        return sum(c.answer.generated.usage.cost(self.prices) for c in self.called)

    def median_ms(self) -> float:
        return statistics.median(c.answer.generated.ms for c in self.called) if self.called else 0.0

    def tokens(self) -> tuple[int, int]:
        return (
            sum(c.answer.generated.usage.input_tokens for c in self.called),
            sum(c.answer.generated.usage.output_tokens for c in self.called),
        )


def expectation(q: BenchmarkQuery) -> Expect:
    return "abstain" if q.kind == "out-of-scope" else "answer"


def evaluate_grounding(
    index: Index,
    embedder: Embedder,
    docs: list[Document],
    bench: BenchmarkSet,
    extra: list[BenchmarkQuery],
    adversarial: list[GroundingQuery],
    generator: Generator,
    reranker: Reranker | None = None,
    retrieval: Retrieval = DEFAULT_RETRIEVAL,
    limit: int | None = None,
    judge: Judge | None = None,
    tracer: Callable[[Answer, Judgement | None], None] | None = None,
) -> Grounding:
    """`tracer`, when given, is called with every answer and its judge reading, so the run can be
    written to the trace log as well as saved."""
    retriever = Retriever(index, embedder, reranker)
    by_id = {d.source_id: d for d in usable(docs)}
    labelled = [*bench.queries, *extra]
    answers: dict[str, Labelled] = {
        q.id: resolve_answer(q.expect, by_id, f"query {q.id}") for q in labelled if q.expect
    }
    todo: list[tuple[str, str, str, Expect, str, Retrieval]] = [
        *(
            (
                q.id,
                q.kind,
                q.text,
                expectation(q),
                q.hypothesis,
                retrieval.model_copy(update={"filter": q.filter}) if q.filter else retrieval,
            )
            for q in labelled
        ),
        *((q.id, q.kind, q.text, q.expect, q.hypothesis, retrieval) for q in adversarial),
    ]
    if limit is not None:
        todo = todo[:limit]

    cases: list[Case] = []
    for qid, kind, text, expect, hypothesis, this in todo:
        # One failed call is one case with outcome "error", tried twice; the run goes on.
        a = answer(text, retriever, generator, this, retries=RETRIES)
        right = answers.get(qid)
        retrieved = cited = flags = None
        if right is not None:
            flags = [right.answered_by(s.chunk) for s in a.context.sources]
            retrieved = any(flags)
            cited = any(right.answered_by(c) for c in a.cited_chunks) if a.outcome == "answered" else None
        judgement = None
        if judge and a.outcome == "answered":
            try:
                judgement = judge.judge(a.context, text, marked(a))
            except (
                ValueError
            ) as error:  # the judge is a reading, not a verdict: its absence is recorded, not fatal
                print(f"{qid}: the judge failed ({str(error)[:120]}); no reading")
        cases.append(Case(qid, kind, expect, hypothesis, a, retrieved, cited, judgement, flags))
        if tracer is not None:
            tracer(a, judgement)
        line = f"{qid}: {a.outcome}" + (f", {a.cited_share:.0%} cited" if a.outcome == "answered" else "")
        if a.outcome == "error":
            line += f" ({a.error})"
        if judgement is not None:
            line += (
                f" · {judge.name} grounded {judgement.grounded:.2f}, fabricated {judgement.fabricated:.2f}"
            )
        print(line)
    judged = [c.judgement for c in cases if c.judgement is not None]
    return Grounding(
        manifest=index.manifest,
        machine=machine(),
        retrieval=retrieval,
        generator=generator.model,
        cases=cases,
        prices=generator.prices,
        backend=generator.name,
        judge=judged[0].model if judged else None,
        effort=str(getattr(generator, "effort", "")),
    )
