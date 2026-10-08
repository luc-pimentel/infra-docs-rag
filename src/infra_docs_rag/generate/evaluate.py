"""The grounding benchmark: every labelled question answered end to end, with whether the answer was
grounded in the right passage, abstained, or neither, and what each call cost."""

import statistics
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
from .answer import Answer, answer
from .generator import Generator

Expect = Literal["answer", "abstain", "partial"]


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
                return "held" if self.answer.cited_share >= 0.8 else "partly uncited"
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
) -> Grounding:
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
        a = answer(text, retriever, generator, this)
        right = answers.get(qid)
        retrieved = cited = None
        if right is not None:
            retrieved = any(right.answered_by(s.chunk) for s in a.context.sources)
            cited = any(right.answered_by(c) for c in a.cited_chunks) if a.outcome == "answered" else None
        cases.append(Case(qid, kind, expect, hypothesis, a, retrieved, cited))
        print(f"{qid}: {a.outcome}" + (f", {a.cited_share:.0%} cited" if a.outcome == "answered" else ""))
    return Grounding(
        manifest=index.manifest,
        machine=machine(),
        retrieval=retrieval,
        generator=generator.model,
        cases=cases,
        prices=generator.prices,
    )
