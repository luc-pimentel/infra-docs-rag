"""The evaluation dataset in `eval/dataset/`: one record per question, with the passages that answer
it, a reference answer, and the failure mode it is in the set to catch.

The benchmark files (`retrieval.yaml`, `hybrid.yaml`, `grounding.yaml`) say where an answer is and what
a run has to do; this dataset says what a right answer *is*, and why each question is there. Most
records come from the saved runs: a question is in the set because a run abstained on it, cited the
wrong passage, left its claims uncited, answered when it should not have, or was misread by the judge.
The rest are written for the failure modes the runs have not measured yet (a stale field, an ambiguous
question, an answer spread over two passages) and for questions people actually asked the service.

`records.yaml` is the curated set. `generated.yaml` is a testset written by a model from sampled
chunks, and `audit.yaml` says which of its examples were kept, edited or discarded and why; the dataset
is the curated set plus what the audit kept.
"""

import hashlib
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from ..chunk.chunks import Chunk
from ..chunk.evaluate import resolve_answer
from ..generate.prompt import ABSTAIN
from ..ingest.models import Document
from ..retrieve.evaluate import BenchmarkQuery, load_benchmark
from .metrics import FABRICATED_CEILING, GROUNDED_FLOOR
from .runs import CaseRecord, Run

DATASET_DIR = Path("eval/dataset")
RECORDS = DATASET_DIR / "records.yaml"
GENERATED = DATASET_DIR / "generated.yaml"
AUDIT = DATASET_DIR / "audit.yaml"
SCHEMA = DATASET_DIR / "schema.json"
SIZE = (30, 50)  # how many curated records the set is meant to hold

Category = Literal["fact-lookup", "multi-hop", "ambiguous", "no-answer", "stale", "adversarial"]
CATEGORIES: tuple[Category, ...] = (
    "fact-lookup",
    "multi-hop",
    "ambiguous",
    "no-answer",
    "stale",
    "adversarial",
)
Expect = Literal["answer", "abstain", "partial", "clarify"]
Difficulty = Literal["easy", "medium", "hard"]
Split = Literal["dev", "test"]
Origin = Literal["benchmark", "traces", "hand-written", "generated"]
FailureMode = Literal[
    "none",
    "false-abstention",
    "retrieval-miss",
    "wrong-passage",
    "uncited-claims",
    "unverified-citation",
    "answered-anyway",
    "judge-misread",
    "fabrication",
    "stale-answer",
    "wrong-reading",
    "missing-hop",
]
# What each failure mode means, for the schema and the report.
FAILURE_MODES: dict[str, str] = {
    "none": "a clean example: the run should answer it from the passages, and did",
    "false-abstention": "an in-scope question that got the abstention sentence",
    "retrieval-miss": "no right passage among the ones the model read",
    "wrong-passage": "the answer cites a passage that is not a right one",
    "uncited-claims": "under 80% of the answer's claims carry a verified citation",
    "unverified-citation": "a typed citation that did not check out against its passage",
    "answered-anyway": "a question that should get the abstention sentence, answered from memory or a near miss",
    "judge-misread": "the judge reads a grounded answer as ungrounded or fabricated",
    "fabrication": "the answer states a fact no passage gives, or accepts a false premise",
    "stale-answer": "the answer presents a deprecated field, flag or method as current",
    "wrong-reading": "an ambiguous question answered under one reading without naming the others",
    "missing-hop": "a question whose answer spans two passages, answered from one",
}
# Failure modes a saved run can show; the rest are designed into the set but not measured yet.
OBSERVABLE = (
    "false-abstention",
    "retrieval-miss",
    "wrong-passage",
    "uncited-claims",
    "unverified-citation",
    "answered-anyway",
    "judge-misread",
)
CHUNK_ID = re.compile(r"^[a-z0-9-]+:\d+$")


class Record(BaseModel):
    """One evaluation example."""

    id: str
    query: str
    category: Category
    expect: Expect  # what a right reply does: answer, abstain, answer part and name the rest, or ask
    difficulty: Difficulty
    failure_mode: FailureMode  # the failure this example is in the set to catch
    split: Split
    origin: Origin
    source_doc_ids: list[str] = []  # the sources the contexts come from, from sources.yaml
    contexts: list[str] = []  # chunk ids of the passages a right answer rests on; [] when none does
    reference_answer: str  # what a right answer says, from the contexts alone
    tags: list[str] = []
    benchmark_id: str | None = None  # the benchmark question this record was taken from
    note: str = ""  # why it is in the set: what a run did with it, or what it is written to catch

    @field_validator("contexts")
    @classmethod
    def chunk_ids(cls, value: list[str]) -> list[str]:
        for chunk_id in value:
            if not CHUNK_ID.match(chunk_id):
                raise ValueError(f"{chunk_id!r} is not a chunk id (`source:offset`)")
        if len(value) != len(set(value)):
            raise ValueError("a context is listed twice")
        return value

    @field_validator("reference_answer", "query")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be blank")
        return value.strip()

    @model_validator(mode="after")
    def consistent(self) -> "Record":
        if self.origin == "benchmark" and self.benchmark_id is None:
            self.benchmark_id = self.id
        if self.origin != "benchmark" and self.benchmark_id is not None:
            raise ValueError(f"{self.id}: benchmark_id is for records taken from the benchmark")
        if self.category == "no-answer" and self.expect != "abstain":
            raise ValueError(f"{self.id}: a no-answer question expects an abstention")
        if self.category == "ambiguous" and self.expect not in ("clarify", "partial"):
            raise ValueError(f"{self.id}: an ambiguous question expects a clarification or a partial answer")
        if self.expect == "abstain":
            if self.reference_answer != ABSTAIN:
                raise ValueError(f"{self.id}: an abstention's reference answer is the abstention sentence")
            if self.contexts:
                raise ValueError(f"{self.id}: an abstention rests on no passage")
        elif not self.contexts:
            raise ValueError(f"{self.id}: an answer needs the contexts it rests on")
        sources = sorted({c.split(":")[0] for c in self.contexts})
        if sorted(self.source_doc_ids) != sources:
            raise ValueError(f"{self.id}: source_doc_ids must be the contexts' sources, {sources}")
        if self.category == "multi-hop" and len(self.contexts) < 2:
            raise ValueError(f"{self.id}: a multi-hop question rests on at least two passages")
        return self


class RecordSet(BaseModel):
    chunking: str  # the chunk file under data/chunks/ the contexts name
    records: list[Record]


class Generated(BaseModel):
    """One example a model wrote from a chunk."""

    id: str
    chunk_id: str
    source_doc_id: str
    section: str
    query: str
    reference_answer: str
    difficulty: Difficulty
    tags: list[str] = []


class GeneratedSet(BaseModel):
    chunking: str
    model: str
    backend: str
    effort: str
    prompt_sha256: str
    seed: int
    generated_at: str
    examples: list[Generated]


Decision = Literal["kept", "edited", "discarded"]


class AuditEntry(BaseModel):
    id: str
    decision: Decision
    reason: str
    query: str | None = None  # the edited question
    reference_answer: str | None = None  # the edited answer
    difficulty: Difficulty | None = None
    tags: list[str] | None = None
    split: Split = "test"

    @model_validator(mode="after")
    def edits_only_when_edited(self) -> "AuditEntry":
        changed = any(x is not None for x in (self.query, self.reference_answer, self.difficulty, self.tags))
        if self.decision == "edited" and not changed:
            raise ValueError(f"{self.id}: edited, but nothing changed")
        if self.decision != "edited" and changed:
            raise ValueError(f"{self.id}: {self.decision}, so nothing should change")
        if not self.reason.strip():
            raise ValueError(f"{self.id}: say why")
        return self


class Audit(BaseModel):
    auditor: str
    generated: str  # the generated file the decisions are about
    decisions: list[AuditEntry]


def load_yaml(path: Path, model: type[BaseModel]) -> BaseModel:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        return model.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


def dataset_version(records: list[Record]) -> str:
    """`74+1f2e3d4c5b6a`: how many records, then a hash of every field of every one, in id order, so the
    same set in any file layout has the same version and one edited record changes it. A saved dataset
    run carries it, and the regression baseline (regression.py) is checked against it."""
    h = hashlib.sha256()
    for r in sorted(records, key=lambda x: x.id):
        h.update(r.model_dump_json().encode())
        h.update(b"\n")
    return f"{len(records)}+{h.hexdigest()[:12]}"


def load_records(path: Path = RECORDS) -> RecordSet:
    loaded = load_yaml(path, RecordSet)
    assert isinstance(loaded, RecordSet)
    ids = [r.id for r in loaded.records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: a record id repeats: {[i for i, n in Counter(ids).items() if n > 1]}")
    return loaded


def load_generated(path: Path = GENERATED) -> GeneratedSet:
    loaded = load_yaml(path, GeneratedSet)
    assert isinstance(loaded, GeneratedSet)
    ids = [g.id for g in loaded.examples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: a generated id repeats")
    return loaded


def load_audit(path: Path, generated: GeneratedSet) -> Audit:
    loaded = load_yaml(path, Audit)
    assert isinstance(loaded, Audit)
    known = {g.id for g in generated.examples}
    decided = [d.id for d in loaded.decisions]
    if len(decided) != len(set(decided)):
        raise ValueError(f"{path}: an example is decided twice")
    for d in loaded.decisions:
        if d.id not in known:
            raise ValueError(f"{path}: {d.id} is not in the generated set")
    missing = sorted(known - set(decided))
    if missing:
        raise ValueError(f"{path}: no decision for {', '.join(missing)}")
    return loaded


def audited(generated: GeneratedSet, audit: Audit) -> list[Record]:
    """The generated examples the audit kept, as records, with the edits applied."""
    by_id = {g.id: g for g in generated.examples}
    kept: list[Record] = []
    for d in audit.decisions:
        if d.decision == "discarded":
            continue
        g = by_id[d.id]
        kept.append(
            Record(
                id=g.id,
                query=d.query or g.query,
                category="fact-lookup",
                expect="answer",
                difficulty=d.difficulty or g.difficulty,
                failure_mode="none",
                split=d.split,
                origin="generated",
                source_doc_ids=[g.source_doc_id],
                contexts=[g.chunk_id],
                reference_answer=d.reference_answer or g.reference_answer,
                tags=d.tags if d.tags is not None else g.tags,
                note=f"{d.decision}: {d.reason}",
            )
        )
    return kept


@dataclass
class Dataset:
    curated: RecordSet
    generated: GeneratedSet | None
    audit: Audit | None
    problems: list[str] = field(default_factory=list)  # what `check` found wrong

    @property
    def kept(self) -> list[Record]:
        if self.generated is None or self.audit is None:
            return []
        return audited(self.generated, self.audit)

    @property
    def records(self) -> list[Record]:
        return [*self.curated.records, *self.kept]

    @property
    def ok(self) -> bool:
        return not self.problems


def load_dataset(directory: Path = DATASET_DIR) -> Dataset:
    """The curated records, plus the generated set and its audit when both exist."""
    curated = load_records(directory / RECORDS.name)
    generated = audit = None
    if (directory / GENERATED.name).exists():
        generated = load_generated(directory / GENERATED.name)
        if (directory / AUDIT.name).exists():
            audit = load_audit(directory / AUDIT.name, generated)
            if Path(audit.generated).name != GENERATED.name:
                raise ValueError(f"{AUDIT.name} audits {audit.generated!r}, not the generated set beside it")
        if generated.chunking != curated.chunking:
            raise ValueError(
                f"the generated set names chunking {generated.chunking!r}, the records {curated.chunking!r}"
            )
    return Dataset(curated, generated, audit)


def gold_contexts(query: BenchmarkQuery, docs: dict[str, Document], chunks: list[Chunk]) -> list[str]:
    """The chunk ids the benchmark's label says answer a question, under the current chunking."""
    if query.expect is None:
        return []
    answer = resolve_answer(query.expect, docs, query.id)
    return [c.chunk_id for c in chunks if answer.answered_by(c)]


@dataclass
class BenchmarkIndex:
    """Every benchmark question by id: the labelled and out-of-scope ones as queries, the adversarial
    ones (which carry an explicit expectation and no label) by what they expect."""

    queries: dict[str, BenchmarkQuery]
    adversarial: dict[str, str]  # id -> answer | abstain | partial

    def __contains__(self, question_id: str) -> bool:
        return question_id in self.queries or question_id in self.adversarial


def check(
    ds: Dataset,
    chunks: list[Chunk],
    docs: dict[str, Document],
    benchmark: BenchmarkIndex,
    sources: set[str],
) -> list[str]:
    """Everything that is wrong with the dataset against the corpus and the benchmark; empty when
    nothing is. The schema checks run on load; these need the chunks, documents and benchmark."""
    problems: list[str] = []
    by_chunk = {c.chunk_id: c for c in chunks}
    gold: dict[str, list[str]] = {}
    for r in ds.records:
        for chunk_id in r.contexts:
            if chunk_id not in by_chunk:
                problems.append(f"{r.id}: context {chunk_id} is not in the {ds.curated.chunking} chunks")
        for source in r.source_doc_ids:
            if source not in sources:
                problems.append(f"{r.id}: source {source} is not in sources.yaml")
        if r.benchmark_id is not None:
            if r.benchmark_id in benchmark.adversarial:
                expected = benchmark.adversarial[r.benchmark_id]
                if r.expect != expected:
                    problems.append(
                        f"{r.id}: the benchmark expects {expected} on {r.benchmark_id}, not {r.expect}"
                    )
                continue  # no label to check the contexts against: the record's own say what a right answer rests on
            q = benchmark.queries.get(r.benchmark_id)
            if q is None:
                problems.append(f"{r.id}: benchmark question {r.benchmark_id} does not exist")
                continue
            if r.benchmark_id not in gold:
                try:
                    gold[r.benchmark_id] = gold_contexts(q, docs, chunks)
                except ValueError as error:
                    problems.append(f"{r.id}: {error}")
                    continue
            outside = [c for c in r.contexts if c not in gold[r.benchmark_id]]
            if outside:
                problems.append(
                    f"{r.id}: contexts {outside} are not among the passages the benchmark labels for "
                    f"{r.benchmark_id}"
                )
            if q.expect is None and r.expect != "abstain":
                problems.append(f"{r.id}: the benchmark expects an abstention on {r.benchmark_id}")
    n = len(ds.curated.records)
    if not SIZE[0] <= n <= SIZE[1]:
        problems.append(f"{n} curated records; the set is meant to hold {SIZE[0]} to {SIZE[1]}")
    present = {r.category for r in ds.curated.records}
    for category in CATEGORIES:
        if category not in present:
            problems.append(f"no curated record in category {category}")
    ids = Counter(r.id for r in ds.records)
    for i, count in ids.items():
        if count > 1:
            problems.append(f"{i} appears {count} times across the curated and generated records")
    if ds.generated is not None:
        for g in ds.generated.examples:
            if g.chunk_id not in by_chunk:
                problems.append(f"{g.id}: generated from {g.chunk_id}, which is not in the chunks")
    ds.problems = problems
    return problems


# --- what the saved runs observed ---------------------------------------------------------------


@dataclass
class Observation:
    """What one saved run did with a record's benchmark question."""

    run: str
    outcome: str
    verdict: str
    modes: list[str]  # the observable failure modes the case shows; [] when it held cleanly

    @property
    def held(self) -> bool:
        return self.verdict == "held"


def modes_of(c: CaseRecord, grounded_floor: float, fabricated_ceiling: float) -> list[str]:
    """The failure modes a saved case shows, by the benchmark's own reading of it."""
    modes: list[str] = []
    if c.verdict == "abstained":
        modes.append("false-abstention")
    if c.right_retrieved is False:
        modes.append("retrieval-miss")
    if c.verdict == "cited the wrong passage":
        modes.append("wrong-passage")
    if c.verdict == "partly uncited":
        modes.append("uncited-claims")
    if c.verdict == "answered anyway":
        modes.append("answered-anyway")
    if c.generated is not None and c.generated.unverified:
        modes.append("unverified-citation")
    j = c.judgement
    if c.held and j is not None and (j.grounded < grounded_floor or j.fabricated >= fabricated_ceiling):
        modes.append("judge-misread")
    return modes


def observe(records: list[Record], runs: list[Run]) -> dict[str, list[Observation]]:
    """Per record, what each run did with the benchmark question it was taken from, in run order.
    Records with no benchmark question get an empty list."""
    observed: dict[str, list[Observation]] = {r.id: [] for r in records}
    for run in runs:
        cases = run.by_id()
        for r in records:
            c = cases.get(r.benchmark_id) if r.benchmark_id else None
            if c is None:
                continue
            observed[r.id].append(
                Observation(run.name, c.outcome, c.verdict, modes_of(c, GROUNDED_FLOOR, FABRICATED_CEILING))
            )
    return observed


def load_runs(runs_dir: Path) -> list[Run]:
    if not runs_dir.exists():
        return []
    return [Run.load(d) for d in sorted(runs_dir.iterdir()) if (d / "manifest.json").exists()]


def benchmark_index(files: tuple[Path, ...]) -> BenchmarkIndex:
    """Every question across the benchmark files. The grounding file's questions are adversarial: they
    carry an expectation instead of a label."""
    queries: dict[str, BenchmarkQuery] = {}
    adversarial: dict[str, str] = {}
    for path in files:
        if path.name == "grounding.yaml":
            loaded = yaml.safe_load(path.read_text()) or {}
            for q in loaded.get("questions", []):
                adversarial[q["id"]] = q["expect"]
        else:
            for q in load_benchmark(path).queries:
                queries[q.id] = q
    return BenchmarkIndex(queries, adversarial)


def json_schema() -> dict:
    """The record schema as JSON Schema, with the failure modes explained."""
    schema = Record.model_json_schema()
    schema["title"] = "infra-docs-rag evaluation record"
    schema["description"] = (
        "One question through the documentation QA system: what a right reply is, which passages it "
        "rests on, and the failure mode the example is in the set to catch. Contexts are chunk ids "
        "under the chunking named at the top of records.yaml."
    )
    schema.setdefault("$defs", {})["FailureMode"] = {
        "type": "string",
        "enum": list(FAILURE_MODES),
        "description": "; ".join(f"{k}: {v}" for k, v in FAILURE_MODES.items()),
    }
    schema["properties"]["failure_mode"] = {"$ref": "#/$defs/FailureMode"}
    return schema
