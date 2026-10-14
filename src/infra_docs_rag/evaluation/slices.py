"""Metrics over a dataset run, overall and per slice, and the rules that read them: the matrix in
`eval/matrix.yaml` maps each failure mode to the metric that measures it, its direction and the
threshold it is held to on the records designed for it; the slice gates in the same file are the
pass/fail rules a change has to clear, on the whole set or on one slice.

The stage 08 metrics (retrieval, groundedness, abstention, cost, latency) are computed on each slice
as they are on a run; this module adds what the dataset makes possible: context recall and precision
against the record's named passages, relevance and agreement with the reference answer from the
reader, and one signal per failure mode the benchmark could not measure (a deprecation named, the
readings of an ambiguous question named, both passages of a two-passage answer cited).
"""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from .dataset import FAILURE_MODES, Record
from .gates import Gate, GateResult, check
from .metrics import FABRICATED_CEILING, GROUNDED_FLOOR, Metric, metrics, share, shown_share
from .runs import CaseRecord, Run

MATRIX = Path("eval/matrix.yaml")
SLICE_FIELDS = ("category", "failure_mode", "difficulty", "split", "origin")
ALL = "all"
READ_FLOOR = 0.5  # a reader probability at or over this reads as yes


def slice_name(field: str, value: str) -> str:
    return f"{field}={value}"


def slices_of(records: list[Record]) -> dict[str, list[Record]]:
    """`all`, then one slice per value of each slice field, in field order."""
    out: dict[str, list[Record]] = {ALL: list(records)}
    for field in SLICE_FIELDS:
        for r in records:
            out.setdefault(slice_name(field, getattr(r, field)), []).append(r)
    return out


def sub_run(run: Run, ids: set[str]) -> Run:
    return Run(manifest=run.manifest, cases=[c for c in run.cases if c.id in ids])


def dataset_metrics(run: Run, records: list[Record]) -> dict[str, Metric]:
    """The stage 08 metrics on the run, plus the dataset's own, keyed by name."""
    by_record = {r.id: r for r in records}
    cases = [c for c in run.cases if c.id in by_record]
    answered = [c for c in cases if c.answered]
    with_contexts = [c for c in cases if by_record[c.id].contexts]
    read = [c for c in answered if c.reading is not None]
    judged = [c for c in answered if c.judgement is not None]
    expect_abstain = [c for c in cases if c.expect == "abstain"]
    expect_answer = [c for c in cases if c.expect != "abstain"]
    multi = [c for c in answered if by_record[c.id].category == "multi-hop"]
    stale = [c for c in read if by_record[c.id].category == "stale"]
    ambiguous = [c for c in read if by_record[c.id].category == "ambiguous"]

    def recall(c: CaseRecord) -> float:
        wanted = set(by_record[c.id].contexts)
        got = {s.chunk_id for s in c.sources}
        return len(wanted & got) / len(wanted)

    def precision(c: CaseRecord) -> float | None:
        wanted = set(by_record[c.id].contexts)
        got = [s.chunk_id for s in c.sources]
        return sum(x in wanted for x in got) / len(got) if got else None

    recalls = [recall(c) for c in with_contexts]
    precisions = [p for p in (precision(c) for c in with_contexts) if p is not None]
    relevant = [c for c in read if c.reading.relevant >= READ_FLOOR]
    agrees = [c for c in read if c.reading.reference_agrees >= READ_FLOOR]
    deprecation = [c for c in stale if c.reading.deprecation_named >= READ_FLOOR]
    readings = [c for c in ambiguous if c.reading.readings_named >= READ_FLOOR]
    hops = [c for c in multi if len(set(c.cited_chunk_ids) & set(by_record[c.id].contexts)) >= 2]
    abstained_right = [c for c in expect_abstain if c.abstained]
    abstained_wrong = [c for c in expect_answer if c.abstained]
    judge_flagged_held = [
        c
        for c in judged
        if c.held and (c.judgement.grounded < GROUNDED_FLOOR or c.judgement.fabricated >= FABRICATED_CEILING)
    ]

    def m(
        name: str, value: float | None, shown: str, better: Literal["higher", "lower"], description: str
    ) -> Metric:
        return Metric(name, value, shown, better, description)

    own = [
        m(
            "retrieval.context_recall",
            statistics.mean(recalls) if recalls else None,
            f"{statistics.mean(recalls):.0%}" if recalls else "–",
            "higher",
            "share of a record's named passages among the ones the model read, averaged over the records that name any",
        ),
        m(
            "retrieval.context_precision",
            statistics.mean(precisions) if precisions else None,
            f"{statistics.mean(precisions):.0%}" if precisions else "–",
            "higher",
            "share of the passages the model read that the record names, averaged over the records that name any",
        ),
        m(
            "relevance.judge_relevant",
            share(relevant, read),
            shown_share(relevant, read),
            "higher",
            f"answers the reader says address the question (P ≥ {READ_FLOOR})",
        ),
        m(
            "relevance.reference_agrees",
            share(agrees, read),
            shown_share(agrees, read),
            "higher",
            f"answers the reader says agree with the reference answer (P ≥ {READ_FLOOR})",
        ),
        m(
            "abstention.expected",
            share(abstained_right, expect_abstain),
            shown_share(abstained_right, expect_abstain),
            "higher",
            "records expecting an abstention that got one",
        ),
        m(
            "abstention.false",
            share(abstained_wrong, expect_answer),
            shown_share(abstained_wrong, expect_answer),
            "lower",
            "records expecting an answer (whole or partial) that got the abstention sentence",
        ),
        m(
            "faithfulness.judge_flagged_held",
            share(judge_flagged_held, judged),
            shown_share(judge_flagged_held, judged),
            "lower",
            "answers whose expectation held by the checks and that the judge reads as ungrounded or fabricated",
        ),
        m(
            "stale.deprecation_named",
            share(deprecation, stale),
            shown_share(deprecation, stale),
            "higher",
            f"stale-category answers the reader says flag the deprecation the passage states (P ≥ {READ_FLOOR})",
        ),
        m(
            "ambiguity.readings_named",
            share(readings, ambiguous),
            shown_share(readings, ambiguous),
            "higher",
            f"ambiguous-category answers the reader says name more than one reading or ask (P ≥ {READ_FLOOR})",
        ),
        m(
            "hops.covered",
            share(hops, multi),
            shown_share(hops, multi),
            "higher",
            "multi-hop answers that cite at least two of the record's named passages",
        ),
    ]
    base = metrics(sub_run(run, set(by_record)))
    # The stage 08 groundedness metrics under the name the matrix uses for them.
    faithful = {
        name.replace("grounding.", "faithfulness."): Metric(
            name.replace("grounding.", "faithfulness."), x.value, x.shown, x.better, x.description
        )
        for name, x in base.items()
        if name.startswith("grounding.")
    }
    return {**base, **faithful, **{x.name: x for x in own}}


def slice_metrics(run: Run, records: list[Record]) -> dict[str, dict[str, Metric]]:
    return {name: dataset_metrics(run, rows) for name, rows in slices_of(records).items()}


# --- the matrix and the slice gates -------------------------------------------------------------

Direction = Literal["higher", "lower"]


class MatrixEntry(BaseModel):
    failure_mode: str
    metric: str
    direction: Direction
    rule: Literal["min", "max"]
    value: float
    slice: str | None = None  # where the threshold is read; default: the records designed for the mode
    blocker: bool = False  # a miss here is a blocker condition, not a target
    why: str = ""

    @model_validator(mode="after")
    def known(self) -> "MatrixEntry":
        if self.failure_mode not in FAILURE_MODES:
            raise ValueError(f"matrix: unknown failure mode {self.failure_mode!r}")
        if (self.direction == "higher") != (self.rule == "min"):
            raise ValueError(
                f"matrix: {self.metric} is better {self.direction}, so its rule is {'min' if self.direction == 'higher' else 'max'}"
            )
        if self.slice is None:
            self.slice = slice_name("failure_mode", self.failure_mode)
        return self


class SliceGate(Gate):
    slice: str = ALL


class MatrixFile(BaseModel):
    matrix: list[MatrixEntry]
    gates: list[SliceGate] = []


def load_matrix(path: Path = MATRIX) -> MatrixFile:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        return MatrixFile.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


@dataclass
class MatrixResult:
    entry: MatrixEntry
    metric: Metric | None  # on the entry's slice; None when the slice is empty or the metric unknown
    passed: bool | None  # None when nothing measures it
    detail: str


def check_matrix(entries: list[MatrixEntry], slices: dict[str, dict[str, Metric]]) -> list[MatrixResult]:
    out: list[MatrixResult] = []
    for e in entries:
        table = slices.get(e.slice or "")
        if table is None:
            out.append(MatrixResult(e, None, None, f"no record in slice {e.slice}"))
            continue
        metric = table.get(e.metric)
        if metric is None:
            raise ValueError(f"{MATRIX} names {e.metric}, which no run measures; one of {', '.join(table)}")
        if metric.value is None:
            out.append(MatrixResult(e, metric, None, "nothing in the slice measures this"))
            continue
        passed = metric.value >= e.value if e.rule == "min" else metric.value <= e.value
        bound = f"{'≥' if e.rule == 'min' else '≤'} {e.value:g}"
        out.append(MatrixResult(e, metric, passed, f"{metric.shown} against {bound}"))
    return out


@dataclass
class SliceGateResult:
    gate: SliceGate
    result: GateResult
    skipped: bool = False  # a no-drop / no-rise gate with no baseline run to compare against

    @property
    def blocks(self) -> bool:
        return not self.skipped and self.result.blocks


def check_slice_gates(
    gates: list[SliceGate],
    candidate: dict[str, dict[str, Metric]],
    baseline: dict[str, dict[str, Metric]] | None,
) -> list[SliceGateResult]:
    out: list[SliceGateResult] = []
    for g in gates:
        table = candidate.get(g.slice)
        if table is None:
            raise ValueError(f"{MATRIX}: gate on {g.metric} names slice {g.slice!r}, which has no record")
        metric = table.get(g.metric)
        if metric is None:
            raise ValueError(
                f"{MATRIX}: gate names {g.metric}, which no run measures; one of {', '.join(table)}"
            )
        before = baseline.get(g.slice, {}).get(g.metric) if baseline else None
        if baseline is None and g.rule in ("no-drop", "no-rise"):
            out.append(
                SliceGateResult(
                    g, GateResult(g, None, metric, False, "no baseline run given; not compared"), skipped=True
                )
            )
            continue
        out.append(SliceGateResult(g, check(g, before, metric)))
    return out


# --- retrieval or generation? --------------------------------------------------------------------

Diagnosis = Literal["held", "retrieval", "generation", "both", "reading"]


def diagnose(c: CaseRecord, r: Record) -> tuple[Diagnosis, str]:
    """Where a record's failure sits: in retrieval (the named passages did not reach the model), in
    generation (they did, and the answer still failed), in both (an abstention expected, a near miss
    retrieved and answered), or in the reading (the checks accept the answer and the judge or the
    reader does not)."""
    if c.held:
        flagged = []
        if c.judgement is not None and (
            c.judgement.grounded < GROUNDED_FLOOR or c.judgement.fabricated >= FABRICATED_CEILING
        ):
            flagged.append("the judge reads it as ungrounded or fabricated")
        if c.reading is not None and c.reading.relevant < READ_FLOOR:
            flagged.append("the reader says it does not address the question")
        if c.reading is not None and c.reading.reference_agrees < READ_FLOOR:
            flagged.append("the reader says it disagrees with the reference")
        if flagged:
            return "reading", "; ".join(flagged)
        return "held", ""
    if c.expect == "abstain":
        if c.outcome == "answered":
            return "both", "a passage cleared the threshold and the model answered from it"
        return "generation", f"outcome {c.outcome}"
    if r.contexts and c.right_retrieved is False:
        return "retrieval", "none of the named passages reached the model"
    if c.abstained:
        return "generation", "abstained with a named passage in hand"
    return "generation", c.verdict


def diagnoses(run: Run, records: list[Record]) -> dict[str, tuple[Diagnosis, str]]:
    by_record = {r.id: r for r in records}
    return {c.id: diagnose(c, by_record[c.id]) for c in run.cases if c.id in by_record}
