"""Regression gates in `eval/gates.yaml`: each names a metric and the rule a candidate run has to clear
against a baseline run, or against a fixed value. A critical gate that fails blocks the change; one
that is not critical is a warning. `compare` runs every gate over two saved runs and lists the
questions whose outcome changed, so a failed gate comes with the answers behind it."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from .metrics import Metric, metrics
from .runs import CaseRecord, Run

GATES = Path("eval/gates.yaml")
Rule = Literal["min", "max", "no-drop", "no-rise"]


class Gate(BaseModel):
    metric: str
    rule: Rule
    value: float | None = None  # min / max: the bound the candidate has to meet
    tolerance: float = 0.0  # no-drop / no-rise: how far past the baseline the candidate may go
    relative: bool = False  # tolerance as a share of the baseline's value, instead of in its units
    critical: bool = True  # a failure blocks; otherwise it warns
    why: str = ""

    @model_validator(mode="after")
    def complete(self) -> "Gate":
        if self.rule in ("min", "max") and self.value is None:
            raise ValueError(f"gate on {self.metric}: rule {self.rule} needs a value")
        if self.rule in ("no-drop", "no-rise") and self.value is not None:
            raise ValueError(f"gate on {self.metric}: rule {self.rule} compares to the baseline, not a value")
        if self.tolerance < 0:
            raise ValueError(f"gate on {self.metric}: tolerance cannot be negative")
        return self

    @property
    def label(self) -> str:
        if self.rule == "min":
            return f"≥ {self.value:g}"
        if self.rule == "max":
            return f"≤ {self.value:g}"
        slack = (
            ""
            if not self.tolerance
            else f" by more than {self.tolerance:.0%} of the baseline"
            if self.relative
            else f" by more than {self.tolerance:g}"
        )
        return ("must not drop" if self.rule == "no-drop" else "must not rise") + slack


class GateSet(BaseModel):
    gates: list[Gate]


def load_gates(path: Path = GATES) -> list[Gate]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        loaded = GateSet.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    return loaded.gates


@dataclass
class GateResult:
    gate: Gate
    baseline: Metric | None
    candidate: Metric
    passed: bool
    detail: str

    @property
    def blocks(self) -> bool:
        return not self.passed and self.gate.critical


def check(gate: Gate, baseline: Metric | None, candidate: Metric) -> GateResult:
    value = candidate.value
    if value is None:
        return GateResult(gate, baseline, candidate, False, "nothing in the candidate run measures this")
    if gate.rule == "min":
        return GateResult(
            gate, baseline, candidate, value >= gate.value, f"{candidate.shown} against ≥ {gate.value:g}"
        )
    if gate.rule == "max":
        return GateResult(
            gate, baseline, candidate, value <= gate.value, f"{candidate.shown} against ≤ {gate.value:g}"
        )
    if baseline is None or baseline.value is None:
        return GateResult(gate, baseline, candidate, False, "the baseline run has nothing to compare against")
    slack = gate.tolerance * abs(baseline.value) if gate.relative else gate.tolerance
    if gate.rule == "no-drop":
        passed = value >= baseline.value - slack
    else:
        passed = value <= baseline.value + slack
    return GateResult(
        gate,
        baseline,
        candidate,
        passed,
        f"{baseline.shown} → {candidate.shown}, {delta(baseline, candidate)}",
    )


def delta(baseline: Metric, candidate: Metric) -> str:
    """The change from baseline to candidate, in the metric's own terms."""
    if baseline.value is None or candidate.value is None:
        return "–"
    d = candidate.value - baseline.value
    if baseline.shown.startswith("$"):
        return f"{d:+.2f} $"
    if baseline.shown.endswith("ms"):
        return f"{d:+,.0f} ms"
    if "%" in baseline.shown or (0 <= baseline.value <= 1 and 0 <= candidate.value <= 1):
        return f"{d:+.0%}" if "(" in baseline.shown or "%" in baseline.shown else f"{d:+.2f}"
    return f"{d:+g}"


@dataclass
class Changed:
    """A question whose result differs between the two runs."""

    id: str
    before: CaseRecord
    after: CaseRecord

    @property
    def what(self) -> list[str]:
        out = []
        if self.before.outcome != self.after.outcome:
            out.append(f"outcome {self.before.outcome} → {self.after.outcome}")
        if self.before.verdict != self.after.verdict:
            out.append(f"verdict {self.before.verdict} → {self.after.verdict}")
        if self.before.right_retrieved != self.after.right_retrieved:
            out.append(
                f"right retrieved {yn(self.before.right_retrieved)} → {yn(self.after.right_retrieved)}"
            )
        if self.before.right_cited != self.after.right_cited:
            out.append(f"right cited {yn(self.before.right_cited)} → {yn(self.after.right_cited)}")
        return out

    @property
    def better(self) -> bool | None:
        """Whether the change is for the better, when held says so; None when both or neither held."""
        if self.before.held == self.after.held:
            return None
        return self.after.held


def yn(value: bool | None) -> str:
    return "–" if value is None else ("yes" if value else "no")


@dataclass
class Comparison:
    baseline: Run
    candidate: Run
    before: dict[str, Metric]
    after: dict[str, Metric]
    results: list[GateResult]
    changed: list[Changed]

    @property
    def blocked(self) -> bool:
        return any(r.blocks for r in self.results)

    @property
    def failed(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed]

    @property
    def warnings(self) -> list[GateResult]:
        return [r for r in self.results if not r.passed and not r.gate.critical]

    @property
    def verdict(self) -> str:
        if self.blocked:
            return f"blocked: {len([r for r in self.results if r.blocks])} critical gate(s) failed"
        if self.warnings:
            return f"passes with {len(self.warnings)} warning(s)"
        return "passes"


def compare(baseline: Run, candidate: Run, gates: list[Gate]) -> Comparison:
    before, after = metrics(baseline), metrics(candidate)
    unknown = [g.metric for g in gates if g.metric not in after]
    if unknown:
        raise ValueError(f"{GATES} names metrics no run has: {', '.join(unknown)}; one of {', '.join(after)}")
    results = [check(g, before.get(g.metric), after[g.metric]) for g in gates]
    earlier = baseline.by_id()
    changed = [
        Changed(c.id, earlier[c.id], c)
        for c in candidate.cases
        if c.id in earlier
        and (
            earlier[c.id].outcome != c.outcome
            or earlier[c.id].verdict != c.verdict
            or earlier[c.id].right_retrieved != c.right_retrieved
            or earlier[c.id].right_cited != c.right_cited
        )
    ]
    return Comparison(baseline, candidate, before, after, results, changed)
