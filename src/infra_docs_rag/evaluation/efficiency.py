"""Cost and latency as quality metrics (stage 20). A change that lifts the score and doubles the bill
or the wait is not a win; every experiment reports what it held, what it spent and how long it took,
side by side, and a change is judged on all three. The budgets are gates (`eval/gates.yaml` for the
release, `eval/matrix.yaml` for a dataset run): the most a request may cost, the longest the 95th
percentile may take, how often a replay has to hit the cache. The tradeoff rule turns a before/after
pair into a verdict: worth it, free, not worth it, a loss."""

import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..pipeline.releases import Releases
from .gates import Gate
from .metrics import Metric, metrics, percentile
from .runs import Run
from .slices import SliceGate

EFFICIENCY_REPORT = Path("reports/efficiency.md")
BUDGET_METRICS = ("cost.", "latency.", "cache.")

# The tradeoff rule: what a change may cost for what it buys.
QUALITY_RECORDS = 2  # held records a change has to buy to count as a quality gain; one is noise
COST_RISE = 0.25  # the most a request may cost more, as a share of the baseline's, for a quality gain
LATENCY_RISE = 0.25  # the same for the 95th percentile
FREE_DROP = 0.05  # a cost or latency drop this large, at the same quality, is a free win


@dataclass
class Numbers:
    """One run's quality, spend and speed."""

    run: str
    version: str
    kind: str  # benchmark | dataset
    n: int
    held: int
    held_share: float
    calls: int
    cached: int
    tokens_in: int
    tokens_out: int
    cost_total: float
    cost_per_request: float
    judge_usd: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    generator: str
    backend: str

    @property
    def tokens(self) -> int:
        return self.tokens_in + self.tokens_out

    @property
    def cache_hit_share(self) -> float | None:
        return self.cached / self.calls if self.calls and self.cached else None

    @property
    def held_per_dollar(self) -> float | None:
        return self.held / self.cost_total if self.cost_total else None

    @property
    def held_per_minute_p95(self) -> float:
        """Records held per minute of 95th-percentile wait: what the quality costs in time."""
        return self.held / (self.p95_ms / 60000) if self.p95_ms else 0.0


def numbers_of(run: Run) -> Numbers:
    m = metrics(run)
    cases = run.cases
    called = [c for c in cases if c.called]
    e2e = [c.end_to_end_ms for c in cases]
    held = m["answers.held"]
    return Numbers(
        run=run.name,
        version=run.manifest.version,
        kind="dataset" if run.manifest.dataset else "benchmark",
        n=len(cases),
        held=int(round((held.value or 0.0) * len(cases))),
        held_share=held.value or 0.0,
        calls=len(called),
        cached=sum(1 for c in called if c.generated.cached),
        tokens_in=sum(c.generated.usage.input_tokens for c in called),
        tokens_out=sum(c.generated.usage.output_tokens for c in called),
        cost_total=m["cost.usd_total"].value or 0.0,
        cost_per_request=m["cost.usd_per_request"].value or 0.0,
        judge_usd=m["cost.judge_usd"].value or 0.0,
        mean_ms=statistics.mean(e2e) if e2e else 0.0,
        p50_ms=m["latency.end_to_end_ms_median"].value or 0.0,
        p95_ms=percentile(e2e, 0.95) or 0.0,
        generator=run.manifest.generator,
        backend=run.manifest.backend,
    )


Verdict = Literal["worth it", "free", "not worth it", "a loss"]


@dataclass
class Tradeoff:
    baseline: Numbers
    candidate: Numbers
    verdict: Verdict
    why: str

    @property
    def held_delta(self) -> int:
        return self.candidate.held - self.baseline.held

    @property
    def cost_delta(self) -> float | None:
        b = self.baseline.cost_per_request
        return (self.candidate.cost_per_request - b) / b if b else None

    @property
    def p95_delta(self) -> float | None:
        b = self.baseline.p95_ms
        return (self.candidate.p95_ms - b) / b if b else None


def share(x: float | None) -> str:
    return "–" if x is None else f"{x:+.0%}"


def tradeoff(b: Numbers, c: Numbers) -> Tradeoff:
    """The verdict on a before/after pair, by the rule in the constants above."""
    held = c.held - b.held
    cost = (c.cost_per_request - b.cost_per_request) / b.cost_per_request if b.cost_per_request else 0.0
    p95 = (c.p95_ms - b.p95_ms) / b.p95_ms if b.p95_ms else 0.0
    over = cost > COST_RISE or p95 > LATENCY_RISE
    cheaper = cost <= -FREE_DROP or p95 <= -FREE_DROP
    if held <= -QUALITY_RECORDS:
        return Tradeoff(b, c, "a loss", f"{-held} fewer records held; nothing saved makes that up")
    if held >= QUALITY_RECORDS:
        if over:
            return Tradeoff(
                b,
                c,
                "not worth it",
                f"buys {held} records at {share(cost)} cost and {share(p95)} p95; the rule allows {COST_RISE:.0%} and {LATENCY_RISE:.0%}",
            )
        return Tradeoff(
            b,
            c,
            "worth it",
            f"buys {held} records at {share(cost)} cost and {share(p95)} p95, within the rule",
        )
    # quality flat
    if cheaper and not over:
        return Tradeoff(
            b,
            c,
            "free",
            f"same quality within {QUALITY_RECORDS} records, {share(cost)} cost and {share(p95)} p95",
        )
    if cost > 0 or p95 > 0:
        return Tradeoff(
            b,
            c,
            "not worth it",
            f"nothing bought: {held:+d} records at {share(cost)} cost and {share(p95)} p95",
        )
    return Tradeoff(
        b, c, "free", f"same quality, same spend and speed ({share(cost)} cost, {share(p95)} p95)"
    )


def release_pairs(runs: dict[str, Run], releases: Releases) -> list[tuple[Run, Run]]:
    """The benchmark runs in release order, each against the one before, a run compared once."""
    order: list[str] = []
    for r in releases.releases:
        if r.run in runs and r.run not in order:
            order.append(r.run)
    return [(runs[a], runs[b]) for a, b in zip(order, order[1:], strict=False)]


def dataset_pairs(runs: dict[str, Run], baseline: str) -> list[tuple[Run, Run]]:
    """The pinned baseline against every other dataset run of the same dataset version."""
    base = runs.get(baseline)
    if base is None:
        return []
    return [
        (base, r)
        for name, r in sorted(runs.items())
        if name != baseline and r.manifest.dataset and r.manifest.dataset == base.manifest.dataset
    ]


def replay_pairs(runs: dict[str, Run]) -> list[tuple[Run, Run]]:
    """A run against its replay through the cache, when both are saved."""
    return [(runs[n], runs[n + "-replay"]) for n in sorted(runs) if n + "-replay" in runs]


@dataclass
class Budget:
    """A cost, latency or cache gate and where it applies."""

    where: str  # release | dataset
    gate: Gate
    slice: str | None = None

    @property
    def label(self) -> str:
        return self.gate.label + (" (skipped when not measured)" if self.gate.skip_unmeasured else "")


def budgets(gates: list[Gate], slice_gates: list[SliceGate]) -> list[Budget]:
    out = [Budget("release", g) for g in gates if g.metric.startswith(BUDGET_METRICS)]
    out += [Budget("dataset", g, g.slice) for g in slice_gates if g.metric.startswith(BUDGET_METRICS)]
    return out


def budget_status(b: Budget, run: Run) -> tuple[bool | None, str]:
    """Whether a run is within a budget: (passed, the value shown). None when not measured."""
    m: Metric | None = metrics(run).get(b.gate.metric)
    if m is None or m.value is None:
        return None, "–"
    if b.gate.rule == "min":
        return m.value >= (b.gate.value or 0.0), m.shown
    if b.gate.rule == "max":
        return m.value <= (b.gate.value or 0.0), m.shown
    return None, m.shown  # a delta rule needs a baseline; not a budget
