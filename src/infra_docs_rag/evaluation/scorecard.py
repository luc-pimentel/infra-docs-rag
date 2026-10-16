"""The evaluation system as an operations workflow (stage 22): one view of quality, regressions,
cost, latency and safety over time. `scorecard` reads everything the repository holds (the saved
runs, the releases, the baseline pin, the freeze, the safety runs, the online windows, the gates)
and writes the latest status of every component beside the trend of every core metric, release by
release and dataset run by dataset run, as `reports/scorecard.md` and `eval/scorecard.json`. No
model call: it is what the nightly job regenerates and what a person opens first."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from ..pipeline.releases import Releases
from .dataset import Record, dataset_version
from .efficiency import budget_status, budgets
from .experiment import Frozen
from .gates import Gate
from .metrics import Metric, metrics
from .online import Window, baseline_rates, drift
from .regression import Baseline, compare
from .runs import Run
from .safety import SafetyPolicy, safety_for
from .slices import MatrixFile, slice_metrics

SCORECARD_REPORT = Path("reports/scorecard.md")
SCORECARD_DATA = Path("eval/scorecard.json")
State = Literal["ok", "warn", "fail", "none"]

# The core metrics, in the order the scorecard shows them: name, label, how to print.
CORE = [
    ("answers.held", "held", "pct"),
    ("retrieval.right_retrieved", "right passage read", "pct"),
    ("retrieval.context_recall", "context recall", "pct"),
    ("grounding.judge_grounded", "grounded (judge)", "pct"),
    ("grounding.judge_fabricated", "fabricated (judge)", "pct"),
    ("abstention.in_scope", "false abstention", "pct"),
    ("abstention.out_of_scope", "out-of-scope abstained", "pct"),
    ("cost.usd_per_request", "cost / request", "usd"),
    ("latency.end_to_end_ms_p95", "p95", "ms"),
]
# The critical slices, followed on the dataset runs.
SLICES = (
    "category=adversarial",
    "category=no-answer",
    "split=test",
    "difficulty=hard",
    "category=fact-lookup",
)
BARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float | None]) -> str:
    """A trend in eight heights; a missing value is a space. Flat series draw at the middle."""
    known = [v for v in values if v is not None]
    if not known:
        return ""
    lo, hi = min(known), max(known)
    out = []
    for v in values:
        if v is None:
            out.append(" ")
        elif hi == lo:
            out.append(BARS[3])
        else:
            out.append(BARS[round((v - lo) / (hi - lo) * (len(BARS) - 1))])
    return "".join(out)


def shown(value: float | None, how: str) -> str:
    if value is None:
        return "–"
    if how == "pct":
        return f"{value:.0%}"
    if how == "usd":
        return f"${value:.4f}"
    if how == "ms":
        return f"{value:,.0f} ms"
    return f"{value:.2f}"


class Status(BaseModel):
    component: str
    state: State
    detail: str


class Point(BaseModel):
    run: str
    version: str
    date: str
    values: dict[str, float | None]  # metric → value


class Series(BaseModel):
    name: str  # releases | dataset
    points: list[Point]

    def trend(self, metric: str) -> list[float | None]:
        """The values in order, a share rounded to the percent the page shows, so a bar moves only
        when the printed number does."""
        out = []
        for p in self.points:
            v = p.values.get(metric)
            out.append(None if v is None else round(v, 2) if abs(v) <= 1 else v)
        return out


class SliceTrend(BaseModel):
    slice: str
    values: list[float | None]  # answers.held per dataset run, in the dataset series' order


class WindowRow(BaseModel):
    window: str
    live: int
    sampled: int
    failed: int
    drifted: list[str]


class Scorecard(BaseModel):
    statuses: list[Status]
    series: list[Series]
    slices: list[SliceTrend]
    windows: list[WindowRow]

    @property
    def worst(self) -> State:
        order = {"fail": 3, "warn": 2, "none": 1, "ok": 0}
        return max((s.state for s in self.statuses), key=lambda s: order[s], default="ok")

    def save(self, path: Path = SCORECARD_DATA) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path


@dataclass
class Inputs:
    """Everything the scorecard reads, loaded by the caller."""

    runs: dict[str, Run]
    releases: Releases
    baseline: Baseline | None
    frozen: Frozen | None
    frozen_now: Frozen | None  # the freeze as the files are now, to compare
    records: list[Record]
    matrix: MatrixFile
    gates: list[Gate]
    safety_policy: SafetyPolicy | None
    windows: list[Window]
    candidate: str | None = None  # the dataset run the committed comparison reads


def statuses(x: Inputs) -> list[Status]:
    out: list[Status] = []
    live = x.releases.live
    if live is None:
        out.append(Status(component="release", state="none", detail="no release yet"))
    else:
        out.append(
            Status(
                component="release",
                state="ok",
                detail=f"{live.name}: `{live.manifest.version}` on run `{live.run}`, {live.created_at[:10]}",
            )
        )
    # the baseline pin
    if x.baseline is None:
        out.append(Status(component="baseline", state="none", detail="no pin; `dataset baseline set`"))
    elif x.baseline.run not in x.runs:
        out.append(
            Status(
                component="baseline",
                state="fail",
                detail=f"the pin names `{x.baseline.run}`, which is not saved",
            )
        )
    else:
        problems = x.baseline.problems(x.runs[x.baseline.run], x.records)
        out.append(
            Status(
                component="baseline",
                state="fail" if problems else "ok",
                detail=("stale: " + "; ".join(problems))
                if problems
                else f"`{x.baseline.run}`, dataset {x.baseline.dataset}",
            )
        )
    # the freeze
    if x.frozen is None or x.frozen_now is None:
        out.append(Status(component="freeze", state="none", detail="no freeze; `dataset freeze`"))
    else:
        moved = x.frozen.problems(x.frozen_now)
        out.append(
            Status(
                component="freeze",
                state="fail" if moved else "ok",
                detail=("moved: " + "; ".join(moved)) if moved else f"holds since {x.frozen.frozen_at[:10]}",
            )
        )
    # the quality gates: the committed comparison
    if x.baseline and x.candidate and x.candidate in x.runs and x.baseline.run in x.runs:
        safety = None
        if x.safety_policy:
            found = safety_for(x.runs[x.candidate].manifest.version)
            if found:
                safety = (found[0].name, found[1], found[1].verdict(x.safety_policy.thresholds))
        try:
            reg = compare(
                x.runs[x.baseline.run],
                x.runs[x.candidate],
                x.records,
                x.matrix,
                allow_confounded=True,
                safety=safety,
            )
            state: State = "fail" if reg.decision == "block" else "warn" if reg.decision == "hold" else "ok"
            out.append(
                Status(
                    component="quality gates",
                    state=state,
                    detail=f"`{x.candidate}` against the baseline: {reg.decision}; {reg.verdict.split(';')[0]}",
                )
            )
        except ValueError as error:
            out.append(Status(component="quality gates", state="fail", detail=str(error)[:160]))
    else:
        out.append(
            Status(component="quality gates", state="none", detail="no candidate dataset run to compare")
        )
    # safety on the release's version
    if x.safety_policy is None:
        out.append(Status(component="safety", state="none", detail="no policy"))
    elif live is None:
        out.append(Status(component="safety", state="none", detail="no release to grade"))
    else:
        found = safety_for(live.manifest.version)
        if found is None:
            out.append(
                Status(
                    component="safety",
                    state="warn",
                    detail=f"no graded safety run for `{live.manifest.version}`",
                )
            )
        else:
            v = found[1].verdict(x.safety_policy.thresholds)
            counts = found[1].counts()
            out.append(
                Status(
                    component="safety",
                    state="fail" if v.blocked else "warn" if v.warnings else "ok",
                    detail=f"`{found[0].name}`: {counts['safe-pass']} pass, {counts['safe-refusal']} refusal, {counts['unsafe-disclosure']} disclosure, {counts['unsupported-claim']} unsupported"
                    + ("; " + "; ".join(v.reasons + v.warnings) if v.reasons or v.warnings else ""),
                )
            )
    # budgets on the release's run
    if live is not None and live.run in x.runs:
        broken = []
        for b in budgets(x.gates, x.matrix.gates):
            if b.where != "release":
                continue
            passed, value = budget_status(b, x.runs[live.run])
            if passed is False:
                broken.append(f"{b.gate.metric} {value} against {b.gate.label}")
        out.append(
            Status(
                component="budgets",
                state="fail" if broken else "ok",
                detail=("; ".join(broken)) if broken else "the release is within every budget",
            )
        )
    # online drift on the latest window
    if not x.windows:
        out.append(Status(component="online", state="none", detail="no window sampled"))
    elif x.baseline and x.baseline.run in x.runs:
        w = x.windows[-1]
        d = drift(w.rates, baseline_rates(x.runs[x.baseline.run]), w.policy)
        flagged = [r.name for r in d if r.drifted]
        failed = sum(1 for s in w.samples if s.failed)
        out.append(
            Status(
                component="online",
                state="warn" if flagged else "ok",
                detail=f"`{w.name}`: {len(w.samples)} of {w.live} live traces sampled, {failed} failed"
                + (f"; drift in {', '.join(flagged)}" if flagged else "; no drift"),
            )
        )
    return out


def series_of(x: Inputs) -> list[Series]:
    out: list[Series] = []
    order: list[str] = []
    for r in x.releases.releases:
        if r.run in x.runs and r.run not in order:
            order.append(r.run)
    out.append(Series(name="releases", points=[point(x.runs[n]) for n in order]))
    dataset_runs = sorted(
        (r for r in x.runs.values() if r.manifest.dataset and not r.name.startswith("safety-")),
        key=lambda r: r.manifest.created_at,
    )
    out.append(Series(name="dataset", points=[point(r) for r in dataset_runs]))
    return out


def point(run: Run) -> Point:
    m = metrics(run)
    return Point(
        run=run.name,
        version=run.manifest.version,
        date=run.manifest.created_at[:10],
        values={name: (m[name].value if name in m else None) for name, _, _ in CORE},
    )


def slice_trends(x: Inputs, dataset_series: Series) -> list[SliceTrend]:
    tables = {}
    for p in dataset_series.points:
        run = x.runs[p.run]
        known = {r.id for r in x.records} & {c.id for c in run.cases}
        rows = [r for r in x.records if r.id in known]
        tables[p.run] = slice_metrics(run, rows)
    out = []
    for s in SLICES:
        values = []
        for p in dataset_series.points:
            m: Metric | None = tables[p.run].get(s, {}).get("answers.held")
            values.append(m.value if m else None)
        out.append(SliceTrend(slice=s, values=values))
    return out


def window_rows(x: Inputs) -> list[WindowRow]:
    rows = []
    base = x.runs.get(x.baseline.run) if x.baseline else None
    for w in x.windows:
        flagged: list[str] = []
        if base is not None:
            flagged = [r.name for r in drift(w.rates, baseline_rates(base), w.policy) if r.drifted]
        rows.append(
            WindowRow(
                window=w.name,
                live=w.live,
                sampled=len(w.samples),
                failed=sum(1 for s in w.samples if s.failed),
                drifted=flagged,
            )
        )
    return rows


def build(x: Inputs) -> Scorecard:
    series = series_of(x)
    return Scorecard(
        statuses=statuses(x),
        series=series,
        slices=slice_trends(x, series[1]),
        windows=window_rows(x),
    )


def frozen_now(records: list[Record], grader: str, judge: str | None) -> Frozen:
    return Frozen.now(dataset_version(records), grader, judge)
