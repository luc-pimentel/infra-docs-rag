"""The judge and the mechanical checks against hand labels: for each automatic signal, how often it
agrees with a person, where it does not, and at which threshold a probability is best read as a
verdict. Nothing here changes a label; the revised rules are what a person writes after reading the
disagreements, and the constants in `metrics.py` are set to match them."""

import statistics
from dataclasses import dataclass, field

from .labels import Label, LabelSet
from .metrics import CITED_FLOOR, FABRICATED_CEILING, GROUNDED_FLOOR
from .runs import CaseRecord, Run

SWEEP = [round(0.3 + 0.05 * i, 2) for i in range(13)]  # 0.30 … 0.90


@dataclass
class Agreement:
    """A yes/no signal against a yes/no label over the same answers."""

    tp: int = 0  # both say yes
    fp: int = 0  # the signal says yes, the person no
    fn: int = 0  # the signal says no, the person yes
    tn: int = 0  # both say no

    @property
    def n(self) -> int:
        return self.tp + self.fp + self.fn + self.tn

    @property
    def agreement(self) -> float:
        return (self.tp + self.tn) / self.n if self.n else 0.0

    @property
    def precision(self) -> float | None:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else None

    @property
    def recall(self) -> float | None:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else None

    @property
    def kappa(self) -> float | None:
        """Cohen's kappa: agreement beyond what the two base rates would give by chance."""
        if not self.n:
            return None
        yes_signal = (self.tp + self.fp) / self.n
        yes_person = (self.tp + self.fn) / self.n
        chance = yes_signal * yes_person + (1 - yes_signal) * (1 - yes_person)
        return (self.agreement - chance) / (1 - chance) if chance < 1 else None


@dataclass
class Disagreement:
    case: CaseRecord
    label: Label
    signal: bool  # what the automatic reading said
    person: bool  # what the person said
    reading: str  # the number behind the signal, as the report shows it


@dataclass
class Signal:
    """One automatic reading compared to one label axis."""

    name: str
    what: str  # the automatic reading
    against: str  # the label axis
    agreement: Agreement
    disagreements: list[Disagreement]
    sweep: list[tuple[float, Agreement]] = field(default_factory=list)  # for a probability: each threshold
    threshold: float | None = None  # the threshold the signal is read at

    @property
    def best(self) -> tuple[float, Agreement] | None:
        """The threshold with the most agreement; ties go to the one nearest the current threshold."""
        if not self.sweep:
            return None
        return max(self.sweep, key=lambda t: (t[1].agreement, -abs(t[0] - (self.threshold or 0.5))))


def tally(pairs: list[tuple[CaseRecord, Label, bool, bool, str]]) -> tuple[Agreement, list[Disagreement]]:
    a = Agreement()
    out: list[Disagreement] = []
    for case, label, signal, person, reading in pairs:
        if signal and person:
            a.tp += 1
        elif signal and not person:
            a.fp += 1
        elif not signal and person:
            a.fn += 1
        else:
            a.tn += 1
        if signal != person:
            out.append(Disagreement(case, label, signal, person, reading))
    return a, out


def probability_signal(
    name: str,
    what: str,
    against: str,
    rows: list[tuple[CaseRecord, Label, float, bool]],
    threshold: float,
    at_least: bool = True,
) -> Signal:
    """A judge probability read as yes at or over (or under, for a floor) a threshold, swept too."""

    def yes(p: float, t: float) -> bool:
        return p >= t if at_least else p < t

    agreement, disagreements = tally(
        [(c, x, yes(p, threshold), person, f"{p:.2f}") for c, x, p, person in rows]
    )
    sweep = [(t, tally([(c, x, yes(p, t), person, "") for c, x, p, person in rows])[0]) for t in SWEEP]
    return Signal(name, what, against, agreement, disagreements, sweep, threshold)


@dataclass
class Calibration:
    run: Run
    labels: LabelSet
    signals: list[Signal]
    cases: list[tuple[CaseRecord, Label]]

    @property
    def judged(self) -> list[tuple[CaseRecord, Label]]:
        return [(c, x) for c, x in self.cases if c.judgement is not None]

    def counts(self) -> dict[str, str]:
        """How the labels fall, per axis."""
        xs = [x for _, x in self.cases]
        n = len(xs)
        gap = [x for x in xs if x.gap_named is not None]
        return {
            "grounded": f"{sum(x.grounded for x in xs)}/{n} yes",
            "fabricated": f"{sum(x.fabricated for x in xs)}/{n} yes",
            "gap_named": f"{sum(bool(x.gap_named) for x in gap)}/{len(gap)} yes ({n - len(gap)} n/a)",
            "correct": ", ".join(f"{sum(x.correct == v for x in xs)} {v}" for v in ("yes", "partial", "no")),
            "citations": ", ".join(
                f"{sum(x.citations == v for x in xs)} {v}" for v in ("right", "wrong", "none")
            ),
        }


def calibrate(run: Run, labels: LabelSet) -> Calibration:
    by_id = run.by_id()
    cases = [(by_id[x.id], x) for x in labels.labels]
    judged = [(c, x) for c, x in cases if c.judgement is not None]
    signals = [
        probability_signal(
            "judge grounded",
            f"jev grounded ≥ {GROUNDED_FLOOR}",
            "grounded: every claim is in a passage",
            [(c, x, c.judgement.grounded, x.grounded) for c, x in judged],
            GROUNDED_FLOOR,
        ),
        probability_signal(
            "judge fabricated",
            f"jev fabricated ≥ {FABRICATED_CEILING}",
            "fabricated: a fact no passage gives",
            [(c, x, c.judgement.fabricated, x.fabricated) for c, x in judged],
            FABRICATED_CEILING,
        ),
        probability_signal(
            "judge gap named",
            "jev gap_named ≥ 0.5",
            "gap named: says what is not covered, or rejects the premise",
            [(c, x, c.judgement.gap_named, bool(x.gap_named)) for c, x in judged if x.gap_named is not None],
            0.5,
        ),
    ]
    mechanical: list[tuple[str, str, str, list[tuple[CaseRecord, Label, bool, bool, str]]]] = [
        (
            "cited share",
            f"cited share ≥ {CITED_FLOOR:.0%}",
            "grounded: every claim is in a passage",
            [(c, x, c.cited_share >= CITED_FLOOR, x.grounded, f"{c.cited_share:.0%}") for c, x in cases],
        ),
        (
            "fully cited",
            "no uncited sentence",
            "grounded: every claim is in a passage",
            [(c, x, not c.uncited, x.grounded, f"{len(c.uncited)} uncited") for c, x in cases],
        ),
        (
            "right cited",
            "a cited passage is a labelled right one",
            "citations: the cited passages are where the answer is",
            [
                (c, x, bool(c.right_cited), x.citations == "right", f"right cited {c.right_cited}")
                for c, x in cases
                if c.right_cited is not None
            ],
        ),
        (
            "verdict held",
            "the benchmark's verdict is `held`",
            "correct: the answer answers the question",
            [(c, x, c.held, x.correct == "yes", c.verdict) for c, x in cases],
        ),
    ]
    for name, what, against, rows in mechanical:
        agreement, disagreements = tally(rows)
        signals.append(Signal(name, what, against, agreement, disagreements))
    return Calibration(run, labels, signals, cases)


def judge_medians(cal: Calibration) -> dict[str, tuple[float, float]]:
    """The judge's median grounded reading on answers the person called grounded and on the rest, and
    the same for fabricated: how far apart the two groups sit."""
    out = {}
    for axis in ("grounded", "fabricated"):
        yes = [getattr(c.judgement, axis) for c, x in cal.judged if getattr(x, axis)]
        no = [getattr(c.judgement, axis) for c, x in cal.judged if not getattr(x, axis)]
        out[axis] = (
            statistics.median(yes) if yes else float("nan"),
            statistics.median(no) if no else float("nan"),
        )
    return out
