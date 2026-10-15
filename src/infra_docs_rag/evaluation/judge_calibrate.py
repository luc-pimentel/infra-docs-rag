"""The rubric grader, the probability judge and the dataset reader against a person's scores on the
rubric: for each dimension, how often the grader gives the same score, how far off it is when it does
not, whether it passes and fails the same answers at the operating threshold, and which threshold would
agree most. The scores come from `eval/calibration.yaml`, read twice by two independent readers and
resolved; the grades from `eval/runs/<run>/grades.json`. Nothing here changes a label."""

from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from .calibrate import SWEEP, Agreement
from .rubric import SCORES, Grade, GradeSet, Rubric, grader_state, passages_of
from .runs import CaseRecord, JudgementRecord, ReadingRecord, Run

CALIBRATION = Path("eval/calibration.yaml")
JUDGE_REPORT = Path("reports/judge.md")
Scores = dict[str, int | None]


class Reading(BaseModel):
    """One reader's scores on an answer, by dimension."""

    groundedness: int | None
    completeness: int | None
    abstention: int | None

    def of(self, dimension: str) -> int | None:
        return getattr(self, dimension)


class CalibrationLabel(BaseModel):
    id: str
    base: str | None = None  # a perturbed answer: the run case whose passages and question it keeps
    perturbation: str = ""  # what was changed, for a perturbed answer
    answer: str | None = None  # the perturbed answer text, with its [n] markers
    groundedness: int | None
    completeness: int | None
    abstention: int | None
    a: Reading
    b: Reading | None = None
    note: str = ""
    resolution: str = ""

    @property
    def perturbed(self) -> bool:
        return self.base is not None

    def final(self, dimension: str) -> int | None:
        return getattr(self, dimension)

    def disputed(self) -> list[str]:
        """The dimensions where the two readers differed."""
        if self.b is None:
            return []
        return [d for d in ("groundedness", "completeness", "abstention") if self.a.of(d) != self.b.of(d)]

    @model_validator(mode="after")
    def consistent(self) -> "CalibrationLabel":
        if (self.base is None) != (self.answer is None):
            raise ValueError(f"{self.id}: a perturbed answer needs both `base` and `answer`")
        if self.base is not None and not self.perturbation:
            raise ValueError(f"{self.id}: a perturbed answer needs `perturbation`, what was changed")
        for d in ("groundedness", "completeness", "abstention"):
            v = self.final(d)
            if v is not None and v not in SCORES:
                raise ValueError(f"{self.id}: {d} is {v}, not one of {SCORES} or null")
        if self.disputed() and not self.resolution.strip():
            raise ValueError(
                f"{self.id}: the readers differ on {', '.join(self.disputed())} and `resolution` is empty"
            )
        return self


class CalibrationSet(BaseModel):
    run: str
    rubric: str = ""
    readers: dict[str, str]
    resolved_by: str = ""
    labels: list[CalibrationLabel]

    @property
    def real(self) -> list[CalibrationLabel]:
        return [x for x in self.labels if not x.perturbed]

    @property
    def perturbed(self) -> list[CalibrationLabel]:
        return [x for x in self.labels if x.perturbed]


def load_calibration(path: Path, run: Run, rubric: Rubric) -> CalibrationSet:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        loaded = CalibrationSet.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    if loaded.run != run.name:
        raise ValueError(f"{path} labels run {loaded.run!r}, not {run.name!r}")
    ids = [x.id for x in loaded.labels]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: an answer is labelled twice")
    cases = run.by_id()
    for x in loaded.labels:
        c = cases.get(x.base or x.id)
        if c is None:
            raise ValueError(f"{path}: {x.base or x.id} is not in run {run.name}")
        if x.perturbed and not x.id.startswith(x.base + "~"):
            raise ValueError(f"{path}: perturbed id {x.id!r} should start with {x.base + '~'!r}")
        for d in rubric.ids:
            if x.final(d) is None and d == "abstention":
                raise ValueError(f"{path}: {x.id}: abstention always applies and cannot be null")
    return loaded


# ---- what the grader reads for each label -----------------------------------------------------


@dataclass
class Item:
    """One labelled answer with the case it comes from and the text that was scored."""

    label: CalibrationLabel
    case: CaseRecord
    answer: str  # the marked answer, or the abstention text, or the perturbed answer
    reference: str
    judgement: JudgementRecord | None = None  # the probability judge's reading of this answer
    reading: ReadingRecord | None = None  # the dataset reader's reading of this answer

    @property
    def id(self) -> str:
        return self.label.id

    @property
    def abstained(self) -> bool:
        return self.label.perturbed is False and self.case.abstained

    def state(self) -> str:
        return grader_state(passages_of(self.case), self.case.question, self.reference, self.answer)


def items_of(cal: CalibrationSet, run: Run, references: dict[str, str]) -> list[Item]:
    """The labelled answers as the grader sees them. `references` maps a case id to its reference
    answer (from the dataset records). A perturbed label keeps its base case's passages, question and
    reference and swaps in its own answer; its judge and reader probabilities come from `grades.json`,
    a real one's from the run."""
    cases = run.by_id()
    out: list[Item] = []
    for x in cal.labels:
        c = cases[x.base or x.id]
        if x.perturbed:
            out.append(Item(x, c, x.answer or "", references.get(c.id, "")))
        else:
            text = c.marked if c.answered else (c.text or " ".join(c.gap_sentences) or c.reason)
            out.append(Item(x, c, text, references.get(c.id, ""), c.judgement, c.reading))
    return out


# ---- agreement on an ordinal score --------------------------------------------------------------


def weighted_kappa(pairs: list[tuple[int, int]]) -> float | None:
    """Cohen's kappa with quadratic weights over scores 0 to 2: 1 is perfect agreement, 0 is what the
    two marginals give by chance, and a disagreement by two costs four times one by one."""
    n = len(pairs)
    if n == 0:
        return None
    k = len(SCORES)
    observed = [[0.0] * k for _ in SCORES]
    for a, b in pairs:
        observed[a][b] += 1
    row = [sum(observed[i]) for i in range(k)]
    col = [sum(observed[i][j] for i in range(k)) for j in range(k)]
    weight = [[((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]
    disagreement = sum(weight[i][j] * observed[i][j] for i in range(k) for j in range(k)) / n
    chance = sum(weight[i][j] * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    return None if chance == 0 else 1 - disagreement / chance


@dataclass
class Confusion:
    """Scores by the grader (rows) against the person (columns), 0 to 2, plus the n/a mismatches."""

    counts: list[list[int]] = field(default_factory=lambda: [[0] * 3 for _ in range(3)])
    grader_na_person_scored: int = 0
    grader_scored_person_na: int = 0
    both_na: int = 0

    @property
    def n(self) -> int:
        return sum(map(sum, self.counts))

    @property
    def exact(self) -> int:
        return sum(self.counts[i][i] for i in range(3))

    @property
    def off_by_two(self) -> int:
        return self.counts[0][2] + self.counts[2][0]


@dataclass
class ScoreDisagreement:
    item: Item
    grader: int | None
    person: int | None
    reason: str  # what the grader said


@dataclass
class DimensionResult:
    """The grader against the person on one dimension."""

    dimension: str
    pass_at: int
    confusion: Confusion
    kappa: float | None
    pass_agreement: Agreement  # at the operating threshold
    pass_sweep: list[tuple[int, Agreement]]  # at pass_at 1 and 2
    disagreements: list[ScoreDisagreement]
    pairs: list[tuple[Item, int | None, int | None]]

    @property
    def exact_share(self) -> float:
        return self.confusion.exact / self.confusion.n if self.confusion.n else 0.0

    @property
    def best_pass_at(self) -> tuple[int, Agreement] | None:
        if not self.pass_sweep:
            return None
        return max(self.pass_sweep, key=lambda t: (t[1].agreement, -abs(t[0] - self.pass_at)))


def tally_passes(rows: list[tuple[bool, bool]]) -> Agreement:
    a = Agreement()
    for grader, person in rows:
        if grader and person:
            a.tp += 1
        elif grader and not person:
            a.fp += 1
        elif not grader and person:
            a.fn += 1
        else:
            a.tn += 1
    return a


def dimension_result(
    dimension: str, rubric: Rubric, pairs: list[tuple[Item, Grade | None]]
) -> DimensionResult:
    pass_at = rubric.pass_at(dimension)
    conf = Confusion()
    scored: list[tuple[int, int]] = []
    disagreements: list[ScoreDisagreement] = []
    triples: list[tuple[Item, int | None, int | None]] = []
    for item, grade in pairs:
        if grade is None:
            continue
        g, p = grade.score(dimension), item.label.final(dimension)
        triples.append((item, g, p))
        reason = grade.scores[dimension].reason if dimension in grade.scores else ""
        if g is None and p is None:
            conf.both_na += 1
            continue
        if g is None or p is None:
            if g is None:
                conf.grader_na_person_scored += 1
            else:
                conf.grader_scored_person_na += 1
            disagreements.append(ScoreDisagreement(item, g, p, reason))
            continue
        conf.counts[g][p] += 1
        scored.append((g, p))
        if g != p:
            disagreements.append(ScoreDisagreement(item, g, p, reason))
    sweep = [(t, tally_passes([(g >= t, p >= t) for g, p in scored])) for t in (1, 2)]
    agreement = dict(sweep)[pass_at]
    return DimensionResult(
        dimension, pass_at, conf, weighted_kappa(scored), agreement, sweep, disagreements, triples
    )


# ---- the probability signals against the same scores ------------------------------------------


@dataclass
class ProbabilitySignal:
    """A jev probability read as yes at a threshold, against a yes/no reading of the person's score."""

    name: str
    what: str
    against: str
    threshold: float
    at_least: bool
    agreement: Agreement
    sweep: list[tuple[float, Agreement]]
    disagreements: list[tuple[Item, float, bool]]  # item, probability, what the person said

    @property
    def best(self) -> tuple[float, Agreement] | None:
        if not self.sweep:
            return None
        return max(self.sweep, key=lambda t: (t[1].agreement, -abs(t[0] - self.threshold)))


def probability_signal(
    name: str,
    what: str,
    against: str,
    rows: list[tuple[Item, float, bool]],
    threshold: float,
    at_least: bool = True,
) -> ProbabilitySignal:
    def yes(p: float, t: float) -> bool:
        return p >= t if at_least else p < t

    def at(t: float) -> Agreement:
        a = Agreement()
        for _, p, person in rows:
            s = yes(p, t)
            if s and person:
                a.tp += 1
            elif s and not person:
                a.fp += 1
            elif not s and person:
                a.fn += 1
            else:
                a.tn += 1
        return a

    disagreements = [(i, p, person) for i, p, person in rows if yes(p, threshold) != person]
    return ProbabilitySignal(
        name, what, against, threshold, at_least, at(threshold), [(t, at(t)) for t in SWEEP], disagreements
    )


# ---- the two readers against each other ----------------------------------------------------------


@dataclass
class ReaderAgreement:
    dimension: str
    n: int  # answers both readers scored
    exact: int
    kappa: float | None
    disputed: list[CalibrationLabel]


def reader_agreement(cal: CalibrationSet, dimension: str) -> ReaderAgreement:
    pairs: list[tuple[int, int]] = []
    n = exact = 0
    disputed: list[CalibrationLabel] = []
    for x in cal.labels:
        if x.b is None:
            continue
        a, b = x.a.of(dimension), x.b.of(dimension)
        if a is None and b is None:
            continue
        n += 1
        if a == b:
            exact += 1
        else:
            disputed.append(x)
        if a is not None and b is not None:
            pairs.append((a, b))
    return ReaderAgreement(dimension, n, exact, weighted_kappa(pairs), disputed)


# ---- the whole calibration -----------------------------------------------------------------------


@dataclass
class JudgeCalibration:
    run: Run
    rubric: Rubric
    cal: CalibrationSet
    grades: GradeSet
    items: list[Item]
    dimensions: list[DimensionResult]
    signals: list[ProbabilitySignal]
    readers: list[ReaderAgreement]
    ungraded: list[str]  # labelled answers with no grade

    @property
    def overall(self) -> Agreement:
        """The grader's overall verdict (every applicable dimension passes) against the person's."""
        by_id = self.grades.by_id()
        a = Agreement()
        for item in self.items:
            g = by_id.get(item.id)
            if g is None:
                continue
            gp = all(self.rubric.passes(d, g.score(d)) is not False for d in self.rubric.ids)
            pp = all(self.rubric.passes(d, item.label.final(d)) is not False for d in self.rubric.ids)
            if gp and pp:
                a.tp += 1
            elif gp and not pp:
                a.fp += 1
            elif not gp and pp:
                a.fn += 1
            else:
                a.tn += 1
        return a

    def counts(self, dimension: str) -> dict[str, int]:
        c: dict[str, int] = {"0": 0, "1": 0, "2": 0, "n/a": 0}
        for x in self.cal.labels:
            v = x.final(dimension)
            c["n/a" if v is None else str(v)] += 1
        return c


def judgement_of(item: Item, grades: GradeSet) -> JudgementRecord | None:
    """The probability judge's reading of an item: from the run for a real answer, from grades.json
    for a perturbed one (`judge grade` runs jev on those too)."""
    if item.judgement is not None:
        return item.judgement
    g = grades.by_id().get(item.id)
    return getattr(g, "judgement", None) if g is not None else None


def reading_of(item: Item, grades: GradeSet) -> ReadingRecord | None:
    if item.reading is not None:
        return item.reading
    g = grades.by_id().get(item.id)
    return getattr(g, "reading", None) if g is not None else None


def calibrate_judge(
    run: Run, rubric: Rubric, cal: CalibrationSet, grades: GradeSet, references: dict[str, str]
) -> JudgeCalibration:
    from .metrics import FABRICATED_CEILING, GROUNDED_FLOOR

    items = items_of(cal, run, references)
    by_id = grades.by_id()
    pairs = [(i, by_id.get(i.id)) for i in items]
    ungraded = [i.id for i, g in pairs if g is None]
    dimensions = [dimension_result(d, rubric, pairs) for d in rubric.ids]

    def judged(axis: str) -> list[tuple[Item, float]]:
        out = []
        for i in items:
            j = judgement_of(i, grades)
            if j is not None and i.label.final("groundedness") is not None:
                out.append((i, getattr(j, axis)))
        return out

    def read(axis: str, dimension: str) -> list[tuple[Item, float]]:
        out = []
        for i in items:
            r = reading_of(i, grades)
            if r is not None and i.label.final(dimension) is not None:
                out.append((i, getattr(r, axis)))
        return out

    g_pass = rubric.pass_at("groundedness")
    c_pass = rubric.pass_at("completeness")
    signals = [
        probability_signal(
            "jev grounded",
            f"jev grounded ≥ {GROUNDED_FLOOR}",
            f"groundedness ≥ {g_pass}",
            [(i, p, (i.label.final("groundedness") or 0) >= g_pass) for i, p in judged("grounded")],
            GROUNDED_FLOOR,
        ),
        probability_signal(
            "jev fabricated",
            f"jev fabricated ≥ {FABRICATED_CEILING}",
            "groundedness = 0",
            [(i, p, i.label.final("groundedness") == 0) for i, p in judged("fabricated")],
            FABRICATED_CEILING,
        ),
        probability_signal(
            "reader agrees",
            "jev reference_agrees ≥ 0.5",
            f"completeness ≥ {c_pass}",
            [
                (i, p, (i.label.final("completeness") or 0) >= c_pass)
                for i, p in read("reference_agrees", "completeness")
            ],
            0.5,
        ),
        probability_signal(
            "reader relevant",
            "jev relevant ≥ 0.5",
            "completeness ≥ 1",
            [(i, p, (i.label.final("completeness") or 0) >= 1) for i, p in read("relevant", "completeness")],
            0.5,
        ),
    ]
    readers = [reader_agreement(cal, d) for d in rubric.ids]
    return JudgeCalibration(run, rubric, cal, grades, items, dimensions, signals, readers, ungraded)
