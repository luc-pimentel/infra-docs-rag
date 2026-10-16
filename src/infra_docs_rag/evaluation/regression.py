"""Evals as regression tests (stage 16). The baseline in `eval/baseline.yaml` pins the dataset run every
change is measured against, with the version of everything that produced it, the dataset included, so a
comparison against a stale pin (the records edited, the index rebuilt, the prompt reworded) is refused
instead of read as a result. `compare` puts a candidate dataset run beside it slice by slice and record
by record, applies the slice gates of `eval/matrix.yaml`, lists every record won and lost, and reaches
the decision: a critical slice that drops past its tolerance blocks the change even when the whole set
improves, because the aggregate can rise on the easy records while the ones the slice was built to
watch quietly fail."""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from .dataset import Record, dataset_version
from .experiment import VARIABLES
from .gates import Changed
from .metrics import Metric
from .runs import Run, git_sha
from .slices import (
    ALL,
    MatrixFile,
    MatrixResult,
    SliceGateResult,
    check_matrix,
    check_slice_gates,
    slice_metrics,
    slices_of,
)

BASELINE = Path("eval/baseline.yaml")
HEADLINE = "answers.held"  # the whole-set number the decision reads the aggregate off
Decision = Literal["ship", "hold", "block"]

# What a pin records of the run it names: every field is one the comparison would be meaningless without.
PINNED = (
    "dataset",
    "version",
    "retrieval_version",
    "prompt_version",
    "index_version",
    "embedding_version",
    "chunking_version",
    "generator",
    "backend",
)


class Baseline(BaseModel):
    """The reference experiment: a saved dataset run and the version of everything behind it."""

    run: str  # a dataset run under eval/runs/
    dataset: str  # `<records>+<hash>` of the records the run answered (dataset.py)
    version: str  # the pipeline version's name
    retrieval_version: str  # `<version>+<hash of the retrieval configuration>`
    prompt_version: str  # `<name>+<hash of the wording>`
    index_version: str
    embedding_version: str
    chunking_version: str
    generator: str  # model id
    backend: str
    judge: str | None = None
    pinned_at: str
    git_sha: str
    note: str = ""

    @classmethod
    def of(cls, run: Run, note: str = "") -> "Baseline":
        m = run.manifest
        if m.lineage is None:
            raise ValueError(f"run {run.name} carries no lineage; a run `dataset run` saves does")
        if m.dataset is None:
            raise ValueError(
                f"run {run.name} carries no dataset version; a run `dataset run` saves since stage 16 does"
            )
        return cls(
            run=run.name,
            dataset=m.dataset,
            version=m.version,
            retrieval_version=m.lineage.retrieval_version,
            prompt_version=m.lineage.prompt_version,
            index_version=m.lineage.index_version,
            embedding_version=m.lineage.embedding_version,
            chunking_version=m.lineage.chunking_version,
            generator=m.generator,
            backend=m.backend,
            judge=m.judge,
            pinned_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=git_sha(),
            note=note,
        )

    @classmethod
    def load(cls, path: Path = BASELINE) -> "Baseline":
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; `dataset baseline set --run <run>` writes it")
        try:
            return cls.model_validate(yaml.safe_load(path.read_text()) or {})
        except ValidationError as error:
            raise ValueError(f"{path}: {error}") from error

    def save(self, path: Path = BASELINE) -> Path:
        header = (
            "# The regression baseline: the saved dataset run every change is compared against, and the\n"
            "# version of everything that produced it. `dataset compare` reads it as the default --baseline\n"
            "# and refuses to compare when it is stale: the run no longer matches the pin, or the dataset is\n"
            "# no longer the one the run answered (then the baseline has to be run again). Written by\n"
            "# `infra-docs-rag dataset baseline set --run <run>`; do not edit by hand.\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True))
        return path

    def problems(self, run: Run, records: list[Record]) -> list[str]:
        """What makes the pin stale: a field of the run that no longer reads as pinned, or a dataset
        that is no longer the one the run answered."""
        out: list[str] = []
        if run.name != self.run:
            out.append(f"the pin names run {self.run}, not {run.name}")
        if run.manifest.lineage is None:
            out.append(f"run {run.name} carries no lineage")
        read = {
            "dataset": run.manifest.dataset,
            "version": run.manifest.version,
            "generator": run.manifest.generator,
            "backend": run.manifest.backend,
            **(
                {
                    f: getattr(run.manifest.lineage, f)
                    for f in (
                        "retrieval_version",
                        "prompt_version",
                        "index_version",
                        "embedding_version",
                        "chunking_version",
                    )
                }
                if run.manifest.lineage
                else {}
            ),
        }
        for f in PINNED:
            if f in read and read[f] != getattr(self, f):
                out.append(f"{f}: the pin says {getattr(self, f)}, the run says {read[f]}")
        current = dataset_version(records)
        if current != self.dataset:
            out.append(
                f"the dataset is {current} now and the baseline answered {self.dataset}; "
                f"run `dataset run --version {self.version}` again and pin it"
            )
        return out


# --- the comparison -------------------------------------------------------------------------------


@dataclass
class Regression:
    """A candidate dataset run against the baseline: every slice before and after, the gates, every
    record whose result changed, and the decision."""

    baseline: Run
    candidate: Run
    records: list[Record]
    before: dict[str, dict[str, Metric]]
    after: dict[str, dict[str, Metric]]
    gates: list[SliceGateResult]
    matrix_before: list[MatrixResult]
    matrix_after: list[MatrixResult]
    changed: list[Changed]
    differs: list[str]  # what is not the same between the two runs: stage versions, dataset, generator…
    missing: list[str]  # record ids one of the runs did not answer
    variables: list[tuple[str, str, str]] = field(
        default_factory=list
    )  # (variable, baseline, candidate) per VARIABLES, from the experiment manifests

    @property
    def confounded(self) -> bool:
        """More than one variable differs: a change below cannot be attributed to one of them."""
        return len(self.differs) > 1

    @property
    def wins(self) -> list[Changed]:
        return [x for x in self.changed if x.better is True]

    @property
    def losses(self) -> list[Changed]:
        return [x for x in self.changed if x.better is False]

    @property
    def neutral(self) -> list[Changed]:
        return [x for x in self.changed if x.better is None]

    @property
    def same_passages(self) -> list[Changed]:
        """Changed records whose model read the same passages in the same order in both runs: the flip
        is the generator's variance, not the change under test, and the report says so."""
        return [x for x in self.changed if same_passages(x)]

    def headline(self, name: str = ALL) -> tuple[Metric, Metric]:
        return self.before[name][HEADLINE], self.after[name][HEADLINE]

    @property
    def aggregate_delta(self) -> float:
        b, a = self.headline()
        return (a.value or 0.0) - (b.value or 0.0)

    @property
    def aggregate_improved(self) -> bool:
        return self.aggregate_delta > 0

    @property
    def blocking(self) -> list[SliceGateResult]:
        return [g for g in self.gates if g.blocks]

    @property
    def warnings(self) -> list[SliceGateResult]:
        return [g for g in self.gates if not g.skipped and not g.result.passed and not g.gate.critical]

    @property
    def masked(self) -> list[SliceGateResult]:
        """Critical gates that fail on a slice while the whole set holds as many records as before or
        more: the regressions an aggregate score would hide."""
        return [g for g in self.blocking if g.gate.slice != ALL and self.aggregate_delta >= 0]

    @property
    def decision(self) -> Decision:
        """`block` when a critical gate fails; `hold` when the gates pass and the change still lost more
        records than it won, or the whole set holds fewer (within tolerance): a person decides; `ship`
        otherwise."""
        if self.blocking:
            return "block"
        if len(self.losses) > len(self.wins) or self.aggregate_delta < 0:
            return "hold"
        return "ship"

    @property
    def verdict(self) -> str:
        b, a = self.headline()
        head = f"{b.shown} → {a.shown} held over the whole set"
        tally = f"{len(self.wins)} record(s) won, {len(self.losses)} lost"
        if self.decision == "block":
            where = ", ".join(f"`{g.gate.slice}` {g.gate.metric}" for g in self.blocking)
            masked = (
                f"; the whole set {'improved' if self.aggregate_improved else 'held as many'}, so the aggregate would have hidden {'it' if len(self.masked) == 1 else 'them'}"
                if self.masked
                else ""
            )
            return f"blocked: {len(self.blocking)} critical gate(s) failed ({where}){masked}; {head}; {tally}"
        if self.decision == "hold":
            why = (
                "more records lost than won"
                if len(self.losses) > len(self.wins)
                else "the whole set holds fewer, within the tolerance"
            )
            return f"hold: every critical gate passes but {why}; {head}; {tally}"
        warn = f" with {len(self.warnings)} warning(s)" if self.warnings else ""
        return f"ship{warn}: every critical gate passes; {head}; {tally}"


def same_passages(x: Changed) -> bool:
    return [s.chunk_id for s in x.before.sources] == [s.chunk_id for s in x.after.sources]


def differences(baseline: Run, candidate: Run) -> list[str]:
    """What is not the same between the two runs, by name. With an experiment manifest on both (stage
    18), the variables of `experiment.VARIABLES`; without, the stage versions from the lineage, the
    dataset, the generator and how it was called, the prompt, the effort, the judge."""
    b, c = baseline.manifest, candidate.manifest
    if b.experiment is not None and c.experiment is not None:
        return b.experiment.differs(c.experiment)
    out: list[str] = []
    if b.lineage and c.lineage:
        out += [f.removesuffix("_version") for f in b.lineage.differs(c.lineage) if f != "generator"]
    else:
        out += [k for k in ("version",) if getattr(b, k) != getattr(c, k)]
        if b.retrieval != c.retrieval:
            out.append("retrieval")
    if b.dataset != c.dataset:
        out.append("dataset")
    for name in ("generator", "backend", "prompt_sha256", "effort", "judge"):
        if getattr(b, name) != getattr(c, name) and name.replace("_sha256", "") not in out:
            out.append(name.replace("_sha256", ""))
    return out


def compare(
    baseline: Run,
    candidate: Run,
    records: list[Record],
    matrix: MatrixFile,
    allow_confounded: bool = False,
) -> Regression:
    """The candidate against the baseline on the records both answered. Refused when the two runs differ
    in more than one variable (a controlled comparison changes one thing), unless `allow_confounded`."""
    differs = differences(baseline, candidate)
    if len(differs) > 1 and not allow_confounded:
        raise ValueError(
            f"{candidate.name} differs from {baseline.name} in {len(differs)} variables ({', '.join(differs)}); "
            "a controlled comparison changes one; --allow-confounded compares anyway and the report says so"
        )
    earlier, later = baseline.by_id(), candidate.by_id()
    missing = [r.id for r in records if r.id not in earlier or r.id not in later]
    shared = [r for r in records if r.id not in set(missing)]
    if not shared:
        raise ValueError(f"the runs {baseline.name} and {candidate.name} share no record with the dataset")
    before = slice_metrics(baseline, shared)
    after = slice_metrics(candidate, shared)
    gates = check_slice_gates(matrix.gates, after, before)
    changed = [
        Changed(r.id, earlier[r.id], later[r.id])
        for r in shared
        if (
            earlier[r.id].outcome != later[r.id].outcome
            or earlier[r.id].verdict != later[r.id].verdict
            or earlier[r.id].right_retrieved != later[r.id].right_retrieved
            or earlier[r.id].right_cited != later[r.id].right_cited
        )
    ]
    return Regression(
        baseline=baseline,
        candidate=candidate,
        records=shared,
        before=before,
        after=after,
        gates=gates,
        matrix_before=check_matrix(matrix.matrix, before),
        matrix_after=check_matrix(matrix.matrix, after),
        changed=changed,
        differs=differs,
        missing=missing,
        variables=[
            (name, baseline.manifest.experiment.value(name), candidate.manifest.experiment.value(name))
            for name in VARIABLES
        ]
        if baseline.manifest.experiment and candidate.manifest.experiment
        else [],
    )


def slice_sizes(records: list[Record]) -> dict[str, int]:
    return {name: len(rows) for name, rows in slices_of(records).items()}
