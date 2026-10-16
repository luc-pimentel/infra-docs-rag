"""Releases: the pipeline versions that were measured, cleared the gates against the one before, and
were made the one that answers questions. `eval/releases.yaml` is the record; it is committed, so the
known-good configuration is in the repository and not in anyone's memory. A release names the saved run
that measured it and carries that run's lineage, so what it pins is a set of content hashes, not a label.

Promoting compares the candidate run against the current release's run with the regression gates;
a critical failure refuses the release and leaves the record untouched. Rolling back makes the release
before the current one current again and points the index store at its index version when it is still
on disk."""

from datetime import UTC, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from ..embed.embedders import MODELS
from ..evaluation.experiment import ExperimentManifest, Frozen
from ..evaluation.gates import Comparison, Gate, compare
from ..evaluation.runs import Run
from .manifest import PipelineManifest
from .store import Store

RELEASES = Path("eval/releases.yaml")


class ReleaseBlocked(Exception):
    def __init__(self, comparison: Comparison) -> None:
        failed = ", ".join(r.gate.metric for r in comparison.results if r.blocks)
        super().__init__(f"{comparison.verdict} ({failed})")
        self.comparison = comparison


class Release(BaseModel):
    name: str  # r1, r2, ...
    created_at: str
    run: str  # the saved run under eval/runs/ that measured it
    manifest: PipelineManifest
    against: str | None = None  # the release it was compared to, when there was one
    gates: str = ""  # the comparison's verdict
    note: str = ""
    experiment: ExperimentManifest | None = (
        None  # every variable the candidate's scores depended on (stage 18)
    )
    frozen: Frozen | None = None  # the eval assets the decision was read under
    safety_run: str | None = None  # the graded safety run the decision read (stage 21)


class Releases(BaseModel):
    current: str | None = None
    releases: list[Release] = []

    @classmethod
    def load(cls, path: Path = RELEASES) -> "Releases":
        if not path.exists():
            return cls()
        try:
            return cls.model_validate(yaml.safe_load(path.read_text()) or {})
        except ValidationError as error:
            raise ValueError(f"{path}: {error}") from error

    def save(self, path: Path = RELEASES) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        head = (
            "# The releases of the pipeline: each one a saved run that cleared the gates against the release\n"
            "# before it, with the lineage of what it measured. `current` is the one that answers questions.\n"
            "# Written by `infra-docs-rag pipeline release` and `pipeline rollback`; do not edit by hand.\n"
        )
        path.write_text(head + yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True))

    def get(self, name: str) -> Release:
        for r in self.releases:
            if r.name == name:
                return r
        raise ValueError(f"no release named {name!r}")

    @property
    def live(self) -> Release | None:
        return self.get(self.current) if self.current else None

    def previous(self) -> Release | None:
        """The release before the current one, in release order."""
        names = [r.name for r in self.releases]
        if self.current is None or names.index(self.current) == 0:
            return None
        return self.releases[names.index(self.current) - 1]

    def next_name(self) -> str:
        return f"r{len(self.releases) + 1}"


def lineage_of(run: Run) -> PipelineManifest:
    if run.manifest.lineage is None:
        raise ValueError(
            f"run {run.name} carries no lineage; it predates the pipeline manifest, "
            "run `uv run python scripts/migrate_runs.py eval/runs/<name>` to fill it in"
        )
    return run.manifest.lineage


def promote(
    candidate: Run,
    releases: Releases,
    gates: list[Gate],
    baseline: Run | None,
    note: str = "",
    frozen: Frozen | None = None,
    safety: tuple[str, bool] | None = None,
) -> tuple[Release, Comparison | None]:
    """Compare the candidate to the current release's run and, when no critical gate fails, add it as
    the current release. `baseline` is the current release's run, loaded by the caller; None when there
    is no release yet, in which case the candidate is accepted as the first."""
    lineage = lineage_of(candidate)
    experiment = candidate.manifest.experiment
    if frozen is not None and experiment is not None:
        moved = frozen.matches(experiment)
        if moved:
            raise ValueError(
                f"run {candidate.name} was read under other assets than the freeze: " + "; ".join(moved)
            )
    if safety is not None and not safety[1]:
        raise ValueError(f"the safety run {safety[0]} blocks {candidate.manifest.version}; not released")
    comparison = None
    live = releases.live
    if live is not None:
        if baseline is None:
            raise ValueError(
                f"the current release {live.name} names run {live.run}, which is needed to compare"
            )
        if baseline.name == candidate.name:
            raise ValueError(f"{candidate.name} is already the current release's run")
        comparison = compare(baseline, candidate, gates)
        if comparison.blocked:
            raise ReleaseBlocked(comparison)
    release = Release(
        name=releases.next_name(),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        run=candidate.name,
        manifest=lineage.model_copy(update={"eval_run_id": candidate.name}),
        against=live.name if live else None,
        gates=comparison.verdict if comparison else "first release, no baseline to compare against",
        note=note,
        experiment=experiment,
        frozen=frozen,
        safety_run=safety[0] if safety else None,
    )
    releases.releases.append(release)
    releases.current = release.name
    return release, comparison


def rollback(releases: Releases) -> tuple[Release, Release]:
    """Make the previous release current. Returns (the one left, the one now current)."""
    live = releases.live
    if live is None:
        raise ValueError("no release to roll back from")
    earlier = releases.previous()
    if earlier is None:
        raise ValueError(f"{live.name} is the first release; there is nothing before it to roll back to")
    releases.current = earlier.name
    return live, earlier


def apply_index(release: Release, index_dir: Path) -> str:
    """Point the index store at the release's index version, when the store holds it. Returns what was
    done, for the reader."""
    model_id = release.manifest.embedding_version.split("@")[0]
    name = next((spec.name for spec in MODELS.values() if spec.model_id == model_id), None)
    if name is None:
        return f"index {release.manifest.index_version} belongs to an unknown model {model_id}; not switched"
    store = Store(index_dir / name)
    wanted = release.manifest.index_version
    if store.current() == wanted:
        return f"index {name}/{wanted} is already current"
    if wanted not in store.versions():
        return f"index {name}/{wanted} is not on disk; run `pipeline run --to embed` at that version to rebuild it"
    store.point(wanted)
    return f"index {name} now points at {wanted}"
