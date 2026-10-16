"""Prompts, retrieval config and eval assets versioned together (stage 18). Every number an eval
produces depends on more than the pipeline: the dataset and the benchmark it was asked, the grader
that read the answers, the gates that judged the scores, the safety policy. The experiment manifest
records all of them for a run, as content hashes, so two runs can be told apart on every variable
that moves a score and a comparison can say which one it changes. The freeze (`eval/frozen.yaml`)
pins the assets a release decision is read under, so the dataset or the grader cannot drift under a
decision without the drift being named."""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import yaml
from pydantic import BaseModel, ValidationError

from ..pipeline.manifest import git_sha

FROZEN = Path("eval/frozen.yaml")
MATRIX = Path("eval/matrix.yaml")
GATES = Path("eval/gates.yaml")
SAFETY = Path("eval/safety.yaml")  # stage 21; `none` until it exists
BENCHMARK_FILES = (Path("eval/retrieval.yaml"), Path("eval/hybrid.yaml"), Path("eval/grounding.yaml"))

# The variables that move a score, and the manifest fields each one covers. A comparison that changes
# one variable is controlled; one that changes more is confounded and the report says so.
VARIABLES: dict[str, tuple[str, ...]] = {
    "retrieval": ("retrieval_version",),
    "prompt": ("prompt_version",),
    "corpus": ("ingest_version", "chunking_version", "embedding_version", "index_version"),
    "model": ("generator", "backend", "effort"),
    "dataset": ("dataset_version", "benchmark_version"),
    "grader": ("grader_version", "judge"),
    "gates": ("matrix_version", "gates_version"),
    "safety": ("safety_version",),
}


def file_version(path: Path) -> str:
    """`<name>+<hash>` of a file's bytes; `none` when the file does not exist."""
    if not path.exists():
        return "none"
    return f"{path.stem}+{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}"


def benchmark_version(paths: tuple[Path, ...] = BENCHMARK_FILES) -> str:
    """`benchmark+<hash>` over the three question files, in order."""
    h = hashlib.sha256()
    for p in paths:
        h.update(p.read_bytes() if p.exists() else b"")
    return f"benchmark+{h.hexdigest()[:12]}"


class ExperimentManifest(BaseModel):
    """Every variable a run's scores depend on, as a version or a content hash."""

    version: str  # the pipeline version's name
    retrieval_version: str
    prompt_version: str
    ingest_version: str
    chunking_version: str
    embedding_version: str
    index_version: str
    generator: str
    backend: str
    effort: str = ""
    dataset_version: str | None = None  # the records (section 9); None on a benchmark run
    benchmark_version: str  # the three question files
    grader_version: str  # `<rubric name>+<hash of the grader's prompt>`
    judge: str | None = None  # the probability judge and reader (jev) the run was read by
    matrix_version: str  # the slice gates and the matrix
    gates_version: str  # the benchmark gates
    safety_version: str = "none"  # the safety policy (stage 21)
    git_sha: str = ""
    created_at: str = ""

    @classmethod
    def of(
        cls,
        manifest: Any,
        dataset_version: str | None,
        grader_version: str,
        benchmark: str | None = None,
        matrix: Path = MATRIX,
        gates: Path = GATES,
        safety: Path = SAFETY,
    ) -> "ExperimentManifest":
        """From a run manifest (`RunManifest`) that carries a lineage."""
        L = manifest.lineage
        if L is None:
            raise ValueError(f"run {manifest.name} carries no lineage; the experiment manifest needs it")
        return cls(
            version=manifest.version,
            retrieval_version=L.retrieval_version,
            prompt_version=L.prompt_version,
            ingest_version=L.ingest_version,
            chunking_version=L.chunking_version,
            embedding_version=L.embedding_version,
            index_version=L.index_version,
            generator=manifest.generator,
            backend=manifest.backend,
            effort=manifest.effort,
            dataset_version=dataset_version,
            benchmark_version=benchmark or benchmark_version(),
            grader_version=grader_version,
            judge=manifest.judge,
            matrix_version=file_version(matrix),
            gates_version=file_version(gates),
            safety_version=file_version(safety),
            git_sha=git_sha(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def differs(self, other: "ExperimentManifest") -> list[str]:
        """The variables that are not the same in `other`, by name, in VARIABLES order."""
        return [
            name
            for name, fields in VARIABLES.items()
            if any(getattr(self, f) != getattr(other, f) for f in fields)
        ]

    def value(self, variable: str) -> str:
        """One variable's value, for a report: the fields it covers, joined."""
        parts = [str(getattr(self, f)) for f in VARIABLES[variable] if getattr(self, f) not in (None, "")]
        return " · ".join(parts) or "–"

    def lines(self) -> list[str]:
        return [f"{name:10} {self.value(name)}" for name in VARIABLES]


class Frozen(BaseModel):
    """The eval assets a release decision is read under, pinned by content: the dataset and the
    benchmark, the grader and the judge, the gates, the safety policy. `dataset freeze` writes it;
    `dataset freeze --check` and `pipeline release` refuse to decide when the files have moved."""

    dataset_version: str
    benchmark_version: str
    grader_version: str
    judge: str | None = None
    matrix_version: str
    gates_version: str
    safety_version: str = "none"
    frozen_at: str
    git_sha: str
    note: str = ""

    FIELDS: ClassVar[tuple[str, ...]] = (
        "dataset_version",
        "benchmark_version",
        "grader_version",
        "judge",
        "matrix_version",
        "gates_version",
        "safety_version",
    )

    @classmethod
    def now(
        cls,
        dataset_version: str,
        grader_version: str,
        judge: str | None,
        note: str = "",
        matrix: Path = MATRIX,
        gates: Path = GATES,
        safety: Path = SAFETY,
        benchmark: tuple[Path, ...] = BENCHMARK_FILES,
    ) -> "Frozen":
        return cls(
            dataset_version=dataset_version,
            benchmark_version=benchmark_version(benchmark),
            grader_version=grader_version,
            judge=judge,
            matrix_version=file_version(matrix),
            gates_version=file_version(gates),
            safety_version=file_version(safety),
            frozen_at=datetime.now(UTC).isoformat(timespec="seconds"),
            git_sha=git_sha(),
            note=note,
        )

    @classmethod
    def load(cls, path: Path = FROZEN) -> "Frozen":
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; `dataset freeze` writes it")
        try:
            return cls.model_validate(yaml.safe_load(path.read_text()) or {})
        except ValidationError as error:
            raise ValueError(f"{path}: {error}") from error

    def save(self, path: Path = FROZEN) -> Path:
        header = (
            "# The freeze: the eval assets a release decision is read under, pinned by content hash. The\n"
            "# dataset and the benchmark questions, the grader (the rubric's prompt) and the judge, the gates\n"
            "# and the matrix, the safety policy. `dataset freeze --check` and `pipeline release` compare the\n"
            "# files to this and refuse when one has moved: a decision is taken under a named set of assets,\n"
            "# and changing one is a deliberate `dataset freeze`, not a side effect. Written by\n"
            "# `infra-docs-rag dataset freeze`; do not edit by hand.\n"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True))
        return path

    def problems(self, current: "Frozen") -> list[str]:
        """What moved since the freeze, field by field."""
        return [
            f"{f}: frozen {getattr(self, f)}, now {getattr(current, f)}"
            for f in self.FIELDS
            if getattr(self, f) != getattr(current, f)
        ]

    ANSWER_FIELDS: ClassVar[tuple[str, ...]] = (
        "dataset_version",
        "benchmark_version",
        "grader_version",
        "judge",
    )

    def matches(self, experiment: ExperimentManifest) -> list[str]:
        """What a run's experiment manifest does not share with the freeze, among the assets that shaped
        its answers and readings (the dataset, the benchmark, the grader, the judge). The gates and the
        policy shape the decision, not the run, and are read from the freeze itself."""
        out = []
        for f in self.ANSWER_FIELDS:
            mine, theirs = getattr(self, f), getattr(experiment, f)
            if f == "dataset_version" and theirs is None:
                continue  # a benchmark run names no dataset
            if mine != theirs:
                out.append(f"{f}: frozen {mine}, the run's {theirs}")
        return out
