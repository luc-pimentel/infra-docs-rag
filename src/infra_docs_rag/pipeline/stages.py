"""The stage graph and what runs it.

A stage consumes artifacts, produces artifacts, and has a version: the hash of the configuration it
runs under. `Pipeline.run` goes through the stages in order and, for each one, fingerprints its inputs.
When the last recorded run of the stage saw the same inputs under the same version and its outputs are
still what it wrote, the stage is up to date and is skipped. Otherwise it runs. A stage that raises is
recorded as failed with the error, nothing after it runs, and the next `run` starts there: the stages
before it are up to date and skip themselves. Each stage writes through a staging path and publishes in
one rename, so a failure leaves no half-written artifact behind.

State is one JSON file, `data/pipeline/state.json`: for each stage, what it read, what it wrote, under
which version, when, and how it ended."""

import hashlib
import json
import re
import shutil
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ..chunk.chunks import DEFAULT_CHUNKING, Chunk, Chunking, chunk_documents
from ..embed.embedders import DEFAULT_MODEL, Embedder
from ..ingest.models import Document
from ..ingest.pipeline import run as ingest_run
from ..ingest.sources import Source, fetch, load_sources
from .manifest import chunking_version, short
from .store import Store

Fingerprint = Callable[[], str | None]  # the artifact's content hash, or None when it is absent


class StageFailed(Exception):
    def __init__(self, stage: str, error: str) -> None:
        super().__init__(f"stage {stage} failed: {error}")
        self.stage = stage
        self.error = error


class StageState(BaseModel):
    status: str  # ok | failed
    version: str
    inputs: dict[str, str]
    outputs: dict[str, str] = {}
    started_at: str
    finished_at: str
    ms: float
    note: str = ""
    error: str | None = None


class State(BaseModel):
    stages: dict[str, StageState] = {}
    updated_at: str = ""

    @classmethod
    def load(cls, path: Path) -> "State":
        return cls.model_validate_json(path.read_text()) if path.exists() else cls()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.updated_at = now()
        tmp = path.with_suffix(".tmp")
        tmp.write_text(self.model_dump_json(indent=2) + "\n")
        tmp.replace(path)


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class Artifact:
    name: str
    path: Path
    fingerprint: Fingerprint
    made_by: str | None = None  # the stage that produces it; None for a file kept in the repository


@dataclass
class Stage:
    name: str
    consumes: list[str]
    produces: list[str]
    version: str  # hash of the configuration the stage runs under
    run: Callable[[], str]  # does the work, returns a one-line note
    what: str = ""  # for the status table


@dataclass
class Outcome:
    stage: str
    status: str  # ok | skipped | failed
    ms: float
    note: str


@dataclass
class Pipeline:
    stages: list[Stage]
    artifacts: dict[str, Artifact]
    state_path: Path
    log: Callable[[str], None] = print

    def names(self) -> list[str]:
        return [s.name for s in self.stages]

    def select(self, start: str | None, stop: str | None, only: str | None) -> list[Stage]:
        names = self.names()
        for given in (start, stop, only):
            if given is not None and given not in names:
                raise ValueError(f"unknown stage {given!r}; one of {', '.join(names)}")
        if only is not None:
            return [s for s in self.stages if s.name == only]
        first = names.index(start) if start else 0
        last = names.index(stop) if stop else len(names) - 1
        if last < first:
            raise ValueError(f"--to {stop} comes before --from {start}")
        return self.stages[first : last + 1]

    def inputs(self, stage: Stage) -> dict[str, str]:
        found: dict[str, str] = {}
        for name in stage.consumes:
            a = self.artifacts[name]
            print_ = a.fingerprint()
            if print_ is None:
                hint = f"; run stage {a.made_by}" if a.made_by else ""
                raise StageFailed(stage.name, f"input {name} is missing ({a.path}){hint}")
            found[name] = print_
        return found

    def outputs(self, stage: Stage) -> dict[str, str]:
        return {
            name: self.artifacts[name].fingerprint() or ""
            for name in stage.produces
            if name in self.artifacts
        }

    def up_to_date(self, stage: Stage, state: State, inputs: dict[str, str]) -> bool:
        prior = state.stages.get(stage.name)
        return (
            prior is not None
            and prior.status == "ok"
            and prior.version == stage.version
            and prior.inputs == inputs
            and all(v for v in prior.outputs.values())
            and self.outputs(stage) == prior.outputs
        )

    def run(
        self,
        start: str | None = None,
        stop: str | None = None,
        only: str | None = None,
        force: bool = False,
    ) -> list[Outcome]:
        """Run the selected stages in order; stop at the first failure. Raises StageFailed after the
        failure is recorded, with the outcomes so far on the exception."""
        state = State.load(self.state_path)
        outcomes: list[Outcome] = []
        for stage in self.select(start, stop, only):
            started = time.perf_counter()
            try:
                inputs = self.inputs(stage)
                if not force and self.up_to_date(stage, state, inputs):
                    outcomes.append(Outcome(stage.name, "skipped", 0.0, "up to date"))
                    self.log(f"{stage.name:9} skipped   up to date")
                    continue
                self.log(f"{stage.name:9} running")
                started_at = now()
                note = stage.run()
                ms = (time.perf_counter() - started) * 1000
                state.stages[stage.name] = StageState(
                    status="ok",
                    version=stage.version,
                    inputs=inputs,
                    outputs=self.outputs(stage),
                    started_at=started_at,
                    finished_at=now(),
                    ms=round(ms, 1),
                    note=note,
                )
                state.save(self.state_path)
                outcomes.append(Outcome(stage.name, "ok", ms, note))
                self.log(f"{stage.name:9} ok        {note} ({ms / 1000:.1f}s)")
            except StageFailed as failure:
                ms = (time.perf_counter() - started) * 1000
                self.record_failure(state, stage, failure.error, ms)
                outcomes.append(Outcome(stage.name, "failed", ms, failure.error))
                self.log(f"{stage.name:9} FAILED    {failure.error}")
                failure.outcomes = outcomes  # type: ignore[attr-defined]
                raise
            except Exception as error:  # noqa: BLE001  the boundary: whatever a stage raises stops here
                ms = (time.perf_counter() - started) * 1000
                detail = f"{type(error).__name__}: {error}"
                self.record_failure(state, stage, detail, ms, traceback.format_exc())
                outcomes.append(Outcome(stage.name, "failed", ms, detail))
                self.log(f"{stage.name:9} FAILED    {detail}")
                failure = StageFailed(stage.name, detail)
                failure.outcomes = outcomes  # type: ignore[attr-defined]
                raise failure from error
        return outcomes

    def record_failure(
        self, state: State, stage: Stage, error: str, ms: float, trace: str | None = None
    ) -> None:
        prior = state.stages.get(stage.name)
        state.stages[stage.name] = StageState(
            status="failed",
            version=stage.version,
            inputs=prior.inputs if prior else {},
            outputs={},
            started_at=now(),
            finished_at=now(),
            ms=round(ms, 1),
            error=error + (f"\n{trace}" if trace else ""),
        )
        state.save(self.state_path)

    def status(self) -> list[tuple[Stage, StageState | None, str]]:
        """Each stage with its recorded state and a word on where it stands now: `ok`, `failed`, `stale`
        (an input changed since it ran, or an earlier stage is not ok) or `never run`."""
        state = State.load(self.state_path)
        out = []
        broken = False
        for stage in self.stages:
            prior = state.stages.get(stage.name)
            if prior is None:
                standing = "never run"
            elif prior.status == "failed":
                standing = "failed"
            elif broken:
                standing = "stale (an earlier stage is not ok)"
            else:
                try:
                    standing = "ok" if self.up_to_date(stage, state, self.inputs(stage)) else "stale"
                except StageFailed:
                    standing = "stale (an input is missing)"
            if standing != "ok":
                broken = True
            out.append((stage, prior, standing))
        return out


# ---------------------------------------------------------------------------------------------------
# Fingerprints: the content identity of each artifact, so that a rewrite that changes nothing but a
# timestamp does not look like a change.


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def file_fingerprint(path: Path) -> str | None:
    return sha(path.read_bytes()) if path.exists() else None


def raw_fingerprint(raw_dir: Path, sources: list[Source]) -> str | None:
    """Every source's content hash: from the download manifest for a URL, from the file for a path."""
    manifest_path = raw_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    parts = []
    for s in sources:
        if s.path:
            local = Path(s.path)
            if not local.exists():
                return None
            parts.append(f"{s.id}={sha(local.read_bytes())}")
        else:
            entry = manifest.get(s.id, {})
            if not entry.get("sha256") or not s.raw_file(raw_dir).exists():
                return None
            parts.append(f"{s.id}={entry['sha256']}")
    return sha("\n".join(parts).encode())


def documents_fingerprint(path: Path) -> str | None:
    """What the records say, timestamps and the ingest version aside: a re-ingest of the same files with
    the same code gives the same fingerprint."""
    if not path.exists():
        return None
    h = hashlib.sha256()
    for line in path.read_text().splitlines():
        if not line:
            continue
        d = json.loads(line)
        h.update(
            json.dumps(
                [
                    d["source_id"],
                    d["provenance"]["source_sha256"],
                    d["content_hash"],
                    d["parse_status"],
                    d["title"],
                    len(d["sections"]),
                    len(d["pages"]),
                    d.get("duplicate"),
                ]
            ).encode()
        )
    return h.hexdigest()[:16]


def index_fingerprint(store: Store) -> str | None:
    return store.current()


def run_fingerprint(directory: Path) -> str | None:
    cases = directory / "cases.jsonl"
    return sha(cases.read_bytes()) if cases.exists() else None


def publish(stage: Callable[[Path], None], final: Path) -> None:
    """Run `stage` writing to a staging path next to `final`, then rename over `final` in one step."""
    staging = final.with_name(f".{final.name}.staging")
    if staging.exists():
        staging.unlink() if staging.is_file() else shutil.rmtree(staging)
    final.parent.mkdir(parents=True, exist_ok=True)
    stage(staging)
    staging.replace(final)


def chunks_slug(chunking: Chunking) -> str:
    return re.sub(r"[^a-z0-9]+", "-", chunking.label.lower()).strip("-")


# ---------------------------------------------------------------------------------------------------
# The graph as it runs here: fetch → ingest → chunk → embed, then evaluate and gate when a generator is
# given. Paths and the embedder come from outside so a test can run the whole graph on fixtures.


@dataclass
class Paths:
    sources: Path = Path("sources.yaml")
    raw_dir: Path = Path("data/raw")
    documents: Path = Path("data/processed/documents.jsonl")
    chunks_dir: Path = Path("data/chunks")
    index_dir: Path = Path("data/index")
    runs_dir: Path = Path("eval/runs")
    state: Path = Path("data/pipeline/state.json")
    benchmark_files: tuple[Path, ...] = (
        Path("eval/retrieval.yaml"),
        Path("eval/hybrid.yaml"),
        Path("eval/grounding.yaml"),
    )
    gates: Path = Path("eval/gates.yaml")
    releases: Path = Path("eval/releases.yaml")


@dataclass
class Config:
    model: str = DEFAULT_MODEL
    chunking: Chunking = field(default_factory=lambda: DEFAULT_CHUNKING)
    refresh: bool = False  # fetch: download again what is already there


@dataclass
class Evaluation:
    """What the evaluate and gate stages need: a run name, a function that produces the run, and what
    goes into the stage version (the generator, the prompt, the pipeline version, the limit)."""

    run_name: str
    version: str  # the stage's version string: everything that changes the answers
    produce: Callable[[Path], None]  # writes the run into the directory it is given
    gate: Callable[[Path], str]  # compares the run to the current release, raises StageFailed when blocked


def load_documents(path: Path) -> list[Document]:
    return [Document.model_validate_json(line) for line in path.read_text().splitlines() if line]


def load_chunks(path: Path) -> list[Chunk]:
    return [Chunk.model_validate_json(line) for line in path.read_text().splitlines() if line]


def build_pipeline(
    paths: Paths,
    config: Config,
    embedder: Callable[[], Embedder],
    evaluation: Evaluation | None = None,
    log: Callable[[str], None] = print,
) -> Pipeline:
    sources = load_sources(paths.sources)
    projects = {s.id: s.project for s in sources if s.project}
    chunks_file = paths.chunks_dir / f"{chunks_slug(config.chunking)}.jsonl"
    store = Store(paths.index_dir / config.model)
    loaded: dict[str, Embedder] = {}

    def model() -> Embedder:
        if "it" not in loaded:
            loaded["it"] = embedder()
        return loaded["it"]

    artifacts = {
        "sources": Artifact("sources", paths.sources, lambda: file_fingerprint(paths.sources)),
        "raw": Artifact("raw", paths.raw_dir, lambda: raw_fingerprint(paths.raw_dir, sources), "fetch"),
        "documents": Artifact(
            "documents", paths.documents, lambda: documents_fingerprint(paths.documents), "ingest"
        ),
        "chunks": Artifact("chunks", chunks_file, lambda: file_fingerprint(chunks_file), "chunk"),
        "index": Artifact("index", store.root, lambda: index_fingerprint(store), "embed"),
    }

    def do_fetch() -> str:
        manifest = fetch(sources, paths.raw_dir, refresh=config.refresh)
        return f"{sum(1 for s in sources if s.url)} url sources on disk, {len(manifest)} in the manifest"

    def do_ingest() -> str:
        count = {"n": 0}

        def write(staging: Path) -> None:
            staging.mkdir(parents=True, exist_ok=True)
            docs = ingest_run(sources, paths.raw_dir, staging / paths.documents.name)
            count["n"] = len(docs)

        staging_dir = paths.documents.parent / ".staging"
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        write(staging_dir)
        paths.documents.parent.mkdir(parents=True, exist_ok=True)
        for name in (paths.documents.name, "pairs.json"):
            (staging_dir / name).replace(paths.documents.parent / name)
        shutil.rmtree(staging_dir, ignore_errors=True)
        docs = load_documents(paths.documents)
        failed = sum(d.parse_status not in ("ok", "ocr_fallback") for d in docs)
        return f"{count['n']} records, {failed} failed or empty, {sum(bool(d.duplicate) for d in docs)} duplicates"

    def do_chunk() -> str:
        docs = load_documents(paths.documents)
        chunks = chunk_documents(docs, config.chunking, model(), projects)
        if not chunks:
            raise StageFailed("chunk", "no usable document produced a chunk")

        def write(staging: Path) -> None:
            staging.write_text("".join(c.model_dump_json() + "\n" for c in chunks))

        publish(write, chunks_file)
        chunks_file.with_suffix(".manifest.json").write_text(
            json.dumps(
                {
                    "chunking": config.chunking.model_dump(),
                    "chunking_version": chunking_version(config.chunking),
                    "tokenizer": model().model_id,
                    "ingest_version": docs[0].provenance.ingest_version if docs else "",
                    "documents": len({c.document_id for c in chunks}),
                    "chunks": len(chunks),
                    "written_at": now(),
                },
                indent=2,
            )
            + "\n"
        )
        return f"{len(chunks)} chunks of {len({c.document_id for c in chunks})} documents ({config.chunking.label})"

    def do_embed() -> str:
        chunks = load_chunks(chunks_file)
        docs = load_documents(paths.documents)
        version, index, new = store.publish(
            model(), chunks, docs[0].provenance.ingest_version, config.chunking
        )
        return f"index {version} {'built' if new else 'already present'}, {index.manifest.chunks} chunks, now current"

    stages = [
        Stage(
            "fetch",
            ["sources"],
            ["raw"],
            short(f"refresh={config.refresh}"),
            do_fetch,
            "download URL sources",
        ),
        Stage(
            "ingest",
            ["sources", "raw"],
            ["documents"],
            short("ingest"),
            do_ingest,
            "parse, clean, deduplicate",
        ),
        Stage(
            "chunk",
            ["documents"],
            ["chunks"],
            short(config.chunking.model_dump_json() + config.model),
            do_chunk,
            "cut records into chunks",
        ),
        Stage(
            "embed",
            ["chunks", "documents"],
            ["index"],
            short(config.model + config.chunking.model_dump_json()),
            do_embed,
            "embed chunks, check and publish the index",
        ),
    ]

    if evaluation is not None:
        run_dir = paths.runs_dir / evaluation.run_name
        artifacts["benchmark"] = Artifact(
            "benchmark",
            paths.benchmark_files[0],
            lambda: sha(b"".join(p.read_bytes() for p in paths.benchmark_files if p.exists())),
        )
        artifacts["run"] = Artifact("run", run_dir, lambda: run_fingerprint(run_dir), "evaluate")
        artifacts["gates"] = Artifact("gates", paths.gates, lambda: file_fingerprint(paths.gates) or "none")
        artifacts["releases"] = Artifact(
            "releases", paths.releases, lambda: file_fingerprint(paths.releases) or "none"
        )

        def do_evaluate() -> str:
            evaluation.produce(run_dir)
            return f"run {evaluation.run_name} saved"

        def do_gate() -> str:
            return evaluation.gate(run_dir)

        stages += [
            Stage(
                "evaluate",
                ["index", "benchmark"],
                ["run"],
                evaluation.version,
                do_evaluate,
                "answer the benchmark, save the run",
            ),
            Stage("gate", ["run", "gates", "releases"], [], short("gate"), do_gate, "compare to the release"),
        ]
    return Pipeline(stages, artifacts, paths.state, log)
