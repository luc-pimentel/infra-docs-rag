"""What `infra-docs-rag pipeline ...` does, and how `answer` finds the released configuration."""

import argparse
import sys
from pathlib import Path

from ..embed.embedders import load_embedder
from ..embed.index import Index
from ..evaluation.experiment import Frozen
from ..evaluation.gates import load_gates
from ..evaluation.runs import RUNS_DIR, Run
from ..evaluation.versions import DEFAULT_VERSION, Version, load_versions, pick
from ..generate.prompts import Prompt, load_prompt
from ..retrieve.retriever import DEFAULT_RETRIEVAL, Retrieval
from .manifest import PipelineManifest
from .releases import RELEASES, ReleaseBlocked, Releases, apply_index, promote, rollback
from .stages import Config, Evaluation, Paths, Pipeline, StageFailed, build_pipeline
from .store import Store

RELEASE = "release"  # what --version takes to mean "the current release's version"


def resolve_version(
    name: str | None, retrieval: Retrieval | None = None, releases_path: Path = RELEASES
) -> tuple[Version, str | None]:
    """The pipeline version a command runs: `--version NAME` from eval/versions.yaml, `--version release`
    the current release's, and with no flag, the release's when there is one and no retrieval flag was
    given, else the defaults. Returns the version and the index version it pins, if any."""
    versions = load_versions()
    custom = retrieval is not None and retrieval != DEFAULT_RETRIEVAL
    live = Releases.load(releases_path).live
    if name == RELEASE or (name is None and not custom and live is not None):
        if live is None:
            raise ValueError(
                f"no release yet; `pipeline release --candidate <run>` makes one (see {releases_path})"
            )
        version = pick(versions, live.manifest.version)
        return version, live.manifest.index_version
    if name is None or name == DEFAULT_VERSION:
        version = versions[DEFAULT_VERSION]
        if custom:
            version = version.model_copy(update={"retrieval": retrieval})
        return version, None
    version = pick(versions, name)
    return version, version.index


def index_for(version: Version, pinned: str | None, index_dir: Path) -> tuple[Index, str]:
    """The index a version reads and its version id. A pinned version that is not on disk falls back to
    the current one, and says so."""
    store = Store(index_dir / version.embedding)
    if pinned and pinned not in store.versions():
        print(
            f"warning: index {version.embedding}/{pinned} is not on disk; reading the current one instead",
            file=sys.stderr,
        )
        pinned = None
    index = store.load(pinned)
    return index, pinned or store.current() or "unknown"


def lineage(
    version: Version, index: Index, index_id: str, prompt: Prompt, backend: str, generator: str = ""
) -> PipelineManifest:
    return PipelineManifest.of(
        version.name, index.manifest, index_id, prompt, backend, version.retrieval, generator
    )


def add_pipeline_commands(commands: argparse._SubParsersAction) -> None:
    pipeline = commands.add_parser(
        "pipeline",
        help="run the stages in order, skipping what is up to date; see where each stands; manage index versions and releases",
    )
    sub = pipeline.add_subparsers(dest="pipeline_command", required=True)
    run = sub.add_parser(
        "run",
        help="fetch → ingest → chunk → embed (→ evaluate → gate with --generator), up-to-date stages skipped",
    )
    run.add_argument("--from", dest="start", metavar="STAGE", help="start here")
    run.add_argument("--to", dest="stop", metavar="STAGE", help="stop after this stage (default embed)")
    run.add_argument("--only", metavar="STAGE", help="this stage alone")
    run.add_argument("--force", action="store_true", help="run the selected stages even when up to date")
    run.add_argument("--refresh", action="store_true", help="fetch: download the sources again")
    run.add_argument("--size", type=int, help="chunk: tokens per chunk (default the current chunking's)")
    run.add_argument("--overlap", type=int, help="chunk: tokens repeated (default the current chunking's)")
    run.add_argument("--generator", help="evaluate: the generator; needed for --to evaluate or gate")
    run.add_argument("--effort", help="evaluate: how hard the model thinks")
    run.add_argument(
        "--version", default=DEFAULT_VERSION, help="evaluate: the pipeline version to answer under"
    )
    run.add_argument("--limit", type=int, help="evaluate: only the first N questions")
    run.add_argument("--no-judge", action="store_true", help="evaluate: skip the judge")
    run.add_argument(
        "--save", metavar="NAME", help="evaluate: the run's name (default <version>-<generator>)"
    )
    sub.add_parser("status", help="where each stage stands, the index versions, the releases")
    index = sub.add_parser("index", help="the index versions of a model: list them, or point at one")
    index.add_argument("--list", action="store_true", help="list the versions (the default)")
    index.add_argument("--use", metavar="VERSION", help="make this version the current one")
    index.add_argument("--remove", metavar="VERSION", help="delete this version from disk")
    release = sub.add_parser(
        "release", help="promote a saved run's pipeline to the current release, if it clears the gates"
    )
    release.add_argument("--candidate", required=True, metavar="RUN", help="a run under eval/runs/")
    release.add_argument("--note", default="", help="why, for the record")
    sub.add_parser("rollback", help="make the previous release current again")
    canary = sub.add_parser(
        "canary",
        help="ask the released pipeline the canary questions (eval/canary.yaml); exit 1 when one fails",
    )
    canary.add_argument("--version", help="a version from eval/versions.yaml (default: the current release)")
    canary.add_argument("--generator", default="claude-code", help="the generator (default claude-code)")
    canary.add_argument("--effort", default="medium", help="how hard the model thinks (default medium)")
    sub.add_parser("health", help="is the service in a state to answer? exit 1 when a check fails")


def paths_and_config(args: argparse.Namespace, model: str, chunking) -> tuple[Paths, Config]:
    update = {}
    if getattr(args, "size", None) is not None:
        update["size"] = args.size
    if getattr(args, "overlap", None) is not None:
        update["overlap"] = args.overlap
    if update:
        chunking = chunking.model_copy(update=update)
    return Paths(), Config(model=model, chunking=chunking, refresh=getattr(args, "refresh", False))


def run_pipeline(args: argparse.Namespace, model: str, chunking, evaluation_factory) -> int:
    """`pipeline run`. Returns the exit code."""
    paths, config = paths_and_config(args, model, chunking)
    wants_eval = args.stop in ("evaluate", "gate") or args.only in ("evaluate", "gate")
    evaluation: Evaluation | None = None
    if wants_eval:
        if not args.generator:
            raise ValueError("--to evaluate and --to gate call the model; name one with --generator")
        evaluation = evaluation_factory(args, paths)
    stop = args.stop or (None if args.only else "embed")
    pipeline = build_pipeline(paths, config, lambda: load_embedder(model), evaluation)
    try:
        pipeline.run(args.start, stop, args.only, args.force)
    except StageFailed as failure:
        print(
            f"stopped at {failure.stage}; fix the cause and run again, the stages before it will skip themselves"
        )
        return 1
    return 0


def status(model: str, chunking, index_dir: Path, releases_path: Path = RELEASES) -> None:
    pipeline = build_pipeline(Paths(), Config(model=model, chunking=chunking), lambda: load_embedder(model))
    print(f"{'stage':9} {'standing':34} {'last run':20} {'took':>7}  note")
    for stage, prior, standing in pipeline.status():
        when = prior.finished_at[:19].replace("T", " ") if prior else "–"
        took = f"{prior.ms / 1000:.1f}s" if prior else "–"
        note = (prior.note or (prior.error or "").splitlines()[0]) if prior else ""
        if len(note) > 110:
            note = note[:107] + "..."
        print(f"{stage.name:9} {standing:34} {when:20} {took:>7}  {note}")
    print()
    describe_indexes(index_dir)
    print()
    describe_releases(Releases.load(releases_path))


def describe_indexes(index_dir: Path) -> None:
    print("index versions")
    if not index_dir.exists():
        print("  none")
        return
    for root in sorted(d for d in index_dir.iterdir() if d.is_dir()):
        store = Store(root)
        current = store.current()
        for v in store.versions():
            m = Index.load(root / v).manifest
            mark = "*" if v == current else " "
            print(f"  {mark} {root.name}/{v}  {m.chunks} chunks, {m.chunking.label}, built {m.built_at[:10]}")


def describe_releases(releases: Releases) -> None:
    print("releases")
    if not releases.releases:
        print("  none; `pipeline release --candidate <run>` makes the first")
        return
    for r in releases.releases:
        mark = "*" if r.name == releases.current else " "
        print(f"  {mark} {r.name}  {r.manifest.version}  run {r.run}  {r.created_at[:10]}  {r.gates}")
    live = releases.live
    if live:
        print(f"current: {live.name}")
        for line in live.manifest.lines():
            print(f"  {line}")


def manage_index(args: argparse.Namespace, model: str, index_dir: Path) -> None:
    store = Store(index_dir / model)
    if args.use:
        store.point(args.use)
        print(f"{model} now points at {args.use}")
    elif args.remove:
        store.remove(args.remove)
        print(f"removed {model}/{args.remove}")
    else:
        describe_indexes(index_dir)


def release(
    args: argparse.Namespace,
    index_dir: Path,
    releases_path: Path = RELEASES,
    frozen: Frozen | None = None,
    safety: tuple[str, bool] | None = None,
) -> int:
    """`frozen` is the freeze the caller checked against the current files and `safety` the candidate
    version's graded safety run (its name, whether it passes); None skips either check."""
    releases = Releases.load(releases_path)
    candidate = Run.load(RUNS_DIR / args.candidate)
    baseline = Run.load(RUNS_DIR / releases.live.run) if releases.live else None
    try:
        made, comparison = promote(
            candidate, releases, load_gates(), baseline, args.note, frozen=frozen, safety=safety
        )
    except ReleaseBlocked as blocked:
        for r in blocked.comparison.results:
            if not r.passed:
                print(f"{'FAIL' if r.gate.critical else 'warn'}  {r.gate.metric}: {r.detail}")
        print(f"refused: {blocked}; {releases.current} stays the current release")
        return 1
    releases.save(releases_path)
    if comparison is not None:
        for r in comparison.results:
            print(
                f"{'pass' if r.passed else ('FAIL' if r.gate.critical else 'warn')}  {r.gate.metric}: {r.detail}"
            )
    print(f"{made.name} is the current release ({made.gates}); {apply_index(made, index_dir)}")
    return 0


def roll_back(index_dir: Path, releases_path: Path = RELEASES) -> None:
    releases = Releases.load(releases_path)
    left, now_live = rollback(releases)
    releases.save(releases_path)
    print(
        f"rolled back from {left.name} to {now_live.name} ({now_live.manifest.version}, run {now_live.run})"
    )
    print(apply_index(now_live, index_dir))
    changed = now_live.manifest.differs(left.manifest)
    if changed:
        print("what changed: " + ", ".join(changed))


__all__ = [
    "add_pipeline_commands",
    "index_for",
    "lineage",
    "load_prompt",
    "manage_index",
    "release",
    "resolve_version",
    "roll_back",
    "run_pipeline",
    "status",
    "Pipeline",
]
