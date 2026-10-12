"""Is the service in a state to answer? Each check is one thing that has to hold: an index to read,
the released index being the one on disk, the records not newer than the index, a way to reach the
model, and the recent traces not full of errors. `infra-docs-rag pipeline health` prints them and
exits 1 when one fails; the capstone's `/health` endpoint returns the same."""

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..embed.embedders import MODELS
from ..pipeline.releases import Releases
from ..pipeline.stages import Pipeline
from ..pipeline.store import Store
from .summary import ERROR_RATE
from .trace import TraceLog

Status = Literal["ok", "warn", "fail"]


@dataclass
class Check:
    name: str
    status: Status
    detail: str


def model_name(embedding_version: str) -> str | None:
    model_id = embedding_version.split("@")[0]
    return next((spec.name for spec in MODELS.values() if spec.model_id == model_id), None)


def checks(
    index_dir: Path,
    default_model: str,
    releases: Releases,
    pipeline: Pipeline | None = None,
    traces: TraceLog | None = None,
    recent: int = 50,
) -> list[Check]:
    out: list[Check] = []
    live = releases.live
    model = (model_name(live.manifest.embedding_version) if live else None) or default_model
    store = Store(index_dir / model)
    current = store.current()
    if current is None:
        out.append(Check("index", "fail", f"no index for {model} under {index_dir}; run `pipeline run`"))
    else:
        out.append(Check("index", "ok", f"{model}/{current} is current"))
    if live is None:
        out.append(Check("release", "warn", "no release; `answer` runs the defaults"))
    elif current and live.manifest.index_version != current:
        out.append(
            Check(
                "release",
                "warn",
                f"{live.name} was measured on index {live.manifest.index_version}; {current} is what answers",
            )
        )
    else:
        out.append(Check("release", "ok", f"{live.name} ({live.manifest.version}), run {live.run}"))
    if pipeline is not None:
        standings = {stage.name: standing for stage, _, standing in pipeline.status()}
        embed = standings.get("embed", "never run")
        if embed == "ok":
            out.append(Check("freshness", "ok", "the index is built from the current records"))
        elif embed == "never run":
            out.append(
                Check("freshness", "warn", "the orchestrator has not run; `pipeline run` records the chain")
            )
        else:
            out.append(
                Check(
                    "freshness", "warn", f"embed stage is {embed}; `pipeline run` brings the index up to date"
                )
            )
    if shutil.which("claude") or os.environ.get("ANTHROPIC_API_KEY"):
        how = "claude on PATH" if shutil.which("claude") else "ANTHROPIC_API_KEY set"
        out.append(Check("generator", "ok", how))
    else:
        out.append(
            Check("generator", "fail", "neither `claude` on PATH nor ANTHROPIC_API_KEY: no way to answer")
        )
    out.append(
        Check("judge", "ok", "jev on PATH")
        if shutil.which("jev")
        else Check("judge", "warn", "jev is not on PATH; benchmarks run with --no-judge")
    )
    if traces is not None:
        last = traces.read(last=recent)
        if not last:
            out.append(Check("traces", "warn", "no traces yet"))
        else:
            errors = sum(1 for t in last if t.outcome == "error")
            degraded = sum(1 for t in last if t.degraded)
            rate = errors / len(last)
            detail = f"last {len(last)}: {errors} errors, {degraded} degraded, newest {last[-1].at[:19].replace('T', ' ')}"
            out.append(Check("traces", "warn" if rate > ERROR_RATE or degraded else "ok", detail))
    return out


def worst(found: list[Check]) -> Status:
    if any(c.status == "fail" for c in found):
        return "fail"
    if any(c.status == "warn" for c in found):
        return "warn"
    return "ok"


def format_checks(found: list[Check]) -> str:
    return "\n".join(f"{c.status:5} {c.name:10} {c.detail}" for c in found) + f"\noverall: {worst(found)}"
