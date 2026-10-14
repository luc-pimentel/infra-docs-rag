"""Bring a saved run up to the current record schema: fill the fields a later version of the code
records (passage headings from the index, the claim share and gap sentences, the verdict under the
current rule, the lineage manifest) from what the run already holds. Nothing that needs a model call is
touched: the judge readings stay as the run's judge read them. The lineage's index version is that of
the current index of the run's embedding model, which is right only while that index is the one the run
was made on (compare the manifests' `built_at`).

    uv run python scripts/migrate_runs.py eval/runs/<name> [...]
"""

import sys
from pathlib import Path

from infra_docs_rag.embed.index import Index
from infra_docs_rag.evaluation.runs import Run
from infra_docs_rag.generate.answer import is_gap
from infra_docs_rag.generate.evaluate import CLAIM_FLOOR
from infra_docs_rag.generate.prompts import load_prompt
from infra_docs_rag.pipeline.manifest import PipelineManifest
from infra_docs_rag.pipeline.store import Store
from infra_docs_rag.retrieve.retriever import locate


def verdict(c) -> str:
    o = c.outcome
    if c.expect == "abstain":
        return "held" if o in ("no-passage", "abstained") else "answered anyway"
    if c.expect == "answer":
        if o == "answered":
            if c.right_cited is False:
                return "cited the wrong passage"
            return "held" if c.claim_share >= CLAIM_FLOOR else "partly uncited"
        return "no-passage" if o == "no-passage" else o
    return "held" if o == "answered" else o


def lineage(run: Run) -> PipelineManifest | None:
    """The run's lineage from its manifest, against the current index of its model; None when that
    index is not the one the run read (a different build time)."""
    m = run.manifest
    store = Store(Path("data/index") / m.index.model)
    current = store.current()
    if current is None or store.load().manifest.built_at != m.index.built_at:
        return None
    made = PipelineManifest.of(
        m.version, m.index, current, load_prompt(), m.backend, m.retrieval, m.generator, run.name
    )
    return made.model_copy(update={"git_sha": m.git_sha, "created_at": m.created_at})


def migrate(directory: Path, chunks: dict) -> None:
    run = Run.load(directory)
    changed = 0
    if run.manifest.lineage is None:
        run.manifest.lineage = lineage(run)
        print(
            f"{directory.name}: lineage "
            + (
                f"filled, index {run.manifest.lineage.index_version}"
                if run.manifest.lineage
                else "left empty: the current index is not the one the run read"
            )
        )
    for c in run.cases:
        for s in c.sources:
            if not s.heading and s.chunk_id in chunks:
                ch = chunks[s.chunk_id]
                path = [ch.project, *ch.section_path] if ch.project else ch.section_path
                s.heading = f"[{s.n}] " + locate(path, ch.pages)
        cited = sum(len(seg.text.strip()) for seg in c.segments if seg.citations and seg.text.strip())
        claims = sum(len(u) for u in c.uncited if not is_gap(u))
        c.gap_sentences = [u for u in c.uncited if is_gap(u)]
        c.claim_share = cited / (cited + claims) if cited + claims else 0.0
        new = verdict(c)
        if new != c.verdict:
            print(f"{directory.name}: {c.id} {c.verdict} → {new}")
            changed += 1
        c.verdict = new
    run.save(directory)
    print(f"{directory.name}: {len(run.cases)} cases rewritten, {changed} verdicts changed")


if __name__ == "__main__":
    index = Index.load(Path("data/index/bge-small"))
    by_id = {c.chunk_id: c for c in index.chunks}
    for arg in sys.argv[1:]:
        migrate(Path(arg), by_id)
