"""Bring a saved run up to the current record schema: fill the fields a later version of the code
records (passage headings from the index, the claim share and gap sentences, the verdict under the
current rule) from what the run already holds. Nothing that needs a model call is touched: the judge
readings stay as the run's judge read them.

    uv run python scripts/migrate_runs.py eval/runs/<name> [...]
"""

import sys
from pathlib import Path

from infra_docs_rag.embed.index import Index
from infra_docs_rag.evaluation.runs import Run
from infra_docs_rag.generate.answer import is_gap
from infra_docs_rag.generate.evaluate import CLAIM_FLOOR
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


def migrate(directory: Path, chunks: dict) -> None:
    run = Run.load(directory)
    changed = 0
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
