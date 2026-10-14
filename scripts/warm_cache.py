"""Fill the generation cache from a saved run, with no model call: every answered case holds the model's
segments and citations for a question against known chunks, which is exactly a cache entry. Warming a
fresh deployment from the release's run means the benchmark questions, and anything a user repeats from
them, cost nothing.

    uv run python scripts/warm_cache.py eval/runs/<name>
"""

import sys
from pathlib import Path

from infra_docs_rag.evaluation.runs import Run
from infra_docs_rag.generate.cache import CACHE, GenerationCache, key_for
from infra_docs_rag.generate.generator import Cited, Generated, Segment, Usage
from infra_docs_rag.retrieve.retriever import Context, Retrieval


class Stub:
    def __init__(self, chunk_id: str) -> None:
        self.chunk = type("Chunk", (), {"chunk_id": chunk_id})()


def warm(directory: Path, cache: GenerationCache) -> int:
    run = Run.load(directory)
    lineage = run.manifest.lineage
    if lineage is None:
        raise SystemExit(f"{directory} has no lineage; the prompt version is needed for the key")
    added = 0
    for c in run.cases:
        g = c.generated
        if g is None or g.stop_reason != "end_turn" or g.refusal:
            continue
        context = Context(
            query=c.question,
            retrieval=Retrieval(),
            sources=[Stub(s.chunk_id) for s in c.sources],
            candidates=0,
            top_score=None,
            below=0,
            over_budget=0,
        )
        key = key_for(g.model, lineage.prompt_version, c.question, context)
        if cache.get(key) is not None:
            continue
        cache.put(
            key,
            Generated(
                segments=[
                    Segment(text=s.text, citations=[Cited(x.document, x.cited_text) for x in s.citations])
                    for s in c.segments
                ],
                stop_reason=g.stop_reason,
                model=g.model,
                usage=Usage(g.usage.input_tokens, g.usage.output_tokens, g.usage.cache_read_tokens),
                ms=g.ms,
                refusal=g.refusal,
                unverified=g.unverified,
            ),
        )
        added += 1
    return added


if __name__ == "__main__":
    cache = GenerationCache(CACHE)
    for arg in sys.argv[1:]:
        n = warm(Path(arg), cache)
        print(f"{arg}: {n} entries added, {len(cache)} in the cache")
