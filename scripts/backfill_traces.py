"""Write `traces.jsonl` beside a dataset run made before stage 17, from what the run and the trace log
already hold: the saved answer, citations, judge and reader readings come from the run's cases and the
log entry the run wrote (matched by run tag and question); the parts the old trace did not record, the
reranker's whole order, what the pack step read and dropped, the tokens packed, come from replaying
the retrieval under the run's version against the current index, which is deterministic. The replay
is checked against the saved context: a record whose replayed passages differ from the ones the model
read is marked `degraded` in its trace and the diagnosis says so. The cases get their `trace_id`.

    uv run python scripts/backfill_traces.py eval/runs/<name> [...]
"""

import sys
from pathlib import Path

from infra_docs_rag.embed.embedders import load_embedder
from infra_docs_rag.evaluation.dataset import dataset_version, load_dataset
from infra_docs_rag.evaluation.diagnose import TRACES_FILE, save_traces
from infra_docs_rag.evaluation.runs import Run
from infra_docs_rag.evaluation.versions import load_versions
from infra_docs_rag.observe.trace import TRACES, Trace, TracedReading, TraceLog, drops, traced
from infra_docs_rag.pipeline import commands as pipeline_commands
from infra_docs_rag.retrieve.rerank import load_reranker
from infra_docs_rag.retrieve.retriever import Retriever


def main(paths: list[str]) -> None:
    ds = load_dataset()
    ds_version = dataset_version(ds.records)
    by_record = {r.id: r for r in ds.records}
    logged = TraceLog(TRACES).read()
    for p in paths:
        run_dir = Path(p)
        run = Run.load(run_dir)
        m = run.manifest
        if (run_dir / TRACES_FILE).exists():
            print(f"{run.name}: {TRACES_FILE} exists; leaving it")
            continue
        version = load_versions()[m.version]
        index, _ = pipeline_commands.index_for(
            version, m.lineage.index_version if m.lineage else None, Path("data/index")
        )
        retriever = Retriever(
            index,
            load_embedder(version.embedding),
            load_reranker(version.retrieval.rerank.model) if version.retrieval.rerank else None,
        )
        tag = f"{m.version}:{run.name}"
        pool = [t for t in logged if t.version == tag]
        print(
            f"{run.name}: {len(pool)} log entries under {tag}; replaying retrieval for {len(run.cases)} cases"
        )
        out: list[Trace] = []
        mismatched = 0
        for c in run.cases:
            if c.id not in by_record:
                continue
            old = next((t for t in pool if t.question == c.question), None)
            ctx = retriever.retrieve(c.question, version.retrieval)
            replayed = [s.chunk.chunk_id for s in ctx.sources]
            saved = [s.chunk_id for s in c.sources]
            degraded = None
            if replayed != saved:
                mismatched += 1
                degraded = f"replayed retrieval read {len(replayed)} passages, the run read {len(saved)}; the ranking below is the replay's"
            base = old.model_dump() if old else {}
            base.update(
                trace_id=old.trace_id if old else f"bf{abs(hash((run.name, c.id))) % 10**10:010d}",
                at=old.at if old else m.created_at,
                question=c.question,
                version=tag,
                retrieval=version.retrieval.label,
                lineage=m.lineage,
                prompt_version=m.lineage.prompt_version if m.lineage else "",
                stages=old.stages
                if old
                else {
                    **{k: round(v, 1) for k, v in ctx.timings.items()},
                    **({"generate": c.generated.ms} if c.generated else {}),
                },
                total_ms=old.total_ms if old else c.end_to_end_ms,
                candidates=ctx.candidates,
                first_stage=[traced(h) for h in ctx.first_stage],
                context=[traced(s.hit) for s in ctx.sources] if not old else old.context,
                top_score=ctx.top_score,
                below=ctx.below,
                degraded=degraded or (old.degraded if old else None),
                outcome=c.outcome,
                reason=c.reason,
                answer=c.marked,
                cited_share=c.cited_share,
                claim_share=c.claim_share,
                uncited=c.uncited,
                generator=c.generated.model if c.generated else "",
                input_tokens=c.generated.usage.input_tokens if c.generated else 0,
                output_tokens=c.generated.usage.output_tokens if c.generated else 0,
                cost_usd=round(c.generated.usage.cost(m.prices), 6) if c.generated else 0.0,
                judgement=(
                    {
                        "grounded": c.judgement.grounded,
                        "fabricated": c.judgement.fabricated,
                        "gap_named": c.judgement.gap_named,
                        "model": c.judgement.model,
                        "ms": c.judgement.ms,
                        "cost_usd": c.judgement.cost_usd,
                    }
                    if c.judgement
                    else None
                ),
                example_id=c.id,
                run=run.name,
                dataset_version=ds_version if m.dataset == ds_version else m.dataset,
                reranked=[traced(h) for h in ctx.reranked],
                ranked=[traced(h) for h in ctx.ranked],
                dropped=[d.model_dump() for d in drops(ctx)],
                tokens=ctx.tokens if ctx.sources else 0,
                budget=version.retrieval.budget,
                over_budget=ctx.over_budget,
                reading=TracedReading(
                    relevant=c.reading.relevant,
                    reference_agrees=c.reading.reference_agrees,
                    deprecation_named=c.reading.deprecation_named,
                    readings_named=c.reading.readings_named,
                    model=c.reading.model,
                    ms=c.reading.ms,
                    cost_usd=c.reading.cost_usd,
                ).model_dump()
                if c.reading
                else None,
            )
            if "citations" in base and old:
                base["citations"] = [x.model_dump() for x in old.citations]
            t = Trace.model_validate(base)
            c.trace_id = t.trace_id
            out.append(t)
            print(
                f"  {c.id}: {'ok' if degraded is None else 'replay differs'}"
                + ("" if old else " (no log entry; built from the case)")
            )
        save_traces(out, run_dir)
        run.save(run_dir)
        print(
            f"{run.name}: wrote {len(out)} traces, {mismatched} with a replay that differs from the saved context"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
