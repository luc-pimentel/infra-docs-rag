from pathlib import Path

import pytest
from test_evaluation import ADVERSARIAL, BENCH
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
from infra_docs_rag.embed.index import build
from infra_docs_rag.generate.answer import answer
from infra_docs_rag.generate.evaluate import evaluate_grounding
from infra_docs_rag.generate.judge import Judgement
from infra_docs_rag.observe.canary import CanaryQuestion, check, load_canary, run_canary
from infra_docs_rag.observe.health import Check, checks, worst
from infra_docs_rag.observe.summary import P95_TOTAL_MS, percentile, summarize, write_report
from infra_docs_rag.observe.trace import Trace, TraceLog, render, trace_answer
from infra_docs_rag.pipeline.manifest import PipelineManifest
from infra_docs_rag.pipeline.releases import Release, Releases
from infra_docs_rag.retrieve.retriever import Rerank, Retrieval, Retriever

ROOT = Path(__file__).parents[1]


@pytest.fixture
def small(fake_embedder, fake_reranker, guide):  # noqa: F811
    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    return index, embedder, fake_reranker("bge-base")


def lineage(index_version="idx1") -> PipelineManifest:
    return PipelineManifest(
        version="v2",
        ingest_version="0.1.0+t",
        chunking_version="c",
        embedding_version="test/fake@r1",
        index_version=index_version,
        prompt_version="cite-v1+abc",
        retrieval_version="v2+def",
    )


def test_a_trace_holds_both_orders_every_stage_time_and_what_was_cited(small, fake_generator, tmp_path):
    index, embedder, reranker = small
    r = Retriever(index, embedder, reranker)
    retrieval = Retrieval(k=2, mode="hybrid", min_score=0.1, rerank=Rerank(model="bge-base", candidates=3))
    gen = fake_generator({"kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])]})
    a = answer("kubectl scale replicas", r, gen, retrieval)
    reading = Judgement(
        grounded=0.9, fabricated=0.1, gap_named=0.0, model="fake-jev", cost_usd=0.001, ms=12.0
    )
    t = trace_answer(a, "v2", lineage(), "cite-v1+abc", reading, gen.prices)
    assert t.outcome == "answered" and t.version == "v2" and t.lineage.index_version == "idx1"
    assert set(t.stages) >= {"embed", "dense", "lexical", "fuse", "rerank", "pack", "generate", "judge"}
    assert t.stages["generate"] == 1.0 and t.stages["judge"] == 12.0 and t.total_ms >= 13.0
    assert len(t.first_stage) == 3 and [h.rank for h in t.first_stage] == [1, 2, 3]
    assert 1 <= len(t.context) <= 2 and all(
        h.relevance is not None and h.first_rank is not None for h in t.context
    )
    assert t.first_stage[0].lexical is not None and t.context[0].cosine is not None
    assert t.citations[0].n == 1 and t.answer.endswith("[1]") and t.cited_share == 1.0
    assert t.judgement.grounded == 0.9 and t.cost_usd > 0 and t.input_tokens == 100
    text = render(t)
    assert (
        "first stage (" in text
        and "context (reranked)" in text
        and "was " in text
        and "judge: fake-jev" in text
    )

    log = TraceLog(tmp_path / "traces.jsonl")
    log.append(t)
    log.append(trace_answer(answer("sourdough recipe", r, gen, retrieval), "v2"))
    back = log.read()
    assert [x.trace_id for x in back] == [t.trace_id, back[1].trace_id] and back[1].outcome in (
        "no-passage",
        "abstained",
    )
    assert log.find(t.trace_id) == t and log.find("nope") is None and log.read(last=1) == [back[1]]


def test_a_reranker_that_fails_degrades_to_the_first_stage(small, fake_generator):
    index, embedder, reranker = small

    class Broken:
        name, model_id, revision, parameters, max_tokens = reranker.name, "x", "r1", 0, 512

        def score(self, query, texts):
            raise RuntimeError("CUDA out of memory")

    r = Retriever(index, embedder, Broken())
    retrieval = Retrieval(k=2, mode="dense", rerank=Rerank(model="bge-base", candidates=3))
    ctx = r.retrieve("kubectl scale replicas", retrieval)
    assert len(ctx.sources) == 2 and ctx.degraded.startswith("reranker bge-base failed (RuntimeError: CUDA")
    assert [h.rank for h in ctx.first_stage] == [1, 2, 3] and ctx.sources[0].hit.rank == 1
    assert "rerank" in ctx.timings
    t = trace_answer(answer("kubectl scale replicas", r, fake_generator(), retrieval))
    assert t.degraded and "degraded:" in render(t)


def test_the_summary_reads_rates_percentiles_and_flags(small, fake_generator):
    index, embedder, reranker = small
    assert percentile([1, 2, 3, 4], 0.5) == 2.5 and percentile([5], 0.95) == 5 and percentile([], 0.5) == 0
    r = Retriever(index, embedder, reranker)
    gen = fake_generator({"kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])]})
    retrieval = Retrieval(k=2, mode="dense", min_score=0.1)
    traces = [
        trace_answer(
            answer("kubectl scale replicas", r, gen, retrieval), "v2", lineage("idx1"), prices=gen.prices
        ),
        trace_answer(answer("sourdough recipe", r, gen, retrieval), "v2", lineage("idx0")),
        trace_answer(answer("maxSurge", r, gen, retrieval), "v1", lineage("idx1")),
    ]
    traces[1].outcome, traces[2].outcome = (
        "no-passage",
        "abstained",
    )  # pinned: the fake geometry is not the point
    traces[2].stages["generate"] = 20_000.0
    traces[2].total_ms = 20_000.0
    s = summarize(traces, current_index="idx1")
    assert s.traces == 3 and s.versions == {"v2": 2, "v1": 1} and s.indexes == {"idx1": 2, "idx0": 1}
    assert s.outcomes == {"answered": 1, "no-passage": 1, "abstained": 1}
    assert s.cited_share_median == 1.0 and s.cost_total > 0 and s.cost_per_1000 > 0
    assert s.total_p95 > P95_TOTAL_MS
    assert any("no passage" in f for f in s.flags) and any("p95" in f for f in s.flags)
    assert any("idx0, not the current idx1" in f for f in s.flags)
    assert summarize([], None).traces == 0 and summarize([], None).flags == []


def test_the_report_renders_the_summary_and_the_examples(small, fake_generator, tmp_path):
    index, embedder, reranker = small
    r = Retriever(index, embedder, reranker)
    gen = fake_generator({"kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])]})
    t = trace_answer(answer("kubectl scale replicas", r, gen, Retrieval(k=2, mode="dense")), "v2", lineage())
    write_report([t], tmp_path / "observability.md", [t], "idx1")
    text = (tmp_path / "observability.md").read_text()
    assert "1 traces" in text and "### kubectl scale replicas" in text and '"trace_id"' in text


def test_the_canary_checks_outcome_source_and_text(small, fake_generator):
    index, embedder, reranker = small
    r = Retriever(index, embedder, reranker)
    gen = fake_generator({"kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])]})
    questions = [
        CanaryQuestion(
            id="scale",
            text="kubectl scale replicas",
            expect="answer",
            source="guide",
            contains="kubectl scale",
        ),
        CanaryQuestion(id="wrong-source", text="kubectl scale replicas", expect="answer", source="other"),
        CanaryQuestion(id="oos", text="sourdough recipe", expect="abstain"),
        CanaryQuestion(id="should-answer", text="maxSurge", expect="answer"),
    ]
    seen = []
    results = run_canary(questions, r, gen, Retrieval(k=2, mode="dense"), on_answer=seen.append)
    assert [x.passed for x in results] == [True, False, True, False] and len(seen) == 4
    assert results[1].detail.startswith("answered but cited guide, not other")
    assert results[3].detail.startswith("expected an answer, got abstained")
    assert check(CanaryQuestion(id="x", text="q", expect="abstain"), results[0].answer) == (
        False,
        "expected an abstention, got answered",
    )
    shipped = load_canary(ROOT / "eval" / "canary.yaml")
    assert len(shipped) == 5 and {q.expect for q in shipped} == {"answer", "abstain"}
    assert all(q.source for q in shipped if q.expect == "answer")


def test_health_reads_the_index_the_release_and_the_traces(small, tmp_path, monkeypatch):
    from infra_docs_rag.pipeline.store import Store

    index, embedder, _ = small
    index_dir = tmp_path / "index"
    Store(index_dir / "bge-small").publish(embedder, index.chunks, "test", DEFAULT_CHUNKING)
    current = Store(index_dir / "bge-small").current()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    found = checks(index_dir, "bge-small", Releases())
    by = {c.name: c for c in found}
    assert by["index"].status == "ok" and current in by["index"].detail
    assert by["release"].status == "warn" and by["generator"].status == "ok"

    pinned = lineage("stale").model_copy(update={"embedding_version": "BAAI/bge-small-en-v1.5@5c38ec7c405e"})
    releases = Releases(
        current="r1",
        releases=[Release(name="r1", created_at="2026-10-12T00:00:00+00:00", run="r", manifest=pinned)],
    )
    log = TraceLog(tmp_path / "traces.jsonl")
    found = checks(index_dir, "bge-small", releases, traces=log)
    by = {c.name: c for c in found}
    assert by["release"].status == "warn" and "measured on index stale" in by["release"].detail
    assert by["traces"].status == "warn" and worst(found) == "warn"
    assert worst([Check("x", "fail", "")]) == "fail" and worst([Check("x", "ok", "")]) == "ok"

    found = checks(tmp_path / "nowhere", "bge-small", Releases())
    assert {c.name: c.status for c in found}["index"] == "fail"


def test_a_benchmark_run_writes_one_trace_per_case(small, fake_generator, guide):  # noqa: F811
    index, embedder, _ = small
    traces: list[Trace] = []
    evaluate_grounding(
        index,
        embedder,
        [guide],
        BENCH,
        [],
        ADVERSARIAL,
        fake_generator(),
        retrieval=Retrieval(k=2, mode="dense"),
        tracer=lambda a, j: traces.append(trace_answer(a, "v1")),
    )
    assert len(traces) == 5 and {t.outcome for t in traces} <= {"abstained", "no-passage", "answered"}
