from pathlib import Path

import pytest

from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
from infra_docs_rag.chunk.evaluate import resolve_answer
from infra_docs_rag.embed.evaluate import Target
from infra_docs_rag.embed.index import build
from infra_docs_rag.ingest.parsers.base import assemble, markdown_parts
from infra_docs_rag.retrieve.evaluate import (
    KINDS,
    BenchmarkQuery,
    BenchmarkSet,
    ThresholdRow,
    evaluate_retrieval,
    load_benchmark,
    pick_threshold,
)
from infra_docs_rag.retrieve.report import write_report
from infra_docs_rag.retrieve.retriever import Filter

BENCHMARK = Path(__file__).parents[1] / "eval" / "retrieval.yaml"

GUIDE = """# Deployments

A Deployment manages a set of Pods.

## Rolling Update

Set maxSurge to allow extra Pods during an update.

## Scaling

Run kubectl scale to change the number of replicas.
"""


def test_the_benchmark_file_is_well_formed():
    bench = load_benchmark(BENCHMARK)
    labelled = bench.in_scope
    assert 25 <= len(labelled) <= 50
    assert {q.kind for q in bench.queries} == set(KINDS)
    assert all(q.expect and q.hypothesis for q in labelled)
    assert len(bench.out_of_scope) >= 3 and all(q.expect is None for q in bench.out_of_scope)
    assert all(q.filter and not q.filter.empty for q in bench.queries if q.kind == "metadata")
    assert all(q.expect.contains for q in labelled if not q.expect.sections)  # a whole document needs a key


def test_a_query_is_labelled_the_way_its_kind_demands():
    target = Target(source="s", sections=["A"])
    with pytest.raises(ValueError, match="no `expect`"):
        BenchmarkQuery(id="x", kind="out-of-scope", text="t", expect=target, hypothesis="h")
    with pytest.raises(ValueError, match="needs one"):
        BenchmarkQuery(id="x", kind="paraphrase", text="t", hypothesis="h")
    with pytest.raises(ValueError, match="filter"):
        BenchmarkQuery(id="x", kind="metadata", text="t", expect=target, hypothesis="h")
    with pytest.raises(ValueError, match="unique"):
        load_benchmark_from(
            [BenchmarkQuery(id="x", kind="paraphrase", text="t", expect=target, hypothesis="h")] * 2
        )


def load_benchmark_from(queries, tmp_path=Path("/tmp")):
    import yaml

    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump({"queries": [q.model_dump(exclude_none=True) for q in queries]}))
    return load_benchmark(path)


def test_a_whole_document_label_needs_the_text_its_chunk_must_contain(make_doc):
    doc = make_doc("notes", "url: https://example.com\nstatusbadge.enabled: true")
    answer = resolve_answer(
        Target(source="notes", sections=[], contains="statusbadge.enabled"), {"notes": doc}, "x"
    )
    assert answer.spans == [(0, len(doc.cleaned_text))] and answer.contains == "statusbadge.enabled"
    with pytest.raises(ValueError, match="whole of notes"):
        resolve_answer(Target(source="notes", sections=[]), {"notes": doc}, "x")
    with pytest.raises(ValueError, match="never says"):
        resolve_answer(Target(source="notes", sections=[], contains="nope"), {"notes": doc}, "x")


def test_the_threshold_pick_keeps_answers_and_rejects_the_unanswerable():
    rows = [
        ThresholdRow(None, answered=5, silent=0, rejected=0, shown=5),
        ThresholdRow(0.6, answered=5, silent=0, rejected=2, shown=3),
        ThresholdRow(
            0.7, answered=4, silent=1, rejected=3, shown=2
        ),  # the same total, but a real question goes silent
        ThresholdRow(0.8, answered=2, silent=3, rejected=3, shown=1),
    ]
    assert pick_threshold(rows).min_score == 0.6


@pytest.fixture
def guide(make_doc):
    text, sections, _ = assemble(markdown_parts(GUIDE))
    doc = make_doc("guide", text)
    doc.sections = sections
    return doc


def test_a_small_benchmark_runs_end_to_end(fake_embedder, guide, tmp_path):
    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    bench = BenchmarkSet(
        queries=[
            BenchmarkQuery(
                id="scale",
                kind="paraphrase",
                text="kubectl scale replicas",
                expect=Target(source="guide", sections=["Scaling"]),
                hypothesis="shares words",
            ),
            BenchmarkQuery(
                id="surge",
                kind="identifier",
                text="maxSurge",
                expect=Target(source="guide", sections=["Rolling Update"], contains="maxSurge"),
                hypothesis="exact word",
            ),
            BenchmarkQuery(
                id="scaling-section",
                kind="metadata",
                text="what does the scaling part say",
                expect=Target(source="guide", sections=["Scaling"]),
                filter=Filter(section="Scaling"),
                hypothesis="the filter leaves one candidate",
            ),
            BenchmarkQuery(id="bread", kind="out-of-scope", text="sourdough recipe", hypothesis="nothing"),
        ]
    )
    ev = evaluate_retrieval(index, embedder, [guide], bench, earlier_ids={"scale"}, sizes=(200,))
    assert [r.rank for r in ev.ranked] == [1, 1, 1] and ev.mrr == 1.0
    assert ev.hits(1) == 3 and ev.earlier == ["scale"]
    assert [u.query.id for u in ev.unanswerable] == ["bread"] and len(ev.unanswerable[0].hits) == len(chunks)
    assert ev.thresholds[0].min_score is None and ev.thresholds[0].answered == 3
    assert len(ev.filtered) == 1 and ev.filtered[0].candidates == 1 and ev.filtered[0].prefiltered == 1
    assert [p.chunks for p in ev.scale] == [200] and ev.embed_ms >= 0
    assert len(ev.plain) == 4 and len(ev.with_filters) == 1
    write_report(ev, tmp_path / "retrieval.md")
    report = (tmp_path / "retrieval.md").read_text()
    assert report.startswith("# Retrieval report") and "## 7. The pick" in report and "`bread`" in report


def test_the_evaluation_insists_on_the_default_chunking(fake_embedder, guide):
    embedder = fake_embedder()
    chunks = chunk_documents([guide], DEFAULT_CHUNKING.model_copy(update={"context": False}), embedder)
    index = build(
        embedder,
        chunks,
        ingest_version="test",
        chunking=DEFAULT_CHUNKING.model_copy(update={"context": False}),
    )
    with pytest.raises(ValueError, match="run `infra-docs-rag embed`"):
        evaluate_retrieval(index, embedder, [guide], BenchmarkSet(queries=[]), sizes=(10,))


def test_the_hybrid_comparison_reports_its_pick(fake_embedder, guide, tmp_path):
    from infra_docs_rag.retrieve.compare import compare
    from infra_docs_rag.retrieve.hybrid_report import write_report as write_hybrid_report
    from infra_docs_rag.retrieve.retriever import DEFAULT_RETRIEVAL

    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    bench = BenchmarkSet(
        queries=[
            BenchmarkQuery(
                id="scale",
                kind="paraphrase",
                text="kubectl scale replicas",
                expect=Target(source="guide", sections=["Scaling"]),
                hypothesis="shares words",
            ),
            BenchmarkQuery(id="bread", kind="out-of-scope", text="sourdough recipe", hypothesis="nothing"),
        ]
    )
    extra = [
        BenchmarkQuery(
            id="surge",
            kind="identifier",
            text="maxSurge",
            expect=Target(source="guide", sections=["Rolling Update"], contains="maxSurge"),
            hypothesis="BM25 should win: the key is in one chunk, whole",
        )
    ]
    cmp = compare(index, embedder, [guide], bench, extra)
    assert cmp.extra == ["surge"] and [q.id for q in cmp.in_scope] == ["scale", "surge"]
    assert cmp.lexical.rank("surge") == 1 and cmp.lexical.rank("bread") is None
    assert DEFAULT_RETRIEVAL.mode == "hybrid" and DEFAULT_RETRIEVAL.fusion.strategy == "weighted"
    assert DEFAULT_RETRIEVAL.rerank is not None and DEFAULT_RETRIEVAL.rerank.model == "bge-base"
    assert DEFAULT_RETRIEVAL.label == (
        "top 5 · hybrid (weighted α=0.5 · depth 50) · rerank bge-base over 20 · min 0.50"
    )
    write_hybrid_report(cmp, tmp_path / "hybrid-search.md")
    report = (tmp_path / "hybrid-search.md").read_text()
    assert report.startswith("# Hybrid search report") and "1 written for this stage" in report
    assert "## 5. The pick, and what still fails" in report
    assert (
        "The first stage of `retrieve` and `search` is `top 5 · hybrid (weighted α=0.5 · depth 50) · min 0.50`"
        in report
    )
    assert "Of the 1 questions written for this stage" in report and "`surge`" in report


def test_the_rerank_comparison_times_each_stage_and_reports_the_pick(
    fake_embedder, fake_reranker, guide, tmp_path
):
    from infra_docs_rag.retrieve.rerank_compare import (
        Expectation,
        compare_rerank,
        configurations,
        load_expectations,
    )
    from infra_docs_rag.retrieve.rerank_report import write_report as write_rerank_report
    from infra_docs_rag.retrieve.retriever import Retrieval

    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    bench = BenchmarkSet(
        queries=[
            BenchmarkQuery(
                id="scale",
                kind="paraphrase",
                text="kubectl scale replicas",
                expect=Target(source="guide", sections=["Scaling"]),
                hypothesis="shares words",
            ),
            BenchmarkQuery(
                id="surge",
                kind="identifier",
                text="maxSurge",
                expect=Target(source="guide", sections=["Rolling Update"], contains="maxSurge"),
                hypothesis="exact word",
            ),
            BenchmarkQuery(id="bread", kind="out-of-scope", text="sourdough recipe", hypothesis="nothing"),
        ]
    )
    first = Retrieval(k=2, mode="dense")
    configs = configurations(first, depths=(3,))
    assert [c.rerank.label if c.rerank else "first" for c in configs] == [
        "first",
        "rerank minilm-l6 over 3",
        "rerank bge-base over 3",
    ]
    (tmp_path / "rerank.yaml").write_text(
        "expectations:\n  - id: scale\n    moves: false\n    hypothesis: already first\n"
    )
    expectations = load_expectations(tmp_path / "rerank.yaml", bench.queries)
    assert expectations == [Expectation(id="scale", moves=False, hypothesis="already first")]
    with pytest.raises(ValueError, match="not a benchmark query"):
        load_expectations(Path(__file__).parents[1] / "eval" / "rerank.yaml", bench.queries[:1])

    rerankers = {"minilm-l6": fake_reranker("minilm-l6"), "bge-base": fake_reranker("bge-base")}
    cmp = compare_rerank(index, embedder, [guide], bench, [], rerankers, expectations, configs=configs)
    assert [r.reranker.name if r.reranker else None for r in cmp.runs] == [None, "minilm-l6", "bge-base"]
    assert cmp.first.rank("scale") == 1 and cmp.first.outcomes["scale"].scored == 0
    reranked = cmp.runs[1]
    assert reranked.rank("surge") == 1 and reranked.outcomes["surge"].scored == 3
    assert reranked.outcomes["surge"].rerank_ms >= 0 and reranked.outcomes["surge"].first_ms >= 0
    assert reranked.rank("bread") is None and reranked.outcomes["bread"].top[0].score == 0.0
    assert cmp.best() in cmp.runs and cmp.parameters == {"minilm-l6": 0, "bge-base": 0}
    assert cmp.pick() in cmp.runs and cmp.pick(budget_ms=0.0) is cmp.first

    write_rerank_report(cmp, tmp_path / "reranking.md")
    report = (tmp_path / "reranking.md").read_text()
    assert report.startswith("# Reranking report") and "## 1. Configurations" in report
    assert "## 2. Where reranking changes the context" in report and "## 3. Latency and cost" in report
    assert "## 4. Relevance as a signal" in report and "## 5. Expectations" in report
    assert "## 6. The pick, and what still fails" in report and "`scale` | stays" in report
