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
