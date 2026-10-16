from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_evaluation import two_runs  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
from infra_docs_rag.evaluation.versions import Version
from infra_docs_rag.generate.cache import CachedGenerator, GenerationCache
from infra_docs_rag.generate.prompts import load_prompt
from infra_docs_rag.observe.trace import TraceLog
from infra_docs_rag.pipeline.store import Store
from infra_docs_rag.retrieve.retriever import Rerank, Retrieval, Retriever
from infra_docs_rag.service.app import Service, create_app


@pytest.fixture
def client(fake_embedder, fake_reranker, fake_generator, guide, tmp_path, monkeypatch):  # noqa: F811
    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    store = Store(tmp_path / "index" / "bge-small")
    index_id, index, _ = store.publish(embedder, chunks, "test", DEFAULT_CHUNKING)
    retrieval = Retrieval(k=2, mode="hybrid", min_score=0.25, rerank=Rerank(model="bge-base", candidates=3))
    version = Version(name="v-test", retrieval=retrieval)
    inner = fake_generator({"kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])]})
    gen = CachedGenerator(inner, GenerationCache(tmp_path / "cache.jsonl"), "cite-v1+t")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    service = Service(
        version=version,
        index=index,
        index_id=index_id,
        retriever=Retriever(index, embedder, fake_reranker("bge-base")),
        generator=gen,
        prompt=load_prompt(),
        judge=None,
        traces=TraceLog(tmp_path / "traces.jsonl"),
        releases_path=tmp_path / "releases.yaml",
        index_dir=tmp_path / "index",
    )
    with TestClient(create_app(service)) as c:
        yield c, inner


def test_answer_returns_the_cited_answer_the_sources_and_a_trace(client):
    c, inner = client
    r = c.post("/answer", json={"question": "kubectl scale replicas"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["outcome"] == "answered" and body["answer"].endswith("[1]") and body["cited_share"] == 1.0
    assert body["citations"][0]["source_id"] == "guide" and body["citations"][0]["n"] == 1
    assert body["sources"][0]["relevance"] is not None and body["version"] == "v-test"
    assert body["lineage"]["prompt_version"].startswith("cite-v1+") and body["stages_ms"]["generate"] == 1.0
    assert not body["cached"] and body["trace_id"]
    again = c.post("/answer", json={"question": "kubectl scale replicas"}).json()
    assert again["cached"] and again["answer"] == body["answer"] and len(inner.calls) == 1
    fresh = c.post("/answer", json={"question": "kubectl scale replicas", "no_cache": True}).json()
    assert not fresh["cached"] and len(inner.calls) == 2

    trace = c.get(f"/traces/{body['trace_id']}")
    assert trace.status_code == 200 and trace.json()["question"] == "kubectl scale replicas"
    assert c.get("/traces/nope").status_code == 404
    assert len(c.get("/traces?last=2").json()) == 2
    summary = c.get("/traces/summary").json()
    assert summary["traces"] == 3 and "outcomes:" in summary["text"]


def test_an_out_of_scope_question_abstains_without_a_call(client):
    c, inner = client
    body = c.post("/answer", json={"question": "sourdough recipe"}).json()
    assert body["outcome"] == "no-passage" and body["answer"].startswith(
        "The indexed documentation does not cover"
    )
    assert (
        body["citations"] == []
        and body["reason"].startswith("the closest chunk scored")
        and len(inner.calls) == 0
    )


def test_a_failing_model_is_an_error_outcome_not_a_500(client):
    c, inner = client

    def boom(context, question):
        raise ValueError("`claude -p` exited 1: rate limited")

    inner.generate = boom
    body = c.post("/answer", json={"question": "kubectl scale replicas", "no_cache": True}).json()
    assert body["outcome"] == "error" and body["answer"].startswith(
        "The indexed documentation does not cover"
    )
    assert "rate limited" in body["reason"] and body["trace_id"]


def test_health_release_and_versions(client):
    c, _ = client
    health = c.get("/health")
    assert health.status_code == 200
    body = health.json()
    assert body["status"] in ("ok", "warn") and {x["name"] for x in body["checks"]} >= {
        "index",
        "release",
        "generator",
    }
    release = c.get("/release").json()
    assert release["serving"]["version"] == "v-test" and release["release"] is None
    versions = c.get("/versions").json()
    assert "default" in versions and "v2-hybrid-rerank" in versions
    assert c.post("/answer", json={"question": ""}).status_code == 422


def test_versions_report(two_runs, tmp_path):  # noqa: F811
    from infra_docs_rag.evaluation.gates import load_gates
    from infra_docs_rag.evaluation.versions_report import write_report
    from infra_docs_rag.pipeline.releases import Releases

    base, cand = two_runs
    write_report(
        [base, cand, base],
        load_gates(Path(__file__).parents[1] / "eval" / "gates.yaml"),
        Releases(),
        tmp_path / "versions.md",
    )
    text = (tmp_path / "versions.md").read_text()
    assert "3 versions of the pipeline" in text and "| right passage retrieved |" in text
    assert "### v1 → v2: **blocked" in text and "### v2 → v1: **passes" in text
    assert "## Rollback decision criteria" in text and "**FAIL**" in text


def test_feedback_is_recorded_against_a_trace(client, tmp_path):
    from infra_docs_rag.observe.feedback import FeedbackLog

    c, _ = client
    c.app.state.service.feedback = FeedbackLog(tmp_path / "feedback.jsonl")
    body = c.post("/answer", json={"question": "kubectl scale replicas"}).json()
    r = c.post("/feedback", json={"trace_id": body["trace_id"], "helpful": False, "note": "too short"})
    assert r.status_code == 200 and r.json()["helpful"] is False and r.json()["at"]
    assert c.post("/feedback", json={"trace_id": "nope", "helpful": True}).status_code == 404
    saved = FeedbackLog(tmp_path / "feedback.jsonl").by_trace()
    assert saved[body["trace_id"]].note == "too short"
