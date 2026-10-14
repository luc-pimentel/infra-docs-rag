from pathlib import Path

import httpx
import pytest
import yaml
from test_evaluation import two_runs  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.chunk.chunks import Chunking
from infra_docs_rag.embed.index import Index, build
from infra_docs_rag.evaluation.gates import load_gates
from infra_docs_rag.evaluation.runs import Run
from infra_docs_rag.ingest import sources as sources_module
from infra_docs_rag.ingest.sources import FetchFailed, Source, fetch
from infra_docs_rag.pipeline.commands import resolve_version
from infra_docs_rag.pipeline.manifest import PipelineManifest, chunking_version, index_version
from infra_docs_rag.pipeline.releases import Release, ReleaseBlocked, Releases, promote, rollback
from infra_docs_rag.pipeline.stages import (
    Artifact,
    Config,
    Paths,
    Pipeline,
    Stage,
    StageFailed,
    State,
    build_pipeline,
    file_fingerprint,
)
from infra_docs_rag.pipeline.store import IndexBuildFailed, Store
from infra_docs_rag.retrieve.retriever import DEFAULT_RETRIEVAL, Retrieval

ROOT = Path(__file__).parents[1]

GUIDE = """# Deployments

A Deployment manages a set of Pods.

## Rolling Update

Set maxSurge to allow extra Pods during an update.

## Scaling

Run kubectl scale to change the number of replicas.
"""


# --- the index store ---------------------------------------------------------------------------


@pytest.fixture
def three(make_chunk):
    return [
        make_chunk("Rollback", "Roll back a Deployment to an earlier revision with kubectl rollout undo."),
        make_chunk("Scaling", "Scale a Deployment by changing its number of replicas."),
        make_chunk("Services", "A Service exposes Pods behind one stable virtual IP address and DNS name."),
    ]


def test_a_new_build_goes_live_and_the_old_version_stays(fake_embedder, three, tmp_path):
    store = Store(tmp_path / "fake")
    embedder = fake_embedder()
    v1, _, new = store.publish(embedder, three, "test", Chunking())
    assert new and store.current() == v1 and store.versions() == [v1]
    v2, _, new = store.publish(embedder, three[:2], "test", Chunking())
    assert new and v2 != v1 and store.current() == v2 and set(store.versions()) == {v1, v2}
    assert Index.load(tmp_path / "fake").manifest.chunks == 2  # the pointer is followed
    store.point(v1)
    assert Index.load(tmp_path / "fake").manifest.chunks == 3
    again, _, new = store.publish(embedder, three, "test", Chunking())
    assert again == v1 and not new  # the same content is the same version; nothing rebuilt
    with pytest.raises(ValueError, match="current"):
        store.remove(v1)
    store.remove(v2)
    assert store.versions() == [v1]


def test_a_build_that_fails_its_check_leaves_the_live_index_alone(fake_embedder, three, tmp_path):
    store = Store(tmp_path / "fake")
    embedder = fake_embedder()
    v1, _, _ = store.publish(embedder, three, "test", Chunking())

    class Lost(type(embedder)):
        """A model whose queries all land on the Services chunk, so a chunk's own text finds another."""

        def embed_queries(self, texts):
            return self.embed_documents([three[2].text] * len(texts))

    with pytest.raises(IndexBuildFailed, match="found src:Doc/Services, not src:Doc/Rollback"):
        store.publish(Lost(), three, "test", Chunking())
    assert store.current() == v1 and store.versions() == [v1]
    assert not [d for d in (tmp_path / "fake").iterdir() if d.name.startswith(".staging")]


def test_a_flat_index_is_adopted_into_a_version_directory(fake_embedder, three, tmp_path):
    index = build(fake_embedder(), three, "test")
    index.save(tmp_path / "fake")
    expected = index_version(tmp_path / "fake")
    store = Store(tmp_path / "fake")
    assert store.current() == expected and (tmp_path / "fake" / expected / "vectors.npy").exists()
    assert not (tmp_path / "fake" / "manifest.json").exists()
    assert Index.load(tmp_path / "fake").manifest.chunks == 3


def test_versions_are_content_hashes(fake_embedder, three, tmp_path):
    assert chunking_version(Chunking()) == chunking_version(Chunking())
    assert chunking_version(Chunking()).startswith("whole-sections+")
    assert chunking_version(Chunking(strategy="fixed", size=256)) != chunking_version(
        Chunking(strategy="fixed", size=512)
    )
    build(fake_embedder(), three, "test").save(tmp_path / "a")
    build(fake_embedder(), three, "test").save(tmp_path / "b")  # later, so a different built_at
    assert index_version(tmp_path / "a") == index_version(tmp_path / "b")
    build(fake_embedder(), three[:2], "test").save(tmp_path / "c")
    assert index_version(tmp_path / "c") != index_version(tmp_path / "a")


# --- the orchestrator --------------------------------------------------------------------------


@pytest.fixture
def chain(tmp_path):
    """Three stages passing one file to the next: A reads `seed` and writes `a`, B writes `b`, C writes
    `c`. B fails while `break_b` is set."""
    seed, a, b, c = (tmp_path / n for n in ("seed", "a", "b", "c"))
    seed.write_text("1")
    switches = {"break_b": False, "ran": []}

    def make(name, src, dst):
        def run():
            switches["ran"].append(name)
            if name == "B" and switches["break_b"]:
                raise RuntimeError("B blew up")
            dst.write_text(f"{name}({src.read_text()})")
            return f"wrote {dst.name}"

        return run

    artifacts = {
        n: Artifact(n, p, (lambda p=p: file_fingerprint(p)), made_by)
        for n, p, made_by in (("seed", seed, None), ("a", a, "A"), ("b", b, "B"), ("c", c, "C"))
    }
    stages = [
        Stage("A", ["seed"], ["a"], "v", make("A", seed, a)),
        Stage("B", ["a"], ["b"], "v", make("B", a, b)),
        Stage("C", ["b"], ["c"], "v", make("C", b, c)),
    ]
    return Pipeline(stages, artifacts, tmp_path / "state.json", log=lambda _: None), switches, (seed, a, b, c)


def test_a_failed_stage_stops_the_run_and_a_retry_runs_only_what_is_not_up_to_date(chain):
    pipeline, switches, (seed, a, b, c) = chain
    switches["break_b"] = True
    with pytest.raises(StageFailed, match="stage B failed: RuntimeError: B blew up") as failure:
        pipeline.run()
    assert [o.status for o in failure.value.outcomes] == ["ok", "failed"]
    state = State.load(pipeline.state_path)
    assert (
        state.stages["A"].status == "ok" and state.stages["B"].status == "failed" and "C" not in state.stages
    )
    assert "B blew up" in state.stages["B"].error and a.exists() and not b.exists() and not c.exists()
    assert [s for _, _, s in pipeline.status()] == ["ok", "failed", "never run"]
    before = a.stat().st_mtime_ns

    switches["break_b"] = False
    switches["ran"].clear()
    outcomes = pipeline.run()
    assert [(o.stage, o.status) for o in outcomes] == [("A", "skipped"), ("B", "ok"), ("C", "ok")]
    assert switches["ran"] == ["B", "C"] and a.stat().st_mtime_ns == before
    assert c.read_text() == "C(B(A(1)))"

    switches["ran"].clear()
    assert [o.status for o in pipeline.run()] == ["skipped"] * 3 and switches["ran"] == []
    assert [s for _, _, s in pipeline.status()] == ["ok", "ok", "ok"]


def test_a_changed_input_reruns_the_stages_after_it(chain):
    pipeline, switches, (seed, a, b, c) = chain
    pipeline.run()
    switches["ran"].clear()
    seed.write_text("2")
    assert [s for _, _, s in pipeline.status()] == ["stale", "stale (an earlier stage is not ok)"] * 1 + [
        "stale (an earlier stage is not ok)"
    ]
    assert [o.status for o in pipeline.run()] == ["ok"] * 3 and c.read_text() == "C(B(A(2)))"

    switches["ran"].clear()
    b.write_text("tampered")  # an output changed under the stage: it is no longer up to date
    outcomes = pipeline.run(only="B")
    assert [(o.stage, o.status) for o in outcomes] == [("B", "ok")] and b.read_text() == "B(A(2))"
    assert [o.status for o in pipeline.run(start="C")] == ["skipped"]  # B wrote the same bytes back
    assert [o.status for o in pipeline.run(force=True)] == ["ok"] * 3
    with pytest.raises(ValueError, match="unknown stage"):
        pipeline.run(only="D")
    with pytest.raises(ValueError, match="comes before"):
        pipeline.run(start="C", stop="A")


def test_a_missing_input_names_the_stage_that_makes_it(chain):
    pipeline, _, _ = chain
    with pytest.raises(StageFailed, match="input a is missing .*; run stage A"):
        pipeline.run(start="B")


def test_the_real_graph_runs_on_a_fixture_corpus_and_skips_what_is_up_to_date(fake_embedder, tmp_path):
    """fetch → ingest → chunk → embed over one Markdown file, with the fake model; then again."""
    page = tmp_path / "guide.md"
    page.write_text(GUIDE)
    sources = tmp_path / "sources.yaml"
    sources.write_text(
        yaml.safe_dump(
            {"sources": [{"id": "guide", "type": "markdown", "path": str(page), "project": "Kubernetes"}]}
        )
    )
    paths = Paths(
        sources=sources,
        raw_dir=tmp_path / "raw",
        documents=tmp_path / "processed" / "documents.jsonl",
        chunks_dir=tmp_path / "chunks",
        index_dir=tmp_path / "index",
        state=tmp_path / "state.json",
    )
    config = Config(model="fake", chunking=Chunking())
    logged = []
    pipeline = build_pipeline(paths, config, lambda: fake_embedder(), log=logged.append)
    outcomes = pipeline.run()
    assert [(o.stage, o.status) for o in outcomes] == [
        (s, "ok") for s in ("fetch", "ingest", "chunk", "embed")
    ]
    assert paths.documents.exists() and (tmp_path / "chunks" / "whole-sections.jsonl").exists()
    assert (tmp_path / "chunks" / "whole-sections.manifest.json").exists()
    store = Store(tmp_path / "index" / "fake")
    assert store.current() and Index.load(store.path()).manifest.chunks == 3
    assert any(
        line.startswith("chunk     ok        3 chunks of 1 documents (whole-sections)") for line in logged
    )
    assert [o.status for o in pipeline.run()] == ["skipped"] * 4
    assert not list((tmp_path / "processed").glob(".staging*"))

    page.write_text(GUIDE + "\n## Pausing\n\nRun kubectl rollout pause to pause a rollout.\n")
    outcomes = pipeline.run()
    assert [(o.stage, o.status) for o in outcomes] == [
        ("fetch", "ok"),
        ("ingest", "ok"),
        ("chunk", "ok"),
        ("embed", "ok"),
    ]
    assert len(store.versions()) == 2 and Index.load(store.path()).manifest.chunks == 4


def test_fetch_keeps_what_it_got_and_names_what_failed(tmp_path, monkeypatch):
    sources = [
        Source(id="good", type="html", url="https://example.com/good"),
        Source(id="bad", type="html", url="https://example.com/bad"),
    ]
    down = {"bad"}
    asked = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path.strip("/"))
        if request.url.path.strip("/") in down:
            return httpx.Response(503, text="later")
        return httpx.Response(200, text=f"<html>{request.url.path}</html>")

    real = httpx.Client
    monkeypatch.setattr(
        sources_module.httpx, "Client", lambda **kw: real(transport=httpx.MockTransport(handler), **kw)
    )
    with pytest.raises(FetchFailed, match="1 source\\(s\\) could not be fetched \\(bad: HTTPStatusError"):
        fetch(sources, tmp_path)
    manifest = yaml.safe_load((tmp_path / "manifest.json").read_text())
    assert manifest["good"]["sha256"] and (tmp_path / "good.html").exists()
    assert (
        "error" in manifest["bad"]
        and "sha256" not in manifest["bad"]
        and not (tmp_path / "bad.html").exists()
    )

    down.clear()
    asked.clear()
    fetch(sources, tmp_path)  # the retry downloads only what is missing
    assert asked == ["bad"]
    manifest = yaml.safe_load((tmp_path / "manifest.json").read_text())
    assert manifest["bad"]["sha256"] and "error" not in manifest["bad"]


# --- releases ----------------------------------------------------------------------------------


def with_lineage(run: Run, version: str, index: str = "idx1", prompt: str = "cite-v1+abc") -> Run:
    lineage = PipelineManifest(
        version=version,
        ingest_version="0.1.0+test",
        chunking_version="whole-sections+x",
        embedding_version="test/fake@r1",
        index_version=index,
        prompt_version=prompt,
        retrieval_version=f"{version}+y",
        generator="fake-model",
        backend="fake",
        eval_run_id=run.name,
    )
    return run.model_copy(update={"manifest": run.manifest.model_copy(update={"lineage": lineage})})


def test_a_release_is_promoted_only_past_the_gates_and_rolls_back_to_the_one_before(two_runs, tmp_path):  # noqa: F811
    base, cand = two_runs
    gates = load_gates(ROOT / "eval" / "gates.yaml")
    good = with_lineage(base, "v1")
    bad = with_lineage(cand, "v2", index="idx2")
    path = tmp_path / "releases.yaml"
    releases = Releases.load(path)
    assert releases.live is None

    with pytest.raises(ValueError, match="carries no lineage"):
        promote(base, releases, gates, None)
    first, comparison = promote(good, releases, gates, None, note="the first")
    assert comparison is None and first.name == "r1" and releases.current == "r1"
    assert first.manifest.eval_run_id == "base-fake" and "first release" in first.gates
    releases.save(path)
    assert Releases.load(path) == releases

    with pytest.raises(ReleaseBlocked, match="blocked: .* critical gate") as blocked:
        promote(bad, releases, gates, baseline=good)
    assert any(r.gate.metric == "abstention.adversarial_held" for r in blocked.value.comparison.failed)
    assert releases.current == "r1" and len(releases.releases) == 1  # the record is untouched

    better = with_lineage(
        base.model_copy(update={"manifest": base.manifest.model_copy(update={"name": "base-2"})}), "v1b"
    )
    second, comparison = promote(better, releases, gates, baseline=good)
    assert comparison is not None and not comparison.blocked and releases.current == "r2"
    assert second.against == "r1"
    with pytest.raises(ValueError, match="already the current release"):
        promote(better, releases, gates, baseline=better)

    left, now = rollback(releases)
    assert (left.name, now.name) == ("r2", "r1") and releases.current == "r1"
    assert now.manifest.differs(left.manifest) == ["retrieval_version"]
    with pytest.raises(ValueError, match="first release"):
        rollback(releases)
    releases.current = None
    with pytest.raises(ValueError, match="no release to roll back"):
        rollback(releases)


def test_the_release_decides_the_version_unless_a_flag_does(tmp_path):
    path = tmp_path / "releases.yaml"
    version, pinned = resolve_version(None, DEFAULT_RETRIEVAL, path)
    assert version.name == "default" and pinned is None
    with pytest.raises(ValueError, match="no release yet"):
        resolve_version("release", None, path)

    lineage = PipelineManifest(
        version="v1-dense",
        ingest_version="x",
        chunking_version="x",
        embedding_version="x",
        index_version="idx1",
        prompt_version="x",
        retrieval_version="x",
    )
    Releases(
        current="r1",
        releases=[Release(name="r1", created_at="2026-10-12T00:00:00+00:00", run="r", manifest=lineage)],
    ).save(path)
    version, pinned = resolve_version(None, DEFAULT_RETRIEVAL, path)
    assert version.name == "v1-dense" and version.retrieval.mode == "dense" and pinned == "idx1"
    version, pinned = resolve_version(None, Retrieval(k=3), path)  # a flag was given: the flags win
    assert version.name == "default" and version.retrieval.k == 3 and pinned is None
    version, _ = resolve_version("v2-hybrid-rerank", DEFAULT_RETRIEVAL, path)
    assert (
        version.name == "v2-hybrid-rerank"
        and version.prompt == "cite-v1"
        and version.embedding == "bge-small"
    )
    with pytest.raises(ValueError, match="unknown version"):
        resolve_version("v9", None, path)


# --- the failure boundary inside a benchmark run ---------------------------------------------------


def test_a_failed_model_call_is_one_case_not_the_end_of_the_run(fake_embedder, fake_generator, guide):  # noqa: F811
    from test_evaluation import ADVERSARIAL, BENCH

    from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
    from infra_docs_rag.generate.evaluate import evaluate_grounding

    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    inner = fake_generator({"kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])]})
    failures = {"maxSurge": 5, "kubectl scale replicas": 1}  # maxSurge never works; scale works on the retry

    class Flaky:
        name, model, prices = inner.name, inner.model, inner.prices

        def generate(self, context, question):
            if failures.get(question, 0) > 0:
                failures[question] -= 1
                raise ValueError("`claude -p` exited 1: rate limited")
            return inner.generate(context, question)

    class BrokenJudge:
        name = "broken"

        def judge(self, context, question, answer_text):
            raise ValueError("jev is down")

    ev = evaluate_grounding(
        index,
        embedder,
        [guide],
        BENCH,
        [],
        ADVERSARIAL,
        Flaky(),
        retrieval=Retrieval(k=2, mode="dense"),
        judge=BrokenJudge(),
    )
    by_id = {c.id: c for c in ev.cases}
    assert by_id["scale"].outcome == "answered" and by_id["scale"].held and by_id["scale"].judgement is None
    surge = by_id["surge"]
    assert surge.outcome == "error" and surge.verdict == "error" and "rate limited" in surge.answer.error
    assert surge.answer.reason.startswith("the call failed") and surge.answer.generated is None
    assert len(ev.cases) == 5 and surge not in ev.called
