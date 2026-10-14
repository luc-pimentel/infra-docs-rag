from pathlib import Path

from test_evaluation import ADVERSARIAL, BENCH
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
from infra_docs_rag.embed.index import build
from infra_docs_rag.generate.answer import answer
from infra_docs_rag.generate.cache import CachedGenerator, GenerationCache, key_for
from infra_docs_rag.latency.configs import CONFIGS
from infra_docs_rag.latency.profile import (
    BatchTiming,
    Workload,
    p50,
    p95,
    profile,
    questions_of,
    time_batches,
)
from infra_docs_rag.latency.report import load_workload, save_workload, write_report
from infra_docs_rag.retrieve.retriever import Rerank, Retrieval, Retriever


def test_the_cache_skips_the_call_for_a_repeat(fake_embedder, fake_reranker, fake_generator, guide, tmp_path):  # noqa: F811
    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    r = Retriever(index, embedder, fake_reranker("bge-base"))
    inner = fake_generator({"kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])]})
    cache = GenerationCache(tmp_path / "generations.jsonl")
    gen = CachedGenerator(inner, cache, "cite-v1+abc")
    retrieval = Retrieval(k=2, mode="dense", min_score=0.1)

    first = answer("kubectl scale replicas", r, gen, retrieval)
    assert first.outcome == "answered" and not first.generated.cached and gen.misses == 1 and len(cache) == 1
    second = answer("kubectl scale replicas", r, gen, retrieval)
    assert second.generated.cached and second.text == first.text and second.generated.usage.input_tokens == 0
    assert second.citations[0].chunk.chunk_id == first.citations[0].chunk.chunk_id and gen.hits == 1
    assert len(inner.calls) == 1  # the model was called once

    # A different prompt version, question or context is a different key.
    ctx = first.context
    key = key_for("fake-model", "cite-v1+abc", "kubectl scale replicas", ctx)
    assert key_for("fake-model", "cite-v2+xyz", "kubectl scale replicas", ctx) != key
    assert key_for("fake-model", "cite-v1+abc", "kubectl scale replicas?", ctx) != key
    other = Retriever(index, embedder).retrieve("kubectl scale replicas", Retrieval(k=3, mode="dense"))
    assert (
        len(other.sources) == 3
        and key_for("fake-model", "cite-v1+abc", "kubectl scale replicas", other) != key
    )

    # The file holds it: a new cache object sees the entry.
    assert GenerationCache(tmp_path / "generations.jsonl").get(key) is not None
    # A refusal is not stored.
    refusing = CachedGenerator(fake_generator({}, stop_reason="refusal", refusal="x"), cache, "p")
    refusing.generate(ctx, "anything")
    assert refusing.misses == 1 and len(cache) == 1


def test_a_profile_times_stages_and_reads_the_labels(fake_embedder, fake_reranker, guide, tmp_path):  # noqa: F811
    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    questions = questions_of([guide], BENCH, [], ADVERSARIAL)
    assert len(questions) == 5 and sum(q.labelled is not None for q in questions) == 2
    retrieval = Retrieval(k=2, mode="hybrid", min_score=0.25, rerank=Rerank(model="bge-base", candidates=3))
    p = profile("test", retrieval, index, embedder, questions, fake_reranker("bge-base"), "a note")
    assert len(p.observations) == 5 and len(p.labelled) == 2 and len(p.out_of_scope) == 1
    assert {"embed", "dense", "lexical", "fuse", "rerank", "pack"} <= set(p.observations[0].timings)
    assert 1 <= p.right_retrieved <= 2 and 0.25 <= p.mrr <= 1.0
    assert p.oos_abstentions == 1 and p.false_abstentions <= 1 and 1 <= p.calls <= 4
    assert p.passage_tokens_p50 > 0 and p.generation_cost(400) > 0
    assert p.retrieve_p50 > 0 and p.stage_p95("rerank") >= p.stage_p50("rerank")
    assert p50([1, 2, 3]) == 2 and round(p95([1, 2, 3, 4]), 2) == 3.85 and p50([]) == 0 and p95([]) == 0

    batches = time_batches(embedder, [c.embedded for c in chunks], sizes=(2, 8))
    assert [b.batch_size for b in batches] == [2, 8] and all(b.chunks == len(chunks) for b in batches)
    assert embedder.batch_size == 16

    w = Workload([p], batches, "test machine")
    save_workload(w, tmp_path / "latency.json")
    back = load_workload(tmp_path / "latency.json")
    q = back.profiles[0]
    assert (
        q.name == "test"
        and q.right_retrieved == p.right_retrieved
        and q.mrr == p.mrr
        and q.retrieval == retrieval
    )
    assert q.passage_tokens_p50 == p.passage_tokens_p50 and back.batches[0] == BatchTiming(
        2, batches[0].seconds, len(chunks)
    )
    write_report(back, {}, tmp_path / "latency-cost.md")
    text = (tmp_path / "latency-cost.md").read_text()
    assert "## 1. The request path" in text and "| test |" in text or "| **test** |" in text
    assert "## 3. The matrix" in text


def test_the_configurations_are_the_released_pipeline_and_cuts_to_it():
    base, _ = CONFIGS["v2 (released)"]
    assert base.rerank.candidates == 20 and base.k == 5 and base.mode == "hybrid"
    assert CONFIGS["rerank 10"][0].rerank.candidates == 10 and CONFIGS["top 3"][0].k == 3
    assert (
        CONFIGS["no rerank"][0].rerank is None and CONFIGS["minilm-l6 over 20"][0].rerank.model == "minilm-l6"
    )
    assert CONFIGS["relevance floor 0.10"][0].rerank.min_relevance == 0.10 and CONFIGS["minilm-l6 + top 3"][0].k == 3
    assert all(note for _, note in CONFIGS.values())


def test_the_shipped_profiles_are_consistent_when_present():
    saved = Path(__file__).parents[1] / "eval" / "latency.json"
    if not saved.exists():
        return
    w = load_workload(saved)
    names = [p.name for p in w.profiles]
    assert "v2 (released)" in names and len(w.batches) >= 2
    base = next(p for p in w.profiles if p.name == "v2 (released)")
    assert len(base.observations) == 69 and len(base.labelled) == 53 + 0 or len(base.labelled) in (53, 59)
    assert all(len(p.observations) == len(base.observations) for p in w.profiles)
