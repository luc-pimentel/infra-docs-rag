from pathlib import Path

import pytest

from infra_docs_rag.embed.evaluate import (
    KINDS,
    Comparison,
    Query,
    QuerySet,
    Target,
    compare,
    load_queries,
    mixed_maps,
    probe,
    probe_set,
    resolve,
)
from infra_docs_rag.embed.index import build

QUERIES = Path(__file__).parents[1] / "eval" / "queries.yaml"


@pytest.fixture
def chunks(make_chunk):
    return [
        make_chunk("Rollback", "Roll back a Deployment to an earlier revision with kubectl rollout undo."),
        make_chunk("Scaling", "Scale a Deployment by changing its number of replicas."),
        make_chunk("Services", "A Service exposes Pods behind one stable virtual IP address and DNS name."),
        make_chunk("Conclusion", "Security is a shared job.", parent="Summary"),
        make_chunk("Conclusion", "People make the community.", parent="Closing"),
    ]


@pytest.fixture
def query_set():
    return QuerySet(
        queries=[
            Query(
                id="undo",
                kind="paraphrase",
                text="undo a rollout to an earlier revision",
                expect=Target(source="src", sections=["Rollback"]),
                hypothesis="shares words with the answer",
            ),
            Query(
                id="replicas",
                kind="identifier",
                text="replicas",
                expect=Target(source="src", sections=["Scaling"]),
                hypothesis="exact word",
            ),
        ],
        decoys=[Target(source="src", sections=["Services"])],
    )


def test_the_queries_file_is_well_formed():
    query_set = load_queries(QUERIES)
    assert len(query_set.queries) == 11
    assert {q.kind for q in query_set.queries} == set(KINDS)
    assert all(q.expect.sections and q.hypothesis for q in query_set.queries)


def test_a_target_must_name_exactly_one_section(chunks):
    found = resolve(Target(source="src", sections=["Summary › Conclusion"]), chunks)
    assert [c.text for c in found] == ["Security is a shared job."]
    with pytest.raises(ValueError, match="matches 2 chunks"):
        resolve(Target(source="src", sections=["Conclusion"]), chunks)
    with pytest.raises(ValueError, match="matches 0 chunks"):
        resolve(Target(source="other", sections=["Rollback"]), chunks)


def test_the_probe_ranks_each_querys_right_section(fake_embedder, chunks, query_set):
    probe_chunks, right = probe_set(query_set, chunks)
    assert [c.section for c in probe_chunks] == ["Rollback", "Scaling", "Services"]
    result = probe(fake_embedder(), query_set, probe_chunks, right)
    assert result.scores.shape == (2, 3)
    assert [result.rank(q) for q in range(2)] == [1, 1]
    assert all(result.margin(q) > 0 for q in range(2))


def test_a_decoy_cannot_be_a_right_section(chunks, query_set):
    query_set.decoys.append(Target(source="src", sections=["Scaling"]))
    with pytest.raises(ValueError, match="also a right section"):
        probe_set(query_set, chunks)


def test_comparison_counts_hits_and_mean_reciprocal_rank(fake_embedder, chunks, query_set):
    embedder = fake_embedder()
    result = compare(embedder, build(embedder, chunks, ingest_version="test"), query_set)
    assert result.ranks == [1, 1] and result.mrr == 1.0
    assert len(result.latency_ms) == 2

    manifest = result.manifest
    spread = Comparison(manifest, parameters=0, index_bytes=0, ranks=[1, 3, 10], latency_ms=[1, 1, 1])
    assert (spread.hits(1), spread.hits(5), spread.hits(5, only=[1, 2])) == (1, 2, 1)
    assert spread.mrr == pytest.approx((1 + 1 / 3 + 1 / 10) / 3)


def test_mixed_maps_are_refused_and_then_measured(fake_embedder, chunks, query_set):
    index = build(fake_embedder("a"), chunks, ingest_version="test")
    mixed = mixed_maps(index, fake_embedder("b"), query_set)
    assert "not comparable" in mixed.refusal
    assert len(mixed.ranks) == 2 and [len(top) for top in mixed.top] == [3, 3]
    with pytest.raises(ValueError, match="nothing mixed"):
        mixed_maps(index, fake_embedder("a"), query_set)
