import numpy as np
import pytest

from infra_docs_rag.embed.index import Index, ModelMismatch, build

CHUNKS = {
    "Rollback": "Roll back a Deployment to an earlier revision with kubectl rollout undo.",
    "Scaling": "Scale a Deployment by changing its number of replicas.",
    "Services": "A Service exposes Pods behind one stable virtual IP address and DNS name.",
}


@pytest.fixture
def chunks(make_chunk):
    return [make_chunk(section, text) for section, text in CHUNKS.items()]


def test_the_lessons_cosine_check(fake_embedder, make_chunk):
    """A = [1, 0], B = [0.8, 0.6], C = [0.2, 0.98]: A–B scores 0.80 and A–C 0.20, so B ranks first."""
    embedder = fake_embedder(dimensions=2, fixed={"A": [1, 0], "B": [0.8, 0.6], "C": [0.2, 0.98]})
    index = build(embedder, [make_chunk("C"), make_chunk("B")], ingest_version="test")
    hits = index.search(embedder, "A", k=2)
    assert [h.chunk.section for h in hits] == ["B", "C"]
    assert [round(h.score, 2) for h in hits] == [0.80, 0.20]


def test_vectors_are_stored_at_length_one(fake_embedder, chunks):
    index = build(fake_embedder(), chunks, ingest_version="test")
    assert np.allclose(np.linalg.norm(index.vectors, axis=1), 1)
    assert index.manifest.chunks == 3 and index.manifest.dimensions == 64
    assert index.manifest.truncated == 3  # every chunk is longer than the fake model's 8 tokens


def test_search_puts_the_chunk_that_shares_the_query_words_first(fake_embedder, chunks):
    embedder = fake_embedder()
    index = build(embedder, chunks, ingest_version="test")
    hits = index.search(embedder, "kubectl rollout undo to an earlier revision", k=2)
    assert len(hits) == 2 and hits[0].chunk.section == "Rollback"
    assert hits[0].score > hits[1].score


def test_search_refuses_another_model_even_at_the_same_size(fake_embedder, chunks):
    index = build(fake_embedder("a"), chunks, ingest_version="test")
    with pytest.raises(ModelMismatch, match="test/a.*test/b"):
        index.search(fake_embedder("b"), "rollback")


def test_search_refuses_a_new_revision_of_the_same_model(fake_embedder, chunks):
    index = build(fake_embedder("a", revision="r1"), chunks, ingest_version="test")
    with pytest.raises(ModelMismatch):
        index.search(fake_embedder("a", revision="r2"), "rollback")


def test_save_and_load_round_trip(fake_embedder, chunks, tmp_path):
    embedder = fake_embedder()
    index = build(embedder, chunks, ingest_version="test")
    index.save(tmp_path / "fake")
    loaded = Index.load(tmp_path / "fake")
    assert loaded.manifest == index.manifest and loaded.chunks == index.chunks
    assert np.array_equal(loaded.vectors, index.vectors)
    assert loaded.search(embedder, "replicas")[0].chunk.section == "Scaling"


def test_load_rejects_vectors_that_do_not_match_the_manifest(fake_embedder, chunks, tmp_path):
    build(fake_embedder(), chunks, ingest_version="test").save(tmp_path / "fake")
    np.save(tmp_path / "fake" / "vectors.npy", np.zeros((3, 32), dtype=np.float32))
    with pytest.raises(ValueError, match="does not match its manifest"):
        Index.load(tmp_path / "fake")
