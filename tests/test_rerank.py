import pytest

from infra_docs_rag.embed.index import Hit
from infra_docs_rag.retrieve.rerank import RERANKERS, RerankedHit, rerank
from infra_docs_rag.retrieve.retriever import Rerank, Retrieval, gate, relevance


@pytest.fixture
def candidates(make_chunk):
    a = make_chunk("A", "Scale a Deployment by changing its number of replicas.")
    b = make_chunk("B", "Roll back a Deployment to an earlier revision with kubectl rollout undo.")
    c = make_chunk("C", "Rollout and release layers of the cloud native stack.")
    return [Hit(1, 0.9, a), Hit(2, 0.8, b), Hit(3, 0.7, c)]


def test_reranking_reorders_by_relevance_and_keeps_the_first_stage(fake_reranker, candidates):
    result = rerank("kubectl rollout undo", candidates, fake_reranker(), k=2)
    assert [h.chunk.section for h in result.hits] == ["B", "C"]  # B has all three words, C one, A none
    assert [h.rank for h in result.hits] == [1, 2] and result.scored == 3 and result.ms >= 0
    best = result.hits[0]
    assert isinstance(best, RerankedHit) and best.score == pytest.approx(1.0)
    assert best.first_rank == 2 and best.first.score == 0.8 and best.first is candidates[1]
    assert gate(best) == 0.8 and relevance(best) == pytest.approx(1.0)
    assert relevance(candidates[0]) is None


def test_ties_keep_the_first_stage_order(fake_reranker, candidates):
    same = fake_reranker(fixed={c.chunk.embedded: 0.5 for c in candidates})
    result = rerank("anything", candidates, same, k=3)
    assert [h.chunk.section for h in result.hits] == ["A", "B", "C"]
    assert rerank("anything", [], same, k=3).hits == []


def test_a_rerank_names_a_known_model_and_enough_candidates():
    assert Rerank().model in RERANKERS and Rerank().candidates == 20
    assert Rerank(model="minilm-l6", candidates=10).label == "rerank minilm-l6 over 10"
    assert Rerank(min_relevance=0.3).label == "rerank bge-base over 20 · relevance 0.30"
    with pytest.raises(ValueError, match="unknown reranker"):
        Rerank(model="cohere")
    with pytest.raises(ValueError, match="cannot fill the top 5"):
        Retrieval(k=5, rerank=Rerank(candidates=3))
    assert Retrieval(k=3, rerank=Rerank(candidates=3)).label == "top 3 · rerank bge-base over 3"
