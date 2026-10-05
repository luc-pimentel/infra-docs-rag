import pytest

from infra_docs_rag.embed.index import Hit
from infra_docs_rag.retrieve.fusion import Fusion, HybridHit, fuse, normalized


@pytest.fixture
def rankings(make_chunk):
    a, b, c, d = (make_chunk(s) for s in "ABCD")
    dense = [Hit(1, 0.80, a), Hit(2, 0.70, b), Hit(3, 0.60, c), Hit(4, 0.20, d)]
    lexical = [Hit(1, 9.0, c), Hit(2, 4.0, b)]  # BM25 had nothing to say about A and D
    return dense, lexical


def test_rrf_rewards_the_chunk_both_sides_like(rankings):
    dense, lexical = rankings
    fused = fuse(dense, lexical, Fusion(strategy="rrf", rrf_k=60), k=4)
    # C is 1st for BM25 and 3rd for cosine (1/61 + 1/63), a hair over B at 2nd on both (2/62); A and D are dense-only
    assert [h.chunk.section for h in fused] == ["C", "B", "A", "D"]
    assert [h.rank for h in fused] == [1, 2, 3, 4]
    b = fused[1]
    assert isinstance(b, HybridHit) and (b.dense, b.dense_rank, b.lexical, b.lexical_rank) == (
        0.70,
        2,
        4.0,
        2,
    )
    assert b.score == pytest.approx(1 / 62 + 1 / 62)
    d = fused[3]
    assert (d.lexical, d.lexical_rank) == (0.0, None) and d.score == pytest.approx(1 / 64)


def test_rrf_only_counts_ranks_within_the_depth(rankings):
    dense, lexical = rankings
    fused = fuse(dense, lexical, Fusion(strategy="rrf", depth=1), k=4)
    assert [h.chunk.section for h in fused] == ["A", "C"]  # one candidate from each side, nothing deeper
    assert [h.score for h in fused] == pytest.approx([1 / 61, 1 / 61])  # a tie goes to the better dense rank


def test_weighted_fusion_follows_alpha(rankings):
    dense, lexical = rankings
    assert normalized(dense, depth=4) == {
        "src:Doc/A": 1.0,
        "src:Doc/B": pytest.approx(5 / 6),
        "src:Doc/C": pytest.approx(4 / 6),
        "src:Doc/D": 0.0,
    }
    assert normalized(lexical, depth=2) == {"src:Doc/C": 1.0, "src:Doc/B": 0.0}
    assert normalized([], depth=2) == {}
    dense_heavy = fuse(dense, lexical, Fusion(strategy="weighted", alpha=1.0), k=2)
    assert [h.chunk.section for h in dense_heavy] == ["A", "B"]
    lexical_heavy = fuse(dense, lexical, Fusion(strategy="weighted", alpha=0.0), k=2)
    assert [h.chunk.section for h in lexical_heavy] == [
        "C",
        "A",
    ]  # C is BM25's 1; the rest tie at 0, dense rank breaks it
    even = fuse(dense, lexical, Fusion(strategy="weighted", alpha=0.5), k=4)
    assert even[0].chunk.section == "C" and even[0].score == pytest.approx(0.5 * 4 / 6 + 0.5 * 1.0)


def test_a_single_score_normalizes_to_one(make_chunk):
    only = [Hit(1, 3.0, make_chunk("A"))]
    assert normalized(only, depth=5) == {"src:Doc/A": 1.0}
    fused = fuse(only, only, Fusion(strategy="weighted"), k=1)
    assert fused[0].score == 1.0


def test_labels():
    assert Fusion().label == "rrf k=60 · depth 50"
    assert Fusion(strategy="weighted", alpha=0.3).label == "weighted α=0.3 · depth 50"
