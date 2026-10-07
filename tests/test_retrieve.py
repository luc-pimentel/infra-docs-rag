import numpy as np
import pytest

from infra_docs_rag.chunk.chunks import Chunk
from infra_docs_rag.embed.index import build
from infra_docs_rag.retrieve.rerank import RerankedHit
from infra_docs_rag.retrieve.retriever import Filter, Rerank, Retrieval, Retriever, mask


def chunk(section, text, source="src", project="Kubernetes", pages=None, parent="Doc") -> Chunk:
    return Chunk(
        chunk_id=f"{source}:{section}",
        source_id=source,
        document_id=f"doc_{source}",
        source_uri=f"https://example.com/{source}",
        project=project,
        section_path=[parent, section],
        start=0,
        end=len(text),
        pages=pages or [],
        text=text,
    )


CHUNKS = [
    chunk("Rollback", "Roll back a Deployment to an earlier revision with kubectl rollout undo."),
    chunk("Scaling", "Scale a Deployment by changing its number of replicas."),
    chunk(
        "Layers",
        "Rollout and release layers of the cloud native stack.",
        source="paper",
        project="CNCF",
        pages=[9, 10],
    ),
]
QUERY = "kubectl rollout undo to an earlier revision"


@pytest.fixture
def retriever(fake_embedder):
    embedder = fake_embedder()
    return Retriever(build(embedder, CHUNKS, ingest_version="test"), embedder)


def test_a_filter_requires_every_field_it_sets():
    assert Filter().empty and all(Filter().matches(c) for c in CHUNKS)
    assert [c.section for c in CHUNKS if Filter(project="cncf").matches(c)] == [
        "Layers"
    ]  # case does not matter
    assert [c.section for c in CHUNKS if Filter(source="src", section="Doc › scaling").matches(c)] == [
        "Scaling"
    ]
    assert [c.section for c in CHUNKS if Filter(page=9).matches(c)] == ["Layers"]
    assert not any(Filter(project="CNCF", page=3).matches(c) for c in CHUNKS)
    assert Filter(project="Argo CD", page=9).label == "project = Argo CD · page = 9"
    assert mask(CHUNKS, Filter(source="paper")).tolist() == [False, False, True]


def test_rank_within_a_mask(fake_embedder):
    embedder = fake_embedder()
    index = build(embedder, CHUNKS, ingest_version="test")
    vector = embedder.embed_queries([QUERY])[0]
    assert [h.chunk.section for h in index.rank(vector, 3)][0] == "Rollback"
    only_scaling = index.rank(vector, 3, np.array([False, True, False]))
    assert [h.chunk.section for h in only_scaling] == ["Scaling"] and only_scaling[0].rank == 1
    assert index.rank(vector, 3, np.array([False, False, False])) == []
    with pytest.raises(ValueError, match="marks 2 chunks"):
        index.rank(vector, 3, np.array([True, False]))


def test_filtering_before_ranking_finds_what_filtering_after_it_loses(retriever):
    before = retriever.retrieve(QUERY, Retrieval(k=1, filter=Filter(source="paper")))
    assert [s.chunk.section for s in before.sources] == ["Layers"] and before.candidates == 1
    after = retriever.retrieve(QUERY, Retrieval(k=1, filter=Filter(source="paper"), prefilter=False))
    assert after.empty and after.candidates == 1 and after.top_score is None  # the top hit was Rollback


def test_hits_under_the_threshold_are_dropped_and_counted(retriever):
    everything = retriever.retrieve(QUERY, Retrieval(k=3))
    scores = [s.hit.score for s in everything.sources]
    assert len(scores) == 3 and scores == sorted(scores, reverse=True) and everything.below == 0
    strict = retriever.retrieve(QUERY, Retrieval(k=3, min_score=scores[0] - 1e-6))
    assert [s.chunk.section for s in strict.sources] == ["Rollback"] and strict.below == 2
    nothing = retriever.retrieve(QUERY, Retrieval(k=3, min_score=0.999))
    assert nothing.empty and nothing.below == 3 and nothing.top_score == pytest.approx(scores[0])


def test_the_context_numbers_passages_and_cites_each_one(retriever):
    context = retriever.retrieve(QUERY, Retrieval(k=2))
    assert [s.n for s in context.sources] == [1, 2]
    first = context.sources[0]
    assert first.heading == "[1] Kubernetes › Doc › Rollback"
    assert context.text.startswith("[1] Kubernetes › Doc › Rollback\nRoll back a Deployment")
    assert "\n\n[2] " in context.text
    assert first.tokens == 12 and context.tokens == sum(s.tokens for s in context.sources)
    paged = retriever.retrieve(QUERY, Retrieval(k=1, filter=Filter(page=10)))
    assert paged.sources[0].heading == "[1] CNCF › Doc › Layers, pages 9-10"


def test_the_budget_keeps_the_first_passage_and_stops_before_overflowing(retriever):
    full = retriever.retrieve(QUERY, Retrieval(k=3))
    assert len(full.sources) == 3 and full.over_budget == 0
    one = retriever.retrieve(QUERY, Retrieval(k=3, budget=full.sources[0].tokens))
    assert len(one.sources) == 1 and one.over_budget == 2
    assert len(retriever.retrieve(QUERY, Retrieval(k=3, budget=1)).sources) == 1


def test_a_label_says_what_the_retrieval_does():
    assert Retrieval().label == "top 5"
    assert Retrieval(
        k=3, min_score=0.6, filter=Filter(project="Argo CD"), prefilter=False, budget=800
    ).label == ("top 3 · min 0.60 · project = Argo CD, after ranking · 800 tokens")
    with pytest.raises(ValueError):
        Retrieval(k=0)


def test_the_retriever_refuses_another_models_index(fake_embedder):
    index = build(fake_embedder("a"), CHUNKS, ingest_version="test")
    with pytest.raises(Exception, match="not comparable"):
        Retriever(index, fake_embedder("b"))


def test_modes_share_the_filter_and_the_floor(retriever):
    plain = retriever.retrieve(QUERY, Retrieval(k=2, mode="lexical"))
    assert [s.chunk.section for s in plain.sources] == ["Rollback", "Layers"]  # Scaling shares no term
    assert plain.retrieval.label == "top 2 · lexical"
    with pytest.raises(ValueError, match="lexical mode takes no min_score"):
        Retrieval(mode="lexical", min_score=0.5)

    hybrid = retriever.retrieve(QUERY, Retrieval(k=3, mode="hybrid", min_score=0.0))
    assert hybrid.sources[0].chunk.section == "Rollback"
    first = hybrid.sources[0].hit
    assert (
        first.dense == pytest.approx(hybrid.top_score) and first.lexical_rank == 1
    )  # the floor sees the cosine side
    assert hybrid.retrieval.label == "top 3 · hybrid (rrf k=60 · depth 50) · min 0.00"

    filtered = retriever.retrieve(QUERY, Retrieval(k=1, mode="hybrid", filter=Filter(source="paper")))
    assert [s.chunk.section for s in filtered.sources] == ["Layers"] and filtered.candidates == 1


def test_a_rerank_reorders_the_first_stage_and_the_floors_read_each_stage(fake_embedder, fake_reranker):
    embedder = fake_embedder()
    index = build(embedder, CHUNKS, ingest_version="test")
    with pytest.raises(ValueError, match="no reranker is loaded"):
        Retriever(index, embedder).retrieve(
            QUERY, Retrieval(k=2, rerank=Rerank(model="minilm-l6", candidates=3))
        )
    with pytest.raises(ValueError, match="'bge-base' is loaded"):
        Retriever(index, embedder, fake_reranker("bge-base")).retrieve(
            QUERY, Retrieval(k=2, rerank=Rerank(model="minilm-l6", candidates=3))
        )

    retriever = Retriever(index, embedder, fake_reranker("minilm-l6"))
    plain = retriever.retrieve("layers of the stack", Retrieval(k=3, mode="dense"))
    reranked = retriever.retrieve(
        "layers of the stack", Retrieval(k=3, mode="dense", rerank=Rerank(model="minilm-l6", candidates=3))
    )
    assert reranked.sources[0].chunk.section == "Layers" and reranked.retrieval.label.endswith("over 3")
    first = reranked.sources[0].hit
    assert isinstance(first, RerankedHit) and first.score == pytest.approx(1.0)  # every query word is there
    assert first.first_rank == [s.chunk.section for s in plain.sources].index("Layers") + 1
    assert reranked.top_score == pytest.approx(first.first.score) and reranked.top_relevance == pytest.approx(
        1.0
    )
    assert retriever.last_rerank is not None and retriever.last_rerank.scored == 3

    strict = retriever.retrieve(
        "layers of the stack",
        Retrieval(k=3, mode="dense", rerank=Rerank(model="minilm-l6", candidates=3, min_relevance=0.9)),
    )
    assert [s.chunk.section for s in strict.sources] == ["Layers"] and strict.below == 2
    nothing = retriever.retrieve(
        "sourdough",
        Retrieval(k=3, mode="dense", rerank=Rerank(model="minilm-l6", candidates=3, min_relevance=0.5)),
    )
    assert nothing.empty and nothing.top_relevance == 0.0
