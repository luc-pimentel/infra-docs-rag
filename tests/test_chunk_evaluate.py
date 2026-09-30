from types import SimpleNamespace

import pytest

from infra_docs_rag.chunk.chunks import Chunk, Chunking
from infra_docs_rag.chunk.evaluate import Answer, answers, best, run, union
from infra_docs_rag.embed.evaluate import Query, QuerySet, Target
from infra_docs_rag.embed.index import Hit
from infra_docs_rag.ingest.parsers.base import assemble, markdown_parts

GUIDE = """# Deployments

A Deployment manages a set of Pods.

## Rolling Update

Set maxSurge to allow extra Pods during an update.

## Scaling

Run kubectl scale to change the number of replicas.
"""


def chunk(start: int, end: int, text: str = "", source: str = "guide") -> Chunk:
    return Chunk(
        chunk_id=f"{source}:{start}",
        source_id=source,
        document_id=f"doc_{source}",
        source_uri=f"https://example.com/{source}",
        section_path=["Doc", "Section"],
        start=start,
        end=end,
        text=text or "x" * (end - start),
    )


@pytest.fixture
def guide(make_doc):
    text, sections, _ = assemble(markdown_parts(GUIDE))
    doc = make_doc("guide", text)
    doc.sections = sections
    return doc


def query(qid: str, text: str, sections: list[str], contains: str | None = None) -> Query:
    return Query(
        id=qid,
        kind="identifier" if contains else "paraphrase",
        text=text,
        expect=Target(source="guide", sections=sections, contains=contains),
        hypothesis="-",
    )


def test_a_chunk_answers_when_it_overlaps_half_of_the_smaller_of_it_and_the_section():
    answer = Answer("guide", [(100, 200)])
    assert answer.answered_by(chunk(100, 200))  # the whole section
    assert answer.answered_by(chunk(120, 160))  # a piece of it
    assert answer.answered_by(chunk(50, 260))  # a window that holds all of it
    assert not answer.answered_by(chunk(180, 400))  # a window that only grazes its end
    assert not answer.answered_by(chunk(100, 200, source="other"))


def test_a_query_about_an_identifier_needs_a_chunk_that_contains_it():
    answer = Answer("guide", [(0, 100)], contains="maxSurge")
    assert not answer.answered_by(chunk(0, 50, "Set the rollout strategy for an update."))
    assert answer.answered_by(chunk(50, 100, "Set maxSurge to allow extra Pods."))


def test_answer_share_counts_overlapping_hits_once():
    answer = Answer("guide", [(100, 200)])
    hits = [
        Hit(1, 0.9, chunk(100, 150)),
        Hit(2, 0.8, chunk(140, 180)),
        Hit(3, 0.7, chunk(100, 200, source="other")),
    ]
    assert answer.coverage(hits) == pytest.approx(0.8)
    assert union([(5, 9), (0, 3), (2, 6)]) == [(0, 9)]


def test_answers_resolve_section_titles_to_their_spans(guide):
    found = answers(QuerySet(queries=[query("scale", "replicas", ["Scaling"])], decoys=[]), [guide])
    scaling = next(s for s in guide.sections if s.title == "Scaling")
    assert found[0].spans == [(scaling.start, scaling.end)]
    with pytest.raises(ValueError, match="matches 0 sections"):
        answers(QuerySet(queries=[query("none", "x", ["Missing"])], decoys=[]), [guide])
    with pytest.raises(ValueError, match="not a usable document"):
        bad = query("elsewhere", "x", ["Scaling"])
        bad.expect.source = "other"
        answers(QuerySet(queries=[bad], decoys=[]), [guide])


def test_a_run_ranks_the_first_right_chunk_and_scores_a_miss_as_zero(fake_embedder, guide):
    query_set = QuerySet(
        queries=[
            query("scale", "kubectl scale replicas", ["Scaling"]),
            query("surge", "maxSurge", ["Scaling"], contains="maxSurge"),  # Scaling never says maxSurge
        ],
        decoys=[],
    )
    found = answers(query_set, [guide])
    chunking = Chunking(strategy="section-aware", size=64)
    result = run(fake_embedder(max_tokens=512), [guide], query_set, found, chunking, projects={})
    assert result.ranks == [1, None]
    assert result.mrr == pytest.approx(0.5)
    assert result.hits(1) == 1 and result.hits(5) == 1
    assert result.shape.spread == 1 and result.shape.cut_off == 0
    assert result.coverage[0] == pytest.approx(1.0)
    assert result.read[0] == sum(result.lengths[h.chunk.chunk_id] for h in result.top[0])


def test_best_takes_the_highest_mrr_then_more_of_the_answer_then_fewer_tokens():
    def fake(label, mrr, coverage, read):
        return SimpleNamespace(label=label, mrr=mrr, mean_coverage=coverage, median_read=read)

    assert best([fake("a", 0.8, 0.5, 900), fake("b", 0.9, 0.2, 2000)]).label == "b"
    assert best([fake("a", 0.8, 0.5, 900), fake("b", 0.8, 0.6, 2000)]).label == "b"
    assert best([fake("a", 0.8, 0.5, 900), fake("b", 0.8, 0.5, 2000)]).label == "a"
