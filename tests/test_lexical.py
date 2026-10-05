import numpy as np
import pytest

from infra_docs_rag.retrieve.lexical import Lexical, tokenize


def test_identifiers_survive_whole_and_in_parts():
    tokens = tokenize("Set progressDeadlineSeconds, keep_firing_for and argocd.argoproj.io/skip-reconcile.")
    assert "progressdeadlineseconds" in tokens and {"progress", "deadline", "seconds"} <= set(tokens)
    assert "keep_firing_for" in tokens and {"keep", "firing"} <= set(tokens)
    assert "argocd.argoproj.io/skip-reconcile" in tokens  # the trailing period is punctuation, not part of it
    assert tokenize("--to-revision $labels .spec.os.name") == [
        "to-revision",
        "revision",  # `to` is a stopword, so only the whole flag and `revision` go in
        "$labels",  # one part only, so nothing to add
        "spec.os.name",
        "spec",
        "os",
        "name",
    ]


def test_stopwords_and_case_are_dropped():
    assert tokenize("How do I set maxSurge?") == ["set", "maxsurge", "max", "surge"]
    assert tokenize("the of a") == []


@pytest.fixture
def lexical(make_chunk):
    return Lexical.build(
        [
            make_chunk(
                "Rolling Update",
                "The maxSurge field sets how many Pods can be created above the desired count.",
            ),
            make_chunk(
                "Unavailable",
                "The maxUnavailable field sets how many Pods can be unavailable during the update.",
            ),
            make_chunk("Scaling", "Scale a Deployment by changing the number of replicas."),
        ]
    )


def test_the_exact_key_wins_over_its_near_twin(lexical):
    hits = lexical.rank("maxSurge", 3)
    # The twin shares `max` through its parts, so it is ranked, but only the whole token `maxsurge` is rare
    assert [h.chunk.section for h in hits] == ["Rolling Update", "Unavailable"]
    assert hits[0].rank == 1 and hits[0].score > 2 * hits[1].score
    split = lexical.rank("max surge pods", 3)
    assert [h.chunk.section for h in split][:2] == [
        "Rolling Update",
        "Unavailable",
    ]  # both have `max` and `pods`


def test_a_query_with_no_shared_term_ranks_nothing(lexical):
    assert lexical.rank("sourdough starter", 3) == []
    assert lexical.scores("sourdough").tolist() == [0.0, 0.0, 0.0]


def test_rank_within_a_mask_and_explain(lexical):
    only_last_two = np.array([False, True, True])
    assert [h.chunk.section for h in lexical.rank("maxSurge pods", 3, only_last_two)] == ["Unavailable"]
    with pytest.raises(ValueError, match="marks 2 chunks"):
        lexical.rank("pods", 3, np.array([True, False]))
    terms = {term: n for term, n, _ in lexical.explain("maxSurge Pods elsewhere")}
    assert terms == {"maxsurge": 1, "max": 2, "surge": 1, "pods": 2}
