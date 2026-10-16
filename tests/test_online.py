from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from test_observe import small  # noqa: F401  (a fixture)
from test_regression import BASE_SCRIPT, make_run
from test_regression import records as dataset_records  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.evaluation.online import (
    POLICY,
    Cluster,
    Policy,
    Sample,
    Window,
    baseline_rates,
    clusters,
    drift,
    harvest,
    harvest_prompt,
    is_live,
    judge_samples,
    load_policy,
    load_windows,
    merge_into_records,
    random_pick,
    rules_for,
    sample,
    save_harvested,
    to_record,
    window_name,
    window_rates,
)
from infra_docs_rag.evaluation.online_report import write_report
from infra_docs_rag.generate.answer import answer
from infra_docs_rag.generate.judge import Judgement
from infra_docs_rag.observe.feedback import Feedback, FeedbackLog
from infra_docs_rag.observe.trace import Trace, TracedHit, trace_answer
from infra_docs_rag.retrieve.retriever import Rerank, Retrieval, Retriever

ROOT = Path(__file__).parents[1]
T0 = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


def live(trace_id: str, question: str, outcome: str = "answered", **overrides) -> Trace:
    base = dict(
        trace_id=trace_id,
        at=(T0 + timedelta(hours=int(trace_id[-1], 16))).isoformat(timespec="seconds"),
        question=question,
        version="v3-budget",
        retrieval="top 5",
        outcome=outcome,
        top_score=0.75,
        claim_share=1.0 if outcome == "answered" else 0.0,
        context=[
            TracedHit(
                rank=1, chunk_id="k8s-deployment-md:10", citation="k8s › Deployments", score=0.9, cosine=0.75
            )
        ]
        if outcome == "answered"
        else [],
        tokens=600 if outcome == "answered" else 0,
        budget=1200,
        answer="Run kubectl rollout undo.[1]" if outcome == "answered" else "",
    )
    base.update(overrides)
    return Trace(**base)


def test_the_rules_read_the_trace_and_the_feedback():
    policy = Policy()
    assert rules_for(live("a1", "how do I undo a release?"), policy, None) == []
    assert rules_for(live("a2", "undo"), policy, Feedback(trace_id="a2", helpful=False)) == ["dissatisfied"]
    assert rules_for(live("a3", "undo", claim_share=0.5), policy, None) == ["uncited"]
    assert rules_for(live("a4", "undo", top_score=0.55), policy, None) == ["low-confidence"]
    assert rules_for(live("a5", "sourdough", outcome="no-passage"), policy, None) == ["no-answer"]
    assert rules_for(live("a6", "undo", tokens=1100), policy, None) == ["long-context"]
    assert rules_for(live("a7", "undo", over_budget=2), policy, None) == ["long-context"]
    assert rules_for(live("a8", " ".join(["word"] * 26)), policy, None) == ["long-query"]
    for q in ("what is maxSurge", "set spec.template.metadata", "scrape_interval vs x", "use --to-revision"):
        assert rules_for(live("a9", q), policy, None) == ["identifier"], q
    assert rules_for(live("aa", "undo", outcome="error"), policy, None) == ["error"]
    both = rules_for(
        live("ab", "what is maxSurge", claim_share=0.3), policy, Feedback(trace_id="ab", helpful=False)
    )
    assert both == ["dissatisfied", "uncited", "identifier"]
    assert random_pick("x", 1.0) and not random_pick("x", 0.0)
    assert random_pick("abc", 0.5) == random_pick("abc", 0.5)  # stable
    assert is_live(live("ac", "q")) and not is_live(live("ad", "q", version="v3:run"))
    assert not is_live(live("ae", "q", run="ds")) and not is_live(live("af", "q", version="canary:v3"))
    assert window_name(datetime(2026, 10, 15).date()) == "2026-W42"


def test_a_window_is_sampled_judged_clustered_and_read_for_drift(dataset_records, fake_generator, tmp_path):  # noqa: F811
    policy = Policy(random_share=0.0, max_samples=3)
    traces = [
        live("b1", "how do I undo a release?"),
        live("b2", "sourdough", outcome="no-passage"),
        live("b3", "what is maxSurge", claim_share=0.4),
        live("b4", "undo", top_score=0.5),
        live("b5", "a clean one"),
        live("b6", "a run's", version="v3:run"),
    ]
    feedback = {"b1": Feedback(trace_id="b1", helpful=False, note="wrong")}
    picked, n_live = sample(traces, feedback, policy, T0 - timedelta(days=1), T0 + timedelta(days=1))
    assert n_live == 5 and [s.trace_id for s in picked] == [
        "b1",
        "b2",
        "b3",
    ]  # capped at 3, rule-picked first
    by = {s.trace_id: s for s in picked}
    assert by["b1"].rules == ["dissatisfied"] and by["b1"].helpful is False and by["b1"].note == "wrong"
    assert by["b1"].primary == "dissatisfied" and by["b1"].failed and by["b3"].failed and not by["b2"].failed
    assert by["b2"].top_source is None and by["b1"].top_source == "k8s-deployment-md"
    picked, _ = sample(
        traces, feedback, Policy(random_share=1.0), T0 - timedelta(days=1), T0 + timedelta(days=1)
    )
    assert [s.rules for s in picked if s.trace_id == "b5"] == [["random"]]

    class ScriptedJudge:
        name = "fake"

        def read_state(self, text):
            g = 0.2 if "maxSurge" in text else 0.9
            return Judgement(
                grounded=g, fabricated=0.1, gap_named=0.0, model="fake-jev", cost_usd=0.001, ms=5
            )

    from infra_docs_rag.chunk.chunks import Chunk

    chunk = Chunk(
        chunk_id="k8s-deployment-md:10",
        source_id="k8s-deployment-md",
        document_id="d",
        source_uri="u",
        project="Kubernetes",
        section_path=["Deployments", "Rolling Back"],
        start=0,
        end=10,
        text="Run kubectl rollout undo to roll back.",
    )
    samples, _ = sample(traces, feedback, policy, T0 - timedelta(days=1), T0 + timedelta(days=1))
    read = judge_samples(
        samples, {t.trace_id: t for t in traces}, {chunk.chunk_id: chunk}, ScriptedJudge(), log=lambda _: None
    )
    assert read == 2  # b1 and b3 answered; b2 abstained
    by = {s.trace_id: s for s in samples}
    assert by["b3"].judgement.grounded == 0.2 and by["b2"].judgement is None
    window = Window(
        name="2026-W42",
        since=(T0 - timedelta(days=1)).isoformat(),
        until=(T0 + timedelta(days=1)).isoformat(),
        live=5,
        samples=samples,
        policy=policy,
    )
    path = window.save(tmp_path / "online")
    assert path.name == "2026-W42.json" and load_windows(tmp_path / "online")[0].samples[0].trace_id == "b1"
    assert load_windows(tmp_path / "nowhere") == []
    groups = clusters(samples)
    assert [c.key for c in groups] == [
        "dissatisfied · answered · k8s-deployment-md",
        "no-answer · no-passage · no passage",
        "uncited · answered · k8s-deployment-md",
    ]
    assert isinstance(groups[0], Cluster)
    rows, index, embedder = dataset_records
    run = make_run(rows, index, embedder, fake_generator, BASE_SCRIPT, "ds-base")
    base = baseline_rates(run)
    assert base["no_answer_rate"] == pytest.approx(2 / 6) and base["grounded_share"] == 1.0
    win = window_rates(traces, T0 - timedelta(days=1), T0 + timedelta(days=1), samples)
    assert win["no_answer_rate"] == pytest.approx(1 / 5) and win["grounded_share"] == 0.5
    rates = drift(win, base, policy)
    by_name = {r.name: r for r in rates}
    assert by_name["grounded_share"].drifted and by_name["grounded_share"].delta == pytest.approx(-0.5)
    assert not by_name["no_answer_rate"].drifted and by_name["no_answer_rate"].fmt(0.2) == "20%"
    assert by_name["query_words_median"].fmt(3.0) == "3"

    out = tmp_path / "online.md"
    write_report([window], [win], [rates], [groups], run, 5, tmp_path / "none.yaml", out)
    text = out.read_text()
    for heading in (
        "## Summary",
        "## 1. The sampling policy",
        "## 2. Window `2026-W42`",
        "**Failure clusters**",
        "**Drift**",
        "## 3. Promoted",
        "## 4. Reading",
    ):
        assert heading in text
    assert "| `grounded_share` |" in text and "**drift**" in text and "Nothing promoted yet" in text
    assert "| `b1` | `dissatisfied` | answered | 0.75 | 100% | not helpful (wrong) | 0.90 · 0.10 |" in text


def test_a_failed_sample_becomes_a_record_through_the_writer(tmp_path):
    from infra_docs_rag.chunk.chunks import Chunk

    chunk = Chunk(
        chunk_id="k8s-deployment-md:10",
        source_id="k8s-deployment-md",
        document_id="d",
        source_uri="u",
        project="Kubernetes",
        section_path=["Deployments", "Rolling Back"],
        start=0,
        end=10,
        text="Run kubectl rollout undo to roll back.",
    )
    t = live("c1", "how do I undo a release?", claim_share=0.4)
    s = Sample(
        trace_id="c1",
        at=t.at,
        question=t.question,
        version="v3",
        outcome="answered",
        rules=["uncited"],
        words=6,
        claim_share=0.4,
        passages=1,
        top_source="k8s-deployment-md",
    )
    prompt = harvest_prompt(t, {chunk.chunk_id: chunk})
    assert (
        "QUESTION\nhow do I undo a release?" in prompt
        and "[1] Kubernetes › Deployments › Rolling Back" in prompt
    )
    assert "WHAT THE SERVICE ANSWERED\nRun kubectl rollout undo.[1]" in prompt

    class Writer:
        def ask(self, text):
            class Reply:
                text = (
                    '{"expect": "answer", "reference_answer": "Run kubectl rollout undo.", "contexts": [1], '
                    '"category": "fact-lookup", "difficulty": "easy", "failure_mode": "uncited-claims", '
                    '"tags": ["kubernetes", "rollback"], "why": "the service left claims uncited"}'
                )

            return Reply()

    records = harvest(
        [s],
        {"c1": t},
        {chunk.chunk_id: chunk},
        Writer(),
        "section-aware-512-64-context",
        10,
        log=lambda _: None,
    )
    assert len(records) == 1
    r = records[0]
    assert r.id == "live-c1" and r.origin == "traces" and r.split == "test" and r.contexts == [chunk.chunk_id]
    assert r.source_doc_ids == ["k8s-deployment-md"] and r.failure_mode == "uncited-claims"
    assert "sampled as uncited" in r.note and "40% of claims cited" in r.note and "for review" in r.note
    abstain = to_record(
        s,
        t,
        {
            "expect": "abstain",
            "reference_answer": "The indexed documentation does not cover this question.",
            "contexts": [1],
            "category": "no-answer",
            "failure_mode": "answered-anyway",
        },
        {chunk.chunk_id: chunk},
        "c",
    )
    assert abstain.contexts == [] and abstain.source_doc_ids == []
    path = save_harvested(records, "section-aware-512-64-context", "2026-W42", tmp_path / "harvested.yaml")
    assert path.read_text().startswith("# Records harvested") and "live-c1" in path.read_text()
    records_file = tmp_path / "records.yaml"
    records_file.write_text(
        "# the records\nchunking: section-aware-512-64-context\nrecords:\n  - id: scale\n    query: q\n"
    )
    assert merge_into_records(records, records_file) == 1 and merge_into_records(records, records_file) == 0
    text = records_file.read_text()
    assert "# the records" in text and "\n  - id: live-c1\n" in text and text.count("- id:") == 2
    bad = harvest(
        [s],
        {"c1": t},
        {chunk.chunk_id: chunk},
        type("W", (), {"ask": lambda self, x: type("R", (), {"text": "nope"})()})(),
        "c",
        10,
        log=lambda _: None,
    )
    assert bad == []


def test_a_live_answer_traces_as_live_and_the_repo_policy_loads(small, fake_generator):  # noqa: F811
    index, embedder, reranker = small
    r = Retriever(index, embedder, reranker)
    gen = fake_generator({"kubectl scale replicas": [("Run kubectl scale.", [1])]})
    a = answer(
        "kubectl scale replicas",
        r,
        gen,
        Retrieval(k=2, mode="hybrid", min_score=0.1, rerank=Rerank(model="bge-base", candidates=3)),
    )
    t = trace_answer(a, "v-test", None, "cite-v1+t", None, gen.prices)
    assert is_live(t) and rules_for(t, Policy(low_confidence_top_score=0.0), None) == []
    policy = load_policy(ROOT / POLICY)
    assert policy.window_days == 7 and set(policy.rules) == {
        "dissatisfied",
        "uncited",
        "low-confidence",
        "no-answer",
        "long-context",
        "long-query",
        "identifier",
        "error",
        "random",
    }
    log = FeedbackLog(Path("/nonexistent/feedback.jsonl"))
    assert log.read() == [] and log.by_trace() == {}
