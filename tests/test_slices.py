from pathlib import Path

import pytest
import yaml
from test_dataset import record
from test_evaluation import ScriptedJudge
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
from infra_docs_rag.embed.index import build
from infra_docs_rag.evaluation.dataset import FAILURE_MODES
from infra_docs_rag.evaluation.dataset_judge import QUESTIONS, JevReader, Reading, reading_state
from infra_docs_rag.evaluation.dataset_run import expect_of, kind_of, run_dataset, to_run
from infra_docs_rag.evaluation.metrics_report import write_report
from infra_docs_rag.evaluation.runs import Run
from infra_docs_rag.evaluation.slices import (
    ALL,
    MatrixEntry,
    SliceGate,
    check_matrix,
    check_slice_gates,
    dataset_metrics,
    diagnoses,
    load_matrix,
    slice_metrics,
    slices_of,
)
from infra_docs_rag.generate.prompt import ABSTAIN
from infra_docs_rag.retrieve.retriever import Retrieval

ROOT = Path(__file__).parents[1]


class ScriptedReader:
    name = "fake-reader"

    def __init__(self, readings=None):
        self.readings = readings or {}  # answer text fragment → (relevant, agrees, deprecation, readings)
        self.calls = []

    def read(self, context, question, reference, answer_text):
        self.calls.append((question, reference))
        r, a, d, n = next((v for k, v in self.readings.items() if k in answer_text), (0.9, 0.9, 0.9, 0.9))
        return Reading(
            relevant=r,
            reference_agrees=a,
            deprecation_named=d,
            readings_named=n,
            model="fake",
            cost_usd=0.001,
            ms=5,
        )


@pytest.fixture
def dataset_run(fake_embedder, fake_generator, guide):  # noqa: F811
    """Six records through the small guide: a clean one, a wrong-passage one, a false abstention, a
    no-answer that is answered anyway, a multi-hop and a stale one."""
    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    scaling = next(c.chunk_id for c in chunks if c.section == "Scaling")
    rolling = next(c.chunk_id for c in chunks if c.section == "Rolling Update")
    records = [
        record(id="scale", query="kubectl scale replicas", contexts=[scaling]),
        record(id="surge", query="maxSurge", contexts=[rolling], failure_mode="wrong-passage", split="test"),
        record(
            id="silent",
            query="change the number of replicas",
            contexts=[scaling],
            failure_mode="false-abstention",
        ),
        record(
            id="bread",
            query="sourdough replicas recipe",
            category="no-answer",
            expect="abstain",
            reference_answer=ABSTAIN,
            contexts=[],
            source_doc_ids=[],
            failure_mode="answered-anyway",
            origin="hand-written",
        ),
        record(
            id="two",
            query="scale and update replicas",
            category="multi-hop",
            contexts=[scaling, rolling],
            failure_mode="missing-hop",
            origin="hand-written",
            difficulty="hard",
        ),
        record(
            id="old",
            query="old replicas flag",
            category="stale",
            contexts=[scaling],
            failure_mode="stale-answer",
            origin="hand-written",
        ),
    ]
    generator = fake_generator(
        {
            "kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])],
            "maxSurge": [("maxSurge allows extra Pods.", [2])],  # cites the wrong passage
            "sourdough replicas recipe": [("Use kubectl scale, it is like bread.", [1])],  # answered anyway
            "scale and update replicas": [("Scale with kubectl scale.", [1]), (" Surge with maxSurge.", [2])],
            "old replicas flag": [
                ("Use the old flag.", [1, 2])
            ],  # cites both read passages: the named one is among them
        }
    )
    judge = ScriptedJudge({"like bread": (0.2, 0.8, 0.0)})
    reader = ScriptedReader({"old flag": (0.9, 0.3, 0.2, 0.9), "like bread": (0.3, 0.1, 0.9, 0.9)})
    dg = run_dataset(
        records,
        index,
        embedder,
        generator,
        retrieval=Retrieval(k=2, mode="dense"),
        judge=judge,
        reader=reader,
        log=lambda _: None,
    )
    return records, to_run(dg, "ds-fake", "v1", "the fake pipeline"), reader


def test_a_record_maps_to_the_run_format():
    assert (
        kind_of(
            record(
                category="no-answer",
                expect="abstain",
                reference_answer=ABSTAIN,
                contexts=[],
                source_doc_ids=[],
            )
        )
        == "out-of-scope"
    )
    assert (
        kind_of(record(category="stale")) == "stale"
        and kind_of(record(category="adversarial")) == "adversarial"
    )
    assert (
        expect_of(record(category="ambiguous", expect="clarify")) == "partial"
        and expect_of(record()) == "answer"
    )


def test_the_run_marks_right_passages_by_the_record_contexts_and_saves_the_readings(dataset_run, tmp_path):
    records, run, reader = dataset_run
    by = run.by_id()
    assert by["scale"].held and by["scale"].right_retrieved and by["scale"].right_cited
    assert by["surge"].verdict == "cited the wrong passage" and by["surge"].right_retrieved
    assert (
        by["silent"].outcome == "abstained" and by["silent"].verdict == "abstained"
    )  # no script → abstention
    assert (
        by["bread"].verdict == "answered anyway" and by["bread"].right_retrieved is None
    )  # no contexts, no flag
    assert by["two"].held and set(by["two"].cited_chunk_ids) == set(records[4].contexts)
    assert by["old"].reading is not None and by["old"].reading.deprecation_named == 0.2
    assert by["silent"].reading is None  # nothing to read on an abstention
    assert all(ref for _, ref in reader.calls) and len(reader.calls) == 5
    saved = run.save(tmp_path / "ds-fake")
    loaded = Run.load(saved)
    assert loaded.by_id()["old"].reading.relevant == 0.9 and loaded.manifest.questions == 6
    old_format = Run.load(ROOT / "eval/runs/v3-budget-claude-code")  # a run saved before readings existed
    assert old_format.cases[0].reading is None


def test_dataset_metrics_add_context_recall_relevance_and_the_failure_mode_signals(dataset_run):
    records, run, _ = dataset_run
    m = dataset_metrics(run, records)
    assert m["retrieval.context_recall"].value == pytest.approx(
        (1 + 1 + 1 + 1 + 1) / 5
    )  # five records name passages, k=2 reads both
    assert m["retrieval.context_precision"].value == pytest.approx((0.5 + 0.5 + 0.5 + 1.0 + 0.5) / 5)
    assert m["relevance.judge_relevant"].shown == "4/5 (80%)"  # bread read as irrelevant; five answered
    assert m["relevance.reference_agrees"].shown == "3/5 (60%)"  # old and bread disagree
    assert m["abstention.expected"].shown == "0/1 (0%)" and m["abstention.false"].shown == "1/5 (20%)"
    assert m["hops.covered"].shown == "1/1 (100%)" and m["stale.deprecation_named"].shown == "0/1 (0%)"
    assert m["ambiguity.readings_named"].value is None  # no ambiguous record
    assert m["faithfulness.judge_fabricated"].name == "faithfulness.judge_fabricated"
    assert m["grounding.judge_fabricated"].value == m["faithfulness.judge_fabricated"].value
    assert (
        m["faithfulness.judge_flagged_held"].shown == "0/5 (0%)"
    )  # bread is flagged, did not hold mechanically
    sliced = slice_metrics(run, records)
    assert set(slices_of(records)) == set(sliced) and ALL in sliced
    assert sliced["category=no-answer"]["abstention.expected"].value == 0.0
    assert sliced["failure_mode=wrong-passage"]["faithfulness.right_cited"].shown == "0/1 (0%)"
    assert (
        sliced["split=test"]["answers.held"].value == 0.0
        and sliced["split=dev"]["answers.held"].shown == "3/5 (60%)"
    )


def test_the_matrix_and_the_slice_gates_read_the_slices(dataset_run):
    records, run, _ = dataset_run
    sliced = slice_metrics(run, records)
    entries = [
        MatrixEntry(failure_mode="none", metric="answers.held", direction="higher", rule="min", value=0.9),
        MatrixEntry(
            failure_mode="answered-anyway",
            metric="abstention.expected",
            direction="higher",
            rule="min",
            value=0.9,
            blocker=True,
        ),
        MatrixEntry(
            failure_mode="wrong-reading",
            metric="ambiguity.readings_named",
            direction="higher",
            rule="min",
            value=0.8,
        ),
        MatrixEntry(
            failure_mode="stale-answer",
            metric="stale.deprecation_named",
            direction="higher",
            rule="min",
            value=0.8,
            slice=ALL,
        ),
    ]
    results = check_matrix(entries, sliced)
    assert [r.passed for r in results] == [True, False, None, False]
    assert results[0].entry.slice == "failure_mode=none" and results[3].entry.slice == ALL
    assert "no record in slice" in results[2].detail
    with pytest.raises(ValueError, match="rule is max"):
        MatrixEntry(failure_mode="none", metric="x", direction="lower", rule="min", value=1)
    with pytest.raises(ValueError, match="unknown failure mode"):
        MatrixEntry(failure_mode="nope", metric="x", direction="higher", rule="min", value=1)
    with pytest.raises(ValueError, match="no run measures"):
        check_matrix(
            [MatrixEntry(failure_mode="none", metric="made.up", direction="higher", rule="min", value=1)],
            sliced,
        )
    gates = [
        SliceGate(
            slice="category=no-answer", metric="abstention.expected", rule="min", value=1.0, why="blocker"
        ),
        SliceGate(slice=ALL, metric="answers.held", rule="no-drop", tolerance=0.03, why="delta"),
        SliceGate(
            slice=ALL, metric="relevance.judge_relevant", rule="min", value=0.5, critical=False, why="warn"
        ),
    ]
    results = check_slice_gates(gates, sliced, None)
    assert [g.result.passed for g in results] == [False, False, True]
    assert (
        results[0].blocks and results[1].skipped and not results[1].blocks
    )  # no baseline: skipped, not failed
    assert "no baseline" in results[1].result.detail
    again = check_slice_gates(gates, sliced, sliced)
    assert again[1].result.passed  # against itself
    with pytest.raises(ValueError, match="has no record"):
        check_slice_gates(
            [SliceGate(slice="category=ghost", metric="answers.held", rule="min", value=1)], sliced, None
        )


def test_diagnosis_tells_retrieval_from_generation(dataset_run):
    records, run, _ = dataset_run
    d = diagnoses(run, records)
    assert d["scale"] == ("held", "")
    assert d["surge"][0] == "generation" and "wrong passage" in d["surge"][1]
    assert d["silent"] == ("generation", "abstained with a named passage in hand")
    assert d["bread"][0] == "both"
    assert d["old"][0] == "reading" and "disagrees with the reference" in d["old"][1]


def test_the_metrics_report_writes_every_section(dataset_run, tmp_path):
    records, run, _ = dataset_run
    sliced = slice_metrics(run, records)
    matrix = load_matrix(ROOT / "eval/matrix.yaml")
    results = check_matrix(matrix.matrix, sliced)
    present = [g for g in matrix.gates if g.slice in sliced]  # the six records cover a few slices
    gates = check_slice_gates(present, sliced, sliced)
    out = tmp_path / "metrics.md"
    write_report(run, records, sliced, results, gates, diagnoses(run, records), run, out)
    text = out.read_text()
    for heading in (
        "## Summary",
        "## 1. The matrix",
        "## 2. Scores",
        "## 3. Retrieval problem",
        "## 4. The gates",
        "## 5. Reading",
    ):
        assert heading in text
    assert "| `answered-anyway` | `abstention.expected` | higher | ≥ 0.9 · blocker |" in text
    assert "### `surge` · fact-lookup · `wrong-passage` · expected answer: cited the wrong passage" in text
    assert "**both**" in text and "`split=test`" in text
    assert {e.failure_mode for e in matrix.matrix} == set(FAILURE_MODES)


def test_the_reader_builds_the_state_with_the_reference_and_runs_the_cli(tmp_path):
    import json

    from infra_docs_rag.retrieve.retriever import Context, Filter, Retrieval

    ctx = Context(
        query="q",
        retrieval=Retrieval(filter=Filter()),
        sources=[],
        candidates=0,
        top_score=None,
        below=0,
        over_budget=0,
    )
    state = reading_state(ctx, "the question", "the reference", "the answer [1]")
    assert state.index("QUESTION") < state.index("REFERENCE ANSWER") < state.index("ANSWER\nthe answer")
    script = tmp_path / "jev"
    reply = {
        "answers": {k: {"noul": 0.8} for k in QUESTIONS},
        "model": "jev-test",
        "cost_usd": 0.004,
        "latency_ms": 12,
    }
    script.write_text(f"#!/bin/sh\ncat > '{tmp_path}/stdin'\ncat <<'JSON'\n{json.dumps(reply)}\nJSON\n")
    script.chmod(0o755)
    reader = JevReader(binary=str(script))
    r = reader.read(ctx, "the question", "the reference", "the answer")
    assert r.relevant == 0.8 and r.model == "jev-test" and r.cost_usd == 0.004
    assert "REFERENCE ANSWER\nthe reference" in (tmp_path / "stdin").read_text()
    assert json.loads(reader.request.read_text())["questions"].keys() == QUESTIONS.keys()
    bad = tmp_path / "bad"
    bad.write_text("#!/bin/sh\necho nope\n")
    bad.chmod(0o755)
    with pytest.raises(ValueError, match="unexpected"):
        JevReader(binary=str(bad)).read(ctx, "q", "r", "a")


def test_the_repo_matrix_loads_and_names_every_failure_mode():
    matrix = load_matrix(ROOT / "eval/matrix.yaml")
    assert {e.failure_mode for e in matrix.matrix} == set(FAILURE_MODES)
    assert any(g.rule == "no-drop" for g in matrix.gates) and any(
        g.slice == "category=no-answer" for g in matrix.gates
    )
    raw = yaml.safe_load((ROOT / "eval/matrix.yaml").read_text())
    assert len(raw["matrix"]) == len(FAILURE_MODES)
