from pathlib import Path

import pytest
from test_dataset import record
from test_evaluation import ScriptedJudge
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)
from test_slices import ScriptedReader

from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
from infra_docs_rag.embed.index import build
from infra_docs_rag.evaluation.dataset import dataset_version, load_dataset
from infra_docs_rag.evaluation.dataset_run import run_dataset, to_run
from infra_docs_rag.evaluation.regression import BASELINE, Baseline, compare, differences
from infra_docs_rag.evaluation.regression_report import write_report
from infra_docs_rag.evaluation.runs import RUNS_DIR, Run
from infra_docs_rag.evaluation.slices import ALL, MatrixEntry, MatrixFile, SliceGate, load_matrix
from infra_docs_rag.generate.prompt import ABSTAIN
from infra_docs_rag.pipeline.manifest import PipelineManifest
from infra_docs_rag.retrieve.retriever import Retrieval

ROOT = Path(__file__).parents[1]

MATRIX = MatrixFile(
    matrix=[
        MatrixEntry(failure_mode="none", metric="answers.held", direction="higher", rule="min", value=0.9),
        MatrixEntry(
            failure_mode="answered-anyway",
            metric="abstention.expected",
            direction="higher",
            rule="min",
            value=0.9,
            blocker=True,
        ),
    ],
    gates=[
        SliceGate(
            slice="category=no-answer", metric="abstention.expected", rule="min", value=1.0, why="blocker"
        ),
        SliceGate(slice=ALL, metric="answers.held", rule="no-drop", tolerance=0.03, why="whole set"),
        SliceGate(slice="split=test", metric="answers.held", rule="no-drop", tolerance=0.05, why="held out"),
        SliceGate(
            slice="category=multi-hop",
            metric="hops.covered",
            rule="no-drop",
            tolerance=0.2,
            critical=False,
            why="five records",
        ),
    ],
)


def lineage(**overrides) -> PipelineManifest:
    base = dict(
        version="v1",
        ingest_version="test",
        chunking_version="section-aware+c1",
        embedding_version="fake@r1",
        index_version="i1",
        prompt_version="cite-v1+p1",
        retrieval_version="v1+r1",
        generator="fake-model",
        backend="fake",
    )
    base.update(overrides)
    return PipelineManifest(**base)


@pytest.fixture
def records(fake_embedder, guide):  # noqa: F811
    """Six records through the small guide, and the index they are answered from."""
    embedder = fake_embedder(max_tokens=512)
    chunks = chunk_documents([guide], DEFAULT_CHUNKING, embedder, {"guide": "Kubernetes"})
    index = build(embedder, chunks, ingest_version="test", chunking=DEFAULT_CHUNKING)
    scaling = next(c.chunk_id for c in chunks if c.section == "Scaling")
    rolling = next(c.chunk_id for c in chunks if c.section == "Rolling Update")
    rows = [
        record(id="scale", query="kubectl scale replicas", contexts=[scaling]),
        record(id="surge", query="maxSurge", contexts=[rolling], failure_mode="wrong-passage", split="test"),
        record(
            id="silent",
            query="change the number of replicas",
            contexts=[scaling],
            failure_mode="false-abstention",
            difficulty="hard",
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
            split="test",
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
    return rows, index, embedder


def make_run(records, index, embedder, fake_generator, script, name, version="v1", **lineage_overrides):
    generator = fake_generator(script)
    judge = ScriptedJudge({"like bread": (0.2, 0.8, 0.0)})
    reader = ScriptedReader({"like bread": (0.3, 0.1, 0.9, 0.9)})
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
    return to_run(
        dg,
        name,
        version,
        f"the {version} pipeline",
        lineage(version=version, **lineage_overrides),
        dataset_version(records),
    )


BASE_SCRIPT = {
    "kubectl scale replicas": [("Run kubectl scale to change the replicas.", [1])],
    "maxSurge": [("maxSurge allows extra Pods.", [2])],  # the wrong passage
    # "change the number of replicas": no script → abstains with the passage in hand
    # "sourdough replicas recipe": no script → abstains, as expected
    "scale and update replicas": [("Scale with kubectl scale.", [1]), (" Surge with maxSurge.", [2])],
    "old replicas flag": [("Use the old flag.", [1, 2])],
}
# The candidate fixes the wrong passage and the false abstention (two wins), cites one passage of the
# two-passage answer, and answers the no-answer record (one loss, on the slice built to catch it).
CAND_SCRIPT = {
    **BASE_SCRIPT,
    "maxSurge": [("maxSurge allows extra Pods.", [1])],
    "change the number of replicas": [("Run kubectl scale.", [1])],
    "sourdough replicas recipe": [("Use kubectl scale, it is like bread.", [1])],
    "scale and update replicas": [("Scale with kubectl scale.", [1]), (" Then surge.", [1])],
}


@pytest.fixture
def two_runs(records, fake_generator):
    rows, index, embedder = records
    base = make_run(rows, index, embedder, fake_generator, BASE_SCRIPT, "ds-base")
    cand = make_run(
        rows, index, embedder, fake_generator, CAND_SCRIPT, "ds-cand", "v2", retrieval_version="v2+r2"
    )
    return rows, base, cand


def test_the_dataset_version_counts_the_records_and_hashes_their_content(records):
    rows, _, _ = records
    v = dataset_version(rows)
    assert v.startswith("6+") and len(v) == len("6+") + 12
    assert dataset_version(list(reversed(rows))) == v  # order does not matter
    edited = [r.model_copy(update={"reference_answer": "Run kubectl scale, carefully."}) for r in rows[:1]]
    assert dataset_version(edited + rows[1:]) != v  # one field of one record does
    assert dataset_version(rows[1:]) != v


def test_a_run_carries_its_dataset_version_and_the_baseline_pins_it(two_runs, tmp_path):
    rows, base, cand = two_runs
    assert base.manifest.dataset == dataset_version(rows)
    pin = Baseline.of(base, note="the first")
    assert pin.run == "ds-base" and pin.retrieval_version == "v1+r1" and pin.dataset == base.manifest.dataset
    path = pin.save(tmp_path / "baseline.yaml")
    loaded = Baseline.load(path)
    assert loaded == pin and path.read_text().startswith("# The regression baseline")
    assert loaded.problems(base, rows) == []
    # the dataset edited since the baseline answered it
    edited = [rows[0].model_copy(update={"reference_answer": "Something else."})] + rows[1:]
    problems = loaded.problems(base, edited)
    assert (
        len(problems) == 1
        and "the dataset is" in problems[0]
        and "run `dataset run --version v1`" in problems[0]
    )
    # a run that is not the one pinned, under a different retrieval
    problems = loaded.problems(cand, rows)
    assert any("names run ds-base, not ds-cand" in p for p in problems)
    assert any(p.startswith("retrieval_version: the pin says v1+r1, the run says v2+r2") for p in problems)
    assert any(p.startswith("version: the pin says v1, the run says v2") for p in problems)
    no_lineage = base.model_copy(deep=True)
    no_lineage.manifest.lineage = None
    with pytest.raises(ValueError, match="carries no lineage"):
        Baseline.of(no_lineage)
    no_dataset = base.model_copy(deep=True)
    no_dataset.manifest.dataset = None
    with pytest.raises(ValueError, match="carries no dataset version"):
        Baseline.of(no_dataset)
    with pytest.raises(FileNotFoundError, match="baseline set"):
        Baseline.load(tmp_path / "missing.yaml")


def test_compare_blocks_a_slice_regression_the_aggregate_hides(two_runs):
    rows, base, cand = two_runs
    reg = compare(base, cand, rows, MATRIX)
    held_before, held_after = reg.headline()
    assert held_before.shown == "4/6 (67%)" and held_after.shown == "5/6 (83%)"
    assert reg.aggregate_improved and reg.aggregate_delta == pytest.approx(1 / 6)
    assert {x.id for x in reg.wins} == {"surge", "silent"} and [x.id for x in reg.losses] == ["bread"]
    assert reg.neutral == [] and reg.missing == []
    assert [x.id for x in reg.same_passages] == [
        "surge",
        "silent",
        "bread",
    ]  # the same retrieval, so every flip is the generator's
    assert reg.differs == ["retrieval"]
    # the whole-set gate passes, the no-answer slice fails, and that is what decides
    by_slice = {(g.gate.slice, g.gate.metric): g for g in reg.gates}
    assert by_slice[(ALL, "answers.held")].result.passed
    assert by_slice[("category=no-answer", "abstention.expected")].blocks
    assert by_slice[("split=test", "answers.held")].result.passed  # surge won, bread lost: 1/2 → 1/2
    assert not by_slice[("category=multi-hop", "hops.covered")].result.passed  # 1/1 → 0/1, a warning
    assert not by_slice[("category=multi-hop", "hops.covered")].blocks
    assert [g.gate.slice for g in reg.masked] == ["category=no-answer"]
    assert reg.decision == "block" and reg.verdict.startswith("blocked: 1 critical gate(s) failed")
    assert "the aggregate would have hidden it" in reg.verdict
    assert len(reg.warnings) == 1
    # the matrix before and after: the blocker on answered-anyway flips to a miss
    after = {x.entry.failure_mode: x.passed for x in reg.matrix_after}
    before = {x.entry.failure_mode: x.passed for x in reg.matrix_before}
    assert before["answered-anyway"] is True and after["answered-anyway"] is False


def test_the_decision_is_ship_or_hold_by_the_tally_when_the_gates_pass(records, fake_generator):
    rows, index, embedder = records
    base = make_run(rows, index, embedder, fake_generator, BASE_SCRIPT, "ds-base")
    better = {k: v for k, v in CAND_SCRIPT.items() if k != "sourdough replicas recipe"}
    cand = make_run(rows, index, embedder, fake_generator, better, "ds-better", "v2")
    reg = compare(base, cand, rows, MATRIX)
    assert reg.decision == "ship" and len(reg.wins) == 2 and not reg.losses and reg.blocking == []
    assert reg.verdict.startswith("ship with 1 warning(s)")  # hops.covered dropped, a warning
    worse = {**BASE_SCRIPT, "kubectl scale replicas": [("Run kubectl scale.", [2])]}  # the wrong passage
    cand = make_run(rows, index, embedder, fake_generator, worse, "ds-worse", "v2")
    reg = compare(base, cand, rows, MatrixFile(matrix=MATRIX.matrix, gates=[]))  # no gate to block
    assert [x.id for x in reg.losses] == ["scale"] and not reg.wins
    assert reg.decision == "hold" and "more records lost than won" in reg.verdict
    same = compare(base, base, rows, MATRIX)
    assert same.decision == "ship" and same.changed == [] and same.differs == []
    with pytest.raises(ValueError, match="share no record"):
        compare(base, cand, [rows[0].model_copy(update={"id": "elsewhere"})], MATRIX)


def test_differences_name_what_is_not_the_same(two_runs):
    _, base, cand = two_runs
    assert differences(base, cand) == ["retrieval"]
    other = cand.model_copy(deep=True)
    other.manifest.dataset = "7+abc"
    other.manifest.effort = "high"
    other.manifest.lineage = other.manifest.lineage.model_copy(update={"prompt_version": "cite-v2+p2"})
    assert differences(base, other) == ["prompt", "retrieval", "dataset", "effort"]


def test_the_regression_report_writes_every_section(two_runs, tmp_path):
    rows, base, cand = two_runs
    reg = compare(base, cand, rows, MATRIX)
    out = tmp_path / "regression.md"
    write_report(reg, out)
    text = out.read_text()
    for heading in (
        "## Summary",
        "## 1. The gates",
        "## 2. Scores by slice",
        "## 3. Record by record",
        "## 4. The matrix",
        "## 5. Reading the report",
    ):
        assert heading in text
    assert "**Decision: BLOCK.**" in text and "**Hidden by the aggregate.**" in text
    assert "The two runs differ in retrieval only" in text
    assert "| `bread` | no-answer | `answered-anyway` | test |" in text and "▼ lost" in text
    assert "3 of them (`surge`, `silent`, `bread`) read the same passages" in text and "| same |" in text
    assert "### `bread` · no-answer · `answered-anyway` · expected abstain: answered anyway" in text
    assert "- `surge` (fact-lookup, `wrong-passage`): cited the wrong passage → held." in text
    assert "| `all` | 6 | 4/6 (67%) → 5/6 (83%) ▲ |" in text
    assert "| `category=no-answer` | 1 |" in text and "1/1 (100%) → 0/1 (0%) ▼" in text
    assert "dataset compare --candidate ds-cand`" in text and "--baseline ds-base" not in text
    write_report(reg, out, default_baseline=False)
    assert "`--baseline ds-base`" in out.read_text()


def test_the_repo_baseline_pin_matches_its_run_and_the_dataset():
    pin = Baseline.load(ROOT / BASELINE)
    run = Run.load(ROOT / RUNS_DIR / pin.run)
    assert pin.problems(run, load_dataset(ROOT / "eval/dataset").records) == []
    matrix = load_matrix(ROOT / "eval/matrix.yaml")
    slice_gates = [g for g in matrix.gates if g.slice != ALL and g.rule in ("no-drop", "no-rise")]
    assert any(g.critical for g in slice_gates), "a critical per-slice regression gate"
