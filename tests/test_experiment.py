from pathlib import Path

import pytest
from test_evaluation import two_runs  # noqa: F401  (a fixture)
from test_regression import BASE_SCRIPT, CAND_SCRIPT, MATRIX, make_run
from test_regression import records as dataset_records  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.evaluation.dataset import load_dataset
from infra_docs_rag.evaluation.experiment import (
    FROZEN,
    VARIABLES,
    ExperimentManifest,
    Frozen,
    benchmark_version,
    file_version,
)
from infra_docs_rag.evaluation.gates import Gate
from infra_docs_rag.evaluation.regression import Baseline, compare, differences
from infra_docs_rag.evaluation.regression_report import write_report
from infra_docs_rag.evaluation.rubric import grader_version, load_rubric
from infra_docs_rag.evaluation.runs import RUNS_DIR, Run
from infra_docs_rag.pipeline.releases import Release, Releases, promote

ROOT = Path(__file__).parents[1]


@pytest.fixture
def assets(tmp_path):
    """A matrix, gates, safety file and benchmark files under tmp, so hashes are the test's own."""
    matrix, gates, safety = tmp_path / "matrix.yaml", tmp_path / "gates.yaml", tmp_path / "safety.yaml"
    matrix.write_text("matrix: []\n")
    gates.write_text("gates: []\n")
    bench = (tmp_path / "retrieval.yaml", tmp_path / "hybrid.yaml", tmp_path / "grounding.yaml")
    for p in bench:
        p.write_text(f"queries: []  # {p.name}\n")
    return matrix, gates, safety, bench


def experiment_of(run, dataset, assets, grader="answer-quality-v1+abc"):
    matrix, gates, safety, bench = assets
    return ExperimentManifest.of(
        run.manifest, dataset, grader, benchmark_version(bench), matrix=matrix, gates=gates, safety=safety
    )


def test_file_and_benchmark_versions_hash_the_content(assets, tmp_path):
    matrix, gates, safety, bench = assets
    v = file_version(matrix)
    assert v.startswith("matrix+") and len(v) == len("matrix+") + 12
    assert file_version(safety) == "none"  # not written yet
    matrix.write_text("matrix: [1]\n")
    assert file_version(matrix) != v
    b = benchmark_version(bench)
    bench[1].write_text("queries: [x]\n")
    assert benchmark_version(bench) != b and b.startswith("benchmark+")


def test_the_experiment_manifest_names_every_variable_and_what_differs(
    dataset_records,  # noqa: F811
    fake_generator,
    assets,
):
    rows, index, embedder = dataset_records
    base = make_run(rows, index, embedder, fake_generator, BASE_SCRIPT, "ds-base")
    e = experiment_of(base, base.manifest.dataset, assets)
    assert e.version == "v1" and e.retrieval_version == "v1+r1" and e.dataset_version == base.manifest.dataset
    assert (
        e.grader_version == "answer-quality-v1+abc" and e.safety_version == "none" and e.judge == "fake-jev"
    )
    assert e.matrix_version.startswith("matrix+") and e.benchmark_version.startswith("benchmark+")
    assert set(VARIABLES) == {
        "retrieval",
        "prompt",
        "corpus",
        "model",
        "dataset",
        "grader",
        "gates",
        "safety",
    }
    assert e.differs(e) == [] and [line.split()[0] for line in e.lines()] == list(VARIABLES)
    other = e.model_copy(
        update={"retrieval_version": "v2+r2", "grader_version": "answer-quality-v2+def", "effort": "high"}
    )
    assert e.differs(other) == ["retrieval", "model", "grader"]
    assert other.value("grader") == "answer-quality-v2+def · fake-jev" and e.value("safety") == "none"
    no_lineage = base.model_copy(deep=True)
    no_lineage.manifest.lineage = None
    with pytest.raises(ValueError, match="carries no lineage"):
        experiment_of(no_lineage, None, assets)


def test_compare_refuses_a_confounded_change_unless_told(dataset_records, fake_generator, assets, tmp_path):  # noqa: F811
    rows, index, embedder = dataset_records
    base = make_run(rows, index, embedder, fake_generator, BASE_SCRIPT, "ds-base")
    cand = make_run(
        rows, index, embedder, fake_generator, CAND_SCRIPT, "ds-cand", "v2", retrieval_version="v2+r2"
    )
    base.manifest.experiment = experiment_of(base, base.manifest.dataset, assets)
    cand.manifest.experiment = experiment_of(cand, cand.manifest.dataset, assets)
    assert differences(base, cand) == ["retrieval"]
    reg = compare(base, cand, rows, MATRIX)
    assert not reg.confounded and [v for v, _, _ in reg.variables] == list(VARIABLES)
    assert ("retrieval", "v1+r1", "v2+r2") in reg.variables
    cand.manifest.experiment = cand.manifest.experiment.model_copy(update={"prompt_version": "cite-v2+zzz"})
    with pytest.raises(ValueError, match="differs from ds-base in 2 variables \\(retrieval, prompt\\)"):
        compare(base, cand, rows, MATRIX)
    reg = compare(base, cand, rows, MATRIX, allow_confounded=True)
    assert reg.confounded and reg.differs == ["retrieval", "prompt"]
    out = tmp_path / "regression.md"
    write_report(reg, out)
    text = out.read_text()
    assert "| retrieval | `v1+r1` | `v2+r2` | **differs** |" in text and "| dataset |" in text
    assert "**Confounded.** The two runs differ in retrieval, prompt" in text


def test_the_freeze_pins_the_assets_and_notices_when_they_move(assets, tmp_path):
    matrix, gates, safety, bench = assets
    frozen = Frozen.now("6+abc", "answer-quality-v1+abc", "jev-1", "first", matrix, gates, safety, bench)
    path = frozen.save(tmp_path / "frozen.yaml")
    loaded = Frozen.load(path)
    assert loaded == frozen and path.read_text().startswith("# The freeze")
    same = Frozen.now("6+abc", "answer-quality-v1+abc", "jev-1", "", matrix, gates, safety, bench)
    assert loaded.problems(same) == []
    matrix.write_text("matrix: [changed]\n")
    safety.write_text("policy: v1\n")
    moved = Frozen.now("7+def", "answer-quality-v1+abc", "jev-1", "", matrix, gates, safety, bench)
    problems = loaded.problems(moved)
    assert len(problems) == 3 and problems[0].startswith("dataset_version: frozen 6+abc, now 7+def")
    assert any(p.startswith("matrix_version") for p in problems) and any(
        p.startswith("safety_version") for p in problems
    )
    with pytest.raises(FileNotFoundError, match="dataset freeze"):
        Frozen.load(tmp_path / "missing.yaml")


def test_a_release_records_the_experiment_and_refuses_a_run_read_under_other_assets(
    two_runs,  # noqa: F811
    assets,
    tmp_path,
):
    good, bad = two_runs
    matrix, gates, safety, bench = assets
    from test_observe import lineage

    good.manifest.lineage = lineage()
    good.manifest.experiment = experiment_of(good, None, assets)
    frozen = Frozen.now("6+abc", "answer-quality-v1+abc", "fake-jev", "", matrix, gates, safety, bench)
    assert frozen.matches(good.manifest.experiment) == []  # a benchmark run names no dataset: not a mismatch
    releases = Releases()
    first, _ = promote(
        good, releases, [Gate(metric="answers.held", rule="min", value=0.0)], None, frozen=frozen
    )
    assert first.experiment == good.manifest.experiment and first.frozen == frozen
    releases.save(tmp_path / "releases.yaml")
    back = Releases.load(tmp_path / "releases.yaml")
    assert (
        back.live.experiment.retrieval_version == "v2+def"
        and back.live.frozen.grader_version == "answer-quality-v1+abc"
    )
    stale = frozen.model_copy(update={"grader_version": "answer-quality-v2+new"})
    with pytest.raises(ValueError, match="read under other assets than the freeze: grader_version"):
        promote(good, Releases(), [], None, frozen=stale)
    assert Release.model_fields["experiment"].default is None  # older records load without one


def test_the_repo_freeze_holds_and_every_run_carries_an_experiment():
    frozen = Frozen.load(ROOT / FROZEN)
    ds = load_dataset(ROOT / "eval/dataset")
    from infra_docs_rag.evaluation.dataset import dataset_version

    runs = [Run.load(d) for d in sorted((ROOT / RUNS_DIR).iterdir()) if (d / "manifest.json").exists()]
    assert runs and all(r.manifest.experiment is not None for r in runs)
    pin = Baseline.load(ROOT / "eval/baseline.yaml")
    baseline = next(r for r in runs if r.name == pin.run)
    current = Frozen.now(
        dataset_version(ds.records),
        grader_version(load_rubric(ROOT / "eval/rubric.yaml")),
        baseline.manifest.judge,
        matrix=ROOT / "eval/matrix.yaml",
        gates=ROOT / "eval/gates.yaml",
        safety=ROOT / "eval/safety.yaml",
        benchmark=tuple(ROOT / p for p in ("eval/retrieval.yaml", "eval/hybrid.yaml", "eval/grounding.yaml")),
    )
    assert frozen.problems(current) == []
    assert frozen.matches(baseline.manifest.experiment) == []
