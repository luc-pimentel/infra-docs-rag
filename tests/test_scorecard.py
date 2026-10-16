from pathlib import Path

import yaml
from test_regression import BASE_SCRIPT, CAND_SCRIPT, MATRIX, make_run
from test_regression import records as dataset_records  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.evaluation.experiment import Frozen
from infra_docs_rag.evaluation.gates import Gate
from infra_docs_rag.evaluation.regression import Baseline
from infra_docs_rag.evaluation.scorecard import (
    CORE,
    SLICES,
    Inputs,
    Scorecard,
    build,
    shown,
    sparkline,
)
from infra_docs_rag.evaluation.scorecard_report import write_report
from infra_docs_rag.pipeline.releases import Release, Releases

ROOT = Path(__file__).parents[1]


def test_sparklines_scale_to_the_series_and_values_print_in_their_units():
    assert sparkline([0.0, 0.5, 1.0]) == "▁▅█" and sparkline([1.0, 1.0]) == "▄▄"
    assert sparkline([None, 0.2, 0.8]) == " ▁█" and sparkline([None]) == "" and sparkline([]) == ""
    assert (
        shown(0.876, "pct") == "88%"
        and shown(0.0412, "usd") == "$0.0412"
        and shown(9088.6, "ms") == "9,089 ms"
    )
    assert shown(None, "pct") == "–" and shown(1.234, "num") == "1.23"


def test_the_scorecard_reads_every_component_and_writes_the_page(dataset_records, fake_generator, tmp_path):  # noqa: F811
    rows, index, embedder = dataset_records
    base = make_run(rows, index, embedder, fake_generator, BASE_SCRIPT, "ds-base")
    cand = make_run(
        rows, index, embedder, fake_generator, CAND_SCRIPT, "ds-cand", "v2", retrieval_version="v2+r2"
    )
    from test_observe import lineage

    bench = make_run(rows, index, embedder, fake_generator, BASE_SCRIPT, "bench-v1")
    bench.manifest.dataset = None
    releases = Releases(
        current="r1",
        releases=[
            Release(name="r1", created_at="2026-10-12T00:00:00+00:00", run="bench-v1", manifest=lineage())
        ],
    )
    pin = Baseline.of(base, "the first")
    frozen = Frozen.now(
        "x", "g+1", "fake-jev", "", tmp_path / "m.yaml", tmp_path / "g.yaml", tmp_path / "s.yaml", ()
    )
    x = Inputs(
        runs={"ds-base": base, "ds-cand": cand, "bench-v1": bench},
        releases=releases,
        baseline=pin,
        frozen=frozen,
        frozen_now=frozen,
        records=rows,
        matrix=MATRIX,
        gates=[Gate(metric="cost.usd_per_request", rule="max", value=0.06, why="budget")],
        safety_policy=None,
        windows=[],
        candidate="ds-cand",
    )
    card = build(x)
    by = {s.component: s for s in card.statuses}
    assert by["release"].state == "ok" and "r1" in by["release"].detail
    assert by["baseline"].state == "ok" and by["freeze"].state == "ok"
    assert (
        by["quality gates"].state == "fail" and "block" in by["quality gates"].detail
    )  # bread answered anyway
    assert by["safety"].state == "none" and by["budgets"].state == "ok" and by["online"].state == "none"
    assert card.worst == "fail"
    names = {s.name: s for s in card.series}
    assert [p.run for p in names["releases"].points] == ["bench-v1"]
    assert [p.run for p in names["dataset"].points] == ["ds-base", "ds-cand"]
    held = names["dataset"].trend("answers.held")
    assert held[0] < held[1] and set(names["dataset"].points[0].values) == {n for n, _, _ in CORE}
    assert [s.slice for s in card.slices] == list(SLICES)
    no_answer = next(s for s in card.slices if s.slice == "category=no-answer")
    assert no_answer.values == [1.0, 0.0]
    # a stale pin and a moved freeze go red; a missing candidate goes grey
    x.frozen_now = frozen.model_copy(update={"grader_version": "g+2"})
    x.baseline = pin.model_copy(update={"retrieval_version": "other"})
    x.candidate = None
    card = build(x)
    by = {s.component: s for s in card.statuses}
    assert by["freeze"].state == "fail" and "grader_version" in by["freeze"].detail
    assert by["baseline"].state == "fail" and by["quality gates"].state == "none"
    out = tmp_path / "scorecard.md"
    write_report(card, out)
    text = out.read_text()
    for heading in (
        "## Status: ❌ fail",
        "## Trends over the releases",
        "## Trends over the dataset runs",
        "**The critical slices**",
        "## The online windows",
        "## Reading the scorecard",
    ):
        assert heading in text
    assert "| held |" in text and "`▁█`" in text and "No window sampled yet." in text
    path = card.save(tmp_path / "scorecard.json")
    back = Scorecard.model_validate_json(path.read_text())
    assert back.worst == "fail" and len(back.series) == 2


def test_the_workflows_run_the_same_saved_run_checks():
    checks = yaml.safe_load((ROOT / ".github/workflows/checks.yml").read_text())
    nightly = yaml.safe_load((ROOT / ".github/workflows/nightly.yml").read_text())
    assert "schedule" in nightly[True] and "workflow_dispatch" in nightly[True]  # `on` parses as True
    steps = {s.get("name"): s for s in nightly["jobs"]["nightly"]["steps"]}
    assert "the scorecard" in steps and "open or update the issue" in steps
    assert steps["open or update the issue"]["if"] == "failure()"
    run_text = " ".join(s.get("run", "") for s in checks["jobs"]["checks"]["steps"])
    nightly_text = " ".join(s.get("run", "") for s in nightly["jobs"]["nightly"]["steps"])
    for command in (
        "dataset baseline check",
        "dataset freeze --check",
        "dataset compare",
        "online report",
        "evaluate-efficiency",
        "safety grade",
        "scorecard",
    ):
        assert command in run_text, command
        assert command in nightly_text, command
    assert (ROOT / "docs/evals.md").read_text().count("| **") >= 6  # the trigger table
