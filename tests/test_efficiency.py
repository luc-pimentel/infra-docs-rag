from pathlib import Path

from test_evaluation import two_runs  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.evaluation.efficiency import (
    COST_RISE,
    QUALITY_RECORDS,
    Numbers,
    budget_status,
    budgets,
    dataset_pairs,
    numbers_of,
    release_pairs,
    replay_pairs,
    tradeoff,
)
from infra_docs_rag.evaluation.efficiency_report import write_report
from infra_docs_rag.evaluation.gates import Gate, check, load_gates
from infra_docs_rag.evaluation.metrics import Metric, metrics
from infra_docs_rag.evaluation.runs import Run
from infra_docs_rag.evaluation.slices import load_matrix
from infra_docs_rag.pipeline.releases import Release, Releases

ROOT = Path(__file__).parents[1]


def numbers(run="r", held=60, n=74, cost=0.04, p95=9000.0, cached=0, calls=70) -> Numbers:
    return Numbers(
        run=run,
        version="v",
        kind="dataset",
        n=n,
        held=held,
        held_share=held / n,
        calls=calls,
        cached=cached,
        tokens_in=1000,
        tokens_out=100,
        cost_total=cost * n,
        cost_per_request=cost,
        judge_usd=0.004,
        mean_ms=p95 * 0.7,
        p50_ms=p95 * 0.65,
        p95_ms=p95,
        generator="m",
        backend="b",
    )


def test_the_tradeoff_rule_reads_quality_against_cost_and_time():
    base = numbers()
    assert tradeoff(base, numbers(held=63)).verdict == "worth it"  # +3 records, same spend
    t = tradeoff(base, numbers(held=63, cost=0.06))  # +3 records at +50% cost
    assert (
        t.verdict == "not worth it" and "buys 3 records at +50% cost" in t.why and f"{COST_RISE:.0%}" in t.why
    )
    assert tradeoff(base, numbers(held=63, p95=12000)).verdict == "not worth it"  # +33% p95
    assert tradeoff(base, numbers(held=63, p95=11000)).verdict == "worth it"  # +22% p95, within
    assert tradeoff(base, numbers(held=61, cost=0.03)).verdict == "free"  # flat, cheaper
    assert tradeoff(base, numbers(held=60, p95=11000)).verdict == "not worth it"  # flat, slower
    assert tradeoff(base, numbers(held=57)).verdict == "a loss"
    assert tradeoff(base, base).verdict == "free" and "same spend and speed" in tradeoff(base, base).why
    assert tradeoff(base, numbers(held=61)).held_delta == 1 < QUALITY_RECORDS
    assert numbers(cached=35).cache_hit_share == 0.5 and base.cache_hit_share is None
    assert base.held_per_dollar == 60 / (0.04 * 74) and base.held_per_minute_p95 == 60 / (9000 / 60000)


def test_numbers_pairs_and_budgets_read_the_saved_runs(two_runs, tmp_path):  # noqa: F811
    good, bad = two_runs
    x = numbers_of(good)
    assert x.run == "base-fake" and x.kind == "benchmark" and x.n == 5 and x.calls >= 1 and x.cached == 0
    assert x.cost_per_request == metrics(good)["cost.usd_per_request"].value and x.p95_ms >= x.p50_ms
    m = metrics(good)
    assert m["latency.end_to_end_ms_p95"].value is not None and m["cost.tokens_total"].value > 0
    assert m["cache.hit_share"].value is None and m["cache.hit_share"].shown == "–"
    # a cached call is counted
    good.cases[0].generated.cached = True
    assert metrics(good)["cache.hit_share"].value == 1 / x.calls
    replay = good.model_copy(deep=True)
    replay.manifest.name = "base-fake-replay"
    runs = {"base-fake": good, "cand-fake": bad, "base-fake-replay": replay}
    releases = Releases(
        current="r2",
        releases=[
            Release(name="r1", created_at="t", run="base-fake", manifest=good.manifest.lineage or _lineage()),
            Release(name="r2", created_at="t", run="cand-fake", manifest=good.manifest.lineage or _lineage()),
            Release(name="r3", created_at="t", run="cand-fake", manifest=good.manifest.lineage or _lineage()),
        ],
    )
    pairs = release_pairs(runs, releases)
    assert [(a.name, b.name) for a, b in pairs] == [("base-fake", "cand-fake")]  # r3 repeats r2's run
    assert len(replay_pairs(runs)) == 1 and replay_pairs(runs)[0][1] is runs["base-fake-replay"]
    good.manifest.dataset = bad.manifest.dataset = "5+abc"
    assert [(a.name, b.name) for a, b in dataset_pairs(runs, "base-fake")] == [("base-fake", "cand-fake")]
    assert dataset_pairs(runs, "nope") == []
    gates = [
        Gate(metric="cost.usd_per_request", rule="max", value=0.06, why="budget"),
        Gate(
            metric="cache.hit_share",
            rule="min",
            value=0.9,
            skip_unmeasured=True,
            critical=False,
            why="replay",
        ),
        Gate(metric="answers.held", rule="min", value=0.5, why="not a budget"),
    ]
    b = budgets(gates, [])
    assert [x.gate.metric for x in b] == ["cost.usd_per_request", "cache.hit_share"] and b[
        0
    ].where == "release"
    assert budget_status(b[0], bad)[0] is True and budget_status(b[1], bad) == (None, "–")
    assert "skipped when not measured" in b[1].label
    # the skip: a run that made no cached call passes the cache gate instead of failing it
    unmeasured = Metric("cache.hit_share", None, "–", "higher", "")
    assert check(gates[1], None, unmeasured).passed and "skipped" in check(gates[1], None, unmeasured).detail
    assert not check(Gate(metric="cache.hit_share", rule="min", value=0.9), None, unmeasured).passed
    out = tmp_path / "efficiency.md"
    write_report(
        [numbers_of(r) for r in runs.values()],
        [("release", tradeoff(numbers_of(good), numbers_of(bad)))],
        b,
        runs,
        "cand-fake",
        "base-fake",
        out,
    )
    text = out.read_text()
    for heading in ("## Summary", "## 1. Tradeoffs", "## 2. The budgets", "## 3. Reading"):
        assert heading in text
    assert "◂ release" in text and "◂ baseline" in text and "| release | `cost.usd_per_request` |" in text


def _lineage():
    from test_observe import lineage

    return lineage()


def test_the_repo_budgets_are_gates_and_the_release_is_within_them():
    gates = load_gates(ROOT / "eval/gates.yaml")
    slice_gates = load_matrix(ROOT / "eval/matrix.yaml").gates
    b = budgets(gates, slice_gates)
    metrics_named = {x.gate.metric for x in b}
    assert {"cost.usd_per_request", "latency.end_to_end_ms_p95", "cache.hit_share"} <= metrics_named
    release = Run.load(ROOT / "eval/runs/v3-budget-claude-code")
    replay = Run.load(ROOT / "eval/runs/v3-budget-claude-code-replay")
    for x in b:
        passed, _ = budget_status(x, release)
        assert passed is not False, x.gate.metric
    cache = next(x for x in b if x.gate.metric == "cache.hit_share")
    assert budget_status(cache, release) == (None, "–") and budget_status(cache, replay)[0] is True
