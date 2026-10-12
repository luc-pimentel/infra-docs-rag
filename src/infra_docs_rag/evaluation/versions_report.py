"""reports/versions.md: every saved version side by side, the gates between each release and the next,
and the criteria under which a release is rolled back. `evaluate-pipeline --runs a b c` writes it."""

from pathlib import Path

from ..pipeline.releases import Releases
from .gates import Gate, compare
from .metrics import metrics
from .runs import Run

ROWS = (
    ("retrieval.right_retrieved", "right passage retrieved"),
    ("retrieval.hit_at_1", "right passage first"),
    ("retrieval.mrr", "MRR"),
    ("retrieval.retrieve_ms_median", "retrieval, median"),
    ("grounding.right_cited", "right passage cited"),
    ("grounding.claim_cited_median", "claims cited, median share"),
    ("grounding.fully_cited", "answers with no uncited sentence"),
    ("grounding.unverified_citations", "typed citations that did not check out"),
    ("grounding.judge_grounded", "judge: grounded"),
    ("grounding.judge_fabricated", "judge: fabricated"),
    ("grounding.flagged", "flagged for a person"),
    ("abstention.out_of_scope", "out-of-scope abstained"),
    ("abstention.in_scope", "false abstentions"),
    ("abstention.adversarial_held", "adversarial held"),
    ("answers.in_scope_held", "in-scope answered right"),
    ("answers.held", "expectations held"),
    ("latency.generate_ms_median", "generation, median"),
    ("latency.end_to_end_ms_median", "end to end, median"),
    ("latency.end_to_end_ms_p90", "end to end, p90"),
    ("cost.tokens_in_per_call", "input tokens per call"),
    ("cost.tokens_out_per_call", "output tokens per call"),
    ("cost.usd_per_1000", "cost per 1,000 questions"),
)


def write_report(runs: list[Run], gates: list[Gate], releases: Releases, path: Path) -> None:
    cols = [f"**{r.manifest.version}**<br>`{r.name}`" for r in runs]
    numbers = [{k: v.shown for k, v in metrics(r).items()} for r in runs]
    out = [
        "# Versions: the pipeline measured end to end, release by release",
        "",
        f"{len(runs)} versions of the pipeline over the same {runs[0].manifest.questions} questions, each a saved run under",
        "`eval/runs/` with the lineage of what answered. The columns are in release order. Every number comes",
        "from the run's cases; the definitions are in [`eval/SPEC.md`](../eval/SPEC.md). Through `claude -p`, the",
        "input token counts include the CLI's own context, so the cost column overstates the API path by the same",
        "factor for every version; the differences between versions are the passages.",
        "",
        "## What each version is",
        "",
    ]
    for r in runs:
        lin = r.manifest.lineage
        out.append(
            f"- **{r.manifest.version}** (`{r.name}`, {r.manifest.created_at[:10]}, commit {r.manifest.git_sha}): "
            f"{r.manifest.description or r.manifest.retrieval.label}. "
            + (
                f"Index `{lin.index_version}`, prompt `{lin.prompt_version}`, retrieval `{lin.retrieval_version}`."
                if lin
                else ""
            )
        )
    out += ["", "## Side by side", "", "| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(runs)]
    for key, label in ROWS:
        out.append(f"| {label} | " + " | ".join(n.get(key, "–") for n in numbers) + " |")
    out += ["", "## The gates between each version and the next", ""]
    for earlier, later in zip(runs, runs[1:], strict=False):
        c = compare(earlier, later, gates)
        out += [
            f"### {earlier.manifest.version} → {later.manifest.version}: **{c.verdict}**",
            "",
            "| Gate | Rule | Before → after | Result |",
            "|---|---|---|---|",
        ]
        for res in c.results:
            word = "pass" if res.passed else ("**FAIL**" if res.gate.critical else "warn")
            out.append(f"| `{res.gate.metric}` | {res.gate.label} | {res.detail} | {word} |")
        improved = [x for x in c.changed if x.better is True]
        regressed = [x for x in c.changed if x.better is False]
        out += [
            "",
            f"{len(c.changed)} question(s) changed: {len(improved)} improved, {len(regressed)} regressed"
            + (
                " (" + ", ".join(f"`{x.id}`: {'; '.join(x.what)}" for x in regressed) + ")"
                if regressed
                else ""
            )
            + ".",
            "",
        ]
    out += [
        "## Rollback decision criteria",
        "",
        "A release is the current one until one of these holds, in which case `pipeline rollback` makes the",
        "previous release current again and points the index at its version (`docs/pipeline.md`):",
        "",
        "1. **A critical gate fails on a re-run.** The gates in [`eval/gates.yaml`](../eval/gates.yaml) are the",
        "   contract: out-of-scope abstention may not drop; false abstentions, judge-fabricated answers and the",
        "   share of answers citing a right passage may not move by more than one question's worth (2 %); the",
        "   median claim share stays over 90 %; at least 8 of 10 adversarial expectations hold; a right passage",
        "   is among the ones read for no fewer labelled questions than before. A release never clears them on",
        "   the way in and fails them later unless the corpus, the model or the prompt changed underneath it,",
        "   and the lineage says which.",
        "2. **The canary fails.** Five questions with a known outcome and source (`eval/canary.yaml`) against the",
        "   released pipeline; one failure that is not the model's judgement call (see",
        "   [`reports/observability.md`](observability.md)) is a rollback, then a look.",
        "3. **The trace log flags a rate.** No-passage share over 30 %, error rate over 5 %, p95 over 15 s, any",
        "   degraded answer, median claim share under 80 %, or traces made on an index that is not the",
        "   released one (`traces --summary`). A flag is a prompt to look; two flags on the same day are a",
        "   rollback.",
        "4. **The release's index is not on disk.** `pipeline health` warns when the release was measured on",
        "   an index version other than the current one. The service keeps answering; the mismatch is",
        "   resolved by re-releasing under the new index or pointing back, never by leaving it.",
        "",
        "What is not a rollback: a warning gate (MRR, cost, median latency inside their tolerances), one",
        "flipped answer on a set of ten, or a judge reading in the unsure band. Those are the next",
        "calibration's questions.",
        "",
    ]
    live = releases.live
    if live is not None:
        out += [
            f"Current release: **{live.name}**, {live.manifest.version}, measured by `{live.run}` "
            f"({live.gates}). The record is [`eval/releases.yaml`](../eval/releases.yaml).",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
