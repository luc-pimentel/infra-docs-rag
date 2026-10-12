"""What the trace log says about the system: outcome rates, stage latencies, grounding, cost, and the
things an operator wants flagged. `infra-docs-rag traces --summary` prints it; `--report` writes it to
reports/observability.md with the last traces rendered in full."""

import statistics
from dataclasses import dataclass, field
from pathlib import Path

from ..generate.judge import BAND
from .trace import Trace, as_json, render

STAGES = ("embed", "dense", "lexical", "fuse", "rerank", "pack", "generate", "judge")

# What gets flagged. Each is a question's worth on a small log, so a flag is a prompt to look, not an alarm.
NO_PASSAGE_RATE = 0.30  # more than this share of questions found nothing: the floor or the corpus moved
ERROR_RATE = 0.05
P95_TOTAL_MS = 15_000  # the latency gate in eval/gates.yaml
CITED_SHARE_MEDIAN = 0.80  # the claim floor the verdict reads


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


@dataclass
class StageLatency:
    name: str
    n: int
    p50: float
    p95: float


@dataclass
class Summary:
    traces: int
    first_at: str
    last_at: str
    versions: dict[str, int]  # version name → traces
    indexes: dict[str, int]  # index version → traces (from the lineage)
    prompts: dict[str, int]
    outcomes: dict[str, int]
    stages: list[StageLatency]
    total_p50: float
    total_p95: float
    top_score_median: float | None
    cited_share_median: float | None
    claim_share_median: float | None
    uncited_answers: int  # answered traces with an uncited sentence that is not a gap sentence
    input_tokens_avg: float
    output_tokens_avg: float
    cost_total: float
    cost_per_1000: float
    degraded: int
    errors: int
    judged: int
    judge_grounded_mean: float | None
    judge_fabricated_mean: float | None
    judge_unsure: int
    flags: list[str] = field(default_factory=list)

    def rate(self, outcome: str) -> float:
        return self.outcomes.get(outcome, 0) / self.traces if self.traces else 0.0


def summarize(traces: list[Trace], current_index: str | None = None) -> Summary:
    n = len(traces)
    answered = [t for t in traces if t.answered]
    called = [t for t in traces if t.generator]
    judged = [t for t in traces if t.judgement is not None]
    outcomes: dict[str, int] = {}
    versions: dict[str, int] = {}
    indexes: dict[str, int] = {}
    prompts: dict[str, int] = {}
    for t in traces:
        outcomes[t.outcome] = outcomes.get(t.outcome, 0) + 1
        versions[t.version] = versions.get(t.version, 0) + 1
        if t.lineage:
            indexes[t.lineage.index_version] = indexes.get(t.lineage.index_version, 0) + 1
        key = t.lineage.prompt_version if t.lineage else t.prompt_version
        if key:
            prompts[key] = prompts.get(key, 0) + 1
    stages = []
    for name in STAGES:
        values = [t.stages[name] for t in traces if name in t.stages]
        if values:
            stages.append(StageLatency(name, len(values), percentile(values, 0.5), percentile(values, 0.95)))
    totals = [t.total_ms for t in traces]
    cost = sum(t.cost_usd for t in called)
    s = Summary(
        traces=n,
        first_at=traces[0].at if traces else "",
        last_at=traces[-1].at if traces else "",
        versions=versions,
        indexes=indexes,
        prompts=prompts,
        outcomes=outcomes,
        stages=stages,
        total_p50=percentile(totals, 0.5),
        total_p95=percentile(totals, 0.95),
        top_score_median=statistics.median([t.top_score for t in traces if t.top_score is not None])
        if any(t.top_score is not None for t in traces)
        else None,
        cited_share_median=statistics.median([t.cited_share for t in answered]) if answered else None,
        claim_share_median=statistics.median([t.claim_share for t in answered]) if answered else None,
        uncited_answers=sum(1 for t in answered if t.claim_share < 1.0),
        input_tokens_avg=statistics.mean([t.input_tokens for t in called]) if called else 0.0,
        output_tokens_avg=statistics.mean([t.output_tokens for t in called]) if called else 0.0,
        cost_total=cost,
        cost_per_1000=cost / len(called) * 1000 if called else 0.0,
        degraded=sum(1 for t in traces if t.degraded),
        errors=outcomes.get("error", 0),
        judged=len(judged),
        judge_grounded_mean=statistics.mean([t.judgement.grounded for t in judged]) if judged else None,
        judge_fabricated_mean=statistics.mean([t.judgement.fabricated for t in judged]) if judged else None,
        judge_unsure=sum(
            1
            for t in judged
            if any(BAND[0] <= p <= BAND[1] for p in (t.judgement.grounded, t.judgement.fabricated))
        ),
    )
    if n:
        if s.rate("no-passage") > NO_PASSAGE_RATE:
            s.flags.append(
                f"{s.rate('no-passage'):.0%} of questions found no passage over the {NO_PASSAGE_RATE:.0%} line: "
                "the questions, the floor or the corpus changed"
            )
        if s.rate("error") > ERROR_RATE:
            s.flags.append(f"{s.errors} model calls failed ({s.rate('error'):.0%})")
        if s.total_p95 > P95_TOTAL_MS:
            s.flags.append(f"p95 end to end is {s.total_p95:,.0f} ms, over the {P95_TOTAL_MS:,} ms gate")
        if s.degraded:
            s.flags.append(f"{s.degraded} question(s) answered with a stage degraded (see the traces)")
        if s.claim_share_median is not None and s.claim_share_median < CITED_SHARE_MEDIAN:
            s.flags.append(
                f"median claim share {s.claim_share_median:.0%} is under the {CITED_SHARE_MEDIAN:.0%} floor"
            )
        if current_index and indexes and any(v != current_index for v in indexes):
            others = [v for v in indexes if v != current_index]
            s.flags.append(
                f"{sum(indexes[v] for v in others)} trace(s) were made on index {', '.join(others)}, not the current {current_index}"
            )
    return s


def format_summary(s: Summary) -> str:
    if not s.traces:
        return "no traces yet; `answer` and the benchmarks write them to data/traces/traces.jsonl"
    lines = [
        f"{s.traces} traces, {s.first_at[:19].replace('T', ' ')} → {s.last_at[:19].replace('T', ' ')}",
        "versions: " + ", ".join(f"{v} ({n})" for v, n in s.versions.items()),
    ]
    if s.indexes:
        lines.append("indexes:  " + ", ".join(f"{v} ({n})" for v, n in s.indexes.items()))
    if s.prompts:
        lines.append("prompts:  " + ", ".join(f"{v} ({n})" for v, n in s.prompts.items()))
    lines.append(
        "outcomes: " + ", ".join(f"{o} {n} ({n / s.traces:.0%})" for o, n in sorted(s.outcomes.items()))
    )
    lines.append("")
    lines.append(f"{'stage':10} {'n':>4} {'p50 ms':>9} {'p95 ms':>9}")
    for st in s.stages:
        lines.append(f"{st.name:10} {st.n:4} {st.p50:9,.0f} {st.p95:9,.0f}")
    lines.append(f"{'total':10} {s.traces:4} {s.total_p50:9,.0f} {s.total_p95:9,.0f}")
    lines.append("")
    if s.top_score_median is not None:
        lines.append(
            f"retrieval: median top cosine {s.top_score_median:.2f}, no passage for {s.rate('no-passage'):.0%}"
        )
    if s.cited_share_median is not None:
        lines.append(
            f"grounding: median cited share {s.cited_share_median:.0%}, claim share {s.claim_share_median:.0%}; "
            f"{s.uncited_answers} answer(s) with an uncited claim"
        )
    if s.judged:
        lines.append(
            f"judge: {s.judged} read, grounded {s.judge_grounded_mean:.2f}, fabricated {s.judge_fabricated_mean:.2f} on average; "
            f"{s.judge_unsure} in the unsure band"
        )
    lines.append(
        f"cost: ${s.cost_total:.4f} over the log, ${s.cost_per_1000:.2f} per 1,000 questions at list price; "
        f"{s.input_tokens_avg:,.0f} in / {s.output_tokens_avg:,.0f} out per call"
    )
    lines.append(f"degraded: {s.degraded} · errors: {s.errors}")
    lines.append("")
    lines.append("flags: " + ("none" if not s.flags else ""))
    lines += [f"  ! {f}" for f in s.flags]
    return "\n".join(lines)


def write_report(traces: list[Trace], path: Path, examples: list[Trace], current_index: str | None) -> None:
    s = summarize(traces, current_index)
    out = [
        "# Observability: what one question leaves behind, and what the log says",
        "",
        "Every question through `answer` or a benchmark run writes one trace to `data/traces/traces.jsonl`:",
        "the question, the pipeline version and its lineage, every stage's time, what the first stage ranked",
        "with its scores, what the reranker made of it, what was packed, what the model wrote and cited, what",
        "the judge read. `infra-docs-rag traces --summary` reads the log back; this report is that summary",
        f"over {s.traces} traces, then {len(examples)} traces in full. The design these feed is in",
        "[`docs/design.md`](../docs/design.md).",
        "",
        "## The log, summarized",
        "",
        "```",
        format_summary(s),
        "```",
        "",
        "## Traces",
        "",
        "Each trace below is `traces --id <id>` as printed. The first-stage list is the fused cosine and BM25",
        "ranking before the cross-encoder, the context is what the model read after it, with where each",
        "passage stood before; the `was n.` says what the reranker moved.",
        "",
    ]
    for t in examples:
        out += [f"### {t.question}", "", "```", render(t), "```", ""]
    if examples:
        out += [
            "### The same trace as the log holds it",
            "",
            "One line of `data/traces/traces.jsonl`, expanded:",
            "",
            "```json",
            as_json(examples[0]),
            "```",
            "",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))
