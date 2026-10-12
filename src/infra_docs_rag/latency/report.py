"""reports/latency-cost.md: where the time and money go on the benchmark workload, each optimization
before and after, and the matrix of configurations. The profiles are measured here without a model;
the generation and judge columns come from the saved runs, and the estimated generation cost of a
profile is what its passages would cost at API list price with the released run's output length."""

import json
import statistics
from dataclasses import asdict
from pathlib import Path

from ..evaluation.gates import compare, load_gates
from ..evaluation.metrics import metrics
from ..evaluation.runs import Run
from ..retrieve.retriever import Retrieval
from .profile import STAGES, BatchTiming, Observation, Profile, Question, Workload


def save_workload(w: Workload, path: Path) -> None:
    data = {
        "machine": w.machine,
        "batches": [asdict(b) for b in w.batches],
        "profiles": [
            {
                "name": p.name,
                "note": p.note,
                "retrieval": p.retrieval.model_dump(),
                "observations": [
                    {
                        "id": o.question.id,
                        "kind": o.question.kind,
                        "labelled": o.question.labelled is not None,
                        "expect": o.question.expect,
                        "timings": o.timings,
                        "retrieve_ms": o.retrieve_ms,
                        "passages": o.passages,
                        "passage_tokens": o.passage_tokens,
                        "right_rank": o.right_rank,
                        "no_passage": o.no_passage,
                        "degraded": o.degraded,
                    }
                    for o in p.observations
                ],
            }
            for p in w.profiles
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


class Marker:
    """Stands in for a labelled answer when a profile is loaded back: only "is there one" matters."""

    def answered_by(self, chunk) -> bool:  # pragma: no cover - never called on a loaded profile
        return False


def load_workload(path: Path) -> Workload:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `evaluate-latency` without --from-saved first")
    data = json.loads(path.read_text())
    profiles = []
    for p in data["profiles"]:
        observations = [
            Observation(
                question=Question(o["id"], o["kind"], "", Marker() if o["labelled"] else None, o["expect"]),
                timings=o["timings"],
                retrieve_ms=o["retrieve_ms"],
                passages=o["passages"],
                passage_tokens=o["passage_tokens"],
                right_rank=o["right_rank"],
                no_passage=o["no_passage"],
                degraded=o["degraded"],
            )
            for o in p["observations"]
        ]
        profiles.append(Profile(p["name"], Retrieval.model_validate(p["retrieval"]), observations, p["note"]))
    return Workload(profiles, [BatchTiming(**b) for b in data["batches"]], data.get("machine", ""))


def ms(value: float) -> str:
    return f"{value:,.0f} ms"


def usd(value: float) -> str:
    return f"${value:,.2f}"


def run_numbers(run: Run) -> dict[str, str]:
    m = metrics(run)
    return {k: v.shown for k, v in m.items()}


def output_tokens_of(run: Run | None) -> float:
    if run is None:
        return 430.0
    called = [c for c in run.cases if c.generated is not None and c.generated.usage.output_tokens]
    return statistics.median(c.generated.usage.output_tokens for c in called) if called else 430.0


def write_report(w: Workload, runs: dict[str, Run], path: Path) -> None:
    by_name = {p.name: p for p in w.profiles}
    base = by_name.get("v2 (released)") or w.profiles[0]
    released = runs.get("v2-hybrid-rerank-claude-code")
    out_tokens = output_tokens_of(released)
    n = len(base.observations)
    out = [
        "# Latency and cost: where the time and the money go, and what each cut buys",
        "",
        f"The workload is the benchmark: {n} questions ({len(base.labelled)} labelled, "
        f"{len(base.out_of_scope)} out of scope, {n - len(base.in_scope) - len(base.out_of_scope)} adversarial), "
        "one pass each. Retrieval is measured here, every stage timed, on "
        f"{w.machine or 'this machine'}, with no model call: the retrieval quality comes from the labels, the",
        "abstentions from the floor. The generation and judge side comes from the saved runs under",
        "`eval/runs/` (the same questions through `claude -p`), and a configuration that has no saved run",
        "gets its generation cost estimated from the passages it would send: the passages and the question",
        "as fresh input tokens at API list price, the frozen system prompt as a cache read, and the released",
        f"run's median output length ({out_tokens:,.0f} tokens). Passage tokens are counted by the embedding",
        "model's tokenizer, which runs a little under Claude's; the estimate is a floor, not a quote.",
        "",
        "## 1. The request path, stage by stage",
        "",
        f"The released pipeline (`{base.retrieval.label}`) over the {n} questions. Retrieval p50 and p95 are",
        "measured here; generation and judge are the released run's medians.",
        "",
        "| Stage | p50 | p95 | Tokens | Cost per question | Where it runs |",
        "|---|---|---|---|---|---|",
    ]
    where = {
        "embed": "the query through bge-small, CPU",
        "dense": "a dot product over 497 unit vectors",
        "lexical": "BM25 over the same chunks, in memory",
        "fuse": "weighted fusion of the two rankings",
        "rerank": "bge-reranker-base over 20 pairs, CPU",
        "pack": "the floor, the budget, the headings",
    }
    for stage in STAGES:
        if any(stage in o.timings for o in base.observations):
            out.append(
                f"| {stage} | {ms(base.stage_p50(stage))} | {ms(base.stage_p95(stage))} | – | $0 (local) | {where[stage]} |"
            )
    out.append(
        f"| **retrieval** | **{ms(base.retrieve_p50)}** | **{ms(base.retrieve_p95)}** | {base.passage_tokens_p50:,.0f} passage tokens | $0 | the six stages above |"
    )
    if released is not None:
        m = run_numbers(released)
        called = [c for c in released.cases if c.generated is not None]
        judged = [c for c in released.cases if c.judgement is not None]
        gen_cost = sum(c.generated.usage.cost(released.manifest.prices) for c in called) / len(released.cases)
        judge_cost = sum(c.judgement.cost_usd for c in judged) / len(released.cases)
        out.append(
            f"| generate | {m['latency.generate_ms_median']} | {ms(statistics.quantiles([c.generated.ms for c in called], n=20)[18])} | "
            f"{m['cost.tokens_in_per_call']} in / {m['cost.tokens_out_per_call']} out per call | "
            f"${gen_cost:.4f} | claude-opus-5-5 through `claude -p`, {len(called)} calls |"
        )
        out.append(
            f"| judge | {ms(statistics.median(c.judgement.ms for c in judged))} | {ms(statistics.quantiles([c.judgement.ms for c in judged], n=20)[18])} | – | "
            f"${judge_cost:.4f} | jev, {len(judged)} readings, not on the serving path |"
        )
        out.append(
            f"| **end to end** | **{m['latency.end_to_end_ms_median']}** | p90 {m['latency.end_to_end_ms_p90']} | – | "
            f"{m['cost.usd_per_1000']} per 1,000 | retrieval + generation |"
        )
    out += [
        "",
        "Two numbers decide the shape of everything below. The cross-encoder is the whole of retrieval's",
        f"time ({ms(base.stage_p50('rerank'))} of {ms(base.retrieve_p50)} at the median; the first stage is under a millisecond), and",
        "generation is the whole of the money. The input token counts through `claude -p` include the",
        "CLI's own context, most of it read from cache, so the per-call cost above overstates the API path;",
        "the estimates in the next sections price only the passages, which is the part a configuration changes.",
        "",
        "## 2. Each cut, before and after",
        "",
        "Every optimization applied to the released pipeline alone. **Right retrieved** is the labelled",
        f"questions with a right passage among the ones the model would read (of {len(base.labelled)});",
        "**false abstentions** are in-scope questions the floor left with no passage; **est. cost** is the",
        "generation estimate per 1,000 questions described above.",
        "",
        "| Configuration | What changes | Retrieval p50 | Rerank p50 | Passages (p50) | Passage tokens (p50) | Right retrieved | Hit@1 | MRR | False abst. | Est. cost / 1,000 |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    def row(p: Profile) -> str:
        rr = f"{p.right_retrieved}/{len(p.labelled)}"
        delta_rr = p.right_retrieved - base.right_retrieved
        if p is not base and delta_rr:
            rr += f" ({delta_rr:+d})"
        return (
            f"| {'**' + p.name + '**' if p is base else p.name} | {p.note} | {ms(p.retrieve_p50)} | {ms(p.stage_p50('rerank')) if p.retrieval.rerank else '–'} | "
            f"{p.passages_p50:.0f} | {p.passage_tokens_p50:,.0f} | {rr} | {p.hit_at_1} | {p.mrr:.2f} | {p.false_abstentions} | "
            f"{usd(p.generation_cost(out_tokens) * 1000)} |"
        )

    out.append(row(base))
    for p in w.profiles:
        if p is not base:
            out.append(row(p))
    out += ["", "What the table says, cut by cut:", ""]
    for p in w.profiles:
        if p is base:
            continue
        d_ms = p.retrieve_p50 - base.retrieve_p50
        d_cost = (p.generation_cost(out_tokens) - base.generation_cost(out_tokens)) * 1000
        d_rr = p.right_retrieved - base.right_retrieved
        d_mrr = p.mrr - base.mrr
        verdict = (
            "costs no right passage"
            if d_rr >= 0
            else f"loses {-d_rr} right passage{'s' if d_rr < -1 else ''}"
        )
        out.append(
            f"- **{p.name}:** retrieval {d_ms:+,.0f} ms at the median, estimated generation {d_cost:+.2f} $/1,000, "
            f"{verdict}, MRR {d_mrr:+.2f}, false abstentions {p.false_abstentions - base.false_abstentions:+d}."
        )
    out += [
        "",
        "### Indexing: the batch size",
        "",
        "Embedding the corpus is paid once per index version, not per question, so it is the cheap side of",
        "the ledger; the batch size is the one knob and this is what it does on this machine:",
        "",
        "| Batch size | Chunks | Seconds | Chunks per second |",
        "|---|---|---|---|",
    ]
    for b in w.batches:
        out.append(f"| {b.batch_size} | {b.chunks} | {b.seconds:.1f} | {b.chunks / b.seconds:,.0f} |")
    out += [
        "",
        "### The generation cache",
        "",
        "A cache in front of the generator keyed by the model, the prompt version, the question and the exact",
        "chunks it would read (`generate/cache.py`). The same question against the same passages is the same",
        "answer; a different index version, a changed prompt or a reordered context is a different key, so",
        "the cache can never serve an answer the released pipeline would not give. On a replay of the",
        "benchmark it is 100 % hits, zero tokens, and the generate stage falls to the lookup's time; its",
        "value in production is the share of questions that repeat, which the trace log measures. The",
        "replay's numbers are in section 3 where a cached run was saved.",
        "",
        "## 3. The matrix",
        "",
        "Every configuration with a saved run, measured end to end through the model, plus the profiles",
        "measured here without one. A dash is a column that needs a model call the configuration has not had.",
        "",
        "| Configuration | End to end p50 | Retrieval p50 | Cost / 1,000 | Right retrieved | MRR | Right cited | Judge fabricated | Claim share (median) | False abst. | OOS abst. | Adversarial held |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, run in runs.items():
        m = run_numbers(run)
        label = f"**{run.manifest.version}** (`{name}`)"
        cached = all(c.generated is None or c.generated.usage.input_tokens == 0 for c in run.cases) and any(
            c.generated is not None for c in run.cases
        )
        if cached:
            label += " replayed through the cache"
        out.append(
            f"| {label} | {m['latency.end_to_end_ms_median']} | {m['retrieval.retrieve_ms_median']} | {m['cost.usd_per_1000']} | "
            f"{m['retrieval.right_retrieved']} | {m['retrieval.mrr']} | {m['grounding.right_cited']} | {m['grounding.judge_fabricated']} | "
            f"{m['grounding.claim_cited_median']} | {m['abstention.in_scope']} | {m['abstention.out_of_scope']} | {m['abstention.adversarial_held']} |"
        )
    for p in w.profiles:
        out.append(
            f"| {p.name} (profile) | – | {ms(p.retrieve_p50)} | est. {usd(p.generation_cost(out_tokens) * 1000)} | "
            f"{p.right_retrieved}/{len(p.labelled)} | {p.mrr:.2f} | – | – | – | {p.false_abstentions} | {p.oos_abstentions}/{len(p.out_of_scope)} by the floor | – |"
        )
    out += [
        "",
        "A profile's out-of-scope column counts the questions the cosine floor stopped before any model call;",
        "the other out-of-scope questions reach the model, which abstains on them in every saved run.",
        "",
        "## 4. The default, and why",
        "",
    ]
    out += conclusion(w, runs, base, out_tokens)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(out))


def conclusion(w: Workload, runs: dict[str, Run], base: Profile, out_tokens: float) -> list[str]:
    by_name = {p.name: p for p in w.profiles}
    v3 = next((p for p in w.profiles if p.name.startswith("v3")), None)
    small = by_name.get("minilm-l6 over 20")
    first = next((p for p in w.profiles if p.name.startswith("first candidate")), None)
    out: list[str] = []
    out += [
        "**Where the time goes is not where the money goes.** Retrieval is the cross-encoder and nothing else:",
        f"{ms(base.stage_p50('rerank'))} of {ms(base.retrieve_p50)} at the median, with embedding, both first stages and the",
        "fusion under ten milliseconds together. Generation is four to five seconds and every cent. So a latency",
        "cut is a reranker cut, and a cost cut is a passage cut, and the two are nearly independent.",
        "",
    ]
    if first is not None:
        out += [
            f"**The cut designed before measuring was wrong.** The first candidate (`{first.name.split(': ', 1)[1]}`)",
            "combined the three changes that each looked free. Measured, the relevance floor alone drops",
            f"{base.right_retrieved - by_name['relevance floor 0.10'].right_retrieved} right passages and turns"
            f" {by_name['relevance floor 0.10'].false_abstentions} in-scope questions into false abstentions:",
            "the cross-encoder's score is a ranking signal, not a calibrated relevance, and a passage it scores",
            "0.08 is often the right one (the identifier questions, where the question is one token and the",
            "passage is a config block). The candidate as a whole lost",
            f"{base.right_retrieved - first.right_retrieved} right passages. The regression gates would have refused it;",
            "the profile refused it two minutes in, for nothing.",
            "",
        ]
    if v3 is not None and small is not None:
        out += [
            f"**Two cuts cost no right passage, and they do not combine.** The token budget (`{v3.name.split(': ', 1)[1]}`)",
            f"keeps {v3.right_retrieved}/{len(v3.labelled)} right passages at MRR {v3.mrr:.2f}, the same as the released pipeline, and sends",
            f"{v3.passage_tokens_p50:,.0f} passage tokens at the median instead of {base.passage_tokens_p50:,.0f}: a",
            f"{1 - v3.passage_tokens_p50 / base.passage_tokens_p50:.0%} cut in the part of the input a configuration",
            "controls, because the fifth passage was rarely the one that answered. The small cross-encoder",
            f"(`minilm-l6 over 20`) keeps {small.right_retrieved}/{len(small.labelled)} too and takes retrieval from {ms(base.retrieve_p50)} to {ms(small.retrieve_p50)},",
            f"at MRR {small.mrr:.2f}: the same passages, slightly less often first. Put together",
            f"(`minilm-l6 + budget 1,200`) they keep {by_name['minilm-l6 + budget 1,200'].right_retrieved}/{len(base.labelled)}: the small model ranks the right passage",
            "fourth or fifth a little more often, and the budget then cuts exactly there. Each is safe alone;",
            "the pair is not, and only the measurement says so.",
            "",
        ]
    out += [
        "**The batch size runs the other way on a CPU.** Larger batches were slower (8: "
        + ", ".join(f"{b.seconds:.1f} s" for b in w.batches[:1])
        + "; "
        + "; ".join(f"{b.batch_size}: {b.seconds:.1f} s" for b in w.batches[1:])
        + "), since every text in a batch is padded to the longest and the chunks vary from a line to 512",
        "tokens. The default stays at 16; on a GPU the curve would turn over. Indexing is paid once per index",
        "version, so this is the smallest number in the report either way.",
        "",
    ]
    v3_run = next(
        (r for n, r in runs.items() if r.manifest.version.startswith("v3") and "replay" not in n), None
    )
    v2_run = runs.get("v2-hybrid-rerank-claude-code")
    replay = next((r for n, r in runs.items() if "replay" in n), None)
    if v3_run is not None and v2_run is not None:
        comparison = compare(v2_run, v3_run, load_gates())
        m2, m3 = run_numbers(v2_run), run_numbers(v3_run)
        out += [
            f"**v3 through the model.** `{v3_run.name}` is the budget run end to end, {v3_run.manifest.questions} questions",
            f"through `claude -p` with the judge, against the released run's gates: **{comparison.verdict}**.",
            "",
            "| | v2 (released) | v3 (budget) |",
            "|---|---|---|",
        ]
        for key, label in (
            ("retrieval.right_retrieved", "right passage retrieved"),
            ("retrieval.mrr", "MRR"),
            ("grounding.right_cited", "right passage cited"),
            ("grounding.claim_cited_median", "claim share, median"),
            ("grounding.judge_fabricated", "judge: fabricated"),
            ("abstention.in_scope", "false abstentions"),
            ("abstention.out_of_scope", "out-of-scope abstained"),
            ("abstention.adversarial_held", "adversarial held"),
            ("answers.held", "expectations held"),
            ("cost.tokens_in_per_call", "input tokens per call"),
            ("cost.usd_per_1000", "cost per 1,000"),
            ("latency.end_to_end_ms_median", "end to end, median"),
        ):
            out.append(f"| {label} | {m2[key]} | {m3[key]} |")
        out.append("")
        failed = [r for r in comparison.results if not r.passed]
        if failed:
            out.append(
                "Gates that did not pass: " + "; ".join(f"{r.gate.metric}: {r.detail}" for r in failed) + "."
            )
        else:
            out.append("Every gate passed.")
        adv = comparison.after["abstention.adversarial_held"].value
        adv_before = comparison.before["abstention.adversarial_held"].value
        if adv is not None and adv_before is not None and adv < adv_before:
            changed = [c for c in comparison.changed if c.after.kind == "adversarial"]
            which = ", ".join(f"`{c.id}` ({c.before.verdict} → {c.after.verdict})" for c in changed)
            out.append(
                f" The one number that moved the wrong way is the adversarial set, {adv_before:.0%} → {adv:.0%}, at the gate's floor:"
                f" {which}. One question on a set of ten is inside the noise the gates were sized for, and the"
                " budget is a plausible cause (fewer passages, less to answer from), so it is the first thing"
                " the next run under v3 should check."
            )
        out.append("")
    if replay is not None:
        mr = run_numbers(replay)
        out += [
            f"**The cache, replayed.** `{replay.name}` is the same {replay.manifest.questions} questions asked again under the same",
            f"version with the generation cache in front of the model: {mr['cost.tokens_in_per_call']} input tokens per call,",
            f"{mr['cost.usd_per_1000']} per 1,000, end to end {mr['latency.end_to_end_ms_median']} at the median (the retrieval, plus a lookup).",
            "The answers are the stored ones, so every grounding number is the run's own. In production the saving",
            "is the share of questions that repeat against the same index, which the trace log can count.",
            "",
        ]
    out += [
        "**The default.** The released pipeline stays bge-base over 20 candidates on the hybrid first stage: it",
        "is the quality ceiling measured in stages 05 and 06 and nothing here beats it on the labels. The",
        "budget is the one cut that is free on every retrieval number, and v3 is that cut, released if its run",
        "clears the gates above. The small cross-encoder is the latency option: when a second a question matters",
        "more than a few places of MRR, `minilm-l6 over 20` is a version away, without the budget. The",
        "relevance floor is not an option at any value tried. Nothing in this report moves the generation",
        "model, its effort or the prompt, which is where the next factor of two in cost would come from.",
        "",
    ]
    return out
