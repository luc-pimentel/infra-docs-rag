"""The numbers read off a saved run: retrieval, groundedness and citation, abstention, and latency and
cost, each a named metric with a direction, so two runs can be compared metric by metric and a gate
can name the one it watches."""

import statistics
from dataclasses import dataclass
from typing import Literal

from ..generate.evaluate import CLAIM_FLOOR
from ..generate.judge import BAND
from .runs import CaseRecord, Run

# The revised scoring rules from the calibration (reports/calibration.md): where the judge's
# probabilities are read as a verdict, and the mechanical floor an answer has to clear.
GROUNDED_FLOOR = 0.5  # jev grounded under this: the answer reads as ungrounded (the bottom of the band)
FABRICATED_CEILING = 0.5  # jev fabricated at or over this: a fact was added
CITED_FLOOR = CLAIM_FLOOR  # share of an answer's claims (gap sentences aside) that has to carry a citation

Better = Literal["higher", "lower"]


@dataclass
class Metric:
    name: str  # `group.metric`
    value: float | None  # None when nothing in the run could be measured
    shown: str  # how the report prints it: `41/53 (77%)`, `$1.42`, `4,900 ms`
    better: Better
    description: str

    @property
    def group(self) -> str:
        return self.name.split(".")[0]


def share(part: list, whole: list) -> float | None:
    return len(part) / len(whole) if whole else None


def shown_share(part: list, whole: list) -> str:
    return f"{len(part)}/{len(whole)} ({len(part) / len(whole):.0%})" if whole else "–"


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


def ms(value: float | None) -> str:
    return "–" if value is None else f"{value:,.0f} ms"


def flags(c: CaseRecord) -> list[str]:
    """Why an answered question is suspect under the revised rules; empty when nothing is."""
    if not c.answered:
        return []
    out: list[str] = []
    if c.right_cited is False:
        out.append("wrong passage cited")
    if c.claim_share < CITED_FLOOR:
        out.append(f"claims cited {c.claim_share:.0%}")
    j = c.judgement
    if j is not None:
        if j.grounded < GROUNDED_FLOOR:
            out.append(f"judge grounded {j.grounded:.2f}")
        if j.fabricated >= FABRICATED_CEILING:
            out.append(f"judge fabricated {j.fabricated:.2f}")
    return out


def metrics(run: Run) -> dict[str, Metric]:
    cases = run.cases
    prices = run.manifest.prices
    in_scope = [c for c in cases if c.in_scope]
    labelled = [c for c in in_scope if c.labelled]
    oos = [c for c in cases if c.kind == "out-of-scope"]
    adv = [c for c in cases if c.kind == "adversarial"]
    answered = [c for c in cases if c.answered]
    answered_labelled = [c for c in labelled if c.answered]
    called = [c for c in cases if c.called]
    judged = [c for c in cases if c.judgement is not None]
    n = len(cases)

    right_retrieved = [c for c in labelled if c.right_retrieved]
    at_one = [c for c in labelled if c.right_rank == 1]
    rr = [1 / c.right_rank if c.right_rank else 0.0 for c in labelled]
    right_cited = [c for c in answered_labelled if c.right_cited]
    fully = [c for c in answered if not c.uncited]
    unverified = sum(c.generated.unverified for c in called)
    judge_grounded = [c for c in judged if c.judgement.grounded >= GROUNDED_FLOOR]
    judge_fabricated = [c for c in judged if c.judgement.fabricated >= FABRICATED_CEILING]
    unsure = [
        c
        for c in judged
        if any(BAND[0] <= p <= BAND[1] for p in (c.judgement.grounded, c.judgement.fabricated))
    ]
    flagged = [c for c in answered if flags(c)]
    oos_abstained = [c for c in oos if c.abstained]
    false_abstentions = [c for c in in_scope if c.abstained]
    adv_held = [c for c in adv if c.held]
    in_scope_held = [c for c in in_scope if c.held]
    held = [c for c in cases if c.held]
    cost = sum(c.generated.usage.cost(prices) for c in called)
    judge_cost = sum(c.judgement.cost_usd for c in judged)
    tokens_in = sum(c.generated.usage.input_tokens for c in called)
    tokens_out = sum(c.generated.usage.output_tokens for c in called)
    generate_ms = [c.generated.ms for c in called]
    end_to_end = [c.end_to_end_ms for c in cases]
    retrieve_ms = [c.retrieve_ms for c in cases]

    def m(name: str, value: float | None, shown: str, better: Better, description: str) -> Metric:
        return Metric(name, value, shown, better, description)

    out = [
        m(
            "retrieval.right_retrieved",
            share(right_retrieved, labelled),
            shown_share(right_retrieved, labelled),
            "higher",
            f"labelled questions with a right passage among the {run.manifest.retrieval.k} the model read",
        ),
        m(
            "retrieval.hit_at_1",
            share(at_one, labelled),
            shown_share(at_one, labelled),
            "higher",
            "labelled questions whose first passage is a right one",
        ),
        m(
            "retrieval.mrr",
            statistics.mean(rr) if rr else None,
            f"{statistics.mean(rr):.2f}" if rr else "–",
            "higher",
            "mean reciprocal rank of the first right passage over the labelled questions",
        ),
        m(
            "retrieval.retrieve_ms_median",
            median(retrieve_ms),
            ms(median(retrieve_ms)),
            "lower",
            "embedding, ranking and reranking per question, at the median",
        ),
        m(
            "grounding.right_cited",
            share(right_cited, answered_labelled),
            shown_share(right_cited, answered_labelled),
            "higher",
            "answered labelled questions that cite a right passage",
        ),
        m(
            "grounding.cited_share_median",
            median([c.cited_share for c in answered]),
            f"{median([c.cited_share for c in answered]):.0%}" if answered else "–",
            "higher",
            "share of answer text carrying a verified citation, at the median over the answers",
        ),
        m(
            "grounding.claim_cited_median",
            median([c.claim_share for c in answered]),
            f"{median([c.claim_share for c in answered]):.0%}" if answered else "–",
            "higher",
            "the same share with the gap sentences left out: what the answer claims about the subject",
        ),
        m(
            "grounding.gap_sentences",
            share([c for c in answered if c.gap_sentences], answered),
            shown_share([c for c in answered if c.gap_sentences], answered),
            "higher",
            "answers with a sentence naming what the passages do not cover",
        ),
        m(
            "grounding.fully_cited",
            share(fully, answered),
            shown_share(fully, answered),
            "higher",
            "answers with no uncited sentence",
        ),
        m(
            "grounding.unverified_citations",
            float(unverified),
            str(unverified),
            "lower",
            "citations the model typed that did not check out against a passage, over the run",
        ),
        m(
            "grounding.judge_grounded",
            share(judge_grounded, judged),
            shown_share(judge_grounded, judged),
            "higher",
            f"answers the judge reads as grounded (P ≥ {GROUNDED_FLOOR})",
        ),
        m(
            "grounding.judge_fabricated",
            share(judge_fabricated, judged),
            shown_share(judge_fabricated, judged),
            "lower",
            f"answers the judge reads as adding a fact (P ≥ {FABRICATED_CEILING})",
        ),
        m(
            "grounding.judge_unsure",
            float(len(unsure)),
            str(len(unsure)),
            "lower",
            f"answers with a grounded or fabricated reading in the {BAND[0]:.1f}–{BAND[1]:.1f} band",
        ),
        m(
            "grounding.flagged",
            share(flagged, answered),
            shown_share(flagged, answered),
            "lower",
            "answers a revised rule flags: wrong passage, claims cited under the floor, or the judge's reading",
        ),
        m(
            "abstention.out_of_scope",
            share(oos_abstained, oos),
            shown_share(oos_abstained, oos),
            "higher",
            "out-of-scope questions that got the abstention sentence",
        ),
        m(
            "abstention.in_scope",
            share(false_abstentions, in_scope),
            shown_share(false_abstentions, in_scope),
            "lower",
            "in-scope questions that got the abstention sentence (false abstentions)",
        ),
        m(
            "abstention.adversarial_held",
            share(adv_held, adv),
            shown_share(adv_held, adv),
            "higher",
            "adversarial questions whose expectation held",
        ),
        m(
            "answers.in_scope_held",
            share(in_scope_held, in_scope),
            shown_share(in_scope_held, in_scope),
            "higher",
            f"in-scope questions answered from a right passage with at least {CITED_FLOOR:.0%} of their claims cited",
        ),
        m(
            "answers.held",
            share(held, cases),
            shown_share(held, cases),
            "higher",
            "questions whose expectation held, over the whole benchmark",
        ),
        m("cost.usd_total", cost, f"${cost:.2f}", "lower", f"{len(called)} calls at list prices"),
        m(
            "cost.usd_per_1000",
            cost / n * 1000 if n else None,
            f"${cost / n * 1000:.2f}" if n else "–",
            "lower",
            "generation cost per thousand questions, abstentions at the threshold included",
        ),
        m(
            "cost.tokens_in_per_call",
            tokens_in / len(called) if called else None,
            f"{tokens_in // len(called):,}" if called else "–",
            "lower",
            "input tokens a call reads on average",
        ),
        m(
            "cost.tokens_out_per_call",
            tokens_out / len(called) if called else None,
            f"{tokens_out // len(called):,}" if called else "–",
            "lower",
            "output tokens a call writes on average",
        ),
        m("cost.judge_usd", judge_cost, f"${judge_cost:.4f}", "lower", f"{len(judged)} judge readings"),
        m(
            "latency.generate_ms_median",
            median(generate_ms),
            ms(median(generate_ms)),
            "lower",
            "the model's time per call, at the median",
        ),
        m(
            "latency.end_to_end_ms_median",
            median(end_to_end),
            ms(median(end_to_end)),
            "lower",
            "retrieval plus generation per question, at the median, over every question",
        ),
        m(
            "latency.end_to_end_ms_p90",
            percentile(end_to_end, 0.9),
            ms(percentile(end_to_end, 0.9)),
            "lower",
            "retrieval plus generation per question, at the 90th percentile",
        ),
    ]
    return {x.name: x for x in out}


GROUPS = {
    "retrieval": "Retrieval",
    "grounding": "Groundedness and citations",
    "abstention": "Abstention",
    "answers": "Expectations",
    "cost": "Cost",
    "latency": "Latency",
}
