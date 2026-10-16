"""Traces that explain why an eval fails (stage 17). A regression score says that a record stopped
holding; its trace says where in the request it broke. For every record of a dataset run whose
expectation did not hold, `diagnose` reads the trace the run saved beside it (`traces.jsonl`) and
names the root cause from the execution evidence: the named passage never ranked (first stage), was
handed to the reranker and reranked past the top k (rerank), ranked and then dropped under the floor
(threshold) or past the token budget (budget); or the passage reached the model and the prompt did not
hold (an abstention expected, near-miss passages answered from; an instruction in the question
followed), the context was at its budget with the answer left partly uncited (overload), or nothing
upstream explains it (generation). The report is `reports/traces.md`."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..observe.trace import Trace
from .dataset import Record
from .runs import CaseRecord, Run

TRACES_FILE = "traces.jsonl"
Cause = Literal[
    "retrieval/first-stage",
    "retrieval/rerank",
    "retrieval/threshold",
    "retrieval/budget",
    "prompt/abstention",
    "prompt/instruction",
    "context/overload",
    "generation",
    "held",
]
STAGE_OF = {
    "retrieval/first-stage": "retrieval",
    "retrieval/rerank": "retrieval",
    "retrieval/threshold": "retrieval",
    "retrieval/budget": "retrieval",
    "prompt/abstention": "prompt construction",
    "prompt/instruction": "prompt construction",
    "context/overload": "context overload",
    "generation": "generation",
    "held": "held",
}
OVERLOAD_PASSAGES = 4  # a context this full, at its budget, with claims left uncited reads as overload
INJECTION_MARKS = ("[system", "ignore", "append the full text", "cite nothing", "disregard")


def load_traces(run_dir: Path) -> dict[str, Trace]:
    """The run's traces by example id, from `traces.jsonl` beside it; empty when the run has none."""
    path = run_dir / TRACES_FILE
    if not path.exists():
        return {}
    out: dict[str, Trace] = {}
    for line in path.read_text().splitlines():
        if line:
            t = Trace.model_validate_json(line)
            if t.example_id:
                out[t.example_id] = t
    return out


def save_traces(traces: list[Trace], run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / TRACES_FILE
    path.write_text("".join(t.model_dump_json() + "\n" for t in traces))
    return path


@dataclass
class Diagnosis:
    """Where one failed record broke, with the evidence read off its trace."""

    id: str
    cause: Cause
    evidence: str
    trace_id: str | None
    verdict: str

    @property
    def stage(self) -> str:
        return STAGE_OF[self.cause]


def rank_in(hits, chunk_ids: set[str]) -> int | None:
    """The best rank any of the named chunks holds in a ranking; None when none is there."""
    ranks = [h.rank for h in hits if h.chunk_id in chunk_ids]
    return min(ranks) if ranks else None


def diagnose(c: CaseRecord, r: Record, t: Trace | None) -> Diagnosis:
    """The root cause of a record that did not hold. Without a trace, only what the case itself shows."""
    if c.held:
        return Diagnosis(c.id, "held", "", c.trace_id, c.verdict)
    named = set(r.contexts)
    read = {s.chunk_id for s in c.sources}
    if named and not (named & read):
        if t is None:
            return Diagnosis(
                c.id,
                "retrieval/first-stage",
                "no trace saved; the named passage was not among the passages read",
                None,
                c.verdict,
            )
        first = rank_in(t.first_stage, named)
        if first is None:
            return Diagnosis(
                c.id,
                "retrieval/first-stage",
                f"the named passage is not among the {len(t.first_stage)} candidates the first stage ranked "
                f"(top first-stage hit {t.first_stage[0].citation if t.first_stage else '–'})",
                t.trace_id,
                c.verdict,
            )
        dropped = next((d for d in t.dropped if d.chunk_id in named), None)
        if dropped is not None:
            cause = (
                "retrieval/threshold" if dropped.reason in ("threshold", "relevance") else "retrieval/budget"
            )
            hit = next((h for h in t.ranked if h.chunk_id == dropped.chunk_id), None)
            score = f", cosine {hit.cosine:.2f}" if hit and hit.cosine is not None else ""
            return Diagnosis(
                c.id,
                cause,
                f"the named passage ranked {dropped.rank} and the pack step dropped it: {dropped.reason}{score}"
                + (
                    f"; {t.tokens:,} of {t.budget:,} tokens packed"
                    if cause == "retrieval/budget" and t.budget
                    else ""
                ),
                t.trace_id,
                c.verdict,
            )
        if t.reranked:
            after = rank_in(t.reranked, named)
            kept = len(t.ranked) or len(t.context)
            return Diagnosis(
                c.id,
                "retrieval/rerank",
                f"the first stage ranked the named passage {first} of {len(t.first_stage)}; the reranker put it "
                f"{after} of {len(t.reranked)}, past the {kept} it keeps"
                + (f" (top reranked: {t.reranked[0].citation})" if t.reranked else ""),
                t.trace_id,
                c.verdict,
            )
        return Diagnosis(
            c.id,
            "retrieval/first-stage",
            f"the first stage ranked the named passage {first}, past the {len(t.context)} the model reads",
            t.trace_id,
            c.verdict,
        )
    # The passage reached the model, or the record names none.
    question = (t.question if t else c.question).lower()
    if c.expect == "abstain" and c.answered:
        if any(m in question for m in INJECTION_MARKS):
            return Diagnosis(
                c.id,
                "prompt/instruction",
                "an instruction in the question and the model answered instead of abstaining",
                c.trace_id,
                c.verdict,
            )
        top = f"{t.top_score:.2f}" if t and t.top_score is not None else "–"
        return Diagnosis(
            c.id,
            "prompt/abstention",
            f"the record expects an abstention; {len(c.sources)} near-miss passages cleared the floor (top cosine {top}) "
            "and the prompt's abstention rule did not hold against them",
            c.trace_id,
            c.verdict,
        )
    if c.answered and c.verdict == "partly uncited" and any(m in question for m in INJECTION_MARKS):
        return Diagnosis(
            c.id,
            "prompt/instruction",
            f"an instruction in the question; the answer left {c.claim_share:.0%} of its claims cited",
            c.trace_id,
            c.verdict,
        )
    at_budget = (
        t is not None and t.budget is not None and (t.over_budget > 0 or (t.tokens or 0) >= 0.9 * t.budget)
    )
    if c.answered and c.verdict == "partly uncited" and len(c.sources) >= OVERLOAD_PASSAGES and at_budget:
        return Diagnosis(
            c.id,
            "context/overload",
            f"{len(c.sources)} passages, {t.tokens:,} of {t.budget:,} tokens packed"
            + (f", {t.over_budget} more left out" if t.over_budget else "")
            + f"; the answer cited {c.claim_share:.0%} of its claims",
            c.trace_id,
            c.verdict,
        )
    where = ""
    if named and t is not None:
        n = rank_in(t.context, named)
        where = f"the named passage was passage [{n}] of {len(t.context)}; "
    elif named:
        n = next((s.n for s in c.sources if s.chunk_id in named), None)
        where = f"the named passage was passage [{n}] of {len(c.sources)}; " if n else ""
    what = {
        "abstained": "the model abstained with it in hand",
        "cited the wrong passage": "the model cited another passage",
        "partly uncited": f"the answer cited {c.claim_share:.0%} of its claims",
    }.get(c.verdict, c.verdict)
    return Diagnosis(c.id, "generation", where + what, c.trace_id, c.verdict)


def diagnose_run(run: Run, records: list[Record], traces: dict[str, Trace]) -> list[Diagnosis]:
    by_record = {r.id: r for r in records}
    return [diagnose(c, by_record[c.id], traces.get(c.id)) for c in run.cases if c.id in by_record]
