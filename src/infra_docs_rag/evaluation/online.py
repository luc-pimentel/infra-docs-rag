"""Online evaluators and drift detection (stage 19). The offline gates read a fixed dataset; live
traffic asks what the dataset did not. A sampling policy (`eval/online.yaml`) picks the traces worth a
second look out of a window of live traffic: the questions the service could not answer, the ones it
answered on a weak match, the ones that filled the context, the long ones, the ones carrying an
identifier, the ones whose answer left claims uncited, the ones a person marked unhelpful, and a
random share of the rest so the sample stays representative. The judge reads the sampled answers
against the passages they were given; the window is saved under `eval/online/` with its samples and
their readings; the report clusters the failures and compares the window's rates with the offline
baseline and the window before (drift). `promote` turns the sampled failures into dataset records,
with a reference answer drafted from the passages for a person to review."""

import hashlib
import json
import re
import statistics
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from ..chunk.chunks import Chunk
from ..generate.judge import Judge, state_of
from ..observe.feedback import Feedback
from ..observe.trace import Trace, TracedJudgement
from ..retrieve.retriever import locate
from .dataset import Record
from .runs import Run

POLICY = Path("eval/online.yaml")
ONLINE_DIR = Path("eval/online")
HARVESTED = Path("eval/dataset/harvested.yaml")
RULE_ORDER = (
    "dissatisfied",
    "uncited",
    "low-confidence",
    "no-answer",
    "long-context",
    "long-query",
    "identifier",
    "error",
    "random",
)
IDENTIFIER = re.compile(r"\b[a-z]+[A-Z][A-Za-z]+\b|\b\w+\.\w+\.\w+\b|\b[a-z]+_[a-z_]+\b|--[a-z][a-z-]+")


class Drift(BaseModel):
    """How far a window's rate may move from the baseline's before the report flags it."""

    no_answer_rate: float = 0.15
    top_score_median: float = 0.10
    claim_share_median: float = 0.10
    grounded_share: float = 0.10
    query_words_median: int = 10


class Policy(BaseModel):
    window_days: int = 7
    low_confidence_top_score: float = 0.60  # the best hit's cosine under this
    long_context_share: float = 0.9  # tokens packed at or over this share of the budget, or passages left out
    long_query_words: int = 25
    uncited_claim_share: float = 0.8
    random_share: float = 0.10  # of the live traces no rule picked
    max_samples: int = 100
    drift: Drift = Drift()
    rules: dict[str, str] = {}  # rule → why it is sampled, for the report


def load_policy(path: Path = POLICY) -> Policy:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        return Policy.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


class Sample(BaseModel):
    """One live trace the policy picked, with what picked it and what the judge read."""

    trace_id: str
    at: str
    question: str
    version: str
    outcome: str
    rules: list[str]
    words: int
    top_score: float | None = None
    claim_share: float = 0.0
    passages: int = 0
    tokens: int | None = None
    budget: int | None = None
    over_budget: int = 0
    top_source: str | None = None  # the source id of the first passage read
    helpful: bool | None = None
    note: str = ""
    judgement: TracedJudgement | None = None

    @property
    def primary(self) -> str:
        return next((r for r in RULE_ORDER if r in self.rules), self.rules[0] if self.rules else "–")

    @property
    def answered(self) -> bool:
        return self.outcome == "answered"

    @property
    def failed(self) -> bool:
        """Worth a record: a person said so, claims uncited, a weak match answered from, an error, or
        the judge reading it as ungrounded or fabricated."""
        if self.helpful is False or self.outcome == "error":
            return True
        if self.answered and self.claim_share < 0.8:
            return True
        if self.judgement is not None and (
            self.judgement.grounded < 0.5 or self.judgement.fabricated >= 0.85
        ):
            return True
        return False


class Window(BaseModel):
    name: str  # YYYY-Www
    since: str
    until: str
    live: int  # live traces seen in the window
    samples: list[Sample]
    policy: Policy
    judged_at: str | None = None

    def save(self, directory: Path = ONLINE_DIR) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.json"
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "Window":
        return cls.model_validate_json(path.read_text())


def load_windows(directory: Path = ONLINE_DIR) -> list[Window]:
    if not directory.exists():
        return []
    return [Window.load(p) for p in sorted(directory.glob("*.json"))]


def is_live(t: Trace) -> bool:
    """A trace the service or `answer` wrote, not a run's or the canary's."""
    return ":" not in t.version and t.example_id is None and t.run is None


def window_name(day: date) -> str:
    y, w, _ = day.isocalendar()
    return f"{y}-W{w:02d}"


def words_of(question: str) -> int:
    return len(question.split())


def rules_for(t: Trace, policy: Policy, fb: Feedback | None) -> list[str]:
    out: list[str] = []
    if fb is not None and not fb.helpful:
        out.append("dissatisfied")
    if t.answered and t.claim_share < policy.uncited_claim_share:
        out.append("uncited")
    if t.answered and t.top_score is not None and t.top_score < policy.low_confidence_top_score:
        out.append("low-confidence")
    if t.outcome in ("no-passage", "abstained"):
        out.append("no-answer")
    if t.budget and ((t.tokens or 0) >= policy.long_context_share * t.budget or t.over_budget > 0):
        out.append("long-context")
    if words_of(t.question) > policy.long_query_words:
        out.append("long-query")
    if IDENTIFIER.search(t.question):
        out.append("identifier")
    if t.outcome == "error":
        out.append("error")
    return out


def random_pick(trace_id: str, share: float) -> bool:
    """A stable coin per trace: the same trace is picked or not whatever the order."""
    return int(hashlib.sha256(trace_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def sample(
    traces: list[Trace],
    feedback: dict[str, Feedback],
    policy: Policy,
    since: datetime,
    until: datetime,
) -> tuple[list[Sample], int]:
    """The samples out of the live traces in [since, until), and how many live traces there were."""
    live = [t for t in traces if is_live(t) and since <= datetime.fromisoformat(t.at) < until]
    picked: list[Sample] = []
    for t in live:
        fb = feedback.get(t.trace_id)
        rules = rules_for(t, policy, fb)
        if not rules and random_pick(t.trace_id, policy.random_share):
            rules = ["random"]
        if not rules:
            continue
        picked.append(
            Sample(
                trace_id=t.trace_id,
                at=t.at,
                question=t.question,
                version=t.version,
                outcome=t.outcome,
                rules=rules,
                words=words_of(t.question),
                top_score=t.top_score,
                claim_share=t.claim_share,
                passages=len(t.context),
                tokens=t.tokens,
                budget=t.budget,
                over_budget=t.over_budget,
                top_source=t.context[0].chunk_id.split(":")[0] if t.context else None,
                helpful=fb.helpful if fb else None,
                note=fb.note if fb else "",
                judgement=t.judgement,
            )
        )
    ruled = [s for s in picked if s.rules != ["random"]]
    random = [s for s in picked if s.rules == ["random"]]
    kept = (ruled + random)[: policy.max_samples]
    return sorted(kept, key=lambda s: s.at), len(live)


def judge_samples(
    samples: list[Sample],
    traces: dict[str, Trace],
    chunks: dict[str, Chunk],
    judge: Judge,
    log=print,
) -> int:
    """Read every answered sample without a judgement, against the passages its trace names. Returns
    how many were read."""
    read = 0
    for s in samples:
        if not s.answered or s.judgement is not None:
            continue
        t = traces.get(s.trace_id)
        if t is None:
            log(f"{s.trace_id}: no trace in the log; not judged")
            continue
        passages = []
        for h in t.context:
            c = chunks.get(h.chunk_id)
            if c is None:
                log(f"{s.trace_id}: chunk {h.chunk_id} is not in the index; not judged")
                break
            heading = f"[{h.rank}] " + locate(
                [c.project, *c.section_path] if c.project else c.section_path, c.pages
            )
            passages.append((heading, c.text))
        else:
            try:
                j = judge.read_state(state_of(passages, t.question, t.answer))
            except ValueError as error:
                log(f"{s.trace_id}: the judge failed ({str(error)[:120]})")
                continue
            s.judgement = TracedJudgement(
                grounded=j.grounded,
                fabricated=j.fabricated,
                gap_named=j.gap_named,
                model=j.model,
                ms=j.ms,
                cost_usd=j.cost_usd,
            )
            read += 1
            log(f"{s.trace_id}: grounded {j.grounded:.2f}, fabricated {j.fabricated:.2f}")
    return read


class Cluster(BaseModel):
    rule: str
    outcome: str
    source: str | None
    samples: list[str]  # trace ids

    @property
    def key(self) -> str:
        return f"{self.rule} · {self.outcome} · {self.source or 'no passage'}"


def clusters(samples: list[Sample], failed_only: bool = True) -> list[Cluster]:
    """The samples grouped by what picked them, what happened and where the first passage came from."""
    groups: dict[tuple[str, str, str | None], list[str]] = {}
    for s in samples:
        if failed_only and not s.failed and s.primary not in ("no-answer", "dissatisfied"):
            continue
        groups.setdefault((s.primary, s.outcome, s.top_source), []).append(s.trace_id)
    out = [Cluster(rule=r, outcome=o, source=src, samples=ids) for (r, o, src), ids in groups.items()]
    return sorted(out, key=lambda c: (-len(c.samples), c.rule, c.outcome))


class Rate(BaseModel):
    name: str
    window: float | None
    baseline: float | None
    previous: float | None = None
    threshold: float
    shown: str  # how to print: pct | num | int

    @property
    def delta(self) -> float | None:
        return None if self.window is None or self.baseline is None else self.window - self.baseline

    @property
    def drifted(self) -> bool:
        return self.delta is not None and abs(self.delta) > self.threshold

    def fmt(self, v: float | None) -> str:
        if v is None:
            return "–"
        return f"{v:.0%}" if self.shown == "pct" else f"{v:.2f}" if self.shown == "num" else f"{v:.0f}"


def rates_of(
    outcomes: list[str],
    top_scores: list[float],
    claim_shares: list[float],
    grounded: list[float],
    words: list[int],
) -> dict[str, float | None]:
    def med(xs):
        return statistics.median(xs) if xs else None

    return {
        "no_answer_rate": (sum(o in ("no-passage", "abstained") for o in outcomes) / len(outcomes))
        if outcomes
        else None,
        "top_score_median": med(top_scores),
        "claim_share_median": med(claim_shares),
        "grounded_share": (sum(g >= 0.5 for g in grounded) / len(grounded)) if grounded else None,
        "query_words_median": med(words),
    }


def window_rates(
    traces: list[Trace], since: datetime, until: datetime, samples: list[Sample]
) -> dict[str, float | None]:
    """The window's rates over every live trace in it (not just the samples), the judge's over the
    samples the judge read."""
    live = [t for t in traces if is_live(t) and since <= datetime.fromisoformat(t.at) < until]
    return rates_of(
        [t.outcome for t in live],
        [t.top_score for t in live if t.top_score is not None],
        [t.claim_share for t in live if t.answered],
        [s.judgement.grounded for s in samples if s.judgement is not None],
        [words_of(t.question) for t in live],
    )


def baseline_rates(run: Run) -> dict[str, float | None]:
    return rates_of(
        [c.outcome for c in run.cases],
        [c.top_score for c in run.cases if c.top_score is not None],
        [c.claim_share for c in run.cases if c.answered],
        [c.judgement.grounded for c in run.cases if c.judgement is not None],
        [words_of(c.question) for c in run.cases],
    )


def drift(
    window: dict[str, float | None],
    baseline: dict[str, float | None],
    policy: Policy,
    previous: dict[str, float | None] | None = None,
) -> list[Rate]:
    d = policy.drift
    spec = [
        ("no_answer_rate", d.no_answer_rate, "pct"),
        ("top_score_median", d.top_score_median, "num"),
        ("claim_share_median", d.claim_share_median, "pct"),
        ("grounded_share", d.grounded_share, "pct"),
        ("query_words_median", float(d.query_words_median), "int"),
    ]
    return [
        Rate(
            name=n,
            window=window.get(n),
            baseline=baseline.get(n),
            previous=(previous or {}).get(n),
            threshold=th,
            shown=shown,
        )
        for n, th, shown in spec
    ]


# --- promotion: from a sampled failure to a dataset record ----------------------------------------

HARVEST_SYSTEM = """You write evaluation records for a question-answering system over the public documentation of Kubernetes, Prometheus and Argo CD. You are given a question a person asked the running service, the passages the service retrieved for it (each under a numbered heading), and what the service answered. Write the record a careful reviewer would: from the passages alone, decide what a right reply is.

Reply with one JSON object and nothing else:
{"expect": "answer" | "abstain" | "partial",
 "reference_answer": "...",
 "contexts": [passage numbers the reference rests on, e.g. [1, 3]],
 "category": "fact-lookup" | "multi-hop" | "ambiguous" | "no-answer" | "stale" | "adversarial",
 "difficulty": "easy" | "medium" | "hard",
 "failure_mode": one of "none", "false-abstention", "retrieval-miss", "wrong-passage", "uncited-claims", "unverified-citation", "answered-anyway", "judge-misread", "fabrication", "stale-answer", "wrong-reading", "missing-hop",
 "tags": ["..."],
 "why": "one sentence: what this example is in the dataset to catch, given what the service did"}

Rules. "abstain" when no passage answers the question: then the reference_answer is exactly "The indexed documentation does not cover this question." and contexts is []. "partial" when the passages answer part of it: answer that part and name the rest as uncovered. Otherwise "answer": one to four sentences from the passages alone, no outside knowledge. category "no-answer" goes with "abstain". failure_mode names what the service's reply got wrong, or "none" when it was right. Write in English."""


class HarvestWriter:
    """`claude -p` with the harvest prompt; the same CLI call as the dataset writer."""

    def __init__(self, inner) -> None:
        self.inner = inner  # a ClaudeCodeWriter; its command is reused with this system prompt

    def ask(self, text: str):
        from ..evaluation.dataset_generate import ClaudeCodeWriter

        assert isinstance(self.inner, ClaudeCodeWriter)
        command = self.inner.command()
        command[command.index("--system-prompt") + 1] = HARVEST_SYSTEM
        original = self.inner.command
        self.inner.command = lambda: command  # type: ignore[method-assign]
        try:
            return self.inner.ask(text)
        finally:
            self.inner.command = original  # type: ignore[method-assign]


def harvest_prompt(t: Trace, chunks: dict[str, Chunk]) -> str:
    lines = [f"QUESTION\n{t.question}", "", "PASSAGES"]
    for h in t.context:
        c = chunks[h.chunk_id]
        heading = f"[{h.rank}] " + locate(
            [c.project, *c.section_path] if c.project else c.section_path, c.pages
        )
        lines.append(f"{heading}\n{c.text}\n")
    if not t.context:
        lines.append("(no passage qualified)")
    lines += ["", "WHAT THE SERVICE ANSWERED", t.answer or f"({t.outcome}: {t.reason or 'no answer'})"]
    return "\n".join(lines)


def to_record(s: Sample, t: Trace, reply: dict, chunks: dict[str, Chunk], chunking: str) -> Record:
    """A dataset record from the writer's reply, with the sample's provenance in the note."""
    numbers = [int(n) for n in reply.get("contexts", [])]
    ids = [h.chunk_id for h in t.context if h.rank in numbers and h.chunk_id in chunks]
    expect = reply["expect"]
    if expect == "abstain":
        ids = []
    category = reply["category"]
    if category == "multi-hop" and len(ids) < 2:
        category = "fact-lookup"  # the writer named one passage; a multi-hop record rests on two
    return Record(
        id=f"live-{s.trace_id[:8]}",
        query=t.question,
        category=category,
        expect=expect,
        difficulty=reply.get("difficulty", "medium"),
        failure_mode=reply.get("failure_mode", "none"),
        split="test",
        origin="traces",
        source_doc_ids=sorted({i.split(":")[0] for i in ids}),
        contexts=ids,
        reference_answer=reply["reference_answer"].strip(),
        tags=list(reply.get("tags", []))[:6],
        note=(
            f"Harvested from live traffic ({s.at[:10]}, trace {s.trace_id}), sampled as {', '.join(s.rules)}; "
            f"the service {s.outcome}"
            + (f" with {s.claim_share:.0%} of claims cited" if s.answered else "")
            + (
                f"; a person marked it {'helpful' if s.helpful else 'not helpful'}"
                if s.helpful is not None
                else ""
            )
            + f". {reply.get('why', '')}".rstrip()
            + " Reference drafted by the model from the passages, for review."
        ),
    )


def harvest(
    samples: list[Sample],
    traces: dict[str, Trace],
    chunks: dict[str, Chunk],
    writer,
    chunking: str,
    n: int,
    log=print,
) -> list[Record]:
    """Up to n records from the failed samples, most telling first."""
    from .dataset_generate import parse

    order = sorted(
        (s for s in samples if s.failed or s.primary == "no-answer"),
        key=lambda s: RULE_ORDER.index(s.primary),
    )
    out: list[Record] = []
    for s in order:
        if len(out) == n:
            break
        t = traces.get(s.trace_id)
        if t is None or any(h.chunk_id not in chunks for h in t.context):
            log(f"{s.trace_id}: trace or chunks missing; skipped")
            continue
        try:
            reply = parse(writer.ask(harvest_prompt(t, chunks)).text)
            record = to_record(s, t, reply, chunks, chunking)
        except (ValueError, KeyError, ValidationError) as error:
            log(f"{s.trace_id}: the writer's reply did not make a record ({str(error)[:160]})")
            continue
        out.append(record)
        log(
            f"{record.id}: {record.category} · expect {record.expect} · {record.failure_mode} · {len(record.contexts)} contexts"
        )
    return out


def save_harvested(records: list[Record], chunking: str, window: str, path: Path = HARVESTED) -> Path:
    header = (
        "# Records harvested from live traffic by `infra-docs-rag online promote` (stage 19): questions\n"
        "# the sampling policy picked out of a window of real requests, each with a reference answer the\n"
        "# model drafted from the passages the service retrieved, for a person to review. `online promote\n"
        "# --merge` appends them to records.yaml; that moves the dataset version, so the baseline is run\n"
        "# again and the freeze renewed, deliberately.\n"
    )
    body = {
        "chunking": chunking,
        "window": window,
        "harvested_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "records": [json.loads(r.model_dump_json(exclude_none=True)) for r in records],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True, width=100))
    return path


def merge_into_records(harvested: list[Record], records_path: Path) -> int:
    """Append the harvested records to records.yaml (as text, so the file's comments stay). Returns how
    many were added; ids already there are skipped."""
    text = records_path.read_text()
    existing = set(re.findall(r"^\s*- id: (\S+)", text, re.M))
    new = [r for r in harvested if r.id not in existing]
    if not new:
        return 0
    block = yaml.safe_dump(
        [json.loads(r.model_dump_json(exclude_none=True)) for r in new],
        sort_keys=False,
        allow_unicode=True,
        width=100,
    )
    block = "\n".join(("  " + line if line else line) for line in block.splitlines())
    records_path.write_text(
        text.rstrip("\n")
        + "\n\n  # Harvested from live traffic (stage 19); see harvested.yaml for the window.\n"
        + block
        + "\n"
    )
    return len(new)
