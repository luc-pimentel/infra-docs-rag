"""The judge specification in `eval/rubric.yaml` and the grader that scores an answer on it.

The rubric names three dimensions, each scored 0, 1 or 2 against one thing: groundedness against the
passages, completeness against the reference answer, abstention against what the passages cover. A
dimension that does not apply is scored null. `RubricGrader` renders the rubric into a system prompt,
sends the passages, the question, the reference answer and the answer through Claude Code's `claude -p`
with a JSON schema, and returns one structured score per dimension with a one-sentence reason. The
grader is a model reading an answer; the hand labels in `eval/calibration.yaml` are the check on it, and
`judge_calibrate.py` compares the two. Grades are saved under the run they read, so the report
regenerates without a model call."""

import hashlib
import json
import os
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import yaml
from pydantic import BaseModel, ValidationError, field_validator, model_validator

from ..generate.generator import CLI_TIMEOUT, DEFAULT_EFFORT, DEFAULT_MODEL, MODELS
from .dataset_generate import parse
from .runs import CaseRecord, JudgementRecord, ReadingRecord

RUBRIC = Path("eval/rubric.yaml")
GRADES = "grades.json"  # under eval/runs/<run>/
SCORES = (0, 1, 2)


class Example(BaseModel):
    score: int
    id: str  # a case in the run the labels were read on
    why: str


class Dimension(BaseModel):
    id: str
    question: str
    reads: str
    applies_when: str
    scores: dict[int, str]
    examples: list[Example] = []

    @field_validator("scores")
    @classmethod
    def three_scores(cls, value: dict[int, str]) -> dict[int, str]:
        if tuple(sorted(value)) != SCORES:
            raise ValueError(f"scores must be defined for exactly {SCORES}, got {sorted(value)}")
        return value

    @model_validator(mode="after")
    def examples_use_defined_scores(self) -> "Dimension":
        for e in self.examples:
            if e.score not in self.scores:
                raise ValueError(f"{self.id}: example {e.id} has score {e.score}, not one of {SCORES}")
        return self


class Operating(BaseModel):
    pass_at: int  # the dimension passes at or over this score

    @field_validator("pass_at")
    @classmethod
    def one_or_two(cls, value: int) -> int:
        if value not in (1, 2):
            raise ValueError(f"pass_at must be 1 or 2, got {value}")
        return value


class Rubric(BaseModel):
    name: str
    abstention_sentence: str
    dimensions: list[Dimension]
    operating: dict[str, Operating]

    @model_validator(mode="after")
    def operating_covers_every_dimension(self) -> "Rubric":
        ids = [d.id for d in self.dimensions]
        if len(ids) != len(set(ids)):
            raise ValueError("a dimension is defined twice")
        if set(self.operating) != set(ids):
            raise ValueError(f"operating thresholds for {sorted(self.operating)}, dimensions are {ids}")
        return self

    @property
    def ids(self) -> list[str]:
        return [d.id for d in self.dimensions]

    def pass_at(self, dimension: str) -> int:
        return self.operating[dimension].pass_at

    def passes(self, dimension: str, score: int | None) -> bool | None:
        """Whether a score passes the dimension; None when the dimension did not apply."""
        return None if score is None else score >= self.pass_at(dimension)


def load_rubric(path: Path = RUBRIC) -> Rubric:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        return Rubric.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


# ---- the grader -------------------------------------------------------------------------------


def system_prompt(rubric: Rubric) -> str:
    """The rubric as the grader's instructions: one dimension at a time, the score definitions as
    written, the output as JSON. The examples are left out so the grader is read on them like on any
    other case."""
    lines = [
        "You grade one answer from a documentation question-answering system. You are given the PASSAGES "
        "the system read (each under the heading it saw it under), the QUESTION, the REFERENCE ANSWER a "
        "person wrote from those passages, and the ANSWER the system gave, with [n] after each span it cited "
        "from passage n.",
        "",
        "Score the ANSWER on each dimension below, one at a time, with the score definitions as written. "
        "Each dimension looks at one thing; a score on one must not depend on the others. Read groundedness "
        "against the PASSAGES only, never against what you know of the software: a claim that is true but in "
        "no passage is not grounded. Read completeness against the REFERENCE ANSWER only. Read abstention "
        "against what the PASSAGES cover.",
        "",
        f'The fixed abstention sentence is: "{rubric.abstention_sentence}" An ANSWER that starts with it is '
        "an abstention; a REFERENCE ANSWER that is it says the passages do not answer the question.",
        "",
    ]
    for d in rubric.dimensions:
        lines += [
            f"## {d.id}",
            f"Question: {d.question}",
            f"Reads: {d.reads}.",
            f"Applies when: {d.applies_when}.",
        ]
        for score in SCORES:
            lines.append(f"- {score}: {' '.join(d.scores[score].split())}")
        lines.append("")
    lines += [
        "Reply with one JSON object and nothing else, of the form "
        + json.dumps({d.id: {"score": 2, "reason": "one sentence"} for d in rubric.dimensions})
        + ". A score is 0, 1 or 2, or null when the dimension does not apply. The reason names the specific "
        "claim, missing point or call that decided the score, in one sentence.",
    ]
    return "\n".join(lines)


def json_schema(rubric: Rubric) -> dict:
    score = {
        "type": "object",
        "properties": {
            "score": {"type": ["integer", "null"], "enum": [0, 1, 2, None]},
            "reason": {"type": "string"},
        },
        "required": ["score", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {d.id: score for d in rubric.dimensions},
        "required": rubric.ids,
        "additionalProperties": False,
    }


def grader_state(passages: list[tuple[str, str]], question: str, reference: str, answer: str) -> str:
    """What the grader reads: the passages under their headings, the question, the reference answer,
    the answer with its markers."""
    lines = ["PASSAGES"]
    for heading, text in passages:
        lines.append(f"{heading}\n{text}")
    lines += ["", "QUESTION", question, "", "REFERENCE ANSWER", reference, "", "ANSWER", answer]
    return "\n".join(lines)


def passages_of(case: CaseRecord) -> list[tuple[str, str]]:
    return [(s.heading or f"[{s.n}] {s.citation}", s.text) for s in case.sources]


class Score(BaseModel):
    score: int | None
    reason: str = ""


class Grade(BaseModel):
    """One answer graded: a score per dimension, what the grader said, and what it cost."""

    id: str
    scores: dict[str, Score]
    model: str
    ms: float
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    judgement: JudgementRecord | None = None  # jev's three readings, for a perturbed answer the run never saw
    reading: ReadingRecord | None = None  # the dataset reader's four, likewise

    def score(self, dimension: str) -> int | None:
        s = self.scores.get(dimension)
        return None if s is None else s.score


class Grader(Protocol):
    name: str
    model: str

    def grade(self, id: str, state: str) -> Grade: ...


def prompt_sha256(rubric: Rubric) -> str:
    return hashlib.sha256(system_prompt(rubric).encode()).hexdigest()[:12]


def grader_version(rubric: Rubric) -> str:
    """`answer-quality-v1+11392d5a1f8c`: the rubric's name, then the hash of the grader's prompt."""
    return f"{rubric.name}+{prompt_sha256(rubric)}"


class RubricGrader:
    """Claude through `claude -p`, the rubric as the system prompt, the scores as a JSON object the CLI
    validates against a schema. Runs in an empty directory with no tools and one turn, on the
    subscription the CLI is logged into."""

    def __init__(
        self,
        rubric: Rubric,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        binary: str = "claude",
        cwd: Path | None = None,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; one of {', '.join(MODELS)}")
        self.name = "claude-code"
        self.rubric = rubric
        self.model = model
        self.effort = effort
        self.prices = MODELS[model]
        self.system = system_prompt(rubric)
        self.schema = json.dumps(json_schema(rubric))
        self.binary = binary
        self.cwd = cwd or Path(tempfile.mkdtemp(prefix="infra-docs-rag-grader-"))

    def command(self) -> list[str]:
        return [
            self.binary,
            "-p",
            "--system-prompt",
            self.system,
            "--tools",
            "",
            "--max-turns",
            "1",
            "--output-format",
            "json",
            "--json-schema",
            self.schema,
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--no-session-persistence",
        ]

    def grade(self, id: str, state: str) -> Grade:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # not a nested session
        started = time.perf_counter()
        try:
            done = subprocess.run(
                self.command(),
                input=state,
                capture_output=True,
                text=True,
                cwd=self.cwd,
                env=env,
                timeout=CLI_TIMEOUT,
            )
        except FileNotFoundError as error:
            raise ValueError(f"{self.binary!r} is not on PATH: install Claude Code") from error
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"`{self.binary} -p` took longer than {CLI_TIMEOUT}s") from error
        ms = (time.perf_counter() - started) * 1000
        if done.returncode != 0:
            raise ValueError(
                f"`{self.binary} -p` exited {done.returncode}: {(done.stderr or done.stdout).strip()[-500:]}"
            )
        try:
            reply = json.loads(done.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(f"`{self.binary} -p` returned no JSON: {done.stdout[:200]!r}") from error
        if reply.get("is_error"):
            raise ValueError(f"`{self.binary} -p` failed: {str(reply.get('result', ''))[:500]}")
        scored = reply.get("structured_output")
        if not isinstance(scored, dict):
            scored = parse(str(reply.get("result") or ""))
        usage = reply.get("usage") or {}
        cache_read = usage.get("cache_read_input_tokens") or 0
        input_tokens = (
            (usage.get("input_tokens") or 0) + (usage.get("cache_creation_input_tokens") or 0) + cache_read
        )
        output_tokens = usage.get("output_tokens") or 0
        cost = (
            (input_tokens - cache_read) * self.prices["input"]
            + cache_read * self.prices["cache_read"]
            + output_tokens * self.prices["output"]
        ) / 1e6
        return Grade(
            id=id,
            scores=self.scores_of(scored),
            model=next(iter(reply.get("modelUsage") or {}), self.model),
            ms=float(reply.get("duration_api_ms") or ms),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )

    def scores_of(self, scored: dict) -> dict[str, Score]:
        out: dict[str, Score] = {}
        for d in self.rubric.ids:
            item = scored.get(d)
            if not isinstance(item, dict) or "score" not in item:
                raise ValueError(f"the grader gave no score for {d}: {json.dumps(scored)[:200]}")
            score = item["score"]
            if score is not None and score not in SCORES:
                raise ValueError(f"the grader scored {d} as {score!r}, not one of {SCORES} or null")
            out[d] = Score(score=score, reason=str(item.get("reason") or "").strip())
        return out


class GradeSet(BaseModel):
    """`eval/runs/<run>/grades.json`: every labelled answer graded, with what graded it."""

    run: str
    labels: str  # the calibration file the ids come from
    rubric: str
    prompt_sha256: str
    model: str
    backend: str
    effort: str
    graded_at: str
    grades: list[Grade]

    def by_id(self) -> dict[str, Grade]:
        return {g.id: g for g in self.grades}

    @property
    def cost_usd(self) -> float:
        return sum(g.cost_usd for g in self.grades)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path: Path) -> "GradeSet":
        if not path.exists():
            raise FileNotFoundError(f"{path} is missing; `judge grade` writes it")
        return cls.model_validate_json(path.read_text())


def new_grade_set(
    run: str, labels: str, rubric: Rubric, grader: RubricGrader, grades: list[Grade]
) -> GradeSet:
    return GradeSet(
        run=run,
        labels=labels,
        rubric=rubric.name,
        prompt_sha256=prompt_sha256(rubric),
        model=grader.model,
        backend=grader.name,
        effort=grader.effort,
        graded_at=datetime.now(UTC).isoformat(timespec="seconds"),
        grades=grades,
    )
