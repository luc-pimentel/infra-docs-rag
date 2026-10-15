"""A second reader for every answer. Jev, TypeSafe AI's System One model through the `jev` CLI, returns
calibrated probabilities and never text, so it can say how likely it is that an answer's claims are in
the passages without writing an answer of its own. Three yes/no questions per answer. Nothing here
changes a verdict: the probabilities sit next to the mechanical checks, and calibrating them against
hand labels is the evaluation stage's job."""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..retrieve.retriever import Context

QUESTIONS = {
    "grounded": {
        "type": "noul",
        "instructions": (
            "Is every factual claim in the ANSWER stated in the PASSAGES? Read the ANSWER sentence by "
            "sentence and look for each claim in the PASSAGES."
        ),
        "criteria": {
            "true": "each statement in the answer can be read in one of the passages, paraphrase included",
            "false": "at least one statement in the answer says something no passage says",
        },
    },
    "fabricated": {
        "type": "noul",
        "instructions": (
            "Does the ANSWER add a fact the PASSAGES do not give: a number, a version, a default value, "
            "a name, a behaviour or a step that appears in no passage?"
        ),
        "criteria": {
            "true": "the answer contains at least one specific fact that no passage contains",
            "false": "every specific fact in the answer appears in a passage",
        },
    },
    "gap_named": {
        "type": "noul",
        "instructions": (
            "Does the ANSWER say what the PASSAGES do not cover, or push back on a premise of the "
            "QUESTION that the PASSAGES contradict, instead of answering as if everything were covered "
            "and true?"
        ),
        "criteria": {
            "true": (
                "the answer names a part of the question the passages do not answer, or says a premise "
                "of the question is wrong according to the passages"
            ),
            "false": "the answer treats the question as fully answerable and its premises as true",
        },
    },
}
BAND = (0.4, 0.6)  # a probability in here means the model cannot tell, not "medium"
CLI_TIMEOUT = 60


@dataclass
class Judgement:
    grounded: float  # P(every claim is in the passages)
    fabricated: float  # P(a fact was added that no passage gives)
    gap_named: float  # P(the answer names what is uncovered, or rejects a false premise)
    model: str
    cost_usd: float
    ms: float

    @property
    def unsure(self) -> list[str]:
        """The questions whose probability sits in the band where the model cannot tell."""
        return [
            name
            for name, p in (
                ("grounded", self.grounded),
                ("fabricated", self.fabricated),
                ("gap_named", self.gap_named),
            )
            if BAND[0] <= p <= BAND[1]
        ]


class Judge(Protocol):
    name: str

    def judge(self, context: Context, question: str, answer_text: str) -> Judgement: ...


def state(context: Context, question: str, answer_text: str) -> str:
    """What the judge reads: the passages by number, each under the heading the model saw it under (a
    page or section claim in the answer comes from there), the question, the answer with its markers."""
    return state_of([(s.heading, s.chunk.text) for s in context.sources], question, answer_text)


def state_of(passages: list[tuple[str, str]], question: str, answer_text: str) -> str:
    """The state from (heading, text) pairs; the heading already starts with `[n]`."""
    lines = ["PASSAGES"]
    for heading, text in passages:
        lines.append(f"{heading}\n{text}")
    lines += ["", "QUESTION", question, "", "ANSWER", answer_text]
    return "\n".join(lines)


class JevJudge:
    """`jev eval` with the three questions, the state on stdin. The API key comes from `jev auth login`;
    nothing is read from the repository."""

    def __init__(self, binary: str = "jev") -> None:
        self.name = "jev"
        self.binary = binary
        self.request = Path(tempfile.mkdtemp(prefix="infra-docs-rag-jev-")) / "questions.json"
        self.request.write_text(json.dumps({"questions": QUESTIONS}))

    def command(self) -> list[str]:
        return [self.binary, "eval", "-f", str(self.request), "--state-file", "-", "-o", "json"]

    def judge(self, context: Context, question: str, answer_text: str) -> Judgement:
        return self.read_state(state(context, question, answer_text))

    def read_state(self, text: str) -> Judgement:
        """The three readings over a state already written (`state` or `state_of`)."""
        try:
            done = subprocess.run(
                self.command(),
                input=text,
                capture_output=True,
                text=True,
                timeout=CLI_TIMEOUT,
            )
        except FileNotFoundError as error:
            raise ValueError(f"{self.binary!r} is not on PATH: install jev, or pass --no-judge") from error
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"`{self.binary} eval` took longer than {CLI_TIMEOUT}s") from error
        if done.returncode != 0:
            raise ValueError(f"`{self.binary} eval` exited {done.returncode}: {done.stderr.strip()[-500:]}")
        try:
            reply = json.loads(done.stdout)
            answers = reply["answers"]
            return Judgement(
                grounded=float(answers["grounded"]["noul"]),
                fabricated=float(answers["fabricated"]["noul"]),
                gap_named=float(answers["gap_named"]["noul"]),
                model=str(reply.get("model") or "jev"),
                cost_usd=float(reply.get("cost_usd") or 0.0),
                ms=float(reply.get("latency_ms") or 0.0),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"`{self.binary} eval` returned something unexpected: {done.stdout[:200]!r}"
            ) from error


def load_judge(binary: str = "jev") -> JevJudge:
    if shutil.which(binary) is None:
        raise ValueError(
            f"{binary!r} is not on PATH: install jev (`jev auth login` once), or pass --no-judge"
        )
    return JevJudge(binary)
