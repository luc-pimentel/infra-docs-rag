"""A second reading of each dataset answer, by the same probability judge the grounding stage uses,
on the questions the dataset can ask and the benchmark cannot: does the answer address the question
(relevance), does it agree with the reference answer, does it say a method is deprecated when the
passage says so, does it name more than one reading of an ambiguous question. Four yes/no questions
per answer, with the reference answer in the state; probabilities, never text."""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..generate.judge import CLI_TIMEOUT
from ..retrieve.retriever import Context

QUESTIONS = {
    "relevant": {
        "type": "noul",
        "instructions": (
            "Does the ANSWER address the QUESTION that was asked? An answer about a neighbouring topic, "
            "or one that answers a different question than the one posed, is not relevant even if it is "
            "true."
        ),
        "criteria": {
            "true": "the answer speaks to what the question asks, in whole or in the part it covers",
            "false": "the answer is about something else, or restates passages without answering",
        },
    },
    "reference_agrees": {
        "type": "noul",
        "instructions": (
            "Does the ANSWER agree with the REFERENCE ANSWER? Compare what each one says the right reply "
            "is: the same facts, commands, values and caveats, in any wording. Extra correct detail in "
            "the ANSWER is fine; a contradiction, a missing main point or a different conclusion is not."
        ),
        "criteria": {
            "true": "the answer reaches the reference's conclusion and keeps its main points",
            "false": "the answer contradicts the reference, misses a main point, or concludes differently",
        },
    },
    "deprecation_named": {
        "type": "noul",
        "instructions": (
            "If a PASSAGE says that a field, flag, value, API or method in the ANSWER is deprecated, "
            "removed or replaced, does the ANSWER say so? Answer true also when no passage mentions any "
            "deprecation at all (there was nothing to flag)."
        ),
        "criteria": {
            "true": "every deprecation the passages state about something the answer recommends is named in the answer, or there is none",
            "false": "the answer presents as current something a passage marks as deprecated, removed or replaced",
        },
    },
    "readings_named": {
        "type": "noul",
        "instructions": (
            "If the QUESTION can reasonably be read in more than one way given the PASSAGES (different "
            "systems, different things with the same name), does the ANSWER name more than one reading, "
            "or ask which one is meant, or say which reading it took? Answer true also when the question "
            "has only one reasonable reading."
        ),
        "criteria": {
            "true": "the answer names the readings, asks which, or says which it answered; or the question is not ambiguous",
            "false": "the question has several readings and the answer picks one without saying so",
        },
    },
}


@dataclass
class Reading:
    relevant: float  # P(the answer addresses the question)
    reference_agrees: float  # P(the answer agrees with the reference answer)
    deprecation_named: float  # P(the answer flags what the passages mark deprecated, or nothing was)
    readings_named: float  # P(the answer names the readings of an ambiguous question, or it is not one)
    model: str
    cost_usd: float
    ms: float


class Reader(Protocol):
    name: str

    def read(self, context: Context, question: str, reference: str, answer_text: str) -> Reading: ...


def reading_state(context: Context, question: str, reference: str, answer_text: str) -> str:
    """What the reader sees: the passages under the headings the model saw, the question, the
    reference answer, the answer with its markers."""
    return reading_state_of(
        [(s.heading, s.chunk.text) for s in context.sources], question, reference, answer_text
    )


def reading_state_of(passages: list[tuple[str, str]], question: str, reference: str, answer_text: str) -> str:
    """The state from (heading, text) pairs; the heading already starts with `[n]`."""
    lines = ["PASSAGES"]
    for heading, text in passages:
        lines.append(f"{heading}\n{text}")
    lines += ["", "QUESTION", question, "", "REFERENCE ANSWER", reference, "", "ANSWER", answer_text]
    return "\n".join(lines)


class JevReader:
    """`jev eval` with the four questions, the state on stdin."""

    def __init__(self, binary: str = "jev") -> None:
        self.name = "jev"
        self.binary = binary
        self.request = Path(tempfile.mkdtemp(prefix="infra-docs-rag-reader-")) / "questions.json"
        self.request.write_text(json.dumps({"questions": QUESTIONS}))

    def command(self) -> list[str]:
        return [self.binary, "eval", "-f", str(self.request), "--state-file", "-", "-o", "json"]

    def read(self, context: Context, question: str, reference: str, answer_text: str) -> Reading:
        return self.read_state(reading_state(context, question, reference, answer_text))

    def read_state(self, text: str) -> Reading:
        """The four readings over a state already written (`reading_state` or `reading_state_of`)."""
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
            return Reading(
                relevant=float(answers["relevant"]["noul"]),
                reference_agrees=float(answers["reference_agrees"]["noul"]),
                deprecation_named=float(answers["deprecation_named"]["noul"]),
                readings_named=float(answers["readings_named"]["noul"]),
                model=str(reply.get("model") or "jev"),
                cost_usd=float(reply.get("cost_usd") or 0.0),
                ms=float(reply.get("latency_ms") or 0.0),
            )
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"`{self.binary} eval` returned something unexpected: {done.stdout[:200]!r}"
            ) from error


def load_reader(binary: str = "jev") -> JevReader:
    if shutil.which(binary) is None:
        raise ValueError(
            f"{binary!r} is not on PATH: install jev (`jev auth login` once), or pass --no-judge"
        )
    return JevReader(binary)
