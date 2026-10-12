"""The canary: a handful of questions with known right answers, asked of a deployed pipeline. A deploy
that changes the index, the prompt or the retrieval and gets these wrong is caught before anyone else
asks. Each question costs one model call, so the set is small and the questions are ones every version
so far has answered right: a failure is the deployment, not the question."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError

from ..generate.answer import Answer, answer
from ..generate.generator import Generator
from ..retrieve.retriever import Retrieval, Retriever

CANARY = Path("eval/canary.yaml")


class CanaryQuestion(BaseModel):
    id: str
    text: str
    expect: Literal["answer", "abstain"]
    source: str | None = None  # the source id an answer has to cite
    contains: str | None = None  # text an answer has to contain
    why: str = ""


class CanarySet(BaseModel):
    questions: list[CanaryQuestion]


def load_canary(path: Path = CANARY) -> list[CanaryQuestion]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        return CanarySet.model_validate(yaml.safe_load(path.read_text()) or {}).questions
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error


@dataclass
class CanaryResult:
    question: CanaryQuestion
    answer: Answer
    passed: bool
    detail: str


def check(q: CanaryQuestion, a: Answer) -> tuple[bool, str]:
    if q.expect == "abstain":
        if a.abstained:
            return True, f"{a.outcome}" + (f" ({a.reason})" if a.reason else "")
        return False, f"expected an abstention, got {a.outcome}"
    if a.outcome != "answered":
        return False, f"expected an answer, got {a.outcome}" + (f" ({a.reason})" if a.reason else "")
    cited = {c.chunk.source_id for c in a.citations}
    if q.source and q.source not in cited:
        return False, f"answered but cited {', '.join(sorted(cited)) or 'nothing'}, not {q.source}"
    if q.contains and q.contains not in a.text:
        return False, f"answered but the text lacks {q.contains!r}"
    return (
        True,
        f"answered, cited {q.source or ', '.join(sorted(cited))}, {a.cited_share:.0%} of the text cited",
    )


def run_canary(
    questions: list[CanaryQuestion],
    retriever: Retriever,
    generator: Generator,
    retrieval: Retrieval,
    on_answer=None,
) -> list[CanaryResult]:
    results = []
    for q in questions:
        a = answer(q.text, retriever, generator, retrieval, retries=1)
        passed, detail = check(q, a)
        results.append(CanaryResult(q, a, passed, detail))
        if on_answer is not None:
            on_answer(a)
        print(f"{'pass' if passed else 'FAIL'}  {q.id}: {detail}")
    return results
