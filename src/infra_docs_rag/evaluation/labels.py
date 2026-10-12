"""Hand labels in `eval/labels.yaml`: a person's reading of the answers in one saved run, on the axes
the judge and the mechanical checks score automatically. The calibration compares the two."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ValidationError, field_validator

from .runs import Run

LABELS = Path("eval/labels.yaml")
Correct = Literal["yes", "partial", "no"]
Citations = Literal["right", "wrong", "none"]


class Label(BaseModel):
    id: str
    grounded: bool  # every claim in the answer can be read in a passage
    fabricated: bool  # the answer states a specific fact no passage gives
    gap_named: bool | None = (
        None  # names what the passages do not cover, or rejects a false premise; None: n/a
    )
    correct: Correct  # the answer answers the question (as the passages allow)
    citations: Citations  # the cited passages are where the answer is, or not, or there are none
    note: str = ""  # what the reader saw; required where a label disagrees with the automatic reading

    @field_validator("correct", mode="before")
    @classmethod
    def yaml_yes_no(cls, value: object) -> object:
        """YAML reads a bare `yes` or `no` as a boolean; take it as the word."""
        return {True: "yes", False: "no"}.get(value, value) if isinstance(value, bool) else value


class LabelSet(BaseModel):
    run: str  # the saved run the answers come from
    labeler: str
    guidelines: str = ""  # where the labeling guidelines are
    revised_rules: str = ""  # what the disagreements taught, written after reading them
    labels: list[Label]


def load_labels(path: Path, run: Run) -> LabelSet:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    try:
        loaded = LabelSet.model_validate(yaml.safe_load(path.read_text()) or {})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    if loaded.run != run.name:
        raise ValueError(f"{path} labels run {loaded.run!r}, not {run.name!r}")
    ids = [x.id for x in loaded.labels]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: a question is labelled twice")
    cases = run.by_id()
    for x in loaded.labels:
        c = cases.get(x.id)
        if c is None:
            raise ValueError(f"{path}: {x.id} is not in run {run.name}")
        if not c.answered:
            raise ValueError(f"{path}: {x.id} has no answer to label ({c.outcome})")
    return loaded
