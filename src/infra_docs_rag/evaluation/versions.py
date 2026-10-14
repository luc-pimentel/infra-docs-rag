"""Named pipeline versions in `eval/versions.yaml`: each one the configuration the benchmark can run end
to end under a name, so two runs can be told apart and compared: the retrieval, the prompt, the
embedding model whose index is read, and, when pinned, which version of that index."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError, model_validator

from ..embed.embedders import DEFAULT_MODEL, MODELS
from ..generate.prompts import DEFAULT_PROMPT, PROMPTS
from ..retrieve.retriever import DEFAULT_RETRIEVAL, Retrieval

VERSIONS = Path("eval/versions.yaml")
DEFAULT_VERSION = "default"


class Version(BaseModel):
    name: str
    description: str = ""
    retrieval: Retrieval
    prompt: str = DEFAULT_PROMPT  # a name in the prompt registry
    embedding: str = DEFAULT_MODEL  # the model whose index is read
    index: str | None = None  # an index version under data/index/<embedding>/; None reads the current one

    @model_validator(mode="after")
    def known(self) -> "Version":
        if self.prompt not in PROMPTS:
            raise ValueError(
                f"version {self.name}: unknown prompt {self.prompt!r}; one of {', '.join(PROMPTS)}"
            )
        if self.embedding not in MODELS:
            raise ValueError(
                f"version {self.name}: unknown embedding {self.embedding!r}; one of {', '.join(MODELS)}"
            )
        return self


class VersionSet(BaseModel):
    versions: list[Version]


def load_versions(path: Path = VERSIONS) -> dict[str, Version]:
    """The named versions, plus `default`: whatever `retrieve` and `answer` do when no flag is given."""
    versions = {
        DEFAULT_VERSION: Version(
            name=DEFAULT_VERSION, description="the current defaults", retrieval=DEFAULT_RETRIEVAL
        )
    }
    if not path.exists():
        return versions
    try:
        loaded = VersionSet.model_validate(yaml.safe_load(path.read_text()) or {"versions": []})
    except ValidationError as error:
        raise ValueError(f"{path}: {error}") from error
    for v in loaded.versions:
        if v.name in versions:
            raise ValueError(f"{path}: version {v.name!r} is named twice")
        versions[v.name] = v
    return versions


def pick(versions: dict[str, Version], name: str) -> Version:
    if name not in versions:
        raise ValueError(f"unknown version {name!r}; one of {', '.join(versions)} (see {VERSIONS})")
    return versions[name]
