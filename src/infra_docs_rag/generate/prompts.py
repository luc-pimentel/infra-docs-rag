"""The prompt registry: every system prompt the generator can be given, under a name, with the hash that
tells one revision of it from another. A pipeline version pins a prompt by name; a saved run records the
name and the hash, so a run made under an older wording of the same name is told apart from one made
under the current wording."""

import hashlib
from dataclasses import dataclass

from .prompt import SYSTEM, SYSTEM_TYPED


@dataclass(frozen=True)
class Prompt:
    name: str
    api: str  # the system prompt for the API, citations attached by the API
    typed: str  # the same for `claude -p`, citations typed by the model
    note: str = ""

    def text(self, backend: str) -> str:
        return self.typed if backend == "claude-code" else self.api

    def sha(self, backend: str) -> str:
        return hashlib.sha256(self.text(backend).encode()).hexdigest()[:12]

    def version(self, backend: str) -> str:
        """`cite-v1+e732a9f0cc40`: the name, then the hash of the wording the backend gets."""
        return f"{self.name}+{self.sha(backend)}"


PROMPTS = {
    p.name: p
    for p in (
        Prompt(
            name="cite-v1",
            api=SYSTEM,
            typed=SYSTEM_TYPED,
            note="The stage 07 prompt: answer from the passages alone, cite every statement, abstain with "
            "a fixed sentence, name what the passages do not cover.",
        ),
    )
}
DEFAULT_PROMPT = "cite-v1"


def load_prompt(name: str = DEFAULT_PROMPT) -> Prompt:
    if name not in PROMPTS:
        raise ValueError(f"unknown prompt {name!r}; one of {', '.join(PROMPTS)}")
    return PROMPTS[name]
