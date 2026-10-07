"""The language model behind an answer. `ClaudeGenerator` calls the Claude API with the passages as
citable documents; the protocol lets a test substitute a scripted model."""

import time
from dataclasses import dataclass, field
from typing import Protocol

from ..retrieve.retriever import Context
from .prompt import SYSTEM, user_content

MODELS = {
    # Claude Opus 5.5, the current Opus: $4 / $20 per million input / output tokens, $0.20 per million read
    # from the cache. Thinking is on by default and `effort` sets how much; forced tool choice and
    # assistant prefill are gone, so the answer format is carried by the system prompt and citations.
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
}
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 2048  # a few sentences, or a short list; the system prompt asks for no more


@dataclass
class Cited:
    """One citation the API attached to a span of the answer: which document (passage number, 1-based)
    and what text in it the span rests on."""

    document: int
    cited_text: str


@dataclass
class Segment:
    """A run of answer text and the citations attached to it; an uncited run has none."""

    text: str
    citations: list[Cited] = field(default_factory=list)


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0

    def cost(self, prices: dict[str, float]) -> float:
        """In dollars, from per-million-token prices."""
        return (
            (self.input_tokens - self.cache_read_tokens) * prices["input"]
            + self.cache_read_tokens * prices["cache_read"]
            + self.output_tokens * prices["output"]
        ) / 1e6


@dataclass
class Generated:
    """What came back from one call."""

    segments: list[Segment]
    stop_reason: str  # end_turn, max_tokens, refusal, ...
    model: str  # the model that answered; a fallback may differ from the one asked
    usage: Usage
    ms: float
    refusal: str | None = None  # the safety classifier's category when stop_reason is refusal

    @property
    def text(self) -> str:
        return "".join(s.text for s in self.segments)


class Generator(Protocol):
    name: str
    model: str
    prices: dict[str, float]

    def generate(self, context: Context, question: str) -> Generated: ...


class ClaudeGenerator:
    """Claude through the official SDK. Credentials come from the environment (`ANTHROPIC_API_KEY`, or an
    `ant auth login` profile); nothing is read from the repository."""

    def __init__(self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT) -> None:
        import anthropic  # slow import, paid only when a model is used

        if model not in MODELS:
            raise ValueError(f"unknown generator model {model!r}; one of {', '.join(MODELS)}")
        self.client = anthropic.Anthropic()
        self.name = "claude"
        self.model = model
        self.effort = effort
        self.prices = MODELS[model]

    def _create(self, context: Context, question: str):
        """One request: the passages as citable documents, then the question."""
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": user_content(context, question)}],
            output_config={"effort": self.effort},
            # A safety classifier may decline a request; the API then re-runs it on Anthropic's
            # recommended fallback model inside the same call instead of returning the refusal.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )

    def generate(self, context: Context, question: str) -> Generated:
        import anthropic

        started = time.perf_counter()
        try:
            response = self._create(context, question)
        except anthropic.AuthenticationError as error:
            raise ValueError("the Anthropic credentials were rejected: check ANTHROPIC_API_KEY") from error
        except anthropic.RateLimitError as error:
            raise ValueError(f"rate limited by the Anthropic API; retry later ({error.message})") from error
        except anthropic.APIStatusError as error:
            raise ValueError(f"the Anthropic API returned {error.status_code}: {error.message}") from error
        except anthropic.APIConnectionError as error:
            raise ValueError(f"could not reach the Anthropic API: {error}") from error
        except TypeError as error:  # the SDK resolves credentials at request time; none were found
            if "authentication" not in str(error):
                raise
            raise ValueError(
                "no Anthropic credentials: set ANTHROPIC_API_KEY or run `ant auth login` first"
            ) from error
        ms = (time.perf_counter() - started) * 1000
        segments = [
            Segment(
                text=block.text,
                citations=[
                    Cited(document=c.document_index + 1, cited_text=c.cited_text)
                    for c in (block.citations or [])
                ],
            )
            for block in response.content
            if block.type == "text"
        ]
        details = response.stop_details
        return Generated(
            segments=segments,
            stop_reason=response.stop_reason or "end_turn",
            model=response.model,
            usage=Usage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                cache_read_tokens=response.usage.cache_read_input_tokens or 0,
            ),
            ms=ms,
            refusal=(details.category or "unspecified") if details is not None else None,
        )


def load_generator(model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT) -> ClaudeGenerator:
    return ClaudeGenerator(model, effort)
