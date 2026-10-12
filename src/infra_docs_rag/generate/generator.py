"""The language model behind an answer. `ClaudeGenerator` calls the Claude API with the passages as
citable documents; `ClaudeCodeGenerator` runs the same prompt through Claude Code's `claude -p`, on
its subscription, and checks the citations the model types; the protocol lets a test substitute a
scripted model."""

import json
import os
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from ..retrieve.retriever import Context
from .prompt import user_content, user_text
from .prompts import DEFAULT_PROMPT, Prompt, load_prompt

MODELS = {
    # Claude Opus 5.5, the current Opus: $4 / $20 per million input / output tokens, $0.20 per million read
    # from the cache. Thinking is on by default and `effort` sets how much; forced tool choice and
    # assistant prefill are gone, so the answer format is carried by the system prompt and citations.
    "claude-opus-5-5": {"input": 4.0, "output": 20.0, "cache_read": 0.20},
}
DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TOKENS = 2048  # a few sentences, or a short list; the system prompt asks for no more

# What `--generator` can name: the API, by model id, or the Claude Code CLI.
GENERATORS = {
    DEFAULT_MODEL: "the Claude API through the official SDK, citations attached by the API",
    "claude-code": "Claude Code's `claude -p` on its subscription, citations typed by the model and checked",
}
CLI_TIMEOUT = 300  # seconds one `claude -p` call may take


@dataclass
class Cited:
    """One citation attached to a span of the answer: which document (passage number, 1-based) and what
    text in it the span rests on."""

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
    unverified: int = 0  # citations the model typed that did not check out against a passage
    cached: bool = False  # answered from the generation cache: no call was made

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

    def __init__(
        self, model: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, prompt: Prompt | None = None
    ) -> None:
        import anthropic  # slow import, paid only when a model is used

        if model not in MODELS:
            raise ValueError(f"unknown generator model {model!r}; one of {', '.join(MODELS)}")
        self.client = anthropic.Anthropic()
        self.name = "claude"
        self.model = model
        self.effort = effort
        self.prices = MODELS[model]
        self.prompt = prompt or load_prompt()
        self.system = self.prompt.api

    def _create(self, context: Context, question: str):
        """One request: the passages as citable documents, then the question."""
        return self.client.beta.messages.create(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=self.system,
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


# A typed citation: `[2]`, or several, at the end of a sentence, before or after its final punctuation;
# its quote follows on the next line(s), opened by `>` and closed by `<`.
MARKERS = re.compile(r"((?:\s*\[\d{1,2}\])+)\s*([.!?:;,]*)\s*$")
QUOTE_LINES = 12  # how far a wrapped quote may run before an unclosed `>` line is taken alone


def normal(text: str) -> str:
    """Text as compared for a quote: backticks and emphasis dropped (the model and the passage do not
    always agree on them) and whitespace collapsed."""
    return " ".join(text.replace("`", "").replace("*", "").split())


def parse_typed(text: str, context: Context) -> tuple[list[Segment], int]:
    """Read an answer whose citations the model typed: each sentence on its own line ending in `[n]`,
    then a `> quote <` line, which may wrap over several lines. A quote found in the passage it names (or,
    failing that, in any passage the sentence names) becomes a citation; a quote found nowhere is dropped
    and counted, and a number with no quote behind it is counted too, so a sentence with nothing verified
    is uncited. Lines are joined back with newlines in whitespace-only segments, so the marked answer
    reads as the model wrote it."""
    passages = {s.n: normal(s.chunk.text) for s in context.sources}
    segments: list[Segment] = []
    typed: list[set[int]] = []  # the numbers each text segment ended with, verified or not
    unverified = 0
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        if line.startswith(">"):
            parts = [line[1:].strip()]
            if not parts[-1].endswith("<"):  # wrapped: take the lines up to the one that closes it
                close = next(
                    (
                        j
                        for j in range(i, min(i + QUOTE_LINES, len(lines)))
                        if lines[j].rstrip().endswith("<")
                    ),
                    None,
                )
                if close is not None:
                    parts += [lines[j].strip() for j in range(i, close + 1) if lines[j].strip()]
                    i = close + 1
            quote = " ".join(parts).removesuffix("<").strip()
            found = normal(quote)
            last = typed[-1] if typed else set()
            n = next((n for n in last if found and found in passages[n]), None)
            if n is None:
                n = next((n for n, p in passages.items() if found and found in p), None)
            if n is None or not typed:
                if not typed or not typed[-1]:
                    unverified += 1  # a stray quote: no number typed for it; a numbered one is counted below
            else:
                next(s for s in reversed(segments) if s.text.strip()).citations.append(
                    Cited(document=n, cited_text=quote)
                )
            continue
        tail = MARKERS.search(line)
        numbers = [int(x) for x in re.findall(r"\[(\d+)\]", tail.group(1))] if tail else []
        known = {n for n in numbers if n in passages}
        unverified += len(numbers) - len(known)  # a number no passage has
        body = line[: tail.start()].rstrip() + tail.group(2) if tail else line
        if not body:  # markers on a line of their own belong to the sentence before them
            if typed:
                typed[-1] |= known
            else:
                unverified += len(known)
            continue
        if segments:
            segments.append(Segment(text="\n"))
        segments.append(Segment(text=body))
        typed.append(known)
    cited = [s for s in segments if s.text.strip()]
    for seg, numbers in zip(cited, typed, strict=True):
        unverified += len(numbers - {c.document for c in seg.citations})
    return segments, unverified


class ClaudeCodeGenerator:
    """Claude through Claude Code's `claude -p`: the same prompt, on the subscription the CLI is logged
    into rather than an API key. The CLI has no citation API, so the system prompt asks for typed
    citations and `parse_typed` checks every quote against its passage. Runs in an empty directory so
    the CLI reads no project instructions, with no tools and one turn."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        binary: str = "claude",
        cwd: Path | None = None,
        prompt: Prompt | None = None,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown generator model {model!r}; one of {', '.join(MODELS)}")
        self.name = "claude-code"
        self.model = model
        self.effort = effort
        self.prices = MODELS[model]
        self.prompt = prompt or load_prompt()
        self.system = self.prompt.typed
        self.binary = binary
        self.cwd = cwd or Path(tempfile.mkdtemp(prefix="infra-docs-rag-claude-"))

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
            "--model",
            self.model,
            "--effort",
            self.effort,
            "--no-session-persistence",
        ]

    def generate(self, context: Context, question: str) -> Generated:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}  # not a nested session
        started = time.perf_counter()
        try:
            done = subprocess.run(
                self.command(),
                input=user_text(context, question),
                capture_output=True,
                text=True,
                cwd=self.cwd,
                env=env,
                timeout=CLI_TIMEOUT,
            )
        except FileNotFoundError as error:
            raise ValueError(
                f"{self.binary!r} is not on PATH: install Claude Code, or use --generator {DEFAULT_MODEL} "
                "with an API key"
            ) from error
        except subprocess.TimeoutExpired as error:
            raise ValueError(f"`{self.binary} -p` took longer than {CLI_TIMEOUT}s") from error
        wall = (time.perf_counter() - started) * 1000
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip()[-500:]
            raise ValueError(f"`{self.binary} -p` exited {done.returncode}: {detail}")
        try:
            reply = json.loads(done.stdout)
        except json.JSONDecodeError as error:
            raise ValueError(f"`{self.binary} -p` returned no JSON: {done.stdout[:200]!r}") from error
        if reply.get("is_error"):
            raise ValueError(f"`{self.binary} -p` failed: {str(reply.get('result', ''))[:500]}")
        usage = reply.get("usage") or {}
        cache_read = usage.get("cache_read_input_tokens") or 0
        stop = reply.get("stop_reason") or "end_turn"
        segments, unverified = parse_typed(str(reply.get("result") or ""), context)
        return Generated(
            segments=segments,
            stop_reason="end_turn" if stop == "stop_sequence" else stop,
            model=next(iter(reply.get("modelUsage") or {}), self.model),
            usage=Usage(
                input_tokens=(usage.get("input_tokens") or 0)
                + (usage.get("cache_creation_input_tokens") or 0)
                + cache_read,
                output_tokens=usage.get("output_tokens") or 0,
                cache_read_tokens=cache_read,
            ),
            ms=float(reply.get("duration_api_ms") or wall),
            refusal="refusal" if stop == "refusal" else None,
            unverified=unverified,
        )


def load_generator(
    name: str = DEFAULT_MODEL, effort: str = DEFAULT_EFFORT, prompt: str = DEFAULT_PROMPT
) -> Generator:
    if name == "claude-code":
        return ClaudeCodeGenerator(effort=effort, prompt=load_prompt(prompt))
    return ClaudeGenerator(name, effort, load_prompt(prompt))
