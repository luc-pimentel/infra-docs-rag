"""Write a testset from the corpus: sample chunks, ask a model for one question and reference answer
per chunk, and save the result for a person to audit (`eval/dataset/audit.yaml`).

Generation runs through Claude Code's `claude -p` like the `claude-code` generator does: no tools, one
turn, an empty working directory, the subscription the CLI is logged into. A generated example is a
candidate, not a record: the audit decides what enters the dataset.
"""

import hashlib
import json
import os
import random
import subprocess
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import yaml

from ..chunk.chunks import Chunk
from ..generate.generator import CLI_TIMEOUT, DEFAULT_MODEL, MODELS
from .dataset import Difficulty, Generated, GeneratedSet

MIN_CHARS = 400  # a chunk shorter than this is a heading and a sentence; nothing to ask about
DEFAULT_N = 30
DEFAULT_SEED = 13
DEFAULT_EFFORT = "low"

SYSTEM = """You write evaluation examples for a question-answering system over the public documentation of \
Kubernetes, Prometheus, Argo CD and two CNCF security whitepapers. You are given one passage from that \
documentation, under the heading it has in its document.

Write ONE question that a practitioner would actually type into such a system and that this passage \
answers. Ask it the way a person who has not read the passage would: name the thing they want to do or \
understand, not the passage, the section or "the documentation". Do not quote the passage's first \
sentence back as a question.

Then write the reference answer: one to three sentences that answer the question using only what the \
passage says. No outside knowledge, no hedging about what the passage does not say. If the passage \
cannot support a self-contained question (it is a table fragment, a list of links, a code block with \
no explanation), say so instead of forcing one.

Difficulty: "easy" when the question shares the passage's own words for the thing it asks about, \
"medium" when it paraphrases them, "hard" when the answer depends on a condition, exception or default \
that the passage states.

Reply with JSON only, no prose around it:
{"query": "...", "reference_answer": "...", "difficulty": "easy|medium|hard", "tags": ["two", "to", "four", "words"]}
or, when the passage cannot support a question:
{"skip": "why"}"""


def prompt_hash() -> str:
    return hashlib.sha256(SYSTEM.encode()).hexdigest()[:12]


def eligible(chunks: list[Chunk], used: set[str]) -> list[Chunk]:
    """Chunks a question can be written from: real sources, enough text, not already a context."""
    return [
        c
        for c in chunks
        if not c.source_id.startswith("fixture-") and len(c.text) >= MIN_CHARS and c.chunk_id not in used
    ]


def sample(chunks: list[Chunk], n: int, seed: int) -> list[Chunk]:
    """`n` chunks, at least one per source and the rest in proportion to each source's share, in a
    fixed order for the seed."""
    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for c in chunks:
        by_source[c.source_id].append(c)
    if not by_source:
        return []
    rng = random.Random(seed)
    sources = sorted(by_source)
    n = min(n, len(chunks))
    quota = {s: 1 for s in sources} if n >= len(sources) else {s: 0 for s in sources}
    remaining = n - sum(quota.values())
    total = sum(len(by_source[s]) for s in sources)
    shares = {s: len(by_source[s]) / total for s in sources}
    for s in sources:
        quota[s] += int(remaining * shares[s])
    leftover = n - sum(quota.values())
    for s in sorted(sources, key=lambda s: (remaining * shares[s]) % 1, reverse=True)[:leftover]:
        quota[s] += 1
    picked: list[Chunk] = []
    for s in sources:
        pool = sorted(by_source[s], key=lambda c: c.chunk_id)
        picked.extend(rng.sample(pool, min(quota[s], len(pool))))
    return picked


def user_text(chunk: Chunk) -> str:
    return f"{chunk.citation()}\n\n{chunk.text}"


@dataclass
class Reply:
    text: str
    ms: float
    input_tokens: int
    output_tokens: int


class ClaudeCodeWriter:
    """`claude -p` with the writer's system prompt."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        effort: str = DEFAULT_EFFORT,
        binary: str = "claude",
        cwd: Path | None = None,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"unknown model {model!r}; one of {', '.join(MODELS)}")
        self.name = "claude-code"
        self.model = model
        self.effort = effort
        self.binary = binary
        self.cwd = cwd or Path(tempfile.mkdtemp(prefix="infra-docs-rag-dataset-"))

    def command(self) -> list[str]:
        return [
            self.binary,
            "-p",
            "--system-prompt",
            SYSTEM,
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

    def ask(self, text: str) -> Reply:
        env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
        started = time.perf_counter()
        try:
            done = subprocess.run(
                self.command(),
                input=text,
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
        usage = reply.get("usage") or {}
        return Reply(
            text=str(reply.get("result") or ""),
            ms=ms,
            input_tokens=(usage.get("input_tokens") or 0)
            + (usage.get("cache_creation_input_tokens") or 0)
            + (usage.get("cache_read_input_tokens") or 0),
            output_tokens=usage.get("output_tokens") or 0,
        )


def parse(text: str) -> dict:
    """The JSON object in a reply, with or without a code fence around it."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise ValueError(f"no JSON object in the reply: {text[:200]!r}")
    return json.loads(text[start : end + 1])


DIFFICULTIES: tuple[Difficulty, ...] = ("easy", "medium", "hard")


@dataclass
class Skipped:
    chunk_id: str
    why: str


@dataclass
class Generation:
    examples: list[Generated]
    skipped: list[Skipped]
    ms: float
    input_tokens: int
    output_tokens: int


def generate(chunks: list[Chunk], writer: ClaudeCodeWriter, log=print) -> Generation:
    examples: list[Generated] = []
    skipped: list[Skipped] = []
    ms = tokens_in = tokens_out = 0
    for i, chunk in enumerate(chunks, 1):
        reply = writer.ask(user_text(chunk))
        ms += reply.ms
        tokens_in += reply.input_tokens
        tokens_out += reply.output_tokens
        try:
            data = parse(reply.text)
        except (ValueError, json.JSONDecodeError) as error:
            skipped.append(Skipped(chunk.chunk_id, f"unreadable reply: {error}"))
            log(f"[{i}/{len(chunks)}] {chunk.chunk_id}: unreadable reply")
            continue
        if "skip" in data:
            skipped.append(Skipped(chunk.chunk_id, str(data["skip"])))
            log(f"[{i}/{len(chunks)}] {chunk.chunk_id}: skipped ({data['skip']})")
            continue
        difficulty = data.get("difficulty")
        if difficulty not in DIFFICULTIES:
            difficulty = "medium"
        query, answer = str(data.get("query", "")).strip(), str(data.get("reference_answer", "")).strip()
        if not query or not answer:
            skipped.append(Skipped(chunk.chunk_id, "reply had no question or no answer"))
            log(f"[{i}/{len(chunks)}] {chunk.chunk_id}: empty reply")
            continue
        examples.append(
            Generated(
                id=f"gen-{len(examples) + 1:02d}",
                chunk_id=chunk.chunk_id,
                source_doc_id=chunk.source_id,
                section=" › ".join(chunk.section_path),
                query=query,
                reference_answer=answer,
                difficulty=difficulty,
                tags=[str(t) for t in data.get("tags", [])][:4],
            )
        )
        log(f"[{i}/{len(chunks)}] {chunk.chunk_id}: {query}")
    return Generation(examples, skipped, ms, tokens_in, tokens_out)


def save(gen: Generation, chunking: str, writer: ClaudeCodeWriter, seed: int, output: Path) -> GeneratedSet:
    generated = GeneratedSet(
        chunking=chunking,
        model=writer.model,
        backend=writer.name,
        effort=writer.effort,
        prompt_sha256=prompt_hash(),
        seed=seed,
        generated_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        examples=gen.examples,
    )
    header = (
        "# A testset written by a model from sampled chunks: one question and reference answer per chunk,\n"
        "# for `infra-docs-rag dataset generate`. Every example here is a candidate; audit.yaml says which\n"
        "# entered the dataset and why. Regenerating with the same seed samples the same chunks.\n"
    )
    skipped = (
        "# Chunks the writer declined, with its reason:\n"
        + "".join(f"#   {s.chunk_id}: {' '.join(s.why.split())}\n" for s in gen.skipped)
        if gen.skipped
        else ""
    )
    body = yaml.safe_dump(generated.model_dump(), sort_keys=False, allow_unicode=True, width=100)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + skipped + "\n" + body)
    return generated
