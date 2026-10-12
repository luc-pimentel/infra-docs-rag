"""A cache in front of the generator. The key is everything that decides the answer: the model, the
prompt version, the question, and the chunks the model would read, in order. The same question against
the same passages under the same prompt is the same answer, so the second time costs nothing and takes
no time; a different index version, a changed prompt or a reordered context is a different key. One
JSON line per entry in `data/cache/generations.jsonl`."""

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ..retrieve.retriever import Context
from .generator import Cited, Generated, Generator, Segment, Usage

CACHE = Path("data/cache/generations.jsonl")


def key_for(model: str, prompt_version: str, question: str, context: Context) -> str:
    ids = "|".join(s.chunk.chunk_id for s in context.sources)
    return hashlib.sha256(f"{model}\n{prompt_version}\n{question}\n{ids}".encode()).hexdigest()[:24]


def dump(g: Generated) -> dict:
    return {
        "segments": [
            {
                "text": s.text,
                "citations": [{"document": c.document, "cited_text": c.cited_text} for c in s.citations],
            }
            for s in g.segments
        ],
        "stop_reason": g.stop_reason,
        "model": g.model,
        "usage": {
            "input_tokens": g.usage.input_tokens,
            "output_tokens": g.usage.output_tokens,
            "cache_read_tokens": g.usage.cache_read_tokens,
        },
        "ms": g.ms,
        "refusal": g.refusal,
        "unverified": g.unverified,
    }


def load(d: dict) -> Generated:
    return Generated(
        segments=[
            Segment(
                text=s["text"],
                citations=[Cited(document=c["document"], cited_text=c["cited_text"]) for c in s["citations"]],
            )
            for s in d["segments"]
        ],
        stop_reason=d["stop_reason"],
        model=d["model"],
        usage=Usage(**d["usage"]),
        ms=d["ms"],
        refusal=d.get("refusal"),
        unverified=d.get("unverified", 0),
    )


class GenerationCache:
    def __init__(self, path: Path = CACHE) -> None:
        self.path = path
        self.entries: dict[str, dict] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line:
                    row = json.loads(line)
                    self.entries[row["key"]] = row["generated"]

    def get(self, key: str) -> Generated | None:
        row = self.entries.get(key)
        return load(row) if row else None

    def put(self, key: str, generated: Generated) -> None:
        row = dump(generated)
        self.entries[key] = row
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as out:
            out.write(json.dumps({"key": key, "generated": row}) + "\n")

    def __len__(self) -> int:
        return len(self.entries)


@dataclass
class CachedGenerator:
    """A generator that answers from the cache when it can. A hit comes back with zero tokens, zero
    cost and the lookup's time, and `hits` counts them; a miss calls the model and is stored. A refusal
    or a cut-off answer is not stored: those are worth asking again."""

    inner: Generator
    cache: GenerationCache
    prompt_version: str
    hits: int = 0
    misses: int = 0

    @property
    def name(self) -> str:
        return self.inner.name

    @property
    def model(self) -> str:
        return self.inner.model

    @property
    def prices(self) -> dict[str, float]:
        return self.inner.prices

    @property
    def effort(self) -> str:
        return str(getattr(self.inner, "effort", ""))

    def generate(self, context: Context, question: str) -> Generated:
        started = time.perf_counter()
        key = key_for(self.model, self.prompt_version, question, context)
        found = self.cache.get(key)
        if found is not None:
            self.hits += 1
            return Generated(
                segments=found.segments,
                stop_reason=found.stop_reason,
                model=found.model,
                usage=Usage(),
                ms=(time.perf_counter() - started) * 1000,
                refusal=found.refusal,
                unverified=found.unverified,
                cached=True,
            )
        self.misses += 1
        generated = self.inner.generate(context, question)
        if generated.stop_reason == "end_turn" and generated.refusal is None:
            self.cache.put(key, generated)
        return generated
