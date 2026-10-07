import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from infra_docs_rag.chunk.chunks import Chunk
from infra_docs_rag.ingest.dedup import content_hash
from infra_docs_rag.ingest.models import Document, ParseStatus, Provenance, SourceType

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """A stand-in for an embedding model: no downloads, same output every run.

    Each word is hashed, salted with the model's name, into one of a few dimensions, so texts that
    share words land close together and two fake models have different maps. Vectors come out at
    whatever length the counts give them; scaling them to length 1 is the index's job. `fixed`
    pins exact vectors for texts that need a known geometry. A token is a run of non-space characters.
    """

    def __init__(self, name="fake", revision="r1", dimensions=64, fixed=None, max_tokens=8):
        self.name = name
        self.model_id = f"test/{name}"
        self.revision = revision
        self.query_prefix = ""
        self.dimensions = dimensions
        self.max_tokens = max_tokens
        self.special_tokens = 0
        self.parameters = 0
        self.fixed = fixed or {}

    def vector(self, text: str) -> np.ndarray:
        if text in self.fixed:
            return np.array(self.fixed[text], dtype=np.float32)
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for word in re.findall(r"\w+", text.lower()):
            vector[int(hashlib.sha256(f"{self.name}:{word}".encode()).hexdigest(), 16) % self.dimensions] += 1
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self.vector(t) for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)

    def token_counts(self, texts: list[str]) -> list[int]:
        return [len(t.split()) for t in texts]

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        return [m.span() for m in re.finditer(r"\S+", text)]


@pytest.fixture
def fake_embedder():
    return FakeEmbedder


@pytest.fixture
def make_chunk():
    def make(section: str, text: str | None = None, source_id: str = "src", parent: str = "Doc") -> Chunk:
        text = text or section
        return Chunk(
            chunk_id=f"{source_id}:{parent}/{section}",
            source_id=source_id,
            document_id=f"doc_{source_id}",
            source_uri=f"https://example.com/{source_id}",
            section_path=[parent, section],
            start=0,
            end=len(text),
            text=text,
        )

    return make


@pytest.fixture
def make_doc():
    def make(source_id: str, text: str) -> Document:
        return Document(
            document_id=f"doc_{source_id}",
            source_id=source_id,
            source_uri=f"https://example.com/{source_id}",
            source_type=SourceType.TEXT,
            title=source_id,
            raw_text=text,
            cleaned_text=text,
            content_hash=content_hash(text),
            parse_status=ParseStatus.OK,
            provenance=Provenance(
                local_path=f"data/raw/{source_id}.txt",
                source_sha256="0" * 64,
                fetched_at=None,
                parser="test",
                ingest_version="test",
                ingested_at="2026-09-28T00:00:00+00:00",
            ),
        )

    return make


class FakeReranker:
    """A stand-in for a cross-encoder: no downloads, same output every run. Relevance is the share of
    the query's words the text contains, so a text that repeats the question scores 1 and one that
    shares nothing scores 0. `fixed` pins a relevance for a text that needs a known one."""

    def __init__(self, name="fake-reranker", fixed=None):
        self.name = name
        self.model_id = f"test/{name}"
        self.revision = "r1"
        self.parameters = 0
        self.max_tokens = 512
        self.fixed = fixed or {}

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        words = set(re.findall(r"\w+", query.lower()))
        out = []
        for text in texts:
            if text in self.fixed:
                out.append(self.fixed[text])
                continue
            found = set(re.findall(r"\w+", text.lower()))
            out.append(len(words & found) / len(words) if words else 0.0)
        return np.asarray(out, dtype=np.float32)


@pytest.fixture
def fake_reranker():
    return FakeReranker


class FakeGenerator:
    """A stand-in for the language model: answers come from a script keyed by question, as a list of
    (text, [passage numbers]) segments; a question with no script gets the abstention sentence. No
    network, same output every run."""

    def __init__(self, script=None, stop_reason="end_turn", refusal=None):
        from infra_docs_rag.generate.generator import Cited, Generated, Segment, Usage
        from infra_docs_rag.generate.prompt import ABSTAIN

        self.name = "fake"
        self.model = "fake-model"
        self.prices = {"input": 1.0, "output": 2.0, "cache_read": 0.1}
        self.script = script or {}
        self.stop_reason = stop_reason
        self.refusal = refusal
        self.calls = []
        self._types = (Cited, Generated, Segment, Usage, ABSTAIN)

    def generate(self, context, question):
        Cited, Generated, Segment, Usage, ABSTAIN = self._types
        self.calls.append((question, [s.chunk.chunk_id for s in context.sources]))
        lines = self.script.get(question, [(ABSTAIN, [])])
        segments = [
            Segment(
                text=text,
                citations=[
                    Cited(document=n, cited_text=context.sources[n - 1].chunk.text[:20]) for n in numbers
                ],
            )
            for text, numbers in lines
        ]
        return Generated(
            segments=segments,
            stop_reason=self.stop_reason,
            model=self.model,
            usage=Usage(input_tokens=100, output_tokens=10, cache_read_tokens=0),
            ms=1.0,
            refusal=self.refusal,
        )


@pytest.fixture
def fake_generator():
    return FakeGenerator
