import hashlib
import re
from pathlib import Path

import numpy as np
import pytest

from infra_docs_rag.embed.chunks import Chunk
from infra_docs_rag.ingest.dedup import content_hash
from infra_docs_rag.ingest.models import Document, ParseStatus, Provenance, SourceType

FIXTURES = Path(__file__).parent / "fixtures"


class FakeEmbedder:
    """A stand-in for an embedding model: no downloads, same output every run.

    Each word is hashed, salted with the model's name, into one of a few dimensions, so texts that
    share words land close together and two fake models have different maps. Vectors come out at
    whatever length the counts give them; scaling them to length 1 is the index's job. `fixed`
    pins exact vectors for texts that need a known geometry.
    """

    def __init__(self, name: str = "fake", revision: str = "r1", dimensions: int = 64, fixed=None):
        self.name = name
        self.model_id = f"test/{name}"
        self.revision = revision
        self.query_prefix = ""
        self.dimensions = dimensions
        self.max_tokens = 8
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
