"""The vector index: one model, one metric, stored as plain files under `data/index/<model>/`.

Vectors are stored at length 1 and every query is scaled the same way, so the dot product is the
cosine similarity. The manifest names the model, revision and vector size the vectors came from,
and search refuses any other embedder: two models can produce vectors of the same size that still
live on different maps.
"""

import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .chunks import Chunk
from .embedders import Embedder


class Manifest(BaseModel):
    model: str
    model_id: str
    revision: str
    dimensions: int
    metric: Literal["cosine"] = "cosine"
    query_prefix: str
    max_tokens: int
    chunks: int
    truncated: int  # chunks longer than max_tokens: the model only saw their beginning
    ingest_version: str
    built_at: str
    build_seconds: float


class ModelMismatch(Exception):
    """The query embedder is not the one the index was built with."""


@dataclass
class Hit:
    rank: int
    score: float
    chunk: Chunk


def unit(vectors: np.ndarray) -> np.ndarray:
    """Scale each vector (row) to length 1, so the dot product is the cosine similarity."""
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("a zero vector has no direction to compare")
    return vectors / norms


@dataclass
class Index:
    manifest: Manifest
    chunks: list[Chunk]
    vectors: np.ndarray  # one row per chunk, each of length 1

    def check(self, embedder: Embedder) -> None:
        built = (self.manifest.model_id, self.manifest.revision, self.manifest.dimensions)
        given = (embedder.model_id, embedder.revision, embedder.dimensions)
        if built != given:
            raise ModelMismatch(
                f"the index holds {built[0]}@{built[1][:7]} vectors ({built[2]} dimensions) but the query "
                f"embedder is {given[0]}@{given[1][:7]} ({given[2]} dimensions); vectors from two models "
                "are not comparable, so re-embed the corpus with the new model instead"
            )

    def search(self, embedder: Embedder, query: str, k: int = 5) -> list[Hit]:
        self.check(embedder)
        return self.rank(embedder.embed_queries([query])[0], k)

    def rank(self, query_vector: np.ndarray, k: int) -> list[Hit]:
        """The k chunks closest to a query vector. It trusts the caller to pass a vector from the
        index's own model; `search` is the entry point that checks."""
        scores = self.vectors @ unit(query_vector)
        order = np.argsort(-scores, kind="stable")[:k]
        return [Hit(rank=n + 1, score=float(scores[i]), chunk=self.chunks[i]) for n, i in enumerate(order)]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        with (directory / "chunks.jsonl").open("w") as out:
            for chunk in self.chunks:
                out.write(chunk.model_dump_json() + "\n")
        (directory / "manifest.json").write_text(self.manifest.model_dump_json(indent=2) + "\n")

    @classmethod
    def load(cls, directory: Path) -> "Index":
        if not (directory / "manifest.json").exists():
            raise FileNotFoundError(f"no index in {directory}; run `infra-docs-rag embed` first")
        manifest = Manifest.model_validate_json((directory / "manifest.json").read_text())
        lines = (directory / "chunks.jsonl").read_text().splitlines()
        chunks = [Chunk.model_validate_json(line) for line in lines if line]
        vectors = np.load(directory / "vectors.npy")
        if len(chunks) != manifest.chunks or vectors.shape != (manifest.chunks, manifest.dimensions):
            raise ValueError(
                f"{directory} does not match its manifest; rebuild it with `infra-docs-rag embed`"
            )
        return cls(manifest, chunks, vectors)


def build(embedder: Embedder, chunks: list[Chunk], ingest_version: str) -> Index:
    texts = [c.text for c in chunks]
    started = time.perf_counter()
    vectors = unit(embedder.embed_documents(texts))
    seconds = time.perf_counter() - started
    manifest = Manifest(
        model=embedder.name,
        model_id=embedder.model_id,
        revision=embedder.revision,
        dimensions=embedder.dimensions,
        query_prefix=embedder.query_prefix,
        max_tokens=embedder.max_tokens,
        chunks=len(chunks),
        truncated=sum(n > embedder.max_tokens for n in embedder.token_counts(texts)),
        ingest_version=ingest_version,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        build_seconds=round(seconds, 2),
    )
    return Index(manifest, chunks, vectors)
