"""Probe and compare embedding models on the labelled queries in `eval/queries.yaml`.

The probe scores every query against a small, hand-picked set of chunks (each query's right
sections plus look-alike decoys), small enough to read the whole similarity matrix. The comparison
runs the same queries against every chunk in each model's index. The mixed-maps check shows what
an index returns when it is searched with another model's vectors of the same size.
"""

import platform
import statistics
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import yaml
from pydantic import BaseModel

from ..chunk.chunks import USABLE, Chunk, Chunking, chunk_documents
from ..ingest.models import Document
from .embedders import DEFAULT_MODEL, Embedder, load_embedder
from .index import Hit, Index, Manifest, ModelMismatch, build, unit

Kind = Literal["paraphrase", "identifier", "metadata"]
KINDS: tuple[Kind, ...] = ("paraphrase", "identifier", "metadata")


class Target(BaseModel):
    source: str  # source id from sources.yaml
    sections: list[str]  # section titles; a repeated title can be written as a path, "Parent › Child"
    contains: str | None = None  # text a piece of a split section needs to answer, such as a config key


class Query(BaseModel):
    id: str
    kind: Kind
    text: str
    expect: Target
    hypothesis: str  # written before any scores were computed


class QuerySet(BaseModel):
    queries: list[Query]
    decoys: list[Target]


def load_queries(path: Path) -> QuerySet:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing")
    query_set = QuerySet.model_validate(yaml.safe_load(path.read_text()))
    ids = [q.id for q in query_set.queries]
    if len(ids) != len(set(ids)):
        raise ValueError("query ids must be unique")
    return query_set


def resolve(target: Target, chunks: list[Chunk]) -> list[Chunk]:
    """The chunks a target names. Each title has to match exactly one section of the source."""
    found: list[Chunk] = []
    for title in target.sections:
        wanted = [part.strip() for part in title.split("›")]
        matches = [
            c for c in chunks if c.source_id == target.source and c.section_path[-len(wanted) :] == wanted
        ]
        if len(matches) != 1:
            raise ValueError(f"`{target.source} › {title}` matches {len(matches)} chunks instead of one")
        found.extend(matches)
    return found


def first_right(hits: list[Hit], right_ids: set[str]) -> int:
    """Rank of the first hit that answers the query."""
    return next(h.rank for h in hits if h.chunk.chunk_id in right_ids)


@dataclass
class Probe:
    """Every query scored against the probe set by one model."""

    model: str
    chunks: list[Chunk]
    right: list[set[int]]  # per query, positions of its right sections in `chunks`
    scores: np.ndarray  # queries × chunks, cosine similarity

    def order(self, q: int) -> list[int]:
        return [int(i) for i in np.argsort(-self.scores[q], kind="stable")]

    def rank(self, q: int) -> int:
        """Where the best right section lands for query q (1 = top)."""
        return next(n + 1 for n, i in enumerate(self.order(q)) if i in self.right[q])

    def best_right(self, q: int) -> int:
        return next(i for i in self.order(q) if i in self.right[q])

    def best_wrong(self, q: int) -> int:
        return next(i for i in self.order(q) if i not in self.right[q])

    def margin(self, q: int) -> float:
        """Best right score minus best wrong score: positive when a right section comes first."""
        return float(self.scores[q][self.best_right(q)] - self.scores[q][self.best_wrong(q)])


def probe_set(query_set: QuerySet, chunks: list[Chunk]) -> tuple[list[Chunk], list[set[int]]]:
    """The chunks every model is probed with: all right sections, then the decoys, each once."""
    picked: list[Chunk] = []

    def position(chunk: Chunk) -> int:
        ids = [c.chunk_id for c in picked]
        if chunk.chunk_id in ids:
            return ids.index(chunk.chunk_id)
        picked.append(chunk)
        return len(picked) - 1

    right = [{position(c) for c in resolve(q.expect, chunks)} for q in query_set.queries]
    answers = {c.chunk_id for c in picked}
    for decoy in query_set.decoys:
        for chunk in resolve(decoy, chunks):
            if chunk.chunk_id in answers:
                raise ValueError(f"decoy `{chunk.citation()}` is also a right section")
            position(chunk)
    return picked, right


def probe(embedder: Embedder, query_set: QuerySet, chunks: list[Chunk], right: list[set[int]]) -> Probe:
    documents = unit(embedder.embed_documents([c.text for c in chunks]))
    queries = unit(embedder.embed_queries([q.text for q in query_set.queries]))
    return Probe(embedder.name, chunks, right, queries @ documents.T)


@dataclass
class Comparison:
    """One model's full index against the labelled queries."""

    manifest: Manifest
    parameters: int
    index_bytes: int
    ranks: list[int]  # per query, rank of the first right section among all chunks
    latency_ms: list[float]  # per query: embed it, score every chunk, sort

    def hits(self, k: int, only: list[int] | None = None) -> int:
        ranks = self.ranks if only is None else [self.ranks[q] for q in only]
        return sum(r <= k for r in ranks)

    @property
    def mrr(self) -> float:
        return statistics.fmean(1 / r for r in self.ranks)


def compare(embedder: Embedder, index: Index, query_set: QuerySet) -> Comparison:
    right = [{c.chunk_id for c in resolve(q.expect, index.chunks)} for q in query_set.queries]
    vectors = embedder.embed_queries([q.text for q in query_set.queries])
    everything = len(index.chunks)
    ranks = [first_right(index.rank(v, everything), ids) for v, ids in zip(vectors, right, strict=True)]
    index.search(embedder, query_set.queries[0].text)  # warm-up: the first call pays one-off costs
    latency = []
    for query in query_set.queries:
        started = time.perf_counter()
        index.search(embedder, query.text, k=5)
        latency.append((time.perf_counter() - started) * 1000)
    return Comparison(index.manifest, embedder.parameters, index.vectors.nbytes, ranks, latency)


@dataclass
class MixedMaps:
    """An index searched with query vectors from another model that happen to be the same size."""

    index_model: str
    query_model: str
    refusal: str  # what `search` says when asked to mix them
    ranks: list[int]  # per query, with the check bypassed
    top: list[list[Hit]]  # per query, the top 3 with the check bypassed


def mixed_maps(index: Index, other: Embedder, query_set: QuerySet) -> MixedMaps:
    try:
        index.search(other, query_set.queries[0].text)
    except ModelMismatch as error:
        refusal = str(error)
    else:
        raise ValueError(f"{other.name} is the index's own model; there is nothing mixed to show")
    right = [{c.chunk_id for c in resolve(q.expect, index.chunks)} for q in query_set.queries]
    vectors = other.embed_queries([q.text for q in query_set.queries])
    everything = len(index.chunks)
    ranks = [first_right(index.rank(v, everything), ids) for v, ids in zip(vectors, right, strict=True)]
    top = [index.rank(v, 3) for v in vectors]
    return MixedMaps(index.manifest.model, other.name, refusal, ranks, top)


@dataclass
class ChunkStats:
    documents: int  # usable, unique documents
    sections: int
    chunks: int
    empty_sections: int  # headings with nothing under them before the next heading
    without_sections: list[str]  # usable documents with no headings, left out until chunking


def chunk_stats(docs: list[Document], chunks: list[Chunk]) -> ChunkStats:
    usable = [d for d in docs if not d.duplicate and d.parse_status in USABLE]
    sections = sum(len(d.sections) for d in usable)
    without = [d.source_id for d in usable if not d.sections]
    return ChunkStats(len(usable), sections, len(chunks), sections - len(chunks), without)


@dataclass
class Evaluation:
    query_set: QuerySet
    stats: ChunkStats
    ingest_version: str
    machine: str
    default_model: str
    probes: dict[str, Probe]
    comparisons: dict[str, Comparison]
    searches: list[tuple[Query, list[Hit]]]  # the default model's answer to one query of each kind
    mixed: MixedMaps | None


def machine() -> str:
    import torch

    chip = platform.processor() or platform.machine()
    if platform.system() == "Darwin":
        chip = (
            subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True
            ).stdout.strip()
            or chip
        )
    return f"{chip}, CPU only ({torch.get_num_threads()} threads), torch {torch.__version__}"


def evaluate(docs: list[Document], query_set: QuerySet, models: list[str]) -> Evaluation:
    """Build every model's index of whole sections in memory, then probe, compare and mix them. The
    indexes are not saved: `data/index/` holds what `embed` builds, with the current chunking."""
    chunks = chunk_documents(docs, Chunking())  # whole sections, the stage before chunking
    version = docs[0].provenance.ingest_version
    probe_chunks, right = probe_set(query_set, chunks)
    embedders = {name: load_embedder(name) for name in models}
    indexes: dict[str, Index] = {}
    probes: dict[str, Probe] = {}
    comparisons: dict[str, Comparison] = {}
    for name, embedder in embedders.items():
        indexes[name] = build(embedder, chunks, version)
        probes[name] = probe(embedder, query_set, probe_chunks, right)
        comparisons[name] = compare(embedder, indexes[name], query_set)
        print(f"{name}: {comparisons[name].hits(1)}/{len(query_set.queries)} right section first")

    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]
    first_of_kind: dict[str, Query] = {}
    for query in query_set.queries:
        first_of_kind.setdefault(query.kind, query)
    searches = [(q, indexes[default].search(embedders[default], q.text, k=5)) for q in first_of_kind.values()]
    same_size = [
        (a, b)
        for a in models
        for b in models
        if a != b and embedders[a].dimensions == embedders[b].dimensions
    ]
    mixed = mixed_maps(indexes[same_size[0][0]], embedders[same_size[0][1]], query_set) if same_size else None
    return Evaluation(
        query_set=query_set,
        stats=chunk_stats(docs, chunks),
        ingest_version=version,
        machine=machine(),
        default_model=default,
        probes=probes,
        comparisons=comparisons,
        searches=searches,
        mixed=mixed,
    )
