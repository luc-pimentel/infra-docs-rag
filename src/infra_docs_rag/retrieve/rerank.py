"""Reranking: a cross-encoder reads the question and each candidate chunk together and scores how well
the chunk answers it.

The first stage (cosine, BM25 or their fusion) scores a question and a chunk separately and compares
the two vectors, which is cheap enough to run over every chunk but blind to how the words interact. A
cross-encoder runs the question and the chunk through one model as a pair, so it can see that "undo"
and "rollout" belong together in this passage and not that one. It costs one model pass per pair, so
it reads only the top `candidates` of the first stage and reorders them. Scores are the model's
relevance after a sigmoid, in [0, 1], so a floor on them means the same for every reranker here.
"""

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..embed.index import Hit


@dataclass(frozen=True)
class RerankerSpec:
    name: str  # used on the command line
    model_id: str  # Hugging Face repository
    revision: str  # pinned commit, so the scores cannot change under a report
    note: str = ""


RERANKERS = {
    spec.name: spec
    for spec in (
        RerankerSpec(
            name="minilm-l6",
            model_id="cross-encoder/ms-marco-MiniLM-L-6-v2",
            revision="233902d25c440f23af6f7d6e94d2946bac0bee0a",
            note="The usual small cross-encoder, trained on MS MARCO passage ranking: 22M parameters, "
            "six layers, reads 512 tokens of question and passage together.",
        ),
        RerankerSpec(
            name="bge-base",
            model_id="BAAI/bge-reranker-base",
            revision="2cfc18c9415c912f9d8155881c133215df768a70",
            note="BGE's base reranker: 278M parameters, twelve layers, multilingual, trained for "
            "retrieval reranking. Twelve times the parameters of minilm-l6.",
        ),
    )
}
DEFAULT_RERANKER = "bge-base"


class Reranker(Protocol):
    name: str
    model_id: str
    revision: str
    parameters: int
    max_tokens: int  # question and passage together; longer pairs are cut off

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """One relevance in [0, 1] per text, for the question."""
        ...


class CrossEncoderReranker:
    def __init__(self, spec: RerankerSpec, device: str = "cpu") -> None:
        import torch
        from sentence_transformers import CrossEncoder  # slow import, paid only when a model is used

        options = {"revision": spec.revision, "device": device, "activation_fn": torch.nn.Sigmoid()}
        try:
            self.model = CrossEncoder(spec.model_id, local_files_only=True, **options)
        except OSError:  # not downloaded yet
            self.model = CrossEncoder(spec.model_id, **options)
        self.name = spec.name
        self.model_id = spec.model_id
        self.revision = spec.revision
        self.parameters = sum(p.numel() for p in self.model.model.parameters())
        self.max_tokens = self.model.max_length

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0, dtype=np.float32)
        scores = self.model.predict([(query, t) for t in texts], batch_size=16, show_progress_bar=False)
        return np.asarray(scores, dtype=np.float32).reshape(-1)


def load_reranker(name: str) -> CrossEncoderReranker:
    return CrossEncoderReranker(RERANKERS[name])


@dataclass
class RerankedHit(Hit):
    """A reranked hit: its relevance as the `score`, and the first-stage hit it came from, with that
    stage's rank and score intact."""

    first: Hit

    @property
    def first_rank(self) -> int:
        return self.first.rank


@dataclass
class Reranked:
    hits: list[RerankedHit]  # the top k, best first
    scored: int  # candidates the reranker read
    ms: float  # the reranker's time, scoring and sorting


def rerank(query: str, candidates: list[Hit], reranker: Reranker, k: int) -> Reranked:
    """Score every candidate against the question and keep the top k by relevance. Ties keep the first
    stage's order. A candidate's text is what the index embedded: the context line, then the text."""
    started = time.perf_counter()
    relevance = reranker.score(query, [h.chunk.embedded for h in candidates])
    order = sorted(range(len(candidates)), key=lambda i: (-float(relevance[i]), candidates[i].rank))
    hits = [
        RerankedHit(rank=n + 1, score=float(relevance[i]), chunk=candidates[i].chunk, first=candidates[i])
        for n, i in enumerate(order[:k])
    ]
    return Reranked(hits=hits, scored=len(candidates), ms=(time.perf_counter() - started) * 1000)
