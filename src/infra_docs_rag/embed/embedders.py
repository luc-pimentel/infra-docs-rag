"""Embedding models. Each one defines its own map, so vectors from two models are never compared.

A model is more than its name: the pinned revision, the instruction some models want in front of
queries, and the token limit past which text is cut off all shape the vectors it produces.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

# BGE's model card recommends this instruction in front of short queries that look for passages.
BGE_QUERY = "Represent this sentence for searching relevant passages: "


@dataclass(frozen=True)
class ModelSpec:
    name: str  # used on the command line and as the index folder name
    model_id: str  # Hugging Face repository
    revision: str  # pinned commit, so the map cannot change under an index built with it
    query_prefix: str = ""
    note: str = ""


MODELS = {
    spec.name: spec
    for spec in (
        ModelSpec(
            name="minilm",
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            revision="1110a243fdf4706b3f48f1d95db1a4f5529b4d41",
            note="The usual sentence-transformers baseline: small and fast, but it reads only the "
            "first 256 tokens of a chunk and was trained on sentence pairs, not queries against passages.",
        ),
        ModelSpec(
            name="bge-small",
            model_id="BAAI/bge-small-en-v1.5",
            revision="5c38ec7c405ec4b44b94cc5a9bb96e735b38267a",
            query_prefix=BGE_QUERY,
            note="Trained for retrieval, same vector size as MiniLM, reads 512 tokens. Queries need the "
            "model card's instruction in front; documents do not.",
        ),
        ModelSpec(
            name="bge-base",
            model_id="BAAI/bge-base-en-v1.5",
            revision="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a",
            query_prefix=BGE_QUERY,
            note="The same recipe as bge-small at three times the parameters and twice the vector size: "
            "more storage and compute per chunk and per query.",
        ),
    )
}
DEFAULT_MODEL = "bge-small"


class Embedder(Protocol):
    name: str
    model_id: str
    revision: str
    query_prefix: str
    dimensions: int
    max_tokens: int  # longer inputs are cut off before embedding
    special_tokens: int  # added around every input, such as [CLS] and [SEP]
    parameters: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def token_counts(self, texts: list[str]) -> list[int]: ...

    def token_spans(self, text: str) -> list[tuple[int, int]]: ...


class SentenceTransformerEmbedder:
    def __init__(self, spec: ModelSpec, device: str = "cpu") -> None:
        from sentence_transformers import SentenceTransformer  # slow import, paid only when a model is used

        options = {"revision": spec.revision, "device": device}
        try:
            self.model = SentenceTransformer(spec.model_id, local_files_only=True, **options)
        except OSError:  # not downloaded yet
            self.model = SentenceTransformer(spec.model_id, **options)
        self.name = spec.name
        self.model_id = spec.model_id
        self.revision = spec.revision
        self.query_prefix = spec.query_prefix
        self.dimensions = self.model.get_embedding_dimension()
        self.max_tokens = self.model.max_seq_length
        self.special_tokens = len(self.model.tokenizer("")["input_ids"])
        self.parameters = sum(p.numel() for p in self.model.parameters())

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prompt=None)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts, prompt=self.query_prefix or None)

    def _encode(self, texts: list[str], prompt: str | None) -> np.ndarray:
        vectors = self.model.encode(
            texts, prompt=prompt, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return np.asarray(vectors, dtype=np.float32)

    def token_counts(self, texts: list[str]) -> list[int]:
        encoded = self.model.tokenizer(texts, add_special_tokens=True, truncation=False, verbose=False)
        return [len(ids) for ids in encoded["input_ids"]]

    def token_spans(self, text: str) -> list[tuple[int, int]]:
        """Where each of the model's tokens sits in `text`, as character offsets."""
        encoded = self.model.tokenizer(
            text, add_special_tokens=False, return_offsets_mapping=True, truncation=False, verbose=False
        )
        return [(start, end) for start, end in encoded["offset_mapping"]]


def load_embedder(name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(MODELS[name])
