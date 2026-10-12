"""The configurations the latency report measures: the released pipeline, each optimization applied to
it alone, and the combination that becomes the next version. Each is a Retrieval; the generation side
is the same model and prompt for all of them."""

from ..retrieve.fusion import Fusion
from ..retrieve.retriever import Rerank, Retrieval

BASE = Retrieval(
    k=5,
    min_score=0.5,
    mode="hybrid",
    fusion=Fusion(strategy="weighted", alpha=0.5),
    rerank=Rerank(model="bge-base", candidates=20),
)


def with_rerank(
    candidates: int | None = None, model: str | None = None, min_relevance: float | None = None, **kw
) -> Retrieval:
    second = BASE.rerank
    rerank = Rerank(
        model=model or second.model,
        candidates=candidates or second.candidates,
        min_relevance=min_relevance if min_relevance is not None else second.min_relevance,
    )
    return BASE.model_copy(update={"rerank": rerank, **kw})


CONFIGS: dict[str, tuple[Retrieval, str]] = {
    "v2 (released)": (BASE, "hybrid, bge-base over 20 candidates, top 5, cosine floor 0.50"),
    "rerank 10": (with_rerank(candidates=10), "the cross-encoder reads 10 candidates instead of 20"),
    "rerank 5": (
        with_rerank(candidates=5),
        "the cross-encoder reads only the first stage's top 5: a reorder, not a rescue",
    ),
    "minilm-l6 over 20": (
        with_rerank(model="minilm-l6"),
        "the small cross-encoder, a twelfth of the parameters",
    ),
    "top 3": (BASE.model_copy(update={"k": 3}), "the model reads 3 passages instead of 5"),
    "relevance floor 0.10": (
        with_rerank(min_relevance=0.10),
        "passages the reranker scores under 0.10 are dropped: adaptive top-k",
    ),
    "v3: budget 1,200 tokens": (
        BASE.model_copy(update={"budget": 1200}),
        "the passages may add up to 1,200 tokens, in rank order; the first always goes in",
    ),
    "no rerank": (
        BASE.model_copy(update={"rerank": None}),
        "the hybrid first stage alone, the stage 05 pipeline",
    ),
    "first candidate: rerank 10 + floor 0.10 + top 4": (
        with_rerank(candidates=10, min_relevance=0.10, k=4),
        "designed before measuring: the three cuts that looked free, together",
    ),
    "minilm-l6 + top 3": (with_rerank(model="minilm-l6", k=3), "the small cross-encoder and 3 passages"),
    "minilm-l6 + top 4": (with_rerank(model="minilm-l6", k=4), "the small cross-encoder and 4 passages"),
    "rerank 10 + top 4": (with_rerank(candidates=10, k=4), "10 candidates and 4 passages"),
    "minilm-l6 over 10 + top 4": (
        with_rerank(model="minilm-l6", candidates=10, k=4),
        "the small cross-encoder over 10, 4 passages",
    ),
    "minilm-l6 + budget 1,200": (
        with_rerank(model="minilm-l6", budget=1200),
        "the small cross-encoder over 20 and the budget together",
    ),
}
