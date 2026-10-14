"""An answer: the retrieved passages, what the model wrote from them, and every citation mapped back to
the chunk it came from. Three outcomes: no passage qualified, so nothing was asked of the model; the
model declined because the passages do not cover the question; or an answer, with the share of its
text that carries a citation."""

import re
import time
from dataclasses import dataclass, field

from ..chunk.chunks import Chunk
from ..retrieve.retriever import DEFAULT_RETRIEVAL, Context, Retrieval, Retriever
from .generator import Generated, Generator
from .prompt import ABSTAIN

Outcome = str  # "no-passage" | "abstained" | "answered" | "refused" | "cut-off" | "error"

# A sentence that names what the passages do not cover, or what the answer cannot say from them: the
# prompt asks for it, it has nothing to cite, and it is not a claim about the subject.
GAP = re.compile(
    r"\b(passages?|documentation|docs?|excerpts?|text)\b[^.]{0,80}\b(do(es)? ?n[o']t|don't|doesn't|cannot|can't|never|aren't|isn't|is not|are not)\b"
    r"|\b(not|isn't|aren't) (covered|given|shown|stated|included|mentioned|described)\b"
    r"|\bI (can't|cannot|can only|could not) (say|tell|answer|confirm)\b",
    re.IGNORECASE,
)


def is_gap(sentence: str) -> bool:
    return GAP.search(sentence) is not None


@dataclass
class Citation:
    """A span of the answer and the chunk it rests on."""

    n: int  # the passage's number in the context, what the answer shows as [n]
    chunk: Chunk
    cited_text: str  # the passage text the API tied the span to
    span: str  # the answer text the citation is attached to


@dataclass
class Answer:
    question: str
    context: Context
    outcome: Outcome
    text: str = ""  # what the model wrote; empty when nothing was asked of it
    citations: list[Citation] = field(default_factory=list)
    uncited: list[str] = field(default_factory=list)  # answer segments with no citation, whitespace aside
    generated: Generated | None = None
    retrieve_ms: float = 0.0  # embedding, ranking and reranking, before any call to the model
    error: str | None = None  # what went wrong calling the model, when the outcome is "error"

    @property
    def abstained(self) -> bool:
        return self.outcome in ("no-passage", "abstained")

    @property
    def cited_chunks(self) -> list[Chunk]:
        """Each cited chunk once, in citation order."""
        seen: dict[str, Chunk] = {}
        for c in self.citations:
            seen.setdefault(c.chunk.chunk_id, c.chunk)
        return list(seen.values())

    @property
    def cited_share(self) -> float:
        """How much of the answer's text, whitespace aside, carries a citation; 0 when there is no text."""
        cited = sum(len(c.span.strip()) for c in self.citations if c.span.strip())
        total = cited + sum(len(u) for u in self.uncited)
        return cited / total if total else 0.0

    @property
    def gap_sentences(self) -> list[str]:
        """The uncited sentences that name what the passages do not cover."""
        return [u for u in self.uncited if is_gap(u)]

    @property
    def claim_share(self) -> float:
        """The cited share with the gap sentences left out: how much of what the answer claims about the
        subject carries a citation. 0 when there is no text."""
        cited = sum(len(c.span.strip()) for c in self.citations if c.span.strip())
        claims = sum(len(u) for u in self.uncited if not is_gap(u))
        total = cited + claims
        return cited / total if total else 0.0

    @property
    def reason(self) -> str:
        """Why there is no answer, for the reader."""
        ctx, r = self.context, self.context.retrieval
        if self.outcome == "no-passage":
            if ctx.candidates == 0:
                return f"no chunk matches {r.filter.label}"
            if ctx.top_score is None:
                return f"none of the top {r.k} chunks matches {r.filter.label}"
            if (
                r.rerank is not None
                and r.rerank.min_relevance is not None
                and (r.min_score is None or ctx.top_score >= r.min_score)
            ):
                return f"the reranker's best relevance was {ctx.top_relevance or 0:.2f}, under {r.rerank.min_relevance:.2f}"
            return f"the closest chunk scored {ctx.top_score:.2f}, under the {r.min_score:.2f} threshold"
        if self.outcome == "abstained":
            return "the model found no passage that answers the question"
        if self.outcome == "refused":
            return f"the model declined ({self.generated.refusal if self.generated else 'refusal'})"
        if self.outcome == "cut-off":
            return "the answer hit the token limit"
        if self.outcome == "error":
            return f"the call failed: {self.error}"
        return ""


def read(context: Context, generated: Generated) -> tuple[Outcome, list[Citation], list[str]]:
    """What the model's reply means: an abstention is the exact sentence at the start; a reply with no
    citation at all is treated as one too, since nothing in it rests on a passage."""
    citations: list[Citation] = []
    uncited: list[str] = []
    by_n = {s.n: s for s in context.sources}
    for seg in generated.segments:
        if seg.citations:
            for c in seg.citations:
                source = by_n.get(c.document)
                if source is not None:
                    citations.append(Citation(c.document, source.chunk, c.cited_text, seg.text))
        elif seg.text.strip():
            uncited.append(seg.text.strip())
    text = generated.text.strip()
    if generated.stop_reason == "refusal":
        return "refused", citations, uncited
    if generated.stop_reason == "max_tokens":
        return "cut-off", citations, uncited
    if text.startswith(ABSTAIN) or not citations:
        return "abstained", citations, uncited
    return "answered", citations, uncited


def answer(
    question: str,
    retriever: Retriever,
    generator: Generator,
    retrieval: Retrieval = DEFAULT_RETRIEVAL,
    retries: int | None = None,
) -> Answer:
    """Retrieve, and generate only when a passage qualified. A call that fails (the API or CLI raised)
    propagates, unless `retries` is given: then it is tried that many more times and, when it still
    fails, the answer's outcome is "error" with the reason, so one bad call does not stop a benchmark."""
    started = time.perf_counter()
    context = retriever.retrieve(question, retrieval)
    retrieve_ms = (time.perf_counter() - started) * 1000
    if context.empty:
        return Answer(question=question, context=context, outcome="no-passage", retrieve_ms=retrieve_ms)
    for attempt in range(1 + (retries or 0)):
        try:
            generated = generator.generate(context, question)
            break
        except ValueError as error:
            if retries is None:
                raise
            if attempt == retries:
                return Answer(
                    question=question,
                    context=context,
                    outcome="error",
                    retrieve_ms=retrieve_ms,
                    error=str(error)[:300],
                )
    outcome, citations, uncited = read(context, generated)
    return Answer(
        question=question,
        context=context,
        outcome=outcome,
        text=generated.text.strip(),
        citations=citations,
        uncited=uncited,
        generated=generated,
        retrieve_ms=retrieve_ms,
    )


def marked(a: Answer) -> str:
    """The answer with [n] after each cited span, the way a reader sees it."""
    if a.generated is None:
        return ""
    parts: list[str] = []
    for seg in a.generated.segments:
        parts.append(seg.text)
        numbers = sorted({c.document for c in seg.citations})
        if numbers:
            parts.append("".join(f"[{n}]" for n in numbers))
    return "".join(parts).strip()


def format_answer(a: Answer) -> str:
    """What `infra-docs-rag answer` prints: the answer with its markers, then the sources it cites, then
    what the call cost."""
    lines = []
    if a.outcome == "no-passage":
        lines.append(f"{ABSTAIN} No passage qualified: {a.reason}.")
    elif a.outcome in ("refused", "cut-off", "error"):
        lines.append(f"No answer: {a.reason}.")
    else:
        lines.append(marked(a))
    cited = a.cited_chunks
    if cited:
        lines += ["", "Sources"]
        for s in a.context.sources:
            if any(s.chunk.chunk_id == c.chunk_id for c in cited):
                lines.append(f"[{s.n}] {s.chunk.citation()}")
                lines.append(f"    {s.chunk.source_uri}")
    g = a.generated
    if g is not None:
        pct = f"{a.cited_share:.0%} of the text cited" if a.outcome == "answered" else a.outcome
        lines += [
            "",
            f"{g.model} · {pct} · {g.usage.input_tokens} in, {g.usage.output_tokens} out · {g.ms:,.0f} ms",
        ]
    return "\n".join(lines)
