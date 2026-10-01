"""Compare chunking strategies on the labelled queries in `eval/queries.yaml`.

Every configuration is cut from the same documents and gets its own index from the same model, kept
in memory. A chunk answers a query when it overlaps a labelled section by at least half of the
smaller of the two: any piece of a split section, or a window that holds most of a short section.
For a query about one identifier the chunk must also contain the identifier, since a piece of a long
section that never mentions the key cannot answer a question about it.
"""

import statistics
from dataclasses import dataclass

from ..embed.embedders import Embedder
from ..embed.evaluate import QuerySet, Target
from ..embed.index import Hit, Index, build
from ..ingest.models import Document
from .chunks import DEFAULT_CHUNKING, Chunk, Chunking, Strategy, chunk_documents, has_text, usable
from .split import Span

K = 5  # the chunks a generator would read for one question
SIZES = (128, 256, 512)
GRID: tuple[Strategy, ...] = ("fixed", "sentences", "section-aware")
SMALL = 32  # tokens; a chunk this short says little on its own


def overlap(a: Span, b: Span) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]))


def union(spans: list[Span]) -> list[Span]:
    """The same characters, with overlapping spans merged."""
    merged: list[Span] = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


@dataclass
class Answer:
    """Where a query's answer sits: the spans of its labelled sections, and the text it must contain."""

    source: str
    spans: list[Span]
    contains: str | None = None

    def answered_by(self, chunk: Chunk) -> bool:
        if chunk.source_id != self.source or (self.contains and self.contains not in chunk.text):
            return False
        size = chunk.end - chunk.start
        return any(2 * overlap((chunk.start, chunk.end), s) >= min(size, s[1] - s[0]) for s in self.spans)

    def coverage(self, hits: list[Hit]) -> float:
        """Share of the labelled sections' text inside the chunks of `hits`."""
        found = union([(h.chunk.start, h.chunk.end) for h in hits if h.chunk.source_id == self.source])
        total = sum(end - start for start, end in self.spans)
        return sum(overlap(f, s) for f in found for s in self.spans) / total


def resolve_answer(target: Target, by_id: dict[str, Document], label: str) -> Answer:
    """Where a target's answer sits in its document. A title has to match exactly one section with text
    of its own, the same sections the embedding stage's chunks are. No sections at all means the whole
    document, for a file without headings, and then `contains` says which piece of it answers."""
    doc = by_id.get(target.source)
    if doc is None:
        raise ValueError(f"{label} names {target.source}, which is not a usable document")
    if not target.sections:
        if not target.contains:
            raise ValueError(f"{label} names the whole of {target.source}; say what the chunk has to contain")
        if target.contains not in doc.cleaned_text:
            raise ValueError(f"{label}: {target.source} never says `{target.contains}`")
        return Answer(target.source, [(0, len(doc.cleaned_text))], target.contains)
    spans: list[Span] = []
    for title in target.sections:
        wanted = [part.strip() for part in title.split("›")]
        matches = [s for s in doc.sections if s.path[-len(wanted) :] == wanted and has_text(doc, s)]
        if len(matches) != 1:
            raise ValueError(f"`{target.source} › {title}` matches {len(matches)} sections instead of one")
        spans.append((matches[0].start, matches[0].end))
    return Answer(target.source, spans, target.contains)


def answers(query_set: QuerySet, docs: list[Document]) -> list[Answer]:
    """Each query's answer, from the sections it names."""
    by_id = {d.source_id: d for d in usable(docs)}
    return [resolve_answer(q.expect, by_id, f"query {q.id}") for q in query_set.queries]


def sections_touched(chunk: Chunk, doc: Document) -> int:
    """How many sections with text of their own a chunk's span reaches into; 1 for a document without
    headings."""
    touched = [s for s in doc.sections if s.start < chunk.end and s.end > chunk.start and has_text(doc, s)]
    return max(1, len(touched))


@dataclass
class Shape:
    """What one configuration's chunks look like, in the model's tokens."""

    chunks: int
    documents: int
    median: float
    largest: int
    small: int  # chunks under SMALL tokens
    cut_off: int  # chunks longer than the model reads
    spread: float  # sections per chunk, on average: 1 when every chunk stays inside one section


def shape(chunks: list[Chunk], lengths: list[int], max_tokens: int, docs: list[Document]) -> Shape:
    by_id = {d.source_id: d for d in docs}
    return Shape(
        chunks=len(chunks),
        documents=len({c.source_id for c in chunks}),
        median=statistics.median(lengths),
        largest=max(lengths),
        small=sum(n < SMALL for n in lengths),
        cut_off=sum(n > max_tokens for n in lengths),
        spread=statistics.fmean(sections_touched(c, by_id[c.source_id]) for c in chunks),
    )


@dataclass
class Run:
    """One configuration's index against every labelled query."""

    chunking: Chunking
    index: Index
    shape: Shape
    lengths: dict[str, int]  # per chunk id, tokens the model reads, special tokens included
    ranks: list[int | None]  # per query, rank of the first chunk that answers it; None when none does
    coverage: list[float]  # per query, share of the labelled sections' text in the top K
    read: list[int]  # per query, tokens in the top K chunks
    top: list[list[Hit]]  # per query, the top K

    @property
    def chunks(self) -> list[Chunk]:
        return self.index.chunks

    def hits(self, k: int, only: list[int] | None = None) -> int:
        ranks = self.ranks if only is None else [self.ranks[q] for q in only]
        return sum(r is not None and r <= k for r in ranks)

    @property
    def mrr(self) -> float:
        return statistics.fmean(1 / r if r else 0.0 for r in self.ranks)

    @property
    def mean_coverage(self) -> float:
        return statistics.fmean(self.coverage)

    @property
    def median_read(self) -> float:
        return statistics.median(self.read)


def run(
    embedder: Embedder,
    docs: list[Document],
    query_set: QuerySet,
    found: list[Answer],
    chunking: Chunking,
    projects: dict[str, str],
) -> Run:
    chunks = chunk_documents(docs, chunking, embedder, projects)
    index = build(embedder, chunks, docs[0].provenance.ingest_version, chunking)
    counts = embedder.token_counts([c.embedded for c in chunks])
    lengths = dict(zip((c.chunk_id for c in chunks), counts, strict=True))
    if len(lengths) != len(chunks):
        raise ValueError(f"{chunking.label} gave two chunks the same id")
    vectors = embedder.embed_queries([q.text for q in query_set.queries])
    ranks: list[int | None] = []
    coverage: list[float] = []
    read: list[int] = []
    top: list[list[Hit]] = []
    for vector, answer in zip(vectors, found, strict=True):
        ranked = index.rank(vector, len(chunks))
        ranks.append(next((h.rank for h in ranked if answer.answered_by(h.chunk)), None))
        top.append(ranked[:K])
        coverage.append(answer.coverage(ranked[:K]))
        read.append(sum(lengths[h.chunk.chunk_id] for h in ranked[:K]))
    result = Run(
        chunking=chunking,
        index=index,
        shape=shape(chunks, counts, embedder.max_tokens, usable(docs)),
        lengths=lengths,
        ranks=ranks,
        coverage=coverage,
        read=read,
        top=top,
    )
    print(
        f"{chunking.label}: {len(chunks)} chunks, right chunk first for {result.hits(1)}, MRR {result.mrr:.2f}"
    )
    return result


def best(runs: list[Run]) -> Run:
    """The highest MRR; ties go to more of the answer in the top K, then to fewer tokens read."""
    return max(runs, key=lambda r: (round(r.mrr, 6), round(r.mean_coverage, 6), -r.median_read))


@dataclass
class Showcase:
    """The labelled section whose key sits deepest, to show where each strategy cuts it."""

    query: int  # position in the query set
    source: str
    span: Span
    key: str
    key_at: int  # tokens of the section before the key
    tokens: int  # tokens in the whole section, as the model would read it
    seen: bool  # whether the model reads as far as the key when the section is one chunk


def showcase(embedder: Embedder, docs: list[Document], found: list[Answer]) -> Showcase | None:
    by_id = {d.source_id: d for d in usable(docs)}
    deepest: Showcase | None = None
    for q, answer in enumerate(found):
        if not answer.contains:
            continue
        text = by_id[answer.source].cleaned_text
        for start, end in answer.spans:
            at = text.find(answer.contains, start, end)
            if at < 0:
                continue
            before, whole = embedder.token_counts([text[start:at], text[start:end]])
            key_at = before - embedder.special_tokens
            seen = (
                key_at + len(embedder.token_spans(answer.contains))
                <= embedder.max_tokens - embedder.special_tokens
            )
            if deepest is None or key_at > deepest.key_at:
                deepest = Showcase(q, answer.source, (start, end), answer.contains, key_at, whole, seen)
    return deepest


@dataclass
class Benchmark:
    query_set: QuerySet
    answers: list[Answer]
    docs: list[Document]  # the usable ones
    ingest_version: str
    model: str
    max_tokens: int
    baseline: Run  # whole sections: the embedding stage's chunks
    baseline_context: Run  # whole sections with the context line
    grid: list[Run]  # every strategy at every size, overlap an eighth of the size, without and with context
    best: Run  # the highest MRR of all
    pick: Run  # the highest MRR among configurations whose chunks each stay inside one section
    default: Run  # what `embed` builds
    no_overlap: Run  # the default without overlap
    showcase: Showcase | None

    @property
    def runs(self) -> list[Run]:
        return [self.baseline, self.baseline_context, *self.grid]


def benchmark(
    embedder: Embedder, docs: list[Document], query_set: QuerySet, projects: dict[str, str]
) -> Benchmark:
    found = answers(query_set, docs)
    runs: dict[str, Run] = {}

    def measure(chunking: Chunking) -> Run:
        if chunking.label not in runs:
            runs[chunking.label] = run(embedder, docs, query_set, found, chunking, projects)
        return runs[chunking.label]

    baseline, baseline_context = measure(Chunking()), measure(Chunking(context=True))
    grid = [
        measure(Chunking(strategy=strategy, size=size, overlap=size // 8, context=context))
        for context in (False, True)
        for strategy in GRID
        for size in SIZES
    ]
    everything = [baseline, baseline_context, *grid]
    return Benchmark(
        query_set=query_set,
        answers=found,
        docs=usable(docs),
        ingest_version=docs[0].provenance.ingest_version,
        model=embedder.name,
        max_tokens=embedder.max_tokens,
        baseline=baseline,
        baseline_context=baseline_context,
        grid=grid,
        best=best(everything),
        pick=best([r for r in everything if r.shape.spread == 1]),
        default=measure(DEFAULT_CHUNKING),
        no_overlap=measure(DEFAULT_CHUNKING.model_copy(update={"overlap": 0})),
        showcase=showcase(embedder, docs, found),
    )
