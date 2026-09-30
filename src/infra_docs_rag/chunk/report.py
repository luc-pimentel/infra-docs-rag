"""`reports/chunking.md`: the evidence for the chunking stage, generated from a benchmark, plus what
`infra-docs-rag chunk` prints."""

import re
from pathlib import Path

from ..embed.report import block, format_hits, ordinal
from ..ingest.report import cell, compact_json, excerpt
from .chunks import DEFAULT_CHUNKING, Chunk, Chunking, has_text
from .evaluate import GRID, SIZES, SMALL, Benchmark, K, Run, Shape

DESCRIBE = {
    "whole-sections": "one chunk per section, however long: the embedding stage's chunks. Documents without "
    "headings are left out.",
    "fixed": "windows of whole words, the same number of tokens each, blind to sentences and headings.",
    "sentences": "whole sentences, and whole lines for lists, code and config, packed up to the size. Chunks run "
    "across headings.",
    "section-aware": "one chunk per section when it fits. A longer section is cut at paragraphs, then lines, "
    "sentences and words, and every piece keeps the section's path.",
}

FIELDS = [
    ("chunk_id", "string", "Source id and start offset; unique within an index."),
    ("source_id", "string", "The source in `sources.yaml`."),
    ("document_id", "string", "The ingestion record the chunk was cut from."),
    ("source_uri", "string", "Where the document was downloaded from."),
    ("project", "string or null", "The documentation set, from `sources.yaml`."),
    ("section_path", "list of strings", "Headings from the top of the document down to the chunk's section."),
    ("start, end", "integers", "The span of the document's cleaned text, end-exclusive."),
    ("pages", "list of integers", "PDF pages the span touches; empty for other formats."),
    ("text", "string", "Exactly `cleaned_text[start:end]`: what a hit shows and a generator reads."),
    ("context", "string", "The line embedded in front of the text; empty when the context line is off."),
]


def format_shapes(rows: list[tuple[Chunking, Shape]], max_tokens: int) -> str:
    """The table `infra-docs-rag chunk` prints."""
    lines = [
        f"{'chunking':32} {'chunks':>7} {'documents':>9} {'median':>7} {'largest':>8} "
        f"{'under ' + str(SMALL):>9} {'sections':>9} {'cut off':>8}"
    ]
    for chunking, s in rows:
        lines.append(
            f"{chunking.label:32} {s.chunks:>7} {s.documents:>9} {s.median:>7.0f} {s.largest:>8} "
            f"{s.small:>9} {s.spread:>9.2f} {s.cut_off:>8}"
        )
    lines.append(
        f"\nSizes are what the model reads: the text, the context line and the special tokens. It reads "
        f"{max_tokens}; longer chunks are cut off. `sections` is how many sections a chunk reaches into, on "
        "average."
    )
    return "\n".join(lines)


def format_chunks(chunks: list[tuple[Chunk, int]], snippet: int = 160) -> str:
    lines: list[str] = []
    for chunk, tokens in chunks:
        lines.append(f"- {chunk.citation()} [{chunk.start}:{chunk.end}], {tokens} tokens")
        if chunk.context:
            lines.append(f"  context: {chunk.context}")
        lines.append(f"  {excerpt(chunk.text, snippet)}")
    return "\n".join(lines) or "(no chunks)"


def rank(value: int | None) -> str:
    return "–" if value is None else str(value)


def placed(value: int | None) -> str:
    return "nowhere" if value is None else ordinal(value)


def percent(value: float) -> str:
    return f"{value:.0%}"


def listing(items: list[str]) -> str:
    """`a`, `a and b`, `a, b and c`."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def name(run: Run, bench: Benchmark) -> str:
    """A configuration's label, with what it is to the benchmark."""
    notes = []
    if run.chunking == DEFAULT_CHUNKING:
        notes.append("default")
    if run is bench.best:
        notes.append("highest MRR")
    return f"`{run.chunking.label}`" + (f" ({', '.join(notes)})" if notes else "")


def find(bench: Benchmark, chunking: Chunking) -> Run | None:
    everything = [*bench.runs, bench.default, bench.no_overlap]
    return next((r for r in everything if r.chunking == chunking), None)


def at_default(bench: Benchmark, strategy: str, size: int | None = None) -> Run | None:
    """A grid configuration at the default's size and context setting."""
    d = DEFAULT_CHUNKING
    size = size or d.size
    assert size is not None
    return find(bench, Chunking(strategy=strategy, size=size, overlap=size // 8, context=d.context))


def first_answer(run: Run, bench: Benchmark, q: int) -> Chunk | None:
    """The best-ranked chunk that answers query q in a run's index."""
    return next((h.chunk for h in run.top[q] if bench.answers[q].answered_by(h.chunk)), None) or next(
        (c for c in run.chunks if bench.answers[q].answered_by(c)), None
    )


def write_report(bench: Benchmark, output: Path) -> None:
    queries = bench.query_set.queries
    n = len(queries)
    base, default = bench.baseline, bench.default
    by_source = {doc.source_id: doc for doc in bench.docs}
    d = DEFAULT_CHUNKING
    one_step = 0.5 / n  # what one query dropping from first to second place does to MRR

    out: list[str] = []
    add = out.append
    add("# Chunking report\n")
    add(
        "Generated by `uv run infra-docs-rag evaluate-chunking` from `data/processed/documents.jsonl` (ingest "
        f"version `{bench.ingest_version}`) and the labelled queries in [`eval/queries.yaml`](../eval/queries.yaml). "
        f"Every index is built with `{bench.model}`, which reads {bench.max_tokens} tokens.\n"
    )

    # Summary
    add("## Summary\n")
    add("| | |\n|---|---|")
    add(
        "| Strategies | `whole-sections`, the embedding stage's chunks, against `fixed`, `sentences` and "
        "`section-aware` |"
    )
    add(
        f"| Grid | {len(GRID)} strategies × {', '.join(map(str, SIZES))} tokens × the context line off and on: "
        f"{len(bench.grid)} indexes, plus whole sections with and without the line. Overlap is an eighth of "
        "the size. |"
    )
    add(
        f"| Default | `{default.chunking.label}`: right chunk first for {default.hits(1)}/{n} queries, in the top "
        f"{K} for {default.hits(K)}/{n}, MRR {default.mrr:.2f}. Whole sections: {base.hits(1)}/{n}, "
        f"{base.hits(K)}/{n}, {base.mrr:.2f} |"
    )
    if bench.best is not default:
        add(
            f"| Highest MRR | `{bench.best.chunking.label}`, {bench.best.mrr:.2f}, with chunks that reach into "
            f"{bench.best.shape.spread:.1f} sections each on average |"
        )
    add(
        f"| Cut off | {default.shape.cut_off} default chunks run past the {bench.max_tokens} tokens the model "
        f"reads; {base.shape.cut_off} whole sections do |"
    )
    missing = sorted({doc.source_id for doc in bench.docs} - {c.source_id for c in base.chunks})
    add(
        f"| Documents | {default.shape.documents} chunked, against {base.shape.documents} for whole sections"
        + (f", which leave out {listing([f'`{m}`' for m in missing])} (no headings)" if missing else "")
        + " |"
    )
    add("")

    # 1. Strategies
    add("## 1. How each strategy cuts the corpus\n")
    add("Each strategy cuts every usable document's cleaned text:\n")
    for strategy, description in DESCRIBE.items():
        add(f"- **`{strategy}`**: {description}")
    add(
        f"\nA size caps what the model reads per chunk: the text plus the special tokens around it, and the "
        f"context line when it is on (section 3). Sizes are in `{bench.model}`'s tokens, from one pass of its "
        f"tokenizer over each document. *Sections per chunk* is how many sections a chunk reaches into, on "
        f"average: 1 means every chunk stays inside one section, so its citation names the section its text "
        f"came from.\n"
    )
    add(
        f"| Chunking | Chunks | Documents | Median tokens | Largest | Under {SMALL} tokens | Sections per chunk "
        "| Cut off |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in [base, *(r for r in bench.grid if not r.chunking.context)]:
        s = r.shape
        add(
            f"| `{r.chunking.label}` | {s.chunks} | {s.documents} | {s.median:.0f} | {s.largest} | {s.small} "
            f"| {s.spread:.2f} | {s.cut_off} |"
        )
    add("")

    sc = bench.showcase
    if sc:
        query = queries[sc.query]
        runs = [base, *(at_default(bench, s) for s in GRID)]
        add("### One section, cut every way\n")
        add(
            f'The `{cell(query.expect.sections[0])}` section of `{sc.source}` answers "{query.text}". The key '
            f"comes {sc.key_at} tokens into a section of {sc.tokens}"
            + (
                f", past the {bench.max_tokens} tokens `{bench.model}` reads, so as one whole chunk the model never "
                "sees it"
                if not sc.seen
                else ""
            )
            + ". Here is where each strategy cuts the section at the default's settings; offsets are characters "
            "from the start of the section, negative when a chunk starts in the section before.\n"
        )
        add(f"| Chunking | Piece | Characters | Tokens | Holds `{sc.key}` | Rank for the query |")
        add("|---|---:|---|---:|---|---:|")
        for r in runs:
            if r is None:
                continue
            pieces = sorted(
                (
                    c
                    for c in r.chunks
                    if c.source_id == sc.source and c.start < sc.span[1] and c.end > sc.span[0]
                ),
                key=lambda c: c.start,
            )
            for i, c in enumerate(pieces, 1):
                tokens = r.lengths[c.chunk_id]
                if sc.key not in c.text:
                    holds = "no"
                elif tokens <= bench.max_tokens:
                    holds = "**yes**"
                elif c.start == sc.span[0] and not sc.seen:
                    holds = f"yes, past the first {bench.max_tokens} tokens"
                else:
                    holds = "yes, in a chunk that is cut off"
                start, end = c.start - sc.span[0], c.end - sc.span[0]
                where = f"{start} to {end}".replace("-", "−")
                label = f"`{r.chunking.label}`" if i == 1 else ""
                ranked = rank(r.ranks[sc.query]) if i == 1 else ""
                add(f"| {label} | {i} | {where} | {tokens} | {holds} | {ranked} |")
        add("")

    # 2. Retrieval
    add(f"## 2. Retrieval: the same {n} queries against every index\n")
    add(
        "Each configuration gets its own index from the same model and answers the embedding stage's labelled "
        "queries. A chunk counts as the right one when it overlaps a labelled section by at least half of the "
        "smaller of the two: any piece of a split section, or a window that holds most of a short section. For "
        "a query about one identifier it must also contain the identifier (`contains` in the queries file), so "
        "a piece of a long section that never mentions the key does not count.\n"
    )
    add(
        f"*First*, *top {K}* and MRR are as in the [embeddings report](embeddings.md). *Answer in top {K}* is "
        f"the share of the labelled sections' text the top {K} chunks hold between them, on average: what a "
        f"generator would have to work with. *Tokens read* is the median size of the top {K} put together, "
        f"what it would cost to read them. With {n} queries, one query dropping from first to second place "
        f"moves MRR by {one_step:.2f}.\n"
    )
    add(
        f"| Chunking | Chunks | First | Top {K} | MRR | Answer in top {K} | Tokens read | Sections per chunk |"
    )
    add("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in bench.runs:
        add(
            f"| {name(r, bench)} | {r.shape.chunks} | {r.hits(1)}/{n} | {r.hits(K)}/{n} | {r.mrr:.2f} "
            f"| {percent(r.mean_coverage)} | {r.median_read:.0f} | {r.shape.spread:.2f} |"
        )
    add("")

    columns = [base, *(r for r in (at_default(bench, s) for s in GRID) if r is not None)]
    add(
        "Rank of the first right chunk, per query, for whole sections and for each strategy at the default's "
        "size and context setting (– when no chunk counts):\n"
    )
    add("| Query | Kind | " + " | ".join(f"`{r.chunking.label}`" for r in columns) + " |")
    add("|---|---|" + "---:|" * len(columns))
    for q, query in enumerate(queries):
        add(f"| `{query.id}` | {query.kind} | " + " | ".join(rank(r.ranks[q]) for r in columns) + " |")
    add("")

    add(f"### Pick: `{bench.pick.chunking.label}`\n")
    pick = (
        f"`embed` builds its index with the configuration that scores the highest MRR among those whose chunks "
        f"each stay inside one section. The answers this project will generate cite the document, page and "
        f"section they come from, and a chunk that runs across headings can only be filed under one of them. That "
        f"leaves whole sections and `section-aware`, and among them `{bench.pick.chunking.label}` scores "
        f"highest, at MRR {bench.pick.mrr:.2f}."
    )
    if bench.best.shape.spread > 1:
        b = bench.best
        pick += (
            f" The grid's highest, `{b.chunking.label}`, scores {b.mrr:.2f}: {b.mrr - bench.pick.mrr:.2f} more, "
            f"against {one_step:.2f} for one query moving down a place. Its chunks reach into "
            f"{b.shape.spread:.1f} sections each on average, and its top {K} cost {b.median_read:.0f} tokens to "
            f"read against {bench.pick.median_read:.0f}."
        )
    add(pick + "\n")
    if bench.pick.chunking != DEFAULT_CHUNKING:
        add(
            f"**This run picks `{bench.pick.chunking.label}`, but `DEFAULT_CHUNKING` is `{d.label}`: update "
            "the default in `src/infra_docs_rag/chunk/chunks.py`.**\n"
        )

    add("### Before and after\n")
    gains = sorted(
        (q for q in range(n) if base.ranks[q] != default.ranks[q]),
        key=lambda q: (
            (1 / default.ranks[q] if default.ranks[q] else 0) - (1 / base.ranks[q] if base.ranks[q] else 0)
        ),
        reverse=True,
    )
    shown = [q for q in gains if (default.ranks[q] or 10**9) < (base.ranks[q] or 10**9)][:2]
    worst = max(range(n), key=lambda q: default.ranks[q] or 10**9)
    if (default.ranks[worst] or 10**9) > 1 and worst not in shown:
        shown.append(worst)
    add(
        f"Whole sections against `{d.label}` for the queries that gained most, and the one the default still "
        "ranks lowest. Top 3 of each:\n"
    )
    for q in shown:
        query = queries[q]
        add(
            f'**"{query.text}"** ({query.kind}): whole sections put the right chunk {placed(base.ranks[q])}, '
            f"the default {placed(default.ranks[q])}.\n"
        )
        add(block("whole sections\n\n" + format_hits(base.top[q][:3], 110)))
        add(block(f"{d.label}\n\n" + format_hits(default.top[q][:3], 110)))

    add("### Size and overlap\n")
    context_word = "with" if d.context else "without"
    for strategy in GRID:
        sized = [at_default(bench, strategy, size) for size in SIZES]
        if any(r is None for r in sized):
            continue
        add(
            f"- **`{strategy}`** {context_word} the context line, at {', '.join(map(str, SIZES))} tokens: MRR "
            + ", ".join(f"{r.mrr:.2f}" for r in sized)
            + f"; answer in top {K} "
            + ", ".join(percent(r.mean_coverage) for r in sized)
            + "; tokens read "
            + ", ".join(f"{r.median_read:.0f}" for r in sized)
            + "."
        )
    winners = [
        strategy
        for strategy in GRID
        for context in (False, True)
        if max(
            (r for r in bench.grid if r.chunking.strategy == strategy and r.chunking.context == context),
            key=lambda r: r.mrr,
        ).chunking.size
        == max(SIZES)
    ]
    sizes = (
        f"The largest size scored the best MRR in {len(winners)} of {2 * len(GRID)} strategy and context "
        "settings. Bigger chunks hold more of each answer and give the model more words to match, and cost "
        "more to read."
    )
    cut = 0
    if d.strategy == "section-aware":
        pieces: dict[tuple[str, int], int] = {}
        for c in default.chunks:
            _, section = by_source[c.source_id].locate(c.start)
            key = (c.source_id, section.start if section else -1)
            pieces[key] = pieces.get(key, 0) + 1
        cut = sum(count > 1 for count in pieces.values())
        sizes += (
            f" At {d.size} tokens `section-aware` keeps {len(pieces) - cut} of {len(pieces)} sections whole (a "
            f"document without headings counts as one) and cuts the other {cut}, the ones too long to fit with "
            "their context line."
        )
    add("\n" + sizes + "\n")
    nov = bench.no_overlap
    same = f"{nov.mrr:.2f}" == f"{default.mrr:.2f}"
    add(
        "Without overlap the default scores "
        + (f"the same MRR, {nov.mrr:.2f}," if same else f"MRR {nov.mrr:.2f} instead of {default.mrr:.2f},")
        + f" with {percent(nov.mean_coverage)} of the answer in the top {K} instead of "
        f"{percent(default.mean_coverage)}. Overlap only changes the sections that get cut"
        + (f", {cut} here," if cut else "")
        + f" and costs {default.shape.chunks - nov.shape.chunks} extra chunks; it stays, since the text on "
        "both sides of each cut then sits together in one piece.\n"
    )

    # 3. Context
    add("## 3. The context line\n")
    add(
        "With the context line on, the project and section path go in front of each chunk's text when it is "
        "embedded, plus the pages for a PDF. The chunk's text itself does not change, and the line counts "
        "against the size. For example:\n"
    )
    example = (first_answer(default, bench, sc.query) if sc else None) or default.chunks[0]
    add(block(example.embedded[: len(example.context) + 200] + "…"))
    metadata = [q for q, query in enumerate(queries) if query.kind == "metadata"]
    add(
        "| Chunking | MRR without → with | First without → with | "
        + " | ".join(f"`{queries[q].id}` without → with" for q in metadata)
        + " |"
    )
    add("|---|---|---|" + "---|" * len(metadata))
    pairs = [(base, bench.baseline_context)]
    for strategy in GRID:
        for size in SIZES:
            off = find(bench, Chunking(strategy=strategy, size=size, overlap=size // 8))
            on = find(bench, Chunking(strategy=strategy, size=size, overlap=size // 8, context=True))
            if off and on:
                pairs.append((off, on))
    for off, on in pairs:
        label = off.chunking.label
        add(
            f"| `{label}` | {off.mrr:.2f} → {on.mrr:.2f} | {off.hits(1)} → {on.hits(1)} | "
            + " | ".join(f"{rank(off.ranks[q])} → {rank(on.ranks[q])}" for q in metadata)
            + " |"
        )
    rose = [off.chunking.label for off, on in pairs if on.mrr > off.mrr]
    add(
        f"\nThe line raised MRR in {len(rose)} of {len(pairs)} configurations. A piece cut from the middle of a "
        "section loses its heading; the line puts the heading back, with the project name, which the section's "
        "own text may never mention. A fixed window or a run of sentences is filed under the section it starts "
        "in, so its line can name the wrong section for the rest of the window.\n"
    )
    for q in metadata:
        query = queries[q]
        chunk = first_answer(default, bench, q)
        with_line = [r for r in bench.runs if r.chunking.context]
        top = min(with_line, key=lambda r: r.ranks[q] or 10**9)
        bullet = (
            f'- **"{query.text}"**: the default puts the right chunk {placed(default.ranks[q])}'
            + (f", with the line `{chunk.context}`" if chunk and chunk.context else "")
            + f". The best any configuration with the line does is {placed(top.ranks[q])} (`{top.chunking.label}`)."
        )
        window = first_answer(top, bench, q)
        if window and top.shape.spread > 1:
            doc = by_source[window.source_id]
            titles = [
                f"`{sec.title}`"
                for sec in doc.sections
                if sec.start < window.end and sec.end > window.start and has_text(doc, sec)
            ]
            if len(titles) > 1:
                bullet += (
                    f" Its right chunk is a window over {listing(titles)}, so it carries the words of the sections "
                    "around the answer as well."
                )
        add(bullet)
    for q in metadata:
        if all((r.ranks[q] or 10**9) > K for r in bench.runs):
            chunk = first_answer(default, bench, q)
            add(
                f'\nNo configuration puts "{queries[q].text}" in the top {K}, with the line or without it.'
                + (
                    " The page it asks about is in the chunk records (section 4), where a filter can match it at "
                    "retrieval time; an embedding of the text, even with the page in its context line, did not."
                    if chunk and chunk.pages
                    else ""
                )
            )
    add("")

    # 4. Schema and records
    add("## 4. Chunk schema and five records\n")
    mismatched = sum(c.text != by_source[c.source_id].cleaned_text[c.start : c.end] for c in default.chunks)
    add(
        "Every chunk carries where it came from, down to the character, so a hit can be cited and a miss "
        "can be traced. "
        + (
            f"All {len(default.chunks)} chunks of the default index are exactly `cleaned_text[start:end]` of "
            "their document."
            if not mismatched
            else f"{mismatched} of the default index's {len(default.chunks)} chunks differ from "
            "`cleaned_text[start:end]` of their document."
        )
        + "\n"
    )
    add("| Field | Type | Meaning |\n|---|---|---|")
    for field, kind, meaning in FIELDS:
        add(f"| `{field}` | {kind} | {meaning} |")
    add("")

    records: list[Chunk] = []

    def keep(chunk: Chunk | None) -> None:
        if chunk and chunk.chunk_id not in {c.chunk_id for c in records}:
            records.append(chunk)

    paraphrase = next((q for q, query in enumerate(queries) if query.kind == "paraphrase"), None)
    if paraphrase is not None:
        keep(first_answer(default, bench, paraphrase))
    if sc:
        keep(first_answer(default, bench, sc.query))
    for q in metadata:
        keep(first_answer(default, bench, q))
    headless = next((doc.source_id for doc in bench.docs if not doc.sections), None)
    keep(next((c for c in default.chunks if c.source_id == headless), None))
    for q in range(n):
        if len(records) >= 5:
            break
        keep(first_answer(default, bench, q))
    add(
        f"Five records from the default index: the chunk that answers a paraphrase, a piece of the "
        f"`{cell(queries[sc.query].expect.sections[0]) if sc else 'long'}` section, the chunks that answer the "
        "metadata queries, and the first chunk of a document without headings. Text is shortened here.\n"
    )
    for chunk in records[:5]:
        record = chunk.model_dump()
        if len(chunk.text) > 240:
            record["text"] = chunk.text[:240].rstrip() + "…"
        add(block(compact_json(record), "json"))

    indexes = len({r.chunking.label for r in [*bench.runs, bench.default, bench.no_overlap]})
    add("## Reproduce\n")
    add(
        "```sh\n"
        "uv run infra-docs-rag chunk                 # cut every document with each strategy into data/chunks/\n"
        f"uv run infra-docs-rag embed                 # index the default: {d.label}\n"
        f'uv run infra-docs-rag search "{queries[0].text}"\n'
        f"uv run infra-docs-rag evaluate-chunking     # rebuild all {indexes} indexes, rewrite this report\n"
        "```"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(re.sub(r"\n{3,}", "\n\n", "\n".join(out)) + "\n")
