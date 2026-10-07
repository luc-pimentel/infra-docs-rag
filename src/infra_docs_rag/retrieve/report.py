"""`reports/retrieval.md`: the evidence for the retrieval stage, generated from an evaluation, plus what
`infra-docs-rag retrieve` prints."""

import re
import statistics
from pathlib import Path

from ..chunk.evaluate import Answer, K
from ..chunk.report import listing, placed, rank
from ..embed.index import Hit, Manifest
from ..embed.report import block, format_hits
from ..ingest.report import cell, excerpt
from .evaluate import K_GRID, KINDS, BenchmarkQuery, Evaluation, Filtered
from .fusion import HybridHit
from .rerank import RerankedHit
from .retriever import Context, Filter

# What each kind of question tests; true whatever the scores come out as.
TESTS = {
    "paraphrase": "the question and its answer share a meaning but few words. Dense retrieval's home ground.",
    "identifier": "a config key, flag or acronym. The right chunk has to contain it (`contains`), so a piece of a "
    "long section that never mentions the key does not count.",
    "metadata": "the answer depends on something the text does not say: which project, which document, which page. "
    "Each one carries the filter its wording implies.",
    "out-of-scope": "nothing in the corpus answers it. The right result is no passage, which only a threshold "
    "can produce: a ranking always has a top chunk.",
}


def count(n: int, noun: str) -> str:
    return f"{n} {noun}" + ("" if n == 1 else "s")


def flags(criteria: Filter) -> str:
    """The command line flags for a filter: `--project "Argo CD" --page 9`."""
    parts = []
    for name, value in criteria.model_dump().items():
        if value is not None:
            text = str(value)
            parts.append(f"--{name} " + (f'"{text}"' if " " in text or "›" in text else text))
    return " ".join(parts)


def format_context(manifest: Manifest, context: Context, snippet: int | None = None) -> str:
    """What `infra-docs-rag retrieve` prints: the passages a generator reads, then where each came from."""
    r = context.retrieval
    lines = [f"{manifest.model_id} · {manifest.chunks} chunks · {r.label}"]
    parts = []
    if not r.filter.empty:
        parts.append(count(context.candidates, "chunk") + " eligible")
    if context.sources:
        parts.append(f"{count(len(context.sources), 'passage')}, {context.tokens} tokens")
    if context.below:
        parts.append(f"{context.below} under the threshold")
    if context.over_budget:
        parts.append(f"{context.over_budget} over the budget")
    if parts:
        lines.append(" · ".join(parts))
    if context.empty:
        if context.candidates == 0:
            reason = f"no chunk matches {r.filter.label}"
        elif context.top_score is None:
            reason = f"none of the top {r.k} chunks matches {r.filter.label}"
        elif (
            r.rerank is not None
            and r.rerank.min_relevance is not None
            and (r.min_score is None or context.top_score >= r.min_score)
        ):
            reason = (
                f"the reranker's best relevance was {context.top_relevance or 0:.2f}, under the "
                f"{r.rerank.min_relevance:.2f} floor, so the candidates probably do not answer this question"
            )
        else:
            reason = (
                f"the closest chunk scored {context.top_score:.2f}, under the {r.min_score:.2f} threshold, so "
                "the indexed documentation probably does not cover this question"
            )
        return "\n".join([*lines, "", f"No passage: {reason}."])
    lines.append("")
    for s in context.sources:
        lines += [s.heading, excerpt(s.chunk.text, snippet) if snippet else s.chunk.text, ""]
    lines.append("Sources")
    for s in context.sources:
        lines.append(f"[{s.n}] {s.hit.score:.3f}  {s.chunk.citation()} · {s.tokens} tokens{sides(s.hit)}")
        lines.append(f"    {s.chunk.source_uri}")
    return "\n".join(lines)


def sides(hit: Hit) -> str:
    """What each side of a hybrid hit said: ` · cosine 0.62 (3rd) · bm25 7.1 (1st)`; nothing for a plain hit.
    A reranked hit says where the first stage had it, then what that stage's sides said."""
    if isinstance(hit, RerankedHit):
        return f" · first stage #{hit.first_rank} at {hit.first.score:.3f}{sides(hit.first)}"
    if not isinstance(hit, HybridHit):
        return ""
    lexical = f"bm25 {hit.lexical:.1f} (#{hit.lexical_rank})" if hit.lexical_rank else "bm25 0"
    return f" · cosine {hit.dense:.3f} (#{hit.dense_rank}) · {lexical}"


def format_hybrid_search(manifest: Manifest, hits: list[Hit], label: str, snippet: int = 150) -> str:
    """What `infra-docs-rag search --mode hybrid` prints: the fused ranking, each hit with both sides."""
    lines = [f"{manifest.model_id} · {manifest.chunks} chunks · {label}", ""]
    for hit in hits:
        body = hit.chunk.text.removeprefix(hit.chunk.section).strip()
        lines += [
            f"{hit.rank}. {hit.score:.4f}  {hit.chunk.citation()}{sides(hit)}",
            f"   {hit.chunk.source_uri}",
        ]
        lines += [f"   {excerpt(body, snippet)}", ""]
    return "\n".join(lines).rstrip() or "(no chunk matches)"


def answered_by(ev: Evaluation, query_id: str) -> str:
    """`k8s-service › Headless Services` and 2 more, or what a whole-document label needs."""
    return answered_by_label(ev.answers[query_id], next(q for q in ev.benchmark.queries if q.id == query_id))


def answered_by_label(answer: Answer, query: BenchmarkQuery) -> str:
    assert query.expect is not None
    if not query.expect.sections:
        return f"`{answer.source}`, the piece that says `{cell(answer.contains)}`"
    first = query.expect.sections[0].split("›")[-1].strip()
    more = len(query.expect.sections) - 1
    return f"`{answer.source} › {cell(first)}`" + (f" and {more} more" if more else "")


def write_report(ev: Evaluation, output: Path) -> None:
    bench = ev.benchmark
    queries = bench.queries
    n_in, n_out = len(ev.ranked), len(ev.unanswerable)
    n = n_in + n_out
    m = ev.manifest
    d = ev.default
    plain = dict(zip((q.id for q in queries), ev.plain, strict=True))
    with_filters = dict(zip((q.id for q in queries if q.filter is not None), ev.with_filters, strict=True))
    in_scope = bench.in_scope
    low, high = K_GRID[0], K_GRID[-1]

    out: list[str] = []
    add = out.append
    add("# Retrieval report\n")
    add(
        f"Generated by `uv run infra-docs-rag evaluate-retrieval` from the `{m.model}` index that `embed` built "
        f"(`{m.chunking.label}`, {m.chunks} chunks, ingest version `{m.ingest_version}`) and the labelled "
        "queries in [`eval/retrieval.yaml`](../eval/retrieval.yaml). Timings are from one run on "
        f"{ev.machine}.\n"
    )

    # Summary
    kinds = {kind: sum(q.kind == kind for q in queries) for kind in KINDS}
    answered = sum(ev.answered(q, plain[q.id]) for q in in_scope)
    silent = sum(plain[q.id].empty for q in in_scope)
    rejected = sum(plain[q.id].empty for q in bench.out_of_scope)
    read = statistics.median(plain[q.id].tokens for q in queries)
    add("## Summary\n")
    add("| | |\n|---|---|")
    add(
        f"| Index | `{m.model}`, {m.chunks} chunks cut `{m.chunking.label}`, cosine similarity, exact search "
        "over every chunk |"
    )
    add(
        f"| Queries | {n}: "
        + " · ".join(f"{count} {kind}" for kind, count in kinds.items())
        + f". {n_in} name the passages that answer them; {n_out} have no answer in the corpus |"
    )
    add(
        f"| Ranking | Right chunk first for {ev.hits(1)}/{n_in} queries, in the top {K} for {ev.hits(K)}/{n_in}, "
        f"MRR {ev.mrr:.2f} |"
    )
    add(
        f"| Default | `{d.label}`: a right passage for {answered}/{n_in} questions"
        + (f", {silent} left with no passage" if silent else "")
        + f", no passage for {rejected}/{n_out} unanswerable questions, {read:.0f} tokens read per question "
        "(median) |"
    )
    if ev.pick != d:
        add(f"| Benchmark's pick | `{ev.pick.label}` (section 4) |")
    if ev.filtered:
        moved = [
            f
            for f in ev.filtered
            if f.prefiltered is not None and (f.unfiltered is None or f.prefiltered < f.unfiltered)
        ]
        lost = sum(f.postfiltered is None for f in ev.filtered)
        add(
            f"| Filters | The metadata queries' own filters move the right chunk up for {len(moved)} of "
            f"{len(ev.filtered)}; applying the same filters after ranking leaves {lost} of them with nothing |"
        )
    small, big = ev.scale[0], ev.scale[-1]
    add(
        f"| Flat index | Scoring and sorting {small.chunks:,} vectors takes {small.score_ms:.2f} ms, "
        f"{big.chunks:,} take {big.score_ms:.0f} ms and {big.megabytes / 1e3:.1f} GB; embedding the query takes "
        f"{ev.embed_ms:.0f} ms (section 6) |"
    )
    add("")

    # 1. Benchmark
    add("## 1. The benchmark set\n")
    add(
        f"{n} questions in [`eval/retrieval.yaml`](../eval/retrieval.yaml)"
        + (
            f", {len(ev.earlier)} of them the embedding stage's queries unchanged, so the numbers stay comparable "
            "with the earlier reports"
            if ev.earlier
            else ""
        )
        + ". Each in-scope question names the passages that answer it: a source id and the titles of its "
        "sections, resolved to character spans of the document. A chunk counts as right when it overlaps a "
        "labelled section by at least half of the smaller of the two, the chunking benchmark's rule, and for "
        "a question about one identifier it also has to contain it. A document without headings is labelled "
        "whole, with the text its right piece has to contain. Every hypothesis was written before the query "
        "was scored.\n"
    )
    add("Each kind of question tests one thing:\n")
    for kind in KINDS:
        add(f"- **{kind}** ({kinds[kind]}): {TESTS[kind]}")
    add("")
    add("| Query | Kind | Question | Answered by | Filter |")
    add("|---|---|---|---|---|")
    for q in queries:
        where = answered_by(ev, q.id) if q.expect else "nothing in the corpus"
        add(f"| `{q.id}` | {q.kind} | {cell(q.text)} | {where} | {q.filter.label if q.filter else ''} |")
    add("")

    # 2. Retriever
    add("## 2. The retriever\n")
    add(
        '`uv run infra-docs-rag retrieve "<question>"` embeds the question with the index\'s own model, keeps '
        "the chunks that match the filter, ranks them, drops any that score under the threshold, and packs "
        f"the top {d.k} into the passages a generator reads: in rank order, each under a numbered heading that "
        "names its project, section and pages, with a source list after them. An answer cites `[2]`; the list "
        "says what `[2]` is. `search` shows the same ranking as raw hits, scores included, and marks the ones "
        "the threshold would drop.\n"
    )
    first_paraphrase = next(q for q in queries if q.kind == "paraphrase")
    add(
        block(
            f'$ uv run infra-docs-rag retrieve "{first_paraphrase.text}"\n'
            + format_context(m, plain[first_paraphrase.id], 160)
        )
    )
    paragraph = (
        "Filters narrow the eligible chunks before ranking, so the top k is the top k of what qualifies. "
        "Filtering after ranking instead (`--post-filter`) keeps only what survives of the unfiltered top k, "
        "which can be nothing (section 5). A question whose wording names a project, a document or a page "
        "carries that as a filter."
    )
    with_filter = [q for q in queries if q.filter is not None and q.id in with_filters]
    helped = next(
        (q for q in with_filter if ev.answered(q, with_filters[q.id]) and not ev.answered(q, plain[q.id])),
        None,
    )
    example = helped or (with_filter[0] if with_filter else None)
    if example is not None:
        assert example.filter is not None
        f = next((f for f in ev.filtered if f.query.id == example.id), None)
        if f is not None:
            paragraph += (
                f' For "{example.text}" the right chunk ranks {placed(f.unfiltered)} of {m.chunks} without the '
                f"filter and {placed(f.prefiltered)} of {count(f.candidates, 'eligible chunk')} with it:"
            )
        add(paragraph + "\n")
        add(
            block(
                f'$ uv run infra-docs-rag retrieve "{example.text}" {flags(example.filter)}\n'
                + format_context(m, with_filters[example.id], 160)
            )
        )
    else:
        add(paragraph + "\n")
    refused = next((q for q in bench.out_of_scope if plain[q.id].empty), None)
    passed = next((q for q in bench.out_of_scope if not plain[q.id].empty), None)
    if refused is not None:
        add(
            "A question the corpus does not cover still has a closest chunk. Under the threshold, the retriever "
            "returns nothing for it:\n"
        )
        add(
            block(
                f'$ uv run infra-docs-rag retrieve "{refused.text}"\n'
                + format_context(m, plain[refused.id], 120)
            )
        )
        if passed is not None:
            add(
                f'"{passed.text}" is just as unanswerable, but its closest chunk scores '
                f"{plain[passed.id].top_score:.2f}, above the threshold, so it gets passages like any other "
                "question. Section 4 is about where that line can go.\n"
            )
    elif passed is not None:
        add(
            "A question the corpus does not cover still has a closest chunk, and without a threshold the "
            "retriever hands it over as if it were an answer:\n"
        )
        add(
            block(
                f'$ uv run infra-docs-rag retrieve "{passed.text}"\n'
                + format_context(m, plain[passed.id], 120)
            )
        )

    # 3. k
    add("## 3. How many chunks: k\n")
    add(
        f"One full ranking per question gives every cutoff at once. *In the top k* counts the {n_in} in-scope "
        "questions whose first right chunk ranks there; *tokens read* is the median size of the top k "
        "passages put together, what a generator would have to read. MRR does not depend on k: "
        f"{ev.mrr:.2f} here, where one question dropping from first to second place moves it by "
        f"{0.5 / n_in:.2f}.\n"
    )
    labelled_kinds = [kind for kind in KINDS if kind != "out-of-scope" and kinds[kind]]
    add(
        "| k | In the top k | "
        + " | ".join(f"{kind} ({len(ev.of_kind(kind))})" for kind in labelled_kinds)
        + " | Tokens read |"
    )
    add("|---:|---:|" + "---:|" * len(labelled_kinds) + "---:|")
    for k in K_GRID:
        add(
            f"| {k} | {ev.hits(k)}/{n_in} | "
            + " | ".join(str(ev.hits(k, ev.of_kind(kind))) for kind in labelled_kinds)
            + f" | {ev.read(k):.0f} |"
        )
    add("")
    add(
        f"Going from the top {low} to the top {K} finds {ev.hits(K) - ev.hits(low)} more questions' answers "
        f"for {ev.read(K) - ev.read(low):.0f} more tokens per question; going on to the top {high} finds "
        f"{ev.hits(high) - ev.hits(K)} more for another {ev.read(high) - ev.read(K):.0f} tokens. "
        + (
            f"The {n_in - ev.hits(high)} questions still missing at {high} are listed in section 7."
            if ev.hits(high) < n_in
            else f"Every answer is in the top {high}."
        )
        + "\n"
    )

    # 4. Threshold
    add("## 4. How similar is similar enough: the threshold\n")
    right_scores = [r.right_score for r in ev.ranked if r.rank is not None and r.rank <= K]
    oos_scores = [u.top_score for u in ev.unanswerable]
    add(
        "Cosine similarity is a ranking signal, not a probability, but a threshold is still the only way to "
        "return nothing. The question is whether the scores of right chunks and the best scores of unanswerable "
        f"questions can be told apart. Right chunks in the top {K} score between {min(right_scores):.2f} and "
        f"{max(right_scores):.2f} (median {statistics.median(right_scores):.2f}); the best chunk for an "
        f"unanswerable question scores between {min(oos_scores):.2f} and {max(oos_scores):.2f} (median "
        f"{statistics.median(oos_scores):.2f}). "
        + (
            f"{sum(s >= min(right_scores) for s in oos_scores)} of the {n_out} unanswerable questions score above "
            "the lowest right chunk, so no threshold separates them perfectly."
            if any(s >= min(right_scores) for s in oos_scores)
            else "Every unanswerable question scores under the lowest right chunk, so a threshold between the "
            "two separates them."
        )
        + "\n"
    )
    add("| Unanswerable question | Closest chunk | Score |")
    add("|---|---|---:|")
    for u in sorted(ev.unanswerable, key=lambda u: -u.top_score):
        add(f"| {cell(u.query.text)} | `{cell(u.hits[0].chunk.citation())}` | {u.top_score:.2f} |")
    add("")
    add(
        f"Each threshold at k = {K}: *answers kept* counts the in-scope questions whose first right chunk is in "
        f"the top {K} and passes; *left silent* the in-scope questions where nothing passes; *rejected* the "
        f"unanswerable questions where nothing passes; *shown* the median passages per question, over all {n}.\n"
    )
    add("| Threshold | Answers kept | Left silent | Rejected | Shown |")
    add("|---|---:|---:|---:|---:|")
    none = ev.thresholds[0]
    for row in ev.thresholds:
        label = "none" if row.min_score is None else f"{row.min_score:.2f}"
        mark = " ◂" if row.min_score == ev.pick.min_score else ""
        add(
            f"| {label}{mark} | {row.answered}/{none.answered} | {row.silent} | {row.rejected}/{n_out} | {row.shown:.0f} |"
        )
    add("")
    pick = ev.pick
    picked = next(r for r in ev.thresholds if r.min_score == pick.min_score)
    if pick.min_score is None:
        add(
            "The pick is no threshold: every value that rejects an unanswerable question costs at least as many "
            "real answers.\n"
        )
    else:
        lost = [
            r
            for r in ev.ranked
            if r.rank is not None
            and r.rank <= K
            and r.right_score is not None
            and r.right_score < pick.min_score
        ]
        add(
            f"The pick is {pick.min_score:.2f}: the most answers kept plus unanswerable questions rejected, with "
            f"ties going to fewer real questions left silent and then to fewer passages shown. It rejects "
            f"{picked.rejected} of the {n_out} unanswerable questions and keeps {picked.answered} of the "
            f"{none.answered} answers"
            + (
                ", losing " + listing([f'"{r.query.text}" ({r.right_score:.2f})' for r in lost])
                if lost
                else ""
            )
            + f". {n_out} unanswerable questions are a small sample: this threshold is a first estimate, to be "
            "revisited when the evaluation stage adds abstention cases.\n"
        )
        through = [u for u in ev.unanswerable if u.top_score >= pick.min_score]
        if through:
            nearest = max(through, key=lambda u: u.top_score)
            add(
                f"The {count(len(through), 'question')} it lets through score between "
                f"{min(u.top_score for u in through):.2f} and {max(u.top_score for u in through):.2f}, inside the "
                f'range of right answers. The closest chunk to "{nearest.query.text}" is '
                f"`{cell(nearest.hits[0].chunk.citation())}`: the same subject, and cosine similarity cannot tell a "
                "question a section almost answers from one it does. Rejecting those needs a signal that reads the "
                "question against the chunk, which is a reranker's job, or the answer against the passages, which "
                "is the generation stage's.\n"
            )

    # 5. Filters
    if ev.filtered:
        add("## 5. Filters: before and after ranking\n")
        add(
            "A metadata question's filter is the project, document or page its wording names. Applied before "
            "ranking, the ranking runs over the eligible chunks only, however few; applied after, it keeps "
            f"what survives of the unfiltered top {K}. The rank of the first right chunk each way (– when no "
            "chunk counts):\n"
        )
        add("| Query | Filter | Eligible chunks | No filter | Filter before ranking | Filter after ranking |")
        add("|---|---|---:|---:|---:|---:|")
        for f in ev.filtered:
            assert f.query.filter is not None
            add(
                f"| `{f.query.id}` | {f.query.filter.label} | {f.candidates} | {rank(f.unfiltered)} | "
                f"{rank(f.prefiltered)} | {rank(f.postfiltered)} |"
            )
        add("")
        lost_all = [f for f in ev.filtered if f.postfiltered is None]
        add(
            f"Filtering before ranking puts the right chunk in the top {K} for "
            f"{sum(f.prefiltered is not None and f.prefiltered <= K for f in ev.filtered)} of the {len(ev.filtered)} "
            f"metadata questions, against {sum(f.unfiltered is not None and f.unfiltered <= K for f in ev.filtered)} "
            f"with no filter. Filtering after ranking leaves {len(lost_all)} with nothing: the right chunk was not in "
            f"the unfiltered top {K}, so there was nothing left to filter. "
            + (
                "This is the whole case for filtering first, and the reason a filter has to be part of the "
                "index's search rather than a step after it."
                if lost_all
                else f"At this size the two agree, but only because every right chunk was already in the top {K}."
            )
            + "\n"
        )
        unmoved = [
            f
            for f in ev.filtered
            if f.prefiltered is not None and f.prefiltered == f.unfiltered and f.prefiltered > K
        ]
        if unmoved:
            add(
                "A filter only helps when the chunks it removes are the ones in the way. For "
                + listing([f"`{f.query.id}`" for f in unmoved])
                + " the chunks ahead of the right one come from the same project or document, so the filter "
                "leaves the ranking as it was.\n"
            )
        paged: Filtered | None = next(
            (f for f in ev.filtered if f.query.filter and f.query.filter.page), None
        )
        if paged is not None:
            assert paged.query.filter is not None
            add(
                f'"{paged.query.text}" with `{flags(paged.query.filter)}`: {count(paged.candidates, "chunk")} '
                f"{'is' if paged.candidates == 1 else 'are'} eligible, and the right one ranks "
                f"{placed(paged.prefiltered)} among them against {placed(paged.unfiltered)} among all {m.chunks}. "
                "The text never says which page it is on; the chunk record does.\n"
            )
            add(block(format_hits(paged.top, 110)))

    # 6. Scale
    add("## 6. Index design: when exact search stops being enough\n")
    add(
        f"The index is flat: every query is scored against every chunk ({m.dimensions} multiplications each) "
        "and the scores are sorted. That is exact, and at this size it is nearly free next to embedding the "
        f"query, which takes {ev.embed_ms:.0f} ms. Random unit vectors of the same size show where it stops being "
        "free:\n"
    )
    add("| Vectors | Memory | Score and sort |")
    add("|---:|---:|---:|")
    for point in ev.scale:
        memory = f"{point.megabytes / 1e3:.1f} GB" if point.megabytes >= 1e3 else f"{point.megabytes:.0f} MB"
        add(f"| {point.chunks:,} | {memory} | {point.score_ms:.2f} ms |")
    add("")
    over = next((p for p in ev.scale if p.score_ms > ev.embed_ms), None)
    add(
        (
            f"Up to {max(p.chunks for p in ev.scale if p.score_ms <= ev.embed_ms):,} vectors the scan costs less than "
            "embedding the query"
            if any(p.score_ms <= ev.embed_ms for p in ev.scale)
            else "Even the smallest size here costs more than embedding the query"
        )
        + (
            f"; at {over.chunks:,} it costs {over.score_ms / ev.embed_ms:.0f} times as much and the vectors alone "
            f"take {over.megabytes / 1e3:.1f} GB of memory."
            if over
            else f"; even at {ev.scale[-1].chunks:,} it costs less."
        )
        + f" This corpus has {m.chunks} chunks, so the flat index stays. The time to change is when the scan, "
        "not the model, sets the latency, and the change is an approximate index (HNSW, IVF) that trades a "
        "little recall for a scan over a fraction of the vectors. Two things about this stage carry over: "
        "exact search is the recall ceiling an approximate index has to be measured against, and metadata "
        "filters are no longer a free mask. Filtering a graph before the walk breaks the walk, and filtering "
        "after it is section 5's lossy case at scale.\n"
    )

    # 7. Pick and misses
    add("## 7. The pick, and what still fails\n")
    add(
        f"`retrieve` and `search` use `{d.label}`"
        + (
            f"; the benchmark picks `{pick.label}`. **Update `DEFAULT_RETRIEVAL` in "
            "`src/infra_docs_rag/retrieve/retriever.py`.**"
            if pick != d
            else ", the benchmark's pick."
        )
    )
    default_row = next((r for r in ev.thresholds if r.min_score == d.min_score), None)
    reasons = [
        f"k is {d.k} because the top {d.k} holds the answer for {ev.hits(d.k)} of {n_in} questions and the top "
        f"{high} adds {ev.hits(high) - ev.hits(d.k)} more for {ev.read(high) / ev.read(d.k):.1f} times the tokens "
        "(section 3)"
    ]
    if d.min_score is None:
        reasons.append("there is no threshold because section 4 found none that keeps every answer")
    elif default_row is not None:
        reasons.append(
            f"the threshold is {d.min_score:.2f} because it keeps {default_row.answered} of the {none.answered} "
            f"answers in the top {K}, leaves {count(default_row.silent, 'real question')} silent and rejects "
            f"{default_row.rejected} of the {n_out} unanswerable ones (section 4)"
        )
    reasons.append("filters apply before ranking because section 5 shows what filtering after it loses")
    add(
        "Three choices, each from a section above: "
        + "; ".join(reasons)
        + ". The filter itself comes from the question: a project picker, a document the user is reading, or a "
        "page they name.\n"
    )
    misses = [r for r in ev.ranked if r.rank is None or r.rank > K]
    if misses:
        add(f"The {len(misses)} in-scope questions whose right chunk is not in the top {K}:\n")
        for r in sorted(misses, key=lambda r: (r.rank is None, r.rank or 0), reverse=True):
            top = r.hits[0].chunk
            filtered = next((f for f in ev.filtered if f.query.id == r.query.id), None)
            add(
                f'- **"{r.query.text}"** ({r.query.kind}): the right chunk is {placed(r.rank)}'
                + (f" ({r.right_score:.2f})" if r.right_score is not None else "")
                + f"; first instead is `{cell(top.citation())}` ({r.top_score:.2f})."
                + (f" With its filter the right chunk is {placed(filtered.prefiltered)}." if filtered else "")
            )
        kinds_missed = {r.query.kind for r in misses}
        follow = []
        if "identifier" in kinds_missed:
            follow.append(
                "an exact keyword match for identifiers, which is what hybrid search (stage 05) adds"
            )
        if "paraphrase" in kinds_missed:
            follow.append(
                "a second look at the top candidates by a model that reads the question and the chunk "
                "together, which is reranking (stage 06)"
            )
        if "metadata" in kinds_missed:
            follow.append(
                "for the metadata questions, a filter that removes what is actually in the way; section 5 shows "
                "that a document filter does not when the competition sits inside the same document"
            )
        if follow:
            add(f"\nWhat these need is {listing(follow)}.\n")
    else:
        add(f"Every in-scope question has its right chunk in the top {K}.\n")

    add("## Reproduce\n")
    example = queries[0]
    add(
        "```sh\n"
        f"uv run infra-docs-rag embed                 # the {m.model} index this report measures\n"
        f'uv run infra-docs-rag search "{example.text}"\n'
        f'uv run infra-docs-rag retrieve "{example.text}"'
        + (f" {flags(example.filter)}" if example.filter else "")
        + "\n"
        "uv run infra-docs-rag evaluate-retrieval    # rerun the benchmark, rewrite this report\n"
        "```"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(re.sub(r"\n{3,}", "\n\n", "\n".join(out)) + "\n")
