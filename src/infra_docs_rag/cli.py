"""Command line: `infra-docs-rag fetch | ingest | report | all | chunk | embed | search | retrieve |
evaluate | evaluate-chunking | evaluate-retrieval | evaluate-hybrid | evaluate-rerank | answer | evaluate-grounding |
calibrate | evaluate-pipeline | traces | pipeline run | pipeline status | pipeline index | pipeline release |
pipeline rollback | pipeline canary | pipeline health`, run from the repo root."""

import argparse
import re
from pathlib import Path

import yaml

from .chunk.chunks import DEFAULT_CHUNKING, STRATEGIES, Chunking, chunk_documents
from .chunk.evaluate import GRID, benchmark, shape
from .chunk.report import format_chunks, format_shapes
from .chunk.report import write_report as write_chunking_report
from .embed.embedders import DEFAULT_MODEL, MODELS, load_embedder
from .embed.evaluate import evaluate, load_queries
from .embed.index import Index, ModelMismatch
from .embed.report import format_search
from .embed.report import write_report as write_embeddings_report
from .evaluation.calibrate import calibrate
from .evaluation.calibration_report import write_report as write_calibration_report
from .evaluation.gates import compare as compare_runs
from .evaluation.gates import load_gates
from .evaluation.labels import LABELS, load_labels
from .evaluation.metrics import metrics
from .evaluation.report import write_report as write_evaluation_report
from .evaluation.runs import RUNS_DIR, Run, from_grounding
from .evaluation.versions import DEFAULT_VERSION, Version
from .generate.answer import answer as grounded_answer
from .generate.answer import format_answer, marked
from .generate.evaluate import evaluate_grounding, load_grounding
from .generate.generator import DEFAULT_EFFORT, GENERATORS, load_generator
from .generate.generator import DEFAULT_MODEL as DEFAULT_GENERATOR
from .generate.judge import load_judge
from .generate.prompt import render
from .generate.prompts import DEFAULT_PROMPT, PROMPTS, load_prompt
from .generate.report import write_report as write_grounding_report
from .ingest.models import Document
from .ingest.pipeline import run
from .ingest.report import write_report
from .ingest.sources import fetch, load_sources
from .observe.canary import load_canary, run_canary
from .observe.health import checks, format_checks, worst
from .observe.summary import format_summary, summarize
from .observe.summary import write_report as write_observability_report
from .observe.trace import TRACES, TraceLog, as_json, trace_answer
from .observe.trace import render as render_trace
from .pipeline import commands as pipeline_commands
from .pipeline.releases import RELEASES, Releases
from .pipeline.stages import Evaluation, Paths
from .pipeline.store import Store
from .retrieve.compare import compare, load_all
from .retrieve.evaluate import evaluate_retrieval, load_benchmark
from .retrieve.fusion import STRATEGIES as FUSIONS
from .retrieve.fusion import Fusion
from .retrieve.hybrid_report import write_report as write_hybrid_report
from .retrieve.report import format_context, format_hybrid_search
from .retrieve.report import write_report as write_retrieval_report
from .retrieve.rerank import RERANKERS, load_reranker
from .retrieve.rerank_compare import compare_rerank, load_expectations
from .retrieve.rerank_report import write_report as write_rerank_report
from .retrieve.retriever import DEFAULT_RETRIEVAL, MODES, Filter, Rerank, Retrieval, Retriever

SOURCES = Path("sources.yaml")
RAW_DIR = Path("data/raw")
DOCUMENTS = Path("data/processed/documents.jsonl")
REPORT = Path("reports/ingestion.md")
CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")
QUERIES = Path("eval/queries.yaml")
BENCHMARK = Path("eval/retrieval.yaml")
EMBEDDINGS_REPORT = Path("reports/embeddings.md")
CHUNKING_REPORT = Path("reports/chunking.md")
RETRIEVAL_REPORT = Path("reports/retrieval.md")
HYBRID_QUERIES = Path("eval/hybrid.yaml")
HYBRID_REPORT = Path("reports/hybrid-search.md")
RERANK_EXPECTATIONS = Path("eval/rerank.yaml")
RERANK_REPORT = Path("reports/reranking.md")
GROUNDING_QUESTIONS = Path("eval/grounding.yaml")
GROUNDING_REPORT = Path("reports/grounding.md")
CALIBRATION_REPORT = Path("reports/calibration.md")
EVALUATION_REPORT = Path("reports/evaluation.md")
OBSERVABILITY_REPORT = Path("reports/observability.md")
CANARY = Path("eval/canary.yaml")


def load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `infra-docs-rag ingest` first")
    return [Document.model_validate_json(line) for line in path.read_text().splitlines() if line]


def load_projects() -> dict[str, str]:
    """Source id to the documentation set it belongs to, from sources.yaml."""
    return {s.id: s.project for s in load_sources(SOURCES) if s.project}


def chunk(chunkings: list[Chunking], model: str, source: str, samples: int) -> None:
    docs = load_documents(DOCUMENTS)
    embedder, projects = load_embedder(model), load_projects()
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for chunking in chunkings:
        chunks = chunk_documents(docs, chunking, embedder, projects)
        lengths = embedder.token_counts([c.embedded for c in chunks])
        output = CHUNKS_DIR / (re.sub(r"[^a-z0-9]+", "-", chunking.label.lower()).strip("-") + ".jsonl")
        output.write_text("".join(c.model_dump_json() + "\n" for c in chunks))
        rows.append((chunking, shape(chunks, lengths, embedder.max_tokens, docs), chunks, lengths))
    print(
        f"Chunks of {len({c.source_id for _, _, cs, _ in rows for c in cs})} documents, in {embedder.name} tokens\n"
    )
    print(format_shapes([(chunking, s) for chunking, s, _, _ in rows], embedder.max_tokens))
    print(f"\nWritten to {CHUNKS_DIR}/. The first {samples} chunks of `{source}`, as each strategy cuts it:")
    for chunking, _, chunks, lengths in rows:
        mine = [(c, n) for c, n in zip(chunks, lengths, strict=True) if c.source_id == source][:samples]
        print(f"\n## {chunking.label}\n\n{format_chunks(mine)}")


def embed(model: str) -> None:
    """Build the index into a staging directory, check it, and make it the current version of its model
    (see pipeline/store.py); the previous version stays on disk."""
    docs = load_documents(DOCUMENTS)
    embedder = load_embedder(model)
    chunks = chunk_documents(docs, DEFAULT_CHUNKING, embedder, load_projects())
    version, index, new = Store(INDEX_DIR / model).publish(
        embedder, chunks, docs[0].provenance.ingest_version, DEFAULT_CHUNKING
    )
    m = index.manifest
    print(
        f"embedded {m.chunks} chunks ({m.chunking.label}) with {m.model_id} ({m.dimensions} dimensions) "
        f"in {m.build_seconds:.1f}s into {INDEX_DIR / model}/{version}/"
        + (", now current" if new else "; the same index was already there, now current")
    )
    if m.truncated:
        print(
            f"{m.truncated} chunks are longer than {m.max_tokens} tokens; the model saw only their beginning"
        )


def retriever(version: Version, pinned: str | None) -> tuple[Retriever, Index, str]:
    """The retriever a version runs: its embedding model's index (the pinned version when it is on disk),
    and the reranker its retrieval asks for."""
    retrieval = version.retrieval
    index, index_id = pipeline_commands.index_for(version, pinned, INDEX_DIR)
    reranker = load_reranker(retrieval.rerank.model) if retrieval.rerank is not None else None
    return Retriever(index, load_embedder(version.embedding), reranker), index, index_id


def resolve(args: argparse.Namespace) -> tuple[Version, str | None]:
    """The version `search`, `retrieve` and `answer` run: --version, else the release when the flags are
    the defaults, else the flags (see pipeline/commands.py). `--model` picks the index when no version does."""
    version, pinned = pipeline_commands.resolve_version(args.version, retrieval_from(args), RELEASES)
    if args.version is None and version.name == DEFAULT_VERSION and args.model != DEFAULT_MODEL:
        version = version.model_copy(update={"embedding": args.model})
    if getattr(args, "prompt", None):
        version = version.model_copy(update={"prompt": args.prompt})
    return version, pinned


def search(query: str, version: Version, pinned: str | None) -> None:
    """Every hit the retrieval would rank, scores included; the ones under the threshold say so."""
    retrieval = version.retrieval
    r, _, _ = retriever(version, pinned)
    context = r.retrieve(query, retrieval.model_copy(update={"min_score": None, "budget": None}))
    hits = [s.hit for s in context.sources]
    if retrieval.mode == "dense":
        print(format_search(r.index.manifest, hits, note=retrieval.label, min_score=retrieval.min_score))
    else:
        print(format_hybrid_search(r.index.manifest, hits, retrieval.label))


def retrieve(query: str, version: Version, pinned: str | None) -> None:
    """The passages a generator would read, numbered and cited."""
    r, _, _ = retriever(version, pinned)
    print(format_context(r.index.manifest, r.retrieve(query, version.retrieval)))


def answer(
    query: str,
    version: Version,
    pinned: str | None,
    generator: str,
    effort: str,
    show_prompt: bool,
    show_trace: bool = False,
    judge: bool = False,
) -> None:
    """A grounded answer, or the prompt that would produce one, under a pipeline version. Every answer
    leaves a trace in the log; --trace prints it, --judge adds the judge's reading to it."""
    r, index, index_id = retriever(version, pinned)
    prompt = load_prompt(version.prompt)
    retrieval = version.retrieval
    if show_prompt:
        print(
            f"[version {version.name} · index {version.embedding}/{index_id} · prompt {prompt.version(generator)}]"
        )
        print(render(r.retrieve(query, retrieval), query, prompt.text(generator)))
        return
    gen = load_generator(generator, effort, version.prompt)
    a = grounded_answer(query, r, gen, retrieval)
    reading = None
    if judge and a.outcome == "answered":
        reading = load_judge().judge(a.context, query, marked(a))
    lineage = pipeline_commands.lineage(
        version, index, index_id, prompt, gen.name, a.generated.model if a.generated else ""
    )
    t = trace_answer(a, version.name, lineage, prompt.version(gen.name), reading, gen.prices)
    TraceLog(TRACES).append(t)
    if show_trace:
        print(render_trace(t))
        return
    print(format_answer(a))
    if reading is not None:
        print(
            f"judge: {reading.model} · grounded {reading.grounded:.2f} · fabricated {reading.fabricated:.2f} · gap named {reading.gap_named:.2f}"
        )
    print(f"trace {t.trace_id} · {TRACES}")


def run_benchmark(
    version: Version,
    pinned: str | None,
    generator: str,
    effort: str,
    limit: int | None,
    judge: bool,
    name: str,
):
    """The grounding benchmark under a pipeline version, and the run to save, lineage included."""
    bench, extra = load_all(BENCHMARK, HYBRID_QUERIES)
    adversarial = load_grounding(GROUNDING_QUESTIONS, {q.id for q in [*bench.queries, *extra]})
    second = version.retrieval.rerank
    index, index_id = pipeline_commands.index_for(version, pinned, INDEX_DIR)
    gen = load_generator(generator, effort, version.prompt)
    prompt = load_prompt(version.prompt)
    lineage = pipeline_commands.lineage(version, index, index_id, prompt, gen.name, gen.model)
    log = TraceLog(TRACES)
    grounding = evaluate_grounding(
        index,
        load_embedder(version.embedding),
        load_documents(DOCUMENTS),
        bench,
        extra,
        adversarial,
        gen,
        load_reranker(second.model) if second is not None else None,
        retrieval=version.retrieval,
        limit=limit,
        judge=load_judge() if judge else None,
        tracer=lambda a, j: log.append(
            trace_answer(a, f"{version.name}:{name}", lineage, prompt.version(gen.name), j, gen.prices)
        ),
    )
    lineage = lineage.model_copy(update={"generator": grounding.generator})
    return grounding, from_grounding(grounding, name, version.name, version.description, lineage)


def evaluation_for(args: argparse.Namespace, paths: Paths) -> Evaluation:
    """The evaluate and gate stages of `pipeline run`: the benchmark under --version, saved as a run,
    then compared to the current release's run."""
    from .pipeline.stages import StageFailed

    version, pinned = pipeline_commands.resolve_version(args.version)
    effort = args.effort or DEFAULT_EFFORT
    name = args.save or f"{version.name}-{args.generator}"
    prompt = load_prompt(version.prompt)

    def produce(directory: Path) -> None:
        _, saved = run_benchmark(version, pinned, args.generator, effort, args.limit, not args.no_judge, name)
        saved.save(directory)

    def gate(directory: Path) -> str:
        releases = Releases.load(paths.releases)
        if releases.live is None:
            return "no release to compare against; `pipeline release --candidate` makes the first"
        comparison = compare_runs(Run.load(RUNS_DIR / releases.live.run), Run.load(directory), load_gates())
        write_evaluation_report(comparison, EVALUATION_REPORT)
        if comparison.blocked:
            raise StageFailed(
                "gate", f"{comparison.verdict} against {releases.live.name}; see {EVALUATION_REPORT}"
            )
        return f"{comparison.verdict} against {releases.live.name}; wrote {EVALUATION_REPORT}"

    stage_version = f"{version.name}|{version.retrieval.model_dump_json()}|{prompt.version(args.generator)}|{args.generator}|{effort}|{args.limit}|{not args.no_judge}"
    return Evaluation(run_name=name, version=stage_version, produce=produce, gate=gate)


def retrieval_from(args: argparse.Namespace) -> Retrieval:
    criteria = Filter(project=args.project, source=args.source, section=args.section, page=args.page)
    min_score = args.min_score
    if min_score is None and args.mode != "lexical":  # not given: the default floor, where there is a cosine
        min_score = DEFAULT_RETRIEVAL.min_score
    if min_score is not None and min_score < 0:
        min_score = None
    return Retrieval(
        k=args.k,
        min_score=min_score,
        filter=criteria,
        prefilter=not args.post_filter,
        budget=getattr(args, "budget", None),
        mode=args.mode,
        fusion=Fusion(strategy=args.fusion, alpha=args.alpha, rrf_k=args.rrf_k, depth=args.depth),
        rerank=None
        if args.rerank == "none"
        else Rerank(model=args.rerank, candidates=args.candidates, min_relevance=args.min_relevance),
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="infra-docs-rag", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_cmd = commands.add_parser("fetch", help="download the URL sources into data/raw/")
    fetch_cmd.add_argument("--refresh", action="store_true", help="download again even if present")
    commands.add_parser("ingest", help="parse, clean and deduplicate into data/processed/")
    commands.add_parser("report", help="write reports/ingestion.md from the processed records")
    commands.add_parser("all", help="fetch, ingest and report")
    chunk_cmd = commands.add_parser(
        "chunk", help="cut every document with each strategy into data/chunks/ and compare the chunks"
    )
    chunk_cmd.add_argument("--strategy", choices=STRATEGIES, help="one strategy (default: all of them)")
    chunk_cmd.add_argument("--size", type=int, default=DEFAULT_CHUNKING.size, help="tokens per chunk")
    chunk_cmd.add_argument("--overlap", type=int, default=DEFAULT_CHUNKING.overlap, help="tokens repeated")
    chunk_cmd.add_argument(
        "--context", action=argparse.BooleanOptionalAction, default=DEFAULT_CHUNKING.context
    )
    chunk_cmd.add_argument("--source", default="argocd-getting-started-stable", help="document to sample")
    chunk_cmd.add_argument("--samples", type=int, default=2, help="chunks to print per strategy")
    embed_cmd = commands.add_parser(
        "embed", help=f"embed every chunk ({DEFAULT_CHUNKING.label}) into data/index/"
    )
    search_cmd = commands.add_parser(
        "search", help="print the chunks that rank highest for a query, with their scores"
    )
    retrieve_cmd = commands.add_parser(
        "retrieve", help="print the passages a generator would read for a query, numbered and cited"
    )
    answer_cmd = commands.add_parser(
        "answer",
        help="answer a question from the retrieved passages, every statement cited, or say the docs do not cover it",
    )
    answer_cmd.add_argument(
        "--generator",
        choices=list(GENERATORS),
        default=DEFAULT_GENERATOR,
        help="; ".join(f"{name}: {how}" for name, how in GENERATORS.items())
        + f" (default {DEFAULT_GENERATOR})",
    )
    answer_cmd.add_argument(
        "--effort",
        choices=["low", "medium", "high", "xhigh", "max"],
        default=DEFAULT_EFFORT,
        help=f"how hard the model thinks (default {DEFAULT_EFFORT})",
    )
    answer_cmd.add_argument(
        "--show-prompt",
        action="store_true",
        help="print the request the model would get instead of calling it",
    )
    answer_cmd.add_argument(
        "--prompt",
        choices=list(PROMPTS),
        help=f"the system prompt, by name in the registry (default the version's, {DEFAULT_PROMPT})",
    )
    answer_cmd.add_argument(
        "--trace",
        action="store_true",
        help="print the trace (every stage's input, output and time) instead of the answer",
    )
    answer_cmd.add_argument("--judge", action="store_true", help="have the judge read the answer too")
    threshold = (
        f"default {DEFAULT_RETRIEVAL.min_score:.2f}"
        if DEFAULT_RETRIEVAL.min_score is not None
        else "default none"
    )
    for cmd in (search_cmd, retrieve_cmd, answer_cmd):
        cmd.add_argument("query")
        cmd.add_argument(
            "--version",
            help="the pipeline version to run, from eval/versions.yaml, or `release` for the current release's "
            "(default: the release's when there is one and no retrieval flag is given, else the flags)",
        )
        cmd.add_argument(
            "-k",
            type=int,
            default=DEFAULT_RETRIEVAL.k,
            help=f"how many chunks (default {DEFAULT_RETRIEVAL.k})",
        )
        cmd.add_argument(
            "--min-score",
            type=float,
            help=f"cosine similarity a chunk needs; search marks the ones under it, retrieve drops them ({threshold}, -1 for none; lexical mode has none)",
        )
        fusion = DEFAULT_RETRIEVAL.fusion
        cmd.add_argument(
            "--mode",
            choices=MODES,
            default=DEFAULT_RETRIEVAL.mode,
            help=f"dense ranks by cosine similarity, lexical by BM25, hybrid fuses the two (default {DEFAULT_RETRIEVAL.mode})",
        )
        cmd.add_argument(
            "--fusion",
            choices=FUSIONS,
            default=fusion.strategy,
            help=f"hybrid: how to merge (default {fusion.strategy})",
        )
        cmd.add_argument(
            "--alpha",
            type=float,
            default=fusion.alpha,
            help=f"hybrid weighted: share of the dense score (default {fusion.alpha})",
        )
        cmd.add_argument(
            "--rrf-k",
            type=int,
            default=fusion.rrf_k,
            help=f"hybrid rrf: the rank constant (default {fusion.rrf_k})",
        )
        cmd.add_argument(
            "--depth",
            type=int,
            default=fusion.depth,
            help=f"hybrid: candidates taken from each ranking (default {fusion.depth})",
        )
        second = DEFAULT_RETRIEVAL.rerank
        cmd.add_argument(
            "--rerank",
            choices=["none", *RERANKERS],
            default=second.model if second is not None else "none",
            help="cross-encoder that reorders the first stage's top candidates "
            f"(default {second.model if second is not None else 'none'})",
        )
        cmd.add_argument(
            "--candidates",
            type=int,
            default=second.candidates if second is not None else Rerank().candidates,
            help="first-stage hits the reranker reads "
            f"(default {second.candidates if second is not None else Rerank().candidates})",
        )
        cmd.add_argument(
            "--min-relevance",
            type=float,
            default=second.min_relevance if second is not None else None,
            help="reranker score in [0, 1] a hit needs; retrieve drops the ones under it (default "
            + (
                f"{second.min_relevance:.2f}"
                if second is not None and second.min_relevance is not None
                else "none"
            )
            + ")",
        )
        cmd.add_argument("--project", help="only chunks from this documentation set, e.g. 'Argo CD'")
        cmd.add_argument("--source", help="only chunks from this source id in sources.yaml")
        cmd.add_argument(
            "--section", help="only chunks under this heading ('Parent › Child' for a repeated one)"
        )
        cmd.add_argument("--page", type=int, help="only chunks that touch this PDF page")
        cmd.add_argument(
            "--post-filter",
            action="store_true",
            help="rank every chunk first, then drop what the filter rejects",
        )
    retrieve_cmd.add_argument("--budget", type=int, help="most tokens the passages may add up to")
    for cmd in (chunk_cmd, embed_cmd, search_cmd, retrieve_cmd, answer_cmd):
        cmd.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}")
    commands.add_parser("evaluate", help="probe and compare every model, write reports/embeddings.md")
    commands.add_parser(
        "evaluate-chunking", help="compare chunking strategies and sizes, write reports/chunking.md"
    )
    commands.add_parser(
        "evaluate-retrieval",
        help="run the retrieval benchmark on the default index, write reports/retrieval.md",
    )
    commands.add_parser(
        "evaluate-hybrid",
        help="compare dense, BM25 and fused rankings on the benchmark, write reports/hybrid-search.md",
    )
    commands.add_parser(
        "evaluate-rerank",
        help="rerank the first stage's candidates with each cross-encoder, write reports/reranking.md",
    )
    grounding_cmd = commands.add_parser(
        "evaluate-grounding",
        help="answer every labelled and adversarial question end to end, write reports/grounding.md",
    )
    grounding_cmd.add_argument(
        "--generator",
        choices=list(GENERATORS),
        default=DEFAULT_GENERATOR,
        help="; ".join(f"{name}: {how}" for name, how in GENERATORS.items())
        + f" (default {DEFAULT_GENERATOR})",
    )
    grounding_cmd.add_argument("--effort", default=DEFAULT_EFFORT, help=f"default {DEFAULT_EFFORT}")
    grounding_cmd.add_argument("--limit", type=int, help="only the first N questions, to try the run cheaply")
    grounding_cmd.add_argument(
        "--no-judge",
        action="store_true",
        help="skip the jev reading of each answer (grounded, fabricated, gap named)",
    )
    grounding_cmd.add_argument(
        "--version",
        default=DEFAULT_VERSION,
        help=f"the pipeline version to run, from eval/versions.yaml (default {DEFAULT_VERSION}: the current flags)",
    )
    grounding_cmd.add_argument(
        "--save",
        metavar="NAME",
        help="save the run as eval/runs/NAME/ (default <version>-<generator>)",
    )
    grounding_cmd.add_argument(
        "--no-report",
        action="store_true",
        help="save the run but leave reports/grounding.md alone",
    )
    calibrate_cmd = commands.add_parser(
        "calibrate",
        help="compare the judge and the mechanical checks to the hand labels, write reports/calibration.md",
    )
    calibrate_cmd.add_argument("--labels", type=Path, default=LABELS, help=f"default {LABELS}")
    calibrate_cmd.add_argument(
        "--run", help="the saved run to read the answers from (default: the one the labels name)"
    )
    pipeline_cmd = commands.add_parser(
        "evaluate-pipeline",
        help="compare two saved runs metric by metric against the gates, write reports/evaluation.md; exit 1 when a critical gate fails",
    )
    pipeline_cmd.add_argument("--baseline", help="a run under eval/runs/")
    pipeline_cmd.add_argument("--candidate", help="a run under eval/runs/")
    pipeline_cmd.add_argument(
        "--list-metrics", action="store_true", help="print every metric a gate can name"
    )
    pipeline_cmd.add_argument("--list-runs", action="store_true", help="print the saved runs")
    traces_cmd = commands.add_parser(
        "traces",
        help=f"summarize the trace log ({TRACES}); --last N or --id ID prints traces; --report writes reports/observability.md",
    )
    traces_cmd.add_argument("--last", type=int, help="print the last N traces")
    traces_cmd.add_argument("--id", help="print one trace")
    traces_cmd.add_argument("--json", action="store_true", help="print traces as JSON instead of rendered")
    traces_cmd.add_argument(
        "--report",
        action="store_true",
        help="write the summary and the last traces to reports/observability.md",
    )
    traces_cmd.add_argument(
        "--examples", type=int, default=4, help="traces rendered in full in the report (default 4)"
    )
    pipeline_commands.add_pipeline_commands(commands)
    args = parser.parse_args(argv)

    try:
        if args.command in ("fetch", "ingest", "report", "all"):
            sources = load_sources(SOURCES)
            if args.command in ("fetch", "all"):
                fetch(sources, RAW_DIR, refresh=getattr(args, "refresh", False))
            if args.command in ("ingest", "all"):
                docs = run(sources, RAW_DIR, DOCUMENTS)
                print(f"ingested {len(docs)} documents into {DOCUMENTS}")
            if args.command in ("report", "all"):
                write_report(load_documents(DOCUMENTS), sources, DOCUMENTS.with_name("pairs.json"), REPORT)
                print(f"wrote {REPORT}")
        elif args.command == "chunk":
            sized = [args.strategy] if args.strategy else list(GRID)
            chunkings = [] if args.strategy not in (None, "whole-sections") else [Chunking()]
            chunkings += [
                Chunking(strategy=s, size=args.size, overlap=args.overlap, context=args.context)
                for s in sized
                if s != "whole-sections"
            ]
            chunk(chunkings, args.model, args.source, args.samples)
        elif args.command == "embed":
            embed(args.model)
        elif args.command == "search":
            search(args.query, *resolve(args))
        elif args.command == "retrieve":
            retrieve(args.query, *resolve(args))
        elif args.command == "answer":
            answer(
                args.query,
                *resolve(args),
                args.generator,
                args.effort,
                args.show_prompt,
                args.trace,
                args.judge,
            )
        elif args.command == "traces":
            log = TraceLog(TRACES)
            if args.id:
                found = log.find(args.id)
                if found is None:
                    raise ValueError(f"no trace {args.id} in {TRACES}")
                print(as_json(found) if args.json else render_trace(found))
            elif args.last:
                for t in log.read(last=args.last):
                    print(as_json(t) if args.json else render_trace(t))
                    print()
            else:
                everything = log.read()
                current = Store(INDEX_DIR / DEFAULT_MODEL).current()
                print(format_summary(summarize(everything, current)))
                if args.report:
                    examples = [
                        t for t in everything if t.version.startswith("canary:") or ":" not in t.version
                    ]
                    write_observability_report(
                        everything, OBSERVABILITY_REPORT, examples[-args.examples :], current
                    )
                    print(f"wrote {OBSERVABILITY_REPORT}")
        elif args.command == "evaluate":
            evaluation = evaluate(load_documents(DOCUMENTS), load_queries(QUERIES), list(MODELS))
            write_embeddings_report(evaluation, EMBEDDINGS_REPORT)
            print(f"wrote {EMBEDDINGS_REPORT}")
        elif args.command == "evaluate-chunking":
            docs, queries = load_documents(DOCUMENTS), load_queries(QUERIES)
            write_chunking_report(
                benchmark(load_embedder(DEFAULT_MODEL), docs, queries, load_projects()), CHUNKING_REPORT
            )
            print(f"wrote {CHUNKING_REPORT}")
        elif args.command == "evaluate-retrieval":
            earlier = {q.id for q in load_queries(QUERIES).queries}
            result = evaluate_retrieval(
                Index.load(INDEX_DIR / DEFAULT_MODEL),
                load_embedder(DEFAULT_MODEL),
                load_documents(DOCUMENTS),
                load_benchmark(BENCHMARK),
                earlier,
            )
            write_retrieval_report(result, RETRIEVAL_REPORT)
            print(f"wrote {RETRIEVAL_REPORT}")
        elif args.command == "evaluate-hybrid":
            bench, extra = load_all(BENCHMARK, HYBRID_QUERIES)
            comparison = compare(
                Index.load(INDEX_DIR / DEFAULT_MODEL),
                load_embedder(DEFAULT_MODEL),
                load_documents(DOCUMENTS),
                bench,
                extra,
            )
            write_hybrid_report(comparison, HYBRID_REPORT)
            print(f"wrote {HYBRID_REPORT}")
        elif args.command == "evaluate-rerank":
            bench, extra = load_all(BENCHMARK, HYBRID_QUERIES)
            result = compare_rerank(
                Index.load(INDEX_DIR / DEFAULT_MODEL),
                load_embedder(DEFAULT_MODEL),
                load_documents(DOCUMENTS),
                bench,
                extra,
                {name: load_reranker(name) for name in RERANKERS},
                load_expectations(RERANK_EXPECTATIONS, [*bench.queries, *extra]),
            )
            write_rerank_report(result, RERANK_REPORT)
            print(f"wrote {RERANK_REPORT}")
        elif args.command == "evaluate-grounding":
            version, pinned = pipeline_commands.resolve_version(args.version)
            name = args.save or f"{version.name}-{args.generator}"
            grounding, saved = run_benchmark(
                version, pinned, args.generator, args.effort, args.limit, not args.no_judge, name
            )
            print(f"saved the run to {saved.save(RUNS_DIR / saved.name)}/")
            if not args.no_report:
                write_grounding_report(grounding, GROUNDING_REPORT)
                print(f"wrote {GROUNDING_REPORT}")
        elif args.command == "pipeline":
            if args.pipeline_command == "run":
                if pipeline_commands.run_pipeline(args, DEFAULT_MODEL, DEFAULT_CHUNKING, evaluation_for):
                    parser.exit(1)
            elif args.pipeline_command == "status":
                pipeline_commands.status(DEFAULT_MODEL, DEFAULT_CHUNKING, INDEX_DIR)
            elif args.pipeline_command == "index":
                pipeline_commands.manage_index(args, DEFAULT_MODEL, INDEX_DIR)
            elif args.pipeline_command == "release":
                if pipeline_commands.release(args, INDEX_DIR):
                    parser.exit(1)
            elif args.pipeline_command == "rollback":
                pipeline_commands.roll_back(INDEX_DIR)
            elif args.pipeline_command == "canary":
                version, pinned = pipeline_commands.resolve_version(args.version)
                r, index, index_id = retriever(version, pinned)
                gen = load_generator(args.generator, args.effort, version.prompt)
                prompt = load_prompt(version.prompt)
                lineage = pipeline_commands.lineage(version, index, index_id, prompt, gen.name, gen.model)
                log = TraceLog(TRACES)
                print(
                    f"canary under {version.name} · index {version.embedding}/{index_id} · prompt {prompt.version(gen.name)}"
                )
                results = run_canary(
                    load_canary(CANARY),
                    r,
                    gen,
                    version.retrieval,
                    on_answer=lambda a: log.append(
                        trace_answer(
                            a, f"canary:{version.name}", lineage, prompt.version(gen.name), None, gen.prices
                        )
                    ),
                )
                failed = [x for x in results if not x.passed]
                print(f"{len(results) - len(failed)}/{len(results)} passed")
                if failed:
                    parser.exit(1)
            elif args.pipeline_command == "health":
                from .pipeline.stages import Config, Paths, build_pipeline

                found = checks(
                    INDEX_DIR,
                    DEFAULT_MODEL,
                    Releases.load(RELEASES),
                    build_pipeline(Paths(), Config(), lambda: load_embedder(DEFAULT_MODEL)),
                    TraceLog(TRACES),
                )
                print(format_checks(found))
                if worst(found) == "fail":
                    parser.exit(1)
        elif args.command == "calibrate":
            labelled_run = args.run or (yaml.safe_load(args.labels.read_text()) or {}).get("run", "")
            if not labelled_run:
                raise ValueError(f"{args.labels} names no run and --run was not given")
            labelled = Run.load(RUNS_DIR / labelled_run)
            calibration = calibrate(labelled, load_labels(args.labels, labelled))
            write_calibration_report(calibration, CALIBRATION_REPORT)
            for s in calibration.signals:
                print(f"{s.name}: {s.agreement.agreement:.0%} agreement over {s.agreement.n}")
            print(f"wrote {CALIBRATION_REPORT}")
        elif args.command == "evaluate-pipeline":
            if args.list_runs:
                for d in sorted(RUNS_DIR.iterdir()) if RUNS_DIR.exists() else []:
                    if (d / "manifest.json").exists():
                        m = Run.load(d).manifest
                        print(
                            f"{d.name}: {m.version}, {m.generator} via {m.backend}, {m.questions} questions, {m.created_at[:10]}"
                        )
                return
            if args.list_metrics:
                if not RUNS_DIR.exists() or not any(RUNS_DIR.iterdir()):
                    raise ValueError("no saved run to list the metrics of; run evaluate-grounding first")
                sample = Run.load(
                    next(d for d in sorted(RUNS_DIR.iterdir()) if (d / "manifest.json").exists())
                )
                for m in metrics(sample).values():
                    print(f"{m.name}: {m.description} ({m.better} is better)")
                return
            if not args.baseline or not args.candidate:
                raise ValueError(
                    "--baseline and --candidate name two runs under eval/runs/ (see --list-runs)"
                )
            comparison = compare_runs(
                Run.load(RUNS_DIR / args.baseline), Run.load(RUNS_DIR / args.candidate), load_gates()
            )
            write_evaluation_report(comparison, EVALUATION_REPORT)
            for r in comparison.results:
                print(
                    f"{'pass' if r.passed else ('FAIL' if r.gate.critical else 'warn')}  {r.gate.metric}: {r.detail}"
                )
            print(f"{comparison.verdict}; wrote {EVALUATION_REPORT}")
            if comparison.blocked:
                parser.exit(1)
    except (FileNotFoundError, ModelMismatch, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
