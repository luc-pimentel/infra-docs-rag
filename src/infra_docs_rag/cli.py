"""Command line: `infra-docs-rag fetch | ingest | report | all | chunk | embed | search | retrieve |
evaluate | evaluate-chunking | evaluate-retrieval | evaluate-hybrid`, run from the repo root."""

import argparse
import re
from pathlib import Path

from .chunk.chunks import DEFAULT_CHUNKING, STRATEGIES, Chunking, chunk_documents
from .chunk.evaluate import GRID, benchmark, shape
from .chunk.report import format_chunks, format_shapes
from .chunk.report import write_report as write_chunking_report
from .embed.embedders import DEFAULT_MODEL, MODELS, load_embedder
from .embed.evaluate import evaluate, load_queries
from .embed.index import Index, ModelMismatch, build
from .embed.report import format_search
from .embed.report import write_report as write_embeddings_report
from .ingest.models import Document
from .ingest.pipeline import run
from .ingest.report import write_report
from .ingest.sources import fetch, load_sources
from .retrieve.compare import compare, load_all
from .retrieve.evaluate import evaluate_retrieval, load_benchmark
from .retrieve.fusion import STRATEGIES as FUSIONS
from .retrieve.fusion import Fusion
from .retrieve.hybrid_report import write_report as write_hybrid_report
from .retrieve.report import format_context, format_hybrid_search
from .retrieve.report import write_report as write_retrieval_report
from .retrieve.retriever import DEFAULT_RETRIEVAL, MODES, Filter, Retrieval, Retriever

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
    docs = load_documents(DOCUMENTS)
    embedder = load_embedder(model)
    chunks = chunk_documents(docs, DEFAULT_CHUNKING, embedder, load_projects())
    index = build(embedder, chunks, docs[0].provenance.ingest_version, DEFAULT_CHUNKING)
    index.save(INDEX_DIR / model)
    m = index.manifest
    print(
        f"embedded {m.chunks} chunks ({m.chunking.label}) with {m.model_id} ({m.dimensions} dimensions) "
        f"in {m.build_seconds:.1f}s into {INDEX_DIR / model}/"
    )
    if m.truncated:
        print(
            f"{m.truncated} chunks are longer than {m.max_tokens} tokens; the model saw only their beginning"
        )


def retriever(model: str) -> Retriever:
    return Retriever(Index.load(INDEX_DIR / model), load_embedder(model))


def search(query: str, model: str, retrieval: Retrieval) -> None:
    """Every hit the retrieval would rank, scores included; the ones under the threshold say so."""
    r = retriever(model)
    context = r.retrieve(query, retrieval.model_copy(update={"min_score": None, "budget": None}))
    hits = [s.hit for s in context.sources]
    if retrieval.mode == "dense":
        print(format_search(r.index.manifest, hits, note=retrieval.label, min_score=retrieval.min_score))
    else:
        print(format_hybrid_search(r.index.manifest, hits, retrieval.label))


def retrieve(query: str, model: str, retrieval: Retrieval) -> None:
    """The passages a generator would read, numbered and cited."""
    r = retriever(model)
    print(format_context(r.index.manifest, r.retrieve(query, retrieval)))


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
    search_cmd = commands.add_parser("search", help="print the chunks closest to a query, with their scores")
    retrieve_cmd = commands.add_parser(
        "retrieve", help="print the passages a generator would read for a query, numbered and cited"
    )
    threshold = (
        f"default {DEFAULT_RETRIEVAL.min_score:.2f}"
        if DEFAULT_RETRIEVAL.min_score is not None
        else "default none"
    )
    for cmd in (search_cmd, retrieve_cmd):
        cmd.add_argument("query")
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
        cmd.add_argument(
            "--mode",
            choices=MODES,
            default="dense",
            help="dense ranks by cosine similarity, lexical by BM25, hybrid fuses the two (default dense)",
        )
        cmd.add_argument(
            "--fusion",
            choices=FUSIONS,
            default=Fusion().strategy,
            help=f"hybrid: how to merge (default {Fusion().strategy})",
        )
        cmd.add_argument(
            "--alpha",
            type=float,
            default=Fusion().alpha,
            help=f"hybrid weighted: share of the dense score (default {Fusion().alpha})",
        )
        cmd.add_argument(
            "--rrf-k",
            type=int,
            default=Fusion().rrf_k,
            help=f"hybrid rrf: the rank constant (default {Fusion().rrf_k})",
        )
        cmd.add_argument(
            "--depth",
            type=int,
            default=Fusion().depth,
            help=f"hybrid: candidates taken from each ranking (default {Fusion().depth})",
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
    for cmd in (chunk_cmd, embed_cmd, search_cmd, retrieve_cmd):
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
            search(args.query, args.model, retrieval_from(args))
        elif args.command == "retrieve":
            retrieve(args.query, args.model, retrieval_from(args))
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
    except (FileNotFoundError, ModelMismatch, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
