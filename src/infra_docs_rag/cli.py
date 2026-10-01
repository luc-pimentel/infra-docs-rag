"""Command line: `infra-docs-rag fetch | ingest | report | all | chunk | embed | search | evaluate |
evaluate-chunking`, run from the repo root."""

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

SOURCES = Path("sources.yaml")
RAW_DIR = Path("data/raw")
DOCUMENTS = Path("data/processed/documents.jsonl")
REPORT = Path("reports/ingestion.md")
CHUNKS_DIR = Path("data/chunks")
INDEX_DIR = Path("data/index")
QUERIES = Path("eval/queries.yaml")
EMBEDDINGS_REPORT = Path("reports/embeddings.md")
CHUNKING_REPORT = Path("reports/chunking.md")


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


def search(query: str, model: str, k: int) -> None:
    index = Index.load(INDEX_DIR / model)
    print(format_search(index.manifest, index.search(load_embedder(model), query, k)))


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
    search_cmd = commands.add_parser("search", help="print the chunks closest to a query")
    search_cmd.add_argument("query")
    search_cmd.add_argument("-k", type=int, default=5, help="how many chunks to print (default 5)")
    for cmd in (chunk_cmd, embed_cmd, search_cmd):
        cmd.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}")
    commands.add_parser("evaluate", help="probe and compare every model, write reports/embeddings.md")
    commands.add_parser(
        "evaluate-chunking", help="compare chunking strategies and sizes, write reports/chunking.md"
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
            search(args.query, args.model, args.k)
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
    except (FileNotFoundError, ModelMismatch, ValueError) as error:
        parser.exit(1, f"error: {error}\n")
