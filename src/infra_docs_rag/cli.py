"""Command line: `infra-docs-rag fetch | ingest | report | all | embed | search | evaluate`, run from
the repo root."""

import argparse
from pathlib import Path

from .embed.chunks import section_chunks
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
INDEX_DIR = Path("data/index")
QUERIES = Path("eval/queries.yaml")
EMBEDDINGS_REPORT = Path("reports/embeddings.md")


def load_documents(path: Path) -> list[Document]:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; run `infra-docs-rag ingest` first")
    return [Document.model_validate_json(line) for line in path.read_text().splitlines() if line]


def embed(model: str) -> None:
    docs = load_documents(DOCUMENTS)
    index = build(load_embedder(model), section_chunks(docs), docs[0].provenance.ingest_version)
    index.save(INDEX_DIR / model)
    m = index.manifest
    print(
        f"embedded {m.chunks} chunks with {m.model_id} ({m.dimensions} dimensions) "
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
    embed_cmd = commands.add_parser("embed", help="embed every section into data/index/<model>/")
    search_cmd = commands.add_parser("search", help="print the sections closest to a query")
    search_cmd.add_argument("query")
    search_cmd.add_argument("-k", type=int, default=5, help="how many sections to print (default 5)")
    for cmd in (embed_cmd, search_cmd):
        cmd.add_argument("--model", choices=MODELS, default=DEFAULT_MODEL, help=f"default {DEFAULT_MODEL}")
    commands.add_parser("evaluate", help="probe and compare every model, write reports/embeddings.md")
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
        elif args.command == "embed":
            embed(args.model)
        elif args.command == "search":
            search(args.query, args.model, args.k)
        elif args.command == "evaluate":
            evaluation = evaluate(load_documents(DOCUMENTS), load_queries(QUERIES), list(MODELS), INDEX_DIR)
            write_embeddings_report(evaluation, EMBEDDINGS_REPORT)
            print(f"wrote {EMBEDDINGS_REPORT}")
    except (FileNotFoundError, ModelMismatch) as error:
        parser.exit(1, f"error: {error}\n")
