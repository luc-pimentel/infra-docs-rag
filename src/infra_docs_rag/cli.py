"""Command line: `infra-docs-rag fetch | ingest | report | all`, run from the repo root."""

import argparse
from pathlib import Path

from .ingest.models import Document
from .ingest.pipeline import run
from .ingest.report import write_report
from .ingest.sources import fetch, load_sources

SOURCES = Path("sources.yaml")
RAW_DIR = Path("data/raw")
DOCUMENTS = Path("data/processed/documents.jsonl")
REPORT = Path("reports/ingestion.md")


def load_documents(path: Path) -> list[Document]:
    return [Document.model_validate_json(line) for line in path.read_text().splitlines() if line]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="infra-docs-rag", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fetch_cmd = commands.add_parser("fetch", help="download the URL sources into data/raw/")
    fetch_cmd.add_argument("--refresh", action="store_true", help="download again even if present")
    commands.add_parser("ingest", help="parse, clean and deduplicate into data/processed/")
    commands.add_parser("report", help="write reports/ingestion.md from the processed records")
    commands.add_parser("all", help="fetch, ingest and report")
    args = parser.parse_args(argv)

    sources = load_sources(SOURCES)
    if args.command in ("fetch", "all"):
        fetch(sources, RAW_DIR, refresh=getattr(args, "refresh", False))
    if args.command in ("ingest", "all"):
        docs = run(sources, RAW_DIR, DOCUMENTS)
        print(f"ingested {len(docs)} documents into {DOCUMENTS}")
    if args.command in ("report", "all"):
        write_report(load_documents(DOCUMENTS), sources, DOCUMENTS.with_name("pairs.json"), REPORT)
        print(f"wrote {REPORT}")
