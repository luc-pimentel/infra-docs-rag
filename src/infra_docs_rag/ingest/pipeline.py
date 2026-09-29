"""Source files in, normalized and deduplicated records out."""

import hashlib
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

from .. import __version__
from .dedup import content_hash, mark_duplicates
from .models import Document, Provenance
from .parsers import parse
from .parsers.base import assemble
from .sources import Source, load_manifest


def ingest_version() -> str:
    """Package version plus the commit that produced the records ("+dirty" if uncommitted)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"], capture_output=True, text=True
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return __version__
    return f"{__version__}+{commit}{'.dirty' if dirty.strip() else ''}"


def ingest_source(source: Source, raw_dir: Path, manifest: dict, version: str) -> Document:
    local = source.raw_file(raw_dir)
    if not local.exists():
        raise FileNotFoundError(f"{local} is missing; run `infra-docs-rag fetch` first")
    data = local.read_bytes()
    parsed = parse(
        source.type, data, fallback_title=PurePosixPath(urlparse(source.uri).path).name or local.name
    )
    cleaned, sections, pages = assemble(parsed.parts)
    return Document(
        document_id="doc_" + hashlib.sha256(source.uri.encode()).hexdigest()[:12],
        source_id=source.id,
        source_uri=source.uri,
        source_type=source.type,
        title=parsed.title,
        raw_text=parsed.raw_text,
        cleaned_text=cleaned,
        content_hash=content_hash(cleaned) if cleaned else None,
        sections=sections,
        pages=pages,
        parse_status=parsed.status,
        parse_error=parsed.error,
        provenance=Provenance(
            local_path=local.as_posix(),
            source_sha256=hashlib.sha256(data).hexdigest(),
            fetched_at=manifest.get(source.id, {}).get("fetched_at"),
            parser=parsed.parser,
            ingest_version=version,
            ingested_at=datetime.now(UTC).isoformat(timespec="seconds"),
        ),
    )


def run(sources: list[Source], raw_dir: Path, output: Path) -> list[Document]:
    manifest = load_manifest(raw_dir)
    version = ingest_version()
    docs = [ingest_source(source, raw_dir, manifest, version) for source in sources]
    pairs = mark_duplicates(docs)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as out:
        for doc in docs:
            out.write(doc.model_dump_json() + "\n")
    output.with_name("pairs.json").write_text(json.dumps([asdict(p) for p in pairs], indent=1) + "\n")
    return docs
