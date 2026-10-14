"""The source list (`sources.yaml`) and downloading it into `data/raw/`."""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import httpx
import yaml
from pydantic import BaseModel, model_validator

from .models import SourceType

USER_AGENT = "infra-docs-rag/0.1 (+https://github.com/luc-pimentel/infra-docs-rag)"
KNOWN_SUFFIXES = {".md", ".pdf", ".yaml", ".txt"}
DEFAULT_SUFFIX = {
    SourceType.HTML: ".html",
    SourceType.MARKDOWN: ".md",
    SourceType.TEXT: ".txt",
    SourceType.PDF: ".pdf",
}


class Source(BaseModel):
    id: str
    type: SourceType
    url: str | None = None
    path: str | None = None  # local files, such as the broken-PDF fixtures
    project: str | None = None  # the documentation set it belongs to, such as "Argo CD"
    license: str | None = None
    note: str | None = None

    @model_validator(mode="after")
    def one_location(self) -> "Source":
        if (self.url is None) == (self.path is None):
            raise ValueError(f"source {self.id!r} needs exactly one of url or path")
        return self

    @property
    def uri(self) -> str:
        return self.url or f"file:{self.path}"

    def raw_file(self, raw_dir: Path) -> Path:
        if self.path:
            return Path(self.path)
        suffix = PurePosixPath(urlparse(self.url).path).suffix.lower()
        if suffix not in KNOWN_SUFFIXES:
            suffix = DEFAULT_SUFFIX[self.type]
        return raw_dir / f"{self.id}{suffix}"


def load_sources(path: Path) -> list[Source]:
    entries = yaml.safe_load(path.read_text())["sources"]
    sources = [Source(**entry) for entry in entries]
    ids = [s.id for s in sources]
    if len(ids) != len(set(ids)):
        raise ValueError("source ids must be unique")
    return sources


def load_manifest(raw_dir: Path) -> dict[str, dict]:
    manifest = raw_dir / "manifest.json"
    return json.loads(manifest.read_text()) if manifest.exists() else {}


class FetchFailed(Exception):
    """One or more sources could not be downloaded; the others were, and the manifest says which."""

    def __init__(self, failures: dict[str, str]) -> None:
        self.failures = failures
        shown = list(failures.items())[:3]
        listed = "; ".join(f"{sid}: {why}" for sid, why in shown)
        more = f"; and {len(failures) - len(shown)} more" if len(failures) > len(shown) else ""
        super().__init__(
            f"{len(failures)} source(s) could not be fetched ({listed}{more}); data/raw/manifest.json has each error"
        )


def fetch(sources: list[Source], raw_dir: Path, refresh: bool = False) -> dict[str, dict]:
    """Download every URL source once; the manifest records when and what was fetched. A source that
    fails is recorded with its error and the rest are still downloaded; the manifest is written after
    each one, so a retry only downloads what is still missing. Raises FetchFailed at the end when any
    source failed, with the manifest already on disk."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw_dir)
    failures: dict[str, str] = {}
    with httpx.Client(follow_redirects=True, timeout=60, headers={"User-Agent": USER_AGENT}) as client:
        for source in sources:
            target = source.raw_file(raw_dir)
            have = target.exists() and bool(manifest.get(source.id, {}).get("sha256"))
            if source.path or (have and not refresh):
                for key in ("error", "error_at"):  # a refresh that failed earlier; the file is on disk
                    manifest.get(source.id, {}).pop(key, None)
                continue
            try:
                response = client.get(source.url)
                response.raise_for_status()
            except httpx.HTTPError as error:
                why = f"{type(error).__name__}: {error}"[:200]
                failures[source.id] = why
                entry = manifest.setdefault(source.id, {"url": source.url})
                entry["error"] = why
                entry["error_at"] = datetime.now(UTC).isoformat(timespec="seconds")
                print(f"failed {source.id}: {why}")
                continue
            target.write_bytes(response.content)
            manifest[source.id] = {
                "url": source.url,
                "final_url": str(response.url),
                "file": target.name,
                "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
                "sha256": hashlib.sha256(response.content).hexdigest(),
                "bytes": len(response.content),
                "content_type": response.headers.get("content-type"),
            }
            print(f"fetched {source.id} ({len(response.content):,} bytes)")
            (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (raw_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if failures:
        raise FetchFailed(failures)
    return manifest
