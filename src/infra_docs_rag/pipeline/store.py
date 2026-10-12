"""Versioned indexes on disk. `data/index/<model>/` holds one directory per index version, named by the
hash of its content, and a `current` file naming the one that answers questions. A build goes into a
staging directory next to them, is checked, and is renamed into place in one step; `current` is
rewritten last. So a build that fails leaves the live index as it was, every earlier version stays
until it is removed, and rolling back is rewriting one file.

An index laid out the old way (its files straight under `data/index/<model>/`) is adopted the first
time it is touched: moved into its version directory and pointed at."""

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from ..chunk.chunks import Chunk, Chunking
from ..embed.embedders import Embedder
from ..embed.index import Index, build
from .manifest import index_version

CURRENT = "current"
FILES = ("manifest.json", "chunks.jsonl", "vectors.npy")


class IndexBuildFailed(Exception):
    """The new index did not pass its checks; the live one was left alone."""


@dataclass
class Store:
    """The index versions of one model."""

    root: Path  # data/index/<model>

    def adopt(self) -> str | None:
        """Move a flat index into its version directory and point at it. Returns the version, or None
        when there was nothing flat to adopt."""
        if not (self.root / "manifest.json").exists():
            return None
        version = index_version(self.root)
        target = self.root / version
        target.mkdir(parents=True, exist_ok=True)
        for name in FILES:
            shutil.move(str(self.root / name), str(target / name))
        self.point(version)
        return version

    def current(self) -> str | None:
        self.adopt()
        pointer = self.root / CURRENT
        if not pointer.exists():
            return None
        version = pointer.read_text().strip()
        return version if (self.root / version / "manifest.json").exists() else None

    def path(self, version: str | None = None) -> Path:
        """The directory of a version, the current one by default."""
        version = version or self.current()
        if version is None:
            raise FileNotFoundError(f"no index in {self.root}; run `infra-docs-rag embed` first")
        directory = self.root / version
        if not (directory / "manifest.json").exists():
            raise FileNotFoundError(
                f"no index version {version!r} in {self.root} (see `pipeline index --list`)"
            )
        return directory

    def versions(self) -> list[str]:
        self.adopt()
        if not self.root.exists():
            return []
        found = [d.name for d in self.root.iterdir() if (d / "manifest.json").exists()]
        return sorted(found, key=lambda v: (self.root / v / "manifest.json").stat().st_mtime)

    def point(self, version: str) -> None:
        """Make a version the current one; the write is one small file, done last."""
        if not (self.root / version / "manifest.json").exists():
            raise FileNotFoundError(f"no index version {version!r} in {self.root}")
        tmp = self.root / f".{CURRENT}.tmp"
        tmp.write_text(version + "\n")
        tmp.replace(self.root / CURRENT)

    def load(self, version: str | None = None) -> Index:
        return Index.load(self.path(version))

    def publish(
        self, embedder: Embedder, chunks: list[Chunk], ingest_version: str, chunking: Chunking
    ) -> tuple[str, Index, bool]:
        """Build, check, then make live. Returns the version, the index, and whether it was new (False
        when the same content was already there, in which case nothing was written)."""
        self.adopt()
        self.root.mkdir(parents=True, exist_ok=True)
        index = build(embedder, chunks, ingest_version, chunking)
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=self.root))
        try:
            index.save(staging)
            check(Index.load(staging), embedder, chunks)
            version = index_version(staging)
            target = self.root / version
            if (target / "manifest.json").exists():
                shutil.rmtree(staging)
                self.point(version)
                return version, Index.load(target), False
            staging.rename(target)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self.point(version)
        return version, index, True

    def remove(self, version: str) -> None:
        if version == self.current():
            raise ValueError(f"{version} is the current index; point at another one first")
        shutil.rmtree(self.path(version))


def check(index: Index, embedder: Embedder, chunks: list[Chunk]) -> None:
    """What a build has to pass before it goes live: it reloads, it holds every chunk it was given, and a
    chunk's own text finds that chunk first."""
    if len(index.chunks) != len(chunks):
        raise IndexBuildFailed(f"the index holds {len(index.chunks)} chunks but {len(chunks)} were given")
    if [c.chunk_id for c in index.chunks] != [c.chunk_id for c in chunks]:
        raise IndexBuildFailed("the index's chunks are not the ones given, or not in their order")
    if not chunks:
        raise IndexBuildFailed("an index with no chunks answers nothing")
    probe = chunks[0]
    hits = index.search(embedder, probe.text, k=1)
    if not hits or hits[0].chunk.chunk_id != probe.chunk_id:
        found = hits[0].chunk.chunk_id if hits else "nothing"
        raise IndexBuildFailed(f"searching the first chunk's own text found {found}, not {probe.chunk_id}")
