"""The lineage of a result: which version of each stage produced it. Every version here is either a
pinned id (a model revision, a commit) or a hash of the content it names, so two manifests that read the
same were produced by the same pipeline, and a result can be traced to the prompt, chunking, embedding
model and index behind it without trusting a label."""

import hashlib
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from ..chunk.chunks import Chunking
from ..embed.index import Manifest as IndexManifest
from ..generate.prompts import Prompt
from ..retrieve.retriever import Retrieval


def short(text: str, n: int = 12) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:n]


def slug(text: str) -> str:
    out = "".join(c if c.isalnum() else "-" for c in text.lower())
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


def chunking_version(chunking: Chunking | None) -> str:
    """`section-aware-512-64-context+1f2e3d4c5b6a`: the label, then the hash of every setting."""
    c = chunking or Chunking()
    return f"{slug(c.label)}+{short(c.model_dump_json())}"


def embedding_version(model_id: str, revision: str) -> str:
    """`BAAI/bge-small-en-v1.5@5c38ec7c405e`: the model and the commit it is pinned to."""
    return f"{model_id}@{revision[:12]}"


def retrieval_version(name: str, retrieval: Retrieval) -> str:
    """`v2-hybrid-rerank+9a8b7c6d5e4f`: the version's name, then the hash of the configuration."""
    return f"{name}+{short(retrieval.model_dump_json())}"


def index_version(directory: Path) -> str:
    """The hash of what an index holds: its vectors, its chunks and the manifest fields that say how
    they were made. The build time is left out, so the same build twice has the same version."""
    manifest = IndexManifest.model_validate_json((directory / "manifest.json").read_text())
    h = hashlib.sha256()
    h.update(manifest.model_dump_json(exclude={"built_at", "build_seconds", "ingest_version"}).encode())
    h.update((directory / "chunks.jsonl").read_bytes())
    h.update((directory / "vectors.npy").read_bytes())
    return h.hexdigest()[:12]


def git_sha() -> str:
    try:
        done = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    except FileNotFoundError:
        return "unknown"
    return done.stdout.strip() or "unknown"


class PipelineManifest(BaseModel):
    """What produced a result. Carried by every saved run and by every release."""

    version: str  # the pipeline version's name, from eval/versions.yaml, or `default`
    ingest_version: str  # package version and commit the records were produced at
    chunking_version: str
    embedding_version: str
    index_version: str
    prompt_version: str
    retrieval_version: str
    generator: str = ""  # model id
    backend: str = ""  # how it was called
    eval_run_id: str | None = None  # the saved run under eval/runs/ that measured this pipeline
    git_sha: str = ""
    created_at: str = ""

    @classmethod
    def of(
        cls,
        version: str,
        index: IndexManifest,
        index_id: str,
        prompt: Prompt,
        backend: str,
        retrieval: Retrieval,
        generator: str = "",
        eval_run_id: str | None = None,
    ) -> "PipelineManifest":
        return cls(
            version=version,
            ingest_version=index.ingest_version,
            chunking_version=chunking_version(index.chunking),
            embedding_version=embedding_version(index.model_id, index.revision),
            index_version=index_id,
            prompt_version=prompt.version(backend),
            retrieval_version=retrieval_version(version, retrieval),
            generator=generator,
            backend=backend,
            eval_run_id=eval_run_id,
            git_sha=git_sha(),
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        )

    def differs(self, other: "PipelineManifest") -> list[str]:
        """The stage versions that are not the same in `other`, by field name."""
        fields = (
            "ingest_version",
            "chunking_version",
            "embedding_version",
            "index_version",
            "prompt_version",
            "retrieval_version",
            "generator",
        )
        return [f for f in fields if getattr(self, f) != getattr(other, f)]

    def lines(self) -> list[str]:
        return [
            f"version     {self.version}",
            f"ingest      {self.ingest_version}",
            f"chunking    {self.chunking_version}",
            f"embedding   {self.embedding_version}",
            f"index       {self.index_version}",
            f"prompt      {self.prompt_version}",
            f"retrieval   {self.retrieval_version}",
            f"generator   {self.generator or '–'}" + (f" via {self.backend}" if self.backend else ""),
            f"eval run    {self.eval_run_id or '–'}",
        ]
