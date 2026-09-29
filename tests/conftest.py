from pathlib import Path

import pytest

from infra_docs_rag.ingest.dedup import content_hash
from infra_docs_rag.ingest.models import Document, ParseStatus, Provenance, SourceType

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def make_doc():
    def make(source_id: str, text: str) -> Document:
        return Document(
            document_id=f"doc_{source_id}",
            source_id=source_id,
            source_uri=f"https://example.com/{source_id}",
            source_type=SourceType.TEXT,
            title=source_id,
            raw_text=text,
            cleaned_text=text,
            content_hash=content_hash(text),
            parse_status=ParseStatus.OK,
            provenance=Provenance(
                local_path=f"data/raw/{source_id}.txt",
                source_sha256="0" * 64,
                fetched_at=None,
                parser="test",
                ingest_version="test",
                ingested_at="2026-09-28T00:00:00+00:00",
            ),
        )

    return make
