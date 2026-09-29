from infra_docs_rag.embed.chunks import section_chunks
from infra_docs_rag.ingest.models import Duplicate, Page, ParseStatus
from infra_docs_rag.ingest.parsers.base import assemble, markdown_parts

GUIDE = """# Deployments

A Deployment manages a set of Pods.

## Rolling Back a Deployment

### Checking Rollout History

Run kubectl rollout history to list the revisions.

## Scaling

Run kubectl scale to change the number of replicas.
"""


def sectioned(make_doc, source_id: str = "guide"):
    text, sections, _ = assemble(markdown_parts(GUIDE))
    doc = make_doc(source_id, text)
    doc.sections = sections
    return doc


def test_one_chunk_per_section_that_has_text_under_its_heading(make_doc):
    doc = sectioned(make_doc)
    chunks = section_chunks([doc])
    # "Rolling Back a Deployment" is followed straight by its first child, so it has no text of its own
    assert [c.section for c in chunks] == ["Deployments", "Checking Rollout History", "Scaling"]
    for chunk in chunks:
        assert chunk.text == doc.cleaned_text[chunk.start : chunk.end]
        assert chunk.text.startswith(chunk.section)
    assert (
        chunks[1].citation() == "guide › Deployments › Rolling Back a Deployment › Checking Rollout History"
    )


def test_duplicates_and_failed_parses_are_skipped(make_doc):
    kept, copy, broken = (sectioned(make_doc, name) for name in ("kept", "copy", "broken"))
    copy.duplicate = Duplicate(of=kept.document_id, kind="exact", similarity=1.0)
    broken.parse_status = ParseStatus.FAILED
    assert {c.source_id for c in section_chunks([kept, copy, broken])} == {"kept"}


def test_chunks_cite_the_pdf_pages_they_span(make_doc):
    doc = sectioned(make_doc)
    scaling = doc.sections[-1].start
    doc.pages = [
        Page(number=1, method="text", start=0, end=scaling - 2),
        Page(number=2, method="text", start=scaling, end=len(doc.cleaned_text)),
    ]
    chunks = section_chunks([doc])
    assert [c.pages for c in chunks] == [[1], [1], [2]]
    assert chunks[-1].citation() == "guide › Deployments › Scaling, page 2"
