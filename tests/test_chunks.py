import pytest

from infra_docs_rag.chunk.chunks import Chunking, chunk_documents
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

# The same guide with a Scaling section of 43 words, too long for a 20-token chunk
LONG = (
    GUIDE
    + """
Autoscaling picks the number for you. It watches CPU use and adds Pods when the load grows.

Scaling to zero stops every Pod but keeps the Deployment, so it can come back later.
"""
)
SIZED = ("fixed", "sentences", "section-aware")


def sectioned(make_doc, source_id: str = "guide", markdown: str = GUIDE):
    text, sections, _ = assemble(markdown_parts(markdown))
    doc = make_doc(source_id, text)
    doc.sections = sections
    return doc


@pytest.fixture
def tokenizer(fake_embedder):
    return fake_embedder(max_tokens=1000)  # one token per word


def test_one_chunk_per_section_that_has_text_under_its_heading(make_doc):
    doc = sectioned(make_doc)
    chunks = chunk_documents([doc], Chunking())
    # "Rolling Back a Deployment" is followed straight by its first child, so it has no text of its own
    assert [c.section for c in chunks] == ["Deployments", "Checking Rollout History", "Scaling"]
    for chunk in chunks:
        assert chunk.text == doc.cleaned_text[chunk.start : chunk.end]
        assert chunk.text.startswith(chunk.section)
    assert (
        chunks[1].citation() == "guide › Deployments › Rolling Back a Deployment › Checking Rollout History"
    )


def test_duplicates_and_failed_parses_are_skipped(make_doc, tokenizer):
    kept, copy, broken = (sectioned(make_doc, name) for name in ("kept", "copy", "broken"))
    copy.duplicate = Duplicate(of=kept.document_id, kind="exact", similarity=1.0)
    broken.parse_status = ParseStatus.FAILED
    for chunking in (Chunking(), Chunking(strategy="section-aware", size=20)):
        assert {c.source_id for c in chunk_documents([kept, copy, broken], chunking, tokenizer)} == {"kept"}


def test_chunks_cite_the_pdf_pages_they_span(make_doc):
    doc = sectioned(make_doc)
    scaling = doc.sections[-1].start
    doc.pages = [
        Page(number=1, method="text", start=0, end=scaling - 2),
        Page(number=2, method="text", start=scaling, end=len(doc.cleaned_text)),
    ]
    chunks = chunk_documents([doc], Chunking())
    assert [c.pages for c in chunks] == [[1], [1], [2]]
    assert chunks[-1].citation() == "guide › Deployments › Scaling, page 2"


def test_section_aware_keeps_short_sections_whole_and_cuts_long_ones_at_paragraphs(make_doc, tokenizer):
    doc = sectioned(make_doc, markdown=LONG)
    chunks = chunk_documents([doc], Chunking(strategy="section-aware", size=20), tokenizer)
    assert [c.section for c in chunks] == ["Deployments", "Checking Rollout History"] + ["Scaling"] * 3
    scaling = chunks[2:]
    assert scaling[0].text == "Scaling\n\nRun kubectl scale to change the number of replicas."
    assert scaling[1].text.startswith("Autoscaling picks") and scaling[1].text.endswith("load grows.")
    assert scaling[2].text.startswith("Scaling to zero")
    assert all(c.section_path == ["Deployments", "Scaling"] for c in scaling)


@pytest.mark.parametrize("strategy", SIZED)
@pytest.mark.parametrize("context", [False, True])
def test_every_chunk_is_an_exact_slice_that_fits_the_size(make_doc, tokenizer, strategy, context):
    doc = sectioned(make_doc, markdown=LONG)
    chunking = Chunking(strategy=strategy, size=40, overlap=4, context=context)
    chunks = chunk_documents([doc], chunking, tokenizer, projects={"guide": "Kubernetes"})
    assert chunks
    for chunk in chunks:
        assert chunk.text == doc.cleaned_text[chunk.start : chunk.end] == chunk.text.strip()
        assert tokenizer.token_counts([chunk.embedded])[0] <= 40
    assert len({c.chunk_id for c in chunks}) == len(chunks)


@pytest.mark.parametrize("strategy", ["fixed", "sentences"])
def test_windows_cover_the_whole_text_and_are_filed_under_the_section_they_start_in(
    make_doc, tokenizer, strategy
):
    doc = sectioned(make_doc, markdown=LONG)
    chunks = chunk_documents([doc], Chunking(strategy=strategy, size=12), tokenizer)
    assert " ".join(c.text for c in chunks).split() == doc.cleaned_text.split()  # nothing lost, no overlap
    for chunk in chunks:
        _, section = doc.locate(chunk.start)
        assert chunk.section_path == section.path


def test_sentence_chunks_end_where_a_sentence_or_a_line_does(make_doc, tokenizer):
    doc = sectioned(make_doc, markdown=LONG)
    text = doc.cleaned_text
    chunks = chunk_documents([doc], Chunking(strategy="sentences", size=20), tokenizer)
    assert len(chunks) > 3
    assert all(c.end == len(text) or text[c.end] == "\n" or text[c.end - 1] in ".!?" for c in chunks)


def test_the_context_line_names_the_project_and_section_and_the_pages_of_a_pdf(make_doc, tokenizer):
    doc = sectioned(make_doc)
    doc.pages = [Page(number=4, method="text", start=0, end=len(doc.cleaned_text))]
    chunks = chunk_documents(
        [doc], Chunking(strategy="section-aware", size=64, context=True), tokenizer, {"guide": "Kubernetes"}
    )
    assert chunks[-1].context == "Kubernetes › Deployments › Scaling, page 4"
    assert chunks[-1].embedded == f"{chunks[-1].context}\n\n{chunks[-1].text}"
    assert chunks[-1].project == "Kubernetes"
    without = chunk_documents(
        [doc], Chunking(strategy="section-aware", size=64), tokenizer, {"guide": "Kubernetes"}
    )
    assert without[-1].context == "" and without[-1].embedded == without[-1].text


def test_a_document_without_headings_is_chunked_under_its_title(make_doc, tokenizer):
    notes = make_doc("notes", "Plain notes with no headings at all. They still hold answers.")
    assert chunk_documents([notes], Chunking()) == []  # whole sections leave it out
    for strategy in SIZED:
        chunks = chunk_documents([notes], Chunking(strategy=strategy, size=64), tokenizer)
        assert [c.section_path for c in chunks] == [["notes"]]
        assert chunks[0].text == notes.cleaned_text


def test_a_size_with_no_room_left_for_text_is_refused(make_doc, tokenizer):
    doc = sectioned(make_doc)
    with pytest.raises(ValueError, match="too little room for text"):
        chunk_documents(
            [doc], Chunking(strategy="section-aware", size=16, context=True), tokenizer, {"guide": "K"}
        )
    with pytest.raises(ValueError, match="needs the model's tokenizer"):
        chunk_documents([doc], Chunking(strategy="fixed", size=64))


def test_chunking_settings_must_agree():
    with pytest.raises(ValueError, match="needs one"):
        Chunking(strategy="fixed")
    with pytest.raises(ValueError, match="takes no size"):
        Chunking(size=256)
    with pytest.raises(ValueError, match="under half"):
        Chunking(strategy="fixed", size=100, overlap=50)
    assert Chunking().label == "whole-sections"
    assert Chunking(strategy="section-aware", size=256, overlap=32, context=True).label == (
        "section-aware 256/32 + context"
    )
    assert Chunking(strategy="fixed", size=512).label == "fixed 512"
