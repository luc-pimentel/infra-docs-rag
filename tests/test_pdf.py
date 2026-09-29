import shutil

import pymupdf
import pytest
from conftest import FIXTURES

from infra_docs_rag.ingest.clean import join_wrapped_lines
from infra_docs_rag.ingest.models import ParseStatus
from infra_docs_rag.ingest.parsers.base import assemble
from infra_docs_rag.ingest.parsers.pdf import parse_pdf

TOPICS = ["scheduling", "networking", "storage", "observability"]


@pytest.mark.parametrize(
    ("name", "status"),
    [
        ("truncated.pdf", ParseStatus.FAILED),
        ("encrypted.pdf", ParseStatus.FAILED),
        ("blank.pdf", ParseStatus.EMPTY),
    ],
)
def test_broken_files_get_a_status_instead_of_crashing(name, status):
    parsed = parse_pdf((FIXTURES / name).read_bytes(), name)
    assert parsed.status == status
    assert parsed.error


@pytest.mark.skipif(shutil.which("tesseract") is None, reason="needs Tesseract installed")
def test_scanned_page_is_read_with_ocr():
    parsed = parse_pdf((FIXTURES / "scanned.pdf").read_bytes(), "scanned.pdf")
    text, _, pages = assemble(parsed.parts)
    assert parsed.status == ParseStatus.OCR_FALLBACK
    assert pages[0].method == "ocr"
    assert "liveness probe" in text.lower()


def test_running_headers_and_page_numbers_are_removed():
    doc = pymupdf.open()
    for number, topic in enumerate(TOPICS, start=1):
        page = doc.new_page()
        page.insert_text((72, 40), "ACME Platform Handbook", fontsize=9)
        page.insert_text((72, 100), f"Chapter {number}", fontsize=20)
        body = f"This chapter explains how the platform handles {topic} for every team. " * 6
        page.insert_textbox(pymupdf.Rect(72, 130, 540, 400), body, fontsize=11)
        page.insert_text((300, 780), str(number), fontsize=9)
    parsed = parse_pdf(doc.tobytes(), "handbook.pdf")
    text, sections, pages = assemble(parsed.parts)
    assert "ACME Platform Handbook" not in text
    assert [s.title for s in sections] == [f"Chapter {n}" for n in range(1, 5)]
    assert [p.number for p in pages] == [1, 2, 3, 4]
    assert not any(line.strip().isdigit() for line in text.splitlines())


def test_wrapped_lines_and_soft_hyphens_are_rejoined():
    assert (
        join_wrapped_lines(["Kubernetes orches­", "trates containers"]) == "Kubernetes orchestrates containers"
    )
    assert join_wrapped_lines(["a cloud-", "native platform"]) == "a cloud-native platform"
