"""Generate the broken-on-purpose files behind the parse-failure table.

Run from the repo root: `uv run python scripts/make_fixtures.py`
"""

from pathlib import Path

import pymupdf

FIXTURES = Path("tests/fixtures")
SCAN_TEXT = (
    "Liveness and readiness probes\n\n"
    "A failing liveness probe tells the kubelet to restart the container. "
    "A failing readiness probe keeps the Pod out of Service endpoints until it recovers, "
    "so traffic only reaches replicas that can answer."
)
FILLER = "Rolling updates replace Pods gradually, keeping the Deployment available. " * 60
BOILERPLATE_HTML = """<!doctype html>
<html>
<head><title>Docs home | Example Project</title></head>
<body>
<header><nav><a href="/">Home</a> <a href="/docs/">Docs</a> <a href="/blog/">Blog</a>
<a href="/community/">Community</a></nav></header>
<div class="cookie-banner">We use cookies to improve your experience. <button>Accept</button></div>
<aside><ul><li><a href="/v1.31/">v1.31</a></li><li><a href="/v1.30/">v1.30</a></li></ul></aside>
<footer><a href="/privacy/">Privacy</a> <a href="/terms/">Terms</a>
<p>&copy; 2026 Example Project Authors</p></footer>
</body>
</html>
"""


def text_pdf(text: str) -> pymupdf.Document:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_textbox(pymupdf.Rect(72, 72, 540, 740), text, fontsize=12)
    return doc


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)

    # A scan: the page is a picture of text, with no text layer.
    source = text_pdf(SCAN_TEXT)
    picture = source[0].get_pixmap(dpi=150)
    scan = pymupdf.open()
    scan.new_page(width=source[0].rect.width, height=source[0].rect.height).insert_image(
        source[0].rect, pixmap=picture
    )
    scan.save(FIXTURES / "scanned.pdf", garbage=4, deflate=True)

    blank = pymupdf.open()
    blank.new_page()
    blank.save(FIXTURES / "blank.pdf")

    whole = text_pdf(FILLER).tobytes(garbage=4, deflate=True)
    (FIXTURES / "truncated.pdf").write_bytes(whole[: int(len(whole) * 0.4)])

    text_pdf(SCAN_TEXT).save(
        FIXTURES / "encrypted.pdf",
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="owner-secret",
        user_pw="reader-secret",
    )

    (FIXTURES / "boilerplate-only.html").write_text(BOILERPLATE_HTML)
    print(f"wrote fixtures to {FIXTURES}/")


if __name__ == "__main__":
    main()
