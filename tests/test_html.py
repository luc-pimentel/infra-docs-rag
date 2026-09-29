from conftest import FIXTURES

from infra_docs_rag.ingest.models import ParseStatus
from infra_docs_rag.ingest.parsers.base import assemble
from infra_docs_rag.ingest.parsers.html import parse_html

PAGE = """<html><head><title>Probes | Example Docs</title></head><body>
<nav><a href="/">Home</a> <a href="/docs/">Docs</a> <a href="/blog/">Blog</a></nav>
<main><article>
<h1>Probes<a class="td-heading-self-link" href="#probes"></a></h1>
<p>The kubelet runs probes against every container to decide whether it is healthy, whether it
has finished starting, and whether it is ready to receive traffic from a Service.</p>
<h2>Liveness probes</h2>
<p>When a liveness probe keeps failing, the kubelet kills the container and restarts it according
to the restart policy of the Pod, which recovers applications stuck in a deadlock.</p>
<h4>Note:</h4>
<p>Probes that call external dependencies can restart healthy containers during an outage.</p>
</article></main>
<footer><p>Copyright 2026 Example Docs authors. Privacy policy. Terms of use.</p></footer>
</body></html>"""


def test_main_content_keeps_headings_and_drops_navigation():
    parsed = parse_html(PAGE.encode(), "probes.html")
    text, sections, _ = assemble(parsed.parts)
    assert parsed.status == ParseStatus.OK
    assert [s.title for s in sections] == ["Probes", "Liveness probes"]
    assert "Note:" in text  # a callout label, not a section
    assert "Blog" not in text and "Privacy policy" not in text
    assert "Blog" in parsed.raw_text  # the naive extraction still had it


def test_navigation_only_page_is_empty():
    parsed = parse_html((FIXTURES / "boilerplate-only.html").read_bytes(), "boilerplate-only.html")
    assert parsed.status == ParseStatus.EMPTY
