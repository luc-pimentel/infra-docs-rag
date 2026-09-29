# infra-docs-rag

Ask questions about Kubernetes, Prometheus and Argo CD and get answers grounded in their public
documentation, each one citing the document, page and section it came from. When the docs don't
cover a question, the service says so instead of guessing.

**Status:** work in progress. Unticked items below are planned, not built.

## Roadmap

Each stage ships into this repo, so the service grows with the checklist.

**Getting the data in**

- [x] 01 · Document ingestion: PDF, HTML and text into normalized records, with deduplication,
  provenance and parse diagnostics ([report](reports/ingestion.md))
- [ ] 02 · Embeddings for semantic retrieval
- [ ] 03 · Chunking and document segmentation

**Search**

- [ ] 04 · Top-k retrieval and index design
- [ ] 05 · Hybrid search: dense + keyword
- [ ] 06 · Reranking

**Answers**

- [ ] 07 · Grounded generation with citations

**Proof**

- [ ] 08 · Evaluation: retrieval quality, groundedness, abstention and regression checks

**Production**

- [ ] 09 · Pipeline orchestration, versioning and failure isolation
- [ ] 10 · System design and observability
- [ ] 11 · Latency and cost optimization
- [ ] 12 · Capstone: FastAPI service, eval report across three versions, design doc, demo and a
  postmortem

## Corpus

Kubernetes, Prometheus and Argo CD docs as HTML pages and Markdown sources, Argo CD's commented
reference config as plain text, and two CNCF security whitepapers as PDFs. [`sources.yaml`](sources.yaml)
lists every document with its license. A few broken-on-purpose files (a scan, a blank page, a
truncated and an encrypted PDF, a navigation-only page) exercise the failure paths.

## Ingestion

Every source becomes one record: raw and cleaned text, title, section and page spans, parse status,
and provenance (source hash, parser, pipeline version). Spans point into the cleaned text, so any
chunk cut from it traces back to its file, page and section.

- **HTML:** trafilatura strips navigation, sidebars and footers; headings are kept as sections.
- **Markdown:** front matter gives the title; Hugo shortcodes, comments and link targets are removed.
- **PDF:** PyMuPDF with layout-aware paragraphs, font-size headings, header, footer and table of
  contents removal, and Tesseract OCR for pages without a text layer.
- **Duplicates:** exact copies by content hash, near copies by Jaccard similarity over 5-word
  shingles.
- **Failures:** damaged, encrypted, blank and navigation-only files get a status and a reason
  instead of crashing the run.

## Run it

Needs [uv](https://docs.astral.sh/uv/), plus Tesseract for scanned PDFs (`brew install tesseract`).

```sh
uv sync
uv run infra-docs-rag all   # download the sources, ingest them, write the report
uv run pytest
```

Downloads go to `data/raw/`, records to `data/processed/documents.jsonl`, and the evidence to
[`reports/ingestion.md`](reports/ingestion.md).
