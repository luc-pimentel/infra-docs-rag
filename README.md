# infra-docs-rag

Ask questions about Kubernetes, Prometheus and Argo CD and get answers grounded in their public
documentation, each one citing the document, page and section it came from. When the docs don't
cover a question, the service says so instead of guessing.

**Status:** work in progress. Unticked items below are planned, not built.

## Roadmap

Each stage ships into this repo, so the service grows with the checklist.

**Getting the data in**

- [ ] 01 · Document ingestion: PDF, HTML and text into normalized records, with deduplication,
  provenance and parse diagnostics
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

Public documentation from Kubernetes, Prometheus and Argo CD: HTML pages, Markdown files and PDFs,
including a few scanned PDFs to exercise the OCR fallback.
