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
- [x] 02 · Embeddings for semantic retrieval: every section embedded with local models into a cosine
  index you can search, with three models compared on labelled queries ([report](reports/embeddings.md))
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

## Embeddings

Each section from ingestion becomes one chunk, heading included, until real chunking lands. A local
[sentence-transformers](https://www.sbert.net/) model embeds every chunk, and the index is plain
files under `data/index/<model>/`: the vectors, the chunks with their provenance, and a manifest
naming the model, its pinned revision, the vector size and the metric.

- **Models:** `bge-small` by default, plus `bge-base` and `minilm`, each pinned to a Hugging Face
  commit. BGE queries get the instruction its model card recommends; documents don't.
- **Search:** vectors are stored at length 1, so a dot product gives the cosine similarity. Search
  refuses any model other than the one the index was built with: vectors from two models are not
  comparable, even when they are the same size.
- **Evaluation:** [`eval/queries.yaml`](eval/queries.yaml) holds labelled queries (reworded questions,
  config keys, and questions that depend on metadata), each with a hypothesis written before it was
  scored. `evaluate` probes every model against a small hand-picked set of chunks, compares them on
  the full index, and writes the [report](reports/embeddings.md).

## Run it

Needs [uv](https://docs.astral.sh/uv/), plus Tesseract for scanned PDFs (`brew install tesseract`).

```sh
uv sync
uv run infra-docs-rag all        # download the sources, ingest them, write the report
uv run infra-docs-rag embed      # embed every section with bge-small (--model to pick another)
uv run infra-docs-rag search "How do I undo a bad release?"
uv run infra-docs-rag evaluate   # rebuild all three indexes, compare them, write the report
uv run pytest
```

Downloads go to `data/raw/`, records to `data/processed/documents.jsonl`, indexes to `data/index/`,
and the evidence to [`reports/ingestion.md`](reports/ingestion.md) and
[`reports/embeddings.md`](reports/embeddings.md). The first `embed` downloads its model from Hugging
Face (about 130 MB for bge-small; `evaluate` needs all three, about 660 MB).
