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
- [x] 03 · Chunking and document segmentation: fixed, sentence and section-aware chunks compared at
  three sizes on labelled queries; every chunk an exact, cited slice of its document, sized to what the
  model reads ([report](reports/chunking.md))

**Search**

- [x] 04 · Top-k retrieval and index design: a retriever with metadata filters applied before ranking, a
  similarity threshold that returns nothing for questions the docs don't cover, and prompt-ready context
  with numbered, cited passages, tuned on a 49-question benchmark ([report](reports/retrieval.md))
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

## Chunking

Retrieval returns chunks, not whole documents. Four strategies cut the same cleaned text, sized in
the embedding model's own tokens:

- **Whole sections:** one chunk per section, however long. The embedding stage's baseline.
- **Fixed:** windows of whole words, the same number of tokens each.
- **Sentences:** whole sentences, and whole lines for lists, code and config, packed up to the size.
- **Section-aware:** one chunk per section when it fits. A longer section is cut at paragraphs, then
  lines, sentences and words, and every piece keeps the section's path.

A chunk's text is always an exact slice of its document's cleaned text, so its offsets, section path,
pages and project trace it back to the source. An optional context line (project, section path and
pages) is embedded in front of the text, so a piece cut from the middle of a section still says what
it is about.

`evaluate-chunking` runs every strategy at 128, 256 and 512 tokens, with and without the context
line, against the labelled queries and writes the [report](reports/chunking.md). `embed` uses
section-aware chunks of up to 512 tokens with the context line: a section stays whole unless the
model can't read it in one piece, and every chunk cites exactly one section.

## Embeddings

A local [sentence-transformers](https://www.sbert.net/) model embeds every chunk, context line
included, and the index is plain files under `data/index/<model>/`: the vectors, the chunks with their
provenance, and a manifest naming the model, its pinned revision, the vector size, the metric and how
the chunks were cut.

- **Models:** `bge-small` by default, plus `bge-base` and `minilm`, each pinned to a Hugging Face
  commit. BGE queries get the instruction its model card recommends; documents don't.
- **Search:** vectors are stored at length 1, so a dot product gives the cosine similarity. Search
  refuses any model other than the one the index was built with: vectors from two models are not
  comparable, even when they are the same size.
- **Evaluation:** [`eval/queries.yaml`](eval/queries.yaml) holds labelled queries (reworded questions,
  config keys, and questions that depend on metadata), each with a hypothesis written before it was
  scored. `evaluate` probes every model against a small hand-picked set of chunks, compares them on
  whole-section indexes it keeps in memory, and writes the [report](reports/embeddings.md).

## Retrieval

`retrieve` turns a question into the passages a generator reads. It embeds the question with the index's
own model, keeps the chunks that match the filter, ranks them, drops any under the similarity threshold,
and packs the top k in rank order, each under a numbered heading that names its project, section and
pages, with a source list after them. An answer can then cite `[2]`, and the list says what `[2]` is.

- **Filters:** `--project`, `--source`, `--section` and `--page` narrow the eligible chunks *before*
  ranking, so the top k is the top k of what qualifies. `--post-filter` shows the alternative, filtering
  the unfiltered top k, which can leave nothing.
- **Threshold:** `--min-score` is the cosine similarity a chunk needs. Under it, the retriever returns no
  passage rather than the least wrong one, which is how a question the docs don't cover gets "the docs
  don't cover this" instead of a confident guess. The default, 0.50, is a floor that catches unrelated
  questions and costs no answer; the report shows why a near miss needs a different signal.
- **Budget:** `--budget` caps the tokens the passages add up to.
- **Benchmark:** [`eval/retrieval.yaml`](eval/retrieval.yaml) holds 49 labelled questions: the embedding
  stage's 11, 32 more written against the corpus, and 6 the corpus cannot answer. `evaluate-retrieval`
  ranks every chunk for every question once and reads off what k, the threshold and the filters each
  change, times exact search up to a million synthetic vectors, and writes the [report](reports/retrieval.md).

`search` prints the same ranking as raw hits with their scores, and marks the ones the threshold would
drop.

## Run it

Needs [uv](https://docs.astral.sh/uv/), plus Tesseract for scanned PDFs (`brew install tesseract`).

```sh
uv sync
uv run infra-docs-rag all                # download the sources, ingest them, write the report
uv run infra-docs-rag chunk              # cut every document with each strategy, compare the chunks
uv run infra-docs-rag embed              # embed the default chunks with bge-small (--model for another)
uv run infra-docs-rag search "How do I undo a bad release?"
uv run infra-docs-rag retrieve "What is on page 9 of the whitepaper?" --source cncf-security-whitepaper --page 9
uv run infra-docs-rag evaluate           # compare the three models on whole sections, write the report
uv run infra-docs-rag evaluate-chunking  # compare chunking strategies and sizes, write the report
uv run infra-docs-rag evaluate-retrieval # run the retrieval benchmark on the default index, write the report
uv run pytest
```

Downloads go to `data/raw/`, records to `data/processed/documents.jsonl`, chunks to `data/chunks/`,
indexes to `data/index/`, and the evidence to [`reports/ingestion.md`](reports/ingestion.md),
[`reports/chunking.md`](reports/chunking.md), [`reports/embeddings.md`](reports/embeddings.md) and
[`reports/retrieval.md`](reports/retrieval.md).
The first `embed` downloads its model from Hugging Face (about 130 MB for bge-small; `evaluate`
needs all three, about 660 MB).
