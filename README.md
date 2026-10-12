# infra-docs-rag

Ask questions about Kubernetes, Prometheus and Argo CD and get answers grounded in their public
documentation, each one citing the document, page and section it came from. When the docs don't
cover a question, the service says so instead of guessing.

```sh
uv run infra-docs-rag answer "How do I undo a bad release?"
```

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
- [x] 05 · Hybrid search: a BM25 index over the same chunks, fused with the cosine ranking by reciprocal
  rank or weighted score, compared on the benchmark plus ten questions written for keyword search to win
  ([report](reports/hybrid-search.md))
- [x] 06 · Reranking: a cross-encoder reads the question with each of the first stage's top candidates
  and reorders them; two rerankers at two depths compared on quality, latency and cost, with the
  before-and-after context for every question it changes ([report](reports/reranking.md))

**Answers**

- [x] 07 · Grounded generation with citations: Claude answers from the retrieved passages, every
  statement tied to the passage it came from, and replies with a fixed sentence when the passages do
  not cover the question; benchmarked on the labelled questions plus an adversarial set, each answer
  read by a second model for grounding ([report](reports/grounding.md))

**Proof**

- [x] 08 · Evaluation: the benchmark as a versioned spec with five label axes, every run saved with its
  pipeline's manifest, the judge and the mechanical checks calibrated against hand labels, and
  regression gates that compare two pipeline versions end to end and block a degrading change
  ([spec](eval/SPEC.md), [calibration](reports/calibration.md), [evaluation](reports/evaluation.md))

**Production**

- [x] 09 · Pipeline orchestration, versioning and failure isolation: the stages as a graph with typed
  artifacts between them, each run skipping what is up to date and stopping at the stage that failed;
  every result carrying the version of the prompt, chunking, embedding model, index and retrieval that
  produced it; indexes built next to the live one and switched when they pass their checks; releases
  that clear the gates, and a rollback ([contracts](docs/pipeline.md), [report](reports/orchestration.md))
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
included, and the index is plain files under `data/index/<model>/<version>/`: the vectors, the chunks
with their provenance, and a manifest naming the model, its pinned revision, the vector size, the metric
and how the chunks were cut. The version is the hash of that content, and `data/index/<model>/current`
names the one that answers questions (see Orchestration).

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
own model, keeps the chunks that match the filter, ranks them (by the fused cosine and BM25 scores unless
`--mode` says otherwise, see Hybrid search, then reordered by a cross-encoder, see Reranking), drops any
under the similarity threshold,
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

## Hybrid search

Cosine similarity blurs exact terms: a config key, a flag, a metric name reads like its neighbours
(`loadBalancerIP` like `loadBalancerClass`, `api_http_requests_total` like `http_requests_total`).
A BM25 index over the same chunks ranks by the terms themselves, and `retrieve` and `search` fuse the
two rankings by default.

- **Tokenizer:** an identifier goes into the BM25 index whole and in parts, so `maxSurge` yields
  `maxsurge`, `max` and `surge`: the parts let "max surge" reach it, the whole token tells it from
  `maxUnavailable`.
- **Fusion:** `--fusion rrf` adds `1 / (k + rank)` per list; `--fusion weighted` rescales each list's
  scores to [0, 1] and adds them with `--alpha` on the dense side. The default is weighted, α=0.5,
  over the top `--depth` of each ranking.
- **Modes:** `--mode dense` is the cosine ranking alone, `--mode lexical` BM25 alone, `--mode hybrid`
  the fusion. Filters apply before ranking in every mode. The threshold stays a cosine floor: in hybrid
  mode it reads a hit's dense component, and lexical mode has none, since a BM25 score has no fixed
  scale and a zero is not an abstention signal: a common word always matches something.
- **Benchmark:** [`eval/hybrid.yaml`](eval/hybrid.yaml) adds ten identifier questions to the retrieval
  benchmark, each with a hypothesis written first. `evaluate-hybrid` ranks every question under dense,
  BM25 and four fusions and writes the [report](reports/hybrid-search.md): where dense loses, what the
  fusion keeps and costs, and what still fails.

## Reranking

The first stage scores the question and each chunk separately and compares vectors, so it can rank
every chunk but cannot see how the words interact. A cross-encoder reads the question and one chunk
together and scores how well the chunk answers it. That costs one model pass per pair, so `retrieve`
runs it only over the first stage's top candidates and reorders them.

- **Rerankers:** `bge-base` (`BAAI/bge-reranker-base`, 278M parameters) by default, `minilm-l6`
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`, 22M) for a twelfth of the cost; both pinned to a commit.
  Scores are relevances in [0, 1] after a sigmoid, so a floor means the same for either.
- **Flags:** `--rerank none|minilm-l6|bge-base`, `--candidates` for how many first-stage hits it reads
  (20 by default), `--min-relevance` for a floor on its score. The cosine threshold still reads the
  first-stage hit underneath each reranked one. `search` shows where the first stage had each hit.
- **Benchmark:** `evaluate-rerank` runs the first stage alone and each reranker over 20 and 50
  candidates on the 59 labelled questions, times each step, and writes the
  [report](reports/reranking.md): every question whose context changes with the order before and
  after, latency and cost per configuration, what the reranker's relevance says about questions the
  corpus cannot answer, and whether the expectations in [`eval/rerank.yaml`](eval/rerank.yaml) held.
  The default is the best configuration under two seconds a question.

## Answers

`answer` retrieves the passages, then asks Claude (`claude-opus-5-5`) to answer from them and nothing
else; the answer prints with `[n]` after each cited span and the sources under it. Two ways to reach
the model:

- `--generator claude-opus-5-5` (the default): the API through the official SDK. Each passage goes in
  as a document block with the API's citations enabled, so every cited span comes back tied to the
  passage it was taken from rather than to a number the model typed. Credentials come from
  `ANTHROPIC_API_KEY` or an `ant auth login` profile.
- `--generator claude-code`: the same prompt through Claude Code's `claude -p` (no tools, one turn, an
  empty working directory), billed to the subscription the CLI is logged into. There is no citation
  API there, so the model types each citation as the passage number and a verbatim quote, and the
  quote is checked against the passage: one that is not in it is dropped and the sentence counts as
  uncited. A citation here is one that checked out, not one the model merely typed.

- **Abstention, twice:** a question whose top chunk is under the cosine floor gets *"The indexed
  documentation does not cover this question."* without a call. One that passes the floor but whose
  passages do not answer it gets the same sentence from the model, which the system prompt asks for
  verbatim; a reply with no citation at all counts as one too.
- **Grounding measures:** the share of answer text that carries a citation, the uncited sentences,
  and, for a labelled question, whether a cited chunk is the right one.
- **Prompt:** `--show-prompt` prints the request as the model sees it. `--effort` sets how hard the
  model thinks (`medium` by default). Nothing is read from the repository. The system prompt is an entry
  in a small registry (`generate/prompts.py`), named and hashed; `--prompt` picks one.
- **Version:** with no retrieval flag, `answer`, `retrieve` and `search` run the current release's
  pipeline version (see Orchestration); any flag wins over it, and `--version <name>` runs a version
  from `eval/versions.yaml`.
- **Benchmark:** `evaluate-grounding` answers the 59 labelled questions and the adversarial set in
  [`eval/grounding.yaml`](eval/grounding.yaml) (instructions planted in the question, false premises,
  memory bait, half-covered questions, a question in Portuguese), each with an expectation written first,
  and writes the [report](reports/grounding.md): the prompt, eight answers in full, every question's
  outcome, the out-of-scope and adversarial responses with an error analysis, cost and latency, and
  what the judge read. `--limit N` runs the first N questions to try it cheaply.
- **Judge:** every answer is also read by Jev (TypeSafe AI's System One model) through the `jev` CLI, a model that
  returns calibrated probabilities and never text: is every claim in the passages, was a fact added
  that no passage gives, and does the answer name what the passages do not cover or reject a false
  premise. The probabilities sit next to the mechanical checks and change no verdict; a reading in the
  0.4–0.6 band is flagged as one the judge cannot tell and a person should read. `--no-judge` skips it.

## Evaluation

Every stage so far measured itself; this one makes the measurement an artifact. [`eval/SPEC.md`](eval/SPEC.md)
is the evaluation spec: the 69 questions as one versioned benchmark, the schema for expected evidence,
expected abstention and acceptable citation behavior, the labeling guidelines, the metric definitions and
the gates.

- **Saved runs:** `evaluate-grounding` writes every answer to `eval/runs/<name>/` next to a manifest
  naming the git commit, the index, the retrieval configuration, the generator, its effort and a hash of
  the prompt, plus the lineage: the version of every stage that produced the answers (see Orchestration).
  `--version` picks a pipeline from [`eval/versions.yaml`](eval/versions.yaml) (`v1-dense`, the stage 04
  pipeline; `v2-hybrid-rerank`, the current default), each naming its retrieval, prompt and embedding
  model; `--save` names the run. Labels and comparisons read saved runs, so no number costs a second
  model call.
- **Calibration:** [`eval/labels.yaml`](eval/labels.yaml) holds a person's reading of 20 answers from the
  default pipeline's run, on five axes (grounded, fabricated, gap named, correct, citations), chosen to
  include every answer where the judge and the mechanical checks disagreed. `calibrate` compares each
  automatic signal to the person, agreement, κ, precision and recall, sweeps each judge probability
  across thresholds, lists every disagreement with the answer in full, and writes the
  [report](reports/calibration.md), whose last section is the revised scoring rules. Three came out of
  this round: the judge now reads each passage under the heading the model saw it under (page and
  section claims come from there, and without it the two page questions read as fabrication); a
  sentence that names what the passages do not cover is left out of the cited share that decides a
  verdict; and the judge's grounded floor is 0.50, not 0.70, since nothing under 0.70 was ungrounded by
  hand. The constants in `evaluation/metrics.py` are those rules.
- **Regression gates:** `evaluate-pipeline --baseline <run> --candidate <run>` reads two saved runs,
  computes the retrieval, groundedness and citation, abstention, cost and latency metrics on each,
  applies [`eval/gates.yaml`](eval/gates.yaml), lists the questions whose result changed with the
  regressions in full, and writes the [report](reports/evaluation.md). A critical gate that fails exits
  1, which is what blocks a change; a non-critical one warns. The shipped comparison is v1 against v2,
  which is the question stages 05 and 06 answered one metric at a time, asked once more end to end.

## Orchestration

The stages above are one graph, and `pipeline run` runs it: fetch, ingest, chunk, embed, and with
`--generator`, evaluate and gate. [`docs/pipeline.md`](docs/pipeline.md) is the contract for each stage
(what it reads and writes, where its state lives, what its version hashes, where a failure stops) and
[`reports/orchestration.md`](reports/orchestration.md) is the graph exercised on the corpus: a run, a
failure, a retry, a re-index, a refused release, a rollback.

- **Up to date:** every artifact has a content fingerprint (the records' hashes and parse statuses,
  the chunks' bytes, the index version), and a stage whose last run saw the same inputs under the same
  configuration, with its outputs intact, is skipped. Rewriting the same content is not a change.
- **Failure:** a stage that raises is recorded with its error, nothing after it runs, and the next run
  starts there. A source that will not download, a file that will not parse, a model call that fails
  inside a benchmark, a judge that is down: each is one entry in its artifact, not the end of the run.
  `pipeline status` shows where every stage stands.
- **Lineage:** every saved run and every release carries a `PipelineManifest`: the ingest, chunking,
  embedding, index, prompt and retrieval versions, each a pinned id or a hash of the content, and the
  run that measured it.
- **Index versions:** `embed` builds into a staging directory, checks the result, and renames it into
  `data/index/<model>/<version>/`; `current` is rewritten last. `pipeline index --list | --use | --remove`
  manages the versions; earlier ones stay until removed.
- **Releases:** `pipeline release --candidate <run>` compares a saved run to the current release's with
  the gates and, when it clears them, records it in [`eval/releases.yaml`](eval/releases.yaml) and makes
  it current; a critical failure refuses it. `pipeline rollback` makes the previous release current and
  points the index at its version. `answer` runs the current release unless told otherwise.

## Run it

Needs [uv](https://docs.astral.sh/uv/), plus Tesseract for scanned PDFs (`brew install tesseract`).
`answer` and `evaluate-grounding` need either an Anthropic API key or, with `--generator claude-code`,
[Claude Code](https://docs.anthropic.com/en/docs/claude-code) logged in; the judge needs
the `jev` CLI with `jev auth login` done once, or `--no-judge`.

```sh
uv sync
uv run infra-docs-rag all                # download the sources, ingest them, write the report
uv run infra-docs-rag chunk              # cut every document with each strategy, compare the chunks
uv run infra-docs-rag embed              # embed the default chunks with bge-small (--model for another)
uv run infra-docs-rag search "How do I undo a bad release?"
uv run infra-docs-rag search loadBalancerIP --mode dense    # the cosine ranking alone; --mode lexical for BM25 alone
uv run infra-docs-rag search "Why won't my pod get scheduled?" --rerank none   # the first stage, no cross-encoder
uv run infra-docs-rag answer "What does keep_firing_for do?"                   # a cited answer, or the abstention sentence
uv run infra-docs-rag answer "What does keep_firing_for do?" --show-prompt     # the request instead of the answer
uv run infra-docs-rag retrieve "What is on page 9 of the whitepaper?" --source cncf-security-whitepaper --page 9
uv run infra-docs-rag evaluate           # compare the three models on whole sections, write the report
uv run infra-docs-rag evaluate-chunking  # compare chunking strategies and sizes, write the report
uv run infra-docs-rag evaluate-retrieval # run the retrieval benchmark on the default index, write the report
uv run infra-docs-rag evaluate-hybrid    # compare dense, BM25 and fused rankings on the benchmark, write the report
uv run infra-docs-rag evaluate-rerank    # rerank the first stage with each cross-encoder, write the report
uv run infra-docs-rag evaluate-grounding                         # answer every labelled and adversarial question, write the report (calls the API)
uv run infra-docs-rag evaluate-grounding --generator claude-code # the same through `claude -p`, on the Claude Code subscription
uv run infra-docs-rag evaluate-grounding --generator claude-code --version v1-dense --no-report   # a named pipeline version, saved under eval/runs/
uv run infra-docs-rag calibrate                                  # the judge and the checks against eval/labels.yaml, write the report
uv run infra-docs-rag evaluate-pipeline --baseline v1-dense-claude-code --candidate v2-hybrid-rerank-claude-code   # the gates; exit 1 when one blocks
uv run infra-docs-rag evaluate-pipeline --list-runs              # the saved runs; --list-metrics for the names a gate can use
uv run infra-docs-rag pipeline run                               # fetch → ingest → chunk → embed, skipping what is up to date
uv run infra-docs-rag pipeline run --from chunk --size 256       # a second chunking and index version, next to the first
uv run infra-docs-rag pipeline status                            # where each stage stands, the index versions, the releases
uv run infra-docs-rag pipeline index --use 3971d9f1e10e          # point bge-small at an index version; --list, --remove
uv run infra-docs-rag pipeline release --candidate v2-hybrid-rerank-claude-code   # promote a run's pipeline past the gates; exit 1 when refused
uv run infra-docs-rag pipeline rollback                          # the previous release again
uv run pytest
```

Downloads go to `data/raw/`, records to `data/processed/documents.jsonl`, chunks to `data/chunks/`,
indexes to `data/index/<model>/<version>/`, the orchestrator's state to `data/pipeline/state.json`, and
the evidence to [`reports/ingestion.md`](reports/ingestion.md),
[`reports/chunking.md`](reports/chunking.md), [`reports/embeddings.md`](reports/embeddings.md),
[`reports/retrieval.md`](reports/retrieval.md), [`reports/hybrid-search.md`](reports/hybrid-search.md),
[`reports/reranking.md`](reports/reranking.md), [`reports/grounding.md`](reports/grounding.md),
[`reports/calibration.md`](reports/calibration.md), [`reports/evaluation.md`](reports/evaluation.md) and
[`reports/orchestration.md`](reports/orchestration.md); saved benchmark runs go to `eval/runs/` and
releases to `eval/releases.yaml`, both committed.
The first `embed` downloads its model from Hugging Face (about 130 MB for bge-small; `evaluate`
needs all three, about 660 MB). The first `retrieve` or `search` downloads the default reranker
(about 1.1 GB for bge-base; `evaluate-rerank` needs both, about 1.2 GB); `--rerank none` needs no
download.
