# System design: running this as a service

[`docs/pipeline.md`](pipeline.md) is the contract between stages. This document is the system around
them: the components, what talks to what over which interface, where state lives, where the deployment
boundaries fall, how the corpus is kept fresh, how a question is observed, and what happens when a part
fails. Two columns run through it: **as built** is what this repository does today, on one machine, in
one process; **as deployed** is the shape it takes with more than one machine and more than one user.
The stage code is the same in both; what changes is who runs it, where, and how its state is shared.

## Components

| Component | As built | As deployed | Interface |
|---|---|---|---|
| **Ingestion job** | `pipeline run` fetch → ingest → chunk → embed, one process, started by hand | a scheduled job (cron, Airflow, a GitHub Action) running the same command on a worker with the models cached; one run at a time | `sources.yaml` in; `data/raw/manifest.json`, `documents.jsonl`, `chunks`, a new index version out; the state file says what ran |
| **Parser workers** | `ingest` parses every source in sequence (`parsers/`); PDFs through PyMuPDF with Tesseract for scanned pages | the same parsers fanned out per source, since no source depends on another; a damaged file is one record with a status, never a failed job | `Source` in, `Document` out; the record carries its parse status and provenance |
| **Vector store** | plain files: `vectors.npy` at unit length, `chunks.jsonl`, `manifest.json`, one directory per index version under `data/index/<model>/`, a `current` pointer | the same files on object storage, pulled by each retrieval replica at start and on a version change; or, past a few hundred thousand chunks, a vector database with the manifest's fields as its collection metadata (`reports/retrieval.md` times exact search to a million vectors; it is fine to there) | `Index.load(path)`; `rank(vector, k, mask)` → hits with cosine scores |
| **Lexical index** | BM25 built in memory from the index's chunks at retriever start (`retrieve/lexical.py`), about a second for 500 chunks | built once per index version and stored next to the vectors, or a search engine's BM25 over the same chunk ids; either way it is derived from the chunks file and has no state of its own | `rank(query, k, mask)` → hits with BM25 scores |
| **Retrieval service** | `Retriever` in the `answer` process: embed the question, rank, fuse, rerank, threshold, pack | a stateless replica per index version holding the vectors, the BM25 index, the embedding model and the cross-encoder in memory; the capstone's FastAPI service is this plus generation | `Retrieval` config + question in; `Context` out (passages under numbered headings, scores, timings, what fell back) |
| **Generation service** | `ClaudeGenerator` (API, citations attached by the API) or `ClaudeCodeGenerator` (`claude -p`, citations typed and checked), one call per question, one retry | the API path only; the CLI path is for the subscription on a laptop. Prompt caching keeps the frozen system prompt cheap; the per-question passages are the variable part | `Context` + question in; `Generated` (segments with citations, usage, stop reason) out; `Answer` after reading it |
| **Eval pipeline** | `evaluate-grounding` over 69 questions → a saved run with its lineage; `calibrate`; `evaluate-pipeline`; `pipeline release` against the gates | the same, run in CI on every change to a prompt, the retrieval defaults or the chunking, against the last release's run; the judge call is the only paid part | saved runs under `eval/runs/`, the gates, the release record |
| **Monitoring** | one trace per question to `data/traces/traces.jsonl`; `traces --summary`; `pipeline health`; the canary | traces shipped to a log store (one JSON object per line is already the format) with the summary's numbers as metrics: outcome rates, stage p50/p95, cost per 1,000, degraded count, lineage drift; the canary on a schedule after each deploy | `Trace` (see below) |
| **Deployment topology** | one machine: the repository, `data/`, the models in the Hugging Face cache | see the next section | – |

## Deployment boundaries

Four boundaries, each a place where one side can change without the other noticing.

1. **Corpus → index.** The ingestion job owns `data/` up to the index version. Nothing downstream reads
   `documents.jsonl` or the chunks at query time; retrieval reads one index version. So an ingestion run
   can rebuild everything while the service answers from the current version, and the switch is the
   `current` pointer (as built) or the version id in the release the replicas are told to load (as deployed).
2. **Index → retrieval replicas.** A replica loads one index version at start and never reloads it:
   a new version is a new replica set, rolled out while the old one still answers. The index version
   is in every trace, so a question answered by the old set during a rollout says so.
3. **Retrieval → generation.** The `Context` is the whole interface: the model sees passages and a
   question, nothing about how they were ranked. The generator can be swapped, the prompt revised, or
   the model called through a different path, and retrieval is untouched; the prompt version in the
   lineage says which wording answered.
4. **Serving → evaluation.** The benchmark never runs in the serving path. It runs on the same code,
   writes a run, and the release decision reads runs. The serving path reads the release record.

## Storage layers

| Layer | What | Where, as built | Identity | Lifetime |
|---|---|---|---|---|
| Raw sources | the fetched files and the download manifest | `data/raw/` (not committed) | sha256 per file | until refetched |
| Records | normalized documents with provenance | `data/processed/documents.jsonl` | content hash + parse status per record | rebuilt by ingest; previous kept until the rename |
| Chunks | one file per chunking | `data/chunks/<chunking>.jsonl` | file bytes | one per chunking, coexisting |
| Index versions | vectors, chunks, manifest | `data/index/<model>/<version>/` | hash of content | until removed |
| Pipeline state | what each stage read, wrote, when, how it ended | `data/pipeline/state.json` | – | per machine |
| Traces | one per question | `data/traces/traces.jsonl` | trace id | append-only |
| Benchmark runs | every answer of a run with the lineage | `eval/runs/<name>/` (committed) | run name | forever |
| Config | sources, versions, gates, canary, labels | `sources.yaml`, `eval/*.yaml` (committed) | git | forever |
| Release record | the releases and which one is current | `eval/releases.yaml` (committed) | git | forever |
| Models | embedding models, cross-encoders, pinned to a commit | the Hugging Face cache | model id + revision | cache |

As deployed, the first five move to object storage keyed by the same identities, and the committed
files stay in git: they are the part a reviewer reads.

## Freshness and update flows

**A source changed upstream.** `pipeline run --refresh` downloads every URL source again. A file whose
sha256 is unchanged changes nothing downstream (the raw fingerprint is the set of sha256s). A changed
file changes the records' fingerprint, which reruns chunk and embed and produces a new index version
with a new id. The release still points at the old index version; `pipeline health` says the release
was measured on one index and another is current, and the operator either runs the benchmark under the
new index and releases it, or points back. Nothing in that chain is implicit.

**A source was added or removed.** An edit to `sources.yaml` changes the `sources` fingerprint, which
reruns fetch and everything after it. A removed source leaves its raw file on disk and its record out
of the next `documents.jsonl`.

**A document in the corpus is broken.** It is a record with `parse_status` `failed` or `empty` and a
reason; `usable()` leaves it out of chunking; the ingestion report lists it. It never reaches a user.

**The prompt or the retrieval defaults changed.** No artifact changes. A new entry in `versions.yaml`,
a benchmark run under it, and `pipeline release`; the gates compare it to the current release's run,
and the lineage's prompt and retrieval versions say what moved.

**The embedding model changed.** A new model name is a new index directory; the index refuses any
embedder but the one it was built with, so a replica cannot read the wrong one by accident.

**The schedule, as deployed:** refresh nightly; a benchmark run under the current release's version
when the index version changed; release by hand after reading the comparison; the canary after every
release and every hour.

## Observability

Every question through `answer`, the canary or a benchmark run leaves one trace: the question; the
pipeline version and its lineage (index, prompt, retrieval, embedding versions); every stage's time
(embed, dense, lexical, fuse, rerank, pack, generate, judge); what the first stage ranked, each with
its fused score, cosine and BM25; what the model read after the reranker, each with its relevance and
where it stood before; the floor's decision; the outcome and reason; the answer with its citation
markers; every citation with the passage text it rests on; the cited and claim shares; the uncited
sentences; the model, tokens and cost; the judge's three probabilities; what fell back, if anything.
[`reports/observability.md`](../reports/observability.md) has four in full and the log summarized.

`traces --summary` turns the log into the operator's numbers: outcome rates, per-stage p50 and p95,
median top cosine, median cited and claim shares, cost per thousand, degraded and error counts, which
versions and indexes answered, and flags when a rate crosses a line (no-passage share over 30 %, error
rate over 5 %, p95 over the 15 s gate, any degraded answer, median claim share under the 80 % floor,
traces from an index that is not the current one). `pipeline health` is the point-in-time version:
index present, release and index agree, index built from the current records, a way to reach the
model, the judge present, the recent traces clean.

What is **not** built: shipping the traces anywhere, sampling (every question is traced; at a few
hundred a day that is right, at a few hundred a minute the retrieval detail would be sampled and the
outcome and timings kept for all), and alerting (the flags print; nothing pages).

## Failure modes and deploy controls

| Failure mode | How it is detected | What happens now (as built) | Fallback | Deploy control |
|---|---|---|---|---|
| **Empty retrieval** (no passage over the floor) | `Context.empty`; the trace's outcome `no-passage` with the top cosine and the floor | the fixed abstention sentence, no model call, the reason printed (closest cosine vs the floor) | there is none by design: the least wrong passage is worse than no passage; the summary flags a no-passage share over 30 % as a corpus or floor problem | the floor is in the version; a change to it is a benchmark run against the out-of-scope and in-scope abstention gates |
| **Stale index** (records newer than the index, or the release measured on another index version) | `pipeline status` shows embed as stale; `pipeline health` warns when the release's index version is not the current one; the summary flags traces from a non-current index | the service answers from whatever `current` points at; nothing switches on its own | keep answering from the released index; it is the measured one | a new index version is not served until a run under it is released; `pipeline release` records the index version it was measured on |
| **Model timeout or error** | the generator raises (`ValueError` with the CLI or API error) after the 300 s CLI timeout or the SDK's | `answer`: the error propagates and the command exits 1 with the reason. Benchmark and canary: one retry, then the case is outcome `error` with the reason, the run continues; the summary flags an error rate over 5 % | the capstone's endpoint returns the abstention sentence with `outcome: error` and the trace id, never a half answer; a second model id as a fallback generator is a one-line change in `MODELS` | the canary after a deploy exercises the model path; a timeout there fails the canary |
| **Malformed document** (damaged, encrypted, blank, navigation-only, a page with no text layer) | `parse_status` and `parse_error` on the record; `reports/ingestion.md` lists them | the record is kept with its status, left out of chunking; a scanned page goes through OCR and is marked | none needed; the document is simply not in the index, and a question it would have answered abstains | an ingestion run that fails more sources than the last one should not be released; today that is read off the ingestion report by hand |
| **Prompt regression** (a wording change that loses citations, abstains more, or fabricates) | the benchmark under the new prompt against the gates: claim share median, judge fabricated, in-scope abstention, adversarial held; the canary's whitepaper question needs the passage headings | `pipeline release` refuses the run when a critical gate fails, exit 1, record untouched | `pipeline rollback` to the previous release; the prompt version is in every trace, so the questions answered under the bad prompt can be listed | a prompt is a registry entry with a hash; a version pins it; no prompt change reaches `answer` without a release |
| **Index regression** (a chunking or embedding change that loses the right passages) | the retrieval gates (right passage retrieved, MRR) on the benchmark run under the new index; the canary's identifier question | as above: refused at release | `pipeline index --use <previous>` for the index alone, or `pipeline rollback` for the whole release, which repoints the index when the version is on disk | an index version is content-addressed; the release names it; a rebuild of the same content is the same version and nothing moves |
| **Canary checks** | `pipeline canary`: five questions, each with the outcome and source it has to produce; exit 1 on any failure; traces under `canary:<version>` | run by hand after a release | a failed canary is the signal to roll back; the trace says which stage changed | as deployed, the canary gates the rollout: the new replica set gets traffic when it passes |
| **Fallback behavior** (a stage degrades rather than fails) | `Context.degraded` and the trace's `degraded` line; the summary flags any | the reranker raising falls back to the first-stage order, logged; the judge raising leaves the case without a reading; a filter that matches nothing returns no passage with the reason | each fallback is the next best thing that is still correct, never a guess: first-stage order is a worse ranking of the same passages, not different passages | a degraded answer is counted; a deploy whose canary ran degraded is not healthy |

Two controls hold the table together. **Nothing reaches `answer` without a release**, and a release is
a saved run that cleared the gates against the one before it. **Everything that answers is named by
content**, so a rollback is a pointer move and the questions answered under a bad version can be found.

## What a design review would push on

- **One corpus, three projects, 500 chunks.** The exact search, the in-memory BM25 and the
  single-process orchestrator are right at this size and would be the first three things to change at
  a hundred times it. The contracts do not change; the implementations behind them do.
- **The cross-encoder is two seconds a question on CPU.** It is the biggest latency item after the
  model; stage 11 is where its candidate count, its model and a cache are weighed against the quality it
  buys.
- **The judge is a reading, not a verdict.** It changes no decision at serving time and the gates use
  it as one signal among several. A service that blocked answers on a judge would need the judge's
  latency and failure modes in this table.
- **No authentication, rate limiting or tenancy.** The capstone's service is a demo surface behind a
  reverse proxy, not a product edge.
