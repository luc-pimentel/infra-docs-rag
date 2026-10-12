# infra-docs-rag: the design, concisely

The long version is [`design.md`](design.md) (components, boundaries, storage, failure modes) and
[`pipeline.md`](pipeline.md) (stage contracts). This is the page to read first: what was built, what
each decision cost and bought, and where the numbers are. The demo script is [`demo.md`](demo.md);
the postmortem is [`postmortem.md`](postmortem.md).

## The system in one paragraph

A question comes in. It is embedded with bge-small, ranked against 497 chunks by cosine and by BM25,
the two rankings are fused, a cross-encoder reorders the top 20, the cosine floor drops anything under
0.50, and the passages that remain are packed under numbered headings up to 1,200 tokens. Claude
answers from those passages and nothing else, each sentence tied to the passage it rests on, or
replies with one fixed sentence when they do not cover the question. A second model reads the answer
for grounding. Every step is timed and written to a trace. Which prompt, index, chunking, embedding
model and retrieval configuration did this is a content-addressed manifest on every answer and every
benchmark run; nothing reaches the service without a run that cleared ten regression gates against
the previous release.

## Corpus

Kubernetes, Prometheus and Argo CD public docs: 17 HTML and Markdown pages, Argo CD's commented
reference config as text, two CNCF security whitepapers as PDF, and five broken files on purpose (a
scan, a blank page, a truncated and an encrypted PDF, a navigation-only page). Chosen because the
questions an SRE asks of them are the three kinds retrieval has to handle: paraphrases ("how do I undo
a bad release"), identifiers (`keep_firing_for`, `maxSurge`, `--iptables-localhost-nodeports`) and
metadata ("what is on page 9"). 22 records, 14 usable, 497 chunks. [`sources.yaml`](../sources.yaml)
lists every document with its license.

## Ingestion: what was hard

- **PDFs.** Font-size headings, headers and footers repeated on every page, a table of contents that
  reads as text, and one scanned page with no text layer (Tesseract). Every chunk keeps its page
  numbers, which is what the page questions and the citations need.
- **Duplicates.** The same page reached by two URLs (with and without a trailing slash) and the
  Markdown source of an HTML page: exact copies by content hash, near copies by Jaccard over 5-word
  shingles; the copy is kept as a record and left out of the index.
- **Failure as data.** A damaged file is a record with a status and a reason, not a failed run.
  [`reports/ingestion.md`](../reports/ingestion.md).

## Retrieval: the architecture and what each piece bought

Measured on 59 labelled questions (49 plus 10 written for keyword search), each with a hypothesis
written before it was scored.

| Stage | Decision | What it bought | Report |
|---|---|---|---|
| 02 | bge-small (pinned commit) over MiniLM and bge-base | retrieval-trained, 512 tokens, same size as MiniLM; bge-base is twice the vectors for little | [embeddings](../reports/embeddings.md) |
| 03 | section-aware chunks, 512 tokens, 64 overlap, a context line embedded in front | a section stays whole unless it cannot; every chunk cites exactly one section and one page range | [chunking](../reports/chunking.md) |
| 04 | top 5, cosine floor 0.50, filters applied before ranking | the floor returns nothing for a question the corpus cannot answer, at no cost to the ones it can; the filter makes "page 9" a 1-chunk search | [retrieval](../reports/retrieval.md) |
| 05 | BM25 over the same chunks, identifiers indexed whole and in parts, weighted fusion α 0.5 | `loadBalancerIP` stops reading like `loadBalancerClass`; the identifier questions go from lost to found | [hybrid](../reports/hybrid-search.md) |
| 06 | bge-reranker-base over the first stage's top 20 | right passage retrieved 46 → 51 of 53, first 36 → 46; two seconds a question on CPU | [reranking](../reports/reranking.md) |
| 11 | passages capped at 1,200 tokens | 31 % fewer passage tokens, no right passage lost; the fifth passage was rarely the one | [latency and cost](../reports/latency-cost.md) |

The relevance floor on the reranker's score, which looked like the obvious fourth cut, lost 7 right
passages: its score ranks, it does not calibrate.

## Grounding

Claude answers from the passages with citations attached by the API (or, through `claude -p`, typed
and checked against the passage text: a quote that is not in the passage is dropped and the sentence
counts as uncited). The abstention is a fixed sentence matched verbatim, so it is a string check, not
a judgement. Two abstentions, not one: the floor before any call, the model after reading. The
adversarial set (instructions in the question, false premises, memory bait, a question in Portuguese)
holds 8 to 9 of 10. [`reports/grounding.md`](../reports/grounding.md).

## Evaluation

The benchmark is a versioned spec ([`eval/SPEC.md`](../eval/SPEC.md)): 69 questions, five label axes,
metric definitions, ten gates. Every run is saved with its lineage; comparisons read saved runs, so no
number costs a second model call. The judge (Jev, calibrated probabilities) was calibrated against 22
hand labels: it was right about grounding 91 % of the time once it was shown the passage headings,
and three scoring rules changed as a result. Three versions measured end to end:
[`reports/versions.md`](../reports/versions.md). A release is a run that cleared the gates against the
one before it; a critical failure is refused with exit 1; `pipeline rollback` is one command.

## Observability

One trace per question (every stage's time, both rankings with scores, the context with where each
passage stood before the reranker, the answer, the citations, the judge, the lineage, what fell back).
A summary over the log with flags. A canary of five known questions; its first run caught a question
that was a model judgement call, not a deployment signal. Health checks. The service exposes all of it.
[`reports/observability.md`](../reports/observability.md).

## Cost controls

Generation is the money; retrieval is the time. The token budget is the one free cut (v3). A
generation cache keyed by model, prompt version, question and the exact chunks read makes a repeat
cost nothing; a replay of the benchmark through it was 68/68 hits. The small cross-encoder is a
version away when a second a question matters more than three places of MRR. Through `claude -p` the
input counts include the CLI's context, so the per-call cost in the reports is an upper bound on the
API path. [`reports/latency-cost.md`](../reports/latency-cost.md).

## The service

`infra-docs-rag serve`: FastAPI, one process, the released pipeline loaded once. `POST /answer`
returns the cited answer, the sources with their scores, the lineage, the stage timings and a trace
id; `GET /health` is the health checks (503 on a failure); `GET /release`, `/versions`, `/traces/{id}`,
`/traces/summary`. A model call that fails is an `error` outcome with the abstention sentence and the
trace id, never a half answer or a 500. The ingestion and indexing jobs are `pipeline run`; the
versioned configs are `eval/versions.yaml`, `eval/gates.yaml`, `eval/releases.yaml`.

## What I would do next

Seed negatives into the calibration set (every label so far is grounded, so recall is unmeasured).
Put the canary on a schedule and the traces in a store with retention. Try a smaller model for the
abstention decision alone. Replace the in-memory BM25 and the exact search when the corpus is a hundred
times this size, behind the same interfaces.
