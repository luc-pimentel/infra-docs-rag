# Demo script

Fifteen minutes, one terminal, no slides. Every command below is real and its output is what the
reports show; the model calls go through `claude -p` on a Claude Code subscription (or the API with
`ANTHROPIC_API_KEY`). Times are on a laptop CPU. The design doc is [`design.md`](design.md); the
numbers behind every claim are in [`reports/`](../reports/).

## 0. Before: what the system is (1 min)

> Documentation for Kubernetes, Prometheus and Argo CD: 17 pages and two PDFs, 497 chunks. Ask a
> question, get an answer that cites the document, section and page it came from, or the sentence
> "The indexed documentation does not cover this question." Every answer leaves a trace. Nothing
> reaches the service without a benchmark run that cleared the regression gates.

```sh
uv run infra-docs-rag pipeline status        # the four stages, up to date; the index versions; the releases
```

## 1. A question, and what it leaves behind (3 min)

```sh
uv run infra-docs-rag answer "How do I undo a bad release?" --generator claude-code
```

Point at: the `[1]` markers after each sentence, the sources under the answer with their URLs, the
`100% of the text cited` line, the trace id.

```sh
uv run infra-docs-rag traces --last 1
```

Point at: the first-stage list (the fused cosine and BM25 ranking, 20 candidates), the context after
the cross-encoder with `was n.` for what it moved, the stage times (rerank 1.5 s, generate 4 s, the
rest under 10 ms), the lineage line (index, prompt, retrieval, embedding versions).

```sh
uv run infra-docs-rag answer "How do I configure Istio sidecar injection?" --generator claude-code
uv run infra-docs-rag answer "What is a good recipe for sourdough bread?" --generator claude-code
```

Point at: the first abstains in the model's words after reading the closest passages (Istio appears
once in the corpus, as a repository name); the second never reaches the model, the cosine floor stops
it at 0.46 against 0.50. Two kinds of "no", both traced.

## 2. The service (2 min)

```sh
uv run infra-docs-rag serve &
curl -s localhost:8000/health | jq .status
curl -s localhost:8000/release | jq .serving
curl -s localhost:8000/answer -H 'content-type: application/json' \
  -d '{"question": "What does keep_firing_for do?"}' | jq '{outcome, answer, citations: [.citations[].citation], total_ms, cached}'
```

Point at: the same answer the CLI gives, the lineage in `/release`, `cached: true` on a repeat.

## 3. How it got here: the benchmark and the gates (4 min)

```sh
uv run infra-docs-rag evaluate-pipeline --list-runs
uv run infra-docs-rag evaluate-pipeline --baseline v1-dense-claude-code --candidate v2-hybrid-rerank-claude-code
```

Point at: 69 questions (49 retrieval, 10 identifier, 10 adversarial), five label axes, ten gates.
v1 (dense only) to v2 (hybrid plus cross-encoder): right passage retrieved 46 → 51 of 53, false
abstentions 5 → 3, cost flat, +2 s a question. Open `reports/evaluation.md` at the "questions that
changed" section: 12 improved, 3 regressed, each with the answer before and after.

```sh
uv run infra-docs-rag pipeline release --candidate v1-dense-claude-code --note "on purpose"
```

Point at: four critical gates fail, exit 1, the release record untouched. Then
`reports/versions.md` for the three versions side by side.

## 4. When it breaks (3 min)

```sh
HTTPS_PROXY=http://127.0.0.1:9 uv run infra-docs-rag pipeline run --refresh   # every download refused
uv run infra-docs-rag pipeline status                                           # fetch failed, the rest stale
uv run infra-docs-rag pipeline run                                              # the retry: fetch ok, the rest skipped
```

Then the postmortem ([`postmortem.md`](postmortem.md)): the judge read passages without their headings
and called two correct answers fabricated; the calibration against hand labels found it; one function
now builds the passage for both the generator and the judge.

## 5. What it costs, and the one cut that was free (2 min)

Open `reports/latency-cost.md` section 2. Point at: the cross-encoder is all of retrieval's time and
none of the money; the relevance floor that looked free lost 7 right passages; the token budget keeps
every right passage at 31 % fewer passage tokens and is v3; the small cross-encoder is the latency
option and the two do not combine.

## Questions that come up

- **Why not a vector database?** 497 chunks; exact search is 0.2 ms. The index is files with a
  manifest, content-addressed; the store interface is where a database would go.
- **Why a cross-encoder on CPU?** It is what moved right-passage-retrieved from 46 to 51. Two seconds
  a question is inside the 15 s gate; `minilm-l6` is a version away at 0.3 s.
- **Why is the judge not the verdict?** It was wrong on two of 54 in a way that looked right. It is one
  gate among ten and it is calibrated against hand labels.
- **What would you do with a week?** Seed negatives into the calibration set; move the traces to a
  store with retention; put the canary on a schedule; try a smaller generator for the abstention
  decision alone.
