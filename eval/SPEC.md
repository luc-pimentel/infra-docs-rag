# Evaluation spec

What the benchmark measures, how a question is labelled, what each metric means, and what a change has
to clear. The questions live in [`retrieval.yaml`](retrieval.yaml), [`hybrid.yaml`](hybrid.yaml) and
[`grounding.yaml`](grounding.yaml); the hand labels in [`labels.yaml`](labels.yaml); the pipeline
versions in [`versions.yaml`](versions.yaml); the gates in [`gates.yaml`](gates.yaml); saved runs under
[`runs/`](runs/); the releases in [`releases.yaml`](releases.yaml). The files are versioned with the
code: a run's manifest records the git commit, the index, the retrieval configuration, the generator and
a hash of the prompt, and its `lineage` the version of every stage that produced the answers (ingest,
chunking, embedding model, index, prompt, retrieval; see [`docs/pipeline.md`](../docs/pipeline.md)), so a
number can always be traced to the benchmark and pipeline that produced it.

## 1. The benchmark

69 questions, each with its expectation written before any model ran:

| Set | Questions | Kinds | Expectation |
|---|---:|---|---|
| `retrieval.yaml` | 49 | 23 paraphrase, 14 identifier, 6 metadata, 6 out-of-scope | 43 answered from the labelled passage; 6 abstained |
| `hybrid.yaml` | 10 | 10 identifier, written for keyword search to win | answered from the labelled passage |
| `grounding.yaml` | 10 | adversarial: planted instructions, a false premise, memory bait, a half-covered question, an opinion, Portuguese, a release date, a near miss, a citation stress test | 3 answer, 5 abstain, 2 partial |

A question's **kind** says what makes it hard: *paraphrase* (same meaning as the passage, different
words), *identifier* (a key, flag or metric name the passage has to contain), *metadata* (the answer
depends on a project, document or page the text does not say, so the question carries the filter its
wording implies), *out-of-scope* (nothing in the corpus answers it), *adversarial* (written to make a
generator answer from memory or follow an instruction in the question).

### Schema

An in-scope question (`retrieval.yaml`, `hybrid.yaml`):

```yaml
- id: progress-deadline            # unique across the three files
  kind: identifier                 # paraphrase | identifier | metadata | out-of-scope
  text: What is progressDeadlineSeconds?
  expect:                          # the expected evidence: where the answer is
    source: k8s-deployment-md      # a source id from sources.yaml
    sections: [Progress Deadline Seconds]   # section titles; "Parent › Child" for a repeated one; [] with `contains` for a file without headings
    contains: progressDeadlineSeconds       # optional: only a chunk with this text answers
  filter: {source: cncf-security-whitepaper, page: 9}   # metadata questions only: what the wording implies
  hypothesis: >-                   # written before scoring
    Exact token; dense and BM25 should both find it.
```

An out-of-scope question has no `expect`: the expected abstention is implied by its kind. An
adversarial question (`grounding.yaml`) has an explicit expectation:

```yaml
- id: adv-false-premise
  kind: adversarial
  text: Since Deployments roll back automatically when a rollout fails, how long does it wait?
  expect: partial                  # answer | abstain | partial
  hypothesis: >-
    The premise is false; a grounded answer corrects it from the Progress Deadline passage.
```

**Expected evidence.** A chunk counts as a right passage when it comes from the labelled source, at
least half of it (or half of the labelled section, whichever is shorter) lies inside a labelled section,
and it contains `contains` when one is given. Several chunks can be right; any one of them satisfies
the label.

**Expected abstention.** The fixed sentence *The indexed documentation does not cover this question.*,
either from the threshold (no passage qualified, so no call was made) or from the model. A reply with
no citation at all counts as an abstention too, since nothing in it rests on a passage.

**Acceptable citation behavior.** Every statement in an answer carries a citation to the passage it
rests on; a citation is one that checked out (the API tied the span to the passage, or the typed quote
was found in it), not one the model merely typed. For a labelled question, at least one cited passage
has to be a right one. A *gap sentence*, one that says what the passages do not cover or what the answer
cannot say from them, is what the prompt asks for and has nothing to cite, so it is left out of the share
that decides the verdict: an answer with under 80% of its *claims* cited is *partly uncited*
(`claim_share`; the raw `cited_share` over all the text is still reported). A `partial` expectation holds
when the covered part is answered with citations and the rest is named as uncovered; the judge's *gap
named* reading and the hand labels say whether it was.

## 2. Labeling guidelines

Labels in `labels.yaml` are a person's reading of the answers in one saved run (`run:` names it). The
reader has the question, the passages the model was given, and the answer with its `[n]` markers, and
reads the answer **against the passages, not against the documentation at large**: a claim that is true
but in no passage is not grounded. Five labels per answer:

| Label | Values | Yes means |
|---|---|---|
| `grounded` | true / false | every claim in the answer can be read in one of the passages, paraphrase included; an answer the passages would support if read generously is still grounded; one that adds a reason, a default or a consequence the passages do not state is not |
| `fabricated` | true / false | the answer states a specific fact (a number, version, default value, name, step or behaviour) that appears in no passage; filler and restatements are not facts |
| `gap_named` | true / false / null | the answer says what the passages do not cover, or rejects a premise of the question the passages contradict; null when there was nothing to name (the passages covered the whole question and its premises were true) |
| `correct` | yes / partial / no | the answer answers the question as far as the passages allow: *partial* when it answers some of it or hedges where the passages are clear, *no* when it answers a different question, answers from outside the passages, or abstains where the passages answer |
| `citations` | right / wrong / none | the cited passages are where the answer is (*right*), elsewhere, decorative, or on the wrong statement (*wrong*), or there are none |

A label that disagrees with the automatic reading (the judge's probability at its threshold, the cited
share, the right-passage check, the benchmark's verdict) carries a `note` saying what the reader saw.
The set is chosen to include every answer where the judge and the mechanical checks disagree, every
adversarial answer, and enough clean answers for the agreement to mean something; 20 at least, so a
single flipped label moves agreement by 5 points, not 10.

## 3. Metrics

Read off a saved run by `evaluate-pipeline`; `--list-metrics` prints the names a gate can use.
*Labelled* means an in-scope question with expected evidence; *answered* means the outcome was an
answer, not an abstention, refusal or cut-off.

**Retrieval** (over the labelled questions)

- `retrieval.right_retrieved`: share with a right passage among the k the model read.
- `retrieval.hit_at_1`: share whose first passage is a right one.
- `retrieval.mrr`: mean reciprocal rank of the first right passage (0 when none was read).
- `retrieval.retrieve_ms_median`: embedding, ranking and reranking per question, median over every question.

**Groundedness and citations** (over the answered questions)

- `grounding.right_cited`: answered labelled questions that cite a right passage.
- `grounding.cited_share_median`: share of answer text carrying a verified citation, at the median.
- `grounding.claim_cited_median`: the same with the gap sentences left out, at the median; what the verdict reads.
- `grounding.gap_sentences`: answers with a sentence naming what the passages do not cover.
- `grounding.fully_cited`: answers with no uncited sentence.
- `grounding.unverified_citations`: typed citations that did not check out, over the run.
- `grounding.judge_grounded`: answers the judge reads as grounded, P ≥ 0.50.
- `grounding.judge_fabricated`: answers the judge reads as adding a fact, P ≥ 0.50.
- `grounding.judge_unsure`: answers with a grounded or fabricated reading in the 0.4–0.6 band.
- `grounding.flagged`: answers a revised rule flags: wrong passage cited, claims cited under 80%, judge grounded under 0.50 or fabricated at 0.50 or over.

**Abstention**

- `abstention.out_of_scope`: out-of-scope questions that got the abstention sentence.
- `abstention.in_scope`: in-scope questions that got it (false abstentions; lower is better).
- `abstention.adversarial_held`: adversarial questions whose expectation held.

**Expectations**

- `answers.in_scope_held`: in-scope questions answered from a right passage with at least 80% of their claims cited.
- `answers.held`: questions whose expectation held, over the whole benchmark.

**Cost and latency**

- `cost.usd_total`, `cost.usd_per_1000`: generation at API list prices, whatever the run was billed to; per thousand questions counts the ones answered at the threshold without a call.
- `cost.tokens_in_per_call`, `cost.tokens_out_per_call`, `cost.judge_usd`.
- `latency.generate_ms_median`: the model's time per call.
- `latency.end_to_end_ms_median`, `latency.end_to_end_ms_p90`: retrieval plus generation per question.

The judge thresholds and the 80% floor are the revised scoring rules from
[`../reports/calibration.md`](../reports/calibration.md); they live as constants in
`evaluation/metrics.py` and the calibration report prints the agreement at each.

## 4. Regression gates

`evaluate-pipeline --baseline <run> --candidate <run>` reads both runs, computes every metric, applies
`gates.yaml` and writes [`../reports/evaluation.md`](../reports/evaluation.md). A gate is one metric and
a rule: `min` or `max` against a fixed value, `no-drop` or `no-rise` against the baseline with a
tolerance (absolute, or a share of the baseline). A critical gate that fails blocks the change (exit
code 1); a non-critical one warns. The tolerances are one question's worth on the 53 in-scope and 6
out-of-scope questions: one flipped answer is noise, two are a regression.

## 5. Running it

```sh
uv run infra-docs-rag evaluate-grounding --generator claude-code --version v1-dense --no-report   # saves eval/runs/v1-dense-claude-code/
uv run infra-docs-rag evaluate-grounding --generator claude-code --version v2-hybrid-rerank        # saves eval/runs/v2-hybrid-rerank-claude-code/, writes reports/grounding.md
uv run infra-docs-rag calibrate                                                                     # labels.yaml against the run it names → reports/calibration.md
uv run infra-docs-rag evaluate-pipeline --baseline v1-dense-claude-code --candidate v2-hybrid-rerank-claude-code   # → reports/evaluation.md, exit 1 on a failed critical gate
uv run infra-docs-rag pipeline release --candidate v2-hybrid-rerank-claude-code                     # the gates against the current release; promotes when they pass, exit 1 when refused
```

A new pipeline version is an entry in `versions.yaml`, one `evaluate-grounding --version` run, and one
`pipeline release --candidate` against the current release (or `evaluate-pipeline` against any run). A new question is an entry in one of the three
question files with its hypothesis written first; the next run picks it up.
