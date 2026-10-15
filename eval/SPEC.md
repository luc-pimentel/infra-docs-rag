# Evaluation spec

What the benchmark measures, how a question is labelled, what each metric means, and what a change has
to clear. The questions live in [`retrieval.yaml`](retrieval.yaml), [`hybrid.yaml`](hybrid.yaml) and
[`grounding.yaml`](grounding.yaml); the hand labels in [`labels.yaml`](labels.yaml) and, on the rubric,
in [`calibration.yaml`](calibration.yaml); the judge rubric in [`rubric.yaml`](rubric.yaml); the pipeline
versions in [`versions.yaml`](versions.yaml); the gates in [`gates.yaml`](gates.yaml) and, per slice of
the dataset, in [`matrix.yaml`](matrix.yaml); saved runs under [`runs/`](runs/); the regression baseline
in [`baseline.yaml`](baseline.yaml); the releases in [`releases.yaml`](releases.yaml). The files are versioned with the
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
- `grounding.judge_fabricated`: answers the judge reads as adding a fact, P ≥ 0.85.
- `grounding.judge_unsure`: answers with a grounded or fabricated reading in the 0.4–0.6 band.
- `grounding.flagged`: answers a revised rule flags: wrong passage cited, claims cited under 80%, judge grounded under 0.50 or fabricated at 0.85 or over (the fabricated ceiling moved from 0.50 after the stage 15 calibration, section 8).

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

## 6. The dataset

[`dataset/records.yaml`](dataset/records.yaml) is the evaluation dataset proper: where the benchmark
files say *where* an answer is and what a run has to do, a record says *what a right answer is*, which
passages it rests on, and the failure mode the example is in the set to catch. The record schema is
[`dataset/schema.json`](dataset/schema.json) (generated from the model in `evaluation/dataset.py`); the
report is [`../reports/dataset.md`](../reports/dataset.md).

```yaml
- id: mh-undo-after-history-zero
  query: I set revisionHistoryLimit to 0 on my Deployment. Can I still run kubectl rollout undo?
  category: multi-hop            # fact-lookup | multi-hop | ambiguous | no-answer | stale | adversarial
  expect: answer                 # answer | abstain | partial | clarify
  difficulty: hard               # easy | medium | hard
  failure_mode: missing-hop      # the failure this example is here to catch; `none` for a clean one
  split: test                    # dev | test
  origin: hand-written           # benchmark | traces | hand-written | generated
  source_doc_ids: [k8s-deployment-md]
  contexts: [k8s-deployment-md:54258, k8s-deployment-md:24763]   # chunk ids under the file's `chunking`
  reference_answer: >-           # from the contexts alone
    No. ...
  tags: [kubernetes, deployment, rollback]
  benchmark_id: ...              # the benchmark question it was taken from (benchmark records only)
  note: >-                       # why it is in the set
    ...
```

**Where records come from.** Most are benchmark questions, chosen because a saved run did something
wrong with them: the saved runs are the record of real failures, and `dataset report` prints every
run's verdict next to each record so the designed failure mode can be checked against what was seen.
`traces` records are questions people asked the running service (`data/traces/`) that the benchmark
did not have. `hand-written` records cover the failure modes the runs cannot show yet: a deprecated
field presented as current (*stale*), a question with several readings (*ambiguous*), an answer that
needs two passages (*multi-hop*). `generated` records come from [`dataset/generated.yaml`](dataset/generated.yaml),
a testset a model wrote from sampled chunks, filtered by [`dataset/audit.yaml`](dataset/audit.yaml):
each generated example is kept, edited or discarded with a reason, and only the first two enter.

**Expectations.** `answer` means an answer with every claim cited from the contexts; `abstain` the
fixed sentence from section 1; `partial` an answer to the part the passages cover with the rest named
as uncovered (the false-premise and mixed-scope questions, and a stale method asked for by name);
`clarify` a reply that names the readings the passages support and asks which, or answers each
briefly, for an ambiguous question. The reference answer for an abstention is the sentence itself; for
everything else it is what a right answer says, in one to four sentences, from the contexts alone.

**Failure modes.** `none`, `false-abstention`, `retrieval-miss`, `wrong-passage`, `uncited-claims`,
`unverified-citation`, `answered-anyway`, `judge-misread` are the ones a saved run can show, by its
verdict, the right-passage flags, the typed citations that did not verify and the judge's reading at
the calibrated thresholds; `fabrication`, `stale-answer`, `wrong-reading` and `missing-hop` are
designed into the set and not measured yet. The report lists which.

**Splits.** `dev` holds the records whose answers were read when the scoring rules were revised (the
hand labels and the calibration report) plus a share of the new ones; `test` is held out. A rule tuned
on dev is checked on test.

**Checks.** `dataset check` validates the schema, that every context exists in the named chunking and
its source in `sources.yaml`, that a benchmark record's contexts are among the passages the benchmark
labels as right for its question (so a relabelled benchmark surfaces here), that an adversarial
record's expectation matches the benchmark's, that every category is present and the curated set
holds 30 to 50 records, and that the audit decides every generated example. It exits 1 on a problem.

## 7. Metrics by failure mode

[`matrix.yaml`](matrix.yaml) maps each failure mode of the dataset to the metric that measures it, the
direction that is better and the threshold it is held to on the records designed for that mode;
[`../reports/metrics.md`](../reports/metrics.md) is the baseline score report over a dataset run. A
dataset run (`dataset run`) answers every record through a pipeline version and saves the result under
`runs/` in the same format as a benchmark run, with two differences: a right passage is one the record's
`contexts` name, and each answer carries the dataset reader's four probabilities beside the judge's
three.

**The reader.** The same probability judge as section 3, asked four questions the benchmark could not
ask because it had no reference answer: *relevant* (the answer addresses the question asked),
*reference_agrees* (it reaches the reference answer's conclusion and keeps its main points),
*deprecation_named* (when a passage marks something the answer recommends as deprecated, removed or
replaced, the answer says so; true when there is nothing to flag), *readings_named* (when the question
has several readings, the answer names them, asks which, or says which it took; true when it has one).
The state is the passages under their headings, the question, the reference answer and the marked
answer. A probability at or over 0.50 reads as yes. Section 8 calibrates the reader's two main
questions against the rubric labels; the gates on its readings stay loose until that agreement is good.

**Metrics.** Every stage 08 metric, computed on a slice as on a run (the `grounding.*` names also
appear as `faithfulness.*`), plus:

- `retrieval.context_recall`: share of the record's named passages among the ones the model read,
  averaged over the records that name any; `retrieval.context_precision`: share of the passages read
  that the record names.
- `relevance.judge_relevant`, `relevance.reference_agrees`: the reader's two main questions, over the
  answered records it read.
- `abstention.expected`: records expecting an abstention that got one; `abstention.false`: records
  expecting an answer, whole or partial, that got the abstention sentence.
- `faithfulness.judge_flagged_held`: answers the checks accept (right passage cited, claims cited) that
  the judge reads as ungrounded or fabricated: the judge-misread signal.
- `stale.deprecation_named`, `ambiguity.readings_named`: the reader's two mode-specific questions on
  the stale and ambiguous slices.
- `hops.covered`: multi-hop answers that cite at least two of the record's named passages.

**Slices.** `all`, then `<field>=<value>` for category, failure_mode, difficulty, split and origin.
The matrix reads each failure mode's metric on `failure_mode=<mode>` unless the entry names another
slice; a slice of five records moves by 20% when one flips, so the slices locate and the whole set
decides.

**Rules.** A matrix entry is a target (`min` or `max` against a value) the report checks; `blocker:
true` marks the misses the pipeline must not ship with (a no-answer record answered, a fabricated
fact). The `gates` are the pass/fail rules a change has to clear, on one slice each, with the rules of
section 4: `min` / `max` against a value, `no-drop` / `no-rise` against a baseline dataset run with a
tolerance. `dataset metrics --run <run> [--baseline <run>]` writes the report and exits 1 when a
critical gate fails; without a baseline the delta gates are skipped, not failed. That exit code is the
PR check: `.github/workflows/checks.yml` runs it on the committed baseline run beside the tests.

**Diagnosis.** For every record whose expectation did not hold, the report says where the failure
sits: *retrieval* when none of the named passages reached the model, *generation* when one did and the
answer still failed (an abstention with the passage in hand, the wrong passage cited, claims left
uncited), *both* when an abstention was expected and a near miss cleared the threshold and was
answered, and *reading* for a record that held by the checks and that the judge or the reader
disputes.

## 8. The judge rubric and its calibration

[`rubric.yaml`](rubric.yaml) is the judge specification: three dimensions, each scored 0, 1 or 2
against one thing, so that a score on one says nothing about the others, and null where the dimension
does not apply.

| Dimension | Reads the answer against | 2 | 1 | 0 | null when |
|---|---|---|---|---|---|
| `groundedness` | the passages only | every claim in a passage (gap sentences aside) | a claim beyond the passages, no specific fact invented | a specific fact no passage gives, or one a passage contradicts | the answer is the abstention sentence |
| `completeness` | the reference answer | every main point of the reference | a main point missing, hedged, or one reading of several | the reference's conclusion missed, contradicted, or an abstention where it answers | the reference is the abstention sentence |
| `abstention` | what the passages cover | the right call, marked (the gap named, the premise corrected) | the right call, unmarked | the wrong call: answered what is not covered, abstained with the answer in hand, a false premise built on | never |

Each score has an example from a labelled answer. `operating` holds the threshold a score passes at
per dimension; the calibration sets it and a person updates it when the report says so. The rubric
does not score currency (a deprecated field presented as current) or the order of an answer; the
reader's deprecation question (section 7) is the check on those.

**The grader.** `judge grade` renders the rubric into a system prompt (the definitions as written, not
the examples), sends the passages under their headings, the question, the reference answer and the
marked answer through `claude -p` with a JSON schema, and saves one score and one-sentence reason per
dimension to `runs/<run>/grades.json`, with the model, effort, prompt hash and cost. It grades only
the answers in `calibration.yaml`; answers already graded are kept unless `--regrade`.

**The calibration set.** [`calibration.yaml`](calibration.yaml) is a person's scores on the rubric
for 25 to 40 answers of one saved run, every one read twice by independent readers (`a`, `b`) and
resolved where they differ (`resolution` is required there). Most are answers as the run wrote them,
chosen to cover every verdict, every expectation and the answers the probability judge or the reader
read lowest; some are *perturbed*: a real answer with one failure written in (a fact the passage
contradicts, an abstention replaced by an answer from memory, a premise accepted), kept with its base
case's passages, question and reference, so that each score on each dimension has examples and the
grader is measured on failures it has to catch. A perturbed label carries `base`, `perturbation` and
`answer`; `judge grade` asks jev the stage 08 and stage 14 questions about it too, since the run never
saw it.

**The report.** `judge calibrate` compares, per dimension, the grader's score to the person's: exact
agreement, a quadratic weighted κ over the three scores, the confusion matrix, and the pass/fail
agreement at each threshold; the overall verdict (every applicable dimension passes); the stage 08
judge (grounded, fabricated) and the stage 14 reader (relevant, agrees with the reference) swept
against the same labels; the two readers against each other; and every answer where the grader and
the person differ, with both reasons. It writes [`../reports/judge.md`](../reports/judge.md) and exits
1 when a labelled answer has no grade; that is the PR check, with no model call.

## 9. Evals as regression tests

A change to the pipeline (a retriever setting, a reranker, a prompt, a model) is measured the same way
every time: the records of section 6 through the changed pipeline (`dataset run --version <name>`),
then the saved run against the baseline, slice by slice and record by record (`dataset compare`). The
report is [`../reports/regression.md`](../reports/regression.md).

**The baseline.** [`baseline.yaml`](baseline.yaml) pins the reference dataset run and the version of
everything that produced it: the dataset (`<records>+<hash>` over every field of every record, which a
run saved by `dataset run` carries on its manifest as `dataset`), the pipeline version, the retrieval
configuration's version, the prompt's, the index's, the embedding model's, the chunking's, the generator
and how it was called. `dataset baseline set --run <run>` writes it; `dataset baseline check` reads the
run and the records back and exits 1 when anything no longer matches: the pin is stale when the records
were edited after the baseline answered them (the baseline has to be run again under the same version)
or when the run on disk is not the one the pin describes. `dataset compare` refuses a stale pin, so a
comparison is always against the reference the pin names; `--baseline <run>` compares against any run
instead.

**The comparison.** Every metric of section 7 on every slice, before and after; the slice gates of
[`matrix.yaml`](matrix.yaml) on the candidate against the baseline (the `no-drop` and `no-rise` rules
now have a baseline to read); every record whose outcome, verdict or right-passage checks changed, as
a win (holds now, did not before) or a loss (held, no longer does), with the answers behind each loss
and the passages both runs read, and whether those passages were the same (then the flip is the
generator's variance, not the change under test: a dataset run does not read the generation cache);
and the matrix before and after. The two runs' manifests are compared
field by field, so the report says what differs (retrieval alone, or the prompt, the index, the dataset
too) and a change is attributed to the stage that changed.

**The decision.** *Block* when a critical gate fails, on any slice, whatever the whole set did: the
per-slice `no-drop` gates (adversarial, no-answer, the held-out split, the hard records, fact lookup)
are what make a change that lifts the aggregate and drops a critical slice block anyway, and the report
names the gates the aggregate would have hidden. *Hold* when every critical gate passes and the change
still lost more records than it won, or the whole set holds fewer within the tolerance: the tolerances
are one record's worth, so this is the case where the gates cannot tell noise from a regression and a
person reads the losses. *Ship* otherwise, with the non-critical gates that failed as warnings. The
tolerances are set by slice size, in [`matrix.yaml`](matrix.yaml): two records on a slice of 34 or more
block, a slice of five or six warns. `dataset compare` exits 1 on *block*; `--no-block` writes the
report and exits 0, for regenerating the committed record of an experiment that was not shipped. The
PR check runs `dataset baseline check` and regenerates the committed comparison, and fails when the
report drifts from what the saved runs say.

```sh
uv run infra-docs-rag dataset run --generator claude-code --version v4-rerank-50   # the candidate: every record through the changed pipeline
uv run infra-docs-rag dataset compare --candidate dataset-v4-rerank-50-claude-code   # → reports/regression.md, exit 1 on block
uv run infra-docs-rag dataset baseline check                                          # the pin against its run and the records
uv run infra-docs-rag dataset baseline set --run <run> --note '…'                     # a shipped change becomes the next baseline
```
