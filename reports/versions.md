# Versions: the pipeline measured end to end, release by release

3 versions of the pipeline over the same 69 questions, each a saved run under
`eval/runs/` with the lineage of what answered. The columns are in release order. Every number comes
from the run's cases; the definitions are in [`eval/SPEC.md`](../eval/SPEC.md). Through `claude -p`, the
input token counts include the CLI's own context, so the cost column overstates the API path by the same
factor for every version; the differences between versions are the passages.

## What each version is

- **v1-dense** (`v1-dense-claude-code`, 2026-10-12, commit 1b66515): dense retrieval alone, as after stage 04; cosine top 5 over the 0.50 floor, no reranker. Index `3971d9f1e10e`, prompt `cite-v1+e732a9f0cc40`, retrieval `v1-dense+97e3615c26c9`.
- **v2-hybrid-rerank** (`v2-hybrid-rerank-claude-code`, 2026-10-12, commit b228565): hybrid weighted fusion (α 0.5) reranked by bge-base over 20 candidates, the stage 06 default. Index `3971d9f1e10e`, prompt `cite-v1+e732a9f0cc40`, retrieval `v2-hybrid-rerank+cc59ba7751e4`.
- **v3-budget** (`v3-budget-claude-code`, 2026-10-12, commit 220b563): v2 with the passages capped at 1,200 tokens in rank order; the stage 11 cut that costs no right passage. Index `3971d9f1e10e`, prompt `cite-v1+e732a9f0cc40`, retrieval `v3-budget+a9ed5d977ad2`.

## Side by side

| | **v1-dense**<br>`v1-dense-claude-code` | **v2-hybrid-rerank**<br>`v2-hybrid-rerank-claude-code` | **v3-budget**<br>`v3-budget-claude-code` |
|---|---|---|---|
| right passage retrieved | 46/53 (87%) | 51/53 (96%) | 51/53 (96%) |
| right passage first | 36/53 (68%) | 46/53 (87%) | 46/53 (87%) |
| MRR | 0.75 | 0.91 | 0.91 |
| retrieval, median | 31 ms | 1,555 ms | 1,570 ms |
| right passage cited | 46/48 (96%) | 50/50 (100%) | 51/51 (100%) |
| claims cited, median share | 100% | 100% | 100% |
| answers with no uncited sentence | 29/54 (54%) | 36/56 (64%) | 38/56 (68%) |
| typed citations that did not check out | 7 | 10 | 2 |
| judge: grounded | 52/54 (96%) | 54/56 (96%) | 56/56 (100%) |
| judge: fabricated | 3/54 (6%) | 1/56 (2%) | 0/56 (0%) |
| flagged for a person | 12/54 (22%) | 4/56 (7%) | 1/56 (2%) |
| out-of-scope abstained | 6/6 (100%) | 6/6 (100%) | 6/6 (100%) |
| false abstentions | 5/53 (9%) | 3/53 (6%) | 2/53 (4%) |
| adversarial held | 8/10 (80%) | 9/10 (90%) | 8/10 (80%) |
| in-scope answered right | 40/53 (75%) | 48/53 (91%) | 50/53 (94%) |
| expectations held | 54/69 (78%) | 63/69 (91%) | 64/69 (93%) |
| generation, median | 4,121 ms | 4,603 ms | 4,348 ms |
| end to end, median | 4,134 ms | 6,181 ms | 5,714 ms |
| end to end, p90 | 6,661 ms | 7,981 ms | 8,249 ms |
| input tokens per call | 55,920 | 56,350 | 54,862 |
| output tokens per call | 415 | 421 | 417 |
| cost per 1,000 questions | $42.90 | $41.66 | $39.79 |

## The gates between each version and the next

### v1-dense → v2-hybrid-rerank: **passes**

| Gate | Rule | Before → after | Result |
|---|---|---|---|
| `abstention.out_of_scope` | must not drop | 6/6 (100%) → 6/6 (100%), +0% | pass |
| `abstention.in_scope` | must not rise by more than 0.02 | 5/53 (9%) → 3/53 (6%), -4% | pass |
| `grounding.right_cited` | must not drop by more than 0.02 | 46/48 (96%) → 50/50 (100%), +4% | pass |
| `grounding.judge_fabricated` | must not rise by more than 0.02 | 3/54 (6%) → 1/56 (2%), -4% | pass |
| `grounding.claim_cited_median` | ≥ 0.9 | 100% against ≥ 0.9 | pass |
| `retrieval.right_retrieved` | must not drop by more than 0.02 | 46/53 (87%) → 51/53 (96%), +9% | pass |
| `retrieval.mrr` | must not drop by more than 0.05 | 0.75 → 0.91, +0.15 | pass |
| `abstention.adversarial_held` | ≥ 0.8 | 9/10 (90%) against ≥ 0.8 | pass |
| `cost.usd_per_1000` | must not rise by more than 25% of the baseline | $42.90 → $41.66, -$1.24 | pass |
| `latency.end_to_end_ms_median` | ≤ 15000 | 6,181 ms against ≤ 15000 | pass |

15 question(s) changed: 12 improved, 3 regressed (`lock-config`: outcome answered → abstained; verdict held → abstained; right retrieved yes → no; right cited yes → –, `configmap-live-update`: verdict held → partly uncited, `argocd-cm-url`: outcome answered → abstained; verdict held → abstained; right cited yes → –).

### v2-hybrid-rerank → v3-budget: **passes**

| Gate | Rule | Before → after | Result |
|---|---|---|---|
| `abstention.out_of_scope` | must not drop | 6/6 (100%) → 6/6 (100%), +0% | pass |
| `abstention.in_scope` | must not rise by more than 0.02 | 3/53 (6%) → 2/53 (4%), -2% | pass |
| `grounding.right_cited` | must not drop by more than 0.02 | 50/50 (100%) → 51/51 (100%), +0% | pass |
| `grounding.judge_fabricated` | must not rise by more than 0.02 | 1/56 (2%) → 0/56 (0%), -2% | pass |
| `grounding.claim_cited_median` | ≥ 0.9 | 100% against ≥ 0.9 | pass |
| `retrieval.right_retrieved` | must not drop by more than 0.02 | 51/53 (96%) → 51/53 (96%), +0% | pass |
| `retrieval.mrr` | must not drop by more than 0.05 | 0.91 → 0.91, +0.00 | pass |
| `abstention.adversarial_held` | ≥ 0.8 | 8/10 (80%) against ≥ 0.8 | pass |
| `cost.usd_per_1000` | must not rise by more than 25% of the baseline | $41.66 → $39.79, -$1.87 | pass |
| `latency.end_to_end_ms_median` | ≤ 15000 | 5,714 ms against ≤ 15000 | pass |

5 question(s) changed: 3 improved, 2 regressed (`whitepaper-page`: verdict held → partly uncited, `adv-portuguese`: outcome answered → abstained; verdict held → abstained).

## Rollback decision criteria

A release is the current one until one of these holds, in which case `pipeline rollback` makes the
previous release current again and points the index at its version (`docs/pipeline.md`):

1. **A critical gate fails on a re-run.** The gates in [`eval/gates.yaml`](../eval/gates.yaml) are the
   contract: out-of-scope abstention may not drop; false abstentions, judge-fabricated answers and the
   share of answers citing a right passage may not move by more than one question's worth (2 %); the
   median claim share stays over 90 %; at least 8 of 10 adversarial expectations hold; a right passage
   is among the ones read for no fewer labelled questions than before. A release never clears them on
   the way in and fails them later unless the corpus, the model or the prompt changed underneath it,
   and the lineage says which.
2. **The canary fails.** Five questions with a known outcome and source (`eval/canary.yaml`) against the
   released pipeline; one failure that is not the model's judgement call (see
   [`reports/observability.md`](observability.md)) is a rollback, then a look.
3. **The trace log flags a rate.** No-passage share over 30 %, error rate over 5 %, p95 over 15 s, any
   degraded answer, median claim share under 80 %, or traces made on an index that is not the
   released one (`traces --summary`). A flag is a prompt to look; two flags on the same day are a
   rollback.
4. **The release's index is not on disk.** `pipeline health` warns when the release was measured on
   an index version other than the current one. The service keeps answering; the mismatch is
   resolved by re-releasing under the new index or pointing back, never by leaving it.

What is not a rollback: a warning gate (MRR, cost, median latency inside their tolerances), one
flipped answer on a set of ten, or a judge reading in the unsure band. Those are the next
calibration's questions.

Current release: **r4**, v3-budget, measured by `v3-budget-claude-code` (passes). The record is [`eval/releases.yaml`](../eval/releases.yaml).
