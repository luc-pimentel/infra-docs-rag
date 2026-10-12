# Latency and cost: where the time and the money go, and what each cut buys

The workload is the benchmark: 69 questions (53 labelled, 6 out of scope, 10 adversarial), one pass each. Retrieval is measured here, every stage timed, on Apple M4, CPU only (4 threads), torch 2.14.0, with no model call: the retrieval quality comes from the labels, the
abstentions from the floor. The generation and judge side comes from the saved runs under
`eval/runs/` (the same questions through `claude -p`), and a configuration that has no saved run
gets its generation cost estimated from the passages it would send: the passages and the question
as fresh input tokens at API list price, the frozen system prompt as a cache read, and the released
run's median output length (426 tokens). Passage tokens are counted by the embedding
model's tokenizer, which runs a little under Claude's; the estimate is a floor, not a quote.

## 1. The request path, stage by stage

The released pipeline (`top 5 · hybrid (weighted α=0.5 · depth 50) · rerank bge-base over 20 · min 0.50`) over the 69 questions. Retrieval p50 and p95 are
measured here; generation and judge are the released run's medians.

| Stage | p50 | p95 | Tokens | Cost per question | Where it runs |
|---|---|---|---|---|---|
| embed | 7 ms | 10 ms | – | $0 (local) | the query through bge-small, CPU |
| dense | 0 ms | 0 ms | – | $0 (local) | a dot product over 497 unit vectors |
| lexical | 0 ms | 0 ms | – | $0 (local) | BM25 over the same chunks, in memory |
| fuse | 0 ms | 0 ms | – | $0 (local) | weighted fusion of the two rankings |
| rerank | 1,559 ms | 1,764 ms | – | $0 (local) | bge-reranker-base over 20 pairs, CPU |
| pack | 1 ms | 1 ms | – | $0 (local) | the floor, the budget, the headings |
| **retrieval** | **1,567 ms** | **1,773 ms** | 1,410 passage tokens | $0 | the six stages above |
| generate | 4,603 ms | 6,883 ms | 56,350 in / 421 out per call | $0.0417 | claude-opus-5-5 through `claude -p`, 68 calls |
| judge | 374 ms | 464 ms | – | $0.0001 | jev, 56 readings, not on the serving path |
| **end to end** | **6,181 ms** | p90 7,981 ms | – | $41.66 per 1,000 | retrieval + generation |

Two numbers decide the shape of everything below. The cross-encoder is the whole of retrieval's
time (1,559 ms of 1,567 ms at the median; the first stage is under a millisecond), and
generation is the whole of the money. The input token counts through `claude -p` include the
CLI's own context, most of it read from cache, so the per-call cost above overstates the API path;
the estimates in the next sections price only the passages, which is the part a configuration changes.

## 2. Each cut, before and after

Every optimization applied to the released pipeline alone. **Right retrieved** is the labelled
questions with a right passage among the ones the model would read (of 53);
**false abstentions** are in-scope questions the floor left with no passage; **est. cost** is the
generation estimate per 1,000 questions described above.

| Configuration | What changes | Retrieval p50 | Rerank p50 | Passages (p50) | Passage tokens (p50) | Right retrieved | Hit@1 | MRR | False abst. | Est. cost / 1,000 |
|---|---|---|---|---|---|---|---|---|---|---|
| **v2 (released)** | hybrid, bge-base over 20 candidates, top 5, cosine floor 0.50 | 1,567 ms | 1,559 ms | 5 | 1,410 | 51/53 | 46 | 0.91 | 0 | $14.02 |
| rerank 10 | the cross-encoder reads 10 candidates instead of 20 | 909 ms | 902 ms | 5 | 1,388 | 50/53 (-1) | 46 | 0.90 | 0 | $13.90 |
| rerank 5 | the cross-encoder reads only the first stage's top 5: a reorder, not a rescue | 464 ms | 457 ms | 5 | 1,397 | 49/53 (-2) | 46 | 0.89 | 0 | $13.77 |
| minilm-l6 over 20 | the small cross-encoder, a twelfth of the parameters | 341 ms | 334 ms | 5 | 1,428 | 51/53 | 44 | 0.88 | 0 | $14.11 |
| top 3 | the model reads 3 passages instead of 5 | 1,567 ms | 1,560 ms | 3 | 858 | 50/53 (-1) | 46 | 0.90 | 0 | $11.89 |
| relevance floor 0.10 | passages the reranker scores under 0.10 are dropped: adaptive top-k | 1,611 ms | 1,603 ms | 3 | 814 | 44/53 (-7) | 42 | 0.81 | 7 | $10.08 |
| v3: budget 1,200 tokens | the passages may add up to 1,200 tokens, in rank order; the first always goes in | 1,661 ms | 1,653 ms | 4 | 969 | 51/53 | 46 | 0.91 | 0 | $12.31 |
| no rerank | the hybrid first stage alone, the stage 05 pipeline | 6 ms | – | 5 | 1,397 | 49/53 (-2) | 43 | 0.85 | 0 | $13.77 |
| first candidate: rerank 10 + floor 0.10 + top 4 | designed before measuring: the three cuts that looked free, together | 985 ms | 978 ms | 3 | 793 | 43/53 (-8) | 42 | 0.80 | 7 | $9.43 |
| minilm-l6 + top 3 | the small cross-encoder and 3 passages | 339 ms | 332 ms | 3 | 878 | 48/53 (-3) | 44 | 0.86 | 0 | $11.94 |
| minilm-l6 + top 4 | the small cross-encoder and 4 passages | 341 ms | 334 ms | 4 | 1,192 | 50/53 (-1) | 44 | 0.87 | 0 | $12.96 |
| rerank 10 + top 4 | 10 candidates and 4 passages | 972 ms | 964 ms | 4 | 1,160 | 49/53 (-2) | 46 | 0.89 | 0 | $12.92 |
| minilm-l6 over 10 + top 4 | the small cross-encoder over 10, 4 passages | 231 ms | 223 ms | 4 | 1,186 | 49/53 (-2) | 44 | 0.87 | 0 | $12.97 |
| minilm-l6 + budget 1,200 | the small cross-encoder over 20 and the budget together | 337 ms | 330 ms | 3 | 928 | 49/53 (-2) | 44 | 0.87 | 0 | $12.18 |

What the table says, cut by cut:

- **rerank 10:** retrieval -659 ms at the median, estimated generation -0.11 $/1,000, loses 1 right passage, MRR -0.01, false abstentions +0.
- **rerank 5:** retrieval -1,104 ms at the median, estimated generation -0.25 $/1,000, loses 2 right passages, MRR -0.02, false abstentions +0.
- **minilm-l6 over 20:** retrieval -1,227 ms at the median, estimated generation +0.09 $/1,000, costs no right passage, MRR -0.03, false abstentions +0.
- **top 3:** retrieval -0 ms at the median, estimated generation -2.12 $/1,000, loses 1 right passage, MRR -0.00, false abstentions +0.
- **relevance floor 0.10:** retrieval +44 ms at the median, estimated generation -3.94 $/1,000, loses 7 right passages, MRR -0.09, false abstentions +7.
- **v3: budget 1,200 tokens:** retrieval +93 ms at the median, estimated generation -1.71 $/1,000, costs no right passage, MRR +0.00, false abstentions +0.
- **no rerank:** retrieval -1,561 ms at the median, estimated generation -0.25 $/1,000, loses 2 right passages, MRR -0.06, false abstentions +0.
- **first candidate: rerank 10 + floor 0.10 + top 4:** retrieval -582 ms at the median, estimated generation -4.59 $/1,000, loses 8 right passages, MRR -0.10, false abstentions +7.
- **minilm-l6 + top 3:** retrieval -1,229 ms at the median, estimated generation -2.08 $/1,000, loses 3 right passages, MRR -0.04, false abstentions +0.
- **minilm-l6 + top 4:** retrieval -1,227 ms at the median, estimated generation -1.06 $/1,000, loses 1 right passage, MRR -0.03, false abstentions +0.
- **rerank 10 + top 4:** retrieval -595 ms at the median, estimated generation -1.10 $/1,000, loses 2 right passages, MRR -0.01, false abstentions +0.
- **minilm-l6 over 10 + top 4:** retrieval -1,337 ms at the median, estimated generation -1.05 $/1,000, loses 2 right passages, MRR -0.04, false abstentions +0.
- **minilm-l6 + budget 1,200:** retrieval -1,231 ms at the median, estimated generation -1.83 $/1,000, loses 2 right passages, MRR -0.04, false abstentions +0.

### Indexing: the batch size

Embedding the corpus is paid once per index version, not per question, so it is the cheap side of
the ledger; the batch size is the one knob and this is what it does on this machine:

| Batch size | Chunks | Seconds | Chunks per second |
|---|---|---|---|
| 8 | 497 | 12.3 | 41 |
| 16 | 497 | 14.1 | 35 |
| 64 | 497 | 16.9 | 29 |

### The generation cache

A cache in front of the generator keyed by the model, the prompt version, the question and the exact
chunks it would read (`generate/cache.py`). The same question against the same passages is the same
answer; a different index version, a changed prompt or a reordered context is a different key, so
the cache can never serve an answer the released pipeline would not give. On a replay of the
benchmark it is 100 % hits, zero tokens, and the generate stage falls to the lookup's time; its
value in production is the share of questions that repeat, which the trace log measures. The
replay's numbers are in section 3 where a cached run was saved.

## 3. The matrix

Every configuration with a saved run, measured end to end through the model, plus the profiles
measured here without one. A dash is a column that needs a model call the configuration has not had.

| Configuration | End to end p50 | Retrieval p50 | Cost / 1,000 | Right retrieved | MRR | Right cited | Judge fabricated | Claim share (median) | False abst. | OOS abst. | Adversarial held |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **v1-dense** (`v1-dense-claude-code`) | 4,134 ms | 31 ms | $42.90 | 46/53 (87%) | 0.75 | 46/48 (96%) | 3/54 (6%) | 100% | 5/53 (9%) | 6/6 (100%) | 8/10 (80%) |
| **v2-hybrid-rerank** (`v2-hybrid-rerank-claude-code`) | 6,181 ms | 1,555 ms | $41.66 | 51/53 (96%) | 0.91 | 50/50 (100%) | 1/56 (2%) | 100% | 3/53 (6%) | 6/6 (100%) | 9/10 (90%) |
| **v3-budget** (`v3-budget-claude-code`) | 5,714 ms | 1,570 ms | $39.79 | 51/53 (96%) | 0.91 | 51/51 (100%) | 0/56 (0%) | 100% | 2/53 (4%) | 6/6 (100%) | 8/10 (80%) |
| **v3-budget** (`v3-budget-claude-code-replay`) replayed through the cache | 1,578 ms | 1,578 ms | $0.00 | 51/53 (96%) | 0.91 | 51/51 (100%) | – | 100% | 2/53 (4%) | 6/6 (100%) | 8/10 (80%) |
| v2 (released) (profile) | – | 1,567 ms | est. $14.02 | 51/53 | 0.91 | – | – | – | 0 | 1/6 by the floor | – |
| rerank 10 (profile) | – | 909 ms | est. $13.90 | 50/53 | 0.90 | – | – | – | 0 | 1/6 by the floor | – |
| rerank 5 (profile) | – | 464 ms | est. $13.77 | 49/53 | 0.89 | – | – | – | 0 | 1/6 by the floor | – |
| minilm-l6 over 20 (profile) | – | 341 ms | est. $14.11 | 51/53 | 0.88 | – | – | – | 0 | 1/6 by the floor | – |
| top 3 (profile) | – | 1,567 ms | est. $11.89 | 50/53 | 0.90 | – | – | – | 0 | 1/6 by the floor | – |
| relevance floor 0.10 (profile) | – | 1,611 ms | est. $10.08 | 44/53 | 0.81 | – | – | – | 7 | 4/6 by the floor | – |
| v3: budget 1,200 tokens (profile) | – | 1,661 ms | est. $12.31 | 51/53 | 0.91 | – | – | – | 0 | 1/6 by the floor | – |
| no rerank (profile) | – | 6 ms | est. $13.77 | 49/53 | 0.85 | – | – | – | 0 | 1/6 by the floor | – |
| first candidate: rerank 10 + floor 0.10 + top 4 (profile) | – | 985 ms | est. $9.43 | 43/53 | 0.80 | – | – | – | 7 | 4/6 by the floor | – |
| minilm-l6 + top 3 (profile) | – | 339 ms | est. $11.94 | 48/53 | 0.86 | – | – | – | 0 | 1/6 by the floor | – |
| minilm-l6 + top 4 (profile) | – | 341 ms | est. $12.96 | 50/53 | 0.87 | – | – | – | 0 | 1/6 by the floor | – |
| rerank 10 + top 4 (profile) | – | 972 ms | est. $12.92 | 49/53 | 0.89 | – | – | – | 0 | 1/6 by the floor | – |
| minilm-l6 over 10 + top 4 (profile) | – | 231 ms | est. $12.97 | 49/53 | 0.87 | – | – | – | 0 | 1/6 by the floor | – |
| minilm-l6 + budget 1,200 (profile) | – | 337 ms | est. $12.18 | 49/53 | 0.87 | – | – | – | 0 | 1/6 by the floor | – |

A profile's out-of-scope column counts the questions the cosine floor stopped before any model call;
the other out-of-scope questions reach the model, which abstains on them in every saved run.

## 4. The default, and why

**Where the time goes is not where the money goes.** Retrieval is the cross-encoder and nothing else:
1,559 ms of 1,567 ms at the median, with embedding, both first stages and the
fusion under ten milliseconds together. Generation is four to five seconds and every cent. So a latency
cut is a reranker cut, and a cost cut is a passage cut, and the two are nearly independent.

**The cut designed before measuring was wrong.** The first candidate (`rerank 10 + floor 0.10 + top 4`)
combined the three changes that each looked free. Measured, the relevance floor alone drops
7 right passages and turns 7 in-scope questions into false abstentions:
the cross-encoder's score is a ranking signal, not a calibrated relevance, and a passage it scores
0.08 is often the right one (the identifier questions, where the question is one token and the
passage is a config block). The candidate as a whole lost
8 right passages. The regression gates would have refused it;
the profile refused it two minutes in, for nothing.

**Two cuts cost no right passage, and they do not combine.** The token budget (`budget 1,200 tokens`)
keeps 51/53 right passages at MRR 0.91, the same as the released pipeline, and sends
969 passage tokens at the median instead of 1,410: a
31% cut in the part of the input a configuration
controls, because the fifth passage was rarely the one that answered. The small cross-encoder
(`minilm-l6 over 20`) keeps 51/53 too and takes retrieval from 1,567 ms to 341 ms,
at MRR 0.88: the same passages, slightly less often first. Put together
(`minilm-l6 + budget 1,200`) they keep 49/53: the small model ranks the right passage
fourth or fifth a little more often, and the budget then cuts exactly there. Each is safe alone;
the pair is not, and only the measurement says so.

**The batch size runs the other way on a CPU.** Larger batches were slower (8: 12.3 s; 16: 14.1 s; 64: 16.9 s), since every text in a batch is padded to the longest and the chunks vary from a line to 512
tokens. The default stays at 16; on a GPU the curve would turn over. Indexing is paid once per index
version, so this is the smallest number in the report either way.

**v3 through the model.** `v3-budget-claude-code` is the budget run end to end, 69 questions
through `claude -p` with the judge, against the released run's gates: **passes**.

| | v2 (released) | v3 (budget) |
|---|---|---|
| right passage retrieved | 51/53 (96%) | 51/53 (96%) |
| MRR | 0.91 | 0.91 |
| right passage cited | 50/50 (100%) | 51/51 (100%) |
| claim share, median | 100% | 100% |
| judge: fabricated | 1/56 (2%) | 0/56 (0%) |
| false abstentions | 3/53 (6%) | 2/53 (4%) |
| out-of-scope abstained | 6/6 (100%) | 6/6 (100%) |
| adversarial held | 9/10 (90%) | 8/10 (80%) |
| expectations held | 63/69 (91%) | 64/69 (93%) |
| input tokens per call | 56,350 | 54,862 |
| cost per 1,000 | $41.66 | $39.79 |
| end to end, median | 6,181 ms | 5,714 ms |

Every gate passed.
 The one number that moved the wrong way is the adversarial set, 90% → 80%, at the gate's floor: `adv-portuguese` (held → abstained). One question on a set of ten is inside the noise the gates were sized for, and the budget is a plausible cause (fewer passages, less to answer from), so it is the first thing the next run under v3 should check.

**The cache, replayed.** `v3-budget-claude-code-replay` is the same 69 questions asked again under the same
version with the generation cache in front of the model: 0 input tokens per call,
$0.00 per 1,000, end to end 1,578 ms at the median (the retrieval, plus a lookup).
The answers are the stored ones, so every grounding number is the run's own. In production the saving
is the share of questions that repeat against the same index, which the trace log can count.

**The default.** The released pipeline stays bge-base over 20 candidates on the hybrid first stage: it
is the quality ceiling measured in stages 05 and 06 and nothing here beats it on the labels. The
budget is the one cut that is free on every retrieval number, and v3 is that cut, released if its run
clears the gates above. The small cross-encoder is the latency option: when a second a question matters
more than a few places of MRR, `minilm-l6 over 20` is a version away, without the budget. The
relevance floor is not an option at any value tried. Nothing in this report moves the generation
model, its effort or the prompt, which is where the next factor of two in cost would come from.
