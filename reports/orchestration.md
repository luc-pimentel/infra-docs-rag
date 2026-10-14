# Orchestration: the pipeline run, broken, retried, re-indexed and rolled back

The stage contracts are in [`docs/pipeline.md`](../docs/pipeline.md). This is the chain exercised on
the real corpus on 2026-10-12, every command as typed and its output as printed (the model-loading
progress bars and 14 of 17 identical fetch errors cut). No model was called: the release steps read the
two benchmark runs saved in stage 08, and the one index build is the local bge-small model, 13 seconds.

## 1. A first run, and a second

The state file did not exist, so every stage ran. Fetch found the 17 URL sources already on disk with
their hashes in the manifest and downloaded nothing. Ingest rewrote the records (1.4 s, Tesseract
included), chunk cut them, and embed built the index into staging, checked it, and found that its
content hash was the version already live: the index built in stage 04 on 2026-09-30 and the one built
now from the same records are byte for byte the same, so nothing was replaced.

```
$ uv run infra-docs-rag pipeline run
fetch     running
fetch     ok        17 url sources on disk, 17 in the manifest (0.0s)
ingest    running
ingest    ok        22 records, 4 failed or empty, 4 duplicates (1.4s)
chunk     running
chunk     ok        497 chunks of 14 documents (section-aware 512/64 + context) (3.4s)
embed     running
embed     ok        index 3971d9f1e10e already present, 497 chunks, now current (12.8s)
```

The second run fingerprinted every input, found each stage's last run saw the same ones under the
same configuration with its outputs intact, and skipped all four. The ingest time in each record had
changed between the two runs; the records' content identity had not, which is what the fingerprint reads.

```
$ uv run infra-docs-rag pipeline run
fetch     skipped   up to date
ingest    skipped   up to date
chunk     skipped   up to date
embed     skipped   up to date
```

## 2. A stage fails, the rest wait, the retry runs only what it must

A refresh with every connection refused (the proxy variables point at a closed port). Each source's
error went into its manifest entry as it happened; the files already on disk were not touched. The
stage ended failed, the pipeline stopped before ingest, and the exit code was 1.

```
$ HTTPS_PROXY=http://127.0.0.1:9 HTTP_PROXY=http://127.0.0.1:9 uv run infra-docs-rag pipeline run --refresh
fetch     running
failed k8s-pods: ConnectError: [Errno 61] Connection refused
failed k8s-pods-no-slash: ConnectError: [Errno 61] Connection refused
failed k8s-service: ConnectError: [Errno 61] Connection refused
… 14 more …
fetch     FAILED    FetchFailed: 17 source(s) could not be fetched (k8s-pods: ConnectError: [Errno 61] Connection refused; k8s-pods-no-slash: ConnectError: [Errno 61] Connection refused; k8s-service: ConnectError: [Errno 61] Connection refused; and 14 more); data/raw/manifest.json has each error
stopped at fetch; fix the cause and run again, the stages before it will skip themselves
[exit 1]
```

Status shows the failed stage with its error and the three after it as stale, not because their
inputs changed but because a stage before them is not ok:

```
$ uv run infra-docs-rag pipeline status
stage     standing                           last run                took  note
fetch     failed                             2026-10-12 20:52:31     0.0s  FetchFailed: 17 source(s) could not be fetched (k8s-pods: ConnectError: [Errno 61] Connection refused; k8s-...
ingest    stale (an earlier stage is not ok) 2026-10-12 20:49:18     1.4s  22 records, 4 failed or empty, 4 duplicates
chunk     stale (an earlier stage is not ok) 2026-10-12 20:50:13     2.2s  497 chunks of 14 documents (section-aware 512/64 + context)
embed     stale (an earlier stage is not ok) 2026-10-12 20:50:26    12.6s  index 3971d9f1e10e already present, 497 chunks, now current
```

The retry, with the network back (and no `--refresh`): fetch ran, found every file on disk with a
hash, cleared the stale errors, and ended ok. Its output fingerprint was what ingest had already seen,
so ingest, chunk and embed skipped themselves.

```
$ uv run infra-docs-rag pipeline run
fetch     running
fetch     ok        17 url sources on disk, 17 in the manifest (0.0s)
ingest    skipped   up to date
chunk     skipped   up to date
embed     skipped   up to date
```

A failure inside a benchmark run is contained the same way one level down: a model call that raises
is tried once more and then becomes one case with outcome `error` and the reason, and the run
continues (`tests/test_pipeline.py::test_a_failed_model_call_is_one_case_not_the_end_of_the_run`).

## 3. A re-index next to the live one, and the switch back

The same pipeline from `chunk` on, with chunks of 256 tokens instead of 512. The chunk stage's
configuration changed, so it ran and wrote a second chunks file beside the first; embed built a second
index version beside the first and made it current.

```
$ uv run infra-docs-rag pipeline run --from chunk --size 256
chunk     running
chunk     ok        804 chunks of 14 documents (section-aware 256/64 + context) (2.1s)
embed     running
embed     ok        index 57a69e802a39 built, 804 chunks, now current (10.6s)

$ uv run infra-docs-rag pipeline index --list
index versions
  * bge-base/005a0870338a  388 chunks, whole-sections, built 2026-09-29
    bge-small/3971d9f1e10e  497 chunks, section-aware 512/64 + context, built 2026-09-30
  * bge-small/57a69e802a39  804 chunks, section-aware 256/64 + context, built 2026-10-12
  * minilm/2541a676bcea  388 chunks, whole-sections, built 2026-09-29
```

Rolling the index back is rewriting the pointer; the 256-token index stays on disk:

```
$ uv run infra-docs-rag pipeline index --use 3971d9f1e10e
bge-small now points at 3971d9f1e10e
```

Status now says chunk is stale (its last run was at 256 tokens, the configuration is 512 again) and
embed with it. The next plain run cut the 512-token chunks again, got the same bytes, and embed found
the same index version already present and simply pointed at it: 15 seconds, nothing rebuilt.

```
$ uv run infra-docs-rag pipeline run
fetch     skipped   up to date
ingest    skipped   up to date
chunk     running
chunk     ok        497 chunks of 14 documents (section-aware 512/64 + context) (2.2s)
embed     running
embed     ok        index 3971d9f1e10e already present, 497 chunks, now current (12.6s)
```

A build that fails its checks is covered by a test rather than shown here: a model whose queries all
land on one chunk fails the "a chunk's own text finds that chunk" check, the staging directory is
removed, and `current` is what it was
(`tests/test_pipeline.py::test_a_build_that_fails_its_check_leaves_the_live_index_alone`).

## 4. Releases: promote, refuse, roll back, roll forward

The two runs saved in stage 08 carry the lineage of what answered them (filled in by
`scripts/migrate_runs.py` from their manifests, against the index they were made on). The first
release has nothing to compare against and is accepted as the baseline:

```
$ uv run infra-docs-rag pipeline release --candidate v1-dense-claude-code --note "the stage 04 pipeline, the first release"
r1 is the current release (first release, no baseline to compare against); index bge-small/3971d9f1e10e is already current
```

The second candidate is compared to the first with every gate in `eval/gates.yaml`; it clears all ten
and becomes current:

```
$ uv run infra-docs-rag pipeline release --candidate v2-hybrid-rerank-claude-code --note "hybrid fusion and the cross-encoder, stages 05 and 06"
pass  abstention.out_of_scope: 6/6 (100%) → 6/6 (100%), +0%
pass  abstention.in_scope: 5/53 (9%) → 3/53 (6%), -4%
pass  grounding.right_cited: 46/48 (96%) → 50/50 (100%), +4%
pass  grounding.judge_fabricated: 3/54 (6%) → 1/56 (2%), -4%
pass  grounding.claim_cited_median: 100% against ≥ 0.9
pass  retrieval.right_retrieved: 46/53 (87%) → 51/53 (96%), +9%
pass  retrieval.mrr: 0.75 → 0.91, +0.15
pass  abstention.adversarial_held: 9/10 (90%) against ≥ 0.8
pass  cost.usd_per_1000: $42.90 → $41.66, -$1.24
pass  latency.end_to_end_ms_median: 6,181 ms against ≤ 15000
r2 is the current release (passes); index bge-small/3971d9f1e10e is already current
```

Promoting the dense pipeline back over it is a regression on four critical gates. It is refused,
the record is not written, and the exit code is 1:

```
$ uv run infra-docs-rag pipeline release --candidate v1-dense-claude-code --note "a regression, on purpose"
FAIL  abstention.in_scope: 3/53 (6%) → 5/53 (9%), +4%
FAIL  grounding.right_cited: 50/50 (100%) → 46/48 (96%), -4%
FAIL  grounding.judge_fabricated: 1/56 (2%) → 3/54 (6%), +4%
FAIL  retrieval.right_retrieved: 51/53 (96%) → 46/53 (87%), -9%
warn  retrieval.mrr: 0.91 → 0.75, -0.15
refused: blocked: 4 critical gate(s) failed (abstention.in_scope, grounding.right_cited, grounding.judge_fabricated, retrieval.right_retrieved); r2 stays the current release
[exit 1]
```

A rollback is the other way to get there, and it is allowed: the previous release is a known-good
configuration by definition. It names what differs between the two lineages, here only the retrieval
(both releases read the same index under the same prompt):

```
$ uv run infra-docs-rag pipeline rollback
rolled back from r2 to r1 (v1-dense, run v1-dense-claude-code)
index bge-small/3971d9f1e10e is already current
what changed: retrieval_version
```

`answer` follows the release. With no retrieval flag, it now runs the dense pipeline, and says so in
the first line of the prompt it would send:

```
$ uv run infra-docs-rag answer "How do I undo a bad release?" --show-prompt | head -3
[version v1-dense · index bge-small/3971d9f1e10e · prompt cite-v1+d9301ff2273d]
[system]
You answer questions about Kubernetes, Prometheus and Argo CD from documentation passages supplied with each question, and from nothing else.
```

(The prompt hash here is the API wording's; the runs were made through `claude -p`, whose
typed-citation wording hashes to `e732a9f0cc40`. Same prompt name, two backends, two hashes.)

Rolling forward is a release again, compared to r1 and accepted. The record keeps all three, which is
the point of a record:

```
$ uv run infra-docs-rag pipeline release --candidate v2-hybrid-rerank-claude-code --note "rolled forward after the rollback demo"
pass  abstention.out_of_scope: 6/6 (100%) → 6/6 (100%), +0%
…
r3 is the current release (passes); index bge-small/3971d9f1e10e is already current

$ uv run infra-docs-rag pipeline status
…
releases
    r1  v1-dense  run v1-dense-claude-code  2026-10-12  first release, no baseline to compare against
    r2  v2-hybrid-rerank  run v2-hybrid-rerank-claude-code  2026-10-12  passes
  * r3  v2-hybrid-rerank  run v2-hybrid-rerank-claude-code  2026-10-12  passes
current: r3
  version     v2-hybrid-rerank
  ingest      0.1.0+8e4b1a0
  chunking    section-aware-512-64-context+825a7940b716
  embedding   BAAI/bge-small-en-v1.5@5c38ec7c405e
  index       3971d9f1e10e
  prompt      cite-v1+e732a9f0cc40
  retrieval   v2-hybrid-rerank+cc59ba7751e4
  generator   claude-opus-5-5 via claude-code
  eval run    v2-hybrid-rerank-claude-code
```

[`eval/releases.yaml`](../eval/releases.yaml) is that record, committed.

## What this does not show

- A rollback that has to rebuild an index. Both releases here read the same index version; a release
  whose index is gone from disk prints that and leaves the pointer alone, and `pipeline run --to embed`
  at that chunking rebuilds it under the same version id (the hash is of the content).
- A prompt change. The registry holds one prompt; a second entry would be a new `prompt_version` on
  every run made with it and a `what changed: prompt_version` on the rollback.
- The evaluate and gate stages running under `pipeline run --to gate`. They are the same code as
  `evaluate-grounding` and `evaluate-pipeline`, wired as stages, and cost a benchmark run of the model;
  the test suite runs them on fixtures.
