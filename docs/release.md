# Release checklist

What a pipeline version has to clear before it answers questions, and the command that verifies each
item mechanically. A release is a decision taken under a named set of assets; every item below is a
file or a saved run in the repository, not a memory.

| # | Check | Command | Refuses when |
|---|---|---|---|
| 1 | The eval assets are frozen and have not moved: the dataset, the benchmark questions, the grader's prompt, the judge, the gates and the matrix, the safety policy | `dataset freeze --check` | a file's hash differs from [`eval/frozen.yaml`](../eval/frozen.yaml); `dataset freeze --note '…'` refreezes deliberately |
| 2 | The regression baseline is the run it says and answered the dataset as it is | `dataset baseline check` | the pinned run's manifest or the records' version differ from [`eval/baseline.yaml`](../eval/baseline.yaml) |
| 3 | The candidate is a named pipeline version with one variable changed against the baseline | `dataset compare --candidate <run>` | the experiment manifests differ in more than one of retrieval, prompt, corpus, model, dataset, grader, gates, safety |
| 4 | The dataset gates pass on every critical slice, and the record-level tally is not net negative | `dataset compare --candidate <run>` | a critical slice gate fails (block), or the gates pass and the change lost more records than it won (hold: a person reads the losses) |
| 5 | Every failed record has a named cause, and none of the new losses is a retrieval or prompt failure the change introduced without a reason | `dataset diagnose --run <run>` | (a reading, not a gate: the report in [`reports/traces.md`](../reports/traces.md)) |
| 6 | The benchmark gates pass against the current release | `evaluate-pipeline --baseline <release run> --candidate <run>` | a critical gate in [`eval/gates.yaml`](../eval/gates.yaml) fails |
| 7 | The candidate run was read under the frozen assets | `pipeline release --candidate <run>` | the run's experiment manifest names another dataset, benchmark, grader, judge, gate set or policy than the freeze |
| 8 | The canary answers its five known questions under the candidate version | `pipeline canary --version <name>` | an outcome, source or text check fails |
| 9 | The release records what it was decided with: the lineage, the experiment manifest, the freeze, the comparison's verdict | `pipeline release --candidate <run> --note '…'` | (written by the command; `eval/releases.yaml`) |

After the release: `dataset baseline set --run <the candidate's dataset run>` makes it the next
baseline, and `pipeline rollback` is the way back when the canary or the traces say so.

The order matters: 1 and 2 before anything is measured (so the numbers are read under what the
decision will cite), 3 to 6 on the saved runs (no model call; the PR check reruns 1, 2, 4 and 5 on
the committed files), 7 to 9 at the release. A step that refuses names what moved; nothing here is
overridden by a flag except `dataset compare --allow-confounded`, which writes **Confounded** into the
report so the decision carries the caveat.
