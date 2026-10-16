"""Write the experiment manifest (stage 18) onto a run saved before it existed, from what the run and
the repository hold: the lineage and the generator from the run's manifest, the dataset version it
carries, the grader as the rubric reads today, the benchmark, the gates and the matrix as the files
are now. The hashes of the files are the current ones, which is right only while the files are the
ones the run was read under (the git log of eval/ says).

    uv run python scripts/backfill_experiments.py eval/runs/<name> [...]
"""

import sys
from pathlib import Path

from infra_docs_rag.evaluation.experiment import ExperimentManifest
from infra_docs_rag.evaluation.rubric import grader_version, load_rubric
from infra_docs_rag.evaluation.runs import Run


def main(paths: list[str]) -> None:
    rubric = load_rubric()
    for p in paths:
        run_dir = Path(p)
        run = Run.load(run_dir)
        if run.manifest.experiment is not None:
            print(f"{run.name}: has an experiment manifest; leaving it")
            continue
        run.manifest.experiment = ExperimentManifest.of(
            run.manifest, run.manifest.dataset, grader_version(rubric)
        )
        run.save(run_dir)
        e = run.manifest.experiment
        print(
            f"{run.name}: {e.version} · {e.retrieval_version} · {e.prompt_version} · dataset {e.dataset_version or '–'} · grader {e.grader_version}"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
