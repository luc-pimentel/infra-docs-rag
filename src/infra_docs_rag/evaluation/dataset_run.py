"""Answer every dataset record through the pipeline and save the result as a run under `eval/runs/`,
in the same format as a benchmark run, so the stage 08 metrics, the gates and the comparisons read
it unchanged. What a dataset run adds: a right passage is one the record's `contexts` name (no
section label to resolve), every record carries an expectation and a reference answer, and each
answer gets the dataset reader's four probabilities beside the judge's three."""

from collections.abc import Callable
from dataclasses import dataclass

from ..embed.embedders import Embedder
from ..embed.evaluate import machine
from ..embed.index import Index
from ..generate.answer import Answer, answer, marked
from ..generate.evaluate import RETRIES, Case, Grounding
from ..generate.generator import Generator
from ..generate.judge import Judge, Judgement
from ..pipeline.manifest import PipelineManifest
from ..retrieve.rerank import Reranker
from ..retrieve.retriever import DEFAULT_RETRIEVAL, Retrieval, Retriever
from .dataset import Record
from .dataset_judge import Reader, Reading
from .runs import Expect, ReadingRecord, Run, from_grounding

# A record's category as the case kind the run format knows: the stage 08 metrics count out-of-scope
# and adversarial cases apart from the in-scope ones, and the rest keep the category's name.
KIND_OF = {"no-answer": "out-of-scope", "adversarial": "adversarial"}


def kind_of(r: Record) -> str:
    return KIND_OF.get(r.category, r.category)


def expect_of(r: Record) -> Expect:
    """`clarify` is a partial answer in the run's terms: answered, with the rest named."""
    return "partial" if r.expect == "clarify" else r.expect


@dataclass
class DatasetGrounding:
    grounding: Grounding
    readings: dict[str, Reading]  # per record id, when the reader ran


def run_dataset(
    records: list[Record],
    index: Index,
    embedder: Embedder,
    generator: Generator,
    reranker: Reranker | None = None,
    retrieval: Retrieval = DEFAULT_RETRIEVAL,
    limit: int | None = None,
    judge: Judge | None = None,
    reader: Reader | None = None,
    tracer: Callable[[Answer, Judgement | None], None] | None = None,
    log: Callable[[str], None] = print,
) -> DatasetGrounding:
    retriever = Retriever(index, embedder, reranker)
    todo = records[:limit] if limit is not None else records
    cases: list[Case] = []
    readings: dict[str, Reading] = {}
    for r in todo:
        a = answer(r.query, retriever, generator, retrieval, retries=RETRIES)
        flags = retrieved = cited = None
        if r.contexts:
            wanted = set(r.contexts)
            flags = [s.chunk.chunk_id in wanted for s in a.context.sources]
            retrieved = any(flags)
            cited = any(c.chunk_id in wanted for c in a.cited_chunks) if a.outcome == "answered" else None
        judgement = reading = None
        if a.outcome == "answered":
            if judge is not None:
                try:
                    judgement = judge.judge(a.context, r.query, marked(a))
                except ValueError as error:
                    log(f"{r.id}: the judge failed ({str(error)[:120]}); no reading")
            if reader is not None:
                try:
                    reading = reader.read(a.context, r.query, r.reference_answer, marked(a))
                except ValueError as error:
                    log(f"{r.id}: the reader failed ({str(error)[:120]}); no reading")
        cases.append(Case(r.id, kind_of(r), expect_of(r), r.note, a, retrieved, cited, judgement, flags))
        if reading is not None:
            readings[r.id] = reading
        if tracer is not None:
            tracer(a, judgement)
        line = f"{r.id}: {a.outcome}" + (
            f", {a.claim_share:.0%} of claims cited" if a.outcome == "answered" else ""
        )
        if a.outcome == "error":
            line += f" ({a.error})"
        if judgement is not None:
            line += f" · grounded {judgement.grounded:.2f}, fabricated {judgement.fabricated:.2f}"
        if reading is not None:
            line += f" · relevant {reading.relevant:.2f}, agrees {reading.reference_agrees:.2f}"
        log(line)
    judged = [c.judgement for c in cases if c.judgement is not None]
    grounding = Grounding(
        manifest=index.manifest,
        machine=machine(),
        retrieval=retrieval,
        generator=generator.model,
        cases=cases,
        prices=generator.prices,
        backend=generator.name,
        judge=judged[0].model if judged else None,
        effort=str(getattr(generator, "effort", "")),
    )
    return DatasetGrounding(grounding, readings)


def to_run(
    dg: DatasetGrounding,
    name: str,
    version: str = "default",
    description: str = "",
    lineage: PipelineManifest | None = None,
    dataset: str | None = None,
) -> Run:
    """The dataset run to save: a benchmark-format run with each case's reading attached and the
    version of the dataset it answered (`dataset_version`) on its manifest."""
    run = from_grounding(dg.grounding, name, version, description, lineage)
    run.manifest.dataset = dataset
    for c in run.cases:
        rd = dg.readings.get(c.id)
        if rd is not None:
            c.reading = ReadingRecord(
                relevant=rd.relevant,
                reference_agrees=rd.reference_agrees,
                deprecation_named=rd.deprecation_named,
                readings_named=rd.readings_named,
                model=rd.model,
                cost_usd=rd.cost_usd,
                ms=rd.ms,
            )
    return run
