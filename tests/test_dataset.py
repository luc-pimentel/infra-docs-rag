import json
from pathlib import Path

import pytest
import yaml
from test_evaluation import two_runs  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.chunk.chunks import DEFAULT_CHUNKING, chunk_documents
from infra_docs_rag.evaluation.dataset import (
    CATEGORIES,
    FAILURE_MODES,
    OBSERVABLE,
    SIZE,
    Audit,
    AuditEntry,
    BenchmarkIndex,
    Dataset,
    Generated,
    GeneratedSet,
    Record,
    RecordSet,
    audited,
    benchmark_index,
    check,
    json_schema,
    load_dataset,
    load_records,
    modes_of,
    observe,
)
from infra_docs_rag.evaluation.dataset_generate import (
    SYSTEM,
    ClaudeCodeWriter,
    eligible,
    generate,
    parse,
    sample,
    save,
)
from infra_docs_rag.evaluation.dataset_report import write_report
from infra_docs_rag.generate.prompt import ABSTAIN
from infra_docs_rag.retrieve.evaluate import BenchmarkQuery

ROOT = Path(__file__).parents[1]
ABSTENTION = ABSTAIN


def record(**overrides) -> Record:
    base = dict(
        id="scale",
        query="kubectl scale replicas",
        category="fact-lookup",
        expect="answer",
        difficulty="easy",
        failure_mode="none",
        split="dev",
        origin="benchmark",
        source_doc_ids=["guide"],
        contexts=["guide:0"],
        reference_answer="Run kubectl scale.",
    )
    base.update(overrides)
    return Record(**base)


@pytest.fixture
def guide_chunks(fake_embedder, guide):  # noqa: F811
    return chunk_documents([guide], DEFAULT_CHUNKING, fake_embedder(max_tokens=512), {"guide": "Kubernetes"})


def test_a_record_checks_its_own_consistency():
    r = record()
    assert r.benchmark_id == "scale"  # a benchmark record defaults to its own id
    with pytest.raises(ValueError, match="abstention sentence"):
        record(expect="abstain", reference_answer="Nope.", contexts=[], source_doc_ids=[])
    with pytest.raises(ValueError, match="rests on no passage"):
        record(expect="abstain", reference_answer=ABSTENTION)
    with pytest.raises(ValueError, match="needs the contexts"):
        record(contexts=[], source_doc_ids=[])
    with pytest.raises(ValueError, match="source_doc_ids must be"):
        record(source_doc_ids=["other"])
    with pytest.raises(ValueError, match="not a chunk id"):
        record(contexts=["guide-0"])
    with pytest.raises(ValueError, match="at least two passages"):
        record(category="multi-hop", contexts=["guide:0"])
    with pytest.raises(ValueError, match="expects an abstention"):
        record(category="no-answer", expect="answer")
    with pytest.raises(ValueError, match="clarification or a partial"):
        record(category="ambiguous", expect="answer")
    with pytest.raises(ValueError, match="benchmark_id is for"):
        record(origin="hand-written", benchmark_id="scale")
    ok = record(
        category="no-answer", expect="abstain", reference_answer=ABSTENTION, contexts=[], source_doc_ids=[]
    )
    assert ok.contexts == []


def test_the_audit_turns_kept_and_edited_examples_into_records_and_refuses_half_decisions():
    generated = GeneratedSet(
        chunking="section-aware-512-64-context",
        model="claude-opus-5-5",
        backend="claude-code",
        effort="low",
        prompt_sha256="abc",
        seed=1,
        generated_at="2026-10-13T00:00:00+00:00",
        examples=[
            Generated(
                id="gen-01",
                chunk_id="guide:0",
                source_doc_id="guide",
                section="Scaling",
                query="How do I scale?",
                reference_answer="kubectl scale.",
                difficulty="easy",
                tags=["scale"],
            ),
            Generated(
                id="gen-02",
                chunk_id="guide:5",
                source_doc_id="guide",
                section="Rolling Update",
                query="maxSurge?",
                reference_answer="Extra pods.",
                difficulty="medium",
            ),
            Generated(
                id="gen-03",
                chunk_id="guide:9",
                source_doc_id="guide",
                section="Links",
                query="What links?",
                reference_answer="Some.",
                difficulty="easy",
            ),
        ],
    )
    audit = Audit(
        auditor="test",
        generated="eval/dataset/generated.yaml",
        decisions=[
            AuditEntry(id="gen-01", decision="kept", reason="fine"),
            AuditEntry(
                id="gen-02",
                decision="edited",
                reason="too terse",
                query="What does maxSurge allow?",
                split="dev",
            ),
            AuditEntry(id="gen-03", decision="discarded", reason="a list of links"),
        ],
    )
    kept = audited(generated, audit)
    assert [r.id for r in kept] == ["gen-01", "gen-02"]
    assert kept[0].origin == "generated" and kept[0].split == "test" and kept[0].contexts == ["guide:0"]
    assert kept[1].query == "What does maxSurge allow?" and kept[1].split == "dev"
    assert kept[1].reference_answer == "Extra pods." and kept[1].note.startswith("edited:")
    with pytest.raises(ValueError, match="nothing changed"):
        AuditEntry(id="x", decision="edited", reason="hm")
    with pytest.raises(ValueError, match="nothing should change"):
        AuditEntry(id="x", decision="kept", reason="hm", query="changed")
    with pytest.raises(ValueError, match="say why"):
        AuditEntry(id="x", decision="kept", reason=" ")


def test_check_reads_the_dataset_against_the_chunks_sources_and_benchmark(guide_chunks, guide):  # noqa: F811
    scaling = next(c.chunk_id for c in guide_chunks if c.section == "Scaling")
    rolling = next(c.chunk_id for c in guide_chunks if c.section == "Rolling Update")
    docs = {"guide": guide}
    bench = BenchmarkIndex(
        queries={
            "scale": BenchmarkQuery(
                id="scale",
                kind="paraphrase",
                text="kubectl scale replicas",
                expect={"source": "guide", "sections": ["Scaling"]},
                hypothesis="h",
            ),
            "bread": BenchmarkQuery(id="bread", kind="out-of-scope", text="sourdough", hypothesis="h"),
        },
        adversarial={"adv-memory": "abstain"},
    )
    good = RecordSet(
        chunking="section-aware-512-64-context",
        records=[
            record(contexts=[scaling]),
            record(
                id="bread",
                category="no-answer",
                expect="abstain",
                reference_answer=ABSTENTION,
                contexts=[],
                source_doc_ids=[],
            ),
            record(
                id="adv-memory",
                category="adversarial",
                expect="abstain",
                reference_answer=ABSTENTION,
                contexts=[],
                source_doc_ids=[],
            ),
            record(id="surge", origin="hand-written", category="stale", contexts=[rolling]),
            record(id="two", origin="hand-written", category="multi-hop", contexts=[scaling, rolling]),
            record(
                id="amb",
                origin="hand-written",
                category="ambiguous",
                expect="clarify",
                contexts=[scaling],
            ),
        ],
    )
    ds = Dataset(good, None, None)
    problems = check(ds, guide_chunks, docs, bench, {"guide"})
    assert [p for p in problems if "curated records" not in p] == []  # only the size rule (6 < 30)
    assert not ds.ok

    bad = RecordSet(
        chunking="section-aware-512-64-context",
        records=[
            record(contexts=[rolling]),  # labelled Scaling, cites Rolling Update
            record(id="bread", contexts=[scaling]),  # the benchmark expects an abstention
            record(
                id="adv-memory", category="adversarial", contexts=[scaling]
            ),  # adversarial expects abstain
            record(id="ghost", contexts=["guide:999999"]),  # no such chunk, no such benchmark question
            record(id="src", origin="hand-written", source_doc_ids=["nope"], contexts=["nope:0"]),
        ],
    )
    problems = check(Dataset(bad, None, None), guide_chunks, docs, bench, {"guide"})
    text = "\n".join(problems)
    assert "not among the passages the benchmark labels" in text
    assert "expects an abstention on bread" in text
    assert "expects abstain on adv-memory" in text
    assert "guide:999999 is not in" in text and "ghost does not exist" in text
    assert "source nope is not in sources.yaml" in text
    for category in CATEGORIES:
        present = category in ("fact-lookup", "adversarial")  # the two the bad set has
        assert (f"no curated record in category {category}" in text) is not present


def test_observe_reads_each_run_for_the_failure_modes_it_showed(two_runs):  # noqa: F811
    base, cand = two_runs
    records = [
        record(),  # scale: held in base, partly uncited in cand
        record(id="surge", query="maxSurge"),
        record(
            id="adv-memory",
            category="adversarial",
            expect="abstain",
            reference_answer=ABSTENTION,
            contexts=[],
            source_doc_ids=[],
            failure_mode="answered-anyway",
        ),
        record(id="unrelated", origin="hand-written", category="stale"),
    ]
    observed = observe(records, [base, cand])
    assert [o.run for o in observed["scale"]] == ["base-fake", "cand-fake"]
    assert observed["scale"][0].modes == [] and observed["scale"][0].held
    assert "uncited-claims" in observed["scale"][1].modes
    assert "wrong-passage" in observed["surge"][1].modes
    assert "answered-anyway" in observed["adv-memory"][1].modes
    assert observed["unrelated"] == []
    for run in (base, cand):
        for c in run.cases:
            assert set(modes_of(c, 0.5, 0.5)) <= set(OBSERVABLE)


def test_the_report_writes_every_section(two_runs, guide_chunks, tmp_path):  # noqa: F811
    base, cand = two_runs
    scaling = next(c.chunk_id for c in guide_chunks if c.section == "Scaling")
    curated = RecordSet(
        chunking="section-aware-512-64-context",
        records=[
            record(contexts=[scaling]),
            record(
                id="surge", query="maxSurge", contexts=[scaling], failure_mode="wrong-passage", split="test"
            ),
            record(
                id="st",
                origin="hand-written",
                category="stale",
                failure_mode="stale-answer",
                contexts=[scaling],
            ),
        ],
    )
    generated = GeneratedSet(
        chunking="section-aware-512-64-context",
        model="claude-opus-5-5",
        backend="claude-code",
        effort="low",
        prompt_sha256="abc",
        seed=1,
        generated_at="2026-10-13T00:00:00+00:00",
        examples=[
            Generated(
                id="gen-01",
                chunk_id=scaling,
                source_doc_id="guide",
                section="Scaling",
                query="How do I scale?",
                reference_answer="kubectl scale.",
                difficulty="easy",
            )
        ],
    )
    audit = Audit(
        auditor="test",
        generated="generated.yaml",
        decisions=[
            AuditEntry(id="gen-01", decision="edited", reason="wordier", query="How do I scale a Deployment?")
        ],
    )
    ds = Dataset(curated, generated, audit, problems=["3 curated records; the set is meant to hold 30 to 50"])
    out = tmp_path / "dataset.md"
    write_report(ds, observe(curated.records, [base, cand]), guide_chunks, [base.name, cand.name], out)
    text = out.read_text()
    for heading in (
        "## Summary",
        "## 1. The schema",
        "## 2. The curated set",
        "## 3. What the runs showed",
        "## 4. The generated",
        "## 5. Coverage",
        "## 6. Running it",
        "## Problems",
    ):
        assert heading in text
    assert "| `surge` | `wrong-passage` | ✓ |" in text  # designed wrong-passage, seen in cand
    assert "Not measured by any run yet: `stale-answer`" in text
    assert "~~How do I scale?~~ → How do I scale a Deployment?" in text
    assert "1 kept" not in text.split("## 4")[0] or "0 kept, 1 edited" in text
    for mode in FAILURE_MODES:
        assert f"`{mode}`" in text


def test_sampling_is_stratified_and_repeatable(guide_chunks):
    a = sample(guide_chunks, 3, seed=7)
    b = sample(guide_chunks, 3, seed=7)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b] and len(a) == 3
    assert sample(guide_chunks, 100, seed=1) and len(sample(guide_chunks, 100, seed=1)) == len(guide_chunks)
    assert sample([], 5, seed=1) == []
    used = {guide_chunks[0].chunk_id}
    assert all(c.chunk_id != guide_chunks[0].chunk_id for c in eligible(guide_chunks, used))


def test_parse_reads_json_with_or_without_a_fence():
    assert parse('{"query": "q", "reference_answer": "a"}')["query"] == "q"
    assert parse('```json\n{"skip": "a table"}\n```')["skip"] == "a table"
    assert parse('Here you go:\n{"query": "q"}\nThanks')["query"] == "q"
    with pytest.raises(ValueError, match="no JSON object"):
        parse("nothing here")


def test_generate_runs_the_cli_per_chunk_and_keeps_what_it_can_read(guide_chunks, tmp_path):
    """A `claude` that prints one scripted JSON reply per call, in order."""
    replies = [
        {
            "result": '{"query": "How do I scale?", "reference_answer": "kubectl scale.", "difficulty": "easy", "tags": ["scale"]}',
            "usage": {"input_tokens": 10, "output_tokens": 5},
        },
        {"result": '{"skip": "a list of links"}', "usage": {}},
        {"result": "not json at all", "usage": {}},
        {
            "result": '```json\n{"query": "maxSurge?", "reference_answer": "Extra pods.", "difficulty": "weird"}\n```',
            "usage": {},
        },
    ]
    for i, r in enumerate(replies):
        (tmp_path / f"reply.{i}").write_text(json.dumps(r))
    counter = tmp_path / "n"
    counter.write_text("0")
    script = tmp_path / "claude"
    script.write_text(
        "#!/bin/sh\n"
        f'n=$(cat "{counter}"); echo $((n+1)) > "{counter}"\n'
        f'cat > "{tmp_path}/stdin.$n"\n'
        f'cat "{tmp_path}/reply.$n"\n'
    )
    script.chmod(0o755)
    writer = ClaudeCodeWriter(effort="low", binary=str(script), cwd=tmp_path)
    assert "--system-prompt" in writer.command() and SYSTEM in writer.command()
    four = (guide_chunks * 4)[:4]  # the guide has fewer chunks than replies; ids may repeat here
    gen = generate(four, writer, log=lambda _: None)
    assert [g.id for g in gen.examples] == ["gen-01", "gen-02"]
    assert gen.examples[0].query == "How do I scale?" and gen.examples[0].chunk_id == four[0].chunk_id
    assert gen.examples[1].difficulty == "medium"  # an unknown difficulty falls back
    assert [s.why for s in gen.skipped][0] == "a list of links" and "unreadable" in gen.skipped[1].why
    assert gen.input_tokens == 10 and gen.output_tokens == 5
    assert four[0].text in (tmp_path / "stdin.0").read_text()
    out = tmp_path / "generated.yaml"
    saved = save(gen, "section-aware-512-64-context", writer, 13, out)
    assert saved.seed == 13 and saved.model == "claude-opus-5-5" and len(saved.prompt_sha256) == 12
    loaded = yaml.safe_load(out.read_text())
    assert loaded["examples"][0]["id"] == "gen-01" and "# Chunks the writer declined" in out.read_text()
    with pytest.raises(ValueError, match="unknown model"):
        ClaudeCodeWriter(model="gpt-7")
    with pytest.raises(ValueError, match="not on PATH"):
        ClaudeCodeWriter(binary=str(tmp_path / "missing"), cwd=tmp_path).ask("x")


def test_the_json_schema_names_every_failure_mode():
    schema = json_schema()
    assert schema["title"] == "infra-docs-rag evaluation record"
    assert schema["$defs"]["FailureMode"]["enum"] == list(FAILURE_MODES)
    assert set(schema["properties"]) >= {
        "query",
        "contexts",
        "reference_answer",
        "tags",
        "source_doc_ids",
        "difficulty",
        "failure_mode",
        "split",
    }


def test_the_repo_dataset_loads_checks_and_matches_its_schema_file():
    ds = load_dataset(ROOT / "eval/dataset")
    assert SIZE[0] <= len(ds.curated.records) <= SIZE[1]
    assert {r.category for r in ds.curated.records} == set(CATEGORIES)
    assert {r.split for r in ds.curated.records} == {"dev", "test"}
    chunks_file = ROOT / "data/chunks" / f"{ds.curated.chunking}.jsonl"
    if not chunks_file.exists():
        pytest.skip("the chunks are not built in this checkout")
    from infra_docs_rag.chunk.chunks import usable
    from infra_docs_rag.ingest.sources import load_sources
    from infra_docs_rag.pipeline.stages import Paths, load_chunks, load_documents

    chunks = load_chunks(chunks_file)
    docs = {d.source_id: d for d in usable(load_documents(ROOT / "data/processed/documents.jsonl"))}
    sources = {s.id for s in load_sources(ROOT / "sources.yaml")}
    bench = benchmark_index(tuple(ROOT / p for p in Paths().benchmark_files))
    assert check(ds, chunks, docs, bench, sources) == []
    schema_file = ROOT / "eval/dataset/schema.json"
    if schema_file.exists():
        assert json.loads(schema_file.read_text()) == json_schema()


def test_load_records_refuses_a_repeated_id(tmp_path):
    path = tmp_path / "records.yaml"
    r = record().model_dump()
    path.write_text(yaml.safe_dump({"chunking": "x", "records": [r, r]}))
    with pytest.raises(ValueError, match="repeats"):
        load_records(path)
