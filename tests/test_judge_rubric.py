import json
from pathlib import Path

import pytest
import yaml
from test_evaluation import two_runs  # noqa: F401  (a fixture)
from test_retrieve_evaluate import guide  # noqa: F401  (a fixture)

from infra_docs_rag.evaluation.judge_calibrate import (
    CalibrationSet,
    calibrate_judge,
    items_of,
    load_calibration,
    reader_agreement,
    weighted_kappa,
)
from infra_docs_rag.evaluation.judge_report import write_report
from infra_docs_rag.evaluation.rubric import (
    RUBRIC,
    Grade,
    GradeSet,
    Rubric,
    RubricGrader,
    Score,
    grader_state,
    json_schema,
    load_rubric,
    new_grade_set,
    passages_of,
    prompt_sha256,
    system_prompt,
)
from infra_docs_rag.evaluation.runs import JudgementRecord, ReadingRecord

ROOT = Path(__file__).parents[1]


def test_the_repo_rubric_loads_and_renders_into_the_grader_prompt():
    rubric = load_rubric(ROOT / RUBRIC)
    assert rubric.ids == ["groundedness", "completeness", "abstention"]
    assert rubric.pass_at("groundedness") == 1 and rubric.passes("completeness", 1) is True
    assert rubric.passes("groundedness", None) is None and rubric.passes("abstention", 1) is False
    prompt = system_prompt(rubric)
    for d in rubric.dimensions:
        assert f"## {d.id}" in prompt and f"Question: {d.question}" in prompt
        assert all(f"- {s}: " in prompt for s in (0, 1, 2))
        # the examples are not shown to the grader, so it is read on them like on any other case
        assert all(e.id not in prompt for e in d.examples)
    assert rubric.abstention_sentence in prompt and '"score": 2' in prompt
    schema = json_schema(rubric)
    assert schema["required"] == rubric.ids and schema["additionalProperties"] is False
    assert schema["properties"]["abstention"]["properties"]["score"]["enum"] == [0, 1, 2, None]
    assert len(prompt_sha256(rubric)) == 12
    # every example names a real answer: a run case, or a perturbed answer in the calibration file
    cal = yaml.safe_load((ROOT / "eval/calibration.yaml").read_text())
    labelled = {x["id"] for x in cal["labels"]} | {x.get("base") for x in cal["labels"]}
    for d in rubric.dimensions:
        for e in d.examples:
            assert e.id in labelled, f"{d.id} example {e.id} is not a labelled answer"


def test_a_rubric_rejects_missing_scores_and_thresholds():
    base = {
        "name": "t",
        "abstention_sentence": "No.",
        "dimensions": [
            {
                "id": "g",
                "question": "q",
                "reads": "r",
                "applies_when": "a",
                "scores": {0: "x", 1: "y", 2: "z"},
            }
        ],
        "operating": {"g": {"pass_at": 2}},
    }
    Rubric.model_validate(base)
    with pytest.raises(ValueError, match="exactly"):
        Rubric.model_validate({**base, "dimensions": [{**base["dimensions"][0], "scores": {0: "x", 2: "z"}}]})
    with pytest.raises(ValueError, match="operating thresholds"):
        Rubric.model_validate({**base, "operating": {}})
    with pytest.raises(ValueError, match="pass_at must be 1 or 2"):
        Rubric.model_validate({**base, "operating": {"g": {"pass_at": 0}}})
    with pytest.raises(ValueError, match="not one of"):
        Rubric.model_validate(
            {
                **base,
                "dimensions": [{**base["dimensions"][0], "examples": [{"score": 3, "id": "x", "why": "w"}]}],
            }
        )


@pytest.fixture
def fake_claude(tmp_path):
    def install(reply, returncode=0):
        script = tmp_path / "claude"
        script.write_text(
            "#!/bin/sh\n"
            f'printf "%s\\n" "$@" > "{tmp_path}/args"\n'
            f'cat > "{tmp_path}/stdin"\n'
            f"cat <<'JSON'\n{json.dumps(reply)}\nJSON\n"
            f"exit {returncode}\n"
        )
        script.chmod(0o755)
        return script

    return install


def test_the_grader_runs_claude_with_the_schema_and_reads_the_structured_scores(fake_claude, tmp_path):
    rubric = load_rubric(ROOT / RUBRIC)
    scored = {
        "groundedness": {"score": 1, "reason": "a consequence beyond the passages"},
        "completeness": {"score": None, "reason": "the reference is the abstention sentence"},
        "abstention": {"score": 0, "reason": "answered from memory"},
    }
    script = fake_claude(
        {
            "is_error": False,
            "result": json.dumps(scored),
            "structured_output": scored,
            "usage": {"input_tokens": 100, "cache_read_input_tokens": 40, "output_tokens": 50},
            "modelUsage": {"claude-opus-5-5": {}},
            "duration_api_ms": 1234,
        }
    )
    grader = RubricGrader(rubric, binary=str(script), cwd=tmp_path)
    g = grader.grade("x", "PASSAGES\n...")
    assert (g.score("groundedness"), g.score("completeness"), g.score("abstention")) == (1, None, 0)
    assert g.scores["abstention"].reason == "answered from memory" and g.ms == 1234
    assert g.input_tokens == 140 and g.output_tokens == 50
    assert g.cost_usd == pytest.approx((100 * 4.0 + 40 * 0.2 + 50 * 20.0) / 1e6)
    args = (tmp_path / "args").read_text()  # one argument per line; the system prompt spans several
    assert args.startswith("-p\n--system-prompt\n" + grader.system + "\n--tools\n\n--max-turns\n1\n")
    lines = args.splitlines()
    assert json.loads(lines[lines.index("--json-schema") + 1]) == json_schema(rubric)
    assert lines[-5:] == ["--model", "claude-opus-5-5", "--effort", "medium", "--no-session-persistence"]
    assert (tmp_path / "stdin").read_text() == "PASSAGES\n..."

    # without structured_output the JSON in the result text is read
    g = RubricGrader(
        rubric, binary=str(fake_claude({"result": "```json\n" + json.dumps(scored) + "\n```"})), cwd=tmp_path
    ).grade("x", "s")
    assert g.score("groundedness") == 1
    with pytest.raises(ValueError, match="no score for completeness"):
        RubricGrader(
            rubric,
            binary=str(fake_claude({"structured_output": {"groundedness": {"score": 2}}})),
            cwd=tmp_path,
        ).grade("x", "s")
    with pytest.raises(ValueError, match="not one of"):
        RubricGrader(
            rubric,
            binary=str(
                fake_claude({"structured_output": {**scored, "abstention": {"score": 3, "reason": ""}}})
            ),
            cwd=tmp_path,
        ).grade("x", "s")
    with pytest.raises(ValueError, match="exited 2"):
        RubricGrader(rubric, binary=str(fake_claude({}, returncode=2)), cwd=tmp_path).grade("x", "s")
    with pytest.raises(ValueError, match="failed: quota"):
        RubricGrader(
            rubric, binary=str(fake_claude({"is_error": True, "result": "quota"})), cwd=tmp_path
        ).grade("x", "s")
    with pytest.raises(ValueError, match="unknown model"):
        RubricGrader(rubric, model="gpt", binary=str(script))


def test_a_grade_set_saves_and_loads_back(tmp_path, fake_claude):
    rubric = load_rubric(ROOT / RUBRIC)
    grader = RubricGrader(rubric, binary=str(fake_claude({})), cwd=tmp_path)
    g = Grade(
        id="a",
        scores={
            "groundedness": Score(score=2),
            "completeness": Score(score=None),
            "abstention": Score(score=1),
        },
        model="m",
        ms=1,
        cost_usd=0.5,
    )
    gs = new_grade_set("run-x", "labels.yaml", rubric, grader, [g])
    gs.save(tmp_path / "grades.json")
    back = GradeSet.load(tmp_path / "grades.json")
    assert back.run == "run-x" and back.by_id()["a"].score("abstention") == 1 and back.cost_usd == 0.5
    assert back.prompt_sha256 == prompt_sha256(rubric) and back.graded_at[:2] == "20"
    with pytest.raises(FileNotFoundError, match="judge grade"):
        GradeSet.load(tmp_path / "none.json")


def test_weighted_kappa_rewards_near_misses_over_far_ones():
    assert weighted_kappa([(2, 2), (1, 1), (0, 0)]) == 1.0
    assert weighted_kappa([(2, 0), (0, 2)]) == -1.0
    assert weighted_kappa([(2, 2), (2, 2)]) is None  # no variance: chance is everything
    assert weighted_kappa([]) is None
    near = weighted_kappa([(2, 2), (1, 2), (0, 0), (2, 1)])
    far = weighted_kappa([(2, 2), (0, 2), (0, 0), (2, 0)])
    assert near is not None and far is not None and near > far


RUBRIC_TEXT = """
name: t
abstention_sentence: The indexed documentation does not cover this question.
dimensions:
  - {id: groundedness, question: q, reads: r, applies_when: a, scores: {0: x, 1: y, 2: z}}
  - {id: completeness, question: q, reads: r, applies_when: a, scores: {0: x, 1: y, 2: z}}
  - {id: abstention, question: q, reads: r, applies_when: a, scores: {0: x, 1: y, 2: z}}
operating: {groundedness: {pass_at: 2}, completeness: {pass_at: 1}, abstention: {pass_at: 1}}
"""

CALIBRATION = """
run: cand-fake
rubric: r.yaml
readers: {a: first, b: second}
resolved_by: a
labels:
  - id: scale
    groundedness: 2
    completeness: 2
    abstention: 2
    a: {groundedness: 2, completeness: 2, abstention: 2}
    b: {groundedness: 1, completeness: 2, abstention: 2}
    note: filler is not a claim
    resolution: a stands
  - id: surge
    groundedness: 2
    completeness: 1
    abstention: 1
    a: {groundedness: 2, completeness: 1, abstention: 1}
    b: {groundedness: 2, completeness: 1, abstention: 1}
    note: the wrong passage, same text
  - id: adv-memory
    groundedness: 0
    completeness: null
    abstention: 0
    a: {groundedness: 0, completeness: null, abstention: 0}
    b: {groundedness: 0, completeness: null, abstention: 0}
    note: answered from memory
  - id: scale~five
    base: scale
    perturbation: a number no passage gives
    answer: Run kubectl scale to change the replicas.[1] The default is 5 replicas.
    groundedness: 0
    completeness: 2
    abstention: 2
    a: {groundedness: 0, completeness: 2, abstention: 2}
    b: {groundedness: 0, completeness: 2, abstention: 2}
    note: five is invented
"""


def test_calibration_labels_are_checked_against_the_run_and_the_rubric(two_runs, tmp_path):  # noqa: F811
    base, cand = two_runs
    rubric = Rubric.model_validate(yaml.safe_load(RUBRIC_TEXT))
    path = tmp_path / "calibration.yaml"
    path.write_text(CALIBRATION)
    cal = load_calibration(path, cand, rubric)
    assert len(cal.labels) == 4 and [x.id for x in cal.perturbed] == ["scale~five"] and len(cal.real) == 3
    assert cal.labels[0].disputed() == ["groundedness"] and cal.labels[1].disputed() == []
    with pytest.raises(ValueError, match="labels run 'cand-fake', not 'base-fake'"):
        load_calibration(path, base, rubric)
    doc = yaml.safe_load(CALIBRATION)

    def rejects(match, **changes):
        bad = {**doc, "labels": [dict(x) for x in doc["labels"]]}
        bad["labels"][-1].update(changes)
        path.write_text(yaml.safe_dump(bad))
        with pytest.raises(ValueError, match=match):
            load_calibration(path, cand, rubric)

    rejects("labelled twice", id="scale", base=None, answer=None, perturbation="")
    rejects("not in run", id="ghost~x", base="ghost")
    rejects("should start with", id="other~x")
    rejects("needs both", answer=None)
    rejects("needs `perturbation`", perturbation="")
    rejects("abstention always applies", abstention=None)
    rejects("not one of", groundedness=3)
    rejects("`resolution` is empty", b={"groundedness": 1, "completeness": 2, "abstention": 2})
    with pytest.raises(FileNotFoundError):
        load_calibration(tmp_path / "none.yaml", cand, rubric)


def test_the_calibration_compares_the_grades_to_the_person_and_writes_the_report(two_runs, tmp_path):  # noqa: F811
    _, cand = two_runs
    rubric = Rubric.model_validate(yaml.safe_load(RUBRIC_TEXT))
    path = tmp_path / "calibration.yaml"
    path.write_text(CALIBRATION)
    cal = load_calibration(path, cand, rubric)
    references = {
        "scale": "Run kubectl scale.",
        "surge": "maxSurge allows extra Pods.",
        "adv-memory": rubric.abstention_sentence,
    }
    items = items_of(cal, cand, references)
    assert [i.id for i in items] == ["scale", "surge", "adv-memory", "scale~five"]
    assert items[3].answer.endswith("The default is 5 replicas.") and items[3].case.id == "scale"
    assert (
        items[3].judgement is None and items[0].judgement is not None
    )  # a perturbed answer has no run reading
    state = items[0].state()
    assert (
        state.startswith("PASSAGES\n[1] ") and "\n\nREFERENCE ANSWER\nRun kubectl scale.\n\nANSWER\n" in state
    )
    assert grader_state(passages_of(items[0].case), "q", "r", "a").endswith(
        "QUESTION\nq\n\nREFERENCE ANSWER\nr\n\nANSWER\na"
    )

    def grade(id, g, c, a, judgement=None):
        return Grade(
            id=id,
            scores={
                "groundedness": Score(score=g, reason=f"g{g}"),
                "completeness": Score(score=c, reason=f"c{c}"),
                "abstention": Score(score=a, reason=f"a{a}"),
            },
            model="fake",
            ms=1,
            judgement=judgement,
        )

    grades = GradeSet(
        run="cand-fake",
        labels="x",
        rubric="t",
        prompt_sha256="abc",
        model="fake",
        backend="fake",
        effort="low",
        graded_at="2026-10-15T00:00:00+00:00",
        grades=[
            grade("scale", 2, 2, 2),  # exact
            grade(
                "surge", 2, 2, 0
            ),  # completeness one off (passes either way at ≥ 1), abstention one off (flips the pass)
            grade("adv-memory", 2, None, 0),  # groundedness two off: the fabrication missed
            grade(
                "scale~five",
                0,
                2,
                2,
                JudgementRecord(
                    grounded=0.2, fabricated=0.8, gap_named=0.1, model="fake-jev", cost_usd=0.0, ms=1
                ),
            ),
        ],
    )
    jc = calibrate_judge(cand, rubric, cal, grades, references)
    assert jc.ungraded == []
    by = {d.dimension: d for d in jc.dimensions}
    g = by["groundedness"]
    assert (
        g.confusion.n == 4
        and g.confusion.exact == 3
        and g.confusion.off_by_two == 1
        and g.confusion.counts[2][0] == 1
    )
    assert g.exact_share == 0.75 and [d.item.id for d in g.disagreements] == ["adv-memory"]
    assert g.pass_agreement.n == 4 and g.pass_agreement.fp == 1 and g.pass_agreement.agreement == 0.75
    c = by["completeness"]
    assert c.confusion.n == 3 and c.confusion.both_na == 1 and c.confusion.counts[2][1] == 1
    assert c.pass_agreement.agreement == 1.0 and dict(c.pass_sweep)[2].agreement == pytest.approx(2 / 3)
    a = by["abstention"]
    assert (
        a.confusion.exact == 3 and a.pass_agreement.fn == 1 and a.best_pass_at[0] == 2
    )  # surge: person 1, grader 0
    assert a.kappa is not None and g.kappa is not None
    # the overall verdict: every applicable dimension passes
    o = jc.overall
    # scale: both pass; surge: the grader fails abstention, the person passes; adv-memory and scale~five:
    # both fail (abstention 0 and groundedness 0 by both)
    assert (o.tp, o.fp, o.fn, o.tn) == (1, 0, 1, 2)
    # the probability judge: the run's readings on the real answers, the grade's on the perturbed one
    sig = {s.name: s for s in jc.signals}
    grounded = sig["jev grounded"]
    assert (
        grounded.agreement.n == 4
    )  # scale 0.3, surge 0.9, adv-memory 0.5 from the run; scale~five 0.2 from its grade
    fabricated = sig["jev fabricated"]
    assert fabricated.agreement.n == grounded.agreement.n and fabricated.sweep[0][0] == 0.3
    assert (
        any(i.id == "scale~five" for i, _, _ in grounded.disagreements) is False
    )  # 0.2 < 0.5: ungrounded, person 0: agree
    assert sig["reader agrees"].agreement.n == 0  # a benchmark run carries no reader readings
    readers = {r.dimension: r for r in jc.readers}
    assert (
        readers["groundedness"].n == 4
        and readers["groundedness"].exact == 3
        and [x.id for x in readers["groundedness"].disputed] == ["scale"]
    )
    assert readers["completeness"].n == 3 and readers["completeness"].exact == 3
    assert reader_agreement(cal, "abstention").kappa == 1.0

    write_report(jc, tmp_path / "judge.md")
    report = (tmp_path / "judge.md").read_text()
    assert report.startswith("# Judge calibration report") and "4 answers: 3 as the saved run" in report
    assert "| Grader: groundedness | 75% exact over 4" in report
    assert (
        "| `scale~five` | paraphrase | answer | perturbed | 0 / 2 / 2 | 0 / 2 / 2 | 0.20 | 0.80 | – |"
        in report
    )
    assert "### `adv-memory`" in report and "**groundedness**: grader 2, person 0. Grader: g2" in report
    assert "**`scale`** (groundedness)" in report and "> a stands" in report
    assert "| grader \\ person | 0 | 1 | 2 |" in report

    # a labelled answer with no grade is reported and the CLI exits 1 on it
    grades.grades.pop()
    jc = calibrate_judge(cand, rubric, cal, grades, references)
    assert jc.ungraded == ["scale~five"]
    write_report(jc, tmp_path / "judge.md")
    assert "| Not graded | `scale~five` |" in (tmp_path / "judge.md").read_text()


def test_the_repo_calibration_file_loads_against_its_run_and_rubric():
    cal = CalibrationSet.model_validate(yaml.safe_load((ROOT / "eval/calibration.yaml").read_text()))
    assert 25 <= len(cal.labels) <= 40 and len(cal.perturbed) >= 6
    assert (
        sum(1 for x in cal.labels if x.disputed()) >= 2
    )  # the module asks for two resolved disagreements at least
    assert all(x.b is not None for x in cal.labels)
    for d in ("groundedness", "completeness", "abstention"):
        scores = {x.final(d) for x in cal.labels} - {None}
        assert scores == {0, 1, 2}, f"{d} has no example of every score"


def test_judgement_and_reading_records_round_trip_from_dataclasses():
    j = JudgementRecord(grounded=0.1, fabricated=0.9, gap_named=0.2, model="jev", cost_usd=0.0, ms=1)
    r = ReadingRecord(
        relevant=0.5,
        reference_agrees=0.4,
        deprecation_named=1.0,
        readings_named=1.0,
        model="jev",
        cost_usd=0.0,
        ms=1,
    )
    g = Grade(id="x", scores={}, model="m", ms=1, judgement=j, reading=r)
    assert Grade.model_validate_json(g.model_dump_json()).reading.reference_agrees == 0.4
