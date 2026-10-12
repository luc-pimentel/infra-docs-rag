import json

import pytest
from test_generate import CHUNKS, QUERY

from infra_docs_rag.embed.index import build
from infra_docs_rag.generate.judge import BAND, QUESTIONS, JevJudge, Judgement, load_judge, state
from infra_docs_rag.retrieve.retriever import Retrieval, Retriever


@pytest.fixture
def context(fake_embedder):
    embedder = fake_embedder()
    return Retriever(build(embedder, CHUNKS, ingest_version="test"), embedder).retrieve(
        QUERY, Retrieval(k=2, mode="dense")
    )


def test_the_judge_reads_the_passages_the_question_and_the_marked_answer(context):
    text = state(context, QUERY, "Use undo.[1]")
    assert text.startswith("PASSAGES\n[1] Kubernetes › Doc › Rollback\nRoll back a Deployment")
    assert "\n[2] CNCF › Doc › Layers\nRollout and release layers" in text
    assert text.endswith(f"QUESTION\n{QUERY}\n\nANSWER\nUse undo.[1]")
    assert set(QUESTIONS) == {"grounded", "fabricated", "gap_named"}
    assert all(q["type"] == "noul" and set(q["criteria"]) == {"true", "false"} for q in QUESTIONS.values())


def test_a_judgement_knows_when_the_judge_could_not_tell():
    sure = Judgement(grounded=0.93, fabricated=0.05, gap_named=0.97, model="jev-1", cost_usd=0.0, ms=1)
    assert sure.unsure == []
    unsure = Judgement(grounded=0.6, fabricated=0.41, gap_named=0.1, model="jev-1", cost_usd=0.0, ms=1)
    assert unsure.unsure == ["grounded", "fabricated"] and BAND == (0.4, 0.6)


@pytest.fixture
def fake_jev(tmp_path):
    def install(reply, returncode=0):
        script = tmp_path / "jev"
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


def test_the_jev_judge_runs_the_cli_with_its_questions_and_reads_the_probabilities(
    context, fake_jev, tmp_path
):
    script = fake_jev(
        {
            "model": "jev-1.13.0",
            "answers": {
                "grounded": {"type": "noul", "noul": 0.97},
                "fabricated": {"type": "noul", "noul": 0.05},
                "gap_named": {"type": "noul", "noul": 0.5},
            },
            "cost_usd": 0.00002,
            "latency_ms": 394,
        }
    )
    judge = JevJudge(binary=str(script))
    j = judge.judge(context, QUERY, "Use undo.[1]")
    assert (j.grounded, j.fabricated, j.gap_named) == (0.97, 0.05, 0.5)
    assert j.model == "jev-1.13.0" and j.cost_usd == 0.00002 and j.ms == 394 and j.unsure == ["gap_named"]
    args = (tmp_path / "args").read_text().splitlines()
    assert args[:3] == ["eval", "-f", str(judge.request)] and args[3:] == ["--state-file", "-", "-o", "json"]
    assert json.loads(judge.request.read_text()) == {"questions": QUESTIONS}
    assert (tmp_path / "stdin").read_text() == state(context, QUERY, "Use undo.[1]")

    with pytest.raises(ValueError, match="exited 2"):
        JevJudge(binary=str(fake_jev({}, returncode=2))).judge(context, QUERY, "x")
    with pytest.raises(ValueError, match="something unexpected"):
        JevJudge(binary=str(fake_jev({"answers": {}}))).judge(context, QUERY, "x")
    with pytest.raises(ValueError, match="not on PATH"):
        JevJudge(binary=str(tmp_path / "missing")).judge(context, QUERY, "x")
    with pytest.raises(ValueError, match="not on PATH"):
        load_judge(str(tmp_path / "missing"))
