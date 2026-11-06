"""Every run records its LLM calls in llm_transcript.jsonl; --backend replay plays one back."""

import json
from pathlib import Path

import pytest
from openai_mock import MockOpenAI

from taskbot.fake import fake_client
from taskbot.llm import LLMError, LLMOutputError, OpenAICompatClient, ReplayClient
from taskbot.pipeline import run_pipeline
from taskbot.requirements import Requirements, requirements_system_prompt
from taskbot.schema import load_motor_catalog, supported_motors

CATALOG = load_motor_catalog()
FIXTURES = Path(__file__).parent / "fixtures" / "real_runs"
KEYS = {"step", "attempt", "schema", "system_sha256", "user", "reply", "error", "provider", "model", "time_s"}


def fake_run(tmp_path, prompt="a robot that carries a 2 kg box across my office"):
    return run_pipeline(fake_client(supported_motors(CATALOG)), prompt, CATALOG, tmp_path, simulate=False,
                        record=False)


def transcript(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "llm_transcript.jsonl").read_text(encoding="utf-8").splitlines()]


def test_every_run_writes_its_transcript(tmp_path):
    result = fake_run(tmp_path)
    records = transcript(result.out_dir)
    assert [r["step"] for r in records] == ["requirements", "intent"]
    for r in records:
        assert set(r) == KEYS and r["reply"] and r["error"] is None and r["model"] == "scripted"
    assert records[0]["user"] == "a robot that carries a 2 kg box across my office"


def test_failed_attempts_are_recorded_with_their_error():
    from taskbot.llm import FakeClient

    fake = FakeClient({"Requirements": ["not json", json.dumps({"task_summary": "x"})] * 2})
    with pytest.raises(LLMOutputError):
        fake.complete_json("sys", "task", Requirements, step="requirements")
    assert [r["attempt"] for r in fake.transcript] == [1, 2, 3, 4]
    assert fake.transcript[0]["error"].startswith("- reply contains no JSON object")
    assert "Field required" in fake.transcript[1]["error"]


def test_transcript_never_holds_the_api_key():
    key = "sk-SECRET-replay-1"
    with MockOpenAI([{"status": 401, "body": {"error": "bad key Bearer " + key}}]) as m:
        c = OpenAICompatClient(model="m", base_url=m.url, api_key=key, rate_limit_retries=0)
        with pytest.raises(LLMError):
            c.complete_json("sys", "u", Requirements, step="requirements")
    assert key not in json.dumps(c.transcript) and c.transcript[0]["reply"] is None


def test_replay_returns_each_steps_replies_in_order(tmp_path):
    out = fake_run(tmp_path / "a").out_dir
    replay = ReplayClient(out / "llm_transcript.jsonl")
    again = run_pipeline(replay, "a robot that carries a 2 kg box across my office", CATALOG, tmp_path / "b",
                         simulate=False, record=False)
    assert again.passed and [r["step"] for r in replay.transcript] == ["requirements", "intent"]
    with pytest.raises(LLMError, match="no more replies for step 'intent'"):
        replay.complete_json("sys", "u", Requirements, step="intent")


def test_the_real_crash_reproduces_offline():
    """Without the normaliser, the recorded requirements fail exactly as in the real run."""
    replay = ReplayClient(FIXTURES / "windowsill_reach_crash.jsonl")
    with pytest.raises(LLMOutputError) as err:
        replay.complete_json(requirements_system_prompt(), "water the three plants on my windowsill",
                             Requirements, step="requirements")
    assert "workspace corner (0.3, -0.15, 0.275) is 0.358 m from the nearest plausible shoulder position" in str(err.value)
    assert "but reach_m is 0.3 m" in str(err.value)


def test_cli_replay_backend(tmp_path, capsys):
    from taskbot.cli import main

    out = fake_run(tmp_path / "a").out_dir
    code = main(["design", "a robot that carries a 2 kg box across my office", "--backend", "replay",
                 "--transcript", str(out / "llm_transcript.jsonl"), "--out", str(tmp_path / "b"),
                 "--no-sim", "--no-record"])
    assert code == 0 and "PASSED" in capsys.readouterr().out
    assert main(["design", "x", "--backend", "replay", "--out", str(tmp_path / "c")]) == 1
    assert "--backend replay needs --transcript" in capsys.readouterr().err


def test_bench_replay_backend(tmp_path):
    from taskbot.bench import RunMeta, load_suite, make_client_factory, run_bench

    (p,) = [q for q in load_suite(Path(__file__).parent.parent / "bench", "main") if q.id == "mobile_office_box"]
    out = fake_run(tmp_path / "a", p.prompt).out_dir
    folder = tmp_path / "transcripts"
    folder.mkdir()
    (folder / f"{p.id}.jsonl").write_text((out / "llm_transcript.jsonl").read_text(encoding="utf-8"), encoding="utf-8")
    factory = make_client_factory("replay", None, 0, 0.0, folder)
    (rec,) = run_bench([p], tmp_path / "run", factory, RunMeta("replay", "recorded", "replay", "r"), simulate=False)
    assert rec.status == "done" and rec.passed


def test_fixtures_are_valid_transcripts():
    for f in FIXTURES.glob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            assert set(json.loads(line)) == KEYS
