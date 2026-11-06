"""Acceptance: the reconstructed real-run transcripts (tests/fixtures/real_runs) replayed through
the CLI. Each finishes with a report that lists every correction code made; none crashes."""

import json
from pathlib import Path

import pytest

from taskbot.cli import main

FIXTURES = Path(__file__).parent / "fixtures" / "real_runs"
TASK = "water the three plants on my windowsill"


def replay(name, tmp_path, *extra):
    code = main(["design", TASK, "--backend", "replay", "--transcript", str(FIXTURES / f"{name}.jsonl"),
                 "--out", str(tmp_path), "--no-record", *extra])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    transcript = [json.loads(line) for line in (out / "llm_transcript.jsonl").read_text(encoding="utf-8").splitlines()]
    return code, out, run, report, transcript


def corrections_listed(report: str, transcript: list[dict], run: dict) -> None:
    section = report.split("## Corrections made by code", 1)[1].split("\n## ", 1)[0]
    for record in transcript:
        if record["error"] is None:
            for note in record["corrections"]:
                assert f"- {record['step']}: {note}" in section
    for f in run["fallbacks"]:
        assert f"- FALLBACK: {f}" in section
    assert run["corrections"] == section.count("\n- ")


def test_the_crash_now_designs_and_passes_the_sims(tmp_path):
    code, out, run, report, transcript = replay("windowsill_reach_crash", tmp_path)
    assert code == 0 and run["status"] == "PASSED"
    req = json.loads((out / "requirements.json").read_text(encoding="utf-8"))
    assert req["reach_m"] == pytest.approx(0.40)
    assert "- requirements: reach_m 0.4 m derived from the workspace" in report
    assert "intent: no usable intent" in report  # the run never got that far, so code chose
    rounds = sorted((out / "rounds").iterdir())
    checks = json.loads((rounds[-1] / "checks.json").read_text(encoding="utf-8"))
    passed = {c["check"] for c in checks if c["passed"]}
    assert {"reach_targets", "carry_payload", "hold_still", "static_torque", "reach"} <= passed
    assert all(c["passed"] for c in checks)
    corrections_listed(report, transcript, run)


def test_reach_08_z15_is_stopped_by_the_gate_with_a_reason(tmp_path):
    code, out, run, report, transcript = replay("windowsill_reach_08_z15", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE" and "needs at least" in run["reason"]
    assert "**INFEASIBLE: no design was made." in report and "## What would make it possible" in report
    assert "reach_m 1.52 m derived from the workspace" in report and "the model said 0.8 m" in report
    assert "feasibility_retry: the model gave no usable second version" in report
    corrections_listed(report, transcript, run)


def test_base_centred_workspace_is_moved_forward_and_designs(tmp_path):
    code, out, run, report, transcript = replay("windowsill_centred_on_base", tmp_path, "--no-sim")
    assert code == 0 and run["status"] == "PASSED"
    req = json.loads((out / "requirements.json").read_text(encoding="utf-8"))
    assert (req["workspace"]["x_min"], req["workspace"]["x_max"]) == (0.1, 0.4)
    assert "moved it 0.25 m forward (x now 0.1 to 0.4 m, same size)" in report
    corrections_listed(report, transcript, run)


def test_double_margin_finishes_with_what_would_work(tmp_path):
    code, out, run, report, transcript = replay("windowsill_double_margin", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert "INFEASIBLE: no catalog motor can carry the arm TaskBot sized" in report
    assert "a payload under" in report and "an arm reaching under" in report
    corrections_listed(report, transcript, run)
