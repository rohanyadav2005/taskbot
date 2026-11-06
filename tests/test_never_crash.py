"""Every run finishes with report.md and run.json, whatever the LLM writes or fails to write."""

import json

import pytest
from test_feasibility import CUP, HEAVY

from taskbot import cli, hybrid
from taskbot.fallback import FALLBACK_MARK, rule_based
from taskbot.llm import FakeClient, LLMError
from taskbot.pipeline import run_pipeline
from taskbot.schema import load_motor_catalog

CATALOG = load_motor_catalog()
TASK = "water the three plants on my windowsill"


def run(fake, tmp_path, prompt=TASK, **kw):
    return run_pipeline(fake, prompt, CATALOG, tmp_path, simulate=False, record=False, **kw)


def run_json(result):
    return json.loads((result.out_dir / "run.json").read_text(encoding="utf-8"))


def report(result):
    return (result.out_dir / "report.md").read_text(encoding="utf-8")


def test_no_parsable_reply_uses_rule_based_requirements_and_still_designs(tmp_path):
    fake = FakeClient({"Requirements": ["Sure! The robot should water plants."] * 4})
    result = run(fake, tmp_path)
    # Every check passes, but guessed requirements are never a pass: APPROXIMATE, with why.
    assert result.outcome is not None and result.outcome.final.passed and not result.passed
    assert "do less than the task asks" in result.approximate and "guessed these requirements" in result.approximate
    assert any(FALLBACK_MARK in a for a in result.requirements.assumptions)
    assert result.fallbacks[0].startswith("requirements: no reply passed validation")
    assert "rule-based requirements from keywords" in result.fallbacks[0]
    assert "- FALLBACK: requirements:" in report(result) and run_json(result)["status"] == "APPROXIMATE"
    assert "**APPROXIMATE: every check passed" in report(result)


def test_last_parsed_reply_is_used_when_it_can_be_filled(tmp_path):
    bad = {"robot_type": "arm", "task_summary": {"oops": 1}, "payload_bare_kg": 0.25,
           "workspace": {"x_min": 0.15, "x_max": 0.3, "y_min": -0.1, "y_max": 0.1, "z_min": 0.0, "z_max": 0.2}}
    result = run(FakeClient({"Requirements": [bad] * 4}), tmp_path)
    assert "the model's last parsed reply, normalised and filled with defaults" in result.fallbacks[0]
    assert result.requirements.payload_kg == pytest.approx(0.312, abs=1e-3)  # 0.25 x 1.25, to 1 g


@pytest.mark.parametrize("prompt, kind, oos", [
    ("water the three plants on my windowsill", "arm", False),
    ("a robot that carries a 2 kg box across my office", "mobile", False),
    ("a robot that picks toys off the floor and drops them in a basket", "mobile_manipulator", False),
    ("a flying drone that delivers coffee to my room", "arm", True),
])
def test_rule_based_fallback_keywords(prompt, kind, oos):
    d = rule_based(prompt)
    assert d["robot_type"] == kind and any(a.startswith("OUT OF SCOPE:") for a in d["assumptions"]) == oos


def test_unusable_intent_falls_back_to_the_default_intent(tmp_path):
    fake = FakeClient({"Requirements": [CUP], "DesignIntent": ["{not json"] * 4})
    result = run(fake, tmp_path)
    assert result.outcome is not None and result.outcome.intents[0].rationale.startswith("default intent")
    assert any(f.startswith("intent: no usable intent") for f in result.fallbacks)


def test_unusable_feasibility_retry_keeps_the_first_requirements(tmp_path):
    result = run(FakeClient({"Requirements": [HEAVY]}), tmp_path)  # nothing left for the retry
    assert result.infeasible and result.outcome is None
    assert result.fallbacks[0].startswith("feasibility_retry: the model gave no usable second version")
    assert run_json(result)["status"] == "INFEASIBLE"


def test_dead_backend_writes_a_report_and_exits_5(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    code = cli.main(["design", TASK, "--backend", "ollama", "--model", "qwen2.5:7b-instruct",
                     "--out", str(tmp_path), "--no-record"])
    err = capsys.readouterr().err
    assert code == 5 and "error: the LLM backend failed at the requirements step" in err
    assert "is Ollama running? try: ollama ps (and ollama pull qwen2.5:7b-instruct)" in err
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    assert "**LLM UNUSABLE: no design was made" in (out / "report.md").read_text(encoding="utf-8")
    assert json.loads((out / "run.json").read_text(encoding="utf-8"))["status"] == "LLM_UNUSABLE"


def test_a_sizer_bug_is_reported_not_raised(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise ZeroDivisionError("sizer bug")

    monkeypatch.setattr(hybrid, "size_design", boom)
    result = run(FakeClient({"Requirements": [CUP]}), tmp_path)
    assert result.outcome is None and "the sizer failed: ZeroDivisionError: sizer bug" in result.error
    assert run_json(result)["status"] == "DESIGN_FAILED" and (result.out_dir / "report.md").exists()


def test_llm_mode_repair_stops_when_the_backend_stops(tmp_path):
    from conftest import strong_arm, with_motor

    class Stops(FakeClient):
        def _chat(self, system, messages, schema):
            if self._step == "repair":
                raise LLMError("connection reset")
            return super()._chat(system, messages, schema)

    weak = with_motor(strong_arm(), "shoulder_pitch", "mg996r")
    result = run(Stops({"Requirements": [CUP], "RobotSpec": [weak]}), tmp_path, design_mode="llm")
    assert result.outcome.llm_failure == "error"
    assert result.outcome.stop_reason.startswith("the LLM stopped answering after round 1")


@pytest.mark.parametrize("requirements, status", [
    ([CUP], "PASSED"), ([HEAVY, HEAVY], "INFEASIBLE"),
    ([{"task_summary": "fly", "robot_type": "arm", "payload_kg": 0.1, "gripper_needed": False,
       "assumptions": ["OUT OF SCOPE: it flies"]}], "OUT_OF_SCOPE"),
])
def test_every_exit_path_writes_report_and_status(tmp_path, requirements, status):
    result = run(FakeClient({"Requirements": requirements}), tmp_path)
    assert run_json(result)["status"] == status and (result.out_dir / "report.md").exists()
    assert (result.out_dir / "llm_transcript.jsonl").exists()
