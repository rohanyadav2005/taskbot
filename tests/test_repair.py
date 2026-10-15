import json

import pytest

from conftest import example_dict, with_motor
from taskbot.llm import FakeClient
from taskbot.repair import repair_loop, repair_prompt
from taskbot.report import render_report, write_outputs
from taskbot.schema import RobotSpec, load_motor_catalog

CATALOG = load_motor_catalog()
UNDERPOWERED = with_motor(example_dict("arm_4dof"), "shoulder_pitch", "mg996r")
UPGRADED = with_motor(UNDERPOWERED, "shoulder_pitch", "xm540_w270") | {
    "description": UNDERPOWERED["description"] + " Shoulder upgraded to xm540_w270 for torque."
}


def test_motor_upgrade_makes_loop_pass_in_round_2(desk_req):
    fake = FakeClient({"RobotSpec": [UNDERPOWERED, UPGRADED]})
    outcome = repair_loop(fake, desk_req, CATALOG, simulate=False)

    assert outcome.passed
    assert [r.number for r in outcome.rounds] == [1, 2]
    r1, r2 = outcome.rounds
    assert [f.check for f in r1.failures] == ["static_torque"]
    assert "shoulder_pitch needs" in r1.failures[0].message
    assert r2.passed
    assert r1.spec.model_dump() == RobotSpec.model_validate(UNDERPOWERED).model_dump()
    assert r2.spec.model_dump() == RobotSpec.model_validate(UPGRADED).model_dump()
    assert len(fake.calls) == 2  # one design, one repair


def test_repair_request_holds_only_failures_spec_and_requirements(desk_req):
    fake = FakeClient({"RobotSpec": [UNDERPOWERED, UPGRADED]})
    outcome = repair_loop(fake, desk_req, CATALOG, simulate=False)
    system, messages, schema = fake.calls[1]
    prompt = messages[0]["content"]

    assert schema == "RobotSpec"
    assert '"xm540_w270"' in system  # the catalog is still available
    assert prompt.startswith("Your robot design failed engineering checks. Fix it.")
    for r in outcome.rounds[0].results:
        assert (r.message in prompt) == (not r.passed), r.check
    assert "{failures}" not in prompt and "{spec}" not in prompt and "{requirements}" not in prompt
    spec_json = prompt.split("Current spec: ", 1)[1].split("\nRequirements: ", 1)[0]
    assert RobotSpec.model_validate_json(spec_json).name == "arm_4dof"
    assert desk_req.model_dump_json() in prompt


def test_loop_passing_first_time_makes_no_repair_call(desk_req):
    fake = FakeClient({"RobotSpec": [UPGRADED]})
    outcome = repair_loop(fake, desk_req, CATALOG, simulate=False)
    assert outcome.passed and len(outcome.rounds) == 1 and len(fake.calls) == 1


def test_loop_gives_up_after_5_rounds_and_report_says_so(desk_req, tmp_path):
    fake = FakeClient({"RobotSpec": [UNDERPOWERED] * 5})  # the "repair" never changes anything
    outcome = repair_loop(fake, desk_req, CATALOG, simulate=False)
    assert not outcome.passed
    assert len(outcome.rounds) == 5 and len(fake.calls) == 5
    report = render_report(outcome)
    assert "**FAILED: no design passed all checks; it did not pass after 5 rounds" in report
    assert "Do not build it as is." in report
    assert "PASSED" not in report
    assert report.count("The spec did not change.") == 4
    assert "Still failing:" in report and "shoulder_pitch needs" in report


def test_llm_failure_during_repair_stops_loop_honestly(desk_req):
    fake = FakeClient({"RobotSpec": [UNDERPOWERED] + [{"broken": True}] * 4})
    outcome = repair_loop(fake, desk_req, CATALOG, simulate=False)
    assert not outcome.passed and len(outcome.rounds) == 1
    assert "repair request after round 1 failed" in outcome.stop_reason
    assert "the loop stopped early" in render_report(outcome)


def test_write_outputs_keeps_every_round_and_diffs_them(desk_req, tmp_path):
    fake = FakeClient({"RobotSpec": [UNDERPOWERED, UPGRADED]})
    outcome = repair_loop(fake, desk_req, CATALOG, simulate=False)
    out = write_outputs(outcome, tmp_path, CATALOG)

    assert out == tmp_path / "arm_4dof"
    for name in ("requirements.json", "spec.json", "robot.urdf", "report.md"):
        assert (out / name).exists(), name
    for n in (1, 2):
        rnd = out / "rounds" / f"round_{n}"
        RobotSpec.model_validate_json((rnd / "spec.json").read_text())
        results = json.loads((rnd / "checks.json").read_text())
        assert {"check", "passed", "value", "limit", "message"} == set(results[0])
    assert RobotSpec.model_validate_json((out / "spec.json").read_text()).model_dump() == \
        outcome.final.spec.model_dump()

    report = (out / "report.md").read_text()
    assert report.startswith("# Design report: arm_4dof\n\n**PASSED: every check passed in round 2 of 5.**")
    assert '-            "motor_id": "mg996r"' in report
    assert '+            "motor_id": "xm540_w270"' in report
    assert "| static_torque | FAIL (1/4) | pass |" in report
    assert "## Requirements" in report and "| Payload | 0.25 kg |" in report


def test_repair_prompt_lists_each_failure_on_its_own_line(desk_req):
    from taskbot.checks import CheckResult

    spec = RobotSpec.model_validate(UNDERPOWERED)
    fails = [CheckResult(check="reach", passed=False, message="too short"),
             CheckResult(check="budget", passed=False, message="too dear")]
    prompt = repair_prompt(spec, desk_req, fails)
    assert "Failed checks:\n- [reach] too short\n- [budget] too dear\nRules:" in prompt


def test_cli_fake_backend_repairs_and_writes_report(tmp_path, capsys):
    from taskbot import cli

    assert cli.main(["design", "water my plants", "--backend", "fake", "--out", str(tmp_path),
                     "--no-record"]) == 0
    out = capsys.readouterr().out
    assert "Rounds: R1 1 fail  R2 pass" in out and "PASSED" in out
    assert (tmp_path / "fake_arm_4dof" / "report.md").exists()


def test_cli_exit_code_2_when_design_never_passes(tmp_path, monkeypatch):
    from taskbot import cli

    monkeypatch.setattr(cli, "TORQUE_SAFETY_FRACTION", 1e-6)  # fake repair finds no motor strong enough
    assert cli.main(["design", "x", "--backend", "fake", "--out", str(tmp_path),
                     "--max-rounds", "2", "--no-record"]) == 2
    assert "FAILED" in (tmp_path / "fake_arm_4dof" / "report.md").read_text()
