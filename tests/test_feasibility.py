"""Task 6.3: impossible requirements are caught before designing, and repair loops that go in
circles stop. Built around the second real-model run (qwen2.5:7b-instruct, "water the three
plants on my windowsill"): reach 0.8 m, payload 0.5 kg, workspace z 0.1-1.5 m."""

import json
from pathlib import Path

import pytest
from conftest import DESK_REQ, example_dict, strong_arm, with_motor
from pydantic import ValidationError

from taskbot.checks import CheckResult
from taskbot.fake import _load
from taskbot.feasibility import check_feasibility, lightest_gripper
from taskbot.llm import FakeClient
from taskbot.pipeline import run_pipeline
from taskbot.repair import (CONFLICT_NOTE, Round, false_claims, repair_loop, repair_prompt, same_design)
from taskbot.report import render_report
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec, load_motor_catalog

CATALOG = load_motor_catalog()
TASK = "water the three plants on my windowsill"

# The run's requirements, as the model wrote them.
RUN = dict(task_summary=TASK, robot_type="arm", reach_m=0.8, payload_kg=0.5,
           workspace={"x_min": -0.2, "x_max": 0.2, "y_min": -0.2, "y_max": 0.2, "z_min": 0.1, "z_max": 1.5},
           dof_min=3, gripper_needed=True, gripper_kind="parallel", position_tolerance_m=0.01,
           tolerance_reason="within 1 cm of the pot", assumptions=["a watering can holds 0.5 kg"])
# Same reach and payload with a workspace that fits the reach: gets past the validator,
# and the gate has to catch it.
HEAVY = RUN | {"workspace": RUN["workspace"] | {"z_max": 0.6}}
# What the requirements should have been: a cup of water on a windowsill.
CUP = RUN | {"reach_m": 0.4, "payload_kg": 0.3,
             "workspace": {"x_min": 0.15, "x_max": 0.3, "y_min": -0.15, "y_max": 0.15, "z_min": 0.0, "z_max": 0.3},
             "assumptions": ["a plastic cup with 250 ml of water weighs about 0.27 kg; 0.3 kg with margin"]}
HEAVY_ARM = json.loads((Path(__file__).parent / "data" / "watering_arm_heavy.json").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- validator


def test_workspace_beyond_reach_is_rejected_with_the_numbers():
    with pytest.raises(ValidationError) as err:
        Requirements(**RUN)
    msg = str(err.value)
    assert "workspace corner (-0.2, -0.2, 1.5) is 1.23 m from the nearest plausible shoulder position" in msg
    assert "on the base axis 0.3 m up" in msg and "but reach_m is 0.8 m" in msg
    assert "make reach_m and the workspace agree" in msg


def test_json_retry_sends_the_workspace_error_back():
    fake = FakeClient({"Requirements": [RUN, CUP]})
    req = fake.complete_json("sys", TASK, Requirements)
    assert req.reach_m == 0.4 and fake.usage.json_retries == 1
    assert "1.23 m from the nearest plausible shoulder position" in fake.calls[1][1][-1]["content"]


def test_mobile_manipulator_workspace_is_measured_from_the_chassis_top():
    mm = _load("mobile_manipulator_requirements")
    Requirements(**mm)  # the canned toy picker fits
    with pytest.raises(ValidationError, match="over the workspace centre"):
        Requirements(**(mm | {"reach_m": 0.1}))


# --------------------------------------------------------------------------- the gate


def test_gate_catches_the_runs_reach_and_payload():
    f = check_feasibility(Requirements(**HEAVY), CATALOG)
    assert not f.ok
    mass, _, _ = lightest_gripper("parallel", CATALOG)
    (problem,) = f.problems
    assert problem.startswith("Holding 0.5 kg at 0.8 m needs at least 4.6 Nm at the shoulder, even with "
                              f"weightless links and motors and the lightest parallel gripper ({mass:.3g} kg)")
    assert "the strongest motor, xm540_w270, gives 2.83 Nm usable (80% of 3.53 Nm rated)" in problem
    assert any("a payload under 0.274 kg" in o for o in f.options)
    assert any("a reach under 0.495 m" in o for o in f.options)


@pytest.mark.parametrize("req", [DESK_REQ, CUP, _load("mobile_requirements"), _load("mobile_manipulator_requirements")])
def test_feasible_tasks_pass_the_gate(req):
    assert check_feasibility(Requirements(**req), CATALOG).ok


def test_mobile_gate_catches_slopes_no_wheel_can_climb():
    req = Requirements(**(_load("mobile_requirements") | {"max_incline_deg": 30.0, "terrain": "outdoor"}))
    (problem,) = check_feasibility(req, CATALOG).problems
    assert "needs a friction coefficient of 0.69" in problem and "outdoor gives 0.5" in problem


def test_mobile_gate_catches_loads_no_wheel_motor_can_move():
    req = Requirements(**(_load("mobile_requirements") | {"base_payload_kg": 200.0}))
    f = check_feasibility(req, CATALOG)
    assert f.problems[0].startswith("Moving 201 kg (200 kg of payload plus four pololu_37d_50 wheel motors")
    assert "under 45.5 kg of payload could work" in f.options[0]


def _pipeline(tmp_path, requirements: list, designs: list | None = None):
    fake = FakeClient({"Requirements": requirements, "RobotSpec": designs or []})
    events = []
    result = run_pipeline(fake, TASK, CATALOG, tmp_path, max_rounds=1, simulate=False, record=False,
                          on_event=events.append)
    return result, fake, events


def test_infeasible_requirements_go_back_once_then_stop(tmp_path):
    result, fake, events = _pipeline(tmp_path, [HEAVY, HEAVY])
    assert result.outcome is None and result.infeasible and not result.rejected and not result.passed
    second_ask = fake.calls[1][1][0]["content"]
    assert second_ask.startswith(TASK) and "needs at least 4.6 Nm" in second_ask
    assert "keep them realistic for the described scene, don't just shrink them to fit" in second_ask
    assert [c[2] for c in fake.calls] == ["Requirements", "Requirements"]  # never designed
    report = (result.out_dir / "report.md").read_text(encoding="utf-8")
    assert "**INFEASIBLE: no design was made." in report
    assert "## Why no design can pass" in report and "needs at least 4.6 Nm" in report
    assert "## What would make it possible" in report and "a reach under 0.495 m" in report
    assert "sent back to the model once" in report
    assert any(e.startswith("Stopped: the requirements are still infeasible") for e in events)


def test_feasible_second_version_goes_on_to_design(tmp_path):
    result, fake, events = _pipeline(tmp_path, [HEAVY, CUP], [strong_arm()])
    assert result.outcome is not None and not result.infeasible
    assert "Feasibility: the second version can be built." in events
    assert result.requirements.reach_m == 0.4


def test_cli_exit_code_4_for_infeasible(tmp_path, monkeypatch, capsys):
    from taskbot import cli

    monkeypatch.setattr(cli, "get_client", lambda *a: FakeClient({"Requirements": [HEAVY, HEAVY]}))
    assert cli.main(["design", TASK, "--backend", "fake", "--out", str(tmp_path), "--no-record"]) == 4
    out = capsys.readouterr().out
    assert out.startswith("INFEASIBLE: no design was made for: " + TASK) and "4.6 Nm" in out


# --------------------------------------------------------------------------- loop stops


def test_loop_stops_at_once_when_no_design_can_pass(tmp_path):
    """Requirements past the gate (the loop called directly) still stop on static_torque's
    'no design change can make this check pass' instead of spending more rounds."""
    req = Requirements(**HEAVY)
    fake = FakeClient({"RobotSpec": []})  # asking for a repair would fail the test
    outcome = repair_loop(fake, req, CATALOG, initial_spec=RobotSpec.model_validate(HEAVY_ARM), simulate=False)
    assert len(outcome.rounds) == 1 and fake.calls == []
    assert outcome.stop_reason == "the requirements are beyond the motor catalog, so no design can pass"
    assert outcome.infeasible and "no design change can make this check pass" in outcome.infeasible[0]
    report = render_report(outcome)
    assert "**INFEASIBLE: the requirements are beyond the motor catalog" in report
    assert "## Why no design can pass" in report and "## What would make it possible" in report


def test_identical_design_is_pointed_out_then_stops_on_the_second_repeat(desk_req):
    weak = with_motor(example_dict("arm_4dof"), "shoulder_pitch", "mg996r")
    fake = FakeClient({"RobotSpec": [weak, weak, weak]})
    outcome = repair_loop(fake, desk_req, CATALOG, initial_spec=RobotSpec.model_validate(weak), simulate=False)
    assert len(outcome.rounds) == 3 and len(fake.calls) == 2
    assert outcome.stop_reason == "repeating designs: round 3 is the same as round 1 (2 repeats)"
    second = fake.calls[1][1][-1]["content"]
    assert "This design is the same as round 1, which failed with: static_torque (saturated)" in second
    assert "Try a different change." in second


def test_same_design_allows_one_percent():
    a = example_dict("arm_4dof")
    b = json.loads(json.dumps(a))
    b["modules"][1]["segments"][1]["link"]["size_m"][2] *= 1.009
    b["description"] = "different words"
    assert same_design(RobotSpec.model_validate(a), RobotSpec.model_validate(b))
    b["modules"][1]["segments"][1]["link"]["size_m"][2] *= 1.02
    assert not same_design(RobotSpec.model_validate(a), RobotSpec.model_validate(b))


# --------------------------------------------------------------------------- claims and conflicts


def all_ax12a(description: str, length: float = 0.15, material: str = "pla") -> RobotSpec:
    d = example_dict("arm_4dof")
    for seg in d["modules"][1]["segments"]:
        seg["joint"]["motor_id"] = "ax12a"
        seg["link"]["material"] = material
        if seg["link"]["shape"] == "box":
            seg["link"]["size_m"][2] = length
    d["modules"][2]["motor_id"] = "ax12a"
    d["description"] = description
    return RobotSpec.model_validate(d)


def test_false_motor_claim_is_called_out():
    old = all_ax12a("A watering arm.")
    new = all_ax12a("A watering arm. Increased torque by using stronger motors.")
    assert false_claims(old, new) == [
        'You said "Increased torque by using stronger motors.", but no motor changed: every joint still uses ax12a.']


def test_false_link_and_material_claims_are_called_out():
    old = all_ax12a("A watering arm.")
    new = all_ax12a("A watering arm. Lengthened the links. Switched to aluminium for stiffness.")
    claims = false_claims(old, new)
    assert 'You said "Lengthened the links.", but every link has the same shape and size as before.' in claims
    assert 'You said "Switched to aluminium for stiffness.", but every link is still pla.' in claims


def test_true_claims_and_old_sentences_are_left_alone():
    old = all_ax12a("A watering arm. Uses stronger motors.")
    assert false_claims(old, all_ax12a("A watering arm. Uses stronger motors.", length=0.15)) == []
    new = all_ax12a("A watering arm. Lengthened the links.", length=0.2)
    assert false_claims(old, new) == []


def test_false_claim_reaches_the_next_repair_message(desk_req):
    old, new = all_ax12a("A watering arm."), all_ax12a("A watering arm. Used stronger motors.")
    fails = [CheckResult(check="static_torque", passed=False, value=2.0, limit=0.4, message="m", cause="saturated")]
    rounds = [Round(1, old, fails), Round(2, new, fails)]
    prompt = repair_prompt(new, desk_req, fails, rounds)
    assert 'You said "Used stronger motors.", but no motor changed: every joint still uses ax12a.' in prompt


def test_reach_and_torque_together_get_the_trade_off_line(desk_req):
    spec = RobotSpec.model_validate(strong_arm())
    both = [CheckResult(check="reach", passed=False, message="m", cause="unreachable"),
            CheckResult(check="static_torque", passed=False, message="m", cause="saturated")]
    assert CONFLICT_NOTE in repair_prompt(spec, desk_req, both)
    assert "stronger motors near the base (shoulder, elbow) and lighter outer links" in CONFLICT_NOTE
    assert CONFLICT_NOTE not in repair_prompt(spec, desk_req, both[:1])
    assert CONFLICT_NOTE not in repair_prompt(spec, desk_req, both[1:])
