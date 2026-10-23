"""Messages for the second real-model run's failures: a correct yaw/pitch arm whose links were
far too heavy and long, whose requirements no catalog motor can meet, and whose shoulder hit
an oversized base. The design is rebuilt from the run summary in tests/data/."""

import json
from pathlib import Path

import pytest

from taskbot import checks
from taskbot.build_urdf import build_model
from taskbot.checks import CheckResult
from taskbot.prompts import load_prompt
from taskbot.repair import progress_lines
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec

HEAVY = json.loads((Path(__file__).parent / "data" / "watering_arm_heavy.json").read_text(encoding="utf-8"))

# The run's requirements: 0.5 kg (a 0.4 kg watering can + 25%) at 0.5 m.
RUN_REQ = dict(
    task_summary="water the three plants on my windowsill", robot_type="arm", reach_m=0.5, payload_kg=0.5,
    workspace={"x_min": -0.2, "x_max": 0.2, "y_min": -0.2, "y_max": 0.2, "z_min": 0.1, "z_max": 0.6},
    dof_min=3, gripper_needed=True, gripper_kind="parallel", position_tolerance_m=0.01,
    tolerance_reason="1 cm is enough to water a plant", assumptions=[])


def run(check, req=None, d=HEAVY):
    spec = RobotSpec.model_validate(d)
    return check(spec, Requirements(**(req or RUN_REQ)), build_model(spec))


def by_joint(results):
    return {r.message.split(" ", 1)[0]: r for r in results}


def test_static_torque_names_the_heavy_parts():
    shoulder = by_joint(run(checks.static_torque))["shoulder_pitch"]
    # The run's requirements are beyond the catalog, so the shoulder's cause is "infeasible".
    assert not shoulder.passed and shoulder.cause == "infeasible"
    assert ("Biggest loads on shoulder_pitch: wrist_link (aluminium box 0.08 x 0.08 x 0.2 m, with the motor "
            "mounted on it) 3.46 kg -> ") in shoulder.message
    assert "elbow_link (aluminium box 0.08 x 0.08 x 0.3 m" in shoulder.message


def test_no_catalog_motor_is_strong_enough_instead_of_an_impossible_motor():
    for r in run(checks.static_torque):
        if not r.passed:
            assert "No catalog motor is strong enough: the strongest, xm540_w270, is rated 3.53 Nm (2.83 Nm usable)" \
                in r.message
            assert "Choose a motor rated above" not in r.message


def test_over_long_arm_is_pointed_out():
    shoulder = by_joint(run(checks.static_torque))["shoulder_pitch"]
    assert ("The arm stretches 0.865 m but the task needs 0.5 m: shortening the links by about 0.36 m "
            "cuts this torque") in shoulder.message


def test_requirements_beyond_the_catalog_are_said_once_at_the_shoulder():
    results = by_joint(run(checks.static_torque))
    shoulder = results["shoulder_pitch"]
    # The floor uses the lightest catalog gripper of the required kind, the same number as the gate.
    assert ("Even with weightless links and motors, holding 0.5 kg with the lightest parallel gripper (0.0943 kg) "
            "0.5 m out needs 2.85 Nm here, more than any catalog motor allows (2.83 Nm)") in shoulder.message
    assert "no design change can make this check pass" in shoulder.message
    assert "beyond the motor catalog" not in results["elbow_pitch"].message


def test_feasible_requirements_get_no_infeasibility_note():
    req = RUN_REQ | {"reach_m": 0.3, "payload_kg": 0.15,
                     "workspace": {"x_min": 0.1, "x_max": 0.25, "y_min": -0.1, "y_max": 0.1,
                                   "z_min": 0.05, "z_max": 0.2}}
    shoulder = by_joint(run(checks.static_torque, req))["shoulder_pitch"]
    assert not shoulder.passed  # still far too heavy
    assert "beyond the motor catalog" not in shoulder.message
    assert "No catalog motor is strong enough" in shoulder.message


def test_torque_at_speed_says_no_catalog_motor_would_hold_it():
    (r,) = run(checks.torque_at_speed)
    assert not r.passed and r.cause == "saturated"
    assert "no catalog motor would: the strongest, xm540_w270, is rated 3.53 Nm" in r.message
    assert "use a stronger motor" not in r.message


def test_self_collision_with_the_base_explains_why():
    (r,) = run(checks.self_collision)
    assert not r.passed and "Base<->shoulder_link" in r.message
    assert ("shoulder_link hits the base 'Base' when shoulder_pitch swings it down: shoulder_pitch sits only "
            "0.05 m above the base top, and the base is 0.3 x 0.3 m") in r.message


def test_joint_range_just_under_90_degrees_shows_the_decimal():
    d = json.loads(json.dumps(HEAVY))
    d["modules"][1]["segments"][2]["joint"] |= {"lower": -0.785, "upper": 0.785}
    (r,) = run(checks.joint_limits, d=d)
    assert not r.passed
    assert "elbow_pitch spans 89.95° (-0.785 to 0.785 rad) but needs at least 90° (1.571 rad)" in r.message


@pytest.mark.parametrize("check, before, now, text", [
    ("self_collision", 0.427, 0.494, "share of colliding poses went from 42.7% to 49.4%: your last change made this worse"),
    ("joint_limits", 1.57, 1.57, "narrowest joint range went from 90.0° to 90.0°: your last change didn't affect this"),
    ("torque_at_speed", 26.21, 26.24, "went from 26.2 Nm to 26.2 Nm: your last change didn't affect this"),
    ("static_torque", 72.4, 72.6, "torque needed went from 72.4 Nm to 72.6 Nm: your last change didn't affect this"),
    ("static_torque", 11.8, 29.8, "torque needed went from 11.8 Nm to 29.8 Nm: your last change made this worse"),
])
def test_progress_labels_and_small_changes(check, before, now, text):
    limits = {"self_collision": 0.05, "joint_limits": 1.5708, "torque_at_speed": 0.5, "static_torque": 0.4}

    def r(v):
        return [CheckResult(check=check, passed=False, value=v, limit=limits[check], message="m")]
    (line,) = progress_lines(r(before), r(now))
    assert text in line


def test_design_prompt_asks_for_a_light_short_arm():
    design = load_prompt("design")
    assert "Keep the arm light and no longer than the task needs" in design
    assert "fixed_base of about 0.15 x 0.15 m" in design
