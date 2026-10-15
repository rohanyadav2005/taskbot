"""Task 4.2: motor torque-speed model, fixed per-motor gains, hold criteria, failure causes,
and the torque_at_speed check."""

import math
import re

import pytest

from conftest import example_dict, strong_arm, with_motor
from taskbot.build_urdf import build_model
from taskbot.checks import CAUSE_ADVICE, run_checks, torque_at_speed
from taskbot.prompts import load_prompt
from taskbot.repair import format_failures
from taskbot.report import render_report
from taskbot.schema import Motor, RobotSpec, load_motor_catalog
from taskbot.sim.model import build_scene
from taskbot.sim.motor import motor_model
from taskbot.sim_tests import hold_still

CATALOG = load_motor_catalog()
NEMA_SHOULDER = with_motor(example_dict("arm_4dof"), "shoulder_pitch", "nema17_pg27")


def spec_of(d):
    return RobotSpec.model_validate(d)


# --------------------------------------------------------------------------- torque-speed model


def test_every_motor_has_sourced_stall_torque():
    for m in CATALOG.values():
        assert m.stall_torque_nm >= m.rated_torque_nm > 0 and "stall_torque_nm" in m.sources


def test_servo_driving_torque_is_min_of_rated_and_stall_line():
    m = motor_model(CATALOG["sts3215"])  # rated 0.637, stall 1.912, w0 4.4
    assert m.drive_limit(0.0) == pytest.approx(0.637)
    knee = 4.4 * (1 - 0.637 / 1.912)  # where the stall line drops below rated
    assert m.drive_limit(knee - 0.01) == pytest.approx(0.637)
    assert m.drive_limit(0.8 * 4.4) == pytest.approx(1.912 * 0.2)
    assert m.drive_limit(4.4) == 0.0 and m.drive_limit(-9.0) == 0.0


def test_braking_is_capped_at_rated_not_the_line():
    m = motor_model(CATALOG["sts3215"])
    assert m.limits(torque=-1.0, w=4.0) == (pytest.approx(0.637), False)   # braking
    cap, slowed = m.limits(torque=1.0, w=4.0)                              # driving
    assert cap == pytest.approx(1.912 * (1 - 4.0 / 4.4)) and slowed


def test_stepper_without_curve_is_flat_to_half_speed_then_linear():
    m = motor_model(CATALOG["nema17_pg27"])  # rated 3, w0 0.733
    assert m.curve is None and "approx" in m.speed_model
    assert m.drive_limit(0.5 * 0.733) == pytest.approx(3.0)
    assert m.drive_limit(0.75 * 0.733) == pytest.approx(1.5)
    assert m.drive_limit(0.8 * 0.733) == pytest.approx(1.2)
    assert m.drive_limit(0.733) == 0.0


def test_stepper_pull_out_curve_is_interpolated():
    data = CATALOG["nema17_pg27"].model_dump() | {
        "id": "curvy", "pull_out_curve": [(0.0, 3.0), (0.4, 2.0), (0.8, 0.0)]}
    m = motor_model(Motor(**data))
    assert m.speed_model == "pull-out curve"
    assert m.drive_limit(0.2) == pytest.approx(2.5)
    assert m.drive_limit(0.6) == pytest.approx(1.0)
    assert m.drive_limit(1.0) == 0.0


def test_pull_out_curve_only_for_steppers():
    data = CATALOG["sts3215"].model_dump() | {"pull_out_curve": [(0.0, 1.0)]}
    with pytest.raises(ValueError, match="only for steppers"):
        Motor(**data)


# --------------------------------------------------------------------------- fixed gains


def test_dynamixel_gains_come_from_factory_values():
    xm = motor_model(CATALOG["xm430_w350"])  # P=800 I=0 D=0, stall 4.1 N·m
    assert xm.kp == pytest.approx(800 / 128 * 4.1 / 885 * 4096 / (2 * math.pi))
    assert xm.kp == pytest.approx(18.88, abs=0.01)
    assert xm.kd == 0 and xm.ki == 0
    xl = motor_model(CATALOG["xl430_w250"])  # P=640 D=4000, stall 1.5
    assert xl.kp == pytest.approx(640 / 128 * 1.5 / 885 * 4096 / (2 * math.pi))
    assert xl.kd == pytest.approx(4000 / 16 * 1.5 / 885 * 4096 / (2 * math.pi) / 1000)
    assert "P=640 I=0 D=4000" in xl.gain_source


def test_hobby_and_stepper_use_one_conservative_setting():
    for motor_id in ("mg996r", "sts3215", "ax12a", "nema17_pg27"):
        m = motor_model(CATALOG[motor_id])
        assert m.kp == pytest.approx(CATALOG[motor_id].stall_torque_nm / math.radians(20))
        assert m.kd == 0 and m.ki == 0
        assert "stall torque at 20° error" in m.gain_source


def test_gains_do_not_depend_on_payload_or_test():
    spec = spec_of(strong_arm())
    a = build_scene(spec, CATALOG, payload_kg=0.0).motors
    b = build_scene(spec, CATALOG, payload_kg=1.0).motors
    assert [(m.kp, m.kd, m.damping) for m in a] == [(m.kp, m.kd, m.damping) for m in b]


def test_back_emf_and_d_term_are_mujoco_joint_damping():
    robot = build_scene(spec_of(strong_arm()), CATALOG)
    for m, dof in zip(robot.motors, robot.dadr):
        assert robot.model.dof_damping[dof] == pytest.approx(m.damping + m.kd)
        if m.kind != "stepper":
            assert m.damping == pytest.approx(m.stall / m.no_load_speed)


def test_report_lists_controller_settings(two_round_outcome):
    report = render_report(two_round_outcome[0])
    assert "## Controller settings (round 2 design)" in report
    assert "never tuned per test" in report
    assert re.search(r"\| shoulder_pitch \| xm540_w270 \| 48\.8 \| 0 \| 0 \| 3\.38 \| 3\.53 / 10\.6 \|", report)
    assert "Dynamixel factory gains P=640 I=0 D=4000" in report


# --------------------------------------------------------------------------- hold_still criteria


def test_hold_reports_peak_and_final_separately(strong_runs):
    msg = strong_runs["hold_still"].result.message
    peak, final = map(int, re.search(r"peak deviation (\d+) µm .* final error (\d+) µm", msg).groups())
    assert peak >= final > 1000  # fails on final error (> 1 mm) ...
    assert peak < 5000           # ... not on peak


def test_hold_passes_only_when_final_under_1mm_and_peak_under_5mm(light_req):
    result = hold_still(spec_of(strong_arm()), light_req, CATALOG).result
    assert result.passed and result.value < 0.001 and result.limit == 0.001


# --------------------------------------------------------------------------- causes


def test_repair_prompt_explains_every_cause_in_one_line():
    prompt = load_prompt("repair")
    for cause in CAUSE_ADVICE:
        lines = [l for l in prompt.splitlines() if l.strip().startswith(f"- {cause}:")]
        assert len(lines) == 1, cause
    assert "a stronger motor will not help" in prompt


def test_failures_sent_to_llm_carry_their_cause(desk_req):
    spec = spec_of(with_motor(strong_arm(), "shoulder_pitch", "mg996r"))
    fails = [r for r in run_checks(spec, desk_req) if not r.passed]
    text = format_failures(fails)
    assert "- [static_torque] (cause: saturated) shoulder_pitch needs" in text


def slow_strong_catalog():
    """A made-up slow motor: strong at rest, little torque left near its no-load speed."""
    slow = CATALOG["xm540_w270"].model_dump() | {
        "id": "slow_strong", "rated_torque_nm": 3.0, "stall_torque_nm": 4.5,
        "max_speed_rad_s": 0.5, "no_load_speed_rad_s": 0.5}
    return CATALOG | {"slow_strong": Motor(**slow)}


def test_torque_at_speed_fails_speed_limited_for_a_slow_motor(desk_req):
    catalog = slow_strong_catalog()
    spec = RobotSpec.model_validate(with_motor(example_dict("arm_4dof"), "shoulder_pitch", "slow_strong"),
                                    context={"motor_ids": set(catalog)})
    (r,) = torque_at_speed(spec, desk_req, build_model(spec, catalog), catalog)
    assert not r.passed and r.cause == "speed_limited"
    assert r.value > r.limit and r.value < 3.0  # fine at rest, short at speed
    assert re.match(r"shoulder_pitch holds [\d.]+ Nm of gravity .* motor slow_strong gives [\d.]+ Nm "
                    r"at that speed \(3 Nm rated at rest", r.message)
    assert "a faster motor (or different gearing)" in r.message


def test_torque_at_speed_fails_saturated_for_a_weak_motor(desk_req):
    spec = spec_of(with_motor(strong_arm(), "shoulder_pitch", "mg996r"))
    (r,) = torque_at_speed(spec, desk_req, build_model(spec))
    assert not r.passed and r.cause == "saturated" and "even at rest" in r.message


def test_nema17_shoulder_passes_torque_at_speed_after_stepper_fix(desk_req):
    """With the stepper model (rated to 50% speed) the NEMA17 keeps 1.2 N·m at its 80% planned
    peak, more than the 1.04 N·m of gravity at mid-move, so the static check passes it."""
    spec = spec_of(NEMA_SHOULDER)
    (r,) = torque_at_speed(spec, desk_req, build_model(spec))
    assert r.passed
    assert re.search(r"shoulder_pitch holds 1\.04 Nm .* motor nema17_pg27 gives 1\.2 Nm", r.message)
