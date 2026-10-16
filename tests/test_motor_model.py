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
    assert xl.gain_source == "Dynamixel factory gains P=640 D=4000, I=0 (pd)"


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
    assert "Dynamixel factory gains P=640 D=4000, I=0 (pd)" in report
    assert "**Control mode: pd** (factory position loops, no integral term)" in report


# --------------------------------------------------------------------------- hold_still criteria


def test_hold_reports_peak_and_final_separately(strong_runs):
    msg = strong_runs["hold_still"].result.message
    peak, final = map(int, re.search(r"peak deviation (\d+) µm .* final error (\d+) µm", msg).groups())
    assert peak >= final > 1000


def test_hold_limits_come_from_task_tolerance(strong_runs, desk_req):
    # Same physics (strong arm, 0.25 kg, ~2.15 mm steady sag), judged against two tolerances.
    loose = strong_runs["hold_still"].result  # 10 mm task
    assert loose.passed and loose.limit == 0.01 and "(limit 30000 µm)" in loose.message
    tight = hold_still(spec_of(strong_arm()), desk_req.model_copy(update={"position_tolerance_m": 0.002}),
                       CATALOG, payload_mass_error=0.2).result
    assert not tight.passed and tight.limit == 0.002 and tight.cause == "compliance"
    assert tight.value == pytest.approx(loose.value)


# --------------------------------------------------------------------------- causes


def test_repair_prompt_explains_every_cause_in_one_line():
    prompt = load_prompt("repair")
    for cause in CAUSE_ADVICE:
        lines = [l for l in prompt.splitlines() if l.strip().startswith(f"- {cause}:")]
        assert len(lines) == 1, cause
    assert 'setting "control": "pid"' in prompt and "stronger motor will not help" not in prompt


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


# --------------------------------------------------------------------------- Task 4.3: tolerance, pid, options


def test_requirements_need_a_tolerance_and_its_reason(desk_req):
    from pydantic import ValidationError

    from conftest import DESK_REQ
    from taskbot.requirements import Requirements, requirements_system_prompt

    for missing in ("position_tolerance_m", "tolerance_reason"):
        with pytest.raises(ValidationError, match=missing):
            Requirements(**{k: v for k, v in DESK_REQ.items() if k != missing})
    with pytest.raises(ValidationError):
        Requirements(**(DESK_REQ | {"position_tolerance_m": 0.0}))
    prompt = requirements_system_prompt()
    assert "use 5 mm and add that guess to assumptions" in prompt and "M3 screws ~2 mm" in prompt


def test_control_defaults_to_pd_and_only_accepts_pd_or_pid():
    from pydantic import ValidationError

    assert spec_of(strong_arm()).control == "pd"
    assert spec_of(strong_arm() | {"control": "pid"}).control == "pid"
    with pytest.raises(ValidationError):
        spec_of(strong_arm() | {"control": "lqr"})


def test_pid_is_one_fixed_preset_and_only_for_dynamixels():
    from taskbot.sim.motor import PID_PRESET_DYNAMIXEL_I

    assert PID_PRESET_DYNAMIXEL_I == 100
    for mid in ("xl430_w250", "xm430_w350", "xm540_w270"):
        pd, pid = motor_model(CATALOG[mid], "pd"), motor_model(CATALOG[mid], "pid")
        assert pd.ki == 0 and (pid.kp, pid.kd) == (pd.kp, pd.kd)
        stall = CATALOG[mid].stall_torque_nm
        assert pid.ki == pytest.approx(100 / 65536 * stall / 885 * 4096 / (2 * math.pi) / 0.001)
        assert "I=100 (pid preset, conversion unverified)" in pid.gain_source
    for mid in ("mg996r", "nema17_pg27"):
        pid = motor_model(CATALOG[mid], "pid")
        assert pid.ki == 0 and "pid not available on this motor" in pid.gain_source


def test_pid_removes_steady_sag(strong_runs, desk_req):
    pd_final = strong_runs["hold_still"].result.value
    pid = hold_still(spec_of(strong_arm() | {"control": "pid"}), desk_req, CATALOG).result
    assert pid.value < 0.5 * pd_final


def test_compliance_message_ranks_options_from_the_stiffness_breakdown(desk_req):
    tight = desk_req.model_copy(update={"position_tolerance_m": 0.002})
    msg = hold_still(spec_of(strong_arm()), tight, CATALOG).result.message
    assert "Sag by joint position-loop stiffness: wrist_pitch (xl430_w250, Kp 5.52 Nm/rad)" in msg
    options = msg.split("Options, most promising first: ")[1]
    # steady sag spread over Dynamixel joints: pid first, then a stiffer wrist motor, then geometry
    assert re.match(r'1\) control "pid" .*; 2\) a stiffer motor on wrist_pitch: .*xm430_w350 \(Kp 18\.9\)'
                    r'; 3\) shorter or lighter links .*; 4\) bringing the load closer to the base', options)
    assert "will not help" not in msg


def test_compliance_options_drop_pid_when_already_pid(desk_req):
    tight = desk_req.model_copy(update={"position_tolerance_m": 0.0005})
    result = hold_still(spec_of(strong_arm() | {"control": "pid"}), tight, CATALOG).result
    assert not result.passed and result.cause == "compliance"
    assert 'control "pid"' not in result.message.split("Options")[1]


def test_report_shows_pid_mode(two_round_outcome):
    from taskbot.repair import Round, RepairOutcome

    outcome = two_round_outcome[0]
    last = outcome.final
    pid_spec = last.spec.model_copy(update={"control": "pid"})
    report = render_report(RepairOutcome(outcome.requirements, outcome.max_rounds,
                                         [Round(last.number, pid_spec, last.results)]))
    assert "**Control mode: pid** (one fixed integral preset on Dynamixels: Position I Gain 100" in report
    assert "I=100 (pid preset, conversion unverified)" in report


# --------------------------------------------------------------------------- Task 4.4: no double counting, geometric path rule


def test_unloaded_joint_reaches_exactly_its_no_load_speed():
    """Back-EMF counted once: an unloaded joint driven flat out tops out at w0. Counted
    twice (controller and MuJoCo damping both), it would stop near w0 / 2."""
    import mujoco
    import numpy as np

    from taskbot.sim.control import MotorController, step

    robot = build_scene(spec_of(strong_arm()), CATALOG)
    data = mujoco.MjData(robot.model)
    robot.set_pose(data, robot.home())
    ctrl = MotorController(robot, payload_mass_error=0.0)
    q_ref = robot.home().copy()
    q_ref[0] = 3.0  # base yaw: no gravity load
    top = max(step(robot, data, ctrl, q_ref, np.zeros(robot.n)) and data.qvel[robot.dadr[0]]
              for _ in range(500))
    assert top == pytest.approx(robot.motors[0].no_load_speed, rel=1e-3)


def test_mujoco_applies_exactly_the_controllers_total_torque():
    import mujoco
    import numpy as np

    from taskbot.sim.control import MotorController
    from taskbot.sim.plan import min_jerk

    robot = build_scene(spec_of(strong_arm()), CATALOG)
    m = robot.model
    data = mujoco.MjData(m)
    robot.set_pose(data, robot.home())
    ctrl = MotorController(robot, payload_mass_error=0.0)
    q1 = np.array([0.0, 0.03, 1.48, 1.12])
    worst = 0.0
    for k in range(2000):
        q_ref, qd_ref = min_jerk(robot.home(), q1, 2.0, k * m.opt.timestep)
        mujoco.mj_step1(m, data)
        out = ctrl.compute(data, q_ref, qd_ref)
        applied_total = data.qfrc_applied[robot.dadr] - m.dof_damping[robot.dadr] * data.qvel[robot.dadr]
        worst = max(worst, float(np.abs(applied_total - out.tau).max()))
        mujoco.mj_step2(m, data)
    assert worst < 1e-9


def test_lag_behind_schedule_does_not_fail_path_deviation(desk_req):
    from taskbot.sim_tests import reach_targets

    five_mm = desk_req.model_copy(update={"position_tolerance_m": 0.005})
    result = reach_targets(spec_of(strong_arm()), five_mm, CATALOG).result
    dev, lag = map(float, re.search(r"max path deviation ([\d.]+) mm, .* max lag behind the planned "
                                    r"timing ([\d.]+) mm \(reported only\)", result.message).groups())
    assert result.passed
    assert dev < 15 < lag  # far behind schedule, but on the path
