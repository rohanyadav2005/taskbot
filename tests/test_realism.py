"""Task 4.1: torque-speed limits, speed-planned moves, payload-mass error, stricter tracking."""

import re

import numpy as np
import pytest

from conftest import example_dict, strong_arm, with_motor
from taskbot.build_urdf import build_model
from taskbot.checks import joint_speed, static_torque
from taskbot.report import render_report
from taskbot.schema import Motor, RobotSpec, load_motor_catalog
from taskbot.sim.control import MotorController
from taskbot.sim.model import build_scene
from taskbot.sim.plan import plan_moves
from taskbot import sim_tests
from taskbot.sim_tests import (
    OVERSHOOT_FACTOR,
    SimMeasurementError,
    _MoveResult,
    carry_payload,
    hold_still,
    reach_targets,
)

CATALOG = load_motor_catalog()
NEMA_SHOULDER = with_motor(example_dict("arm_4dof"), "shoulder_pitch", "nema17_pg27")
NEMA_YAW = with_motor(example_dict("arm_4dof"), "base_yaw", "nema17_pg27")


def spec_of(d):
    return RobotSpec.model_validate(d)


# --------------------------------------------------------------------------- catalog


def test_every_motor_has_a_sourced_no_load_speed():
    for m in CATALOG.values():
        assert m.no_load_speed_rad_s >= m.max_speed_rad_s > 0
        assert "no_load_speed_rad_s" in m.sources


def test_max_speed_above_no_load_speed_rejected():
    data = CATALOG["sts3215"].model_dump() | {"max_speed_rad_s": 10.0}
    with pytest.raises(ValueError, match="exceeds no_load_speed_rad_s"):
        Motor(**data)


# --------------------------------------------------------------------------- torque-speed line


def test_controller_never_exceeds_torque_available_at_current_speed(desk_req, monkeypatch):
    """Total joint torque (applied minus MuJoCo's damping) stays within each motor's
    torque-speed model at every step, for a servo arm with a stepper shoulder."""
    spec = spec_of(NEMA_SHOULDER)
    seen = []
    original = MotorController.compute

    def spy(self, data, *a, **k):
        out = original(self, data, *a, **k)
        qd = data.qvel[self.robot.dadr]
        for m, tau, w in zip(self.robot.motors, out.tau, qd):
            cap = m.limits(tau, w)[0]
            seen.append(abs(tau) - cap)
        return out

    monkeypatch.setattr(MotorController, "compute", spy)
    carry_payload(spec, desk_req, CATALOG, payload_mass_error=0.0)
    assert max(seen) <= 1e-9


# --------------------------------------------------------------------------- planned durations / joint_speed


def test_nema17_shoulder_demo_gets_longer_moves(desk_req):
    """Regression: the geared NEMA17 (0.73 rad/s) can't do every move in 2 s."""
    spec = spec_of(NEMA_SHOULDER)
    robot = build_scene(spec, CATALOG)
    durations = [mv.duration(robot.no_load_speed) for mv in plan_moves(robot, desk_req)]
    assert min(durations) == 2.0 and max(durations) > 3.0
    # joint_speed passes (5 s is enough) and says which joint came closest
    (r,) = joint_speed(spec, desk_req, build_model(spec))
    assert r.passed and "planned move durations 2.00–3.63 s" in r.message
    assert "shoulder_pitch (nema17_pg27) needs" in r.message
    # the sim reports the longer durations it actually used
    run = reach_targets(spec, desk_req, CATALOG)
    assert "move durations 2.00–3.63 s" in run.result.message


def test_joint_speed_fails_when_a_move_needs_more_than_5_s(desk_req):
    spec = spec_of(NEMA_YAW)
    (r,) = joint_speed(spec, desk_req, build_model(spec))
    assert not r.passed and r.value > r.limit
    assert re.fullmatch(
        r"base_yaw needs 0\.59\d* rad/s to finish the move to corner \(.*\) in 5 s, but motor "
        r"nema17_pg27 gives only 0\.586 rad/s \(80% of its 0\.733 rad/s no-load speed\); "
        r"the move would take 5\.0\d s\. Use a faster motor on base_yaw, .*",
        r.message,
    )


def test_peak_speed_of_planned_moves_stays_under_80_percent(desk_req):
    robot = build_scene(spec_of(NEMA_SHOULDER), CATALOG)
    for mv in plan_moves(robot, desk_req):
        T = mv.duration(robot.no_load_speed)
        assert np.all(mv.peak_speed(T) <= 0.8 * robot.no_load_speed + 1e-9)


# --------------------------------------------------------------------------- payload mass error


def test_mass_error_is_applied_and_ideal_result_kept(strong_runs):
    run = strong_runs["hold_still"]
    assert "controller assumes 0.2 kg" in run.result.message
    assert run.ideal is not None and "controller knows the mass" in run.ideal.message
    assert run.result.value > 100 * max(run.ideal.value, 1e-6)  # the wrong mass really makes it sag
    assert re.search(r"peak deviation \d+ µm \(limit 30000 µm\), final error \d+ µm \(limit 10000 µm, "
                     r"the task tolerance\)", run.result.message)


def test_controller_believes_lighter_payload():
    robot = build_scene(spec_of(strong_arm()), CATALOG, payload_kg=0.25)
    assert MotorController(robot).believed_payload_kg == pytest.approx(0.2)
    assert MotorController(robot, payload_mass_error=0.0).believed_payload_kg == 0.25


def test_zero_drift_raises(desk_req, monkeypatch):
    def frozen(self, seconds, ref):
        return np.tile(self.robot.ee(self.data), (10, 1))  # the tool never moves

    monkeypatch.setattr(sim_tests._Sim, "run", frozen)
    with pytest.raises(SimMeasurementError, match="exactly zero deviation"):
        hold_still(spec_of(strong_arm()), desk_req, CATALOG)


# --------------------------------------------------------------------------- stricter tracking


def move(**kw):
    base = dict(label="corner", tol=0.01, ik_error=0.0, duration=2.0, final_error=0.001, overshoot=0.0,
                settle_s=1.6, peak_dev=0.005, final_contrib=np.zeros(4), peak_contrib=np.zeros(4))
    return _MoveResult(**(base | kw))


def test_overshoot_over_2x_tolerance_fails_even_if_it_settles():
    assert move().failures == []
    assert move(overshoot=OVERSHOOT_FACTOR * 0.01 + 0.001).failures == ["overshot by 21 mm (limit 20 mm)"]
    assert move(tol=0.002, final_error=0.001, overshoot=0.0045).failures == ["overshot by 4.5 mm (limit 4 mm)"]


def test_late_settling_fails_even_if_final_error_is_small():
    late = move(settle_s=3.2)  # deadline is duration + 1 s = 3.0 s
    assert late.failures == ["settled after 3.20 s, later than the 3.00 s deadline"]
    assert move(settle_s=None, final_error=0.03).failures == [
        "never settled within 10 mm (final error 30 mm)"]


def test_path_deviation_over_3x_tolerance_fails():
    assert move(peak_dev=0.031).failures == ["strayed 31 mm sideways from the planned path (limit 30 mm)"]
    assert move(peak_dev=0.029).failures == []


def test_lag_along_the_path_is_reported_not_judged():
    assert move(lag=0.5).failures == []


def test_tracking_message_keeps_settling_time(strong_runs):
    msg = strong_runs["reach_targets"].result.message
    assert re.search(r"settling [\d.]+–[\d.]+ s after move start", msg)
    assert re.search(r"max path deviation [\d.]+ mm, max overshoot [\d.]+ mm", msg)


# --------------------------------------------------------------------------- static vs sim (MG996R)


def test_mg996r_shoulder_fails_both_static_and_sim(desk_req):
    spec = spec_of(with_motor(strong_arm(), "shoulder_pitch", "mg996r"))
    shoulder = [r for r in static_torque(spec, desk_req, build_model(spec))
                if r.message.startswith("shoulder_pitch")][0]
    assert not shoulder.passed and shoulder.value > 5 * shoulder.limit
    assert not carry_payload(spec, desk_req, CATALOG, payload_mass_error=0.0).result.passed


# --------------------------------------------------------------------------- report


def test_report_shows_ideal_rows(two_round_outcome):
    report = render_report(two_round_outcome[0])
    assert re.search(r"\| hold_still \(ideal\) \| pass \(info\) \|", report)
    assert re.search(r"\| carry_payload \(ideal\) \| pass \(info\) \|", report)
    assert "20% lighter" in report
