import re

import mujoco
import numpy as np
import pytest

from conftest import example_dict, strong_arm, with_motor
from taskbot.build_urdf import EE_SITE, build_model
from taskbot.llm import FakeClient
from taskbot.repair import repair_loop
from taskbot.report import render_report, write_outputs
from taskbot.schema import RobotSpec, load_motor_catalog
from taskbot.sim import control
from taskbot.sim.ik import solve_ik
from taskbot.sim.model import build_scene
from taskbot.sim.record import render_gif
from taskbot.sim_tests import (
    carry_payload,
    hold_still,
    min_jerk,
    reach_targets,
    workspace_targets,
)

CATALOG = load_motor_catalog()


def spec_of(d):
    return RobotSpec.model_validate(d)


# --------------------------------------------------------------------------- model / IK


def test_build_model_has_end_effector_site_at_tool_point():
    spec = spec_of(example_dict("arm_2dof"))
    model = build_model(spec)
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE)
    # straight up: base 0.04 + turret 0.04 + upper arm 0.12 + palm 0.015 + fingers 0.05
    np.testing.assert_allclose(data.site_xpos[site], [0, 0, 0.265], atol=1e-9)


def test_ik_reaches_workspace_targets(desk_req):
    robot = build_scene(spec_of(example_dict("arm_4dof")))
    for label, target in workspace_targets(desk_req):
        res = solve_ik(robot, target, robot.home())
        assert res.converged and res.error_m < 1e-3, label
        assert np.all(res.q >= robot.lower) and np.all(res.q <= robot.upper)


def test_ik_respects_joint_limits_even_when_target_needs_more():
    d = example_dict("arm_4dof")
    for seg in d["modules"][1]["segments"][1:]:
        seg["joint"].update(lower=-0.3, upper=0.3)  # pitch joints can barely bend
    robot = build_scene(spec_of(d))
    res = solve_ik(robot, np.array([0.3, 0.0, 0.05]), robot.home())
    assert not res.converged and res.error_m > 0.05
    assert np.all(res.q >= robot.lower - 1e-12) and np.all(res.q <= robot.upper + 1e-12)


def test_min_jerk_starts_and_ends_at_rest():
    q0, q1 = np.array([0.0, 1.0]), np.array([1.0, -1.0])
    p, v = min_jerk(q0, q1, 2.0, 0.0)
    np.testing.assert_allclose(p, q0) and np.testing.assert_allclose(v, 0)
    p, v = min_jerk(q0, q1, 2.0, 2.0)
    np.testing.assert_allclose(p, q1) and np.testing.assert_allclose(v, 0)
    p, _ = min_jerk(q0, q1, 2.0, 1.0)
    np.testing.assert_allclose(p, (q0 + q1) / 2)


# --------------------------------------------------------------------------- control


def test_total_torque_never_above_rated_and_weak_motor_hits_it(desk_req, monkeypatch):
    """Above rated only when a joint is overspun past w0 (1 + rated/stall), where back-EMF
    alone brakes harder than rated (see taskbot.sim.motor)."""
    spec = spec_of(with_motor(strong_arm(), "shoulder_pitch", "mg996r"))
    robot = build_scene(spec, CATALOG, payload_kg=desk_req.payload_kg)
    rated = [CATALOG[j.motor_id].rated_torque_nm for j in robot.joints]
    np.testing.assert_allclose(robot.torque_limit, rated)

    seen = []
    original = control.MotorController.compute

    def spy(self, data, *args, **kwargs):
        out = original(self, data, *args, **kwargs)
        qd = np.abs(data.qvel[self.robot.dadr])
        overspun = np.array([w > m.no_load_speed * (1 + m.rated / m.stall)
                             for m, w in zip(self.robot.motors, qd)])
        seen.append(np.where(overspun, 0.0, np.abs(out.tau) / self.robot.torque_limit))
        return out

    monkeypatch.setattr(control.MotorController, "compute", spy)
    run = hold_still(spec, desk_req, CATALOG)
    ratios = np.array(seen)
    assert ratios.max() <= 1.0 + 1e-9  # never above rated
    assert ratios[:, 1].max() == pytest.approx(1.0)  # the weak shoulder does hit its limit
    assert not run.result.passed and run.result.cause == "saturated"


# --------------------------------------------------------------------------- the three tests


def test_strong_arm_passes_all_three_with_light_payload(two_round_outcome):
    runs = {r.name: r for r in two_round_outcome[0].rounds[-1].sim_runs}  # 0.1 kg payload
    assert set(runs) == {"reach_targets", "carry_payload", "hold_still"}
    for name, run in runs.items():
        assert run.result.passed, run.result.message
        assert run.result.check == name and run.result.cause is None
    assert runs["reach_targets"].result.value < 0.01
    assert runs["carry_payload"].result.value <= 0.2  # longest saturation, s
    assert runs["hold_still"].result.value < 0.001    # final error, m
    assert "9/9 targets met the 10 mm task tolerance" in runs["carry_payload"].result.message


def test_strong_arm_with_desk_payload_sags_about_2mm_within_10mm_tolerance(strong_runs):
    # 0.25 kg, controller believes 0.2 kg: factory Dynamixel gains have no I term, so the
    # missing feedforward torque becomes a steady droop of (missing torque / Kp), ~2 mm here,
    # within the desk task's 10 mm tolerance.
    assert all(r.result.passed for r in strong_runs.values())
    hold = strong_runs["hold_still"]
    assert 0.001 < hold.result.value < 0.005
    assert hold.ideal.value < 1e-5  # with the true mass there is no sag


def test_payload_stays_welded_to_gripper(strong_runs):
    run = strong_runs["carry_payload"]
    model, robot = run.robot.model, run.robot
    data = mujoco.MjData(model)
    data.qpos[:] = run.frames[-1]
    mujoco.mj_kinematics(model, data)
    gap = np.linalg.norm(data.xpos[robot.payload_body] - data.site_xpos[robot.site])
    assert gap < 1e-3


def test_weak_shoulder_fails_carry_payload_naming_the_joint(desk_req):
    spec = spec_of(with_motor(strong_arm(), "shoulder_pitch", "mg996r"))
    result = carry_payload(spec, desk_req, CATALOG, payload_mass_error=0.0).result
    assert not result.passed
    assert result.value > 0.2 and result.limit == 0.2
    assert re.search(r"shoulder_pitch saturated for [\d.]+ s while moving to .* \(motor mg996r: "
                     r"0\.36 Nm rated, 7\.48 rad/s no-load; speed model min\(rated, stall", result.message)
    assert "elbow_pitch saturated" not in result.message
    assert result.cause == "saturated" and result.message.endswith(
        "Cause: saturated: torque limit hit; use a stronger motor, or shorter or lighter outer links.")


def test_unreachable_targets_fail_reach_targets(desk_req):
    spec = spec_of(example_dict("arm_2dof"))
    result = reach_targets(spec, desk_req, CATALOG).result
    assert not result.passed and "unreachable: IK gets no closer than" in result.message


# --------------------------------------------------------------------------- repair loop + report


def test_sim_tests_run_only_after_static_checks_pass(two_round_outcome):
    outcome, _ = two_round_outcome
    r1, r2 = outcome.rounds
    assert [f.check for f in r1.failures] == ["static_torque", "torque_at_speed"] and not r1.sim_runs
    assert {r.check for r in r2.results} >= {"reach_targets", "carry_payload", "hold_still"}
    assert r2.passed and outcome.passed

    report = render_report(outcome)
    assert "| carry_payload | not run | pass |" in report


def test_sim_failure_is_sent_back_for_repair(desk_req, monkeypatch):
    import taskbot.repair as repair

    def carry_payload(spec, req, catalog):
        from taskbot.checks import CheckResult
        from taskbot.sim_tests import SimRun

        ok = spec.name == "fixed"
        res = CheckResult(check="carry_payload", passed=ok, value=0.0 if ok else 1.0, limit=0.2,
                          message="ok" if ok else "shoulder_pitch saturated at its rated 1.37 Nm")
        return SimRun("carry_payload", res, None, np.zeros((1, 1)))

    monkeypatch.setattr(repair, "sim_tests_for", lambda spec, req: [carry_payload])
    fixed = strong_arm() | {"name": "fixed"}
    fake = FakeClient({"RobotSpec": [strong_arm(), fixed]})
    outcome = repair_loop(fake, desk_req, CATALOG)
    assert outcome.passed and len(outcome.rounds) == 2
    assert "- [carry_payload] shoulder_pitch saturated" in fake.calls[1][1][0]["content"]


def test_recording_writes_320x240_gif_at_20fps(strong_runs, tmp_path):
    from PIL import Image

    run = strong_runs["reach_targets"]
    path = tmp_path / "reach_targets.gif"
    frames = run.frames[20:30]  # 10 frames of motion = 0.5 s
    problem = render_gif(run.robot.model, frames, path)
    if problem:
        pytest.skip(problem)
    with Image.open(path) as im:
        assert im.size == (320, 240)
        total_ms = 0
        for i in range(im.n_frames):  # Pillow may merge identical frames; time is kept
            im.seek(i)
            total_ms += im.info["duration"]
    assert total_ms == 10 * 50  # 20 fps


def test_frames_are_recorded_at_20fps(strong_runs):
    run = strong_runs["hold_still"]
    assert len(run.frames) == 5 * 20 + 1  # 5 s hold at 20 fps, plus the initial frame


def test_write_outputs_saves_gifs_and_links_them(two_round_outcome, tmp_path, monkeypatch):
    import taskbot.report as report

    monkeypatch.setattr(report, "render_gif", lambda model, frames, path, **kw: path.parent.mkdir(
        parents=True, exist_ok=True) or path.write_bytes(b"GIF89a") and None)
    outcome, _ = two_round_outcome
    out = write_outputs(outcome, tmp_path, CATALOG)
    for name in ("reach_targets", "carry_payload", "hold_still"):
        assert (out / "sim" / f"{name}.gif").exists()
    text = (out / "report.md").read_text()
    assert "## Simulation recordings (round 2)" in text
    assert (out / "rounds" / "round_2" / "sim_ideal.json").exists()
    assert "![carry_payload](sim/carry_payload.gif)" in text
