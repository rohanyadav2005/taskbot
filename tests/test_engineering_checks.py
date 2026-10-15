import math
import re

import mujoco
import numpy as np
import pytest

from conftest import DESK_REQ, example_dict, with_motor
from taskbot import checks
from taskbot.build_urdf import build_urdf
from taskbot.checks import run_checks
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec, load_motor_catalog

CATALOG = load_motor_catalog()


def compile_(spec_dict):
    spec = RobotSpec.model_validate(spec_dict)
    return spec, mujoco.MjModel.from_xml_string(build_urdf(spec))


def req(**changes):
    return Requirements(**(DESK_REQ | changes))


def by_joint(results):
    return {r.message.split(" ")[0]: r for r in results}


# --------------------------------------------------------------------------- static_torque


def test_underpowered_spec_fails_static_torque():
    spec, model = compile_(with_motor(example_dict("arm_4dof"), "shoulder_pitch", "mg996r"))
    results = by_joint(checks.static_torque(spec, req(payload_kg=1.25), model))
    shoulder = results["shoulder_pitch"]
    assert not shoulder.passed
    assert shoulder.limit == pytest.approx(0.8 * CATALOG["mg996r"].rated_torque_nm)
    assert shoulder.value > shoulder.limit
    assert re.fullmatch(
        r"shoulder_pitch needs [\d.]+ Nm at full extension with 1\.25 kg payload; "
        r"motor mg996r is rated 0\.36 Nm \(80% limit 0\.288 Nm\)\. "
        r"Choose a motor rated above [\d.]+ Nm, or shorten/lighten .*",
        shoulder.message,
    )


def test_torque_matches_mujoco_gravity_with_arm_horizontal(desk_req):
    # With payload 0, our closed-form worst case must equal MuJoCo's gravity
    # torque (qfrc_bias at rest) once the shoulder tips the straight arm horizontal.
    spec, model = compile_(example_dict("arm_4dof"))
    ours = by_joint(checks.static_torque(spec, req(payload_kg=0.0), model))
    data = mujoco.MjData(model)
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    data.qpos[model.jnt_qposadr[jid("shoulder_pitch")]] = math.pi / 2
    mujoco.mj_forward(model, data)
    for name in ("shoulder_pitch", "elbow_pitch", "wrist_pitch"):
        mj = abs(data.qfrc_bias[model.jnt_dofadr[jid(name)]])
        assert ours[name].value == pytest.approx(mj, rel=1e-6), name


def test_payload_adds_weight_times_lever_arm(desk_req):
    spec, model = compile_(example_dict("arm_2dof"))
    empty = by_joint(checks.static_torque(spec, req(payload_kg=0.0), model))
    loaded = by_joint(checks.static_torque(spec, req(payload_kg=1.0), model))
    # Shoulder sits on top of base + turret; tool point is upper arm + palm + fingers above it.
    lever = 0.12 + 0.015 + 0.05
    extra = loaded["shoulder_pitch"].value - empty["shoulder_pitch"].value
    assert extra == pytest.approx(9.81 * 1.0 * lever, rel=1e-3)


def test_vertical_yaw_joint_carries_no_gravity_load(desk_req):
    spec, model = compile_(example_dict("arm_6dof"))
    yaw = by_joint(checks.static_torque(spec, desk_req, model))["base_yaw"]
    assert yaw.passed and yaw.value == 0.0
    assert "stays vertical" in yaw.message


# --------------------------------------------------------------------------- reach


def test_reach_passes_for_arm_that_is_long_enough(desk_req):
    spec, model = compile_(example_dict("arm_4dof"))
    reach, corners = checks.reach(spec, desk_req, model)
    assert reach.passed and reach.value <= 0.01
    assert corners.passed and corners.value == 0


def test_reach_fails_and_names_unreachable_corners():
    spec, model = compile_(example_dict("arm_2dof"))
    reach, corners = checks.reach(spec, req(reach_m=0.5), model)
    assert not reach.passed
    assert "cannot reach 0.5 m" in reach.message and "Lengthen" in reach.message
    assert not corners.passed
    assert corners.value >= 1
    assert re.search(r"\(0\.25, 0\.1, 0\.2\) misses by [\d.]+ cm", corners.message)


# --------------------------------------------------------------------------- self_collision


def folding_arm():
    d = example_dict("arm_4dof")
    segs = d["modules"][1]["segments"]
    segs[1]["joint"].update(lower=-2.8, upper=2.8)
    segs[1]["link"]["size_m"] = [0.03, 0.03, 0.06]
    segs[2]["joint"].update(lower=-3.0, upper=3.0)
    segs[2]["link"]["size_m"] = [0.03, 0.03, 0.25]
    return d


def test_self_collision_passes_for_example(desk_req):
    spec, model = compile_(example_dict("arm_4dof"))
    (r,) = checks.self_collision(spec, desk_req, model)
    assert r.passed and r.value <= 0.05


def test_self_collision_fails_when_links_fold_into_base(desk_req):
    spec, model = compile_(folding_arm())
    (r,) = checks.self_collision(spec, desk_req, model)
    assert not r.passed and r.value > 0.05
    assert "base<->forearm" in r.message  # names the offending pair
    assert "turret<->base" not in r.message and "base<->turret" not in r.message  # adjacent


# --------------------------------------------------------------------------- joint_limits


def test_narrow_joint_range_fails_unless_requirements_allow_it():
    d = example_dict("arm_4dof")
    d["modules"][1]["segments"][3]["joint"].update(lower=-0.5, upper=0.5)
    spec, model = compile_(d)
    (r,) = checks.joint_limits(spec, req(), model)
    assert not r.passed and "wrist_pitch spans 57° (needs >= 90°)" in r.message
    (r,) = checks.joint_limits(spec, req(min_joint_range_rad=0.9), model)
    assert r.passed


def test_range_beyond_two_pi_fails():
    d = example_dict("arm_4dof")
    d["modules"][1]["segments"][0]["joint"].update(lower=-7.0, upper=7.0)
    spec, model = compile_(d)
    (r,) = checks.joint_limits(spec, req(), model)
    assert not r.passed and "use type 'continuous'" in r.message


# --------------------------------------------------------------------------- budget / mass


def test_budget_fails_when_over_and_itemizes():
    spec, model = compile_(example_dict("arm_4dof"))
    (r,) = checks.budget(spec, req(max_budget_usd=300), model)
    motors = 3 * CATALOG["xm430_w350"].price_usd + 2 * CATALOG["xl430_w250"].price_usd
    assert not r.passed and r.value > motors and r.limit == 300
    assert "3x xm430_w350 $310.39" in r.message and "over by $" in r.message


def test_budget_passes_with_no_budget(desk_req):
    spec, model = compile_(example_dict("arm_2dof"))
    (r,) = checks.budget(spec, desk_req, model)
    assert r.passed and r.limit is None and "no budget given" in r.message


def test_heavy_desk_robot_warns_but_passes(desk_req):
    spec, model = compile_(example_dict("arm_6dof"))  # solid steel base
    (r,) = checks.mass(spec, desk_req, model)
    assert r.passed and r.message.startswith("WARNING:") and r.value > 5
    spec, model = compile_(example_dict("arm_2dof"))
    (r,) = checks.mass(spec, desk_req, model)
    assert r.passed and not r.message.startswith("WARNING")
    assert r.value == pytest.approx(model.body_mass.sum())


# --------------------------------------------------------------------------- run_checks


def test_run_checks_returns_every_check(desk_req):
    spec = RobotSpec.model_validate(example_dict("arm_4dof"))
    names = {r.check for r in run_checks(spec, desk_req)}
    assert names == {"static_torque", "reach", "workspace_corners", "self_collision",
                     "joint_limits", "budget", "mass"}


def test_run_checks_reports_unloadable_model(desk_req, monkeypatch):
    def broken(xml):
        raise ValueError("Error: mass and inertia of moving bodies must be larger than mjMINVAL")

    monkeypatch.setattr(mujoco.MjModel, "from_xml_string", broken)
    spec = RobotSpec.model_validate(example_dict("arm_2dof"))
    (r,) = run_checks(spec, desk_req)
    assert r.check == "build" and not r.passed and "mjMINVAL" in r.message


def test_checks_are_deterministic(desk_req):
    spec = RobotSpec.model_validate(example_dict("arm_4dof"))
    a = [r.model_dump() for r in run_checks(spec, desk_req)]
    b = [r.model_dump() for r in run_checks(spec, desk_req)]
    assert a == b
