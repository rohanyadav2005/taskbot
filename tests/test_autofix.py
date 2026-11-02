"""Autofix: deterministic fixes for failures with a known numeric fix, each in one step."""

import json

import pytest
from conftest import DESK_REQ, strong_arm, with_motor
from test_collisions_workspace_best import RUN4, low_shoulder

from taskbot import autofix, checks
from taskbot.autofix import autofix_loop, propose, structural_fix
from taskbot.build_urdf import build_model
from taskbot.checks import run_checks
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec, load_motor_catalog

CATALOG = load_motor_catalog()


def check(req):
    return lambda spec: run_checks(spec, req, CATALOG)


def one_step(d: dict, req: Requirements):
    spec = RobotSpec.model_validate(d)
    steps: list[str] = []
    new = propose(spec, req, run_checks(spec, req, CATALOG), CATALOG, steps)
    assert new is not None, "no fix proposed"
    return new, steps, {r.check: r for r in run_checks(new, req, CATALOG)}


def test_saturated_joint_gets_the_next_stronger_motor_in_one_step():
    req = Requirements(**DESK_REQ)
    before = {r.check: r for r in run_checks(RobotSpec.model_validate(
        with_motor(strong_arm(), "shoulder_pitch", "mg996r")), req, CATALOG) if not r.passed}
    assert "static_torque" in before
    new, steps, after = one_step(with_motor(strong_arm(), "shoulder_pitch", "mg996r"), req)
    assert all(r.passed for r in checks.static_torque(new, req, build_model(new)))
    assert after["torque_at_speed"].passed
    assert any(s.startswith("shoulder_pitch: mg996r → ") and "(needed " in s and "Nm, usable " in s for s in steps)


def test_short_arm_is_lengthened_by_the_shortfall_in_one_step():
    req = Requirements(**(DESK_REQ | {"reach_m": 0.42}))
    spec = RobotSpec.model_validate(strong_arm())
    assert not next(r for r in run_checks(spec, req, CATALOG) if r.check == "reach").passed
    new, steps, after = one_step(strong_arm(), req)
    assert after["reach"].passed, after["reach"].message
    assert any(s.startswith("lengthened the arm by ") for s in steps)
    assert checks.stretch_m(new) > checks.stretch_m(spec)


def test_self_collision_gets_the_least_change_verified_fix_in_one_step():
    req = Requirements(**RUN4)
    new, steps, after = one_step(low_shoulder(), req)
    assert after["self_collision"].passed
    assert any(s.startswith("self_collision: set turret length_m to at least") and "(verified: " in s for s in steps)


def test_short_joint_range_is_widened_in_one_step():
    d = strong_arm()
    d["modules"][1]["segments"][2]["joint"] |= {"lower": 0.0, "upper": 1.2}
    new, steps, after = one_step(d, Requirements(**DESK_REQ))
    assert after["joint_limits"].passed
    assert any(s.startswith("elbow_pitch: range 0 to 1.2 rad ->") for s in steps)


def test_structural_fixes():
    d = strong_arm()
    d["modules"][1]["segments"][2]["joint"] |= {"lower": 1.5, "upper": -1.5}
    d["modules"][1]["segments"][3]["link"]["name"] = "forearm"
    fixed, steps = structural_fix(d)
    j = fixed["modules"][1]["segments"][2]["joint"]
    assert (j["lower"], j["upper"]) == (-1.5, 1.5)
    assert fixed["modules"][1]["segments"][3]["link"]["name"] == "forearm_2"
    assert "elbow_pitch: swapped lower and upper (-1.5 to 1.5 rad)" in steps
    assert "renamed duplicate part forearm -> forearm_2" in steps
    RobotSpec.model_validate(fixed)


def test_loop_stops_on_an_identical_spec(monkeypatch):
    req = Requirements(**DESK_REQ)
    spec = RobotSpec.model_validate(with_motor(strong_arm(), "shoulder_pitch", "mg996r"))
    monkeypatch.setattr(autofix, "propose", lambda s, *a, **k: s)  # a 'fix' that changes nothing
    out = autofix_loop(spec, req, CATALOG, check(req))
    assert out.stop == "the fix would repeat a design already tried" and len(out.rounds) == 1


def test_loop_stops_when_the_layout_itself_cannot_reach():
    from test_chain_diagnosis import ALL_Z, WATERING

    spec = RobotSpec.model_validate(ALL_Z)
    out = autofix_loop(spec, WATERING, CATALOG, check(WATERING))
    assert not all(r.passed for r in out.results)
    assert not any("lengthened the arm" in s for s in out.steps)  # all-z needs a new intent, not length


@pytest.mark.parametrize("bad", ["shoulder_pitch", "elbow_pitch"])
def test_loop_fixes_and_logs(bad):
    req = Requirements(**DESK_REQ)
    spec = RobotSpec.model_validate(with_motor(strong_arm(), bad, "mg996r"))
    out = autofix_loop(spec, req, CATALOG, check(req))
    assert out.stop == "all checks pass", [r.message for r in out.results if not r.passed]
    assert out.steps[0].startswith(f"autofix 1: {bad}: mg996r → ")
    assert json.loads(out.spec.model_dump_json())  # a valid spec
