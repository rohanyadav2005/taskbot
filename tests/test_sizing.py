"""The sizer: DesignIntent + Requirements + catalog -> RobotSpec, every number computed."""

import re

import pytest
from conftest import DESK_REQ
from sizing_cases import arm_cases, mobile_cases

from taskbot import checks
from taskbot.build_urdf import build_model
from taskbot.checks import holding_loads, run_checks
from taskbot.feasibility import check_feasibility
from taskbot.fake import _load
from taskbot.intent import DesignIntent
from taskbot.requirements import Requirements
from taskbot.schema import load_motor_catalog
from taskbot.sizing import SizedDesign, SizingFailure, size_design

CATALOG = load_motor_catalog()
LAYOUTS = ["yaw_pitch_pitch", "yaw_pitch_pitch_pitch", "yaw_pitch_pitch_wristroll"]
TASK = "water the three plants on my windowsill"


def arm_intent(req, layout="yaw_pitch_pitch_pitch") -> DesignIntent:
    return DesignIntent(robot_type="arm", arm_layout=layout, gripper=req.gripper_kind, rationale="r")


# The four real windowsill runs' requirements.
RUN2 = dict(task_summary=TASK, robot_type="arm", reach_m=0.8, payload_kg=0.5,
            workspace={"x_min": -0.2, "x_max": 0.2, "y_min": -0.2, "y_max": 0.2, "z_min": 0.1, "z_max": 1.5},
            dof_min=3, gripper_needed=True, gripper_kind="parallel", position_tolerance_m=0.01,
            tolerance_reason="1 cm", assumptions=[])
RUN3 = RUN2 | {"reach_m": 0.5, "workspace": {"x_min": 0.15, "x_max": 0.4, "y_min": -0.2, "y_max": 0.2,
                                             "z_min": 0.05, "z_max": 0.3}}
RUN4 = RUN2 | {"reach_m": 0.3, "payload_kg": 0.3,
               "workspace": {"x_min": 0.12, "x_max": 0.25, "y_min": -0.12, "y_max": 0.12, "z_min": 0.0, "z_max": 0.25}}


# --------------------------------------------------------------------------- property test

ARM_CASES = arm_cases(30, CATALOG)
MOBILE_CASES = mobile_cases(10, CATALOG)
RESULTS: dict[str, str] = {}


@pytest.mark.parametrize("i", range(30))
def test_sized_arm_passes_every_static_check_or_says_why(i):
    req = ARM_CASES[i]
    res = size_design(arm_intent(req, LAYOUTS[i % 3]), req, CATALOG)
    if isinstance(res, SizedDesign):
        failed = [r.check for r in run_checks(res.spec, req, CATALOG) if not r.passed]
        assert failed == [], f"silently failing design: {failed}"
        RESULTS[f"arm {i}"] = "sized"
    else:
        assert isinstance(res, SizingFailure) and res.reason
        if res.kind == "infeasible":
            assert "strongest catalog motor gives" in res.reason and res.options
        else:
            assert res.failures
        RESULTS[f"arm {i}"] = res.kind


@pytest.mark.parametrize("i", range(10))
def test_sized_mobile_base_passes_every_static_check_or_says_why(i):
    req = MOBILE_CASES[i]
    intent = DesignIntent(robot_type="mobile", gripper="none", drive=["differential", "four_wheel"][i % 2],
                          caster=True if i % 2 == 0 else None, rationale="r")
    res = size_design(intent, req, CATALOG)
    if isinstance(res, SizedDesign):
        assert [r.check for r in run_checks(res.spec, req, CATALOG) if not r.passed] == []
        RESULTS[f"mobile {i}"] = "sized"
    else:
        assert res.reason
        RESULTS[f"mobile {i}"] = res.kind


def test_property_pass_count(capsys):
    """Runs after the parametrized cases: prints the tally (and keeps it from regressing)."""
    if len(RESULTS) < 40:
        pytest.skip("run with the whole module")
    sized = sum(v == "sized" for v in RESULTS.values())
    with capsys.disabled():
        print(f"\nsizer property test: {sized}/40 sized and pass every static check; "
              f"{sum(v == 'infeasible' for v in RESULTS.values())} infeasible with a reason; "
              f"{sum(v == 'checks' for v in RESULTS.values())} returned as failing checks")
    assert sized >= 30


# --------------------------------------------------------------------------- golden cases


def test_run2_is_stopped_before_sizing():
    with pytest.raises(ValueError, match="1.38 m from the nearest plausible shoulder position"):
        Requirements(**RUN2)


def test_run3_is_infeasible_with_a_reason():
    req = Requirements(**RUN3)
    assert not check_feasibility(req, CATALOG).ok  # the gate stops it in the pipeline
    res = size_design(arm_intent(req), req, CATALOG)  # and the sizer agrees, with real masses
    assert isinstance(res, SizingFailure) and res.kind == "infeasible"
    assert res.reason.startswith("shoulder_pitch needs ")
    assert any("a payload under" in o for o in res.options)


def test_run4_sized_design_passes_static_checks_and_the_arm_sims():
    from taskbot.sim_tests import run_sim_tests

    req = Requirements(**RUN4)
    res = size_design(arm_intent(req), req, CATALOG)
    assert isinstance(res, SizedDesign)
    assert [r.check for r in run_checks(res.spec, req, CATALOG) if not r.passed] == []
    runs = {r.name: r.result for r in run_sim_tests(res.spec, req)}
    for name in ("reach_targets", "carry_payload", "hold_still"):
        assert runs[name].passed, runs[name].message


# --------------------------------------------------------------------------- details


def test_sizer_and_static_torque_give_identical_numbers():
    req = Requirements(**RUN4)
    res = size_design(arm_intent(req), req, CATALOG)
    model = build_model(res.spec)
    st = {r.message.split(" ")[0]: r.value for r in checks.static_torque(res.spec, req, model)}
    hl = {jl.joint.name: jl.need for jl in holding_loads(res.spec, req, model)}
    assert st == hl
    for step in res.steps:  # the sizer's logged holding torque is the check's number
        if m := re.match(r"(\w+): \w+ \(needs ([\d.e+-]+) Nm holding", step):
            assert f"{st[m[1]]:.3g}" == m[2]


def test_motors_are_picked_from_the_tip_inward_with_margin():
    req = Requirements(**DESK_REQ)
    res = size_design(arm_intent(req), req, CATALOG)
    assert isinstance(res, SizedDesign)
    joints = [s.joint for s in res.spec.modules[1].segments]
    loads = {jl.joint.name: jl.need for jl in holding_loads(res.spec, req, build_model(res.spec))}
    for j in joints:
        usable = checks.usable_torque(CATALOG[j.motor_id])
        assert usable >= loads[j.name]
    assert all(s.link.shape == "tube" for s in res.spec.modules[1].segments)
    assert res.spec.modules[1].segments[0].joint.type == "continuous"


def test_nearest_mode_relaxes_and_labels_the_design():
    req = Requirements(**RUN3)
    res = size_design(arm_intent(req), req, CATALOG, nearest=True)
    assert isinstance(res, SizedDesign) and res.approximate.startswith("payload_kg 0.5 -> ")
    assert res.requirements.payload_kg < 0.5
    assert [r.check for r in run_checks(res.spec, res.requirements, CATALOG) if not r.passed] == []
    assert any(s.startswith("APPROXIMATE: ") for s in res.steps)


def test_mobile_manipulator_and_caster_note():
    mm = Requirements(**_load("mobile_manipulator_requirements"))
    intent = DesignIntent(robot_type="mobile_manipulator", arm_layout="yaw_pitch_pitch", gripper="parallel",
                          drive="differential", caster=False, arm_mount="front", rationale="r")
    res = size_design(intent, mm, CATALOG)
    assert isinstance(res, SizedDesign)
    assert any(s.startswith("caster added") for s in res.steps)
    assert res.provenance["arm layout"] == "LLM intent" and res.provenance["upper_arm.length_m"] == "sizer"
