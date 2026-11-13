"""Task 11: ground clearance, requirements the task's words don't support, and the sizer's sim margin."""

import json
from pathlib import Path

import pytest

from taskbot.fake import _load
from taskbot.intent import DesignIntent
from taskbot.mobile_checks import MIN_CLEARANCE_M, MIN_WHEEL_RADIUS_M, ground_clearance, min_wheel_radius_m
from taskbot.normalize import INDOOR_MAX_INCLINE_DEG, normalize_requirements
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec, load_motor_catalog
from taskbot.sizing import SizedDesign, size_design

CATALOG = load_motor_catalog()
TOYS = Path(__file__).parent / "fixtures" / "real_runs" / "smoke_2026-11-07" / "4_toys.jsonl"
TOYS_PROMPT = "a robot that picks toys off the floor and drops them in a basket"


def _base(r: float) -> RobotSpec:
    return RobotSpec.model_validate({"name": "b", "modules": [{
        "kind": "wheeled_base", "name": "base", "drive": "differential", "wheel_radius_m": r, "wheel_width_m": 0.025,
        "track_width_m": 0.275, "wheelbase_m": 0.26, "chassis_size_m": [0.3, 0.24, 0.06],
        "wheel_motor_id": "jgb37_520_110", "caster": True}]})


@pytest.mark.parametrize("terrain, r, ok", [("flat", 0.03, True), ("flat", 0.025, False), ("carpet", 0.045, False),
                                            ("carpet", 0.05, True), ("outdoor", 0.07, False), ("outdoor", 0.08, True)])
def test_ground_clearance_by_terrain(terrain, r, ok):
    req = Requirements(**(_load("mobile_requirements") | {"terrain": terrain}))
    results = ground_clearance(_base(r), req, None)
    assert [x.check for x in results] == ["ground_clearance"] * 2
    assert all(x.passed for x in results) == ok
    if not ok:
        bad = next(x for x in results if not x.passed)
        assert bad.cause == "clearance" and f"at least {min_wheel_radius_m(terrain):g} m radius" in bad.message


def test_clearance_values_are_the_documented_conventions():
    assert MIN_CLEARANCE_M == {"flat": 0.015, "carpet": 0.025, "outdoor": 0.04}
    assert MIN_WHEEL_RADIUS_M == {"flat": 0.03, "carpet": 0.04, "outdoor": 0.06}
    # The chassis bottom sits at half the wheel radius, so clearance asks for 2x it in radius.
    assert [min_wheel_radius_m(t) for t in ("flat", "carpet", "outdoor")] == [0.03, 0.05, 0.08]


@pytest.mark.parametrize("terrain", ["flat", "carpet"])
def test_the_sizer_respects_ground_clearance(terrain):
    req = Requirements(**(_load("mobile_requirements") | {"terrain": terrain}))
    intent = DesignIntent(robot_type="mobile", gripper="none", drive="differential", caster=True, rationale="r")
    sized = size_design(intent, req, CATALOG)
    assert isinstance(sized, SizedDesign)
    assert sized.spec.base.wheel_radius_m >= min_wheel_radius_m(terrain) - 1e-9


def _toys_reply() -> dict:
    return json.loads(json.loads(TOYS.read_text(encoding="utf-8").splitlines()[0])["reply"])


def test_invented_slope_and_base_load_are_removed_with_notes():
    """qwen (smoke 2026-11-07 and -12): "a flat surface, such as a carpeted floor, with a maximum
    incline of 15 degrees" and 4 kg (x1.25) on the base, for toys dropped into a basket."""
    d, notes = normalize_requirements(_toys_reply(), TOYS_PROMPT)
    assert d["max_incline_deg"] == INDOOR_MAX_INCLINE_DEG == 5.0 and d["base_payload_kg"] == 0.0
    assert any(n.startswith("max_incline_deg 15° reduced to 5°: the task names no ramp") for n in notes)
    assert any(n.startswith("base_payload_kg 5 kg set to 0: the task drops the items into a separate container")
               for n in notes)


@pytest.mark.parametrize("prompt, incline, base", [
    ("a cart that drives up the garden ramp and drops leaves in a bin", 15, 0.0),
    ("a robot that carries a 2 kg box across my office", 5.0, 5.0),
    ("a robot that picks toys off the floor and carries them in its own basket to the toy box", 5.0, 5.0),
    ("a robot that picks toys off the floor in the yard", 15, 5.0),
    ("", 15, 5.0),  # no task text to judge by: values kept
])
def test_values_the_task_supports_are_kept(prompt, incline, base):
    d, _ = normalize_requirements(_toys_reply(), prompt)
    assert (d["max_incline_deg"], d["base_payload_kg"]) == (incline, base)


def test_sizer_sims_pass_with_a_margin():
    """The sizer asks 70% of each sim test's limit (incline: 20% further), so a design that barely
    passes on one platform isn't chosen; the tests themselves keep their limits."""
    from taskbot.mobile_sim import INCLINE_M, INCLINE_SPARE, incline, turn_in_place
    from taskbot.sizing import SIZER_SIM_MARGIN

    req = Requirements(**_load("mobile_requirements"))
    spec = _base(0.04)
    plain, tight = turn_in_place(spec, req, CATALOG).result, turn_in_place(spec, req, CATALOG, margin=0.7).result
    assert plain.limit == 5.0 and tight.limit == pytest.approx(3.5) and plain.value == pytest.approx(tight.value)
    assert "the sizer's margin: 70% of the test's limits" in tight.message
    up = incline(spec, req, CATALOG, margin=SIZER_SIM_MARGIN).result
    assert f"for {INCLINE_M * (1 + INCLINE_SPARE):g} m by odometry" in up.message and up.limit == pytest.approx(0.007)
