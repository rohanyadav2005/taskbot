"""Task 6.4: joint ranges typed with rounded numbers, hollow tube links, the payload margin
applied once, and the feasibility gate on the third run's numbers."""

import math

import mujoco
import numpy as np
import pytest
from conftest import DESK_REQ, example_dict
from pydantic import ValidationError

from taskbot import checks
from taskbot.build_urdf import DENSITY_KG_M3, build_model, hollow_tube, tube_inertia
from taskbot.feasibility import check_feasibility, holding_torque_floor_nm, lightest_gripper
from taskbot.prompts import load_prompt
from taskbot.requirements import Requirements
from taskbot.schema import Link, RobotSpec, load_motor_catalog

CATALOG = load_motor_catalog()


def with_range(lower: float, upper: float) -> RobotSpec:
    d = example_dict("arm_4dof")
    d["modules"][1]["segments"][2]["joint"] |= {"lower": lower, "upper": upper}
    return RobotSpec.model_validate(d)


def joint_limits(spec):
    (r,) = checks.joint_limits(spec, Requirements(**DESK_REQ), build_model(spec))
    return r


# --------------------------------------------------------------------------- joint ranges


def test_zero_to_1_57_counts_as_90_degrees():
    r = joint_limits(with_range(0.0, 1.57))  # 89.95°: short only by rounding
    assert r.passed and r.limit == pytest.approx(math.pi / 2)


def test_range_just_short_gets_the_exact_value_to_write():
    r = joint_limits(with_range(0.0, 1.545))  # 88.52°
    assert not r.passed
    assert "elbow_pitch spans 88.52° (0 to 1.545 rad) but needs at least 90° (1.571 rad)" in r.message
    assert "set upper to 1.5708 or more (or lower to -0.0258 or less)" in r.message


def test_range_far_short_gets_no_exact_value():
    r = joint_limits(with_range(0.0, 1.2))
    assert not r.passed and "set upper" not in r.message


# --------------------------------------------------------------------------- tubes


def test_tube_mass_and_inertia_match_the_hollow_cylinder_formulas():
    ro, wall, L, rho = 0.015, 0.002, 0.2, DENSITY_KG_M3["aluminium"]
    ri = ro - wall
    m = rho * math.pi * (ro**2 - ri**2) * L
    b = hollow_tube("aluminium", ro, wall, L, np.zeros(3))
    assert b.mass == pytest.approx(m)
    assert np.diag(b.inertia) == pytest.approx([m * (3 * (ro**2 + ri**2) + L**2) / 12] * 2 + [m * (ro**2 + ri**2) / 2])
    # With no hole (inner radius 0) it is the solid rod.
    assert np.diag(tube_inertia(1.0, ro, 0.0, L)) == pytest.approx([(3 * ro**2 + L**2) / 12] * 2 + [ro**2 / 2])
    assert hollow_tube("aluminium", ro, ro - 1e-12, L, np.zeros(3)).mass == pytest.approx(rho * math.pi * ro**2 * L)


def test_tube_link_validation():
    t = Link(name="t", shape="tube", radius_m=0.015, length_m=0.2, material="pla")
    assert t.wall == 0.002 and t.length == 0.2
    with pytest.raises(ValidationError, match="greater than or equal to 0.001"):
        Link(name="t", shape="tube", radius_m=0.015, length_m=0.2, wall_thickness_m=0.0005, material="pla")
    with pytest.raises(ValidationError, match="must be less than radius_m"):
        Link(name="t", shape="tube", radius_m=0.002, length_m=0.2, material="pla")
    with pytest.raises(ValidationError, match="wall_thickness_m is for shape 'tube' only"):
        Link(name="t", shape="cylinder", radius_m=0.015, length_m=0.2, wall_thickness_m=0.002, material="pla")
    with pytest.raises(ValidationError, match="is a tube: give radius_m and length_m"):
        Link(name="t", shape="tube", size_m=(0.02, 0.02, 0.2), material="pla")


def tube_arm(wall: float | None = None) -> RobotSpec:
    d = example_dict("arm_4dof")
    for seg in d["modules"][1]["segments"][1:]:
        L = seg["link"]["size_m"][2]
        seg["link"] = {"name": seg["link"]["name"], "shape": "tube", "radius_m": 0.015, "length_m": L,
                       "material": "aluminium"} | ({"wall_thickness_m": wall} if wall else {})
    return RobotSpec.model_validate(d)


@pytest.mark.parametrize("wall", [None, 0.003])
def test_mujoco_model_has_the_tube_mass_and_inertia(wall):
    spec = tube_arm(wall)
    model = build_model(spec)
    # The last arm link carries no motor (the gripper's sits on its palm), so its body is the tube.
    wrist = spec.modules[1].segments[-1].link
    b = hollow_tube("aluminium", 0.015, wrist.wall, wrist.length, np.zeros(3))
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, wrist.name)
    assert model.body_mass[bid] == pytest.approx(b.mass, rel=1e-5)  # URDF numbers have 6 digits
    assert sorted(model.body_inertia[bid]) == pytest.approx(sorted(np.diag(b.inertia)), rel=1e-5)
    assert "tube r 0.015 m" in checks._link_desc(wrist)


def rod_arm() -> RobotSpec:
    """Run 3's mistake: solid aluminium rods, r 0.03 m."""
    d = example_dict("arm_4dof")
    for seg, L in zip(d["modules"][1]["segments"][1:], (0.15, 0.12, 0.04)):
        seg["link"] = {"name": seg["link"]["name"], "shape": "cylinder", "radius_m": 0.03, "length_m": L,
                       "material": "aluminium"}
    return RobotSpec.model_validate(d)


def test_static_torque_suggests_tubes_with_computed_masses():
    spec = rod_arm()
    req = Requirements(**(DESK_REQ | {"payload_kg": 0.3}))
    model = build_model(spec)
    elbow = next(r for r in checks.static_torque(spec, req, model) if r.message.startswith("elbow_pitch"))
    forearm = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "forearm")
    now = model.body_mass[forearm]
    rho = DENSITY_KG_M3["aluminium"]
    solid = rho * math.pi * 0.03**2 * 0.12
    tube = rho * math.pi * (0.03**2 - 0.028**2) * 0.12
    assert ('Use tubes (shape "tube", 2 mm wall) instead of solid links: '
            f"forearm would drop from {now:.3g} kg to {now - solid + tube:.3g} kg") in elbow.message
    assert "aluminium solid rod r 0.03 m x 0.12 m" in elbow.message


def test_tube_arm_gets_no_tube_suggestion():
    spec = tube_arm()
    req = Requirements(**(DESK_REQ | {"payload_kg": 0.3}))
    assert all("Use tubes" not in r.message for r in checks.static_torque(spec, req, build_model(spec)))


# --------------------------------------------------------------------------- prompts


def test_prompts_ask_for_tubes_and_a_single_margin():
    assert 'Desk-scale arms use hollow tubes, not solid rods or bars: shape "tube"' in load_prompt("design")
    req = load_prompt("requirements")
    assert "each exactly once" in req and "0.25 kg water + 0.05 kg cup = 0.30 kg; x1.25 = 0.38 kg" in req
    assert "Never add the margin to a number that already includes it" in req


# --------------------------------------------------------------------------- feasibility gate (run 3)

RUN3 = dict(task_summary="water the three plants on my windowsill", robot_type="arm", reach_m=0.5,
            payload_kg=0.5, workspace={"x_min": 0.15, "x_max": 0.4, "y_min": -0.2, "y_max": 0.2,
                                       "z_min": 0.05, "z_max": 0.3},
            dof_min=3, gripper_needed=True, gripper_kind="parallel", position_tolerance_m=0.01,
            tolerance_reason="1 cm is enough to water a pot", assumptions=[])


def test_half_a_kilo_at_half_a_metre_is_infeasible():
    f = check_feasibility(Requirements(**RUN3), CATALOG)
    mass, tool, _ = lightest_gripper("parallel", CATALOG)
    floor = holding_torque_floor_nm(0.5, mass, tool, 0.5)
    assert floor == pytest.approx(9.81 * (0.5 * 0.5 + mass * (0.5 - tool)))
    assert not f.ok and f"needs at least {floor:.3g} Nm" in f.problems[0] and "2.83 Nm usable" in f.problems[0]
    # The run's message used its own 0.105 kg gripper. With its mass a tool length nearer, as here,
    # that is 2.90 Nm; with it at the full 0.5 m, 2.97 Nm. Both exceed 2.83 Nm.
    assert holding_torque_floor_nm(0.5, 0.105, tool, 0.5) == pytest.approx(2.90, abs=0.005)
    assert holding_torque_floor_nm(0.5, 0.105, 0.0, 0.5) == pytest.approx(2.97, abs=0.005)


def test_corner_beyond_reach_is_rejected():
    ws = {"x_min": 0.1, "x_max": 0.2, "y_min": -0.2, "y_max": 0.2, "z_min": 0.0, "z_max": 0.6}
    with pytest.raises(ValidationError) as err:
        Requirements(**(RUN3 | {"workspace": ws, "payload_kg": 0.2}))
    msg = str(err.value)
    # 0.532 m from the most generous plausible shoulder (0.15 m up); ~0.61 m from a shoulder 0.05 m up.
    assert "workspace corner (0.2, -0.2, 0.6) is 0.532 m from the nearest plausible shoulder position" in msg
    assert "but reach_m is 0.5 m" in msg
