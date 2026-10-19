import math

import mujoco
import numpy as np
import pytest
from lxml import etree

from conftest import EXAMPLES
from taskbot.build_urdf import (
    DENSITY_KG_M3,
    LINEAR_DRIVE_RADIUS_M,
    Body,
    box_inertia,
    build_urdf,
    combine,
    cylinder_inertia,
    solid_box,
)
from taskbot.schema import ArmChain, Gripper, RobotSpec, load_motor_catalog
from taskbot.view import load_model

CATALOG = load_motor_catalog()


def load(path) -> RobotSpec:
    return RobotSpec.model_validate_json(path.read_text(encoding="utf-8"))


def urdf_tree(path):
    return etree.fromstring(build_urdf(load(path)).encode())


def inertia_matrix(el) -> np.ndarray:
    a = {k: float(v) for k, v in el.attrib.items()}
    return np.array([
        [a["ixx"], a["ixy"], a["ixz"]],
        [a["ixy"], a["iyy"], a["iyz"]],
        [a["ixz"], a["iyz"], a["izz"]],
    ])


# --------------------------------------------------------------------------- formulas


def test_box_inertia_formula():
    I = box_inertia(12.0, 1.0, 2.0, 3.0)
    np.testing.assert_allclose(np.diag(I), [13.0, 10.0, 5.0])


def test_cylinder_inertia_formula():
    I = cylinder_inertia(2.0, 0.5, 3.0)
    np.testing.assert_allclose(np.diag(I), [2 / 12 * (0.75 + 9), 2 / 12 * (0.75 + 9), 0.25])


def test_box_mass_from_density():
    b = solid_box("aluminium", (0.1, 0.2, 0.3), (0, 0, 0))
    assert b.mass == pytest.approx(2700 * 0.006)


def test_combine_uses_parallel_axis_theorem():
    # Two 1 kg point masses at z = ±1: Ixx = Iyy = 2, Izz = 0 about the origin.
    zero = np.zeros((3, 3))
    total = combine([Body(1.0, np.array([0, 0, 1.0]), zero), Body(1.0, np.array([0, 0, -1.0]), zero)])
    assert total.mass == 2.0
    np.testing.assert_allclose(total.com, 0.0)
    np.testing.assert_allclose(np.diag(total.inertia), [2.0, 2.0, 0.0])


# --------------------------------------------------------------------------- URDF output


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_urdf_parses_and_forms_a_tree(path):
    root = urdf_tree(path)
    assert root.tag == "robot"
    links = [l.get("name") for l in root.iter("link")]
    joints = root.findall("joint")
    assert len(links) == len(set(links))
    assert len(joints) == len(links) - 1
    children = [j.find("child").get("link") for j in joints]
    assert len(set(children)) == len(children)
    roots = set(links) - set(children)
    assert roots == {"base"}
    for j in joints:
        assert j.find("parent").get("link") in links


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_mass_positive_and_inertia_positive_definite(path):
    root = urdf_tree(path)
    total = 0.0
    for link in root.iter("link"):
        mass = float(link.find("inertial/mass").get("value"))
        assert mass > 0, link.get("name")
        I = inertia_matrix(link.find("inertial/inertia"))
        eig = np.linalg.eigvalsh(I)
        assert eig.min() > 0, f"{link.get('name')}: eigenvalues {eig}"
        # Physical inertia must satisfy the triangle inequality (MuJoCo checks it too).
        a, b, c = sorted(eig)
        assert a + b >= c * (1 - 1e-9)
        total += mass
    assert total > 0


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_total_mass_is_parts_plus_motors(path):
    spec = load(path)
    root = urdf_tree(path)
    urdf_mass = sum(float(m.get("value")) for m in root.iter("mass"))

    motor_mass = 0.0
    for m in spec.modules:
        if isinstance(m, ArmChain):
            motor_mass += sum(CATALOG[s.joint.motor_id].mass_kg for s in m.segments)
        elif isinstance(m, Gripper) and m.motor_id:
            motor_mass += CATALOG[m.motor_id].mass_kg

    # Structure = urdf mass - motors; must be > 0 and match the arm links we can compute.
    arm_links = 0.0
    for m in spec.modules:
        if isinstance(m, ArmChain):
            for s in m.segments:
                L = s.link
                vol = (math.prod(L.size_m) if L.shape == "box"
                       else math.pi * L.radius_m ** 2 * L.length_m)
                arm_links += DENSITY_KG_M3[L.material] * vol
    base = spec.base
    base_mass = DENSITY_KG_M3[base.material] * math.prod(base.footprint_m) * base.height_m
    gripper_parts = urdf_mass - motor_mass - arm_links - base_mass
    assert gripper_parts > 0  # palm/fingers/cup add some PLA
    assert gripper_parts < 0.1


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_joint_limits_come_from_motor_catalog(path):
    spec = load(path)
    joints = {j.get("name"): j for j in urdf_tree(path).findall("joint")}
    for m in spec.modules:
        if isinstance(m, ArmChain):
            for s in m.segments:
                motor = CATALOG[s.joint.motor_id]
                el = joints[s.joint.name]
                assert el.get("type") == s.joint.type
                lim = el.find("limit")
                assert float(lim.get("effort")) == pytest.approx(motor.stall_torque_nm, rel=1e-5)
                assert float(lim.get("velocity")) == pytest.approx(motor.max_speed_rad_s, rel=1e-5)
                if s.joint.type == "continuous":
                    assert lim.get("lower") is None
                else:
                    assert float(lim.get("lower")) == pytest.approx(s.joint.lower)
                    assert float(lim.get("upper")) == pytest.approx(s.joint.upper)


def test_parallel_finger_limits(spec_dict):
    spec = RobotSpec.model_validate(spec_dict)
    root = etree.fromstring(build_urdf(spec).encode())
    finger = root.find("joint[@name='gripper_finger_left_joint']")
    motor = CATALOG[spec.modules[-1].motor_id]
    lim = finger.find("limit")
    assert finger.get("type") == "prismatic"
    assert float(lim.get("upper")) == pytest.approx(spec.modules[-1].max_opening_m / 2)
    assert float(lim.get("effort")) == pytest.approx(motor.stall_torque_nm / LINEAR_DRIVE_RADIUS_M, rel=1e-5)


def test_motor_mass_sits_on_parent_link_at_joint(spec_dict):
    # Without motors the base COM is at half height; the yaw motor at the top
    # (z = height) pulls it up.
    spec = RobotSpec.model_validate(spec_dict)
    root = etree.fromstring(build_urdf(spec).encode())
    base = root.find("link[@name='base']/inertial")
    h = spec.base.height_m
    plate = DENSITY_KG_M3["pla"] * math.prod(spec.base.footprint_m) * h
    motor = CATALOG["sts3215"].mass_kg
    assert float(base.find("mass").get("value")) == pytest.approx(plate + motor, rel=1e-5)
    com_z = float(base.find("origin").get("xyz").split()[2])
    assert com_z == pytest.approx((plate * h / 2 + motor * h) / (plate + motor), rel=1e-5)


# --------------------------------------------------------------------------- MuJoCo


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: p.stem)
def test_mujoco_loads_every_example(path):
    model = load_model(path)
    spec = load(path)
    expected = [s.joint.name for m in spec.modules if isinstance(m, ArmChain) for s in m.segments]
    if spec.modules[-1].gripper_type == "parallel":
        expected += ["gripper_finger_left_joint", "gripper_finger_right_joint"]
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
    assert names == expected
    assert model.body_mass.sum() > 0

    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)
    assert np.all(np.isfinite(data.qpos))


def test_mujoco_respects_continuous_and_revolute_limits():
    model = load_model(next(p for p in EXAMPLES if p.stem == "arm_6dof"))
    jid = lambda n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)
    assert not model.jnt_limited[jid("wrist_roll")]
    assert model.jnt_limited[jid("elbow_pitch")]
    np.testing.assert_allclose(model.jnt_range[jid("elbow_pitch")], [-2.4, 2.4])


def test_view_main_builds_urdf_before_opening_viewer(tmp_path, monkeypatch):
    from mujoco import viewer

    from taskbot import view

    opened = []
    monkeypatch.setattr(viewer, "launch", lambda model: opened.append(model))
    out = tmp_path / "arm.urdf"
    view.main([str(EXAMPLES[0]), "--urdf-out", str(out)])
    assert len(opened) == 1 and opened[0].njnt > 0
    etree.fromstring(out.read_bytes())
