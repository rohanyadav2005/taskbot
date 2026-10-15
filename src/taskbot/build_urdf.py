"""Deterministic RobotSpec -> URDF.

All geometry, mass and inertia is computed here; the spec only says what parts
to use. Frame conventions:
  * Each link frame sits at the joint that drives it; the link body extends
    along +z and the next joint is at its tip (0, 0, length).
  * The motor for a joint is mounted on the *parent* link at that joint, so a
    joint's own motor does not load it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from lxml import etree

from taskbot.schema import (
    ArmChain,
    FixedBase,
    Gripper,
    Joint,
    Link,
    Motor,
    RobotSpec,
    base_link_name,
    gripper_names,
    load_motor_catalog,
)

DENSITY_KG_M3 = {"pla": 1240.0, "aluminium": 2700.0, "steel": 7850.0}
RGBA = {"pla": "0.9 0.5 0.1 1", "aluminium": "0.75 0.75 0.78 1", "steel": "0.4 0.4 0.45 1"}
MOTOR_RGBA = "0.1 0.1 0.1 1"

# Rack-and-pinion pitch radius assumed for prismatic joints (arm or fingers):
# force = torque / r, linear speed = angular speed * r.
LINEAR_DRIVE_RADIUS_M = 0.01

# Fixed gripper geometry (PLA). Parallel: palm spans the full opening plus both
# fingers along x. Suction: a cup whose diameter is max_opening_m.
FINGER_THICKNESS_M = 0.01
FINGER_DEPTH_M = 0.02
FINGER_LENGTH_M = 0.05
PALM_THICKNESS_M = 0.015
SUCTION_CUP_LENGTH_M = 0.03

AXIS_VEC = {"x": "1 0 0", "y": "0 1 0", "z": "0 0 1"}


# --------------------------------------------------------------------------- mass properties


@dataclass(frozen=True)
class Body:
    """A solid with its centre of mass and inertia about that COM, in link frame."""

    mass: float
    com: np.ndarray
    inertia: np.ndarray  # 3x3
    material: str | None = None  # None for motors


def box_inertia(mass: float, x: float, y: float, z: float) -> np.ndarray:
    return mass / 12.0 * np.diag([y * y + z * z, x * x + z * z, x * x + y * y])


def cylinder_inertia(mass: float, radius: float, length: float) -> np.ndarray:
    """Solid cylinder with its axis along z."""
    i_xy = mass / 12.0 * (3 * radius * radius + length * length)
    return np.diag([i_xy, i_xy, mass * radius * radius / 2.0])


def solid_box(material: str, size, com) -> Body:
    x, y, z = size
    m = DENSITY_KG_M3[material] * x * y * z
    return Body(m, np.asarray(com, float), box_inertia(m, x, y, z), material)


def solid_cylinder(material: str, radius: float, length: float, com) -> Body:
    m = DENSITY_KG_M3[material] * np.pi * radius * radius * length
    return Body(m, np.asarray(com, float), cylinder_inertia(m, radius, length), material)


def motor_body(motor: Motor, com) -> Body:
    """Motor housing treated as a uniform box with the catalog mass."""
    return Body(motor.mass_kg, np.asarray(com, float), box_inertia(motor.mass_kg, *motor.body_size_m))


def combine(bodies: list[Body]) -> Body:
    """Total mass, COM and inertia about the combined COM (parallel-axis theorem)."""
    mass = sum(b.mass for b in bodies)
    com = sum(b.mass * b.com for b in bodies) / mass
    inertia = np.zeros((3, 3))
    for b in bodies:
        d = b.com - com
        inertia += b.inertia + b.mass * (d @ d * np.eye(3) - np.outer(d, d))
    return Body(mass, com, inertia)


# --------------------------------------------------------------------------- URDF assembly


@dataclass
class _LinkDraft:
    name: str
    bodies: list[Body]
    visuals: list[tuple[etree._Element, np.ndarray, str]]  # (geometry, origin xyz, rgba)
    tip_z: float


def _fmt(v) -> str:
    return " ".join(f"{float(x):.6g}" for x in np.atleast_1d(v))


def _box_geom(size) -> etree._Element:
    g = etree.Element("geometry")
    etree.SubElement(g, "box", size=_fmt(size))
    return g


def _cyl_geom(radius: float, length: float) -> etree._Element:
    g = etree.Element("geometry")
    etree.SubElement(g, "cylinder", radius=_fmt(radius), length=_fmt(length))
    return g


def _arm_link_draft(link: Link) -> _LinkDraft:
    L = link.length
    centre = np.array([0.0, 0.0, L / 2])
    if link.shape == "box":
        body = solid_box(link.material, link.size_m, centre)
        geom = _box_geom(link.size_m)
    else:
        body = solid_cylinder(link.material, link.radius_m, L, centre)
        geom = _cyl_geom(link.radius_m, L)
    return _LinkDraft(link.name, [body], [(geom, centre, RGBA[link.material])], L)


def _base_draft(base: FixedBase) -> _LinkDraft:
    size = (*base.footprint_m, base.height_m)
    centre = np.array([0.0, 0.0, base.height_m / 2])
    return _LinkDraft(
        base_link_name(base),
        [solid_box(base.material, size, centre)],
        [(_box_geom(size), centre, RGBA[base.material])],
        base.height_m,
    )


def _add_motor(draft: _LinkDraft, motor: Motor, at_z: float) -> None:
    centre = np.array([0.0, 0.0, at_z])
    draft.bodies.append(motor_body(motor, centre))
    draft.visuals.append((_box_geom(motor.body_size_m), centre, MOTOR_RGBA))


def _joint_limits(joint_type: str, motor: Motor) -> tuple[float, float]:
    """(effort, velocity) for a URDF <limit>, from the motor's peak capability."""
    if joint_type == "prismatic":
        return (
            motor.stall_torque_nm / LINEAR_DRIVE_RADIUS_M,
            motor.max_speed_rad_s * LINEAR_DRIVE_RADIUS_M,
        )
    return motor.stall_torque_nm, motor.max_speed_rad_s


def _emit_link(robot: etree._Element, draft: _LinkDraft) -> None:
    link = etree.SubElement(robot, "link", name=draft.name)
    total = combine(draft.bodies)
    inertial = etree.SubElement(link, "inertial")
    etree.SubElement(inertial, "origin", xyz=_fmt(total.com), rpy="0 0 0")
    etree.SubElement(inertial, "mass", value=_fmt(total.mass))
    I = total.inertia
    etree.SubElement(
        inertial,
        "inertia",
        ixx=_fmt(I[0, 0]), ixy=_fmt(I[0, 1]), ixz=_fmt(I[0, 2]),
        iyy=_fmt(I[1, 1]), iyz=_fmt(I[1, 2]), izz=_fmt(I[2, 2]),
    )
    for tag in ("visual", "collision"):
        for geom, xyz, rgba in draft.visuals:
            el = etree.SubElement(link, tag)
            etree.SubElement(el, "origin", xyz=_fmt(xyz), rpy="0 0 0")
            el.append(_copy(geom))
            if tag == "visual":
                mat = etree.SubElement(el, "material", name=f"rgba_{rgba.replace(' ', '_')}")
                etree.SubElement(mat, "color", rgba=rgba)


def _copy(el: etree._Element) -> etree._Element:
    return etree.fromstring(etree.tostring(el))


def _emit_joint(
    robot: etree._Element,
    name: str,
    jtype: str,
    parent: str,
    child: str,
    xyz,
    axis: str | None = None,
    limits: tuple[float, float] | None = None,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    j = etree.SubElement(robot, "joint", name=name, type=jtype)
    etree.SubElement(j, "parent", link=parent)
    etree.SubElement(j, "child", link=child)
    etree.SubElement(j, "origin", xyz=_fmt(xyz), rpy="0 0 0")
    if axis is not None:
        etree.SubElement(j, "axis", xyz=axis)
    if limits is not None:
        attrs = {"effort": _fmt(limits[0]), "velocity": _fmt(limits[1])}
        if lower is not None:
            attrs |= {"lower": _fmt(lower), "upper": _fmt(upper)}
        etree.SubElement(j, "limit", **attrs)


def _gripper_drafts(g: Gripper, motor: Motor | None) -> list[_LinkDraft]:
    palm_name, *finger_names = gripper_names(g)[0]
    if g.gripper_type == "suction":
        r = g.max_opening_m / 2
        centre = np.array([0.0, 0.0, SUCTION_CUP_LENGTH_M / 2])
        palm = _LinkDraft(
            palm_name,
            [solid_cylinder("pla", r, SUCTION_CUP_LENGTH_M, centre)],
            [(_cyl_geom(r, SUCTION_CUP_LENGTH_M), centre, RGBA["pla"])],
            SUCTION_CUP_LENGTH_M,
        )
        if motor is not None:
            _add_motor(palm, motor, SUCTION_CUP_LENGTH_M / 2)
        return [palm]

    palm_size = (g.max_opening_m + 2 * FINGER_THICKNESS_M, FINGER_DEPTH_M, PALM_THICKNESS_M)
    centre = np.array([0.0, 0.0, PALM_THICKNESS_M / 2])
    palm = _LinkDraft(
        palm_name,
        [solid_box("pla", palm_size, centre)],
        [(_box_geom(palm_size), centre, RGBA["pla"])],
        PALM_THICKNESS_M,
    )
    _add_motor(palm, motor, PALM_THICKNESS_M / 2)
    finger_size = (FINGER_THICKNESS_M, FINGER_DEPTH_M, FINGER_LENGTH_M)
    fingers = []
    for name, side in zip(finger_names, (-1, 1)):
        c = np.array([side * FINGER_THICKNESS_M / 2, 0.0, FINGER_LENGTH_M / 2])
        fingers.append(
            _LinkDraft(name, [solid_box("pla", finger_size, c)],
                       [(_box_geom(finger_size), c, RGBA["pla"])], FINGER_LENGTH_M)
        )
    return [palm, *fingers]


def tool_length_m(gripper: Gripper | None) -> float:
    """Distance from the last arm link's tip to the tool point (fingertips / cup face)."""
    if gripper is None:
        return 0.0
    if gripper.gripper_type == "suction":
        return SUCTION_CUP_LENGTH_M
    return PALM_THICKNESS_M + FINGER_LENGTH_M


def _assemble(spec: RobotSpec, motors: dict[str, Motor]) -> tuple[list[_LinkDraft], list[dict]]:
    drafts: list[_LinkDraft] = [_base_draft(spec.base)]
    joints: list[dict] = []
    prev = drafts[0]

    for module in spec.modules[1:]:
        if isinstance(module, ArmChain):
            for seg in module.segments:
                j: Joint = seg.joint
                motor = motors[j.motor_id]
                _add_motor(prev, motor, prev.tip_z)
                child = _arm_link_draft(seg.link)
                joints.append(dict(
                    name=j.name, jtype=j.type, parent=prev.name, child=child.name,
                    xyz=(0, 0, prev.tip_z), axis=AXIS_VEC[j.axis],
                    limits=_joint_limits(j.type, motor), lower=j.lower, upper=j.upper,
                ))
                drafts.append(child)
                prev = child
        elif isinstance(module, Gripper):
            motor = motors[module.motor_id] if module.motor_id else None
            palm, *fingers = _gripper_drafts(module, motor)
            mount, *finger_joints = gripper_names(module)[1]
            joints.append(dict(name=mount, jtype="fixed", parent=prev.name,
                               child=palm.name, xyz=(0, 0, prev.tip_z)))
            for jname, finger, sign in zip(finger_joints, fingers, ("-1 0 0", "1 0 0")):
                joints.append(dict(
                    name=jname, jtype="prismatic", parent=palm.name, child=finger.name,
                    xyz=(0, 0, palm.tip_z), axis=sign,
                    limits=_joint_limits("prismatic", motor),
                    lower=0.0, upper=module.max_opening_m / 2,
                ))
            drafts += [palm, *fingers]
            prev = palm
        else:  # pragma: no cover - schema allows only one base, at index 0
            raise AssertionError(f"unexpected module {module.kind}")
    return drafts, joints


def structure_mass_by_material(
    spec: RobotSpec, catalog: dict[str, Motor] | None = None
) -> dict[str, float]:
    """kg of each material in the base, links and gripper parts (motors excluded)."""
    drafts, _ = _assemble(spec, catalog if catalog is not None else load_motor_catalog())
    totals: dict[str, float] = {}
    for d in drafts:
        for b in d.bodies:
            if b.material is not None:
                totals[b.material] = totals.get(b.material, 0.0) + b.mass
    return totals


def build_urdf(spec: RobotSpec, catalog: dict[str, Motor] | None = None) -> str:
    """Return the URDF XML for a validated RobotSpec."""
    drafts, joints = _assemble(spec, catalog if catalog is not None else load_motor_catalog())
    robot = etree.Element("robot", name=spec.name)
    # MuJoCo-only hint (other URDF tools ignore it): keep the fixed base and the
    # palm as their own bodies, so their mass stays in the model and contacts
    # between them and their neighbours are filtered as parent/child pairs.
    mj = etree.SubElement(robot, "mujoco")
    etree.SubElement(mj, "compiler", fusestatic="false", discardvisual="true")
    for d in drafts:
        _emit_link(robot, d)
    for j in joints:
        _emit_joint(robot, **j)
    return etree.tostring(robot, pretty_print=True, xml_declaration=True,
                          encoding="utf-8").decode("utf-8")
