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
    WheeledBase,
    base_link_name,
    gripper_names,
    load_motor_catalog,
    wheel_names,
)

DENSITY_KG_M3 = {"pla": 1240.0, "aluminium": 2700.0, "steel": 7850.0, "rubber": 1100.0}
RGBA = {"pla": "0.9 0.5 0.1 1", "aluminium": "0.75 0.75 0.78 1", "steel": "0.4 0.4 0.45 1",
        "rubber": "0.15 0.15 0.15 1"}

# A mobile chassis is a closed box with walls this thick (sheet metal / printed shell);
# arm links and fixed bases stay solid.
CHASSIS_WALL_M = 0.003
# Wheels are solid rubber cylinders; the caster is a PLA ball of half a wheel radius.
WHEEL_MATERIAL = "rubber"
CASTER_FRICTION = 0.01   # a ball caster rolls freely; nearly frictionless contact
# Rubber wheel on each terrain: sliding friction mu, and rolling-resistance coefficient Crr
# (rolling torque = Crr * r * normal force). Typical textbook ranges, not measured for any
# particular wheel: ESTIMATES, unverified.
TERRAIN = {
    "flat": {"mu": 0.8, "crr": 0.01, "note": "hard indoor floor (tile, wood, concrete)"},
    "carpet": {"mu": 0.6, "crr": 0.03, "note": "low-pile carpet"},
    "outdoor": {"mu": 0.5, "crr": 0.05, "note": "paving, packed earth, short grass"},
}
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


def tube_inertia(mass: float, outer: float, inner: float, length: float) -> np.ndarray:
    """Hollow cylinder (thick-walled tube) with its axis along z."""
    r2 = outer * outer + inner * inner
    i_xy = mass / 12.0 * (3 * r2 + length * length)
    return np.diag([i_xy, i_xy, mass * r2 / 2.0])


def hollow_tube(material: str, outer: float, wall: float, length: float, com) -> Body:
    inner = outer - wall
    m = DENSITY_KG_M3[material] * np.pi * (outer * outer - inner * inner) * length
    return Body(m, np.asarray(com, float), tube_inertia(m, outer, inner, length), material)


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
    elif link.shape == "tube":  # collision/visual shape is the outer cylinder
        body = hollow_tube(link.material, link.radius_m, link.wall, L, centre)
        geom = _cyl_geom(link.radius_m, L)
    else:
        body = solid_cylinder(link.material, link.radius_m, L, centre)
        geom = _cyl_geom(link.radius_m, L)
    return _LinkDraft(link.name, [body], [(geom, centre, RGBA[link.material])], L)


def shell_box(material: str, size, com, wall: float = CHASSIS_WALL_M) -> Body:
    """Closed box with walls of thickness `wall`: outer solid minus inner solid."""
    outer = solid_box(material, size, com)
    inner_size = [max(s - 2 * wall, 0.0) for s in size]
    inner = solid_box(material, inner_size, com)
    return Body(outer.mass - inner.mass, outer.com, outer.inertia - inner.inertia, material)


def solid_sphere(material: str, radius: float, com) -> Body:
    m = DENSITY_KG_M3[material] * 4 / 3 * np.pi * radius**3
    return Body(m, np.asarray(com, float), np.eye(3) * 0.4 * m * radius**2, material)


def _sphere_geom(radius: float) -> etree._Element:
    g = etree.Element("geometry")
    etree.SubElement(g, "sphere", radius=_fmt(radius))
    return g


def wheel_layout(b: WheeledBase) -> list[tuple[str, float, float]]:
    """(side name, x, y) of each wheel centre in the chassis frame (axles at z = radius)."""
    half_t, half_b = b.track_width_m / 2, b.wheelbase_m / 2
    if b.drive == "differential":
        return [("left", half_b, half_t), ("right", half_b, -half_t)]
    return [("front_left", half_b, half_t), ("front_right", half_b, -half_t),
            ("rear_left", -half_b, half_t), ("rear_right", -half_b, -half_t)]


def caster_position(b: WheeledBase) -> tuple[float, float, float, float]:
    """(x, y, z, radius) of the caster ball centre in the chassis frame."""
    r = 0.5 * b.wheel_radius_m
    return (-b.wheelbase_m / 2, 0.0, r, r)


def _wheeled_base_drafts(b: WheeledBase, motor: Motor) -> tuple[_LinkDraft, list[_LinkDraft], list[dict]]:
    """Chassis (with wheel motors and battery), wheels, caster, and their joints."""
    size = b.chassis_size_m
    centre = np.array([0.0, 0.0, b.chassis_bottom_m + size[2] / 2])
    chassis = _LinkDraft(b.name, [shell_box(b.chassis_material, size, centre)],
                         [(_box_geom(size), centre, RGBA[b.chassis_material])], b.chassis_top_m)
    battery = np.array([0.0, 0.0, b.chassis_bottom_m + 0.015])
    chassis.bodies.append(Body(b.battery_kg, battery, box_inertia(b.battery_kg, 0.1, 0.06, 0.03)))

    links, joints = [], []
    link_names, joint_names = wheel_names(b)
    r, w = b.wheel_radius_m, b.wheel_width_m
    mx, my, mz = motor.body_size_m
    for (side, x, y), lname, jname in zip(wheel_layout(b), link_names, joint_names):
        sign = 1.0 if y > 0 else -1.0
        # The motor sits inboard of its wheel with its long (z) axis along the axle.
        at = np.array([x, sign * (abs(y) - w / 2 - mz / 2), r])
        chassis.bodies.append(Body(motor.mass_kg, at, box_inertia(motor.mass_kg, mx, mz, my)))
        chassis.visuals.append((_box_geom((mx, mz, my)), at, MOTOR_RGBA))
        wheel = _LinkDraft(lname, [solid_cylinder(WHEEL_MATERIAL, r, w, (0, 0, 0))],
                           [(_cyl_geom(r, w), np.zeros(3), RGBA[WHEEL_MATERIAL])], 0.0)
        links.append(wheel)
        # Wheel frame rotated -90° about x: its z axis (the cylinder axis) is the chassis y axis.
        joints.append(dict(name=jname, jtype="continuous", parent=b.name, child=lname,
                           xyz=(x, y, r), rpy=(-np.pi / 2, 0, 0), axis="0 0 1",
                           limits=_joint_limits("continuous", motor)))
    if b.caster:
        cx, cy, cz, cr = caster_position(b)
        caster = _LinkDraft(link_names[-1], [solid_sphere("pla", cr, (0, 0, 0))],
                            [(_sphere_geom(cr), np.zeros(3), RGBA["pla"])], 0.0)
        links.append(caster)
        joints.append(dict(name=joint_names[-1], jtype="fixed", parent=b.name, child=caster.name,
                           xyz=(cx, cy, cz)))
    return chassis, links, joints


def _base_draft(base: FixedBase) -> _LinkDraft:
    size = (*base.footprint_m, base.height_m)
    centre = np.array([0.0, 0.0, base.height_m / 2])
    return _LinkDraft(
        base_link_name(base),
        [solid_box(base.material, size, centre)],
        [(_box_geom(size), centre, RGBA[base.material])],
        base.height_m,
    )


def _add_motor(draft: _LinkDraft, motor: Motor, at_z: float, at_xy=(0.0, 0.0)) -> None:
    centre = np.array([at_xy[0], at_xy[1], at_z])
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
    rpy=(0.0, 0.0, 0.0),
) -> None:
    j = etree.SubElement(robot, "joint", name=name, type=jtype)
    etree.SubElement(j, "parent", link=parent)
    etree.SubElement(j, "child", link=child)
    etree.SubElement(j, "origin", xyz=_fmt(xyz), rpy=_fmt(rpy))
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
    joints: list[dict] = []
    extra: list[_LinkDraft] = []
    if isinstance(spec.base, WheeledBase):
        root, extra, joints = _wheeled_base_drafts(spec.base, motors[spec.base.wheel_motor_id])
    else:
        root = _base_draft(spec.base)
    drafts: list[_LinkDraft] = [root]
    prev = root

    for module in spec.modules[1:]:
        if isinstance(module, ArmChain):
            for k, seg in enumerate(module.segments):
                j: Joint = seg.joint
                motor = motors[j.motor_id]
                xy = (module.mount_xy_m or (0.0, 0.0)) if (k == 0 and prev is root) else (0.0, 0.0)
                _add_motor(prev, motor, prev.tip_z, xy)
                child = _arm_link_draft(seg.link)
                joints.append(dict(
                    name=j.name, jtype=j.type, parent=prev.name, child=child.name,
                    xyz=(xy[0], xy[1], prev.tip_z), axis=AXIS_VEC[j.axis],
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
    return drafts + extra, joints


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
    # palm as their own bodies so their mass stays in the model, and keep visual
    # info so geoms get the URDF colours (and non-colliding scene geoms added
    # later, like the floor, aren't discarded).
    mj = etree.SubElement(robot, "mujoco")
    etree.SubElement(mj, "compiler", fusestatic="false", discardvisual="false")
    for d in drafts:
        _emit_link(robot, d)
    for j in joints:
        _emit_joint(robot, **j)
    return etree.tostring(robot, pretty_print=True, xml_declaration=True,
                          encoding="utf-8").decode("utf-8")


# --------------------------------------------------------------------------- MuJoCo model

EE_SITE = "end_effector"


def tool_link_name(spec: RobotSpec) -> str:
    """The last arm link; the gripper is welded to it."""
    return [s for m in spec.modules if isinstance(m, ArmChain) for s in m.segments][-1].link.name


def tool_offset_m(spec: RobotSpec) -> float:
    """Tool point (fingertips / cup face) along z in the last arm link's frame."""
    segments = [s for m in spec.modules if isinstance(m, ArmChain) for s in m.segments]
    gripper = next((m for m in spec.modules if isinstance(m, Gripper)), None)
    return segments[-1].link.length + tool_length_m(gripper)


def build_mjspec(spec: RobotSpec, catalog: dict[str, Motor] | None = None,
                 floating: bool = False, terrain: str = "flat"):
    """URDF loaded as an editable mujoco.MjSpec, with an `end_effector` site at the tool
    point (if there is an arm) and contacts between adjacent links excluded.

    For a wheeled base: `floating` gives the chassis a free joint (URDF has no world link,
    so a mobile robot's root link is free-floating by convention; MuJoCo needs it explicit);
    otherwise the chassis is welded in place, as for the arm-only tests. The caster becomes
    a ball joint, the wheels get the terrain's friction (see TERRAIN) with rolling
    resistance (condim 6), and each wheel joint gets a torque actuator `<joint>_motor`
    driven by the wheel speed controller.
    """
    import mujoco

    mjspec = mujoco.MjSpec.from_string(build_urdf(spec, catalog))
    # Adjacent links meet at their joint by construction. MuJoCo skips parent/child
    # contacts itself, except for bodies welded to the world (the fixed base), whose
    # contacts with the first link would otherwise jam the base joint.
    probe = mjspec.compile()
    for b in range(1, probe.nbody):
        parent = probe.body_parentid[b]
        if parent > 0:
            mjspec.add_exclude(bodyname1=probe.body(parent).name, bodyname2=probe.body(b).name)
    if spec.has_arm:
        mjspec.body(tool_link_name(spec)).add_site(
            name=EE_SITE, pos=[0.0, 0.0, tool_offset_m(spec)], size=[0.006, 0, 0],
            rgba=[1, 0.2, 0.2, 1],
        )
    if isinstance(spec.base, WheeledBase):
        _mobile_parts(mjspec, spec.base, floating, terrain)
    return mjspec


def _mobile_parts(mjspec, b: WheeledBase, floating: bool, terrain: str) -> None:
    import mujoco

    t = TERRAIN[terrain]
    if floating:
        mjspec.body(b.name).add_freejoint()
    link_names, joint_names = wheel_names(b)
    for lname, jname in zip(link_names, joint_names):
        if lname.endswith("_caster"):
            body = mjspec.body(lname)
            body.add_joint(type=mujoco.mjtJoint.mjJNT_BALL, name=f"{lname}_ball")
            for g in body.geoms:
                g.friction = [CASTER_FRICTION, 0.0, 0.0]
            continue
        for g in mjspec.body(lname).geoms:
            g.friction = [t["mu"], 0.005, t["crr"] * b.wheel_radius_m]
            g.condim = 6
        mjspec.add_actuator(name=f"{jname}_motor", target=jname,
                            trntype=mujoco.mjtTrn.mjTRN_JOINT)


def build_model(spec: RobotSpec, catalog: dict[str, Motor] | None = None):
    """Compiled mujoco.MjModel of the robot (base fixed in place), with the end-effector site."""
    return build_mjspec(spec, catalog).compile()
