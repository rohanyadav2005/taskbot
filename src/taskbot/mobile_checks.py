"""Static checks for wheeled bases: wheel_torque, top_speed, tip_over, traction, ground_clearance.

Same contract as the arm checks: each takes (spec, requirements, model, catalog) and
returns a list of CheckResults with a cause on failure. `model` is the robot with its base
fixed at the origin (build_model). Loads: the robot, plus payload_kg at the tool point
(if there is an arm) and base_payload_kg on the chassis top centre.

Mobility assumptions, used by all four checks:
    ACCEL_M_S2      0.5 m/s² acceleration while climbing max_incline_deg
    rolling resistance and friction from build_urdf.TERRAIN (estimates)
    torque shared equally between driven wheels
    a differential robot climbs forwards, drive wheels in front, caster behind
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np

from taskbot.build_urdf import TERRAIN, caster_position, wheel_layout
from taskbot.checks import CheckResult
from taskbot.schema import ArmChain, Motor, RobotSpec, WheeledBase, load_motor_catalog
from taskbot.sim.motor import motor_model

if TYPE_CHECKING:
    from taskbot.requirements import Requirements

ACCEL_M_S2 = 0.5
SPEED_FRACTION = 0.8      # top_speed: use at most 80% of the wheel motor's no-load speed
TIP_MARGIN = 0.15         # tip_over: shrink the support polygon by 15% about its centroid
G = 9.81
DIRECTIONS = [(f"{int(a)}°", np.radians(a)) for a in range(0, 360, 45)]


def _catalog(catalog):
    return catalog if catalog is not None else load_motor_catalog()


# --------------------------------------------------------------------------- mass model


@dataclass
class MassModel:
    """Base part (chassis, wheels, caster, base payload) and arm part (arm links, gripper,
    gripper payload) with the arm extended along +x from its mount."""

    base_mass: float
    base_com: np.ndarray
    arm_mass: float
    arm_com: np.ndarray     # arm extended along +x (rotate about mount for other directions)
    arm_com_stowed: np.ndarray
    mount: np.ndarray       # first arm joint position (x, y)
    reach: float            # horizontal distance mount -> tool point when extended

    @property
    def total(self) -> float:
        return self.base_mass + self.arm_mass

    def com(self, angle: float | None = None) -> np.ndarray:
        """System COM with the arm extended towards `angle` (rad, about the mount), or
        stowed (straight up) when angle is None."""
        if angle is None or self.arm_mass == 0:
            arm = self.arm_com_stowed
        else:
            c, s = np.cos(angle), np.sin(angle)
            d = self.arm_com[:2] - self.mount
            arm = np.array([*(self.mount + [c * d[0] - s * d[1], s * d[0] + c * d[1]]), self.arm_com[2]])
        return (self.base_mass * self.base_com + self.arm_mass * arm) / self.total


def _subtree(model, root: int) -> list[int]:
    out = []
    for b in range(1, model.nbody):
        a = b
        while a > 0 and a != root:
            a = model.body_parentid[a]
        if a == root:
            out.append(b)
    return out


def mass_model(spec: RobotSpec, req: Requirements, model: mujoco.MjModel) -> MassModel:
    base: WheeledBase = spec.base
    data = mujoco.MjData(model)
    base_payload = req.base_payload_kg or 0.0
    top = np.array([0.0, 0.0, base.chassis_top_m])
    arm_root = None
    first_arm = next((m for m in spec.modules if isinstance(m, ArmChain)), None)
    if first_arm is not None:
        arm_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, first_arm.segments[0].link.name)
    arm_bodies = set(_subtree(model, arm_root)) if arm_root is not None else set()

    def masses(q: dict[int, float]) -> tuple[float, np.ndarray, float, np.ndarray, np.ndarray]:
        data.qpos[:] = model.qpos0
        for jid, v in q.items():
            data.qpos[model.jnt_qposadr[jid]] = v
        mujoco.mj_kinematics(model, data)
        bm = sum(model.body_mass[b] for b in range(1, model.nbody) if b not in arm_bodies)
        bc = sum(model.body_mass[b] * data.xipos[b] for b in range(1, model.nbody) if b not in arm_bodies)
        bm_total = bm + base_payload
        b_com = (bc + base_payload * top) / bm_total
        am = sum(model.body_mass[b] for b in arm_bodies)
        ac = sum((model.body_mass[b] * data.xipos[b] for b in arm_bodies), np.zeros(3))
        tool = np.zeros(3)
        if first_arm is not None:
            tool = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "end_effector")].copy()
            am += req.payload_kg
            ac = ac + req.payload_kg * tool
        a_com = ac / am if am > 0 else np.zeros(3)
        return bm_total, b_com, am, a_com, tool

    bm, bc, am, ac_stowed, _ = masses({})
    mount = np.array(first_arm.mount_xy_m or (0.0, 0.0)) if first_arm else np.zeros(2)
    ac, reach = ac_stowed, 0.0
    if first_arm is not None:
        # Extended: the first joint that tilts the arm (axis x or y) set to ±90° (within its
        # limits), others at zero; keep the sign that reaches further horizontally.
        best = None
        for seg in first_arm.segments:
            if seg.joint.axis in ("x", "y") and seg.joint.type != "prismatic":
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, seg.joint.name)
                for target in (np.pi / 2, -np.pi / 2):
                    lo = seg.joint.lower if seg.joint.lower is not None else -np.inf
                    hi = seg.joint.upper if seg.joint.upper is not None else np.inf
                    _, _, _, com, tool = masses({jid: float(np.clip(target, lo, hi))})
                    horizontal = np.linalg.norm(tool[:2] - mount)
                    if best is None or horizontal > best[0]:
                        best = (horizontal, com, tool)
                break
        if best is not None:
            reach, ac, tool = best
            # rotate so the extension points along +x from the mount
            d = tool[:2] - mount
            ang = -np.arctan2(d[1], d[0])
            c, s = np.cos(ang), np.sin(ang)
            v = ac[:2] - mount
            ac = np.array([*(mount + [c * v[0] - s * v[1], s * v[0] + c * v[1]]), ac[2]])
    return MassModel(bm, bc, am, ac, ac_stowed, mount, float(reach))


def support_polygon(base: WheeledBase) -> np.ndarray:
    """Ground contact points (x, y), counter-clockwise."""
    pts = [(x, y) for _, x, y in wheel_layout(base)]
    if base.caster:
        cx, cy, _, _ = caster_position(base)
        pts.append((cx, cy))
    pts = np.array(pts, float)
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]


def _margin(point: np.ndarray, poly: np.ndarray) -> float:
    """Signed distance from point to the polygon's edges: positive inside."""
    d = np.inf
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        edge = b - a
        normal = np.array([-edge[1], edge[0]]) / np.linalg.norm(edge)  # inward for CCW
        d = min(d, float(np.dot(point - a, normal)))
    return d


# --------------------------------------------------------------------------- checks


def _climb_force(mm: MassModel, req: Requirements) -> tuple[float, float]:
    """(total traction force needed, incline in rad)."""
    theta = np.radians(req.max_incline_deg)
    crr = TERRAIN[req.terrain]["crr"]
    m = mm.total
    return m * (G * np.sin(theta) + crr * G * np.cos(theta) + ACCEL_M_S2), theta


def wheel_torque(spec: RobotSpec, req: Requirements, model: mujoco.MjModel,
                 catalog: dict[str, Motor] | None = None) -> list[CheckResult]:
    """Torque per driven wheel to accelerate at 0.5 m/s² up max_incline_deg with the full load,
    against what the wheel motor gives at the target speed."""
    base: WheeledBase = spec.base
    motor = motor_model(_catalog(catalog)[base.wheel_motor_id], mode="velocity")
    mm = mass_model(spec, req, model)
    force, _ = _climb_force(mm, req)
    need = force * base.wheel_radius_m / base.driven_wheels
    w = req.speed_m_s / base.wheel_radius_m
    available = motor.drive_limit(w)
    ok = need <= available
    msg = (f"each of {base.driven_wheels} driven wheels needs {need:.3g} Nm to accelerate "
           f"{mm.total:.3g} kg at {ACCEL_M_S2} m/s² up {req.max_incline_deg:g}° on {req.terrain} "
           f"(rolling resistance {TERRAIN[req.terrain]['crr']}); motor {base.wheel_motor_id} gives "
           f"{available:.3g} Nm at {req.speed_m_s:g} m/s ({w:.3g} rad/s with {base.wheel_radius_m:g} m "
           f"wheels; {motor.rated:.3g} Nm rated at rest).")
    cause = None
    if not ok:
        if need > motor.rated:
            cause = "saturated"
            msg += (" Even at rest it is short: use a stronger wheel motor, smaller wheels, more "
                    "driven wheels (four_wheel), or a lighter robot.")
        else:
            cause = "speed_limited"
            msg += (" It has the torque at rest but not at this speed: use a faster wheel motor, or "
                    "larger wheels only if the motor has torque to spare.")
    return [CheckResult(check="wheel_torque", passed=ok, value=need, limit=available,
                        message=msg, cause=cause)]


def top_speed(spec: RobotSpec, req: Requirements, model: mujoco.MjModel,
              catalog: dict[str, Motor] | None = None) -> list[CheckResult]:
    base: WheeledBase = spec.base
    motor = _catalog(catalog)[base.wheel_motor_id]
    top = SPEED_FRACTION * motor.no_load_speed_rad_s * base.wheel_radius_m
    ok = top > req.speed_m_s
    msg = (f"top speed at {SPEED_FRACTION:.0%} of {base.wheel_motor_id}'s {motor.no_load_speed_rad_s:.3g} "
           f"rad/s no-load speed with {base.wheel_radius_m:g} m wheels is {top:.3g} m/s; the task needs "
           f"{req.speed_m_s:g} m/s.")
    if not ok:
        msg += " Use a faster wheel motor or larger wheels (then re-check wheel_torque)."
    return [CheckResult(check="top_speed", passed=ok, value=top, limit=req.speed_m_s, message=msg,
                        cause=None if ok else "speed_limited")]


def tip_over(spec: RobotSpec, req: Requirements, model: mujoco.MjModel,
             catalog: dict[str, Motor] | None = None) -> list[CheckResult]:
    """The centre of mass must project inside the support polygon shrunk by 15%, with the arm
    extended in 8 directions (payload in the gripper) and the robot on max_incline_deg
    facing any of 8 directions. On a slope the COM's projection moves downhill by h·tan(θ)."""
    base: WheeledBase = spec.base
    poly = support_polygon(base)
    if len(poly) < 3:
        return [CheckResult(
            check="tip_over", passed=False, value=0.0, limit=0.0, cause="unstable",
            message=f"{base.name} is differential drive with no caster: only two wheels touch the "
                    "ground, so it would fall over. Add a caster (caster: true) or use four_wheel.")]
    centroid = poly.mean(axis=0)
    shrunk = centroid + (1 - TIP_MARGIN) * (poly - centroid)
    mm = mass_model(spec, req, model)
    theta = np.radians(req.max_incline_deg)
    arm_cases = [("stowed", None)] + (DIRECTIONS if mm.arm_mass > 0 and mm.reach > 0 else [])
    slopes = [("flat", None)] + (DIRECTIONS if theta > 0 else [])
    worst = None
    for arm_label, arm_angle in arm_cases:
        com = mm.com(arm_angle)
        for slope_label, down in slopes:
            p = com[:2].copy()
            if down is not None:
                p += com[2] * np.tan(theta) * np.array([np.cos(down), np.sin(down)])
            m = _margin(p, shrunk)
            if worst is None or m < worst[0]:
                worst = (m, arm_label, slope_label, com)
    m, arm_label, slope_label, com = worst
    case = (f"arm {'extended towards ' + arm_label if arm_label != 'stowed' else 'stowed'}"
            f"{'' if slope_label == 'flat' else f', on the {req.max_incline_deg:g}° slope with downhill towards {slope_label}'}")
    ok = m > 0
    msg = (f"worst case ({case}): centre of mass at height {com[2]:.3g} m projects "
           f"{abs(m) * 1000:.0f} mm {'inside' if ok else 'outside'} the support polygon shrunk by "
           f"{TIP_MARGIN:.0%} (total {mm.total:.3g} kg with payloads"
           + (f", arm reach {mm.reach:.3g} m from its mount at ({mm.mount[0]:.3g}, {mm.mount[1]:.3g})"
              if mm.arm_mass else "") + ").")
    if not ok:
        msg += (" Fixes: a wider track (track_width_m) or longer wheelbase, a lower or heavier "
                "base (lower chassis, heavier battery low down), moving the arm mount towards the "
                "centre of the support polygon, or a shorter arm reach.")
    return [CheckResult(check="tip_over", passed=ok, value=m, limit=0.0, message=msg,
                        cause=None if ok else "unstable")]


def traction(spec: RobotSpec, req: Requirements, model: mujoco.MjModel,
             catalog: dict[str, Motor] | None = None) -> list[CheckResult]:
    """Climbing force needed must be below friction x normal load on the driven wheels (arm
    stowed, payloads on). A differential robot climbs forwards, which moves weight back
    off its front drive wheels."""
    base: WheeledBase = spec.base
    mm = mass_model(spec, req, model)
    force, theta = _climb_force(mm, req)
    mu = TERRAIN[req.terrain]["mu"]
    com = mm.com(None)
    weight = mm.total * G
    if base.drive == "four_wheel" or not base.caster:
        normal = weight * np.cos(theta)
        where = "all four wheels" if base.drive == "four_wheel" else "both wheels"
    else:
        xa = base.wheelbase_m / 2
        xc = caster_position(base)[0]
        normal = weight * (np.cos(theta) * (com[0] - xc) - np.sin(theta) * com[2]) / (xa - xc)
        normal = float(np.clip(normal, 0.0, weight))
        where = "the two front drive wheels"
    grip = mu * normal
    ok = force < grip
    msg = (f"climbing {req.max_incline_deg:g}° at {ACCEL_M_S2} m/s² needs {force:.3g} N of traction; "
           f"{where} carry {normal:.3g} N, so friction (μ={mu} on {req.terrain}) allows {grip:.3g} N.")
    if not ok:
        msg += (" The wheels would slip: put more weight over the driven wheels (move the battery, "
                "payload or arm mount towards them, or lower the centre of mass), or drive all four "
                "wheels.")
    return [CheckResult(check="traction", passed=ok, value=force, limit=grip, message=msg,
                        cause=None if ok else "slip")]


# Ground clearance by terrain (design conventions, unverified): the obstacle a base must roll over
# without touching (flat: thresholds and cable covers about 15 mm; carpet: pile, rug edges and
# carpet-to-floor strips about 20-25 mm; outdoor: kerb gaps, roots and stones about 40 mm). The
# chassis bottom must clear it, and a wheel rolls over a step of up to about half its radius, so
# the wheel radius must be at least about twice the step it meets (flat 0.03 m, carpet 0.04 m,
# outdoor 0.06 m, rounded). In this model the chassis bottom sits at half the wheel radius.
MIN_CLEARANCE_M = {"flat": 0.015, "carpet": 0.025, "outdoor": 0.04}
MIN_WHEEL_RADIUS_M = {"flat": 0.03, "carpet": 0.04, "outdoor": 0.06}


def min_wheel_radius_m(terrain: str) -> float:
    """The smallest wheel that passes ground_clearance on this terrain (both rules)."""
    return max(MIN_WHEEL_RADIUS_M[terrain], 2 * MIN_CLEARANCE_M[terrain])


def ground_clearance(spec: RobotSpec, req: Requirements, model: mujoco.MjModel,
                     catalog: dict[str, Motor] | None = None) -> list[CheckResult]:
    """The chassis bottom clears the terrain's typical obstacle, and the wheels are big enough to
    roll over it (MIN_CLEARANCE_M, MIN_WHEEL_RADIUS_M)."""
    base: WheeledBase = spec.base
    terrain = req.terrain
    need_c, need_r = MIN_CLEARANCE_M[terrain], MIN_WHEEL_RADIUS_M[terrain]
    fix = (f"Use wheels of at least {min_wheel_radius_m(terrain):g} m radius (the chassis bottom sits at half "
           "the wheel radius).")
    c_ok, r_ok = base.chassis_bottom_m >= need_c - 1e-9, base.wheel_radius_m >= need_r - 1e-9
    return [
        CheckResult(check="ground_clearance", passed=c_ok, value=base.chassis_bottom_m, limit=need_c,
                    cause=None if c_ok else "clearance",
                    message=f"chassis bottom {base.chassis_bottom_m * 1000:.0f} mm above the ground; {terrain} needs "
                            f"at least {need_c * 1000:.0f} mm (design convention)." + ("" if c_ok else " " + fix)),
        CheckResult(check="ground_clearance", passed=r_ok, value=base.wheel_radius_m, limit=need_r,
                    cause=None if r_ok else "clearance",
                    message=f"wheel radius {base.wheel_radius_m * 1000:.0f} mm; rolling over {terrain} obstacles needs "
                            f"at least {need_r * 1000:.0f} mm (design convention)." + ("" if r_ok else " " + fix)),
    ]


MOBILE_CHECKS = [wheel_torque, top_speed, tip_over, traction, ground_clearance]
