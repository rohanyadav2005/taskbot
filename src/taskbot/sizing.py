"""Deterministic sizing: DesignIntent + Requirements + motor catalog -> RobotSpec.

The LLM chooses (taskbot.intent); code sizes every number:

    arm       a fixed base (or the chassis), a riser link, then the layout's joints
              (ARM_LAYOUTS). Upper arm and forearm lengths reach the farthest target (workspace
              corners, centre and the reach_m point) from the shoulder exactly (the reach margin is
              already in reach_m, x1.10 in requirements; the sizer adds none), split
              evenly so the nearest target is inside the inner radius |L1 - L2|. The riser
              puts the shoulder at mid-workspace height, raised until self_collision passes
              (checks._search, the same search as the verified collision fixes). Joint limits
              are the IK angles over the targets widened by 15°, kept at least
              min_joint_range_rad, with the shoulder clipped so the upper arm can't swing
              down into the base. Links are tubes (radius from length and payload, 2 mm wall).
    motors    picked from the tip inward, because outer motors load the inner joints: for each
              joint the cheapest supported position motor whose usable torque (80% of rated)
              is at least 1.25 x the holding torque (checks.holding_loads) and whose torque at
              the planned peak speed is at least 1.25 x the gravity torque there
              (checks.speed_loads), repeated until the selection stops changing.
    mobile    the cheapest wheel motor and smallest wheel radius passing wheel_torque with
              1.25 x margin and top_speed (the mobile checks' own functions) and verified with
              the drive_straight and incline sims, which the static estimate alone doesn't
              guarantee near the motor's top speed; the smallest chassis passing tip_over and
              traction.

Every design is checked with every static check before it is returned: the result is a
SizedDesign that passes them, or a SizingFailure that says why ("infeasible": no catalog
choice can carry the load, with what would work; "checks": a design that still fails, for
autofix or a new intent). With nearest=True an infeasible design is relaxed (payload, then
reach) to the nearest feasible one, labelled APPROXIMATE.

To add an arm layout, add its joints to ARM_LAYOUTS (and its name to intent.ArmLayout).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from taskbot import checks
from taskbot.build_urdf import build_model, tool_length_m
from taskbot.checks import CheckResult, holding_loads, run_checks, speed_loads, usable_torque
from taskbot.intent import DesignIntent
from taskbot.mobile_checks import top_speed, wheel_torque
from taskbot.requirements import Requirements
from taskbot.schema import Gripper, Motor, RobotSpec, supported_motors
from taskbot.sim.motor import motor_model

MOTOR_MARGIN = 1.25
LIMIT_WIDEN_RAD = math.radians(15)
MIN_LINK_M = 0.05
WRIST_LINK_M = 0.04
BASE_FOOTPRINT_M = 0.12
BASE_HEIGHT_M = 0.03
RISER_MIN_M = 0.04
RISER_MAX_M = 0.35
RISER_RADIUS_M = 0.025
MAX_PITCH_RAD = 2.6
WRIST_PITCH_RAD = 2.0
MOTOR_PASSES = 5
GEOMETRY_PASSES = 3
GRIPPER = {"parallel": {"max_opening_m": 0.06, "grip_force_n": 10.0},
           "suction": {"max_opening_m": 0.04, "grip_force_n": 5.0}}
CHASSIS_M = (0.30, 0.24, 0.06)
CHASSIS_SCALES = (1.0, 1.25, 1.5, 1.75, 2.0)
WHEEL_WIDTH_M = 0.025
WHEEL_RADIUS_MAX_M = 0.12

# Joints per layout: (joint name, axis, link name). The first is the base yaw on the riser.
ARM_LAYOUTS: dict[str, list[tuple[str, str, str]]] = {
    "yaw_pitch_pitch": [("base_yaw", "z", "turret"), ("shoulder_pitch", "y", "upper_arm"),
                        ("elbow_pitch", "y", "forearm")],
    "yaw_pitch_pitch_pitch": [("base_yaw", "z", "turret"), ("shoulder_pitch", "y", "upper_arm"),
                              ("elbow_pitch", "y", "forearm"), ("wrist_pitch", "y", "wrist")],
    "yaw_pitch_pitch_wristroll": [("base_yaw", "z", "turret"), ("shoulder_pitch", "y", "upper_arm"),
                                  ("elbow_pitch", "y", "forearm"), ("wrist_roll", "z", "wrist")],
}


@dataclass
class SizedDesign:
    spec: RobotSpec
    requirements: Requirements          # what it was sized for (relaxed when approximate)
    steps: list[str] = field(default_factory=list)       # what was computed, and why
    provenance: dict[str, str] = field(default_factory=dict)  # "part.field" -> who chose it
    approximate: str | None = None      # the requirement relaxed (nearest mode)


@dataclass
class SizingFailure:
    kind: str                           # "infeasible" | "checks"
    reason: str
    options: list[str] = field(default_factory=list)
    spec: RobotSpec | None = None
    failures: list[CheckResult] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    provenance: dict[str, str] = field(default_factory=dict)


def _position_motors(catalog: dict[str, Motor]) -> list[Motor]:
    return sorted((m for m in supported_motors(catalog).values() if "position" in m.modes),
                  key=lambda m: (m.price_usd, m.rated_torque_nm))


def _tube(name: str, length: float, payload: float, material: str) -> dict:
    mat = material if material != "auto" else ("pla" if length < 0.25 else "aluminium")
    r = float(np.clip(0.012 + 0.02 * length + 0.01 * payload, 0.012, 0.025))
    return {"name": name, "shape": "tube", "radius_m": round(r, 4), "length_m": round(length, 4), "material": mat}


def _targets(req: Requirements) -> list[np.ndarray]:
    """The points the arm must reach: the workspace corners and centre, and the reach check's
    point (reach_m from the base axis at mid-workspace height). reach_m already holds the reach
    margin (x1.10 over the farthest corner, in requirements); the sizer adds none."""
    from taskbot.sim.plan import workspace_targets

    ws = req.workspace
    pts = [t for _, t in workspace_targets(req)]
    return pts + [np.array([req.reach_m, 0.0, (ws.z_min + ws.z_max) / 2])]


def _arm_lengths(targets, shoulder: np.ndarray, tail: float) -> tuple[float, float, float, float]:
    """(upper arm, forearm, farthest, nearest): exactly reach the farthest target (its margin is
    already in reach_m), split so the inner radius |L1 - (L2 + tail)| is ~0."""
    d = [float(np.linalg.norm(t - shoulder)) for t in targets]
    far, near = max(d), min(d)
    total = far
    l1, l2 = total / 2, total / 2 - tail
    if l2 < MIN_LINK_M:
        l2, l1 = MIN_LINK_M, total - tail - MIN_LINK_M
    return max(l1, MIN_LINK_M), l2, far, near


# --------------------------------------------------------------------------- arm


class _Arm:
    """The arm being sized, as a spec dict with named handles."""

    def __init__(self, intent: DesignIntent, req: Requirements, catalog, base: dict, mount=None):
        self.intent, self.req, self.catalog = intent, req, catalog
        self.layout = ARM_LAYOUTS[intent.arm_layout]
        motors = _position_motors(catalog)
        self.cheapest = motors[0].id
        self.lightest = min(motors, key=lambda m: m.mass_kg).id
        self.base = base
        self.mount = mount
        g = intent.gripper
        self.gripper = None if g == "none" else {
            "kind": "gripper", "name": "gripper", "gripper_type": g, **GRIPPER[g],
            **({"motor_id": self.lightest} if g == "parallel" else {})}
        self.wrist = WRIST_LINK_M if len(self.layout) > 3 else 0.0
        self.tail = self.wrist + tool_length_m(Gripper(**self.gripper) if self.gripper else None)
        self.motors = {name: self.cheapest for name, _, _ in self.layout}
        self.limits: dict[str, tuple[float, float]] = {}
        self.riser = RISER_MIN_M
        self.l1 = self.l2 = 0.1

    def base_top(self) -> float:
        b = self.base
        if b["kind"] == "fixed_base":
            return b["height_m"]
        return 0.5 * b["wheel_radius_m"] + b["chassis_size_m"][2]

    def shoulder(self, riser: float | None = None) -> np.ndarray:
        x, y = self.mount or (0.0, 0.0)
        return np.array([x, y, self.base_top() + (self.riser if riser is None else riser)])

    def lengths(self) -> list[float]:
        out = [self.riser, self.l1, self.l2]
        return out + ([self.wrist] if self.wrist else [])

    def as_dict(self, limits: dict | None = None, riser: float | None = None) -> dict:
        lengths = self.lengths()
        if riser is not None:
            lengths[0] = riser
        segs = []
        for (jname, axis, lname), L in zip(self.layout, lengths):
            if lname == "turret":
                link = {"name": lname, "shape": "tube", "radius_m": RISER_RADIUS_M, "length_m": round(L, 4),
                        "material": "aluminium" if self.intent.link_material == "aluminium" else "pla"}
            else:
                link = _tube(lname, L, self.req.payload_kg, self.intent.link_material)
            if axis == "z":  # base yaw and wrist roll turn freely
                joint = {"name": jname, "type": "continuous", "axis": axis, "motor_id": self.motors[jname]}
            else:
                lo, hi = (limits or self.limits).get(jname, self._wide(jname, axis))
                joint = {"name": jname, "type": "revolute", "axis": axis, "lower": round(lo, 4),
                         "upper": round(hi, 4), "motor_id": self.motors[jname]}
            segs.append({"joint": joint, "link": link})
        arm = {"kind": "arm_chain", "name": "arm", "segments": segs}
        if self.mount is not None:
            arm["mount_xy_m"] = [round(self.mount[0], 4), round(self.mount[1], 4)]
        modules = [self.base, arm] + ([self.gripper] if self.gripper else [])
        return {"name": "sized_robot", "description": "", "modules": json.loads(json.dumps(modules))}

    def spec(self, **kw) -> RobotSpec:
        return RobotSpec.model_validate(self.as_dict(**kw))

    @staticmethod
    def _wide(jname: str, axis: str) -> tuple[float, float]:
        return (-WRIST_PITCH_RAD, WRIST_PITCH_RAD) if jname == "wrist_pitch" else (-MAX_PITCH_RAD, MAX_PITCH_RAD)

    # ---- geometry

    def size_lengths(self, steps: list[str]) -> None:
        targets = _targets(self.req)
        self.l1, self.l2, far, near = _arm_lengths(targets, self.shoulder(), self.tail)
        steps.append(f"upper_arm {self.l1:.3g} m, forearm {self.l2:.3g} m: together with {self.tail:.3g} m of wrist "
                     f"and gripper they reach the farthest target, {far:.3g} m from the shoulder (reach_m "
                     f"{self.req.reach_m:g} m already holds the 10% reach margin; the sizer adds none); the nearest "
                     f"is {near:.3g} m away, inside |upper_arm - (forearm + wrist + gripper)| = "
                     f"{abs(self.l1 - self.l2 - self.tail):.3g} m")

    def size_limits(self, steps: list[str]) -> None:
        from taskbot.sim.ik import solve_ik
        from taskbot.sim.model import build_scene

        self.limits = {}
        robot = build_scene(self.spec(), self.catalog)
        q, qs = robot.home(), []
        for i, t in enumerate(_targets(self.req)):
            ik = solve_ik(robot, t, q, seed=i)
            qs.append(ik.q)
            q = ik.q
        qs = np.array(qs)
        need = self.req.min_joint_range_rad + 1e-3
        for k, (jname, axis, _) in enumerate(self.layout):
            if axis == "z":  # continuous
                continue
            lo = min(float(qs[:, k].min()), 0.0) - LIMIT_WIDEN_RAD
            hi = max(float(qs[:, k].max()), 0.0) + LIMIT_WIDEN_RAD
            cap = WRIST_PITCH_RAD if jname == "wrist_pitch" else MAX_PITCH_RAD
            if jname == "shoulder_pitch":  # the upper arm's tip stays above the base top
                cap = min(cap, math.acos(float(np.clip(-self.riser / self.l1, -1.0, 1.0))))
            lo, hi = max(lo, -cap), min(hi, cap)
            if hi - lo < need:  # widen to the minimum range, inside the cap
                grow = need - (hi - lo)
                lo, hi = max(-cap, lo - grow / 2), min(cap, hi + grow / 2)
                if hi - lo < need:
                    lo, hi = (lo, lo + need) if lo + need <= cap else (hi - need, hi)
            self.limits[jname] = (lo, hi)
        steps.append("joint limits: IK angles over the workspace corners, centre and reach point, widened "
                     f"by {math.degrees(LIMIT_WIDEN_RAD):.0f}°, base yaw and wrist roll continuous: " + ", ".join(
                         f"{j} {lo:.3g} to {hi:.3g} rad" for j, (lo, hi) in self.limits.items()))

    # ---- motors

    def pick_motors(self, steps: list[str], start: int | None = None) -> tuple[str, float, float] | None:
        """Tip inward from joint index `start` (default the tip), cheapest motor with margin.
        Returns (joint, need, best usable) for a joint no motor can carry, else None."""
        motors = _position_motors(self.catalog)
        order = [name for name, _, _ in self.layout][::-1]
        if start is not None:
            order = order[len(order) - 1 - start:]
        failing = None
        reasons: dict[str, str] = {}
        for _ in range(MOTOR_PASSES):
            changed, failing = False, None
            spec = self.spec()
            sl = {s.joint.name: s for s in speed_loads(spec, self.req, self.catalog)}
            for name in order:
                spec = self.spec()
                hl = {jl.joint.name: jl for jl in holding_loads(spec, self.req, build_model(spec, self.catalog))}
                need, s = hl[name].need, sl[name]
                margin = MOTOR_MARGIN
                pick = next((m for m in motors if usable_torque(m) >= margin * need
                             and motor_model(m).drive_limit(s.speed) >= margin * s.gravity), None)
                if pick is None:  # no motor with the margin: the cheapest that still passes the checks
                    margin = 1.0
                    pick = next((m for m in motors if usable_torque(m) >= need
                                 and motor_model(m).drive_limit(s.speed) >= s.gravity), None)
                if pick is None:
                    pick = max(motors, key=lambda m: m.rated_torque_nm)
                    failing = (name, need, usable_torque(pick))
                if pick.id != self.motors[name]:
                    self.motors[name], changed = pick.id, True
                reasons[name] = (f"{name}: {pick.id} (needs {need:.3g} Nm holding, usable {usable_torque(pick):.3g} "
                                 f"Nm; {s.gravity:.3g} Nm at {s.speed:.3g} rad/s, gives "
                                 f"{motor_model(pick).drive_limit(s.speed):.3g} Nm there; "
                                 + (f"margin {MOTOR_MARGIN})" if margin > 1 else
                                    f"no motor has the {MOTOR_MARGIN}x margin, this one passes without it)"))
            if not changed:
                break
        steps.extend(reasons[n] for n, _, _ in self.layout if n in reasons)
        return failing

    # ---- collisions

    def size_riser(self, steps: list[str]) -> bool:
        """Raise the riser until self_collision passes. True if it changed."""
        spec = self.spec()
        frac, _, _ = checks._collision_rate(spec, self.req, build_model(spec, self.catalog))
        if frac <= checks.COLLISION_MAX_FRACTION:
            return False

        def make(h):
            return self.as_dict(riser=h)
        found = checks._search(make, self.riser, RISER_MAX_M, self.req, self.catalog, smallest=True, step=0.005)
        if found is None or abs(found[0] - self.riser) < 1e-6:
            return False
        steps.append(f"turret {found[0]:.3g} m (was {self.riser:.3g} m): links hit the base in {frac:.0%} of "
                     f"sampled poses; this is the shortest riser that passes self_collision ({found[1]:.1%})")
        self.riser = found[0]
        return True


def _size_arm(arm: _Arm, steps: list[str]) -> tuple[str, float, float] | None:
    ws = arm.req.workspace
    arm.riser = float(np.clip((ws.z_min + ws.z_max) / 2 - arm.base_top(), RISER_MIN_M, RISER_MAX_M))
    steps.append(f"turret {arm.riser:.3g} m: puts the shoulder at mid-workspace height "
                 f"({(ws.z_min + ws.z_max) / 2:.3g} m), at least {RISER_MIN_M} m")
    failing = None
    for _ in range(GEOMETRY_PASSES):
        arm.size_lengths(steps)
        arm.limits = {}
        arm.size_limits(steps)
        failing = arm.pick_motors(steps)
        if not arm.size_riser(steps):
            break
    return failing


# --------------------------------------------------------------------------- mobile base


def _base_dict(intent: DesignIntent, scale: float, r: float, motor: str, caster: bool) -> dict:
    cx, cy, cz = CHASSIS_M[0] * scale, CHASSIS_M[1] * scale, CHASSIS_M[2]
    return {"kind": "wheeled_base", "name": "base", "drive": intent.drive, "wheel_radius_m": round(r, 4),
            "wheel_width_m": WHEEL_WIDTH_M, "track_width_m": round(cy + WHEEL_WIDTH_M + 0.01, 4),
            "wheelbase_m": round(cx - 0.04, 4), "chassis_size_m": [round(cx, 4), round(cy, 4), cz],
            "chassis_material": "aluminium", "wheel_motor_id": motor, "caster": caster}


def _size_base(intent, req, catalog, arm: _Arm | None, steps: list[str]) -> tuple[dict | None, str | None]:
    """The cheapest wheel motor and smallest radius with wheel_torque margin and top_speed, that
    drives straight and climbs the incline in simulation, on the smallest chassis that passes
    tip_over and traction. (base dict, failure reason)."""
    from taskbot.mobile_checks import tip_over, traction
    from taskbot.mobile_sim import drive_straight, incline

    caster = bool(intent.caster) if intent.drive == "differential" else False
    if intent.drive == "differential" and not caster:
        caster = True
        steps.append("caster added: a differential base stands on its two wheels and a caster")
    wheel_motors = sorted((m for m in supported_motors(catalog).values() if "velocity" in m.modes),
                          key=lambda m: m.price_usd)

    def spec_for(b):
        if arm is None:
            return RobotSpec.model_validate({"name": "sized_robot", "description": "", "modules": [b]})
        arm.base = b
        return arm.spec()

    last = "no wheel motor in the catalog"
    for scale in CHASSIS_SCALES:
        chosen = None
        for m in wheel_motors:
            r = max(MOTOR_MARGIN * req.speed_m_s / (0.8 * m.no_load_speed_rad_s), 0.02)
            while r <= WHEEL_RADIUS_MAX_M:
                b = _base_dict(intent, scale, r, m.id, caster)
                spec = spec_for(b)
                model = build_model(spec, catalog)
                (wt,), (ts,) = wheel_torque(spec, req, model, catalog), top_speed(spec, req, model, catalog)
                if wt.value * MOTOR_MARGIN <= wt.limit and ts.passed:
                    sims = [t(spec, req, catalog).result for t in (drive_straight, incline)]
                    if all(x.passed for x in sims):
                        chosen = (b, m, r, wt)
                        break
                    last = next(x.message for x in sims if not x.passed)
                else:
                    last = f"wheel_torque with {m.id}: {wt.message}"
                r += 0.005
            if chosen:
                break
        if chosen is None:
            continue
        b, m, r, wt = chosen
        spec = spec_for(b)
        model = build_model(spec, catalog)
        (tip,), (tr,) = tip_over(spec, req, model, catalog), traction(spec, req, model, catalog)
        if tip.passed and tr.passed:
            steps.append(f"wheels: {m.id}, radius {r:.3g} m (each wheel needs {wt.value:.3g} Nm, it gives "
                         f"{wt.limit:.3g} Nm at speed; margin {MOTOR_MARGIN}; the smallest radius that also "
                         "drives straight and climbs the incline in simulation); chassis "
                         f"{b['chassis_size_m'][0]:.3g} x {b['chassis_size_m'][1]:.3g} m, the smallest that passes "
                         "tip_over and traction")
            return b, None
        last = tip.message if not tip.passed else tr.message
    return None, last


# --------------------------------------------------------------------------- options and results


def _options(arm: _Arm, joint: str) -> list[str]:
    """What would work with the strongest motor at `joint`: the largest payload at this reach and
    the longest reach with this payload, the arm's own masses included."""
    from taskbot.checks import stretch_m

    strongest = max(_position_motors(arm.catalog), key=lambda m: m.rated_torque_nm)
    keep = dict(arm.motors)
    arm.motors = {n: (strongest.id if n == joint or keep[n] == strongest.id else keep[n]) for n in keep}
    allowed = usable_torque(strongest)

    def need(req, scale=1.0):
        l1, l2 = arm.l1, arm.l2
        arm.l1, arm.l2 = l1 * scale, l2 * scale
        spec = arm.spec()
        loads = {jl.joint.name: jl.need for jl in holding_loads(spec, req, build_model(spec, arm.catalog))}
        arm.l1, arm.l2 = l1, l2
        return loads[joint], stretch_m(spec)

    out = []
    zero, _ = need(arm.req.model_copy(update={"payload_kg": 0.0}))
    one, stretch = need(arm.req.model_copy(update={"payload_kg": 1.0}))
    per_kg = one - zero
    if per_kg > 0 and allowed > zero:
        out.append(f"At this reach ({stretch:.3g} m arm), a payload under {(allowed - zero) / per_kg:.3g} kg could "
                   f"work with {strongest.id} at {joint}.")
    lo, hi = 0.1, 1.0
    if need(arm.req, lo)[0] <= allowed:
        for _ in range(12):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if need(arm.req, mid)[0] <= allowed else (lo, mid)
        out.append(f"With {arm.req.payload_kg:.3g} kg, an arm reaching under {need(arm.req, lo)[1]:.3g} m could "
                   "work: put the robot closer to the work.")
    arm.motors = keep
    return out


def _provenance(intent: DesignIntent, spec: RobotSpec) -> dict[str, str]:
    p = {"robot_type": "LLM intent", "arm layout": "LLM intent", "gripper type": "LLM intent",
         "link material policy": "LLM intent"}
    if intent.drive:
        p |= {"drive": "LLM intent", "caster": "LLM intent", **({"arm mount": "LLM intent"} if intent.arm_mount else {})}
    for m in spec.modules:
        for seg in getattr(m, "segments", []):
            for f in ("length_m", "radius_m", "material"):
                p[f"{seg.link.name}.{f}"] = "sizer"
            for f in ("lower", "upper", "motor_id"):
                p[f"{seg.joint.name}.{f}"] = "sizer"
        if m.kind == "wheeled_base":
            for f in ("wheel_radius_m", "track_width_m", "wheelbase_m", "chassis_size_m", "wheel_motor_id"):
                p[f"{m.name}.{f}"] = "sizer"
        if m.kind in ("fixed_base",):
            p[f"{m.name}.footprint_m"] = p[f"{m.name}.height_m"] = "sizer (fixed default)"
        if m.kind == "gripper":
            p[f"{m.name}.max_opening_m"] = "sizer (fixed default)"
            if m.motor_id:
                p[f"{m.name}.motor_id"] = "sizer (lightest position motor)"
    return p


def size_design(intent: DesignIntent, req: Requirements, catalog: dict[str, Motor], *,
                nearest: bool = False) -> SizedDesign | SizingFailure:
    steps: list[str] = []
    arm = None
    if req.has_arm:
        if req.is_mobile:
            base = _base_dict(intent, 1.0, 0.04, sorted(
                (m for m in supported_motors(catalog).values() if "velocity" in m.modes),
                key=lambda m: m.price_usd)[0].id, intent.drive == "differential")
            mount = (CHASSIS_M[0] / 2 - 0.06, 0.0) if intent.arm_mount == "front" else (0.0, 0.0)
        else:
            base = {"kind": "fixed_base", "name": "base", "footprint_m": [BASE_FOOTPRINT_M] * 2,
                    "height_m": BASE_HEIGHT_M, "material": "aluminium"}
            mount = None
        arm = _Arm(intent, req, catalog, base, mount)
        failing = _size_arm(arm, steps)
        if req.is_mobile:  # the arm's mass sizes the base; the chassis top moves the shoulder
            for _ in range(2):
                b, why = _size_base(intent, req, catalog, arm, steps)
                if b is None:
                    return SizingFailure("checks", f"no wheel motor, radius and chassis passed: {why}", steps=steps)
                old_top = arm.base_top()
                arm.base = b
                if intent.arm_mount == "front":
                    arm.mount = (b["chassis_size_m"][0] / 2 - 0.06, 0.0)
                if abs(arm.base_top() - old_top) < 1e-3:
                    break
                failing = _size_arm(arm, steps)
        spec = arm.spec()
        if failing is not None:
            joint, need, usable = failing
            reason = (f"{joint} needs {need:.3g} Nm to hold the sized arm out straight with {req.payload_kg:.3g} kg "
                      f"(its own links and motors included), but the strongest catalog motor gives {usable:.3g} Nm "
                      "usable (80% of rated).")
            options = _options(arm, joint)
            if nearest:
                return _nearest(intent, req, catalog, options, reason, steps)
            return SizingFailure("infeasible", reason, options, spec, steps=steps,
                                 provenance=_provenance(intent, spec))
    else:
        b, why = _size_base(intent, req, catalog, None, steps)
        if b is None:
            return SizingFailure("checks", f"no wheel motor, radius and chassis passed: {why}", steps=steps)
        spec = RobotSpec.model_validate({"name": "sized_robot", "description": "", "modules": [b]})
    spec = spec.model_copy(update={"name": _name(req), "description": f"Sized by TaskBot from the intent: {intent.rationale}"})
    failures = [r for r in run_checks(spec, req, catalog) if not r.passed]
    prov = _provenance(intent, spec)
    if failures:
        return SizingFailure("checks", "after sizing, " + "; ".join(
            f"{r.check} still fails" for r in failures), spec=spec, failures=failures, steps=steps, provenance=prov)
    return SizedDesign(spec, req, steps, prov)


def _name(req: Requirements) -> str:
    import re

    words = re.sub(r"[^a-z0-9 ]+", " ", req.task_summary.lower()).split()[:4]
    return "_".join(words) or "sized_robot"


def _nearest(intent, req, catalog, options, reason, steps) -> SizedDesign | SizingFailure:
    """Relax the payload (to 95% of the largest that works), else the reach, and size again."""
    import re

    m = next((re.search(r"a payload under ([\d.]+) kg", o) for o in options if "payload under" in o), None)
    candidates = []
    if m:
        candidates.append(("payload_kg", req.payload_kg, round(float(m[1]) * 0.95, 3)))
    m = next((re.search(r"reaching under ([\d.]+) m", o) for o in options if "reaching under" in o), None)
    if m:
        candidates.append(("reach", req.reach_m, None))
    for field_, old, new in candidates:
        try:
            if field_ == "payload_kg":
                relaxed = Requirements(**(req.model_dump() | {"payload_kg": new}))
                what = f"payload_kg {old:g} -> {new:g} kg"
            else:
                k = 0.9
                ws = req.workspace
                relaxed = Requirements(**(req.model_dump() | {
                    "reach_m": round(req.reach_m * k, 3),
                    "workspace": {"x_min": ws.x_min * k, "x_max": ws.x_max * k, "y_min": ws.y_min * k,
                                  "y_max": ws.y_max * k, "z_min": ws.z_min, "z_max": ws.z_max}}))
                what = f"reach_m {old:g} -> {relaxed.reach_m:g} m (workspace scaled by {k})"
        except ValueError:
            continue
        sized = size_design(intent, relaxed, catalog)
        if isinstance(sized, SizedDesign):
            sized.approximate = what
            sized.steps = steps + [f"APPROXIMATE: {reason} Relaxed {what} to the nearest feasible design."] + sized.steps
            return sized
    return SizingFailure("infeasible", reason, options, steps=steps)
