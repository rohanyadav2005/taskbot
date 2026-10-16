"""Simulation tests generated from the Requirements.

    reach_targets  move to the workspace centre and its 8 corners
    carry_payload  the same moves with the payload welded to the gripper
    hold_still     hold the most heavily loaded target pose, with payload, for 5 s

Every test returns a SimRun whose `.result` is a CheckResult, so it plugs into
the repair loop next to the static checks. Each move is a minimum-jerk joint
trajectory to an IK solution, as long as the motors' speed limits require (at
least 2 s), tracked by each motor's own fixed position loop plus gravity
feedforward (taskbot.sim.control), under each motor's torque-speed model.

carry_payload and hold_still run with the controller believing the payload is
DEFAULT_PAYLOAD_MASS_ERROR lighter than it is; that run decides pass/fail. The
same test with the true mass is kept as `SimRun.ideal` for the report.

Every failure names a cause (CheckResult.cause):
    saturated      a joint wanted more torque than its motor's limit for > 0.2 s
    speed_limited  ...and the limit was low because the joint was moving fast
    compliance     the error happened without any joint saturating
    unreachable    IK found no pose that reaches the target
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from taskbot.checks import CAUSE_ADVICE, CheckResult
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec, load_motor_catalog
from taskbot.sim.control import DEFAULT_PAYLOAD_MASS_ERROR, MotorController, step
from taskbot.sim.model import SimRobot, build_scene
from taskbot.sim.plan import (
    MIN_MOVE_S,
    SPEED_FRACTION,
    min_jerk,
    plan_moves,
    workspace_targets,
)
from taskbot.sim.record import FPS

# Position limits come from the requirements' position_tolerance_m (tol):
PEAK_FACTOR = 3.0        # peak deviation from the planned path / commanded pose < 3 tol
OVERSHOOT_FACTOR = 2.0   # overshoot past a target < 2 tol
SETTLE_DEADLINE_S = 1.0   # error must be under tolerance by move duration + this
SETTLE_CONFIRM_S = 0.5    # ...and stay there for this long after the deadline
SATURATION_MAX_S = 0.2
HOLD_S = 5.0


def _mm(x: float) -> str:
    return f"{x * 1000:.3g} mm"


class SimMeasurementError(RuntimeError):
    """A sim test produced a value that means it measured nothing."""


@dataclass
class SimRun:
    name: str
    result: CheckResult
    robot: SimRobot                                   # scene used, for rendering
    frames: np.ndarray = field(repr=False)            # qpos at FPS, shape (n, nq)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.15)
    camera_distance: float = 1.0
    ideal: CheckResult | None = None                  # same test, controller knows the mass
    track_body: int | None = None                     # follow this body with the camera


# --------------------------------------------------------------------------- simulation


class _Sim:
    """Steps the scene under the controller and records frames and saturation."""

    def __init__(self, robot: SimRobot, q_start: np.ndarray, payload_mass_error: float):
        self.robot = robot
        self.data = mujoco.MjData(robot.model)
        robot.set_pose(self.data, q_start)
        self.ctrl = MotorController(robot, payload_mass_error=payload_mass_error)
        self.dt = robot.model.opt.timestep
        self.frame_every = max(1, round(1 / (FPS * self.dt)))
        self.steps = 0
        self.frames: list[np.ndarray] = [self.data.qpos.copy()]
        n = robot.n
        self.sat_run = np.zeros(n)        # current consecutive saturated time per joint
        self.slow_run = np.zeros(n)       # ...of which speed-limited
        self.sat_max = np.zeros(n)        # longest saturated stretch per joint
        self.sat_max_slow = np.zeros(n)   # speed-limited time within that stretch
        self.sat_where: list[str] = [""] * n
        self.label = ""
        self._ref_data = mujoco.MjData(robot.model)
        self._jacp = np.zeros((3, robot.model.nv))

    def lever_errors(self, q_target: np.ndarray, q: np.ndarray | None = None) -> np.ndarray:
        """Per joint: |joint error| x how far that joint moves the tool per radian (m),
        at the current state or, if given, at arm configuration q."""
        data = self.data
        if q is not None:
            data = self._ref_data
            data.qpos[self.robot.qadr] = q
            mujoco.mj_kinematics(self.robot.model, data)
            mujoco.mj_comPos(self.robot.model, data)
        mujoco.mj_jacSite(self.robot.model, data, self._jacp, None, self.robot.site)
        lever = np.linalg.norm(self._jacp[:, self.robot.dadr], axis=0)
        return lever * np.abs((self.q() if q is None else q) - q_target)

    def run(self, seconds: float, ref) -> np.ndarray:
        """Step for `seconds`, with ref(t) -> (q_ref, qd_ref). Returns tool positions.
        Also records the arm configuration each step (self.q_hist), and tracks the peak
        distance between the tool and where q_ref puts it at that same instant
        (self.peak_lag: lag behind schedule plus any sideways error) and the per-joint
        lever errors at that moment (self.peak_contrib)."""
        tool, q_hist = [], []
        self.peak_lag, self.peak_contrib = 0.0, np.zeros(self.robot.n)
        for k in range(round(seconds / self.dt)):
            q_ref, qd_ref = ref(k * self.dt)
            out = step(self.robot, self.data, self.ctrl, q_ref, qd_ref)
            q_hist.append(self.q())
            self._ref_data.qpos[self.robot.qadr] = q_ref
            mujoco.mj_kinematics(self.robot.model, self._ref_data)
            dev = float(np.linalg.norm(self.robot.ee(self.data) - self._ref_data.site_xpos[self.robot.site]))
            if dev > self.peak_lag:
                self.peak_lag, self.peak_contrib = dev, self.lever_errors(q_ref)
            self.sat_run = np.where(out.saturated, self.sat_run + self.dt, 0.0)
            self.slow_run = np.where(out.saturated, self.slow_run + self.dt * out.speed_limited, 0.0)
            for i in np.flatnonzero(self.sat_run > self.sat_max):
                self.sat_where[i] = self.label
                self.sat_max[i] = self.sat_run[i]
                self.sat_max_slow[i] = self.slow_run[i]
            self.steps += 1
            if self.steps % self.frame_every == 0:
                self.frames.append(self.data.qpos.copy())
            tool.append(self.robot.ee(self.data))
        self.q_hist = np.array(q_hist)
        return np.array(tool)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.robot.qadr].copy()

    def saturation_cause(self, i: int) -> str:
        return "speed_limited" if self.sat_max_slow[i] > 0.5 * self.sat_max[i] else "saturated"


@dataclass
class _MoveResult:
    label: str
    tol: float
    ik_error: float
    duration: float
    final_error: float     # at duration + SETTLE_DEADLINE_S + SETTLE_CONFIRM_S
    overshoot: float       # furthest the tool went past the target along the approach line
    settle_s: float | None  # from move start until the error stays under tolerance
    peak_dev: float        # largest distance from the planned tool path (nearest point on it)
    final_contrib: np.ndarray  # per-joint lever errors at the end of the move
    peak_contrib: np.ndarray   # ...at the moment of peak path deviation
    lag: float = 0.0       # largest distance behind/ahead of the planned position at the same
                           # instant (reported, not judged: lateness along the path isn't a miss)

    @property
    def unreachable(self) -> bool:
        return self.ik_error >= self.tol

    @property
    def settled_in_time(self) -> bool:
        return self.settle_s is not None and self.settle_s <= self.duration + SETTLE_DEADLINE_S

    @property
    def steady_fail(self) -> bool:
        return not self.settled_in_time

    @property
    def transient_fail(self) -> bool:
        return self.peak_dev >= PEAK_FACTOR * self.tol or self.overshoot >= OVERSHOOT_FACTOR * self.tol

    @property
    def failures(self) -> list[str]:
        out = []
        if self.unreachable:
            out.append(f"unreachable: IK gets no closer than {_mm(self.ik_error)}")
        if not self.settled_in_time:
            if self.settle_s is None:
                out.append(f"never settled within {_mm(self.tol)} (final error {_mm(self.final_error)})")
            else:
                out.append(f"settled after {self.settle_s:.2f} s, later than the "
                           f"{self.duration + SETTLE_DEADLINE_S:.2f} s deadline")
        if self.peak_dev >= PEAK_FACTOR * self.tol:
            out.append(f"strayed {_mm(self.peak_dev)} sideways from the planned path "
                       f"(limit {_mm(PEAK_FACTOR * self.tol)})")
        if self.overshoot >= OVERSHOOT_FACTOR * self.tol:
            out.append(f"overshot by {_mm(self.overshoot)} (limit {_mm(OVERSHOOT_FACTOR * self.tol)})")
        return out


def _run_moves(robot: SimRobot, req: Requirements, payload_mass_error: float) -> tuple[_Sim, list[_MoveResult]]:
    tol = req.position_tolerance_m
    sim = _Sim(robot, robot.home(), payload_mass_error)
    results = []
    for mv in plan_moves(robot, req):
        T = mv.duration(robot.no_load_speed)
        start = sim.q()  # actual pose, which may differ from the plan if the last move failed
        sim.label = f"moving to {mv.label}"
        path = sim.run(T + SETTLE_DEADLINE_S + SETTLE_CONFIRM_S, lambda t: min_jerk(start, mv.q1, T, t))
        errors = np.linalg.norm(path - mv.target, axis=1)
        outside = np.flatnonzero(errors >= tol)
        if errors[-1] >= tol:
            settle = None
        else:
            settle = 0.0 if not len(outside) else (outside[-1] + 1) * sim.dt
        approach = mv.target - path[0]
        dist = np.linalg.norm(approach)
        overshoot = float(np.max((path - mv.target) @ (approach / dist), initial=0.0)) if dist > 1e-9 else 0.0
        dev, at, q_near = _path_deviation(robot, sim, path, start, mv.q1, T)
        results.append(_MoveResult(mv.label, tol, mv.ik_error, T, float(errors[-1]), max(overshoot, 0.0),
                                   settle, dev, sim.lever_errors(mv.q1),
                                   sim.lever_errors(q_near, sim.q_hist[at]), sim.peak_lag))
    return sim, results


PATH_SAMPLES = 400


def _path_deviation(robot: SimRobot, sim: _Sim, path: np.ndarray, q0: np.ndarray, q1: np.ndarray,
                    T: float) -> tuple[float, int, np.ndarray]:
    """Largest distance from the tool to the planned tool path (a polyline through the
    tool positions of the min-jerk trajectory), the step where it happens, and the planned
    configuration at the nearest point. Being late or early along the path doesn't count."""
    data = mujoco.MjData(robot.model)
    qs = np.array([min_jerk(q0, q1, T, t)[0] for t in np.linspace(0.0, T, PATH_SAMPLES)])
    pts = np.empty((len(qs), 3))
    for i, q in enumerate(qs):
        data.qpos[robot.qadr] = q
        mujoco.mj_kinematics(robot.model, data)
        pts[i] = data.site_xpos[robot.site]
    a, b = pts[:-1], pts[1:]
    ab = b - a
    ab2 = np.maximum(np.einsum("ij,ij->i", ab, ab), 1e-18)
    best_d = np.full(len(path), np.inf)
    best_seg = np.zeros(len(path), int)
    for start in range(0, len(path), 2000):  # chunks of steps x segments
        p = path[start:start + 2000, None, :]
        t = np.clip(np.einsum("nsj,sj->ns", p - a, ab) / ab2, 0.0, 1.0)
        d = np.linalg.norm(p - (a + t[..., None] * ab), axis=2)
        best_seg[start:start + 2000] = d.argmin(axis=1)
        best_d[start:start + 2000] = d.min(axis=1)
    at = int(best_d.argmax())
    return float(best_d[at]), at, qs[best_seg[at]]


def _view(req: Requirements) -> dict:
    ws = req.workspace
    reach = max(req.reach_m, abs(ws.x_max), abs(ws.x_min), abs(ws.y_max), abs(ws.y_min))
    return {"look_at": (0.0, 0.0, max(ws.z_max, 0.1) / 2), "camera_distance": max(0.6, 3.0 * reach)}


def _move_summary(moves: list[_MoveResult]) -> tuple[bool, str]:
    tol = moves[0].tol
    failed = [m for m in moves if m.failures]
    durations = [m.duration for m in moves]
    settles = [m.settle_s for m in moves if m.settle_s is not None]
    text = (f"{len(moves) - len(failed)}/{len(moves)} targets met the {_mm(tol)} task tolerance (final "
            f"error < {_mm(tol)} by move end + {SETTLE_DEADLINE_S:.0f} s, path deviation < "
            f"{_mm(PEAK_FACTOR * tol)}, overshoot < {_mm(OVERSHOOT_FACTOR * tol)}); "
            f"move durations {min(durations):.2f}–{max(durations):.2f} s (planned for "
            f"<= {SPEED_FRACTION:.0%} of no-load speed, min {MIN_MOVE_S:.0f} s)")
    if settles:
        text += f"; settling {min(settles):.2f}–{max(settles):.2f} s after move start"
    text += (f"; max path deviation {_mm(max(m.peak_dev for m in moves))}, max overshoot "
             f"{_mm(max(m.overshoot for m in moves))}; max lag behind the planned timing "
             f"{_mm(max(m.lag for m in moves))} (reported only)")
    if failed:
        text += "; failed: " + "; ".join(f"{m.label}: {', '.join(m.failures)}" for m in failed)
    return not failed, text


def _saturation(robot: SimRobot, sim: _Sim, payload_kg: float) -> tuple[list[int], float, str]:
    """(joints saturated > 0.2 s, longest saturation, text)."""
    worst = float(sim.sat_max.max(initial=0.0))
    bad = sorted((i for i in range(robot.n) if sim.sat_max[i] > SATURATION_MAX_S),
                 key=lambda i: -sim.sat_max[i])
    if not bad:
        return [], worst, f"no joint saturated for more than {SATURATION_MAX_S} s"
    load = f" with {payload_kg:.3g} kg payload" if payload_kg else ""
    parts = []
    for i in bad:
        j, m = robot.joints[i], robot.motors[i]
        unit, speed_unit = ("N", "m/s") if j.type == "prismatic" else ("Nm", "rad/s")
        parts.append(
            f"{j.name} {sim.saturation_cause(i).replace('_', '-')} for {sim.sat_max[i]:.2f} s while "
            f"{sim.sat_where[i]}{load} (motor {j.motor_id}: {m.rated:.3g} {unit} rated, "
            f"{m.no_load_speed:.3g} {speed_unit} no-load; speed model {m.speed_model})"
        )
    return bad, worst, "; ".join(parts)


def _cause(sim: _Sim, saturated: list[int], moves: list[_MoveResult] = ()) -> str:
    if saturated:
        return sim.saturation_cause(saturated[0])  # the longest saturation
    if any(m.unreachable for m in moves):
        return "unreachable"
    return "compliance"


def _with_cause(msg: str, cause: str | None) -> str:
    return msg if cause is None else f"{msg} Cause: {cause}: {CAUSE_ADVICE[cause]}."


def _compliance_advice(robot: SimRobot, contrib: np.ndarray, steady: bool, control: str,
                       catalog: dict[str, Motor]) -> str:
    """Stiffness breakdown and the fix options, ranked by it.

    Scores (higher first): a stiffer motor on the joint that sags most (its share of the
    error); "pid" (share of the error on Dynamixel joints, x0.3 if the error is transient,
    since an integral only removes steady sag); shorter or lighter outer links (0.25) and
    bringing the load closer to the base (0.2), which lower every joint's load a little.
    """
    from taskbot.sim.motor import motor_model

    total = float(contrib.sum())
    share = contrib / total if total > 0 else np.zeros_like(contrib)
    order = np.argsort(-contrib)
    breakdown = ", ".join(
        f"{robot.joints[i].name} ({robot.motors[i].motor_id}, Kp {robot.motors[i].kp:.3g} Nm/rad) "
        f"{_mm(contrib[i])}" for i in order[:3] if contrib[i] > 0)
    top = int(order[0])
    options = []
    usable = [m for m in catalog.values() if m.supported and "position" in m.modes]
    stiffer = sorted((motor_model(m, control).kp, m.id) for m in usable
                     if robot.joints[top].type != "prismatic"
                     and motor_model(m, control).kp > robot.motors[top].kp * 1.01)
    if stiffer:
        names = ", ".join(f"{mid} (Kp {kp:.3g})" for kp, mid in stiffer[:2])
        options.append((share[top], f"a stiffer motor on {robot.joints[top].name}: {names}"))
    if control == "pd":
        dyn = sum(share[i] for i, m in enumerate(robot.motors) if m.kind == "dynamixel")
        if dyn > 0:
            options.append((dyn * (1.0 if steady else 0.3),
                            'control "pid" (integral action on the Dynamixel joints removes steady sag '
                            'over a few seconds; it does little for errors while moving)'))
    options.append((0.25, "shorter or lighter links outboard of the sagging joints"))
    options.append((0.2, "bringing the load closer to the base (shorter reach to the workspace)"))
    options.sort(key=lambda o: -o[0])
    ranked = "; ".join(f"{k}) {text}" for k, (_, text) in enumerate(options, 1))
    return f"Sag by joint position-loop stiffness: {breakdown}. Options, most promising first: {ranked}."


# --------------------------------------------------------------------------- tests


def _catalog(catalog: dict[str, Motor] | None) -> dict[str, Motor]:
    return catalog if catalog is not None else load_motor_catalog()


def _move_compliance(robot: SimRobot, moves: list[_MoveResult], spec: RobotSpec,
                     catalog: dict[str, Motor]) -> str:
    """Compliance advice for the worst failed move: steady sag if it didn't settle,
    otherwise the error while moving."""
    failed = [m for m in moves if m.failures and not m.unreachable]
    if not failed:
        return ""
    worst = max(failed, key=lambda m: max(m.final_error, m.peak_dev / PEAK_FACTOR) / m.tol)
    steady = worst.steady_fail
    contrib = worst.final_contrib if steady else worst.peak_contrib
    when = "at the end of" if steady else "at the worst moment of"
    return f"({when} the move to {worst.label}) " + _compliance_advice(
        robot, contrib, steady, spec.control, catalog)


def _move_failure_text(robot, sim, moves, spec, catalog, payload_kg) -> tuple[str, str]:
    """(extra message, cause) for a failed reach/carry test."""
    saturated, _, sat_text = _saturation(robot, sim, payload_kg)
    cause = _cause(sim, saturated, moves)
    if saturated:
        return f" {sat_text}.", cause
    if cause == "compliance":
        return " " + _move_compliance(robot, moves, spec, catalog), cause
    return "", cause


def reach_targets(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None) -> SimRun:
    catalog = _catalog(catalog)
    targets = [p for _, p in workspace_targets(req)]
    robot = build_scene(spec, catalog, payload_kg=0.0, markers=targets)
    sim, moves = _run_moves(robot, req, payload_mass_error=0.0)
    ok, text = _move_summary(moves)
    worst = max(m.final_error for m in moves)
    msg, cause = f"Without payload: {text}.", None
    if not ok:
        extra, cause = _move_failure_text(robot, sim, moves, spec, catalog, 0.0)
        msg += extra
    result = CheckResult(check="reach_targets", passed=ok, value=worst, limit=req.position_tolerance_m,
                         message=_with_cause(msg, cause), cause=cause)
    return SimRun("reach_targets", result, robot, np.array(sim.frames), **_view(req))


def _belief(req: Requirements, payload_mass_error: float) -> str:
    return (f"controller assumes {req.payload_kg * (1 - payload_mass_error):.3g} kg"
            if payload_mass_error else "controller knows the mass")


def _carry(spec, req, catalog, payload_mass_error) -> tuple[CheckResult, SimRobot, _Sim]:
    targets = [p for _, p in workspace_targets(req)]
    robot = build_scene(spec, catalog, payload_kg=req.payload_kg, markers=targets)
    sim, moves = _run_moves(robot, req, payload_mass_error)
    reach_ok, text = _move_summary(moves)
    saturated, worst_sat, sat_text = _saturation(robot, sim, req.payload_kg)
    ok = reach_ok and not saturated
    msg = f"With {req.payload_kg:.3g} kg payload ({_belief(req, payload_mass_error)}): {sat_text}; {text}."
    cause = None
    if not ok:
        extra, cause = _move_failure_text(robot, sim, moves, spec, catalog, req.payload_kg)
        if cause == "compliance":
            msg += extra
    result = CheckResult(check="carry_payload", passed=ok, value=worst_sat, limit=SATURATION_MAX_S,
                         message=_with_cause(msg, cause), cause=cause)
    return result, robot, sim


def carry_payload(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None,
                  payload_mass_error: float = DEFAULT_PAYLOAD_MASS_ERROR) -> SimRun:
    catalog = _catalog(catalog)
    result, robot, sim = _carry(spec, req, catalog, payload_mass_error)
    ideal = _carry(spec, req, catalog, 0.0)[0] if payload_mass_error else None
    return SimRun("carry_payload", result, robot, np.array(sim.frames), ideal=ideal, **_view(req))


def _worst_pose(robot: SimRobot, req: Requirements) -> tuple[str, np.ndarray, int, float]:
    """Among the IK solutions for the workspace targets, the pose where some joint needs
    the largest fraction of its rated torque to hold still with the (true) payload."""
    from taskbot.sim.control import gravity_torque

    data = mujoco.MjData(robot.model)
    best = ("home", robot.home(), 0, -1.0)
    for mv in plan_moves(robot, req):
        robot.set_pose(data, mv.q1)
        load = np.abs(gravity_torque(robot, data, robot.payload_kg)) / robot.torque_limit
        j = int(np.argmax(load))
        if load[j] > best[3]:
            best = (mv.label, mv.q1.copy(), j, float(load[j]))
    return best


def _hold(spec, req, catalog, payload_mass_error) -> tuple[CheckResult, SimRobot, _Sim]:
    tol = req.position_tolerance_m
    targets = [p for _, p in workspace_targets(req)]
    robot = build_scene(spec, catalog, payload_kg=req.payload_kg, markers=targets)
    label, pose, joint, load = _worst_pose(robot, req)
    sim = _Sim(robot, pose, payload_mass_error)
    sim.label = f"holding at {label}"
    commanded = robot.ee(sim.data)  # tool position of the commanded pose
    path = sim.run(HOLD_S, lambda t: (pose, np.zeros(robot.n)))
    deviation = np.linalg.norm(path - commanded, axis=1)
    peak, final = float(deviation.max()), float(deviation[-1])
    if peak == 0.0:
        raise SimMeasurementError(
            f"hold_still measured exactly zero deviation holding {label}: the tool didn't move "
            "at all, so the test isn't measuring anything (is the arm or payload missing?)")
    final_ok, peak_ok = final < tol, peak < PEAK_FACTOR * tol
    ok = final_ok and peak_ok
    saturated, _, sat_text = _saturation(robot, sim, req.payload_kg)
    cause = None if ok else _cause(sim, saturated)
    msg = (f"Holding the most loaded target pose ({label}; {robot.joints[joint].name} needs "
           f"{load:.0%} of its rated torque) with {req.payload_kg:.3g} kg payload "
           f"({_belief(req, payload_mass_error)}) for {HOLD_S:.0f} s: peak deviation {peak * 1e6:.0f} µm "
           f"(limit {PEAK_FACTOR * tol * 1e6:.0f} µm), final error {final * 1e6:.0f} µm "
           f"(limit {tol * 1e6:.0f} µm, the task tolerance).")
    if saturated:
        msg += f" {sat_text}."
    elif not ok:
        contrib = sim.lever_errors(pose) if not final_ok else sim.peak_contrib  # steady vs transient
        msg += " " + _compliance_advice(robot, contrib, not final_ok, spec.control, catalog)
    result = CheckResult(check="hold_still", passed=ok, value=final, limit=tol,
                         message=_with_cause(msg, cause), cause=cause)
    return result, robot, sim


def hold_still(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None,
               payload_mass_error: float = DEFAULT_PAYLOAD_MASS_ERROR) -> SimRun:
    catalog = _catalog(catalog)
    result, robot, sim = _hold(spec, req, catalog, payload_mass_error)
    ideal = _hold(spec, req, catalog, 0.0)[0] if payload_mass_error else None
    return SimRun("hold_still", result, robot, np.array(sim.frames), ideal=ideal, **_view(req))


SIM_TESTS = [reach_targets, carry_payload, hold_still]


def run_sim_tests(spec: RobotSpec, req: Requirements,
                  catalog: dict[str, Motor] | None = None) -> list[SimRun]:
    catalog = catalog if catalog is not None else load_motor_catalog()
    return [test(spec, req, catalog) for test in SIM_TESTS]


def controller_settings(robot: SimRobot) -> list[dict]:
    """Per-joint controller and motor-model settings, for the report."""
    rows = []
    for j, m in zip(robot.joints, robot.motors):
        rows.append({"joint": j.name, "motor": m.motor_id, "kind": m.kind, "kp": m.kp, "kd": m.kd,
                     "ki": m.ki, "back_emf_damping": m.damping, "rated": m.rated, "stall": m.stall,
                     "no_load_speed": m.no_load_speed, "gain_source": m.gain_source,
                     "speed_model": m.speed_model, "mode": m.mode})
    for name, m in zip(robot.wheel_names, robot.wheel_motors):
        rows.append({"joint": name, "motor": m.motor_id, "kind": m.kind, "kp": m.kp, "kd": m.kd,
                     "ki": m.ki, "back_emf_damping": m.damping, "rated": m.rated, "stall": m.stall,
                     "no_load_speed": m.no_load_speed, "gain_source": m.gain_source,
                     "speed_model": m.speed_model, "mode": m.mode})
    return rows
