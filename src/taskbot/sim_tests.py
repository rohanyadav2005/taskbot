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

POSITION_TOL_M = 0.01
OVERSHOOT_MAX_M = 0.02
SETTLE_DEADLINE_S = 1.0   # error must be under tolerance by move duration + this
SETTLE_CONFIRM_S = 0.5    # ...and stay there for this long after the deadline
SATURATION_MAX_S = 0.2
HOLD_S = 5.0
HOLD_FINAL_MAX_M = 0.001
HOLD_PEAK_MAX_M = 0.005


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

    def run(self, seconds: float, ref) -> np.ndarray:
        """Step for `seconds`, with ref(t) -> (q_ref, qd_ref). Returns tool positions."""
        tool = []
        for k in range(round(seconds / self.dt)):
            q_ref, qd_ref = ref(k * self.dt)
            out = step(self.robot, self.data, self.ctrl, q_ref, qd_ref)
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
        return np.array(tool)

    def q(self) -> np.ndarray:
        return self.data.qpos[self.robot.qadr].copy()

    def saturation_cause(self, i: int) -> str:
        return "speed_limited" if self.sat_max_slow[i] > 0.5 * self.sat_max[i] else "saturated"


@dataclass
class _MoveResult:
    label: str
    ik_error: float
    duration: float
    final_error: float     # at duration + SETTLE_DEADLINE_S + SETTLE_CONFIRM_S
    overshoot: float       # furthest the tool went past the target along the approach line
    settle_s: float | None  # from move start until the error stays under tolerance

    @property
    def unreachable(self) -> bool:
        return self.ik_error >= POSITION_TOL_M

    @property
    def settled_in_time(self) -> bool:
        return self.settle_s is not None and self.settle_s <= self.duration + SETTLE_DEADLINE_S

    @property
    def failures(self) -> list[str]:
        out = []
        if self.unreachable:
            out.append(f"unreachable: IK gets no closer than {self.ik_error * 100:.1f} cm")
        if not self.settled_in_time:
            if self.settle_s is None:
                out.append(f"never settled within {POSITION_TOL_M * 100:.0f} cm "
                           f"(final error {self.final_error * 100:.1f} cm)")
            else:
                out.append(f"settled after {self.settle_s:.2f} s, later than the "
                           f"{self.duration + SETTLE_DEADLINE_S:.2f} s deadline")
        if self.overshoot > OVERSHOOT_MAX_M:
            out.append(f"overshot by {self.overshoot * 100:.1f} cm (limit {OVERSHOOT_MAX_M * 100:.0f} cm)")
        return out


def _run_moves(robot: SimRobot, req: Requirements, payload_mass_error: float) -> tuple[_Sim, list[_MoveResult]]:
    sim = _Sim(robot, robot.home(), payload_mass_error)
    results = []
    for mv in plan_moves(robot, req):
        T = mv.duration(robot.no_load_speed)
        start = sim.q()  # actual pose, which may differ from the plan if the last move failed
        sim.label = f"moving to {mv.label}"
        path = sim.run(T + SETTLE_DEADLINE_S + SETTLE_CONFIRM_S, lambda t: min_jerk(start, mv.q1, T, t))
        errors = np.linalg.norm(path - mv.target, axis=1)
        outside = np.flatnonzero(errors >= POSITION_TOL_M)
        if errors[-1] >= POSITION_TOL_M:
            settle = None
        else:
            settle = 0.0 if not len(outside) else (outside[-1] + 1) * sim.dt
        approach = mv.target - path[0]
        dist = np.linalg.norm(approach)
        overshoot = float(np.max((path - mv.target) @ (approach / dist), initial=0.0)) if dist > 1e-9 else 0.0
        results.append(_MoveResult(mv.label, mv.ik_error, T, float(errors[-1]), max(overshoot, 0.0), settle))
    return sim, results


def _view(req: Requirements) -> dict:
    ws = req.workspace
    reach = max(req.reach_m, abs(ws.x_max), abs(ws.x_min), abs(ws.y_max), abs(ws.y_min))
    return {"look_at": (0.0, 0.0, max(ws.z_max, 0.1) / 2), "camera_distance": max(0.6, 3.0 * reach)}


def _move_summary(moves: list[_MoveResult]) -> tuple[bool, str]:
    failed = [m for m in moves if m.failures]
    durations = [m.duration for m in moves]
    settles = [m.settle_s for m in moves if m.settle_s is not None]
    text = (f"{len(moves) - len(failed)}/{len(moves)} targets reached within {POSITION_TOL_M * 100:.0f} cm "
            f"by move end + {SETTLE_DEADLINE_S:.0f} s with overshoot <= {OVERSHOOT_MAX_M * 100:.0f} cm; "
            f"move durations {min(durations):.2f}–{max(durations):.2f} s (planned for "
            f"<= {SPEED_FRACTION:.0%} of no-load speed, min {MIN_MOVE_S:.0f} s)")
    if settles:
        text += f"; settling {min(settles):.2f}–{max(settles):.2f} s after move start"
    text += f"; max overshoot {max(m.overshoot for m in moves) * 100:.1f} cm"
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


# --------------------------------------------------------------------------- tests


def reach_targets(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None) -> SimRun:
    targets = [p for _, p in workspace_targets(req)]
    robot = build_scene(spec, catalog, payload_kg=0.0, markers=targets)
    sim, moves = _run_moves(robot, req, payload_mass_error=0.0)
    ok, text = _move_summary(moves)
    worst = max(m.final_error for m in moves)
    msg, cause = f"Without payload: {text}.", None
    if not ok:
        saturated, _, sat_text = _saturation(robot, sim, 0.0)
        if saturated:
            msg += f" {sat_text}."
        cause = _cause(sim, saturated, moves)
    result = CheckResult(check="reach_targets", passed=ok, value=worst, limit=POSITION_TOL_M,
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
    cause = None if ok else _cause(sim, saturated, moves)
    msg = f"With {req.payload_kg:.3g} kg payload ({_belief(req, payload_mass_error)}): {sat_text}; {text}."
    result = CheckResult(check="carry_payload", passed=ok, value=worst_sat, limit=SATURATION_MAX_S,
                         message=_with_cause(msg, cause), cause=cause)
    return result, robot, sim


def carry_payload(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None,
                  payload_mass_error: float = DEFAULT_PAYLOAD_MASS_ERROR) -> SimRun:
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
    ok = final < HOLD_FINAL_MAX_M and peak < HOLD_PEAK_MAX_M
    saturated, _, sat_text = _saturation(robot, sim, req.payload_kg)
    cause = None if ok else _cause(sim, saturated)
    msg = (f"Holding the most loaded target pose ({label}; {robot.joints[joint].name} needs "
           f"{load:.0%} of its rated torque) with {req.payload_kg:.3g} kg payload "
           f"({_belief(req, payload_mass_error)}) for {HOLD_S:.0f} s: peak deviation {peak * 1e6:.0f} µm "
           f"(limit {HOLD_PEAK_MAX_M * 1e6:.0f} µm), final error {final * 1e6:.0f} µm "
           f"(limit {HOLD_FINAL_MAX_M * 1e6:.0f} µm).")
    if saturated:
        msg += f" {sat_text}."
    elif not ok:
        msg += " " + _sag_breakdown(robot, sim, pose)
    result = CheckResult(check="hold_still", passed=ok, value=final, limit=HOLD_FINAL_MAX_M,
                         message=_with_cause(msg, cause), cause=cause)
    return result, robot, sim


def _sag_breakdown(robot: SimRobot, sim: _Sim, pose: np.ndarray) -> str:
    """Which joints' position loops let the tool sag, from final joint error x lever arm."""
    jacp = np.zeros((3, robot.model.nv))
    mujoco.mj_jacSite(robot.model, sim.data, jacp, None, robot.site)
    err = np.abs(sim.q() - pose)
    contrib = np.linalg.norm(jacp[:, robot.dadr], axis=0) * err
    order = np.argsort(-contrib)[:2]
    parts = [f"{robot.joints[i].name} ({robot.motors[i].motor_id}, Kp {robot.motors[i].kp:.3g} Nm/rad) "
             f"{contrib[i] * 1e6:.0f} µm" for i in order if contrib[i] > 0]
    return "Sag by joint position-loop stiffness: " + ", ".join(parts) + "."


def hold_still(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None,
               payload_mass_error: float = DEFAULT_PAYLOAD_MASS_ERROR) -> SimRun:
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
                     "speed_model": m.speed_model})
    return rows
