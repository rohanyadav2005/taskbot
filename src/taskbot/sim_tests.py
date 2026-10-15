"""Simulation tests generated from the Requirements.

    reach_targets  move to the workspace centre and its 8 corners
    carry_payload  the same moves with the payload welded to the gripper
    hold_still     hold the most heavily loaded target pose, with payload, for 5 s

Every test returns a SimRun whose `.result` is a CheckResult, so it plugs into
the repair loop next to the static checks. Each move is a minimum-jerk joint
trajectory solved by IK and tracked by the PD + gravity controller with torques
clipped to the motors' rated values.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from taskbot.checks import CheckResult
from taskbot.requirements import Requirements
from taskbot.schema import Motor, RobotSpec, load_motor_catalog
from taskbot.sim.control import PDGravityController, step
from taskbot.sim.ik import solve_ik
from taskbot.sim.model import SimRobot, build_scene
from taskbot.sim.record import FPS

POSITION_TOL_M = 0.01
MOVE_S = 2.0
SETTLE_S = 1.0           # hold after each move before measuring the final error
SATURATION_MAX_S = 0.2
HOLD_S = 5.0
DRIFT_MAX_M = 0.005


@dataclass
class SimRun:
    name: str
    result: CheckResult
    robot: SimRobot                                   # scene used, for rendering
    frames: np.ndarray = field(repr=False)            # qpos at FPS, shape (n, nq)
    look_at: tuple[float, float, float] = (0.0, 0.0, 0.15)
    camera_distance: float = 1.0


def workspace_targets(req: Requirements) -> list[tuple[str, np.ndarray]]:
    ws = req.workspace
    centre = np.array([(ws.x_min + ws.x_max) / 2, (ws.y_min + ws.y_max) / 2, (ws.z_min + ws.z_max) / 2])
    targets = [("centre", centre)]
    seen = set()
    for x in (ws.x_min, ws.x_max):
        for y in (ws.y_min, ws.y_max):
            for z in (ws.z_min, ws.z_max):
                if (x, y, z) not in seen:
                    seen.add((x, y, z))
                    targets.append((f"corner ({x:.3g}, {y:.3g}, {z:.3g})", np.array([x, y, z])))
    return targets


def min_jerk(q0: np.ndarray, q1: np.ndarray, duration: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of a minimum-jerk path from q0 to q1 at time t."""
    s = np.clip(t / duration, 0.0, 1.0)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration if t < duration else 0.0
    return q0 + (q1 - q0) * pos, (q1 - q0) * vel


class _Sim:
    """Steps the scene under the controller and records frames and saturation."""

    def __init__(self, robot: SimRobot, q_start: np.ndarray):
        self.robot = robot
        self.data = mujoco.MjData(robot.model)
        robot.set_pose(self.data, q_start)
        self.ctrl = PDGravityController(robot)
        self.dt = robot.model.opt.timestep
        self.frame_every = max(1, round(1 / (FPS * self.dt)))
        self.steps = 0
        self.frames: list[np.ndarray] = [self.data.qpos.copy()]
        self.sat_run = np.zeros(robot.n)        # current consecutive saturated time per joint
        self.sat_max = np.zeros(robot.n)        # longest saturated stretch per joint
        self.sat_where: list[str] = [""] * robot.n
        self.label = ""

    def run(self, seconds: float, ref) -> list[np.ndarray]:
        """Step for `seconds`, with ref(t) -> (q_ref, qd_ref). Returns tool positions."""
        tool = []
        for k in range(round(seconds / self.dt)):
            q_ref, qd_ref = ref(k * self.dt)
            out = step(self.robot, self.data, self.ctrl, q_ref, qd_ref)
            self.sat_run = np.where(out.saturated, self.sat_run + self.dt, 0.0)
            longer = self.sat_run > self.sat_max
            for i in np.flatnonzero(longer):
                self.sat_where[i] = self.label
            self.sat_max = np.maximum(self.sat_max, self.sat_run)
            self.steps += 1
            if self.steps % self.frame_every == 0:
                self.frames.append(self.data.qpos.copy())
            tool.append(self.robot.ee(self.data))
        return tool

    def q(self) -> np.ndarray:
        return self.data.qpos[self.robot.qadr].copy()


@dataclass
class _Move:
    label: str
    target: np.ndarray
    ik_error: float
    final_error: float
    settle_s: float | None  # time from move start until the error stays under tolerance


def _do_moves(robot: SimRobot, req: Requirements) -> tuple[_Sim, list[_Move]]:
    sim = _Sim(robot, robot.home())
    moves = []
    for i, (label, target) in enumerate(workspace_targets(req)):
        q0 = sim.q()
        ik = solve_ik(robot, target, q0, seed=i)
        sim.label = f"moving to {label}"
        path = sim.run(MOVE_S + SETTLE_S, lambda t: min_jerk(q0, ik.q, MOVE_S, t))
        errors = np.linalg.norm(np.array(path) - target, axis=1)
        outside = np.flatnonzero(errors >= POSITION_TOL_M)
        if errors[-1] >= POSITION_TOL_M:
            settle = None
        else:
            settle = 0.0 if not len(outside) else (outside[-1] + 1) * sim.dt
        moves.append(_Move(label, target, ik.error_m, float(errors[-1]), settle))
    return sim, moves


def _view(req: Requirements) -> dict:
    ws = req.workspace
    reach = max(req.reach_m, abs(ws.x_max), abs(ws.x_min), abs(ws.y_max), abs(ws.y_min))
    return {"look_at": (0.0, 0.0, max(ws.z_max, 0.1) / 2), "camera_distance": max(0.6, 3.0 * reach)}


def _move_summary(moves: list[_Move]) -> tuple[bool, str]:
    failed = [m for m in moves if m.final_error >= POSITION_TOL_M]
    settles = [m.settle_s for m in moves if m.settle_s is not None]
    text = f"{len(moves) - len(failed)}/{len(moves)} targets within {POSITION_TOL_M * 100:.0f} cm"
    if settles:
        text += f" (settling {min(settles):.2f}–{max(settles):.2f} s after move start)"
    if failed:
        parts = []
        for m in failed:
            p = f"{m.label} final error {m.final_error * 100:.1f} cm"
            if m.ik_error >= POSITION_TOL_M:
                p += f" (unreachable: IK gets no closer than {m.ik_error * 100:.1f} cm)"
            parts.append(p)
        text += "; failed: " + "; ".join(parts)
    return not failed, text


def _saturation_text(robot: SimRobot, sim: _Sim, payload_kg: float) -> tuple[bool, float, str]:
    worst = float(sim.sat_max.max(initial=0.0))
    bad = [i for i in range(robot.n) if sim.sat_max[i] > SATURATION_MAX_S]
    if not bad:
        return True, worst, f"no joint saturated for more than {SATURATION_MAX_S} s"
    parts = []
    for i in sorted(bad, key=lambda i: -sim.sat_max[i]):
        j = robot.joints[i]
        unit = "N" if j.type == "prismatic" else "Nm"
        parts.append(
            f"{j.name} saturated at its rated {robot.torque_limit[i]:.3g} {unit} ({j.motor_id}) "
            f"for {sim.sat_max[i]:.2f} s while {sim.sat_where[i]} with {payload_kg:.3g} kg payload"
        )
    return False, worst, "; ".join(parts) + ". Use a stronger motor on these joints or lighten the outer links."


# --------------------------------------------------------------------------- tests


def reach_targets(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None) -> SimRun:
    targets = [p for _, p in workspace_targets(req)]
    robot = build_scene(spec, catalog, payload_kg=0.0, markers=targets)
    sim, moves = _do_moves(robot, req)
    ok, text = _move_summary(moves)
    worst = max(m.final_error for m in moves)
    msg = f"Without payload: {text}."
    if not ok:
        sat_ok, _, sat_text = _saturation_text(robot, sim, 0.0)
        if not sat_ok:
            msg += " " + sat_text.replace(" with 0 kg payload", "")
        msg += " Lengthen links or add a joint for unreachable targets; stronger motors for ones it can't settle on."
    result = CheckResult(check="reach_targets", passed=ok, value=worst, limit=POSITION_TOL_M, message=msg)
    return SimRun("reach_targets", result, robot, np.array(sim.frames), **_view(req))


def carry_payload(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None) -> SimRun:
    targets = [p for _, p in workspace_targets(req)]
    robot = build_scene(spec, catalog, payload_kg=req.payload_kg, markers=targets)
    sim, moves = _do_moves(robot, req)
    reach_ok, text = _move_summary(moves)
    sat_ok, worst_sat, sat_text = _saturation_text(robot, sim, req.payload_kg)
    msg = f"With {req.payload_kg:.3g} kg payload: {sat_text}; {text}."
    result = CheckResult(check="carry_payload", passed=reach_ok and sat_ok, value=worst_sat,
                         limit=SATURATION_MAX_S, message=msg)
    return SimRun("carry_payload", result, robot, np.array(sim.frames), **_view(req))


def _worst_pose(robot: SimRobot, req: Requirements) -> tuple[str, np.ndarray, int, float]:
    """Among the IK solutions for the workspace targets, the pose where some joint needs
    the largest fraction of its rated torque to hold still with the payload."""
    data = mujoco.MjData(robot.model)
    ctrl = PDGravityController(robot)
    best = ("home", robot.home(), 0, -1.0)
    q = robot.home()
    for i, (label, target) in enumerate(workspace_targets(req)):
        q = solve_ik(robot, target, q, seed=i).q
        robot.set_pose(data, q)
        out = ctrl.compute(data, q, np.zeros(robot.n))  # at rest: pure gravity load
        load = np.abs(out.tau) / robot.torque_limit
        load = np.where(out.saturated, np.inf, load)
        j = int(np.argmax(load))
        if load[j] > best[3]:
            best = (label, q.copy(), j, float(load[j]))
    return best


def hold_still(spec: RobotSpec, req: Requirements, catalog: dict[str, Motor] | None = None) -> SimRun:
    targets = [p for _, p in workspace_targets(req)]
    robot = build_scene(spec, catalog, payload_kg=req.payload_kg, markers=targets)
    label, pose, joint, load = _worst_pose(robot, req)
    sim = _Sim(robot, pose)
    sim.label = f"holding at {label}"
    start = robot.ee(sim.data)
    path = sim.run(HOLD_S, lambda t: (pose, np.zeros(robot.n)))
    drift = float(np.max(np.linalg.norm(np.array(path) - start, axis=1)))
    ok = drift < DRIFT_MAX_M
    load_txt = "more than 100%" if not np.isfinite(load) else f"{load:.0%}"
    msg = (f"Holding the most loaded target pose ({label}; {robot.joints[joint].name} needs "
           f"{load_txt} of its rated torque) with {req.payload_kg:.3g} kg payload for {HOLD_S:.0f} s: "
           f"tool drifted {drift * 1000:.1f} mm (limit {DRIFT_MAX_M * 1000:.0f} mm).")
    sat_ok, _, sat_text = _saturation_text(robot, sim, req.payload_kg)
    if not sat_ok:
        msg += " " + sat_text
    elif not ok:
        msg += f" Use a stronger motor on {robot.joints[joint].name}."
    result = CheckResult(check="hold_still", passed=ok, value=drift, limit=DRIFT_MAX_M, message=msg)
    return SimRun("hold_still", result, robot, np.array(sim.frames), **_view(req))


SIM_TESTS = [reach_targets, carry_payload, hold_still]


def run_sim_tests(spec: RobotSpec, req: Requirements,
                  catalog: dict[str, Motor] | None = None) -> list[SimRun]:
    catalog = catalog if catalog is not None else load_motor_catalog()
    return [test(spec, req, catalog) for test in SIM_TESTS]
