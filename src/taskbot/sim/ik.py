"""Damped least-squares position IK on the end-effector site, within joint limits."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from taskbot.sim.model import SimRobot


@dataclass(frozen=True)
class IKResult:
    q: np.ndarray
    error_m: float
    converged: bool


def _descend(robot: SimRobot, data: mujoco.MjData, q: np.ndarray, target: np.ndarray,
             iters: int, tol: float, damping: float, max_step: float) -> tuple[np.ndarray, float]:
    model = robot.model
    jacp = np.zeros((3, model.nv))
    q = np.clip(q, robot.lower, robot.upper)
    err = np.inf
    for _ in range(iters):
        data.qpos[robot.qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)  # mj_jacSite needs cdof
        e = target - data.site_xpos[robot.site]
        err = float(np.linalg.norm(e))
        if err < tol:
            break
        mujoco.mj_jacSite(model, data, jacp, None, robot.site)
        J = jacp[:, robot.dadr]
        dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(3), e)
        step = np.max(np.abs(dq))
        if step > max_step:
            dq *= max_step / step
        q = np.clip(q + dq, robot.lower, robot.upper)  # respect joint limits every step
    return q, err


def solve_ik(
    robot: SimRobot,
    target: np.ndarray,
    q_init: np.ndarray,
    *,
    tol: float = 1e-3,
    iters: int = 200,
    damping: float = 0.02,
    max_step: float = 0.2,
    restarts: int = 8,
    seed: int = 0,
) -> IKResult:
    """Try from q_init first (keeps the solution near the current pose), then from
    seeded random configurations. Returns the best attempt even if it misses."""
    data = mujoco.MjData(robot.model)
    target = np.asarray(target, float)
    best_q, best_err = _descend(robot, data, q_init, target, iters, tol, damping, max_step)
    rng = np.random.default_rng(seed)
    lo, hi = robot.sample_bounds()
    for _ in range(restarts):
        if best_err < tol:
            break
        q, err = _descend(robot, data, rng.uniform(lo, hi), target, iters, tol, damping, max_step)
        if err < best_err:
            best_q, best_err = q, err
    return IKResult(best_q, best_err, best_err < tol)
