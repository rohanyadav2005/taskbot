"""Move planning shared by the simulation tests and the joint_speed check."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from taskbot.sim.ik import solve_ik
from taskbot.sim.model import SimRobot

if TYPE_CHECKING:
    from taskbot.requirements import Requirements

MIN_MOVE_S = 2.0
SPEED_FRACTION = 0.8      # plan moves so no joint exceeds 80% of its no-load speed
MIN_JERK_PEAK = 1.875     # peak velocity of a min-jerk move = 1.875 * distance / duration


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


@dataclass(frozen=True)
class PlannedMove:
    label: str
    target: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    ik_error: float

    def duration(self, no_load_speed: np.ndarray) -> float:
        """Shortest min-jerk duration (>= 2 s) keeping every joint under 80% of no-load speed."""
        need = MIN_JERK_PEAK * np.abs(self.q1 - self.q0) / (SPEED_FRACTION * no_load_speed)
        return max(MIN_MOVE_S, float(need.max(initial=0.0)))

    def peak_speed(self, duration: float) -> np.ndarray:
        return MIN_JERK_PEAK * np.abs(self.q1 - self.q0) / duration


def plan_moves(robot: SimRobot, req: Requirements) -> list[PlannedMove]:
    """Home -> centre -> each corner; IK seeded from the previous pose."""
    q = robot.home()
    moves = []
    for i, (label, target) in enumerate(workspace_targets(req)):
        ik = solve_ik(robot, target, q, seed=i)
        moves.append(PlannedMove(label, target, q, ik.q, ik.error_m))
        q = ik.q
    return moves


def min_jerk(q0: np.ndarray, q1: np.ndarray, duration: float, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Position and velocity of a minimum-jerk path from q0 to q1 at time t."""
    s = np.clip(t / duration, 0.0, 1.0)
    pos = 10 * s**3 - 15 * s**4 + 6 * s**5
    vel = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration if t < duration else 0.0
    return q0 + (q1 - q0) * pos, (q1 - q0) * vel
