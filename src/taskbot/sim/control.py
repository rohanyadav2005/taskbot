"""Joint position control: PD + gravity compensation, clipped to rated torque.

    tau = clip(Kp (q_ref - q) + Kd (qd_ref - qd) + g(q), ±rated)

Gains come from the inertia each joint currently sees (Kp = I wn², Kd = 2 zeta I wn),
so light wrist joints and heavy shoulder joints get the same closed-loop bandwidth.
I is the effective inertia 1 / (M^-1)_ii of the arm, plus the payload as a point mass
at the tool (m |J_i|²), since the payload is a separate welded body that M lacks.
Because the weld is slightly compliant, at high frequency a joint feels only the
arm's own inertia, so both gains are capped at the explicit-integration stability
limit for that inertia; otherwise a heavy payload on a light wrist makes the loop chatter.

g(q) is MuJoCo's qfrc_bias plus the known payload's weight mapped through the
end-effector Jacobian, since qfrc_bias alone doesn't include the welded payload.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from taskbot.sim.model import SimRobot

DEFAULT_BANDWIDTH_HZ = 4.0
MIN_INERTIA = 1e-6  # kg·m² / kg, so a near-massless joint still gets a positive gain
# Discrete stability margins for gains on an inertia I with timestep dt.
KD_MAX_FRACTION = 0.5    # Kd <= 0.5 * I / dt
KP_MAX_FRACTION = 0.25   # Kp <= 0.25 * I / dt²


@dataclass
class ControlOutput:
    tau: np.ndarray        # applied (clipped) arm torques
    saturated: np.ndarray  # bool per arm joint: wanted more than the rated torque


class PDGravityController:
    def __init__(self, robot: SimRobot, bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ, zeta: float = 1.0):
        self.robot = robot
        self.wn = 2 * np.pi * bandwidth_hz
        self.zeta = zeta
        self._jacp = np.zeros((3, robot.model.nv))
        self._mass = np.zeros((robot.model.nv, robot.model.nv))
        g = robot.model.opt.gravity
        self._payload_force = -robot.payload_kg * g  # force that holds the payload up

    def _inertia(self, data: mujoco.MjData, dofs: np.ndarray) -> np.ndarray:
        """Effective inertia 1 / (M^-1)_ii for these dofs."""
        mujoco.mj_fullM(self.robot.model, data, self._mass)
        inv_diag = np.diag(np.linalg.inv(self._mass))[dofs]
        return np.maximum(1.0 / inv_diag, MIN_INERTIA)

    def compute(self, data: mujoco.MjData, q_ref: np.ndarray, qd_ref: np.ndarray,
                aux_ref: np.ndarray | None = None) -> ControlOutput:
        """Needs the mass matrix and qfrc_bias for the current state: call after mj_step1."""
        r = self.robot
        arm_inertia = self._inertia(data, r.dadr)
        inertia = arm_inertia
        gravity = data.qfrc_bias[r.dadr].copy()
        if r.payload_kg > 0:
            mujoco.mj_jacSite(r.model, data, self._jacp, None, r.site)
            J = self._jacp[:, r.dadr]
            gravity += J.T @ self._payload_force
            inertia = arm_inertia + r.payload_kg * np.sum(J * J, axis=0)

        dt = r.model.opt.timestep
        kp = np.minimum(inertia * self.wn**2, KP_MAX_FRACTION * arm_inertia / dt**2)
        kd = np.minimum(2 * self.zeta * inertia * self.wn, KD_MAX_FRACTION * arm_inertia / dt)
        raw = kp * (q_ref - data.qpos[r.qadr]) + kd * (qd_ref - data.qvel[r.dadr]) + gravity
        tau = np.clip(raw, -r.torque_limit, r.torque_limit)
        data.qfrc_applied[r.dadr] = tau

        if len(r.aux_dadr):  # fingers: hold position with the gripper motor's force
            ref = np.zeros(len(r.aux_dadr)) if aux_ref is None else aux_ref
            m = self._inertia(data, r.aux_dadr)
            aux = (m * self.wn**2 * (ref - data.qpos[r.aux_qadr])
                   - 2 * self.zeta * m * self.wn * data.qvel[r.aux_dadr]
                   + data.qfrc_bias[r.aux_dadr])
            data.qfrc_applied[r.aux_dadr] = np.clip(aux, -r.aux_limit, r.aux_limit)
        return ControlOutput(tau, np.abs(raw) > r.torque_limit + 1e-12)


def step(robot: SimRobot, data: mujoco.MjData, controller: PDGravityController,
         q_ref: np.ndarray, qd_ref: np.ndarray) -> ControlOutput:
    """One physics step with the control law evaluated on the current state."""
    mujoco.mj_step1(robot.model, data)
    out = controller.compute(data, q_ref, qd_ref)
    mujoco.mj_step2(robot.model, data)
    return out
