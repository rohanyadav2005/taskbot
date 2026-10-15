"""Joint position control: PID + gravity compensation, limited by each motor's torque-speed line.

    tau = clip(Kp e + Kd (qd_ref - qd) + (Kp / Ti) ∫e + g_hat(q), available torque)

Available torque follows a linear torque-speed line from rated torque at rest
to zero at the no-load speed: rated * max(0, 1 - |w| / w0). That limits torque
that drives the joint the way it is already turning. Torque that opposes the
motion (braking) is limited to rated torque: a motor brakes at least as hard as
it holds, and capping braking on the same line would let a fast-falling arm run
away with no torque at all.

Gains come from the inertia each joint currently sees (Kp = I wn², Kd = 2 zeta I wn),
so light wrist joints and heavy shoulder joints get the same closed-loop bandwidth.
I is the effective inertia 1 / (M^-1)_ii of the arm, plus the payload as a point mass
at the tool (m |J_i|²), since the payload is a separate welded body that M lacks.
Because the weld is slightly compliant, at high frequency a joint feels only the
arm's own inertia, so both gains are capped at the explicit-integration stability
limit for that inertia; otherwise a heavy payload on a light wrist makes the loop chatter.

g_hat(q) is MuJoCo's qfrc_bias plus the payload's weight mapped through the
end-effector Jacobian (qfrc_bias alone doesn't include the welded payload). The
controller's idea of the payload mass can be wrong: with payload_mass_error = 0.2
it believes the payload is 20% lighter than it is, as a real controller would.

Tuning: 8 Hz bandwidth and a 0.3 s integral time. With a 20% mass error, a 4 Hz
PD loop sags ~1 cm even on strong motors (the error is the missing torque / Kp,
not a lack of torque); 8 Hz is closer to a hobby/Dynamixel position loop (an
XM430 at default gains is roughly 19 N·m/rad). The integral term (conditional
integration, clamped at rated torque, as anti-windup) removes the remaining
steady offset, as a servo's PID does.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from taskbot.sim.model import SimRobot

DEFAULT_BANDWIDTH_HZ = 8.0
DEFAULT_PAYLOAD_MASS_ERROR = 0.2
DEFAULT_INTEGRAL_TIME_S = 0.3
MIN_INERTIA = 1e-6  # kg·m² / kg, so a near-massless joint still gets a positive gain
# Discrete stability margins for gains on an inertia I with timestep dt.
KD_MAX_FRACTION = 0.5    # Kd <= 0.5 * I / dt
KP_MAX_FRACTION = 0.25   # Kp <= 0.25 * I / dt²


def available_torque(raw: np.ndarray, speed: np.ndarray, rated: np.ndarray,
                     no_load_speed: np.ndarray) -> np.ndarray:
    """Torque limit for a requested torque `raw` at joint speed `speed`."""
    motoring = raw * speed > 0
    on_line = rated * np.maximum(0.0, 1.0 - np.abs(speed) / no_load_speed)
    return np.where(motoring, on_line, rated)


@dataclass
class ControlOutput:
    tau: np.ndarray        # applied (clipped) arm torques
    limit: np.ndarray      # torque that was available at this speed
    saturated: np.ndarray  # bool per arm joint: wanted more than was available


class PDGravityController:
    def __init__(self, robot: SimRobot, bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ,
                 zeta: float = 1.0, payload_mass_error: float = DEFAULT_PAYLOAD_MASS_ERROR,
                 integral_time_s: float | None = DEFAULT_INTEGRAL_TIME_S):
        self.robot = robot
        self.wn = 2 * np.pi * bandwidth_hz
        self.zeta = zeta
        self.payload_mass_error = payload_mass_error
        self.integral_time_s = integral_time_s
        self._integral = np.zeros(robot.n)  # integral term, as torque
        self.believed_payload_kg = robot.payload_kg * (1.0 - payload_mass_error)
        self._jacp = np.zeros((3, robot.model.nv))
        self._mass = np.zeros((robot.model.nv, robot.model.nv))
        self._payload_force = -self.believed_payload_kg * robot.model.opt.gravity

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
            inertia = arm_inertia + self.believed_payload_kg * np.sum(J * J, axis=0)

        dt = r.model.opt.timestep
        kp = np.minimum(inertia * self.wn**2, KP_MAX_FRACTION * arm_inertia / dt**2)
        kd = np.minimum(2 * self.zeta * inertia * self.wn, KD_MAX_FRACTION * arm_inertia / dt)
        qd = data.qvel[r.dadr]
        err = q_ref - data.qpos[r.qadr]
        pd = kp * err + kd * (qd_ref - qd) + gravity
        raw = pd + self._integral
        limit = available_torque(raw, qd, r.torque_limit, r.no_load_speed)
        tau = np.clip(raw, -limit, limit)
        if self.integral_time_s:
            # Anti-windup: integrate only while the motor isn't saturated, and never let
            # the integral alone ask for more than rated torque.
            free = np.abs(raw) <= limit
            self._integral = np.where(free, self._integral + kp * err * dt / self.integral_time_s,
                                      self._integral)
            self._integral = np.clip(self._integral, -r.torque_limit, r.torque_limit)
        data.qfrc_applied[r.dadr] = tau

        if len(r.aux_dadr):  # fingers: hold position with the gripper motor's force
            ref = np.zeros(len(r.aux_dadr)) if aux_ref is None else aux_ref
            m = self._inertia(data, r.aux_dadr)
            aqd = data.qvel[r.aux_dadr]
            aux = (m * self.wn**2 * (ref - data.qpos[r.aux_qadr]) - 2 * self.zeta * m * self.wn * aqd
                   + data.qfrc_bias[r.aux_dadr])
            alim = available_torque(aux, aqd, r.aux_limit, r.aux_no_load_speed)
            data.qfrc_applied[r.aux_dadr] = np.clip(aux, -alim, alim)
        return ControlOutput(tau, limit, np.abs(raw) > limit + 1e-12)


def step(robot: SimRobot, data: mujoco.MjData, controller: PDGravityController,
         q_ref: np.ndarray, qd_ref: np.ndarray) -> ControlOutput:
    """One physics step with the control law evaluated on the current state."""
    mujoco.mj_step1(robot.model, data)
    out = controller.compute(data, q_ref, qd_ref)
    mujoco.mj_step2(robot.model, data)
    return out
