"""Joint position control with fixed, per-motor gains (see taskbot.sim.motor).

For each joint, the motor's own position loop plus a host-side gravity feedforward:

    desired total torque = Kp e + Kd (qd_ref - qd) + Ki ∫e dt + g_hat(q) - b qd

where Kp, Kd come from the motor (Dynamixel factory gains, or one conservative
hobby/stepper setting), Ki is zero unless the design chose control = "pid" (then
one fixed preset on Dynamixels; see taskbot.sim.motor), b qd is the motor's
back-EMF, and the result is limited
by the motor's torque-speed model. The velocity terms -b qd and -Kd qd are
applied by MuJoCo as joint damping (integrated implicitly, so they are stable
however light the joint); only the rest is applied as a joint force. Nothing
here is tuned per test.

The integral runs continuously (it isn't reset between moves), integrates only
while the motor isn't saturated, and its contribution is clamped to ±rated torque
(anti-windup; the firmware's own anti-windup isn't documented, so this is a
modelling choice).

g_hat(q) is MuJoCo's qfrc_bias plus the payload's weight mapped through the
end-effector Jacobian (qfrc_bias alone doesn't include the welded payload). The
controller's idea of the payload mass can be wrong: with payload_mass_error = 0.2
it believes the payload is 20% lighter than it is, as a real controller would.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from taskbot.sim.model import SimRobot

DEFAULT_PAYLOAD_MASS_ERROR = 0.2


@dataclass
class ControlOutput:
    tau: np.ndarray            # total arm joint torque (applied minus back-EMF)
    saturated: np.ndarray      # bool per arm joint: wanted more torque than available
    speed_limited: np.ndarray  # bool per arm joint: ...and the limit was low because of speed


def gravity_torque(robot: SimRobot, data: mujoco.MjData, payload_kg: float,
                   jacp: np.ndarray | None = None) -> np.ndarray:
    """Torque each arm joint needs to hold the current pose (at rest) with this payload.
    Needs kinematics and qfrc_bias computed for the current state."""
    g = data.qfrc_bias[robot.dadr].copy()
    if payload_kg > 0:
        jacp = np.zeros((3, robot.model.nv)) if jacp is None else jacp
        mujoco.mj_jacSite(robot.model, data, jacp, None, robot.site)
        g += jacp[:, robot.dadr].T @ (-payload_kg * robot.model.opt.gravity)
    return g


class MotorController:
    def __init__(self, robot: SimRobot, payload_mass_error: float = DEFAULT_PAYLOAD_MASS_ERROR):
        self.robot = robot
        self.payload_mass_error = payload_mass_error
        self.believed_payload_kg = robot.payload_kg * (1.0 - payload_mass_error)
        self._jacp = np.zeros((3, robot.model.nv))
        self._integral = np.zeros(robot.n)  # Ki ∫e dt, as torque

    def compute(self, data: mujoco.MjData, q_ref: np.ndarray, qd_ref: np.ndarray) -> ControlOutput:
        """Needs qfrc_bias for the current state: call after mj_step1."""
        r = self.robot
        g_hat = gravity_torque(r, data, self.believed_payload_kg, self._jacp)
        q, qd = data.qpos[r.qadr], data.qvel[r.dadr]
        n = r.n
        tau, sat, slow = np.zeros(n), np.zeros(n, bool), np.zeros(n, bool)
        dt = r.model.opt.timestep
        for i, m in enumerate(r.motors):
            e = q_ref[i] - q[i]
            desired = (m.kp * e + m.kd * (qd_ref[i] - qd[i]) + self._integral[i] + g_hat[i]
                       - m.damping * qd[i])
            applied, tau[i], sat[i], slow[i] = m.command(desired, qd[i])
            data.qfrc_applied[r.dadr[i]] = applied
            if m.ki and not sat[i]:
                self._integral[i] = np.clip(self._integral[i] + m.ki * e * dt, -m.rated, m.rated)

        for k, m in enumerate(r.aux_motors):  # fingers hold closed with the gripper motor
            qa, da = r.aux_qadr[k], r.aux_dadr[k]
            w = data.qvel[da]
            desired = m.kp * (0.0 - data.qpos[qa]) - m.kd * w + data.qfrc_bias[da] - m.damping * w
            data.qfrc_applied[da] = m.command(desired, w)[0]
        return ControlOutput(tau, sat, slow)


def step(robot: SimRobot, data: mujoco.MjData, controller: MotorController,
         q_ref: np.ndarray, qd_ref: np.ndarray) -> ControlOutput:
    """One physics step with the control law evaluated on the current state."""
    mujoco.mj_step1(robot.model, data)
    out = controller.compute(data, q_ref, qd_ref)
    mujoco.mj_step2(robot.model, data)
    return out
