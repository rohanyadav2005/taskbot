"""Simulation tests for wheeled bases, generated from the Requirements.

    drive_straight          drive 1 m (by wheel odometry) at speed_m_s; heading error < 5°,
                            distance error < 5% (the gap is wheel slip, reported)
    turn_in_place           turn 90° on the spot; error < 5°
    incline                 drive 0.5 m up max_incline_deg without stalling or sliding back
    drive_with_arm_extended mobile manipulators: drive 0.5 m with the arm extended to each
                            side it can reach and the payload in the gripper; roll/pitch < 20°
                            and the tool stays within the task tolerance of its pose

Wheels run each motor's fixed velocity preset (taskbot.sim.motor, velocity mode) under the
same torque-speed limit and back-EMF as the arm joints. Speeds ramp at 0.5 m/s². Turns
close the loop on the chassis yaw, as a robot would with a gyro; straight driving is open
loop on matched wheel speeds, stopping on wheel odometry. Arm joints hold their pose with
the arm controller (taskbot.sim.control).
"""

from __future__ import annotations


import mujoco
import numpy as np

from taskbot.checks import CAUSE_ADVICE, CheckResult
from taskbot.mobile_checks import ACCEL_M_S2
from taskbot.requirements import Requirements
from taskbot.schema import ArmChain, Motor, RobotSpec, WheeledBase, load_motor_catalog
from taskbot.sim.control import DEFAULT_PAYLOAD_MASS_ERROR, MotorController
from taskbot.sim.model import SimRobot, build_scene
from taskbot.sim.record import FPS
from taskbot.sim_tests import SATURATION_MAX_S, SimRun

HEADING_MAX_DEG = 5.0
DISTANCE_ERROR_MAX = 0.05
TURN_ERROR_MAX_DEG = 5.0
TILT_MAX_DEG = 20.0
STRAIGHT_M = 1.0
INCLINE_M = 0.5
ARM_DRIVE_M = 0.5
TURN_GAIN = 2.0            # yaw-rate command per rad of heading error (1/s)
TURN_RATE_MAX = 1.0        # rad/s
SETTLE_S = 1.0


class WheelController:
    """Per wheel: desired total torque = Kv (w_ref - w) + Ki ∫(w_ref - w) dt - b w, through
    the motor's torque-speed limit (MotorModel.command). The integral is clamped to ±rated
    and frozen while the motor is saturated."""

    def __init__(self, robot: SimRobot):
        self.robot = robot
        self._integral = np.zeros(len(robot.wheel_names))
        self.sat_time = np.zeros(len(robot.wheel_names))
        self.slow_time = np.zeros(len(robot.wheel_names))

    def apply(self, data: mujoco.MjData, w_ref: np.ndarray) -> None:
        r = self.robot
        dt = r.model.opt.timestep
        for i, m in enumerate(r.wheel_motors):
            w = data.qvel[r.wheel_dadr[i]]
            e = w_ref[i] - w
            desired = m.kp * e + self._integral[i] - m.damping * w
            applied, _, sat, slow = m.command(desired, w)
            data.ctrl[r.wheel_act[i]] = applied
            if sat:
                self.sat_time[i] += dt
                self.slow_time[i] += dt * slow
            else:
                self._integral[i] = np.clip(self._integral[i] + m.ki * e * dt, -m.rated, m.rated)


class _MobileSim:
    def __init__(self, robot: SimRobot, arm_pose: np.ndarray | None = None,
                 payload_mass_error: float = DEFAULT_PAYLOAD_MASS_ERROR):
        self.robot = robot
        self.data = mujoco.MjData(robot.model)
        self.arm_pose = arm_pose if arm_pose is not None else robot.home()
        robot.set_pose(self.data, self.arm_pose)
        self.wheels = WheelController(robot)
        self.arm = MotorController(robot, payload_mass_error=payload_mass_error) if robot.n else None
        self.dt = robot.model.opt.timestep
        self.frame_every = max(1, round(1 / (FPS * self.dt)))
        self.steps = 0
        self.frames = [self.data.qpos.copy()]
        self.max_tilt = 0.0

    def step(self, w_ref: np.ndarray) -> None:
        m, d = self.robot.model, self.data
        mujoco.mj_step1(m, d)
        self.wheels.apply(d, w_ref)
        if self.arm is not None:
            self.arm.compute(d, self.arm_pose, np.zeros(self.robot.n))
        mujoco.mj_step2(m, d)
        self.steps += 1
        if self.steps % self.frame_every == 0:
            self.frames.append(d.qpos.copy())
        self.max_tilt = max(self.max_tilt, self.tilt_deg())

    # ---- chassis state (what an IMU + ground truth would give)
    def position(self) -> np.ndarray:
        return self.data.xpos[self.robot.chassis_body].copy()

    def yaw(self) -> float:
        R = self.data.xmat[self.robot.chassis_body].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def tilt_deg(self) -> float:
        """Angle between the chassis z axis and the world z axis (roll and pitch combined)."""
        R = self.data.xmat[self.robot.chassis_body].reshape(3, 3)
        return float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1))))

    def odometry(self) -> float:
        """Mean distance rolled by the wheels (radius x angle)."""
        r = self.robot
        return float(np.mean(self.data.qpos[r.wheel_qadr])) * self.radius

    radius = 0.0

    def tool_in_chassis(self) -> np.ndarray:
        r, d = self.robot, self.data
        R = d.xmat[r.chassis_body].reshape(3, 3)
        return R.T @ (d.site_xpos[r.site] - d.xpos[r.chassis_body])


def _sim(spec, req, catalog, *, arm_pose=None, incline=0.0, gripper_payload=0.0):
    base: WheeledBase = spec.base
    robot = build_scene(spec, catalog, payload_kg=gripper_payload, floating=True, terrain=req.terrain,
                        base_payload_kg=req.base_payload_kg or 0.0, incline_deg=incline)
    sim = _MobileSim(robot, arm_pose)
    sim.radius = base.wheel_radius_m
    for _ in range(round(0.3 / sim.dt)):  # let it settle onto its wheels
        sim.step(np.zeros(len(robot.wheel_names)))
    return sim


def _drive(sim: _MobileSim, distance: float, speed: float, timeout: float,
           watch=None) -> tuple[float, bool]:
    """Ramp up to `speed` at ACCEL_M_S2, stop (ramping down) when odometry says `distance`.
    Returns (time taken, reached). `watch(sim)` is called every step."""
    r = sim.radius
    start = sim.odometry()
    v, t = 0.0, 0.0
    while t < timeout:
        travelled = sim.odometry() - start
        remaining = distance - travelled
        if remaining <= 0:
            break
        v_stop = np.sqrt(2 * ACCEL_M_S2 * max(remaining, 0.0))
        v = min(speed, v + ACCEL_M_S2 * sim.dt, v_stop)
        sim.step(np.full(len(sim.robot.wheel_names), v / r))
        if watch:
            watch(sim)
        t += sim.dt
    for _ in range(round(SETTLE_S / sim.dt)):
        sim.step(np.zeros(len(sim.robot.wheel_names)))
        if watch:
            watch(sim)
    return t, t < timeout


def _wheel_saturation(sim: _MobileSim) -> tuple[str | None, str]:
    worst = int(np.argmax(sim.wheels.sat_time))
    t = float(sim.wheels.sat_time[worst])
    if t <= SATURATION_MAX_S:
        return None, ""
    cause = "speed_limited" if sim.wheels.slow_time[worst] > 0.5 * t else "saturated"
    return cause, (f" Wheel {sim.robot.wheel_names[worst]} ({sim.robot.wheel_motors[worst].motor_id}) "
                   f"was {cause.replace('_', '-')} for {t:.2f} s.")


def _run(name: str, result: CheckResult, sim: _MobileSim) -> SimRun:
    return SimRun(name, result, sim.robot, np.array(sim.frames), look_at=(0.5, 0.0, 0.1),
                  camera_distance=1.8, track_body=sim.robot.chassis_body)


# The arm advice in CAUSE_ADVICE ("shorter or lighter outer links") doesn't fit a wheel.
WHEEL_ADVICE = {
    "saturated": ("a wheel motor hit its torque limit; use a stronger wheel motor, smaller wheels, "
                  "more driven wheels (four_wheel), or a lighter robot"),
    "speed_limited": ("a wheel motor ran out of speed; use a faster wheel motor, or larger wheels "
                      "if it has torque to spare"),
}


def _msg(text: str, cause: str | None) -> str:
    if cause is None:
        return text
    return f"{text} Cause: {cause}: {WHEEL_ADVICE.get(cause, CAUSE_ADVICE[cause])}."


# --------------------------------------------------------------------------- tests


def drive_straight(spec: RobotSpec, req: Requirements, catalog=None) -> SimRun:
    sim = _sim(spec, req, catalog)
    p0, yaw0 = sim.position(), sim.yaw()
    nominal = STRAIGHT_M / req.speed_m_s + req.speed_m_s / ACCEL_M_S2
    _, reached = _drive(sim, STRAIGHT_M, req.speed_m_s, timeout=3 * nominal + 3)
    actual = float(np.linalg.norm((sim.position() - p0)[:2]))
    heading = abs(np.degrees(np.angle(np.exp(1j * (sim.yaw() - yaw0)))))
    dist_err = abs(actual - STRAIGHT_M) / STRAIGHT_M
    slip = (sim.odometry() - actual) / max(sim.odometry(), 1e-9)
    ok = reached and heading < HEADING_MAX_DEG and dist_err < DISTANCE_ERROR_MAX
    sat_cause, sat_text = _wheel_saturation(sim)
    cause = None if ok else (sat_cause or "slip")
    text = (f"Drove {STRAIGHT_M:g} m by wheel odometry at {req.speed_m_s:g} m/s on {req.terrain}"
            f"{'' if reached else ' (gave up: the odometry never got there)'}: actually moved {actual:.3f} m "
            f"(error {dist_err:.1%}, limit {DISTANCE_ERROR_MAX:.0%}; wheel slip {slip:.1%}), heading "
            f"error {heading:.2f}° (limit {HEADING_MAX_DEG:g}°).{sat_text}")
    result = CheckResult(check="drive_straight", passed=ok, value=dist_err, limit=DISTANCE_ERROR_MAX,
                         message=_msg(text, cause), cause=cause)
    return _run("drive_straight", result, sim)


def turn_in_place(spec: RobotSpec, req: Requirements, catalog=None) -> SimRun:
    base: WheeledBase = spec.base
    sim = _sim(spec, req, catalog)
    yaw0 = sim.yaw()
    target = np.pi / 2
    half_track = base.track_width_m / 2
    rate = 0.0
    for _ in range(round(8.0 / sim.dt)):
        err = float(np.angle(np.exp(1j * (yaw0 + target - sim.yaw()))))
        rate_cmd = float(np.clip(TURN_GAIN * err, -TURN_RATE_MAX, TURN_RATE_MAX))
        # ramp the wheel rim speed at 0.5 m/s²
        rate += np.clip(rate_cmd - rate, -ACCEL_M_S2 / half_track * sim.dt, ACCEL_M_S2 / half_track * sim.dt)
        w = -sim.robot.wheel_side * rate * half_track / base.wheel_radius_m
        sim.step(w)
    for _ in range(round(SETTLE_S / sim.dt)):
        sim.step(np.zeros(len(sim.robot.wheel_names)))
    turned = np.degrees(np.angle(np.exp(1j * (sim.yaw() - yaw0))))
    err = abs(turned - 90.0)
    ok = err < TURN_ERROR_MAX_DEG
    sat_cause, sat_text = _wheel_saturation(sim)
    cause = None if ok else (sat_cause or "slip")
    text = (f"Turned on the spot towards 90° with yaw feedback ({base.drive} drive"
            f"{', skid steering' if base.drive == 'four_wheel' else ''}): ended at {turned:.2f}° "
            f"(error {err:.2f}°, limit {TURN_ERROR_MAX_DEG:g}°).{sat_text}")
    result = CheckResult(check="turn_in_place", passed=ok, value=err, limit=TURN_ERROR_MAX_DEG,
                         message=_msg(text, cause), cause=cause)
    return _run("turn_in_place", result, sim)


def incline(spec: RobotSpec, req: Requirements, catalog=None) -> SimRun:
    sim = _sim(spec, req, catalog, incline=req.max_incline_deg)
    p0 = sim.position()
    track = {"min_x": 0.0, "last_x": 0.0}

    def watch(s: _MobileSim):
        dx = float(s.position()[0] - p0[0])
        track["min_x"] = min(track["min_x"], dx)
        track["last_x"] = dx

    nominal = INCLINE_M / req.speed_m_s + req.speed_m_s / ACCEL_M_S2
    _, reached = _drive(sim, INCLINE_M, req.speed_m_s, timeout=3 * nominal + 3, watch=watch)
    climbed = float(sim.position()[0] - p0[0])
    slid_back = track["min_x"] < -0.01
    slip = (sim.odometry() - climbed) / max(sim.odometry(), 1e-9)
    ok = reached and not slid_back  # distance accuracy is drive_straight's job; slip is reported
    sat_cause, sat_text = _wheel_saturation(sim)
    cause = None if ok else (sat_cause or "slip")
    progress = (f"climbed {climbed:.3f} m (wheel slip {slip:.1%}, reported only)" if climbed > 0
                else f"went {-climbed:.3f} m backwards down the slope instead")
    text = (f"Drove up a {req.max_incline_deg:g}° slope on {req.terrain} for {INCLINE_M:g} m by odometry: "
            f"{progress}; slid back at most "
            f"{max(0.0, -track['min_x']) * 1000:.0f} mm (limit 10 mm)"
            f"{'' if reached else '; stalled: never covered the distance'}.{sat_text}")
    result = CheckResult(check="incline", passed=ok, value=-track["min_x"], limit=0.01,
                         message=_msg(text, cause), cause=cause)
    return _run("incline", result, sim)


def _extended_poses(spec: RobotSpec, robot: SimRobot) -> list[tuple[str, np.ndarray]]:
    """Arm extended horizontally (the first tilting joint at ±90° within its limits), pointed
    front / left / back / right with the first joint if it is a vertical-axis (yaw) joint;
    otherwise only the direction the arm can point."""
    arm = next(m for m in spec.modules if isinstance(m, ArmChain))
    q = robot.home().copy()
    joints = robot.joints
    tilt = next((i for i, j in enumerate(joints) if j.axis in ("x", "y") and j.type != "prismatic"), None)
    data = mujoco.MjData(robot.model)
    if tilt is not None:
        best = None
        for target in (np.pi / 2, -np.pi / 2):
            qq = q.copy()
            qq[tilt] = np.clip(target, robot.lower[tilt], robot.upper[tilt])
            robot.set_pose(data, qq)
            reach = np.linalg.norm(robot.ee(data)[:2] - np.array(arm.mount_xy_m or (0, 0)))
            if best is None or reach > best[0]:
                best = (reach, qq)
        q = best[1]
    yaw = 0 if joints and joints[0].axis == "z" and joints[0].type != "prismatic" else None
    if yaw is None:
        return [("as built", q)]
    poses = []
    robot.set_pose(data, q)
    d = robot.ee(data)[:2] - np.array(arm.mount_xy_m or (0, 0))
    natural = np.arctan2(d[1], d[0])
    for label, ang in [("front", 0.0), ("left", np.pi / 2), ("back", np.pi), ("right", -np.pi / 2)]:
        qq = q.copy()
        qq[yaw] = float(np.angle(np.exp(1j * (ang - natural))))
        if robot.lower[yaw] <= qq[yaw] <= robot.upper[yaw]:
            poses.append((label, qq))
    return poses


def drive_with_arm_extended(spec: RobotSpec, req: Requirements, catalog=None) -> SimRun:
    tol = req.position_tolerance_m
    probe = build_scene(spec, catalog)
    results, last = [], None
    for label, pose in _extended_poses(spec, probe):
        sim = _sim(spec, req, catalog, arm_pose=pose, gripper_payload=req.payload_kg)
        for _ in range(round(SETTLE_S / sim.dt)):
            sim.step(np.zeros(len(sim.robot.wheel_names)))
        ref = sim.tool_in_chassis()
        dev = {"max": 0.0}

        def watch(s: _MobileSim):
            dev["max"] = max(dev["max"], float(np.linalg.norm(s.tool_in_chassis() - ref)))

        nominal = ARM_DRIVE_M / req.speed_m_s + req.speed_m_s / ACCEL_M_S2
        _, reached = _drive(sim, ARM_DRIVE_M, req.speed_m_s, timeout=3 * nominal + 3, watch=watch)
        results.append((label, sim.max_tilt, dev["max"], reached, sim))
        last = sim
    tipped = [r for r in results if r[1] >= TILT_MAX_DEG]
    sagged = [r for r in results if r[2] >= tol]
    stuck = [r for r in results if not r[3]]
    ok = not tipped and not sagged and not stuck
    parts = "; ".join(f"arm {lab}: max tilt {tilt:.1f}°, tool moved {d * 1000:.1f} mm"
                      for lab, tilt, d, _, _ in results)
    text = (f"Drove {ARM_DRIVE_M:g} m at {req.speed_m_s:g} m/s with the arm extended and "
            f"{req.payload_kg:g} kg in the gripper (limits: tilt {TILT_MAX_DEG:g}°, tool within "
            f"{tol * 1000:g} mm of its pose relative to the chassis): {parts}.")
    cause = None
    if tipped:
        cause = "unstable"
        text += (" Fixes: a wider track or longer wheelbase, a lower or heavier base, moving the "
                 "arm mount towards the middle, or a shorter reach.")
    elif stuck:
        cause = _wheel_saturation(results[0][4])[0] or "slip"
    elif sagged:
        cause = "compliance"
    result = CheckResult(check="drive_with_arm_extended", passed=ok,
                         value=max(r[1] for r in results), limit=TILT_MAX_DEG,
                         message=_msg(text, cause), cause=cause)
    return _run("drive_with_arm_extended", result, last)


MOBILE_SIM_TESTS = [drive_straight, turn_in_place, incline]


def run_mobile_sim_tests(spec: RobotSpec, req: Requirements,
                         catalog: dict[str, Motor] | None = None) -> list[SimRun]:
    catalog = catalog if catalog is not None else load_motor_catalog()
    runs = [t(spec, req, catalog) for t in MOBILE_SIM_TESTS]
    if spec.has_arm and req.has_arm:
        runs.append(drive_with_arm_extended(spec, req, catalog))
    return runs
