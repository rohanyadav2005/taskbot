"""Actuator model: torque-speed limits and position-loop gains derived from the catalog.

Torque limits (total torque at the joint, sign convention: w = joint speed)

    driving (torque in the direction of motion, or at rest):
        servos/Dynamixels: min(rated, stall * max(0, 1 - |w| / w0))
        steppers: the pull-out curve if the catalog has one, else rated up to
                  w0/2 and a straight line to 0 at w0 (an approximation)
    braking (torque against the motion): rated
        A DC motor brakes at least as hard as it holds (back-EMF adds to the
        drive voltage when braking), so the rated (thermal/current) limit is the
        binding one. Putting braking on the driving line instead would leave a
        fast-falling arm with no torque at all to stop it.
        One exception, which is physics: if the joint is driven beyond
        w0 * (1 + rated/stall) (e.g. an arm collapsing under a weak motor), back-EMF
        alone brakes harder than rated (b|w| - stall > rated) and no drive voltage
        can prevent it.

Gains (fixed per motor, never tuned per test)

    dynamixel:  the factory position-loop gains from the control table. The
                firmware computes PWM = KPP*e + KPD*de/dt with KPP = P/128,
                KPD = D/16 (e in encoder pulses, 4096/rev; de/dt per ms), and the
                torque at a PWM fraction u (885 = 100%) is stall * u at rest. So
                Kp = (P/128) * (stall/885) * (4096/2π)          [N·m/rad]
                Kd = (D/16) * (stall/885) * (4096/2π) / 1000   [N·m·s/rad]
                (I gain is 0 by default for every catalog Dynamixel.)
    hobby_servo / stepper: one conservative setting: full stall torque at 20°
                of error (Kp = stall / 0.349 rad), no D, no I.

    Every motor also has back-EMF damping b = stall / w0 [N·m·s/rad]: a motor
    driven by a voltage loses stall/w0 of torque per rad/s, which is the
    torque-speed line above. MuJoCo applies it as joint damping (implicitly, so
    it is stable at any size); the controller adds b*w back to its command so
    that the total joint torque obeys the limits above. Steppers aren't
    voltage-driven DC motors; they get b = rated / w0 as an approximation, small
    enough that their speed curve stays reachable without exceeding stall.

Neither the gearbox's reflected rotor inertia nor friction is modelled.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from taskbot.schema import Motor

PWM_FULL = 885                     # Dynamixel X-series PWM limit = 100% duty
PULSES_PER_RAD = 4096 / (2 * np.pi)
HOBBY_FULL_TORQUE_ERROR_RAD = np.radians(20.0)
STEPPER_FLAT_FRACTION = 0.5


@dataclass(frozen=True)
class MotorModel:
    motor_id: str
    kind: str
    rated: float          # N·m (or N)
    stall: float          # N·m (or N)
    no_load_speed: float  # rad/s (or m/s)
    kp: float             # N·m/rad (or N/m)
    kd: float             # N·m·s/rad
    ki: float             # N·m/(rad·s)
    damping: float        # back-EMF, N·m·s/rad
    curve: tuple[tuple[float, float], ...] | None = None
    gain_source: str = ""
    speed_model: str = ""

    def drive_limit(self, w: float) -> float:
        """Most torque available in the direction of motion at speed w."""
        w = abs(w)
        if self.kind == "stepper":
            if self.curve:
                speeds, torques = zip(*self.curve)
                return float(min(self.rated, np.interp(w, speeds, torques, right=0.0)))
            knee = STEPPER_FLAT_FRACTION * self.no_load_speed
            if w <= knee:
                return self.rated
            return self.rated * max(0.0, (self.no_load_speed - w) / (self.no_load_speed - knee))
        return min(self.rated, self.stall * max(0.0, 1.0 - w / self.no_load_speed))

    def limits(self, torque: float, w: float) -> tuple[float, bool]:
        """(cap on |torque|, whether that cap is below rated because of speed)."""
        if torque * w < 0:  # braking
            return self.rated, False
        cap = self.drive_limit(w)
        return cap, cap < self.rated * (1 - 1e-9)

    @property
    def joint_damping(self) -> float:
        """Velocity damping MuJoCo applies (implicitly) at the joint: back-EMF plus the
        -Kd*w part of the position loop's D term."""
        return self.damping + self.kd

    def command(self, desired: float, w: float) -> tuple[float, float, bool, bool]:
        """Turn a desired total joint torque into what the motor actually does.

        Returns (applied, total, saturated, speed_limited). `applied` goes to MuJoCo,
        which adds -joint_damping*w itself; `total` = applied - joint_damping*w. The
        electrical part (applied - kd*w, i.e. stall * PWM fraction) is limited to ±stall.
        """
        cap, slowed = self.limits(desired, w)
        total = float(np.clip(desired, -cap, cap))
        electrical = float(np.clip(total + self.damping * w, -self.stall, self.stall))
        applied = electrical + self.kd * w
        total = electrical - self.damping * w
        over = abs(desired) > cap + 1e-12
        return applied, total, over, slowed and over

    def linear(self, radius: float) -> MotorModel:
        """The same motor driving a prismatic joint through a pinion of this radius."""
        return replace(
            self, rated=self.rated / radius, stall=self.stall / radius,
            no_load_speed=self.no_load_speed * radius, kp=self.kp / radius**2,
            kd=self.kd / radius**2, ki=self.ki / radius**2, damping=self.damping / radius**2,
            curve=None if self.curve is None else tuple((w * radius, t / radius) for w, t in self.curve),
        )


def motor_model(motor: Motor) -> MotorModel:
    torque_per_pwm = motor.stall_torque_nm / PWM_FULL
    if motor.kind == "dynamixel":
        g = motor.dynamixel_gains
        kp = g.p / 128 * torque_per_pwm * PULSES_PER_RAD
        kd = g.d / 16 * torque_per_pwm * PULSES_PER_RAD / 1000
        ki = 0.0
        if g.i:
            raise NotImplementedError(f"{motor.id}: non-zero Dynamixel I gain isn't modelled")
        source = f"Dynamixel factory gains P={g.p} I={g.i} D={g.d}"
    else:
        kp = motor.stall_torque_nm / HOBBY_FULL_TORQUE_ERROR_RAD
        kd = ki = 0.0
        source = f"conservative {motor.kind.replace('_', ' ')} setting: stall torque at 20° error"
    if motor.kind == "stepper":
        speed_model = ("pull-out curve" if motor.pull_out_curve else
                       "approx.: rated to 50% of no-load speed, then linear to 0")
    else:
        speed_model = "min(rated, stall·(1 − |w|/w0))"
    return MotorModel(
        motor_id=motor.id, kind=motor.kind, rated=motor.rated_torque_nm, stall=motor.stall_torque_nm,
        no_load_speed=motor.no_load_speed_rad_s, kp=kp, kd=kd, ki=ki,
        damping=(motor.rated_torque_nm if motor.kind == "stepper" else motor.stall_torque_nm)
        / motor.no_load_speed_rad_s,
        curve=tuple(map(tuple, motor.pull_out_curve)) if motor.pull_out_curve else None,
        gain_source=source, speed_model=speed_model,
    )
