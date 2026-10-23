"""Can any design from the motor catalog meet these requirements? Checked between the
requirements step and the design step, with bounds no design can beat:

    arm types   holding torque at the shoulder with weightless links and motors: payload at
                the needed reach plus the lightest catalog gripper of the required kind just
                behind it, against the strongest supported motor's usable torque (80% of
                rated). static_torque uses the same floor (holding_torque_floor_nm).
    mobile      traction: the slope, rolling resistance and 0.5 m/s² acceleration need a
                friction coefficient the terrain can't give, whatever the robot weighs.
                Drive: the payloads plus four of a wheel motor (weightless chassis and
                wheels), on four driven wheels of the best radius, at the required speed.

The workspace-within-reach rule is a Requirements validator (requirements.farthest_corner).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from taskbot.build_urdf import TERRAIN, _gripper_drafts, tool_length_m
from taskbot.schema import Gripper, Motor, supported_motors
from taskbot.sim.motor import motor_model

if TYPE_CHECKING:
    from taskbot.requirements import Requirements

G = 9.81
TORQUE_SAFETY_FRACTION = 0.8      # same as checks.TORQUE_SAFETY_FRACTION
ACCEL_M_S2 = 0.5                  # same as mobile_checks.ACCEL_M_S2
SPEED_FRACTION = 0.8              # same as mobile_checks.SPEED_FRACTION
MAX_DRIVEN_WHEELS = 4
LIGHTEST_GRIPPER_OPENING_M = 0.02  # the smallest gripper TaskBot would build
MAX_WHEEL_RADIUS_M = 0.15


@dataclass
class Feasibility:
    problems: list[str] = field(default_factory=list)  # why no design can pass, with numbers
    options: list[str] = field(default_factory=list)   # what would make it possible

    @property
    def ok(self) -> bool:
        return not self.problems


# --------------------------------------------------------------------------- arm


def strongest_position_motor(catalog: dict[str, Motor]) -> Motor:
    return max((m for m in supported_motors(catalog).values() if "position" in m.modes),
               key=lambda m: m.rated_torque_nm)


def lightest_gripper(kind: str | None, catalog: dict[str, Motor]) -> tuple[float, float, str]:
    """(mass, tool length, description) of the lightest gripper of this kind TaskBot can build:
    LIGHTEST_GRIPPER_OPENING_M wide, with the lightest supported position motor."""
    if kind is None:
        return 0.0, 0.0, "no gripper"
    motor = min((m for m in supported_motors(catalog).values() if "position" in m.modes),
                key=lambda m: m.mass_kg)
    g = Gripper(name="g", gripper_type=kind, max_opening_m=LIGHTEST_GRIPPER_OPENING_M, grip_force_n=1.0,
                motor_id=motor.id if kind == "parallel" else None)
    mass = sum(b.mass for d in _gripper_drafts(g, motor if kind == "parallel" else None) for b in d.bodies)
    return mass, tool_length_m(g), f"lightest {kind} gripper ({mass:.3g} kg)"


def holding_torque_floor_nm(payload_kg: float, gripper_kg: float, tool_m: float, lever_m: float) -> float:
    """Gravity torque at the shoulder of a weightless arm holding the payload lever_m out,
    with the gripper's mass a tool length nearer."""
    return G * (payload_kg * lever_m + gripper_kg * max(lever_m - tool_m, 0.0))


def needed_reach_m(req: Requirements) -> float:
    """Farthest horizontal distance the task needs: reach_m or the farthest workspace corner."""
    ws = req.workspace
    corners = max(np.hypot(x, y) for x in (ws.x_min, ws.x_max) for y in (ws.y_min, ws.y_max)) if ws else 0.0
    return max(req.reach_m or 0.0, float(corners))


def _arm(req: Requirements, catalog: dict[str, Motor], out: Feasibility) -> None:
    strongest = strongest_position_motor(catalog)
    usable = TORQUE_SAFETY_FRACTION * strongest.rated_torque_nm
    m_g, tool, gripper = lightest_gripper(req.gripper_kind if req.gripper_needed else None, catalog)
    lever = needed_reach_m(req)
    need = holding_torque_floor_nm(req.payload_kg, m_g, tool, lever)
    if need <= usable:
        return
    out.problems.append(
        f"Holding {req.payload_kg:.3g} kg at {lever:.3g} m needs at least {need:.3g} Nm at the shoulder, "
        f"even with weightless links and motors and the {gripper}; the strongest motor, {strongest.id}, "
        f"gives {usable:.3g} Nm usable ({TORQUE_SAFETY_FRACTION:.0%} of {strongest.rated_torque_nm:.3g} Nm rated).")
    max_payload = usable / (G * lever) - m_g * max(lever - tool, 0.0) / lever
    max_reach = (usable / G + m_g * tool) / (req.payload_kg + m_g) if req.payload_kg + m_g > 0 else float("inf")
    if max_payload > 0:
        out.options.append(f"At {lever:.3g} m, a payload under {max_payload:.3g} kg (margin included) could work "
                           "with a very light arm; a real arm's own weight leaves less.")
    else:
        out.options.append(f"At {lever:.3g} m, even the {gripper} alone is too heavy.")
    out.options.append(f"With {req.payload_kg:.3g} kg, a reach under {max_reach:.3g} m could work with a very "
                       "light arm: put the robot closer to the work.")


# --------------------------------------------------------------------------- mobile


def _mobile(req: Requirements, catalog: dict[str, Motor], out: Feasibility) -> None:
    th = np.radians(req.max_incline_deg)
    terrain = TERRAIN[req.terrain]
    mu, crr = terrain["mu"], terrain["crr"]
    need_mu = (np.sin(th) + crr * np.cos(th) + ACCEL_M_S2 / G) / np.cos(th)
    if need_mu >= mu:
        out.problems.append(
            f"Climbing {req.max_incline_deg:g}° on {req.terrain} while accelerating at {ACCEL_M_S2} m/s² needs a "
            f"friction coefficient of {need_mu:.2f} even with every wheel driven; {req.terrain} gives {mu} "
            "(build_urdf.TERRAIN, an estimate). No robot weight changes that.")
        out.options.append("A gentler max_incline_deg, or grippier terrain, if the task allows it.")
        return
    carried = (req.base_payload_kg or 0.0) + (req.payload_kg if req.has_arm else 0.0)
    best = None  # (most drive force, motor, radius)
    for m in supported_motors(catalog).values():
        if "velocity" not in m.modes:
            continue
        mm = motor_model(m, mode="velocity")
        r_min = req.speed_m_s / (SPEED_FRACTION * m.no_load_speed_rad_s)
        for r in np.linspace(r_min, max(r_min, MAX_WHEEL_RADIUS_M), 60):
            force = MAX_DRIVEN_WHEELS * mm.drive_limit(req.speed_m_s / r) / r
            mass = carried + MAX_DRIVEN_WHEELS * m.mass_kg
            per_kg = G * np.sin(th) + crr * G * np.cos(th) + ACCEL_M_S2
            if best is None or force - mass * per_kg > best[0]:
                best = (force - mass * per_kg, m, r, force, mass, per_kg)
    if best is None or best[0] >= 0:
        return
    _, m, r, force, mass, per_kg = best
    out.problems.append(
        f"Moving {mass:.3g} kg ({carried:.3g} kg of payload plus four {m.id} wheel motors, weightless chassis "
        f"and wheels) at {req.speed_m_s:g} m/s up {req.max_incline_deg:g}° on {req.terrain} needs "
        f"{mass * per_kg:.3g} N of drive force; the best wheel motor, {m.id}, on four driven wheels of "
        f"{r * 100:.3g} cm radius gives {force:.3g} N.")
    spare = force / per_kg - MAX_DRIVEN_WHEELS * m.mass_kg
    out.options.append(f"At this speed and slope, under {max(spare, 0):.3g} kg of payload could work."
                       if spare > 0 else "A lower speed_m_s or gentler max_incline_deg.")


def check_feasibility(req: Requirements, catalog: dict[str, Motor]) -> Feasibility:
    out = Feasibility()
    if req.has_arm:
        _arm(req, catalog, out)
    if req.is_mobile:
        _mobile(req, catalog, out)
    return out


def retry_message(task: str, req: Requirements, f: Feasibility) -> str:
    """Sent with the task to the requirements step once when the gate fails."""
    return (f"{task}\n\nYour requirements for this task were:\n{req.model_dump_json(exclude_none=True)}\n"
            "No robot built from the available motors can meet them:\n"
            + "\n".join(f"- {p}" for p in f.problems)
            + "\nRe-check reach, height and payload against the task; keep them realistic for the described "
              "scene, don't just shrink them to fit. Return the complete corrected requirements as JSON.")
