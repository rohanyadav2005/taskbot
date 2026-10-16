"""Robot description schema: the only thing the LLM is allowed to write.

Everything here is declarative. Geometry conventions, masses and inertias are
derived later by deterministic code (`taskbot.build_urdf`). SI units throughout.

Conventions the LLM needs to know (also stated in field descriptions):
  * Each link extends along its own +z axis from the joint that drives it.
    A box link's length is size_m[2]; a cylinder link's length is length_m.
  * The next joint sits at the tip of the previous link.
  * Modules are listed root-first: one fixed_base, then one or more
    arm_chain modules, then at most one gripper. They attach in order.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PositiveFloat,
    ValidationInfo,
    field_validator,
    model_validator,
)

NAME_PATTERN = r"^[A-Za-z][A-Za-z0-9_]*$"
Name = Annotated[str, Field(pattern=NAME_PATTERN, max_length=64)]

Material = Literal["pla", "aluminium", "steel"]
Vec3 = tuple[PositiveFloat, PositiveFloat, PositiveFloat]
Vec2 = tuple[PositiveFloat, PositiveFloat]
Vec2Signed = tuple[float, float]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- motors


class PositionGains(_Strict):
    """Raw position-loop gains as written in a Dynamixel X-series control table."""

    p: Annotated[int, Field(ge=0)]
    i: Annotated[int, Field(ge=0)]
    d: Annotated[int, Field(ge=0)]


class VelocityGains(_Strict):
    """Raw velocity-loop gains as written in a Dynamixel X-series control table."""

    p: Annotated[int, Field(ge=0)]
    i: Annotated[int, Field(ge=0)]


class Motor(_Strict):
    id: Name
    name: str = Field(min_length=1)
    kind: Literal["hobby_servo", "dynamixel", "stepper", "dc_gearmotor"] = Field(
        description="Decides the controller gains and the torque-speed model used in simulation."
    )
    modes: list[Literal["position", "velocity"]] = Field(
        min_length=1,
        description="Control modes this motor supports: arm joints need position, wheels velocity.",
    )
    voltage_v: PositiveFloat = Field(description="Supply voltage the torque/speed figures are for.")
    stall_torque_nm: PositiveFloat
    rated_torque_nm: PositiveFloat = Field(
        description="Continuous torque the motor can hold without overheating."
    )
    max_speed_rad_s: PositiveFloat = Field(description="Highest output speed the vendor quotes.")
    no_load_speed_rad_s: PositiveFloat = Field(
        description="Output speed at zero load; available torque falls linearly to 0 here."
    )
    mass_kg: PositiveFloat
    body_size_m: Vec3 = Field(description="Housing bounding box (x, y, z).")
    price_usd: Annotated[float, Field(ge=0)]
    dynamixel_gains: PositionGains | None = Field(
        default=None, description="Factory position P/I/D gains (Dynamixel only)."
    )
    dynamixel_velocity_gains: VelocityGains | None = Field(
        default=None, description="Factory velocity P/I gains (Dynamixel only)."
    )
    pull_out_curve: list[tuple[float, float]] | None = Field(
        default=None,
        description="Stepper only: (speed rad/s, torque N·m) points of the pull-out curve.",
    )
    sources: dict[str, str] = Field(
        default_factory=dict,
        description="Where each value came from, keyed by field name.",
    )
    supported: bool = Field(
        default=True, description="False: the simulator can't model this motor faithfully, so "
        "the designer isn't offered it.")
    unsupported_reason: str | None = None

    @model_validator(mode="after")
    def _unsupported_needs_reason(self) -> Motor:
        if not self.supported and not self.unsupported_reason:
            raise ValueError(f"motor {self.id}: unsupported motors need an unsupported_reason")
        return self

    @model_validator(mode="after")
    def _kind_fields(self) -> Motor:
        if (self.kind == "dynamixel") != (self.dynamixel_gains is not None):
            raise ValueError(f"motor {self.id}: dynamixel_gains is required for, and only for, kind 'dynamixel'")
        if self.kind == "dynamixel" and "velocity" in self.modes and self.dynamixel_velocity_gains is None:
            raise ValueError(f"motor {self.id}: a Dynamixel used in velocity mode needs dynamixel_velocity_gains")
        if self.pull_out_curve is not None and self.kind != "stepper":
            raise ValueError(f"motor {self.id}: pull_out_curve is only for steppers")
        return self

    @model_validator(mode="after")
    def _speeds_consistent(self) -> Motor:
        if self.max_speed_rad_s > self.no_load_speed_rad_s:
            raise ValueError(
                f"motor {self.id}: max_speed_rad_s ({self.max_speed_rad_s}) exceeds "
                f"no_load_speed_rad_s ({self.no_load_speed_rad_s})"
            )
        return self

    @model_validator(mode="after")
    def _rated_not_above_stall(self) -> Motor:
        if self.rated_torque_nm > self.stall_torque_nm:
            raise ValueError(
                f"motor {self.id}: rated_torque_nm ({self.rated_torque_nm}) "
                f"exceeds stall_torque_nm ({self.stall_torque_nm})"
            )
        return self


def supported_motors(catalog: dict[str, Motor]) -> dict[str, Motor]:
    """The motors a designer may choose from."""
    return {k: m for k, m in catalog.items() if m.supported}


@lru_cache(maxsize=1)
def load_motor_catalog() -> dict[str, Motor]:
    """Built-in motor catalog from taskbot/data/motors.json, keyed by id."""
    raw = json.loads(
        resources.files("taskbot").joinpath("data/motors.json").read_text("utf-8")
    )
    motors = [Motor.model_validate(m) for m in raw["motors"]]
    catalog = {m.id: m for m in motors}
    if len(catalog) != len(motors):
        raise ValueError("duplicate motor id in motors.json")
    return catalog


# --------------------------------------------------------------------------- parts


class Joint(_Strict):
    name: Name
    type: Literal["revolute", "continuous", "prismatic"]
    axis: Literal["x", "y", "z"]
    lower: float | None = Field(
        default=None,
        description="Lower limit: rad for revolute, m for prismatic. Omit for continuous.",
    )
    upper: float | None = Field(
        default=None,
        description="Upper limit: rad for revolute, m for prismatic. Omit for continuous.",
    )
    motor_id: Name

    @model_validator(mode="after")
    def _limits_match_type(self) -> Joint:
        has = (self.lower is not None, self.upper is not None)
        if self.type == "continuous":
            if any(has):
                raise ValueError(
                    f"joint '{self.name}' is continuous, so it must not set lower/upper; "
                    "remove them or change type to 'revolute'"
                )
            return self
        if not all(has):
            raise ValueError(
                f"joint '{self.name}' is {self.type} and needs both lower and upper limits"
            )
        if self.lower >= self.upper:
            raise ValueError(
                f"joint '{self.name}': lower ({self.lower}) must be less than upper ({self.upper})"
            )
        return self


class Link(_Strict):
    """A rigid link. Its mass is computed from shape, size and material."""

    name: Name
    shape: Literal["box", "cylinder"]
    size_m: Vec3 | None = Field(
        default=None, description="Box only: (x, y, z); z is the length along the link."
    )
    radius_m: PositiveFloat | None = Field(default=None, description="Cylinder only.")
    length_m: PositiveFloat | None = Field(
        default=None, description="Cylinder only: length along the link's z axis."
    )
    material: Material

    @model_validator(mode="after")
    def _dimensions_match_shape(self) -> Link:
        if self.shape == "box":
            if self.size_m is None or self.radius_m is not None or self.length_m is not None:
                raise ValueError(
                    f"link '{self.name}' is a box: give size_m=[x, y, z] and "
                    "no radius_m/length_m"
                )
        else:
            if self.radius_m is None or self.length_m is None or self.size_m is not None:
                raise ValueError(
                    f"link '{self.name}' is a cylinder: give radius_m and length_m "
                    "and no size_m"
                )
        return self

    @property
    def length(self) -> float:
        """Extent along the link's z axis; the next joint sits here."""
        return self.size_m[2] if self.shape == "box" else self.length_m


class Segment(_Strict):
    joint: Joint
    link: Link


# --------------------------------------------------------------------------- modules


class FixedBase(_Strict):
    kind: Literal["fixed_base"] = "fixed_base"
    name: Name
    footprint_m: Vec2 = Field(description="(x, y) size of the base box.")
    height_m: PositiveFloat = Field(description="The arm mounts on top, at this height.")
    material: Material = "aluminium"


class WheeledBase(_Strict):
    """A mobile chassis: a box on wheels. Its frame is on the ground under the chassis centre,
    x forward, z up. Wheel axles are at height wheel_radius_m; the chassis box sits with its
    bottom at half a wheel radius above the ground."""

    kind: Literal["wheeled_base"] = "wheeled_base"
    name: Name
    drive: Literal["differential", "four_wheel"] = Field(
        description="differential: two driven wheels (plus a caster); four_wheel: four driven "
        "wheels, steered by skidding (left and right sides at different speeds).")
    wheel_radius_m: PositiveFloat
    wheel_width_m: PositiveFloat
    track_width_m: PositiveFloat = Field(description="Distance between left and right wheel centres.")
    wheelbase_m: PositiveFloat = Field(
        description="four_wheel: distance between front and rear axles. differential: distance "
        "from the drive axle (at the front) to the caster (at the back).")
    chassis_size_m: Vec3 = Field(description="Chassis box (x length, y width, z height).")
    chassis_material: Material = "aluminium"
    wheel_motor_id: Name
    caster: bool = Field(default=False, description="Differential only: a ball caster at the back.")
    battery_mass_kg: PositiveFloat | None = Field(
        default=None, description="Omit to use the simple estimate (see battery_estimate_kg).")

    @model_validator(mode="after")
    def _drive_layout(self) -> WheeledBase:
        if self.drive == "four_wheel" and self.caster:
            raise ValueError(f"wheeled_base '{self.name}': caster is for differential drive only")
        if self.track_width_m <= self.wheel_width_m:
            raise ValueError(f"wheeled_base '{self.name}': track_width_m must exceed wheel_width_m")
        return self

    @property
    def driven_wheels(self) -> int:
        return 2 if self.drive == "differential" else 4

    def battery_estimate_kg(self) -> float:
        """0.25 kg of battery per pair of driven wheels: a rough 12 V, ~2 Ah pack per two
        small gearmotors. A planning estimate, not a sourced figure."""
        return 0.25 * self.driven_wheels / 2

    @property
    def battery_kg(self) -> float:
        return self.battery_mass_kg if self.battery_mass_kg is not None else self.battery_estimate_kg()

    @property
    def battery_note(self) -> str | None:
        if self.battery_mass_kg is not None:
            return None
        return (f"battery_mass_kg not given: estimated {self.battery_kg:.2f} kg "
                "(0.25 kg per pair of driven wheels)")

    @property
    def chassis_bottom_m(self) -> float:
        return 0.5 * self.wheel_radius_m

    @property
    def chassis_top_m(self) -> float:
        return self.chassis_bottom_m + self.chassis_size_m[2]


class ArmChain(_Strict):
    kind: Literal["arm_chain"] = "arm_chain"
    name: Name
    segments: list[Segment] = Field(min_length=1)
    mount_xy_m: Vec2Signed | None = Field(
        default=None,
        description="Only when the arm sits on a wheeled_base: (x, y) of its first joint on the "
        "chassis top, relative to the chassis centre (x forward). Defaults to (0, 0).")


class Gripper(_Strict):
    kind: Literal["gripper"] = "gripper"
    name: Name
    gripper_type: Literal["parallel", "suction"]
    max_opening_m: PositiveFloat = Field(
        description="Parallel: max finger gap. Suction: cup diameter."
    )
    grip_force_n: PositiveFloat
    motor_id: Name | None = Field(
        default=None, description="Required for parallel grippers (drives the fingers)."
    )

    @model_validator(mode="after")
    def _parallel_needs_motor(self) -> Gripper:
        if self.gripper_type == "parallel" and self.motor_id is None:
            raise ValueError(f"gripper '{self.name}' is parallel and needs a motor_id")
        return self


Module = Annotated[FixedBase | WheeledBase | ArmChain | Gripper, Field(discriminator="kind")]


# --------------------------------------------------------------------------- naming
# Names of the URDF links/joints generated for modules. Shared with build_urdf so
# the uniqueness check below covers everything that ends up in the URDF.


def base_link_name(base: FixedBase | WheeledBase) -> str:
    return base.name


def wheel_names(b: WheeledBase) -> tuple[list[str], list[str]]:
    """(link names, joint names) generated for a wheeled base's wheels and caster."""
    sides = ["left", "right"] if b.drive == "differential" else [
        "front_left", "front_right", "rear_left", "rear_right"]
    links = [f"{b.name}_wheel_{s}" for s in sides]
    joints = [f"{b.name}_wheel_{s}_joint" for s in sides]
    if b.caster:
        links.append(f"{b.name}_caster")
        joints.append(f"{b.name}_caster_mount")
    return links, joints


def gripper_names(g: Gripper) -> tuple[list[str], list[str]]:
    """(link names, joint names) generated for a gripper."""
    links = [f"{g.name}_palm"]
    joints = [f"{g.name}_mount"]
    if g.gripper_type == "parallel":
        links += [f"{g.name}_finger_left", f"{g.name}_finger_right"]
        joints += [f"{g.name}_finger_left_joint", f"{g.name}_finger_right_joint"]
    return links, joints


# --------------------------------------------------------------------------- robot


class RobotSpec(_Strict):
    name: Name
    description: str = ""
    control: Literal["pd", "pid"] = Field(
        default="pd",
        description="Joint position control. 'pd': the motors' factory position loops (no "
        "integral term), so an unmodelled load leaves a steady sag. 'pid': adds one fixed integral "
        "gain preset on Dynamixels, which removes that sag over a few seconds; hobby servos and "
        "steppers have no integral term either way.",
    )
    modules: list[Module] = Field(min_length=1)

    @field_validator("modules")
    @classmethod
    def _serial_order(cls, modules: list[Module]) -> list[Module]:
        kinds = [m.kind for m in modules]
        bases = ("fixed_base", "wheeled_base")
        if kinds[0] not in bases:
            raise ValueError(
                f"modules[0] must be a fixed_base or wheeled_base (the root), got '{kinds[0]}'"
            )
        if sum(k in bases for k in kinds) > 1:
            raise ValueError("only one base (fixed_base or wheeled_base) is allowed, as modules[0]")
        if kinds[0] == "fixed_base" and "arm_chain" not in kinds:
            raise ValueError("at least one arm_chain must follow the fixed_base")
        if "gripper" in kinds and "arm_chain" not in kinds:
            raise ValueError("a gripper needs an arm_chain to attach to")
        for i, m in enumerate(modules):
            if isinstance(m, ArmChain) and m.mount_xy_m is not None and (
                    kinds[0] != "wheeled_base" or i != 1):
                raise ValueError(f"arm_chain '{m.name}': mount_xy_m is only for the first arm "
                                 "on a wheeled_base")
        for i, k in enumerate(kinds):
            if k == "gripper" and i != len(kinds) - 1:
                raise ValueError(
                    f"gripper at modules[{i}] must be the last module; nothing can "
                    "attach after a gripper"
                )
        return modules

    @model_validator(mode="after")
    def _unique_names_and_known_motors(self, info: ValidationInfo) -> RobotSpec:
        seen: dict[str, str] = {}

        def claim(name: str, what: str) -> None:
            if name in seen:
                raise ValueError(
                    f"name '{name}' is used by both {seen[name]} and {what}; "
                    "every module, link and joint name must be unique"
                )
            seen[name] = what

        for i, m in enumerate(self.modules):
            if isinstance(m, ArmChain):
                claim(m.name, f"module {i}")
                for j, s in enumerate(m.segments):
                    claim(s.joint.name, f"joint in {m.name}.segments[{j}]")
                    claim(s.link.name, f"link in {m.name}.segments[{j}]")
            elif isinstance(m, Gripper):
                claim(m.name, f"module {i}")
                links, joints = gripper_names(m)
                for n in links + joints:
                    claim(n, f"generated part of gripper '{m.name}'")
            elif isinstance(m, WheeledBase):
                claim(m.name, f"module {i} (chassis link)")
                links, joints = wheel_names(m)
                for n in links + joints:
                    claim(n, f"generated part of wheeled_base '{m.name}'")
            else:
                claim(m.name, f"module {i} (base link)")

        context = info.context or {}
        catalog = context.get("catalog") or load_motor_catalog()
        motor_ids = context.get("motor_ids")
        if motor_ids is None:
            motor_ids = set(catalog)
        for used, where, mode in self._motor_uses():
            if used not in motor_ids:
                raise ValueError(
                    f"{where} uses unknown motor_id '{used}'; "
                    f"choose one of: {', '.join(sorted(motor_ids))}"
                )
            if used in catalog and mode not in catalog[used].modes:
                able = sorted(k for k in motor_ids if k in catalog and mode in catalog[k].modes)
                raise ValueError(
                    f"{where} needs a motor with {mode} control, but '{used}' only supports "
                    f"{', '.join(catalog[used].modes)}; choose one of: {', '.join(able)}"
                )
        return self

    def _motor_uses(self) -> list[tuple[str, str, str]]:
        """(motor_id, where, control mode it needs)."""
        uses = []
        for m in self.modules:
            if isinstance(m, ArmChain):
                uses += [(s.joint.motor_id, f"joint '{s.joint.name}'", "position") for s in m.segments]
            elif isinstance(m, Gripper) and m.motor_id is not None:
                uses.append((m.motor_id, f"gripper '{m.name}'", "position"))
            elif isinstance(m, WheeledBase):
                uses.append((m.wheel_motor_id, f"wheeled_base '{m.name}' wheels", "velocity"))
        return uses

    @property
    def base(self) -> FixedBase | WheeledBase:
        return self.modules[0]  # guaranteed by _serial_order

    @property
    def is_mobile(self) -> bool:
        return isinstance(self.modules[0], WheeledBase)

    @property
    def has_arm(self) -> bool:
        return any(isinstance(m, ArmChain) for m in self.modules)
