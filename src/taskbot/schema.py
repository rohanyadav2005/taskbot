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


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# --------------------------------------------------------------------------- motors


class PositionGains(_Strict):
    """Raw position-loop gains as written in a Dynamixel X-series control table."""

    p: Annotated[int, Field(ge=0)]
    i: Annotated[int, Field(ge=0)]
    d: Annotated[int, Field(ge=0)]


class Motor(_Strict):
    id: Name
    name: str = Field(min_length=1)
    kind: Literal["hobby_servo", "dynamixel", "stepper"] = Field(
        description="Decides the controller gains and the torque-speed model used in simulation."
    )
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
    pull_out_curve: list[tuple[float, float]] | None = Field(
        default=None,
        description="Stepper only: (speed rad/s, torque N·m) points of the pull-out curve.",
    )
    sources: dict[str, str] = Field(
        default_factory=dict,
        description="Where each value came from, keyed by field name.",
    )

    @model_validator(mode="after")
    def _kind_fields(self) -> Motor:
        if (self.kind == "dynamixel") != (self.dynamixel_gains is not None):
            raise ValueError(f"motor {self.id}: dynamixel_gains is required for, and only for, kind 'dynamixel'")
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


class ArmChain(_Strict):
    kind: Literal["arm_chain"] = "arm_chain"
    name: Name
    segments: list[Segment] = Field(min_length=1)


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


Module = Annotated[FixedBase | ArmChain | Gripper, Field(discriminator="kind")]


# --------------------------------------------------------------------------- naming
# Names of the URDF links/joints generated for modules. Shared with build_urdf so
# the uniqueness check below covers everything that ends up in the URDF.


def base_link_name(base: FixedBase) -> str:
    return base.name


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
    modules: list[Module] = Field(min_length=2)

    @field_validator("modules")
    @classmethod
    def _serial_order(cls, modules: list[Module]) -> list[Module]:
        kinds = [m.kind for m in modules]
        if kinds[0] != "fixed_base":
            raise ValueError(
                f"modules[0] must be a fixed_base (the root), got '{kinds[0]}'"
            )
        if kinds.count("fixed_base") > 1:
            raise ValueError("only one fixed_base is allowed, and it must be modules[0]")
        if "arm_chain" not in kinds:
            raise ValueError("at least one arm_chain must follow the fixed_base")
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
            else:
                claim(m.name, f"module {i} (base link)")

        motor_ids = (info.context or {}).get("motor_ids")
        if motor_ids is None:
            motor_ids = set(load_motor_catalog())
        for used, where in self._motor_uses():
            if used not in motor_ids:
                raise ValueError(
                    f"{where} uses unknown motor_id '{used}'; "
                    f"choose one of: {', '.join(sorted(motor_ids))}"
                )
        return self

    def _motor_uses(self) -> list[tuple[str, str]]:
        uses = []
        for m in self.modules:
            if isinstance(m, ArmChain):
                uses += [(s.joint.motor_id, f"joint '{s.joint.name}'") for s in m.segments]
            elif isinstance(m, Gripper) and m.motor_id is not None:
                uses.append((m.motor_id, f"gripper '{m.name}'"))
        return uses

    @property
    def base(self) -> FixedBase:
        return self.modules[0]  # guaranteed by _serial_order
