"""Measurable requirements extracted from the user's task description (SI units)."""

from __future__ import annotations

import json
import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, NonNegativeFloat, PositiveFloat, model_validator

from taskbot.llm import LLMClient
from taskbot.prompts import load_prompt

OUT_OF_SCOPE_PREFIX = "OUT OF SCOPE:"


class Workspace(BaseModel):
    """Axis-aligned box the gripper must be able to reach, in the base frame (m).

    Origin is the centre of the base's bottom face; z is up.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    @model_validator(mode="after")
    def _min_below_max(self) -> Workspace:
        for axis in "xyz":
            lo, hi = getattr(self, f"{axis}_min"), getattr(self, f"{axis}_max")
            if lo > hi:
                raise ValueError(f"workspace {axis}_min ({lo}) must not exceed {axis}_max ({hi})")
        return self


class Requirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_summary: str = Field(min_length=1)
    robot_type: Literal["arm", "mobile", "mobile_manipulator"] = Field(
        default="arm",
        description="arm: works from one fixed place. mobile: drives around, no arm. "
        "mobile_manipulator: drives and handles objects with an arm.")
    reach_m: PositiveFloat | None = Field(
        default=None, description="Arm types only: required reach including the 10% margin.")
    payload_kg: NonNegativeFloat = Field(
        description="Mass the gripper holds while working (e.g. a watering can and its water, not "
        "the plants), including the 25% margin (0 for mobile).")
    workspace: Workspace | None = Field(
        default=None, description="Arm types only. For a mobile_manipulator, in the robot frame: "
        "origin on the ground under the chassis centre, x forward, z up.")
    dof_min: int | None = Field(default=None, ge=1, le=7, description="Arm types only.")
    gripper_needed: bool
    speed_m_s: PositiveFloat | None = Field(default=None, description="Mobile types: cruising speed.")
    max_incline_deg: float | None = Field(
        default=None, ge=0, le=30, description="Mobile types: steepest slope it must climb.")
    terrain: Literal["flat", "carpet", "outdoor"] | None = Field(
        default=None, description="Mobile types: what it drives on.")
    base_payload_kg: NonNegativeFloat | None = Field(
        default=None, description="Mobile types: mass carried on the chassis (25% margin included).")
    travel_distance_m: PositiveFloat | None = Field(
        default=None, description="Mobile types, optional: distance it must cover per charge.")
    gripper_kind: Literal["parallel", "suction"] | None = Field(
        default=None, description="Required when gripper_needed is true; null otherwise. parallel: "
        "holds objects, tools or containers; suction: flat, smooth, light items only."
    )
    max_budget_usd: PositiveFloat | None = None
    position_tolerance_m: float = Field(
        gt=0, le=0.1,
        description="How precisely the tool must be placed for this task (m), e.g. ~0.01 to water "
        "a plant, ~0.002 to pick M3 screws. Use 0.005 if unclear and say so in assumptions.",
    )
    tolerance_reason: str = Field(min_length=1, description="Why the task needs this precision.")
    min_joint_range_rad: PositiveFloat = Field(
        default=math.pi / 2,
        description="Smallest allowed range for any revolute joint. Default 90°; "
        "lower it only if the task needs less motion, and say why in assumptions.",
    )
    assumptions: list[str] = Field(
        default_factory=list, description="Every value the model guessed, and why."
    )

    @model_validator(mode="after")
    def _fields_match_robot_type(self) -> Requirements:
        arm_fields = ("reach_m", "workspace", "dof_min")
        mobile_fields = ("speed_m_s", "max_incline_deg", "terrain", "base_payload_kg")
        has_arm = self.robot_type in ("arm", "mobile_manipulator")
        is_mobile = self.robot_type in ("mobile", "mobile_manipulator")
        for name in arm_fields:
            if has_arm and getattr(self, name) is None:
                raise ValueError(f"robot_type '{self.robot_type}' needs {name}")
            if not has_arm and getattr(self, name) is not None:
                raise ValueError(f"robot_type 'mobile' has no arm, so {name} must be null")
        for name in mobile_fields + ("travel_distance_m",):
            if is_mobile and name in mobile_fields and getattr(self, name) is None:
                raise ValueError(f"robot_type '{self.robot_type}' needs {name}")
            if not is_mobile and getattr(self, name) is not None:
                raise ValueError(f"robot_type 'arm' doesn't move, so {name} must be null")
        if not has_arm and self.gripper_needed:
            raise ValueError("robot_type 'mobile' has no arm, so gripper_needed must be false")
        return self

    @property
    def has_arm(self) -> bool:
        return self.robot_type in ("arm", "mobile_manipulator")

    @property
    def is_mobile(self) -> bool:
        return self.robot_type in ("mobile", "mobile_manipulator")

    @model_validator(mode="after")
    def _gripper_kind_matches(self) -> Requirements:
        if self.gripper_needed and self.gripper_kind is None:
            raise ValueError("gripper_needed is true, so gripper_kind must be 'parallel' or 'suction'")
        if not self.gripper_needed and self.gripper_kind is not None:
            raise ValueError("gripper_needed is false, so gripper_kind must be null")
        return self

    @property
    def out_of_scope(self) -> list[str]:
        return [a for a in self.assumptions if a.strip().startswith(OUT_OF_SCOPE_PREFIX)]


def requirements_system_prompt() -> str:
    schema = json.dumps(Requirements.model_json_schema(), separators=(",", ":"))
    return f"{load_prompt('requirements').rstrip()}\nSchema: {schema}\n"


def extract_requirements(client: LLMClient, task: str) -> Requirements:
    return client.complete_json(requirements_system_prompt(), task, Requirements)
