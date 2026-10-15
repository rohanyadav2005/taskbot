"""Measurable requirements extracted from the user's task description (SI units)."""

from __future__ import annotations

import json
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
    robot_type: Literal["arm"] = "arm"
    reach_m: PositiveFloat = Field(description="Required reach including the 10% margin.")
    payload_kg: NonNegativeFloat = Field(description="Required payload including the 25% margin.")
    workspace: Workspace
    dof_min: int = Field(ge=1, le=7)
    gripper_needed: bool
    gripper_kind: Literal["parallel", "suction"] | None = Field(
        default=None, description="Required when gripper_needed is true; null otherwise."
    )
    max_budget_usd: PositiveFloat | None = None
    assumptions: list[str] = Field(
        default_factory=list, description="Every value the model guessed, and why."
    )

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
