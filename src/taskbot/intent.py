"""DesignIntent: what the LLM writes in the default "hybrid" design mode.

The LLM chooses; code sizes. An intent holds only choices (the robot type, an arm layout
from ARM_LAYOUTS in taskbot.sizing, the gripper, the link material policy, the drive) and a
short rationale. No lengths, heights, joint limits or motor ids: taskbot.sizing computes
every number from the intent, the requirements and the motor catalog.

To add an arm layout: add its name to ArmLayout here and its joint list to
taskbot.sizing.ARM_LAYOUTS, and give it one line in prompts/intent.md.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from taskbot.llm import LLMClient
from taskbot.prompts import load_prompt
from taskbot.requirements import Requirements

ArmLayout = Literal["yaw_pitch_pitch", "yaw_pitch_pitch_pitch", "yaw_pitch_pitch_wristroll"]


class DesignIntent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    robot_type: Literal["arm", "mobile", "mobile_manipulator"] = Field(
        description="Optional: copied from the requirements.")
    arm_layout: ArmLayout | None = Field(
        default=None, description="Arm types only. yaw_pitch_pitch: base yaw, shoulder and elbow pitch. "
        "yaw_pitch_pitch_pitch: plus a wrist pitch to tilt the tool. yaw_pitch_pitch_wristroll: plus a "
        "wrist roll to turn the tool.")
    gripper: Literal["parallel", "suction", "none"] = Field(
        description="Optional: copied from the requirements' gripper_kind ('none' when no gripper).")
    link_material: Literal["auto", "pla", "aluminium"] = Field(
        default="auto", description="auto: PLA for links under 0.25 m, aluminium for longer ones.")
    drive: Literal["differential", "four_wheel"] | None = Field(
        default=None, description="Mobile types only.")
    caster: bool | None = Field(default=None, description="Mobile types with differential drive only.")
    arm_mount: Literal["front", "centre"] | None = Field(
        default=None, description="mobile_manipulator only: where the arm stands on the chassis.")
    not_modelled: list[str] = Field(
        default_factory=list, description="Things the task asks for that TaskBot doesn't model (a colour, "
        "a camera, a sensor, a sound), each a few words. Empty if none.")
    rationale: str = Field(min_length=1, max_length=600, description="Why these choices, 1-3 sentences.")

    @model_validator(mode="before")
    @classmethod
    def _defaults_from_requirements(cls, data, info: ValidationInfo):
        req = (info.context or {}).get("requirements")
        if isinstance(data, dict) and req is not None:
            data = dict(data)
            data.setdefault("robot_type", req.robot_type)
            data.setdefault("gripper", req.gripper_kind if req.gripper_needed else "none")
        return data

    @model_validator(mode="after")
    def _fields_match_type(self) -> DesignIntent:
        has_arm = self.robot_type in ("arm", "mobile_manipulator")
        mobile = self.robot_type in ("mobile", "mobile_manipulator")
        if has_arm and self.arm_layout is None:
            raise ValueError(f"robot_type '{self.robot_type}' needs arm_layout")
        if not has_arm and (self.arm_layout is not None or self.gripper != "none"):
            raise ValueError("robot_type 'mobile' has no arm: arm_layout must be null and gripper 'none'")
        if mobile and self.drive is None:
            raise ValueError(f"robot_type '{self.robot_type}' needs drive")
        if not mobile and (self.drive is not None or self.caster is not None):
            raise ValueError("robot_type 'arm' doesn't drive: drive and caster must be null")
        if self.caster and self.drive != "differential":
            raise ValueError("caster is for differential drive only")
        if (self.robot_type == "mobile_manipulator") != (self.arm_mount is not None):
            raise ValueError("arm_mount is required for a mobile_manipulator and only for one")
        return self

    @model_validator(mode="after")
    def _matches_requirements(self, info: ValidationInfo) -> DesignIntent:
        req: Requirements | None = (info.context or {}).get("requirements")
        if req is None:
            return self
        if self.robot_type != req.robot_type:
            raise ValueError(f"robot_type is '{self.robot_type}' but the requirements ask for '{req.robot_type}'")
        want = req.gripper_kind if req.gripper_needed else "none"
        if self.gripper != want:
            raise ValueError(f"gripper is '{self.gripper}' but the requirements ask for '{want}'")
        return self


def intent_system_prompt() -> str:
    import json

    schema = json.dumps(DesignIntent.model_json_schema(), separators=(",", ":"))
    return load_prompt("intent").replace("{schema}", schema)


def intent_user_message(req: Requirements, feedback: str | None = None) -> str:
    text = "Requirements:\n" + req.model_dump_json(indent=2)
    if feedback:
        text += "\n\n" + feedback
    return text


def request_intent(client: LLMClient, req: Requirements, feedback: str | None = None) -> DesignIntent:
    from taskbot.normalize import normalize_intent

    return client.complete_json(intent_system_prompt(), intent_user_message(req, feedback), DesignIntent,
                                context={"requirements": req}, step="intent_revision" if feedback else "intent",
                                normalize=lambda d: normalize_intent(d, req))
