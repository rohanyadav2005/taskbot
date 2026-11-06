"""DesignIntent: the LLM's choices in hybrid mode, with no numbers in them."""

import pytest
from pydantic import ValidationError

from taskbot.fake import _load
from taskbot.intent import DesignIntent, intent_system_prompt, request_intent
from taskbot.llm import FakeClient
from taskbot.requirements import Requirements

ARM = {"robot_type": "arm", "arm_layout": "yaw_pitch_pitch_pitch", "gripper": "parallel", "rationale": "r"}
MOBILE = {"robot_type": "mobile", "gripper": "none", "drive": "differential", "caster": True, "rationale": "r"}
MM = {"robot_type": "mobile_manipulator", "arm_layout": "yaw_pitch_pitch", "gripper": "parallel",
      "drive": "differential", "caster": True, "arm_mount": "front", "rationale": "r"}


@pytest.mark.parametrize("d", [ARM, MOBILE, MM])
def test_valid_intents(d):
    assert DesignIntent(**d).link_material == "auto"


@pytest.mark.parametrize("d, match", [
    (ARM | {"arm_layout": None}, "needs arm_layout"),
    (MOBILE | {"arm_layout": "yaw_pitch_pitch"}, "has no arm"),
    (MOBILE | {"drive": None}, "needs drive"),
    (ARM | {"drive": "differential"}, "doesn't drive"),
    (MOBILE | {"drive": "four_wheel", "caster": True}, "caster is for differential"),
    (MM | {"arm_mount": None}, "arm_mount is required"),
    (ARM | {"link_length_m": 0.3}, "Extra inputs are not permitted"),  # no numbers in an intent
])
def test_invalid_intents(d, match):
    with pytest.raises(ValidationError, match=match):
        DesignIntent(**d)


def test_intent_must_match_the_requirements():
    req = Requirements(**_load("mobile_requirements"))
    with pytest.raises(ValidationError, match="robot_type is 'arm' but the requirements ask for 'mobile'"):
        DesignIntent.model_validate(ARM, context={"requirements": req})
    mm = Requirements(**_load("mobile_manipulator_requirements"))
    with pytest.raises(ValidationError, match="gripper is 'suction' but the requirements ask for 'parallel'"):
        DesignIntent.model_validate(MM | {"gripper": "suction"}, context={"requirements": mm})


def test_request_intent_corrects_a_mismatch_instead_of_retrying():
    """Task 8: robot_type and gripper are copied from the requirements by code, with a note."""
    req = Requirements(**_load("mobile_requirements"))
    fake = FakeClient({"DesignIntent": [ARM, MOBILE]})
    intent = request_intent(fake, req)
    assert intent.robot_type == "mobile" and intent.drive == "differential" and len(fake.calls) == 1
    notes = [n for _, n in fake.corrections]
    assert "intent robot_type 'arm' replaced by the requirements' 'mobile'" in notes
    assert fake.calls[0][1][0]["content"].startswith("Requirements:\n")


def test_prompt_has_one_example_per_robot_type_and_no_numbers_to_choose():
    p = intent_system_prompt()
    for t in ("arm", "mobile", "mobile_manipulator"):
        assert f'{{"robot_type":"{t}"' in p
    assert "code computes every length, height, joint limit and motor" in p
    assert '"rationale"' in p and "link_length" not in p
