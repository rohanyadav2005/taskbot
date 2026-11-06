"""Normalisers: code fixes what the LLM writes before validation, and says what it changed."""

import json
from pathlib import Path

import pytest

from taskbot.fake import fake_client
from taskbot.intent import DesignIntent
from taskbot.llm import ReplayClient
from taskbot.normalize import default_intent, normalize_intent, normalize_requirements
from taskbot.pipeline import run_pipeline
from taskbot.requirements import Requirements, extract_requirements
from taskbot.schema import load_motor_catalog, supported_motors

FIXTURES = Path(__file__).parent / "fixtures" / "real_runs"
ARM = {"task_summary": "water plants", "robot_type": "arm", "payload_kg": 0.3, "gripper_needed": True,
       "gripper_kind": "parallel", "dof_min": 3, "position_tolerance_m": 0.01, "tolerance_reason": "r",
       "workspace": {"x_min": 0.15, "x_max": 0.3, "y_min": -0.1, "y_max": 0.1, "z_min": 0.05, "z_max": 0.25}}


def norm(d, task="water plants"):
    out, notes = normalize_requirements(d, task)
    return Requirements(**out), notes


def test_the_real_crash_now_validates_with_a_derived_reach():
    req = extract_requirements(ReplayClient(FIXTURES / "windowsill_reach_crash.jsonl"),
                               "water the three plants on my windowsill")
    assert req.reach_m == 0.4  # 0.358 m x 1.10, rounded up to 1 cm


def test_reach_is_derived_and_the_models_value_kept_in_the_note():
    req, notes = norm(ARM | {"reach_m": 0.9})
    assert req.reach_m == 0.37  # corner (0.3, 0.1, 0.05) is 0.332 m from the shoulder 0.15 m up, x 1.10
    assert any(n.startswith("reach_m 0.37 m derived from the workspace") and "the model said 0.9 m" in n
               for n in notes)
    _, close = norm(ARM | {"reach_m": 0.37})
    assert not any("reach_m" in n for n in close)


def test_workspace_over_the_base_moves_the_smaller_way():
    centred = {"x_min": -0.15, "x_max": 0.15, "y_min": -0.15, "y_max": 0.15, "z_min": 0.0, "z_max": 0.3}
    req, notes = norm(ARM | {"workspace": centred})  # raising would cut a third of it: moved forward
    assert (req.workspace.x_min, req.workspace.x_max, req.workspace.z_min) == (0.1, 0.4, 0.0)
    assert any("moved it 0.25 m forward (x now 0.1 to 0.4 m, same size)" in n for n in notes)
    tall = {"x_min": -0.15, "x_max": 0.15, "y_min": -0.15, "y_max": 0.15, "z_min": 0.05, "z_max": 0.45}
    req, notes = norm(ARM | {"workspace": tall})  # raising cuts 12%: raised
    assert req.workspace.z_min == 0.1 and req.workspace.x_min == -0.15
    assert any("raised z_min from 0.05 to 0.1 m (cuts 12% of its height" in n for n in notes)
    near = {"x_min": 0.05, "x_max": 0.25, "y_min": -0.1, "y_max": 0.1, "z_min": 0.0, "z_max": 0.2}
    req, notes = norm(ARM | {"workspace": near})
    assert (req.workspace.x_min, req.workspace.x_max) == (0.1, 0.3) and req.workspace.z_min == 0.0
    assert any("moved it 0.05 m forward" in n for n in notes)


def test_reversed_and_absurd_values_are_fixed():
    ws = {"x_min": 0.3, "x_max": 0.15, "y_min": -0.1, "y_max": 0.1, "z_min": 0.0, "z_max": 9.0}
    req, notes = norm(ARM | {"workspace": ws})
    assert (req.workspace.x_min, req.workspace.x_max, req.workspace.z_max) == (0.15, 0.3, 1.5)
    assert "workspace x_min and x_max were reversed: swapped" in notes
    assert "workspace z_max 9 m is outside 0 to 1.5 m: clipped to 1.5 m" in notes


def test_fields_for_the_other_robot_type_are_dropped_and_mobile_defaults_filled():
    req, notes = norm(ARM | {"speed_m_s": 0.5, "terrain": "carpet"})
    assert req.speed_m_s is None and "speed_m_s doesn't apply to an arm: set to null" in notes
    mobile = {"task_summary": "carry", "robot_type": "mobile", "payload_kg": 1.0, "gripper_needed": True,
              "workspace": ARM["workspace"], "position_tolerance_m": 0.05, "tolerance_reason": "r"}
    req, notes = norm(mobile)
    assert (req.speed_m_s, req.max_incline_deg, req.terrain, req.base_payload_kg) == (0.3, 5.0, "flat", 0.0)
    assert req.workspace is None and req.payload_kg == 0 and not req.gripper_needed
    assert "speed_m_s missing: used the indoor default 0.3" in notes


def test_gripper_flags_are_made_consistent():
    req, notes = norm(ARM | {"gripper_kind": None})
    assert req.gripper_kind == "parallel" and "gripper_needed with no gripper_kind: used parallel" in notes
    req, notes = norm(ARM | {"gripper_needed": False, "gripper_kind": "suction"})
    assert req.gripper_needed and req.gripper_kind == "suction"


def test_margin_applied_once_from_the_bare_value():
    req, notes = norm(ARM | {"payload_bare_kg": 0.3, "payload_kg": 0.5})
    assert req.payload_kg == 0.375 and req.payload_bare_kg == 0.3
    assert "payload_kg 0.5 kg replaced by 0.3 kg bare x 1.25 = 0.375 kg (the margin applied once, by code)" in notes


def test_strings_unknown_fields_and_missing_tolerance():
    d = ARM | {"payload_kg": "0.3 kg", "colour": "red", "position_tolerance_m": 2.0}
    req, notes = norm(d)
    assert req.payload_kg == 0.3 and req.position_tolerance_m == 0.005
    assert "ignored unknown field 'colour'" in notes and "payload_kg '0.3 kg' read as 0.3" in notes


def test_intent_copies_type_and_gripper_and_fills_defaults():
    mm = Requirements(**json.loads((Path(__file__).parent.parent / "src/taskbot/data/fake/"
                                    "mobile_manipulator_requirements.json").read_text(encoding="utf-8")))
    d, notes = normalize_intent({"robot_type": "arm", "gripper": "suction", "wheels": 6, "rationale": "r"}, mm)
    intent = DesignIntent.model_validate(d, context={"requirements": mm})
    assert (intent.robot_type, intent.gripper, intent.arm_layout, intent.drive, intent.caster, intent.arm_mount) == (
        "mobile_manipulator", "parallel", "yaw_pitch_pitch_pitch", "differential", True, "front")
    assert "intent: ignored unknown field 'wheels'" in notes
    assert default_intent(mm).rationale.startswith("default intent chosen by code")


@pytest.mark.parametrize("terrain, drive", [("flat", "differential"), ("outdoor", "four_wheel")])
def test_default_drive_depends_on_terrain(terrain, drive):
    mobile = Requirements(task_summary="t", robot_type="mobile", payload_kg=0, gripper_needed=False,
                          speed_m_s=0.3, max_incline_deg=2, terrain=terrain, base_payload_kg=1,
                          position_tolerance_m=0.05, tolerance_reason="r")
    assert default_intent(mobile).drive == drive


def test_report_lists_every_correction(tmp_path):
    result = run_pipeline(fake_client(supported_motors(load_motor_catalog())),
                          "a robot that picks toys off the floor and drops them in a basket", load_motor_catalog(),
                          tmp_path, simulate=False, record=False)
    report = (result.out_dir / "report.md").read_text(encoding="utf-8")
    assert "## Corrections made by code" in report
    assert "- requirements: reach_m 0.45 m derived from the workspace" in report
    assert json.loads((result.out_dir / "run.json").read_text(encoding="utf-8"))["corrections"] >= 1
