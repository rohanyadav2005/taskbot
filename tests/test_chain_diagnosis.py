"""Reach failures explain degenerate joint layouts, and repair requests track progress.

The all-z design is round 5 of the first real-model run (qwen2.5:7b-instruct via Ollama,
"water the three plants on my windowsill"), rebuilt from its summary in tests/data/.
"""

import json
from pathlib import Path

import pytest
from conftest import example_dict, strong_arm

from taskbot import checks
from taskbot.build_urdf import build_model
from taskbot.checks import CheckResult, chain_diagnosis
from taskbot.llm import FakeClient
from taskbot.prompts import load_prompt
from taskbot.repair import MOTOR_NOTE, describe_changes, progress_lines, repair_loop, repair_prompt
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec

ALL_Z = json.loads((Path(__file__).parent / "data" / "watering_arm_all_z.json").read_text(encoding="utf-8"))

# The requirements of that run (from its summary and corner list).
WATERING = Requirements(
    task_summary="water the three plants on my windowsill", robot_type="arm", reach_m=0.5, payload_kg=0.5,
    # z_min 0.1, not the run's 0.05: a workspace around the base below 0.1 m is now rejected.
    workspace={"x_min": -0.2, "x_max": 0.2, "y_min": -0.2, "y_max": 0.2, "z_min": 0.1, "z_max": 0.5},
    dof_min=3, gripper_needed=True, gripper_kind="suction", position_tolerance_m=0.01,
    tolerance_reason="1 cm is enough to water a plant", assumptions=[])


def spec_of(d: dict) -> RobotSpec:
    return RobotSpec.model_validate(d)


def segments(d: dict) -> list[dict]:
    return next(m for m in d["modules"] if m["kind"] == "arm_chain")["segments"]


def all_z(n: int = 4, length: float = 0.3, motor: str = "xm540_w270") -> dict:
    d = json.loads(json.dumps(ALL_Z))
    segs = segments(d)[:n]
    for s in segs:
        s["link"]["length_m"] = length
        s["joint"]["motor_id"] = motor
    next(m for m in d["modules"] if m["kind"] == "arm_chain")["segments"] = segs
    return d


def reach_results(d: dict, req: Requirements = WATERING):
    spec = spec_of(d)
    return checks.reach(spec, req, build_model(spec))


# --------------------------------------------------------------------------- diagnosis


def test_all_z_arm_from_the_real_run_is_diagnosed_first():
    reach, corners = reach_results(ALL_Z)
    lead = ("All 4 joints rotate about their own link (axis z) and the links point along z, so the arm can "
            "only spin in place: the tool can never leave the base axis. A reaching arm needs a base yaw "
            "joint (axis z) followed by pitch joints (axis y) at the shoulder and elbow.")
    for r in (reach, corners):
        assert not r.passed and r.cause == "unreachable"
        assert r.message.startswith(lead)
    assert corners.message.index(lead) < corners.message.index("workspace corners unreachable")
    assert "farthest sampled tool position is 0 m from the base axis" in reach.message
    assert corners.value > 0.01 and corners.limit == 0.01  # worst corner miss, in m


def test_no_bending_joint_with_a_slider_is_diagnosed():
    d = all_z(2)
    segments(d)[1]["joint"] |= {"type": "prismatic", "lower": 0.0, "upper": 0.1}
    (note,) = chain_diagnosis(spec_of(d), WATERING)
    assert note.startswith("None of the 2 joints bends the arm")
    assert "pitch joints (axis y)" in note


def test_single_pitch_joint_at_the_base_moves_the_tool_on_a_sphere():
    d = all_z(3, length=0.25)
    segments(d)[0]["joint"]["axis"] = "y"   # shoulder pitch at the base, then two twists
    notes = chain_diagnosis(spec_of(d), WATERING)
    assert notes[0].startswith("Only one joint bends the arm (shoulder, axis y)")
    tool = 0.03  # suction cup
    assert f"sphere of radius {0.75 + tool:.3g} m around shoulder" in notes[0]
    assert "Add an elbow pitch joint (axis y)" in notes[0]
    reach, corners = reach_results(d)
    assert not corners.passed and corners.message.startswith("Only one joint bends the arm")


def test_too_short_arm_states_both_numbers():
    spec = spec_of(example_dict("arm_4dof"))
    req = WATERING.model_copy(update={"reach_m": 0.5, "gripper_kind": "parallel"})
    note, height = chain_diagnosis(spec, req)
    assert height.startswith("The tool can only get between")  # its top, 0.435 m, is under z_max 0.5 m
    # upper_arm 0.13 + forearm 0.12 + wrist 0.04 after shoulder_pitch, plus the parallel gripper.
    from taskbot.build_urdf import tool_length_m
    gripper = spec.modules[-1]
    total = 0.13 + 0.12 + 0.04 + tool_length_m(gripper)
    assert note.startswith("The arm is too short: 0.29 m of links from shoulder_pitch")
    assert f"= {total:.3g} m at full stretch, but the task needs reach_m = 0.5 m" in note
    reach, _ = checks.reach(spec, req, build_model(spec))
    assert reach.message.startswith("The arm is too short")


def test_a_working_arm_gets_no_diagnosis(desk_req):
    spec = spec_of(strong_arm())
    assert chain_diagnosis(spec, desk_req) == []
    reach, corners = checks.reach(spec, desk_req, build_model(spec))
    assert reach.passed and corners.passed and reach.cause is None


# --------------------------------------------------------------------------- repair messages


def test_motor_note_only_when_every_failure_is_reach_type(desk_req):
    spec = spec_of(strong_arm())
    reach_only = [CheckResult(check="reach", passed=False, message="m", cause="unreachable"),
                  CheckResult(check="workspace_corners", passed=False, message="m", cause="unreachable")]
    assert MOTOR_NOTE in repair_prompt(spec, desk_req, reach_only)
    mixed = reach_only + [CheckResult(check="static_torque", passed=False, message="m", cause="saturated")]
    assert MOTOR_NOTE not in repair_prompt(spec, desk_req, mixed)


def test_progress_says_better_worse_or_unchanged():
    def r(value):
        return [CheckResult(check="reach", passed=False, value=value, limit=0.01, message="m")]
    assert progress_lines(r(0.523), r(0.586)) == [
        "- [reach] reach miss went from 52.3 cm to 58.6 cm: your last change made this worse."]
    assert progress_lines(r(0.586), r(0.3)) == [
        "- [reach] reach miss went from 58.6 cm to 30.0 cm: your last change made this better, but it still fails."]
    assert "didn't affect this" in progress_lines(r(0.3), r(0.3))[0]
    assert progress_lines([], r(0.3)) == []  # nothing to compare with


def test_describe_changes_names_parts():
    a, b = spec_of(all_z(3, 0.25, "ax12a")), spec_of(all_z(4, 0.30, "xm540_w270"))
    text = describe_changes(a, b)
    assert "joint shoulder motor_id ax12a -> xm540_w270" in text
    assert "link shoulder_link length_m 0.25 -> 0.3" in text
    assert "added joint gripper_mount (axis z, motor_id xm540_w270)" in text
    assert describe_changes(a, a) == "nothing changed"


def test_repair_loop_replays_the_real_run_with_progress_and_history():
    """Longer links, then stronger motors, on the all-z arm: each request says how it went."""
    designs = [all_z(3, 0.25, "ax12a"), all_z(3, 0.30, "ax12a"), all_z(3, 0.30, "xm430_w350")]
    fake = FakeClient({"RobotSpec": designs[1:]})
    events = []
    outcome = repair_loop(fake, WATERING, initial_spec=spec_of(designs[0]), max_rounds=3,
                          simulate=False, on_event=events.append)
    assert not outcome.passed and len(outcome.rounds) == 3
    first, second = (msgs[-1]["content"] for _, msgs, _ in fake.calls)
    assert first.startswith("Your robot design failed")
    assert "All 3 joints rotate about their own link" in first and MOTOR_NOTE in first
    assert "\nCompared with the round before:\n" not in first and "\nHistory:\n" not in first
    assert "- [reach] reach miss went from" in second and "your last change made this worse." in second
    assert "- Round 1 -> 2 changes: link shoulder_link length_m 0.25 -> 0.3" in second
    assert "- Round 2: failed reach (reach miss" in second
    # Round 3 only swapped motors: reported as not affecting reach, in the live log too.
    assert any("your last change didn't affect this" in e for e in events)


def test_prompts_carry_the_new_guidance():
    design = load_prompt("design")
    assert "A z-axis joint after an upright link only twists it in place" in design
    assert "base_yaw (axis z) -> turret 0.05 m; shoulder_pitch (axis y)" in design
    req = load_prompt("requirements")
    assert "for watering plants it is a cup of 200-300 ml of water plus the cup" in req
    assert "potted desk plant weighs" not in req
    assert "There is no fixed-tool or holder option" in req
    assert "Desk-scale robots should be light: assume PLA or aluminium" in req
    repair = load_prompt("repair")
    assert ("Fix only what each failure's cause points to. Reach and workspace failures are about joint axes, "
            "link lengths and base height, not motors.") in repair


@pytest.mark.parametrize("axis", ["x", "y"])
def test_bending_about_x_or_y_counts_as_pitch(axis):
    d = all_z(3, 0.3)
    segments(d)[1]["joint"]["axis"] = axis
    segments(d)[2]["joint"]["axis"] = axis
    assert chain_diagnosis(spec_of(d), WATERING) == []


def watering(**changes) -> Requirements:
    return Requirements(**(WATERING.model_dump() | changes))


def yaw_pitch_pitch(base_height: float = 0.03, l1: float = 0.2, l2: float = 0.2) -> dict:
    d = example_dict("arm_4dof")
    d["modules"][0]["height_m"] = base_height
    segs = segments(d)[:3]
    segs[1]["link"]["size_m"][2], segs[2]["link"]["size_m"][2] = l1, l2
    segments(d)[:] = segs
    return d


def test_normal_yaw_pitch_pitch_arm_gets_no_diagnosis():
    req = watering(**{
        "reach_m": 0.35, "gripper_kind": "parallel",
        "workspace": {"x_min": 0.15, "x_max": 0.3, "y_min": -0.1, "y_max": 0.1, "z_min": 0.05, "z_max": 0.3}})
    assert chain_diagnosis(spec_of(yaw_pitch_pitch()), req) == []


def test_workspace_above_the_arm_states_the_reachable_heights():
    req = watering(**{
        "reach_m": 0.8, "gripper_kind": "parallel",
        "workspace": {"x_min": 0.1, "x_max": 0.2, "y_min": -0.1, "y_max": 0.1, "z_min": 0.1, "z_max": 0.9}})
    note = next(n for n in chain_diagnosis(spec_of(yaw_pitch_pitch()), req) if n.startswith("The tool can only"))
    # base top 0.03 + turret 0.05 = shoulder at 0.08; links 0.4 + gripper 0.065 above and below it.
    assert note.startswith("The tool can only get between z = -0.385 m and z = 0.545 m: the arm starts on the "
                           "base top at 0.03 m and shoulder_pitch (the first joint that bends it) is at 0.08 m.")
    assert "to reach z = 0.9 m, raise the base or lengthen the links by at least 0.355 m" in note
    _, corners = checks.reach(spec_of(yaw_pitch_pitch()), req, build_model(spec_of(yaw_pitch_pitch())))
    assert not corners.passed and "The tool can only get between" in corners.message


def test_floor_below_a_tall_base_states_the_reachable_heights():
    req = watering(**{
        "reach_m": 0.7, "gripper_kind": "parallel",
        "workspace": {"x_min": 0.1, "x_max": 0.2, "y_min": -0.1, "y_max": 0.1, "z_min": 0.0, "z_max": 0.8}})
    note = next(n for n in chain_diagnosis(spec_of(yaw_pitch_pitch(base_height=0.6)), req)
                if n.startswith("The tool can only"))
    assert "between z = 0.185 m and z = 1.11 m" in note
    assert "to reach z = 0 m, lower the base or shoulder_pitch" in note and "by at least 0.185 m" in note


def test_reach_message_no_longer_sends_the_model_to_the_motors():
    spec = spec_of(example_dict("arm_2dof"))
    req = watering(**{"reach_m": 0.5, "gripper_kind": "parallel"})
    reach, corners = checks.reach(spec, req, build_model(spec))
    for r in (reach, corners):
        assert "torque" not in r.message and "motor" not in r.message
    assert MOTOR_NOTE.startswith("Motor changes won't help this failure.")
