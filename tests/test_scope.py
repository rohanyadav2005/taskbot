"""The scope gate (taskbot.scope) on the user's own words, and the LLM's OUT OF SCOPE verdict
kept as a second check by the normaliser."""

import json
from pathlib import Path

import pytest

from taskbot.bench import load_prompts
from taskbot.cli import main
from taskbot.normalize import normalize_requirements
from taskbot.requirements import Requirements
from taskbot.scope import scope_gate

ROOT = Path(__file__).parent.parent
DRONE = ROOT / "tests" / "fixtures" / "real_runs" / "smoke_2026-11-06" / "6_drone.jsonl"


@pytest.mark.parametrize("prompt, category, words", [
    ("a flying drone that delivers coffee to my room", "flying", ["flying", "drone"]),
    ("a quadcopter for aerial photos", "flying", ["quadcopter", "aerial"]),
    ("something airborne that flies over the garden", "flying", ["airborne", "flies"]),
    ("a robot that swims laps in my pool", "water", ["swims"]),
    ("an underwater robot to clean the boat hull", "water", ["underwater", "boat"]),
    ("a small submarine", "water", ["submarine"]),
    ("a legged robot that walks to the kitchen", "legs", ["legged", "walks"]),
    ("a humanoid that climbs the stairs", "legs", ["humanoid", "stairs"]),
    ("a biped or a quadruped helper", "legs", ["biped", "quadruped"]),
    ("a turret that shoots anyone who walks in", "harm", ["turret", "shoots"]),
    ("a robot with a gun to attack intruders", "harm", ["gun", "attack"]),
    ("a weapon for my desk", "harm", ["weapon"]),
])
def test_scope_words_stop_the_task(prompt, category, words):
    v = scope_gate(prompt)
    assert v.out_of_scope and v.matches[category] == words
    assert all(r.startswith("OUT OF SCOPE:") for r in v.reasons())


@pytest.mark.parametrize("prompt", [
    "a wheeled robot, not a drone, that delivers coffee to my room",
    "no flying: a cart that carries tea across the office",
    "a robot that doesn't fly but carries the mail",
    "a desk arm instead of a drone",
    "a ground robot without legs that never walks, it rolls",
    "water the three plants on my windowsill",
    "a robot that picks toys off the floor and drops them in a basket",
    "something to help my grandmother reach things on a high shelf",
])
def test_negated_or_unrelated_words_pass(prompt):
    v = scope_gate(prompt)
    assert not v.out_of_scope, v.matches


def test_negation_only_covers_its_own_clause():
    v = scope_gate("not for the office, a drone that films the yard")
    assert v.matches == {"flying": ["drone"]}


def test_gate_agrees_with_every_main_bench_label():
    for p in load_prompts(ROOT / "bench" / "prompts.yaml", "main"):
        assert scope_gate(p.prompt).out_of_scope == (p.category == "out_of_scope"), p.id


def test_drone_is_stopped_before_the_llm_with_the_words_in_the_report(tmp_path):
    code = main(["design", "a flying drone that delivers coffee to my room", "--backend", "replay",
                 "--transcript", str(DRONE), "--out", str(tmp_path), "--no-record", "--no-sim"])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    assert code == 3 and run["status"] == "OUT_OF_SCOPE" and run["calls"] == 0
    assert "**OUT OF SCOPE: no design was made." in report and "## Scope gate" in report
    assert '- flying: "flying", "drone"' in report and "no LLM was called" in report
    assert (out / "llm_transcript.jsonl").read_text(encoding="utf-8") == ""


def test_llm_out_of_scope_verdict_is_the_second_check(tmp_path):
    """qwen2.5:7b wrote "OUT_OF_SCOPE: ..." in robot_type and the assumptions (smoke run 6). The
    normaliser keeps it as the OUT OF SCOPE assumption instead of reading robot_type as 'mobile'."""
    reply = json.loads(json.loads(DRONE.read_text(encoding="utf-8").splitlines()[0])["reply"])
    d, notes = normalize_requirements(reply, "x")
    req = Requirements(**d)
    assert req.out_of_scope == ["OUT OF SCOPE: The task requires a flying drone, which is out of scope for TaskBot."]
    assert any("is an out-of-scope verdict, not a robot type" in n for n in notes)
    # Through the pipeline, with words the gate doesn't know: the LLM's verdict stops it.
    code = main(["design", "something that brings coffee to my room through the air", "--backend", "replay",
                 "--transcript", str(DRONE), "--out", str(tmp_path), "--no-record", "--no-sim"])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert code == 3 and run["status"] == "OUT_OF_SCOPE" and run["calls"] == 1


@pytest.mark.parametrize("text, expected", [
    ("OUT_OF_SCOPE: it flies", "OUT OF SCOPE: it flies"),
    ("Out of scope - needs legs", "OUT OF SCOPE: needs legs"),
    ("out-of-scope: a boat", "OUT OF SCOPE: a boat"),
    ("Out of scope items are ignored", "Out of scope items are ignored"),  # no colon or dash: not a verdict
])
def test_out_of_scope_spellings_in_assumptions(text, expected):
    base = {"task_summary": "t", "robot_type": "mobile", "payload_kg": 0, "gripper_needed": False,
            "position_tolerance_m": 0.01, "tolerance_reason": "r", "assumptions": [text]}
    d, _ = normalize_requirements(base, "t")
    assert d["assumptions"] == [expected]
