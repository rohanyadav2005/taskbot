"""Offline fake LLM backend, for tests and demos without an API key.

The prompt picks a canned scenario (taskbot/data/fake/):
    "box" + "carr" / "deliver"       -> mobile: round 1 fails wheel_torque and top_speed
                                        (slow JGB37-520 wheel motors), round 2 swaps in
                                        Pololu 37D gearmotors and passes
    "toy"                            -> mobile_manipulator: round 1 fails tip_over (short
                                        wheelbase), round 2 lengthens it and passes
    "fly" / "drone" / "swim" / "legs" -> out of scope: no design is made
    anything else                    -> fixed arm: canned 4-DOF design; 'repair' swaps any
                                        motor that fails static_torque (or saturates in
                                        simulation) for the cheapest supported one strong
                                        enough. It fixes nothing else.
Requirements are canned per scenario; only task_summary echoes the prompt.
"""

from __future__ import annotations

import json
import re
from importlib import resources

from taskbot.checks import TORQUE_SAFETY_FRACTION
from taskbot.llm import FakeClient
from taskbot.schema import Motor

_TORQUE_FAIL = re.compile(r"^- \[static_torque\](?: \(cause: \w+\))? (\w+) needs ([\d.e+-]+) Nm", re.MULTILINE)
_SATURATED = re.compile(r"(\w+) (?:saturated|speed-limited) for [\d.]+ s while [^(]*\(motor (\w+):")
OUT_OF_SCOPE_WORDS = ("fly", "flies", "flying", "drone", "swim", "swims", "swimming", "underwater", "legged", "legs", "walk up stairs")

ARM_REQUIREMENTS = {
    "robot_type": "arm", "reach_m": 0.3, "payload_kg": 0.25,
    "workspace": {"x_min": 0.1, "x_max": 0.25, "y_min": -0.1, "y_max": 0.1, "z_min": 0.05, "z_max": 0.2},
    "dof_min": 4, "gripper_needed": True, "gripper_kind": "parallel", "max_budget_usd": None,
    "position_tolerance_m": 0.005, "tolerance_reason": "FAKE BACKEND: canned 5 mm default",
    "assumptions": ["FAKE BACKEND: canned requirements, only task_summary reflects the prompt."],
}


def _load(name: str) -> dict:
    return json.loads(resources.files("taskbot").joinpath(f"data/fake/{name}.json").read_text("utf-8"))


def scenario(prompt: str) -> str:
    p = prompt.lower()
    if any(re.search(rf"\b{w}\b", p) for w in OUT_OF_SCOPE_WORDS):
        return "out_of_scope"
    if "toy" in p:
        return "mobile_manipulator"
    if "box" in p and any(w in p for w in ("carr", "deliver", "across", "move")):
        return "mobile"
    return "arm"


def _arm_repair(prompt: str, catalog: dict[str, Motor]) -> dict:
    spec = json.loads(prompt.split("Current spec: ", 1)[1].split("\nRequirements: ", 1)[0])
    # (joint, torque needed, motor it must also be faster than)
    needs = [(j, float(n), None) for j, n in _TORQUE_FAIL.findall(prompt)]
    needs += [(j, TORQUE_SAFETY_FRACTION * catalog[m].rated_torque_nm, catalog[m])
              for j, m in _SATURATED.findall(prompt) if m in catalog]
    for joint_name, need, slower in needs:
        strong = [m for m in catalog.values()
                  if "position" in m.modes and TORQUE_SAFETY_FRACTION * m.rated_torque_nm >= need
                  and (slower is None or m.no_load_speed_rad_s > slower.no_load_speed_rad_s)]
        if not strong:
            continue
        best = min(strong, key=lambda m: m.price_usd)
        for module in spec["modules"]:
            for seg in module.get("segments", []):
                if seg["joint"]["name"] == joint_name:
                    seg["joint"]["motor_id"] = best.id
        spec["description"] += f" Fake repair: {joint_name} -> {best.id}."
    return spec


def fake_client(catalog: dict[str, Motor]) -> FakeClient:
    """`catalog`: the motors the fake may pick from when repairing an arm (supported ones)."""

    def requirements(system, messages, schema):
        prompt = messages[0]["content"]
        kind = scenario(prompt)
        req = ARM_REQUIREMENTS if kind == "arm" else _load(f"{kind}_requirements")
        return dict(req, task_summary=prompt)

    def design(system, messages, schema):
        prompt = messages[-1]["content"]
        req_json = prompt.split("Requirements:", 1)[1]
        kind = json.loads(req_json.strip().split("\nReturn", 1)[0])["robot_type"] \
            if prompt.startswith("Requirements:") else json.loads(
                prompt.split("\nRequirements: ", 1)[1].split("\nReturn", 1)[0])["robot_type"]
        first = prompt.startswith("Requirements:")
        if kind == "arm":
            return _load("arm_design") if first else _arm_repair(prompt, catalog)
        return _load(f"{kind}_design_round1" if first else f"{kind}_design_round2")

    return FakeClient({"Requirements": requirements, "RobotSpec": design})
