"""Offline fake LLM backend, for tests and demos without an API key.

The prompt picks a canned scenario (taskbot/data/fake/):
    "box" + "carr" / "deliver"       -> mobile: round 1 fails wheel_torque and top_speed
                                        (slow JGB37-520 wheel motors), round 2 swaps in
                                        Pololu 37D gearmotors and passes
    "laundry"                        -> mobile, never fixed: 6.25 kg on the base; the 'repair'
                                        returns the round-1 design unchanged every round,
                                        so wheel_torque and top_speed still fail at the end
    "toy"                            -> mobile_manipulator: round 1 fails tip_over (short
                                        wheelbase), round 2 lengthens it and passes
    "fly" / "drone" / "swim" / "legs" -> out of scope: no design is made
    anything else                    -> fixed arm: canned 4-DOF design; 'repair' swaps any
                                        motor that fails static_torque (or saturates in
                                        simulation) for the cheapest supported one strong
                                        enough. It fixes nothing else.
Requirements are canned per scenario; only task_summary echoes the prompt.

With strict=True (the benchmark) only the prompts in FAKE_PROMPTS get an answer; any other
prompt raises NoFakeResponse, which the benchmark records as "no fake response", not as a
design failure.
"""

from __future__ import annotations

import json
import re
from importlib import resources

from taskbot.checks import TORQUE_SAFETY_FRACTION
from taskbot.llm import FakeClient, LLMError
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


# Prompts with canned answers in strict mode (the app's examples and some benchmark prompts).
FAKE_PROMPTS = (
    "Water the three potted plants on my windowsill",
    "a robot that carries a 2 kg box across my office",
    "a robot that picks toys off the floor and drops them in a basket",
    "a drone that flies my mail to the neighbour",
    "a drone that flies a cup of coffee from the kitchen to my desk",
    "Pick M3 screws out of a parts tray and drop them into the holes of a 3D-printed jig",
    "Carry a full 5 kg laundry basket from my bedroom to the washing machine down the hall",
)


class NoFakeResponse(LLMError):
    """Strict fake backend: no canned answer for this prompt."""


def _norm(prompt: str) -> str:
    return " ".join(prompt.lower().split())


def has_fake_response(prompt: str) -> bool:
    return _norm(prompt) in {_norm(p) for p in FAKE_PROMPTS}


def _load(name: str) -> dict:
    return json.loads(resources.files("taskbot").joinpath(f"data/fake/{name}.json").read_text("utf-8"))


def scenario(prompt: str) -> str:
    p = prompt.lower()
    if any(re.search(rf"\b{w}\b", p) for w in OUT_OF_SCOPE_WORDS):
        return "out_of_scope"
    if "laundry" in p:
        return "mobile_stuck"
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


def fake_client(catalog: dict[str, Motor], strict: bool = False) -> FakeClient:
    """`catalog`: the motors the fake may pick from when repairing an arm (supported ones).
    `strict`: answer only FAKE_PROMPTS; raise NoFakeResponse for anything else."""

    def requirements(system, messages, schema):
        prompt = messages[0]["content"]
        if strict and not has_fake_response(prompt):
            raise NoFakeResponse(f"no fake response for: {prompt}")
        kind = scenario(prompt)
        if kind == "arm":
            req = ARM_REQUIREMENTS
        elif kind == "mobile_stuck":
            req = _load("mobile_requirements") | {
                "base_payload_kg": 6.25, "travel_distance_m": 20.0,
                "assumptions": ["FAKE BACKEND: canned requirements for this prompt; only task_summary echoes it.",
                                "a 5 kg laundry basket plus 25% margin = 6.25 kg on the base"]}
        else:
            req = _load(f"{kind}_requirements")
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
        if kind == "mobile" and "laundry" in prompt.lower():
            return _load("mobile_design_round1")  # the stuck scenario: never changes the design
        return _load(f"{kind}_design_round1" if first else f"{kind}_design_round2")

    return FakeClient({"Requirements": requirements, "RobotSpec": design})
