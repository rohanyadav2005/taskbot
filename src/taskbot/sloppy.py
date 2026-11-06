"""A "sloppy model" backend for tests: the mistakes real 7B models make, seeded.

SloppyClient(seed) picks a task (SCENARIOS, by weight) and answers like a weak local model:
mismatched reach, a workspace centred on the base, reversed min/max, the payload margin
applied twice, fields of the other robot type filled in, a missing gripper_kind, unknown
fields, numbers as strings, prose around the JSON, invalid JSON on the first attempt; intents
with the wrong robot_type, misspelled layouts and extra fields. The scene numbers themselves
are sensible (a few variants per task), so a robust pipeline should design nearly every run.
"""

from __future__ import annotations

import json
import random

from pydantic import BaseModel

from taskbot.llm import LLMClient, Message, Usage

# (weight, prompt, requirements as a careful model would write them)
SCENARIOS = [
    (6, "water the three plants on my windowsill",
     {"robot_type": "arm", "payload_kg": 0.34, "gripper_needed": True, "gripper_kind": "parallel", "dof_min": 4,
      "workspace": {"x_min": 0.12, "x_max": 0.3, "y_min": -0.15, "y_max": 0.15, "z_min": 0.02, "z_max": 0.28},
      "position_tolerance_m": 0.01, "tolerance_reason": "within a cm of the pot"}),
    (5, "a desk arm that picks M3 screws out of a tray and drops them into a jar next to it",
     {"robot_type": "arm", "payload_kg": 0.01, "gripper_needed": True, "gripper_kind": "parallel", "dof_min": 4,
      "workspace": {"x_min": 0.12, "x_max": 0.25, "y_min": -0.12, "y_max": 0.12, "z_min": 0.0, "z_max": 0.15},
      "position_tolerance_m": 0.002, "tolerance_reason": "M3 screw holes"}),
    (5, "a robot that carries a 2 kg box across my office",
     {"robot_type": "mobile", "payload_kg": 0.0, "gripper_needed": False, "speed_m_s": 0.4, "max_incline_deg": 3,
      "terrain": "carpet", "base_payload_kg": 2.5, "position_tolerance_m": 0.05, "tolerance_reason": "near a desk"}),
    (5, "a robot that picks toys off the floor and drops them in a basket",
     {"robot_type": "mobile_manipulator", "payload_kg": 0.15, "gripper_needed": True, "gripper_kind": "parallel",
      "dof_min": 3, "workspace": {"x_min": 0.25, "x_max": 0.4, "y_min": -0.1, "y_max": 0.1, "z_min": 0.02,
                                  "z_max": 0.3},
      "speed_m_s": 0.3, "max_incline_deg": 2, "terrain": "carpet", "base_payload_kg": 0.0,
      "position_tolerance_m": 0.01, "tolerance_reason": "toys are a few cm across"}),
    (4, "something to help my grandmother reach things on a high shelf",
     {"robot_type": "arm", "payload_kg": 0.3, "gripper_needed": True, "gripper_kind": "parallel", "dof_min": 4,
      "workspace": {"x_min": 0.15, "x_max": 0.3, "y_min": -0.12, "y_max": 0.12, "z_min": 0.12, "z_max": 0.4},
      "position_tolerance_m": 0.01, "tolerance_reason": "jars are several cm wide"}),
    (1, "a flying drone that delivers coffee to my room",
     {"robot_type": "mobile", "payload_kg": 0.0, "gripper_needed": False, "speed_m_s": 2.0, "max_incline_deg": 0,
      "terrain": "flat", "base_payload_kg": 0.4, "position_tolerance_m": 0.05, "tolerance_reason": "a room",
      "assumptions": ["OUT OF SCOPE: it must fly"]}),
]
VARIANTS = (0.9, 1.0, 1.1)


def pick(seed: int) -> tuple[str, dict]:
    rng = random.Random(seed)
    prompt, req = rng.choices([(p, r) for _, p, r in SCENARIOS], weights=[w for w, _, _ in SCENARIOS])[0]
    k = rng.choice(VARIANTS)
    req = json.loads(json.dumps(req))
    if req.get("workspace"):
        req["workspace"] = {a: round(v * k, 3) for a, v in req["workspace"].items()}
    req["payload_kg"] = round(req["payload_kg"] * k, 3)
    return prompt, req | {"task_summary": prompt}


def _mess_up_requirements(d: dict, rng: random.Random) -> dict:
    d = json.loads(json.dumps(d))
    arm = d.get("workspace") is not None
    if arm:
        d["reach_m"] = round(rng.uniform(0.15, 0.9), 2)                       # mismatched reach
        ws = d["workspace"]
        if rng.random() < 0.3:                                                # centred on the base
            half = (ws["x_max"] - ws["x_min"]) / 2
            ws["x_min"], ws["x_max"] = -half, half
        if rng.random() < 0.3:                                                # reversed min/max
            axis = rng.choice("xyz")
            ws[f"{axis}_min"], ws[f"{axis}_max"] = ws[f"{axis}_max"], ws[f"{axis}_min"]
        if rng.random() < 0.3:                                                # mobile fields on an arm
            d |= {"speed_m_s": 0.5, "terrain": "carpet", "max_incline_deg": 4}
        if rng.random() < 0.3:                                                # missing gripper_kind
            d.pop("gripper_kind", None)
        if rng.random() < 0.3:                                                # margin applied twice
            d["payload_kg"] = round(d["payload_kg"] * 1.25, 3)
            d.setdefault("assumptions", []).append("payload includes the 25% safety margin")
    else:
        if rng.random() < 0.4:                                                # arm fields on a mobile robot
            d |= {"reach_m": 0.4, "dof_min": 3, "gripper_needed": True}
        if rng.random() < 0.3:
            for k in ("max_incline_deg", "terrain"):                         # missing mobile fields
                d.pop(k, None)
    if rng.random() < 0.4:                                                    # unknown fields
        d |= {"color": "red", "notes": "keep it quiet", "units": "metric"}
    if rng.random() < 0.4:                                                    # numbers as strings
        d["payload_kg"] = f"{d['payload_kg']} kg"
        d["position_tolerance_m"] = str(d["position_tolerance_m"])
    if rng.random() < 0.2:
        d.pop("position_tolerance_m", None)
    if rng.random() < 0.2:
        d["robot_type"] = {"arm": "Robotic Arm", "mobile": "mobile robot",
                           "mobile_manipulator": "mobile manipulator"}.get(d["robot_type"], d["robot_type"])
    return d


def _mess_up_intent(req: dict, rng: random.Random) -> dict:
    d = {"robot_type": rng.choice(["arm", "mobile", "mobile_manipulator"]),             # often wrong
         "gripper": rng.choice(["parallel", "suction", "none", "claw"]),
         "rationale": "It should work."}
    if rng.random() < 0.7:
        d["arm_layout"] = rng.choice(["yaw_pitch_pitch", "yaw-pitch-pitch", "6dof", "yaw_pitch_pitch_pitch"])
    if rng.random() < 0.5:
        d["drive"] = rng.choice(["differential", "tank", "four_wheel"])
    if rng.random() < 0.4:
        d |= {"wheels": 3, "link_length_m": 0.3, "motor": "xm540_w270"}                # extra fields, numbers
    if rng.random() < 0.3:
        d["caster"] = "yes"
    return d


def _dress(obj: dict, rng: random.Random) -> str:
    text = json.dumps(obj, indent=rng.choice([None, 2]))
    if rng.random() < 0.4:  # prose around the JSON
        text = f"Sure! Here is the JSON you asked for:\n```json\n{text}\n```\nLet me know if you need changes."
    return text


class SloppyClient(LLMClient):
    provider = "sloppy"

    def __init__(self, seed: int):
        super().__init__(max_retries=3, rate_limit_retries=0)
        self.seed = seed
        self.prompt, self.truth = pick(seed)
        self.rng = random.Random(seed * 7919 + 1)
        self.usage = Usage("sloppy", f"seed-{seed}")
        self._attempts: dict[str, int] = {}

    def _chat(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        n = self._attempts[self._step] = self._attempts.get(self._step, 0) + 1
        if n == 1 and self.rng.random() < 0.3:  # invalid JSON on the first attempt
            return '{"robot_type": "arm", "workspace": {"x_min": 0.1, '
        if schema.__name__ == "Requirements":
            return _dress(_mess_up_requirements(self.truth, self.rng), self.rng)
        if schema.__name__ == "DesignIntent":
            return _dress(_mess_up_intent(self.truth, self.rng), self.rng)
        return "I can't write a full robot spec."  # llm mode: not what this backend is for


def sloppy_client(seed: int) -> SloppyClient:
    return SloppyClient(seed)


