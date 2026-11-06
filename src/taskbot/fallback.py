"""Requirements when the LLM's replies can't be used, so every run still finishes.

    requirements_fallback(prompt, last_data)
        1. the last reply that parsed as JSON, normalised (taskbot.normalize) and filled with the
           documented defaults, if it then validates;
        2. otherwise rule-based requirements from keywords in the prompt (like the
           text-to-robot example's interpreter): flying, swimming, legs, stairs -> out of scope;
           pick up / collect / tidy + floor / room / around -> mobile_manipulator;
           carry / deliver / transport / drive / move ... across -> mobile; anything else (desk,
           windowsill, table, shelf) -> a fixed arm with a 0.3 m-wide workspace in front and a
           0.2 kg payload.
    Both are marked in the requirements' assumptions and in the report.
"""

from __future__ import annotations

import re

from taskbot.normalize import DEFAULT_ARM_PAYLOAD_KG, DEFAULT_WORKSPACE, normalize_requirements
from taskbot.requirements import OUT_OF_SCOPE_PREFIX, Requirements

FALLBACK_MARK = "REQUIREMENTS FALLBACK:"
_OUT_OF_SCOPE = (r"\bfl(y|ies|ying)\b|\bdrones?\b|\bswim", r"\blegs?\b|\blegged\b|\bwalk(s|ing)?\b|\bstairs\b",
                 r"\bhumanoid\b")
_MM = re.compile(r"\b(pick(s|ing)? up|picks?|collect\w*|tidy\w*|gather\w*|fetch\w*)\b.*\b(floor|room|around|house|"
                 r"basket|bring)\b|\b(floor|room)\b.*\b(pick|collect|tidy)", re.S)
_MOBILE = re.compile(r"\b(carr(y|ies|ying)|deliver\w*|transport\w*|drive\w*|patrol\w*|follow\w*|across)\b")


def rule_based(prompt: str) -> dict:
    p = prompt.lower()
    note = (f"{FALLBACK_MARK} the model's replies couldn't be used, so TaskBot guessed these requirements "
            "from keywords in the task; check them")
    base = {"task_summary": prompt.strip() or "unnamed task", "position_tolerance_m": 0.005,
            "tolerance_reason": "not known: TaskBot's default of 5 mm", "assumptions": [note]}
    for pattern in _OUT_OF_SCOPE:
        if re.search(pattern, p):
            base["assumptions"].append(f"{OUT_OF_SCOPE_PREFIX} the task needs a robot that flies, swims, walks or "
                                       "climbs stairs; TaskBot designs fixed arms and wheeled robots")
            return base | {"robot_type": "arm", "payload_kg": DEFAULT_ARM_PAYLOAD_KG, "gripper_needed": True,
                           "gripper_kind": "parallel", "workspace": dict(DEFAULT_WORKSPACE), "dof_min": 3}
    if _MM.search(p):
        return base | {"robot_type": "mobile_manipulator", "payload_kg": DEFAULT_ARM_PAYLOAD_KG, "gripper_needed": True,
                       "gripper_kind": "parallel", "dof_min": 3,
                       "workspace": {"x_min": 0.25, "x_max": 0.4, "y_min": -0.1, "y_max": 0.1,
                                     "z_min": 0.02, "z_max": 0.3}}
    if _MOBILE.search(p):
        m = re.search(r"(\d+(?:\.\d+)?)\s*kg", p)
        payload = float(m[1]) * 1.25 if m else 0.0
        if m:
            base["assumptions"].append(f"the task mentions {m[1]} kg: carried on the base, x1.25 margin")
        return base | {"robot_type": "mobile", "payload_kg": 0.0, "gripper_needed": False,
                       "base_payload_kg": round(payload, 3)}
    return base | {"robot_type": "arm", "payload_kg": DEFAULT_ARM_PAYLOAD_KG, "gripper_needed": True,
                   "gripper_kind": "parallel", "workspace": dict(DEFAULT_WORKSPACE), "dof_min": 3}


def requirements_fallback(prompt: str, last_data: dict | None) -> tuple[Requirements, str, list[str]]:
    """(requirements, what was used, normaliser notes)."""
    if isinstance(last_data, dict):
        from pydantic import ValidationError

        d = dict(last_data)
        notes: list[str] = []
        for _ in range(3):  # drop what still fails; the normaliser's defaults take its place
            d, more = normalize_requirements(d, prompt)
            notes += more
            try:
                req = Requirements(**(d | {"assumptions": list(d.get("assumptions") or []) + [
                    f"{FALLBACK_MARK} the model's last reply didn't pass validation; TaskBot normalised it and "
                    "filled the rest with its documented defaults"]}))
                return req, "the model's last parsed reply, normalised and filled with defaults", notes
            except ValidationError as exc:
                bad = {str(e["loc"][0]) for e in exc.errors() if e["loc"]}
                if not bad:
                    break
                for k in bad:
                    d.pop(k, None)
                notes.append(f"dropped {', '.join(sorted(bad))}: still invalid after normalising")
    d, notes = normalize_requirements(rule_based(prompt), prompt)
    return Requirements(**d), "rule-based requirements from keywords in the task (no usable reply)", notes
