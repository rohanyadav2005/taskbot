import ctypes.util
import json
import os
import sys
from pathlib import Path

import pytest

# Headless Linux: pick an offscreen GL backend for the recording tests before
# mujoco is imported. Tests that need rendering skip if none works.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "osmesa" if ctypes.util.find_library("OSMesa") else "egl")

EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"
EXAMPLES = sorted(EXAMPLES_DIR.glob("*.json"))


@pytest.fixture
def spec_dict():
    """A fresh, mutable copy of the 2-DOF example to break in tests."""
    return json.loads((EXAMPLES_DIR / "arm_2dof.json").read_text())


DESK_REQ = {
    "task_summary": "Move a 200 g cup around a desk.",
    "robot_type": "arm",
    "reach_m": 0.3,
    "payload_kg": 0.25,
    "workspace": {"x_min": 0.1, "x_max": 0.25, "y_min": -0.1, "y_max": 0.1, "z_min": 0.05, "z_max": 0.2},
    "dof_min": 4,
    "gripper_needed": True,
    "gripper_kind": "parallel",
    "assumptions": [],
}


@pytest.fixture
def desk_req():
    from taskbot.requirements import Requirements

    return Requirements(**DESK_REQ)


def example_dict(name: str) -> dict:
    return json.loads((EXAMPLES_DIR / f"{name}.json").read_text())


def with_motor(spec: dict, joint_name: str, motor_id: str) -> dict:
    """Copy of a spec dict with one joint's motor swapped."""
    spec = json.loads(json.dumps(spec))
    for module in spec["modules"]:
        for seg in module.get("segments", []):
            if seg["joint"]["name"] == joint_name:
                seg["joint"]["motor_id"] = motor_id
    return spec


def strong_arm() -> dict:
    """4-DOF example with XM540s at shoulder and elbow: passes every check and sim test."""
    return with_motor(with_motor(example_dict("arm_4dof"), "shoulder_pitch", "xm540_w270"),
                      "elbow_pitch", "xm540_w270")
