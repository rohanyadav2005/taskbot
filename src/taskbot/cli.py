"""Command line entry point.

    taskbot design "a desk arm that waters my plants" [--backend anthropic|ollama|fake]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from importlib import resources
from pathlib import Path

from lxml import etree

from taskbot.build_urdf import build_urdf
from taskbot.design import design_robot
from taskbot.llm import FakeClient, LLMClient, LLMError, make_client
from taskbot.requirements import OUT_OF_SCOPE_PREFIX, Requirements, extract_requirements
from taskbot.schema import ArmChain, Gripper, Motor, RobotSpec, load_motor_catalog

BACKENDS = ("anthropic", "ollama", "fake")


def fake_client() -> FakeClient:
    """Offline demo: canned requirements (echoing the prompt) and a canned 4-DOF arm."""

    def requirements(system, messages, schema):
        return {
            "task_summary": messages[0]["content"],
            "robot_type": "arm",
            "reach_m": 0.33,
            "payload_kg": 0.25,
            "workspace": {"x_min": -0.3, "x_max": 0.3, "y_min": -0.3, "y_max": 0.3,
                          "z_min": 0.0, "z_max": 0.3},
            "dof_min": 4,
            "gripper_needed": True,
            "gripper_kind": "parallel",
            "max_budget_usd": None,
            "assumptions": ["FAKE BACKEND: canned requirements, only task_summary reflects the prompt."],
        }

    design = json.loads(
        resources.files("taskbot").joinpath("data/fake_design.json").read_text("utf-8")
    )
    return FakeClient({"Requirements": requirements, "RobotSpec": lambda *_: design})


def get_client(backend: str, model: str | None) -> LLMClient:
    return fake_client() if backend == "fake" else make_client(backend, model)


def unique_dir(root: Path, name: str) -> Path:
    path, n = root / name, 2
    while path.exists():
        path, n = root / f"{name}_{n}", n + 1
    return path


def summarize(req: Requirements, spec: RobotSpec, urdf: str, catalog: dict[str, Motor], out: Path) -> str:
    segments = [s for m in spec.modules if isinstance(m, ArmChain) for s in m.segments]
    link_sum = sum(s.link.length for s in segments)
    motor_ids = [s.joint.motor_id for s in segments]
    motor_ids += [m.motor_id for m in spec.modules if isinstance(m, Gripper) and m.motor_id]
    cost = sum(catalog[i].price_usd for i in motor_ids)
    mass = sum(float(m.get("value")) for m in etree.fromstring(urdf.encode()).iter("mass"))
    gripper = next((m for m in spec.modules if isinstance(m, Gripper)), None)

    lines = [
        f"Task: {req.task_summary}",
        "",
        "Requirements",
        f"  reach {req.reach_m:.3f} m | payload {req.payload_kg:.3f} kg | DOF >= {req.dof_min}"
        f" | gripper {req.gripper_kind or 'none'}"
        + (f" | budget ${req.max_budget_usd:.0f}" if req.max_budget_usd else ""),
        "  assumptions:",
        *[f"    {'!! ' if a.startswith(OUT_OF_SCOPE_PREFIX) else '- '}{a}" for a in req.assumptions],
        "",
        f"Design: {spec.name}",
        f"  {spec.description}",
        f"  joints ({len(segments)}):",
        *[f"    {s.joint.name:<18} {s.joint.type:<10} axis {s.joint.axis}  {s.joint.motor_id:<12}"
          f" -> {s.link.name} ({s.link.length:.3f} m {s.link.material})" for s in segments],
        f"  gripper: {gripper.gripper_type} {gripper.name}" if gripper else "  gripper: none",
        f"  link length sum {link_sum:.3f} m vs reach {req.reach_m:.3f} m (not checked yet)",
        f"  total mass {mass:.3f} kg | motor cost ${cost:.2f}"
        + (f" (budget ${req.max_budget_usd:.0f})" if req.max_budget_usd else ""),
        "",
        f"Saved to {out}/ : requirements.json, spec.json, robot.urdf",
    ]
    return "\n".join(lines)


def cmd_design(args: argparse.Namespace) -> int:
    catalog = load_motor_catalog()
    try:
        client = get_client(args.backend, args.model)
        print(f"[1/3] requirements ({args.backend})...", file=sys.stderr)
        req = extract_requirements(client, args.prompt)
        print("[2/3] design...", file=sys.stderr)
        spec = design_robot(client, req, catalog)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print("[3/3] URDF...", file=sys.stderr)
    urdf = build_urdf(spec, catalog)

    out = unique_dir(args.out, spec.name)
    out.mkdir(parents=True)
    (out / "requirements.json").write_text(req.model_dump_json(indent=2) + "\n", "utf-8")
    (out / "spec.json").write_text(spec.model_dump_json(indent=2, exclude_none=True) + "\n", "utf-8")
    (out / "robot.urdf").write_text(urdf, "utf-8")
    print(summarize(req, spec, urdf, catalog, out))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="taskbot")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("design", help="prompt -> requirements -> spec -> URDF")
    d.add_argument("prompt", help="what the robot must do, in plain English")
    d.add_argument("--backend", choices=BACKENDS,
                   default=os.environ.get("TASKBOT_BACKEND", "anthropic"))
    d.add_argument("--model", help="model name (default: TASKBOT_MODEL or the backend's default)")
    d.add_argument("--out", type=Path, default=Path("out"), help="output root (default: out/)")
    args = parser.parse_args(argv)
    return cmd_design(args)


if __name__ == "__main__":
    sys.exit(main())
