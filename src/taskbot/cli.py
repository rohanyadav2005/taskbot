"""Command line entry point.

    taskbot design "a desk arm that waters my plants" [--backend anthropic|ollama|fake]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from importlib import resources
from pathlib import Path

from taskbot.checks import TORQUE_SAFETY_FRACTION
from taskbot.llm import FakeClient, LLMClient, LLMError, make_client
from taskbot.repair import DEFAULT_MAX_ROUNDS, RepairOutcome, repair_loop
from taskbot.report import verdict, write_outputs
from taskbot.requirements import OUT_OF_SCOPE_PREFIX, extract_requirements
from taskbot.schema import ArmChain, Gripper, Motor, load_motor_catalog

BACKENDS = ("anthropic", "ollama", "fake")

_TORQUE_FAIL = re.compile(r"^- \[static_torque\] (\w+) needs ([\d.e+-]+) Nm", re.MULTILINE)
_SATURATED = re.compile(r"(\w+) saturated at its rated ([\d.e+-]+) Nm")


def fake_client(catalog: dict[str, Motor]) -> FakeClient:
    """Offline demo. Requirements are canned (echoing the prompt), the first design is a
    canned 4-DOF arm, and 'repair' swaps any motor that fails static_torque, or saturates
    in simulation, for the cheapest catalog motor strong enough. It fixes nothing else."""

    def requirements(system, messages, schema):
        return {
            "task_summary": messages[0]["content"],
            "robot_type": "arm",
            "reach_m": 0.3,
            "payload_kg": 0.25,
            "workspace": {"x_min": 0.1, "x_max": 0.25, "y_min": -0.1, "y_max": 0.1,
                          "z_min": 0.05, "z_max": 0.2},
            "dof_min": 4,
            "gripper_needed": True,
            "gripper_kind": "parallel",
            "max_budget_usd": None,
            "assumptions": ["FAKE BACKEND: canned requirements, only task_summary reflects the prompt."],
        }

    canned = json.loads(
        resources.files("taskbot").joinpath("data/fake_design.json").read_text("utf-8")
    )

    def design(system, messages, schema):
        prompt = messages[-1]["content"]
        if prompt.startswith("Requirements:"):
            return canned
        spec = json.loads(prompt.split("Current spec: ", 1)[1].split("\nRequirements: ", 1)[0])
        needs = [(j, float(n)) for j, n in _TORQUE_FAIL.findall(prompt)]
        needs += [(j, 1.5 * float(rated)) for j, rated in _SATURATED.findall(prompt)]
        for joint_name, need in needs:
            strong = [m for m in catalog.values()
                      if TORQUE_SAFETY_FRACTION * m.rated_torque_nm >= need]
            if not strong:
                continue
            best = min(strong, key=lambda m: m.price_usd)
            for module in spec["modules"]:
                for seg in module.get("segments", []):
                    if seg["joint"]["name"] == joint_name:
                        seg["joint"]["motor_id"] = best.id
            spec["description"] += f" Fake repair: {joint_name} -> {best.id}."
        return spec

    return FakeClient({"Requirements": requirements, "RobotSpec": design})


def get_client(backend: str, model: str | None, catalog: dict[str, Motor]) -> LLMClient:
    return fake_client(catalog) if backend == "fake" else make_client(backend, model)


def summarize(outcome: RepairOutcome, catalog: dict[str, Motor], out: Path) -> str:
    req, final = outcome.requirements, outcome.final
    spec = final.spec
    segments = [s for m in spec.modules if isinstance(m, ArmChain) for s in m.segments]
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
        f"Design: {spec.name} (round {final.number})",
        f"  {spec.description}",
        f"  joints ({len(segments)}):",
        *[f"    {s.joint.name:<18} {s.joint.type:<10} axis {s.joint.axis}  {s.joint.motor_id:<12}"
          f" -> {s.link.name} ({s.link.length:.3f} m {s.link.material})" for s in segments],
        f"  gripper: {gripper.gripper_type} {gripper.name}" if gripper else "  gripper: none",
        "",
        "Rounds: " + "  ".join(
            f"R{r.number} {'pass' if r.passed else f'{len(r.failures)} fail'}" for r in outcome.rounds),
        *[f"  FAIL [{r.check}] {r.message}" for r in final.failures],
        "",
        verdict(outcome),
        f"Saved to {out}/ : report.md, requirements.json, spec.json, robot.urdf, rounds/"
        + (", sim/" if (out / "sim").exists() else ""),
    ]
    return "\n".join(lines)


def cmd_design(args: argparse.Namespace) -> int:
    catalog = load_motor_catalog()
    try:
        client = get_client(args.backend, args.model, catalog)
        print(f"requirements ({args.backend})...", file=sys.stderr)
        req = extract_requirements(client, args.prompt)
        print(f"design + check loop (max {args.max_rounds} rounds)...", file=sys.stderr)
        outcome = repair_loop(client, req, catalog, max_rounds=args.max_rounds,
                              simulate=not args.no_sim)
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if outcome.final.sim_runs and not args.no_record:
        print("recording simulation GIFs...", file=sys.stderr)
    out = write_outputs(outcome, args.out, catalog, record=not args.no_record)
    print(summarize(outcome, catalog, out))
    return 0 if outcome.passed else 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="taskbot")
    sub = parser.add_subparsers(dest="command", required=True)
    d = sub.add_parser("design", help="prompt -> requirements -> design/check/repair -> report")
    d.add_argument("prompt", help="what the robot must do, in plain English")
    d.add_argument("--backend", choices=BACKENDS,
                   default=os.environ.get("TASKBOT_BACKEND", "anthropic"))
    d.add_argument("--model", help="model name (default: TASKBOT_MODEL or the backend's default)")
    d.add_argument("--out", type=Path, default=Path("out"), help="output root (default: out/)")
    d.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    d.add_argument("--no-sim", action="store_true", help="static checks only")
    d.add_argument("--no-record", action="store_true", help="don't render simulation GIFs")
    args = parser.parse_args(argv)
    return cmd_design(args)


if __name__ == "__main__":
    sys.exit(main())
