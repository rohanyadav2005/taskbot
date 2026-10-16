"""Command line entry point.

    taskbot design "a desk arm that waters my plants" [--backend anthropic|ollama|fake]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from taskbot.fake import fake_client
from taskbot.llm import LLMClient, LLMError, make_client
from taskbot.pipeline import run_pipeline
from taskbot.repair import DEFAULT_MAX_ROUNDS, RepairOutcome
from taskbot.report import verdict
from taskbot.requirements import OUT_OF_SCOPE_PREFIX
from taskbot.schema import ArmChain, Gripper, Motor, WheeledBase, load_motor_catalog, supported_motors

BACKENDS = ("anthropic", "ollama", "fake")

def get_client(backend: str, model: str | None, catalog: dict[str, Motor]) -> LLMClient:
    return fake_client(supported_motors(catalog)) if backend == "fake" else make_client(backend, model)


def summarize(outcome: RepairOutcome, catalog: dict[str, Motor], out: Path) -> str:
    req, final = outcome.requirements, outcome.final
    spec = final.spec
    segments = [s for m in spec.modules if isinstance(m, ArmChain) for s in m.segments]
    gripper = next((m for m in spec.modules if isinstance(m, Gripper)), None)
    needs = []
    if req.has_arm:
        needs.append(f"reach {req.reach_m:.3f} m | payload {req.payload_kg:.3f} kg | DOF >= {req.dof_min}"
                     f" | gripper {req.gripper_kind or 'none'}")
    if req.is_mobile:
        needs.append(f"speed {req.speed_m_s:g} m/s | incline {req.max_incline_deg:g}° | {req.terrain}"
                     f" | {req.base_payload_kg:g} kg on the base")
    base = spec.base
    lines = [
        f"Task: {req.task_summary}",
        "",
        f"Requirements ({req.robot_type})",
        *[f"  {n}" for n in needs],
        *([f"  budget ${req.max_budget_usd:.0f}"] if req.max_budget_usd else []),
        "  assumptions:",
        *[f"    {'!! ' if a.startswith(OUT_OF_SCOPE_PREFIX) else '- '}{a}" for a in req.assumptions],
        "",
        f"Design: {spec.name} (round {final.number})",
        f"  {spec.description}",
        f"  joints ({len(segments)}):",
        *[f"    {s.joint.name:<18} {s.joint.type:<10} axis {s.joint.axis}  {s.joint.motor_id:<12}"
          f" -> {s.link.name} ({s.link.length:.3f} m {s.link.material})" for s in segments],
        f"  gripper: {gripper.gripper_type} {gripper.name}" if gripper else "  gripper: none",
        *([f"  base: {base.drive} drive, {base.wheel_motor_id} wheels r={base.wheel_radius_m:g} m, "
           f"track {base.track_width_m:g} m, wheelbase {base.wheelbase_m:g} m"
           + (", caster" if base.caster else "")] if isinstance(base, WheeledBase) else []),
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
        result = run_pipeline(client, args.prompt, catalog, args.out, max_rounds=args.max_rounds,
                              simulate=not args.no_sim, record=not args.no_record,
                              on_event=lambda line: print(line, file=sys.stderr))
    except LLMError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if result.outcome is None:
        print(f"OUT OF SCOPE: no design was made for: {result.requirements.task_summary}")
        for reason in result.out_of_scope:
            print(f"  {reason}")
        print(f"Saved to {result.out_dir}/ : report.md, requirements.json")
        return 3
    print(summarize(result.outcome, catalog, result.out_dir))
    return 0 if result.passed else 2


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
