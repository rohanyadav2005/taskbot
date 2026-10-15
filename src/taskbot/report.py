"""Save a repair run to disk and write report.md."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from taskbot.build_urdf import build_urdf
from taskbot.checks import CheckResult
from taskbot.repair import SIM_TEST_NAMES, RepairOutcome, Round
from taskbot.schema import Motor
from taskbot.sim.record import render_gif


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _num(v: float | None) -> str:
    return "–" if v is None else f"{v:.4g}"


def _is_warning(r: CheckResult) -> bool:
    return r.passed and r.message.startswith("WARNING:")


def _spec_json(rnd: Round) -> str:
    return rnd.spec.model_dump_json(indent=2, exclude_none=True)


def verdict(outcome: RepairOutcome) -> str:
    n = len(outcome.rounds)
    if outcome.passed:
        return f"PASSED: every check passed in round {n} of {outcome.max_rounds}."
    failed = len(outcome.final.failures)
    why = (f"the loop stopped early ({outcome.stop_reason})" if outcome.stop_reason
           else f"it did not pass after {n} rounds, the maximum")
    return (f"FAILED: no design passed all checks; {why}. "
            f"The last design still fails {failed} check(s), listed below. Do not build it as is.")


def render_report(outcome: RepairOutcome, recordings: dict[str, str | None] | None = None) -> str:
    """`recordings` maps sim test name -> None (GIF written to sim/<name>.gif) or why not."""
    req = outcome.requirements
    final = outcome.final
    lines = [f"# Design report: {final.spec.name}", "", f"**{verdict(outcome)}**", ""]

    lines += ["## Requirements", "", "| Field | Value |", "|---|---|"]
    ws = req.workspace
    rows = {
        "Task": req.task_summary,
        "Robot type": req.robot_type,
        "Reach": f"{req.reach_m:.3g} m",
        "Payload": f"{req.payload_kg:.3g} kg",
        "Workspace": f"x [{ws.x_min:.3g}, {ws.x_max:.3g}] y [{ws.y_min:.3g}, {ws.y_max:.3g}] "
                     f"z [{ws.z_min:.3g}, {ws.z_max:.3g}] m",
        "Min DOF": str(req.dof_min),
        "Gripper": req.gripper_kind or "none",
        "Budget": f"${req.max_budget_usd:.2f}" if req.max_budget_usd else "none given",
        "Min joint range": f"{req.min_joint_range_rad:.3g} rad",
    }
    lines += [f"| {k} | {_cell(v)} |" for k, v in rows.items()]
    if req.assumptions:
        lines += ["", "Assumptions:", *[f"- {_cell(a)}" for a in req.assumptions]]

    # Summary: one row per check name, one column per round.
    names = list(dict.fromkeys(r.check for rnd in outcome.rounds for r in rnd.results))
    names += [n for n in SIM_TEST_NAMES if n not in names]
    lines += ["", "## Check results by round", "",
              "| Check | " + " | ".join(f"Round {r.number}" for r in outcome.rounds) + " |",
              "|---|" + "---|" * len(outcome.rounds)]
    for name in names:
        cells = []
        for rnd in outcome.rounds:
            rs = [r for r in rnd.results if r.check == name]
            bad = sum(not r.passed for r in rs)
            if not rs:
                cells.append("not run" if name in SIM_TEST_NAMES else "–")
            elif bad:
                cells.append(f"FAIL ({bad}/{len(rs)})" if len(rs) > 1 else "FAIL")
            else:
                cells.append("warn" if any(map(_is_warning, rs)) else "pass")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")

    for rnd in outcome.rounds:
        lines += ["", f"### Round {rnd.number}: {'all checks pass' if rnd.passed else f'{len(rnd.failures)} failed'}",
                  "", "| Check | Result | Value | Limit | Message |", "|---|---|---|---|---|"]
        for r in rnd.results:
            status = "pass" if r.passed and not _is_warning(r) else ("warn" if r.passed else "**FAIL**")
            lines.append(f"| {r.check} | {status} | {_num(r.value)} | {_num(r.limit)} | {_cell(r.message)} |")

    lines += ["", "Simulation tests run only in rounds where every static check passed."]

    if recordings:
        lines += ["", f"## Simulation recordings (round {final.number})", ""]
        for name, problem in recordings.items():
            lines.append(f"- {name}: " + (f"![{name}](sim/{name}.gif)" if problem is None else problem))

    lines += ["", "## Changes between rounds"]
    if len(outcome.rounds) == 1:
        lines += ["", "Only one round ran, so there are no changes."]
    for a, b in zip(outcome.rounds, outcome.rounds[1:]):
        diff = list(difflib.unified_diff(
            _spec_json(a).splitlines(), _spec_json(b).splitlines(),
            f"round_{a.number}/spec.json", f"round_{b.number}/spec.json", n=2, lineterm=""))
        lines += ["", f"### Round {a.number} → {b.number}", ""]
        lines += ["```diff", *diff, "```"] if diff else ["The spec did not change."]

    lines += ["", "## Final result", "", verdict(outcome)]
    if not outcome.passed:
        lines += ["", "Still failing:", *[f"- [{r.check}] {_cell(r.message)}" for r in final.failures]]
    warnings = [r for r in final.results if _is_warning(r)]
    if warnings:
        lines += ["", "Warnings:", *[f"- [{r.check}] {_cell(r.message)}" for r in warnings]]
    return "\n".join(lines) + "\n"


def unique_dir(root: Path, name: str) -> Path:
    path, n = root / name, 2
    while path.exists():
        path, n = root / f"{name}_{n}", n + 1
    return path


def record_sim(rnd: Round, out: Path) -> dict[str, str | None]:
    """Render the round's sim runs to out/sim/<test>.gif."""
    return {
        run.name: render_gif(run.robot.model, run.frames, out / "sim" / f"{run.name}.gif",
                             look_at=run.look_at, distance=run.camera_distance)
        for run in rnd.sim_runs
    }


def write_outputs(outcome: RepairOutcome, out_root: Path, catalog: dict[str, Motor],
                  record: bool = True) -> Path:
    """out/<name>/{requirements.json, spec.json, robot.urdf, report.md, rounds/round_N/...,
    sim/<test>.gif}. spec.json and robot.urdf are the last round's design, whether or
    not it passed; GIFs are of the last round's sim tests, if they ran.
    """
    out = unique_dir(out_root, outcome.rounds[0].spec.name)
    out.mkdir(parents=True)
    (out / "requirements.json").write_text(outcome.requirements.model_dump_json(indent=2) + "\n", "utf-8")
    for rnd in outcome.rounds:
        d = out / "rounds" / f"round_{rnd.number}"
        d.mkdir(parents=True)
        (d / "spec.json").write_text(_spec_json(rnd) + "\n", "utf-8")
        (d / "checks.json").write_text(
            json.dumps([r.model_dump() for r in rnd.results], indent=2) + "\n", "utf-8")
    (out / "spec.json").write_text(_spec_json(outcome.final) + "\n", "utf-8")
    (out / "robot.urdf").write_text(build_urdf(outcome.final.spec, catalog), "utf-8")
    recordings = record_sim(outcome.final, out) if record else None
    (out / "report.md").write_text(render_report(outcome, recordings), "utf-8")
    return out
