"""Save a repair run to disk and write report.md."""

from __future__ import annotations

import difflib
import json
from pathlib import Path

from taskbot.build_urdf import build_urdf
from taskbot.checks import CheckResult
from taskbot.repair import SIM_TEST_NAMES, RepairOutcome, Round
from taskbot.schema import Motor
from taskbot.sim.control import DEFAULT_PAYLOAD_MASS_ERROR
from taskbot.sim.model import SIM_TIMESTEP_S, build_scene
from taskbot.sim.motor import PID_PRESET_DYNAMIXEL_I
from taskbot.sim.record import render_gif
from taskbot.sim_tests import controller_settings


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

    lines += ["## Requirements", ""]
    lines += requirements_rows(req)
    battery = getattr(final.spec.base, "battery_note", None)
    if battery:
        lines += [f"- Design assumption: {battery}"]

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
        ideal = {run.name: run.ideal for run in rnd.sim_runs if run.ideal is not None}
        for r in rnd.results:
            status = "pass" if r.passed and not _is_warning(r) else ("warn" if r.passed else "**FAIL**")
            lines.append(f"| {r.check} | {status} | {_num(r.value)} | {_num(r.limit)} | {_cell(r.message)} |")
            if r.check in ideal:
                i = ideal[r.check]
                lines.append(f"| {r.check} (ideal) | {'pass' if i.passed else 'fail'} (info) | "
                             f"{_num(i.value)} | {_num(i.limit)} | {_cell(i.message)} |")

    lines += ["", "Simulation tests run only in rounds where every static check passed. "
              "carry_payload and hold_still are judged with the controller believing the payload "
              "is 20% lighter than it is; the \"(ideal)\" rows repeat them with the true mass, "
              "for information only."]
    if req.is_mobile and req.has_arm:
        lines += ["", "**For this mobile manipulator the arm tests (reach_targets, carry_payload, "
                  "hold_still) run with the wheeled base fixed in place**, so they test the arm alone; "
                  "drive_with_arm_extended tests the arm and base together while driving."]

    lines += ["", f"## Controller settings (round {final.number} design)", "",
              "Fixed per motor, never tuned per test (see taskbot/sim/motor.py). Total joint torque = "
              "Kp·e + Kd·(q̇_ref − q̇) + gravity feedforward − b·q̇, limited to the motor's driving "
              "torque at the current speed, and to rated torque when braking. "
              f"Gravity feedforward believes the payload is {DEFAULT_PAYLOAD_MASS_ERROR:.0%} lighter "
              f"than it is (the \"(ideal)\" rows use the true mass). Physics timestep "
              f"{SIM_TIMESTEP_S * 1000:.0f} ms.", "",
              f"**Control mode: {final.spec.control}**"
              + (" (one fixed integral preset on Dynamixels: Position I Gain "
                 f"{PID_PRESET_DYNAMIXEL_I}; the gain conversion is unverified)"
                 if final.spec.control == "pid" else " (factory position loops, no integral term)"), "",
              "| Joint | Motor | Kp (Nm/rad) | Kd (Nm·s/rad) | Ki (Nm/(rad·s)) | Back-EMF b (Nm·s/rad) | Rated / stall (Nm) "
              "| No-load (rad/s) | Gains from | Torque-speed model |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for row in controller_settings(build_scene(final.spec)):
        lines.append(
            f"| {row['joint']} | {row['motor']} | {row['kp']:.3g} | {row['kd']:.3g} | {row['ki']:.3g} | "
            f"{row['back_emf_damping']:.3g} | {row['rated']:.3g} / {row['stall']:.3g} | "
            f"{row['no_load_speed']:.3g} | {_cell(row['gain_source'])} | {_cell(row['speed_model'])} |")
    if final.spec.is_mobile:
        lines += ["", "Wheel rows are velocity loops: Kp is in Nm·s/rad (torque per rad/s of speed "
                  "error) and Ki in Nm/rad."]

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


def requirements_rows(req) -> list[str]:
    """The requirements table and assumptions, as markdown lines."""
    lines = ["| Field | Value |", "|---|---|"]
    rows = {"Task": req.task_summary, "Robot type": req.robot_type}
    if req.has_arm:
        ws = req.workspace
        rows |= {
            "Reach": f"{req.reach_m:.3g} m",
            "Payload in gripper": f"{req.payload_kg:.3g} kg",
            "Workspace": f"x [{ws.x_min:.3g}, {ws.x_max:.3g}] y [{ws.y_min:.3g}, {ws.y_max:.3g}] "
                         f"z [{ws.z_min:.3g}, {ws.z_max:.3g}] m",
            "Min DOF": str(req.dof_min),
            "Gripper": req.gripper_kind or "none",
            "Min joint range": f"{req.min_joint_range_rad:.3g} rad",
        }
    if req.is_mobile:
        rows |= {
            "Speed": f"{req.speed_m_s:.3g} m/s",
            "Max incline": f"{req.max_incline_deg:g}°",
            "Terrain": req.terrain,
            "Payload on base": f"{req.base_payload_kg:.3g} kg",
            "Travel per charge": f"{req.travel_distance_m:g} m" if req.travel_distance_m else "not given",
        }
    rows |= {
        "Budget": f"${req.max_budget_usd:.2f}" if req.max_budget_usd else "none given",
        "Position tolerance": f"{req.position_tolerance_m * 1000:.3g} mm ({req.tolerance_reason})",
    }
    lines += [f"| {k} | {_cell(str(v))} |" for k, v in rows.items()]
    if req.assumptions:
        lines += ["", "Assumptions:", *[f"- {_cell(a)}" for a in req.assumptions]]
    return lines


def write_out_of_scope(req, out_root: Path) -> Path:
    """out/<task>/{requirements.json, report.md} for a task TaskBot won't design."""
    import re

    slug = re.sub(r"[^A-Za-z0-9]+", "_", req.task_summary.lower()).strip("_")[:40] or "out_of_scope"
    out = unique_dir(out_root, slug)
    out.mkdir(parents=True)
    (out / "requirements.json").write_text(req.model_dump_json(indent=2) + "\n", "utf-8")
    lines = [f"# Design report: {slug}", "",
             "**OUT OF SCOPE: no design was made. TaskBot designs fixed arms and wheeled robots; "
             "this task needs something else.**", "", "Why:", *[f"- {_cell(a)}" for a in req.out_of_scope],
             "", "## Requirements (best attempt, for reference)", "", *requirements_rows(req)]
    (out / "report.md").write_text("\n".join(lines) + "\n", "utf-8")
    return out


def unique_dir(root: Path, name: str) -> Path:
    path, n = root / name, 2
    while path.exists():
        path, n = root / f"{name}_{n}", n + 1
    return path


def record_sim(rnd: Round, out: Path) -> dict[str, str | None]:
    """Render the round's sim runs to out/sim/<test>.gif."""
    return {
        run.name: render_gif(run.robot.model, run.frames, out / "sim" / f"{run.name}.gif",
                             look_at=run.look_at, distance=run.camera_distance,
                             track_body=run.track_body)
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
        ideal = [run.ideal.model_dump() for run in rnd.sim_runs if run.ideal is not None]
        if ideal:
            (d / "sim_ideal.json").write_text(json.dumps(ideal, indent=2) + "\n", "utf-8")
    (out / "spec.json").write_text(_spec_json(outcome.final) + "\n", "utf-8")
    (out / "robot.urdf").write_text(build_urdf(outcome.final.spec, catalog), "utf-8")
    recordings = record_sim(outcome.final, out) if record else None
    (out / "report.md").write_text(render_report(outcome, recordings), "utf-8")
    return out
