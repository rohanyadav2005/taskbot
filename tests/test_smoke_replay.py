"""Task 9: the six real smoke-test transcripts (qwen2.5:7b-instruct via Ollama on a laptop CPU,
2026-11-06, tests/fixtures/real_runs/smoke_2026-11-06) replayed through today's code. Each test
says what went wrong in the run and what happens now. A replay answers only the steps the run
recorded; where today's code asks something the run never did, the replay has no reply and code's
documented fallback stands in (said in the test)."""

import json
from pathlib import Path

import pytest

from taskbot.cli import main
from taskbot.repair import Round
from taskbot.report import render_report, verdict
from taskbot.timing import KINDS


pytestmark = pytest.mark.slow
SMOKE = Path(__file__).parent / "fixtures" / "real_runs" / "smoke_2026-11-06"


def replay(name: str, prompt: str, tmp_path, *extra):
    code = main(["design", prompt, "--backend", "replay", "--transcript", str(SMOKE / f"{name}.jsonl"),
                 "--out", str(tmp_path), "--no-record", *extra])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    req = json.loads((out / "requirements.json").read_text(encoding="utf-8"))
    return code, out, run, report, req


def checks(out: Path) -> dict[str, dict]:
    last = sorted((out / "rounds").iterdir(), key=lambda p: int(p.name.split("_")[1]))[-1]
    return {c["check"]: c for c in json.loads((last / "checks.json").read_text(encoding="utf-8"))}


def test_1_windowsill_passes_every_check_and_sim(tmp_path):
    """Then INFEASIBLE: 3.15 Nm at the shoulder for 2.83 usable, and self_collision 14.8%. The
    workspace (x -0.15 to 0.35) surrounded the base; raising z_min left targets in the turret
    column, and reach_m (0.48 m) came from a shoulder 0.15 m up, not the sizer's 0.2 m."""
    code, out, run, report, req = replay("1_windowsill", "water the three plants on my windowsill", tmp_path)
    assert code == 0 and run["status"] == "PASSED" and run["calls"] == 2
    ws = req["workspace"]
    assert (ws["x_min"], ws["x_max"], ws["z_min"]) == (0.1, 0.35, 0.05) and req["reach_m"] == 0.451
    assert req["payload_kg"] == 0.375  # 250 ml of water and a 50 g cup, x 1.25: unchanged
    assert "upper arm 0.07 m shorter than the even split" in report
    c = checks(out)
    assert all(r["passed"] for r in c.values())
    last = sorted((out / "rounds").iterdir())[-1] / "checks.json"
    torque = next(r for r in json.loads(last.read_text(encoding="utf-8"))
                  if r["check"] == "static_torque" and r["message"].startswith("shoulder_pitch"))
    # The check is unchanged (80% of xm540_w270's rated torque); the shoulder needs 2.82 of 2.83 Nm.
    assert torque["value"] <= torque["limit"] and torque["limit"] == pytest.approx(2.83, abs=0.005)
    assert {"reach_targets", "carry_payload", "hold_still", "self_collision"} <= set(c)


def test_2_m3_screws_meet_2_mm_with_the_stiffest_motors(tmp_path):
    """Then FAILED: self_collision 36-40% (the workspace +-0.15 m around the base, z_min raised).
    Now the box is trimmed to the same targets in front; at 2 mm the sizer picks the stiffest
    motors, and the first intent passes, so the run's 3 intent revisions are never asked for."""
    code, out, run, report, req = replay(
        "2_m3_screws", "a desk arm that picks M3 screws out of a tray and drops them into a jar next to it", tmp_path)
    assert code == 0 and run["status"] == "PASSED" and run["calls"] == 2
    assert req["position_tolerance_m"] == 0.002 and req["workspace"]["x_min"] == 0.1
    assert "stiffest that passes, for the tight tolerance" in report
    c = checks(out)
    assert c["reach_targets"]["passed"] and c["carry_payload"]["passed"] and c["self_collision"]["passed"]


def test_3_box_still_passes(tmp_path):
    code, out, run, report, req = replay("3_box", "a robot that carries a 2 kg box across my office", tmp_path)
    assert code == 0 and run["status"] == "PASSED" and req["robot_type"] == "mobile"
    assert set(run["time_by_kind"]) == set(KINDS) | {"other"}
    assert abs(sum(run["time_by_kind"].values()) - run["time_s"]) < 0.05
    kinds = {t["kind"] for t in run["timings"]}
    assert {"llm", "sizing", "static_checks", "sim"} <= kinds
    assert "Time by step: LLM" in report


def test_4_toys_passes_with_the_arm_working_in_front_of_the_chassis(tmp_path):
    """Then INFEASIBLE in 1010 s (912 s of it one LLM call): the +-0.5 m box around the chassis was
    shifted 0.7 m forward, so the gate saw 0.3 kg at 1.4 m. Now the arm covers what it reaches
    once the base stops (x 0.2-0.45 m); the gate still sends 0.5 kg back once (as the run did) and
    the model's 0.3 kg passes. The run never reached the intent step, so the replay has no intent:
    code's default (differential) can't climb the model's 15° carpet, and code's next untried
    intent, four_wheel, is sized with turning verified and a stiffer base yaw."""
    code, out, run, report, req = replay(
        "4_toys", "a robot that picks toys off the floor and drops them in a basket", tmp_path)
    assert code == 0 and run["status"] == "PASSED"
    ws = req["workspace"]
    assert (ws["x_min"], ws["x_max"], ws["y_min"], ws["y_max"]) == (0.2, 0.45, -0.2, 0.2)
    assert req["payload_kg"] == 0.3 and "drives to each target" in report
    assert any(f.startswith("intent: no usable intent") for f in run["fallbacks"])
    assert any("code tried its next untried intent instead" in f and "four_wheel" in f for f in run["fallbacks"])
    assert "turns in place in simulation" in report and "stiffer than mg996r" in report
    assert all(r["passed"] for r in checks(out).values())


def test_5_high_shelf_is_infeasible_and_says_the_shelf_was_cut(tmp_path):
    """Then PASSED with a desk arm 0.25 m tall: the normaliser and the validator measured reach
    from different shoulders, so all 4 replies (each with z 1.5-2.5 m) were rejected, 3 retries
    took 2160 s, and rule-based requirements were designed. Now the first reply validates; the
    shelf (z up to 2.5 m) is cut to 1.5 m and marked; no catalog arm holds 0.5 kg that far out.
    The run had no feasibility retry, so the replay keeps the first requirements."""
    code, out, run, report, req = replay(
        "5_grandmother_shelf", "something to help my grandmother reach things on a high shelf", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE" and run["json_retries"] == 0
    assert req["robot_type"] == "mobile_manipulator" and req["workspace"]["z_max"] == 1.5
    assert "The requirements were also cut short: targets up to z_max = 2.5 m were cut to 1.5 m" in run["reason"]
    assert "Code had already cut these requirements short of the task" in report
    assert not any("rule-based" in f for f in run["fallbacks"])


def test_6_drone_is_stopped_by_the_scope_gate(tmp_path):
    """Then FAILED: qwen wrote "OUT_OF_SCOPE: ..." as robot_type; the normaliser read it as
    'mobile' and a wheeled robot was designed (turn_in_place saturated)."""
    code, out, run, report, req = replay("6_drone", "a flying drone that delivers coffee to my room", tmp_path)
    assert code == 3 and run["status"] == "OUT_OF_SCOPE" and run["calls"] == 0
    assert '- flying: "flying", "drone"' in report


# --------------------------------------------------------------------------- what a FAILED or
# precision-limited run reports


class _Run:
    def __init__(self, achieved):
        self.achieved_tolerance_m = achieved


def _outcome(tol, failures, achieved):
    from taskbot.checks import CheckResult
    from taskbot.hybrid import HybridOutcome
    from taskbot.normalize import normalize_requirements
    from taskbot.requirements import Requirements
    from taskbot.schema import RobotSpec
    from conftest import example_dict

    req = Requirements(**normalize_requirements({
        "task_summary": "pick screws", "robot_type": "arm", "payload_kg": 0.01, "gripper_needed": True,
        "gripper_kind": "parallel", "dof_min": 3, "position_tolerance_m": tol, "tolerance_reason": "r",
        "workspace": {"x_min": 0.15, "x_max": 0.3, "y_min": -0.1, "y_max": 0.1, "z_min": 0.05, "z_max": 0.2}}, "x")[0])
    spec = RobotSpec.model_validate(example_dict("arm_4dof"))
    results = [CheckResult(check=c, passed=False, value=v, limit=lim, cause=cause, message=f"{c} failed")
               for c, v, lim, cause in failures]
    out = HybridOutcome(requirements=req, max_rounds=1)
    out.rounds = [Round(1, spec, results, [_Run(a) for a in achieved])]
    out.stop_reason = "automatic fixes and 3 intent revisions didn't fix it"
    return out, req


def test_tight_tolerance_missed_only_through_compliance_is_infeasible_with_the_best_precision():
    from taskbot.hybrid import _precision_limit

    out, req = _outcome(0.001, [("reach_targets", 0.0021, 0.001, "compliance")], [0.0021, 0.0034, 0.0004])
    _precision_limit(out, req)
    assert out.infeasible_kind == "precision" and "holds about 3.4 mm" in out.infeasible[0]
    assert "within the task's 1 mm" in out.infeasible[0] and "about 3.4 mm" in out.infeasible_options[0]
    assert verdict(out).startswith("INFEASIBLE: no design TaskBot can size from this catalog holds the tool within")
    loose, req = _outcome(0.01, [("reach_targets", 0.012, 0.01, "compliance")], [0.012])
    _precision_limit(loose, req)  # not a tight tolerance: the sizer didn't try the stiffest motors
    assert not loose.infeasible
    saturated, req = _outcome(0.001, [("carry_payload", 0.5, 0.2, "saturated")], [0.003])
    _precision_limit(saturated, req)  # a torque problem, not precision
    assert not saturated.infeasible


def test_failed_report_says_which_check_blocked_it_and_by_how_much():
    out, _ = _outcome(0.01, [("self_collision", 0.359, 0.05, None)], [])
    text = render_report(out)
    assert "## What blocked it" in text
    assert "- **self_collision**: 0.359 against a limit of 0.05 (7.2 x the limit). self_collision failed" in text
    assert "Why TaskBot stopped: automatic fixes and 3 intent revisions didn't fix it." in text
