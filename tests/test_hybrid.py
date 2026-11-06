"""The hybrid design mode (the default): the LLM chooses an intent, code sizes, autofix, and a
revised intent only when automatic fixes run out. All offline (fake backend)."""

import json

import pytest
from sizing_cases import arm_cases

from taskbot import hybrid
from taskbot.bench import Record, RunMeta, compare, render_markdown, write_csv
from taskbot.fake import fake_client
from taskbot.intent import DesignIntent
from taskbot.llm import FakeClient
from taskbot.pipeline import run_pipeline
from taskbot.requirements import Requirements
from taskbot.schema import RobotSpec, load_motor_catalog, supported_motors
from taskbot.sizing import SizedDesign, SizingFailure, size_design

CATALOG = load_motor_catalog()


def run(prompt, tmp_path, **kw):
    catalog = supported_motors(CATALOG)
    events = []
    result = run_pipeline(fake_client(catalog), prompt, CATALOG, tmp_path, record=False,
                          on_event=events.append, **kw)
    return result, events


def test_hybrid_arm_end_to_end_with_sims(tmp_path):
    result, events = run("Water the three potted plants on my windowsill", tmp_path)
    assert result.passed and result.design_mode == "hybrid" and result.approximate is None
    o = result.outcome
    assert len(o.intents) == 1 and o.intents[0].arm_layout == "yaw_pitch_pitch_pitch"
    names = {r.check for r in o.final.results}
    assert {"reach_targets", "carry_payload", "hold_still"} <= names  # the sims ran and passed
    assert any(e.startswith("  sizer: shoulder_pitch: ") for e in events)
    run_json = json.loads((result.out_dir / "run.json").read_text(encoding="utf-8"))
    assert run_json["design_mode"] == "hybrid"
    assert "Design mode: hybrid." in (result.out_dir / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("prompt", ["a robot that carries a 2 kg box across my office",
                                    "a robot that picks toys off the floor and drops them in a basket"])
def test_hybrid_mobile_types_end_to_end(tmp_path, prompt):
    result, _ = run(prompt, tmp_path, simulate=False)
    assert result.passed, [r.message for r in result.outcome.final.failures]
    spec = result.outcome.final.spec
    assert spec.base.kind == "wheeled_base" and spec.base.caster


def all_z_sized(intent, req, catalog, nearest=False):
    """A sizing stand-in that returns an arm that can only spin: no automatic fix applies."""
    from test_chain_diagnosis import ALL_Z

    return SizedDesign(RobotSpec.model_validate(ALL_Z), req, ["stand-in"], {})


def test_failures_autofix_cant_handle_go_back_to_the_llm(tmp_path, monkeypatch):
    calls = []

    def sizing(intent, req, catalog, nearest=False):
        calls.append(intent)
        return all_z_sized(intent, req, catalog) if len(calls) == 1 else size_design(intent, req, catalog)

    monkeypatch.setattr(hybrid, "size_design", sizing)
    result, events = run("Water the three potted plants on my windowsill", tmp_path, simulate=False)
    assert len(result.outcome.intents) == 2 and result.passed
    asked = [e for e in events if e.startswith("Autofix stopped (no automatic fix applies")]
    assert asked and "asking the model for a different intent" in asked[0]


def test_at_most_three_intent_revisions(tmp_path, monkeypatch):
    monkeypatch.setattr(hybrid, "size_design", all_z_sized)
    result, _ = run("Water the three potted plants on my windowsill", tmp_path, simulate=False)
    o = result.outcome
    assert not result.passed and len(o.intents) == 4
    assert o.stop_reason.startswith("automatic fixes and 3 intent revisions didn't fix it")


def test_revision_message_sends_failures_and_asks_for_no_numbers():
    intent = DesignIntent(robot_type="arm", arm_layout="yaw_pitch_pitch", gripper="parallel", rationale="r")
    from taskbot.checks import CheckResult

    msg = hybrid.revision_message(intent, [CheckResult(check="reach", passed=False, message="too short",
                                                       cause="unreachable")])
    assert '"arm_layout":"yaw_pitch_pitch"' in msg and "- [reach] (cause: unreachable) too short" in msg
    assert "don't give any numbers" in msg


# --------------------------------------------------------------------------- infeasible and nearest

def _sized_as_the_pipeline_would(r):
    """The pipeline sizes the normalised requirements (derived reach_m) after the feasibility gate."""
    from taskbot.feasibility import check_feasibility
    from taskbot.normalize import normalize_requirements

    n = Requirements(**normalize_requirements(json.loads(r.model_dump_json()), "x")[0])
    if not check_feasibility(n, CATALOG).ok:
        return None
    return size_design(DesignIntent(robot_type="arm", arm_layout="yaw_pitch_pitch_pitch", gripper=r.gripper_kind,
                                    rationale="r"), n, CATALOG)


# The first random case that passes the gate but no catalog motor can carry once sized.
INFEASIBLE = next(r for r in arm_cases(60, CATALOG)
                  if isinstance(s := _sized_as_the_pipeline_would(r), SizingFailure) and s.kind == "infeasible")


def _client(req):
    def intent(system, messages, schema):
        return {"robot_type": "arm", "arm_layout": "yaw_pitch_pitch_pitch", "gripper": req.gripper_kind,
                "rationale": "r"}
    return FakeClient({"Requirements": [json.loads(req.model_dump_json())], "DesignIntent": intent})


def test_sizer_infeasible_stops_with_exit_code_4(tmp_path, monkeypatch, capsys):
    from taskbot import cli

    monkeypatch.setattr(cli, "get_client", lambda *a: _client(INFEASIBLE))
    assert cli.main(["design", "x", "--backend", "fake", "--out", str(tmp_path), "--no-sim", "--no-record"]) == 4
    out = capsys.readouterr().out
    assert "needs" in out and "strongest catalog motor gives" in out


def test_nearest_builds_an_approximate_design_that_is_never_a_pass(tmp_path, monkeypatch, capsys):
    from taskbot import cli

    monkeypatch.setattr(cli, "get_client", lambda *a: _client(INFEASIBLE))
    code = cli.main(["design", "x", "--backend", "fake", "--out", str(tmp_path), "--no-sim", "--no-record",
                     "--nearest"])
    out = capsys.readouterr().out
    assert code == 2 and "APPROXIMATE: relaxed payload_kg" in out and "not a pass" in out
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    report = (folder / "report.md").read_text(encoding="utf-8")
    assert "**APPROXIMATE: no design can meet the requirements as written" in report


def test_bench_never_counts_approximate_as_a_pass(tmp_path):
    rec = Record("a", "easy_arm", "main", "p", "done", False, passed=False, approximate="payload_kg 0.4 -> 0.3 kg",
                 rounds=1, backend="fake", model="scripted", design_mode="hybrid")
    text = render_markdown([rec], RunMeta("fake", "scripted", "fake", "r"), "r")
    assert "APPROXIMATE (payload_kg 0.4 -> 0.3 kg), not a pass" in text
    assert "(hybrid mode)" in text


def test_compare_two_modes_on_the_same_model(tmp_path):
    rows = {mode: [Record("a", "easy_arm", "main", "p", "done", False, passed=(mode == "hybrid"), rounds=1,
                          backend="openai_compat", model="qwen", design_mode=mode)] for mode in ("llm", "hybrid")}
    for mode, recs in rows.items():
        write_csv(recs, tmp_path / f"{mode}.csv", mode)
    text = compare(tmp_path / "llm.csv", tmp_path / "hybrid.csv")
    assert "(llm mode)" in text and "(hybrid mode)" in text
    assert "## Better (fail → pass) (1)" in text
