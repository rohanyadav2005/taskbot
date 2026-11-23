"""Task 13: the second full benchmark (qwen2.5:7b-instruct via Ollama on a laptop CPU, 2026-11-22;
transcripts in tests/fixtures/real_runs/bench_2026-11-22): hold_still with no payload, internal
errors labelled as such, per-prompt time limits, and the scene rules the infeasible runs needed."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_bench import by_id, fake_factory

from taskbot import bench, scene, timing
from taskbot.bench import TIMED_OUT, Record, RunMeta, run_bench, sleep_warning
from taskbot.cli import main
from taskbot.feasibility import in_short
from taskbot.normalize import normalize_requirements
from taskbot.requirements import Requirements
from taskbot.schema import load_motor_catalog

RUNS = Path(__file__).parent / "fixtures" / "real_runs" / "bench_2026-11-22"
CATALOG = load_motor_catalog()
ARM = {"task_summary": "t", "robot_type": "arm", "gripper_needed": True, "gripper_kind": "parallel", "dof_min": 3,
       "position_tolerance_m": 0.005, "tolerance_reason": "r",
       "workspace": {"x_min": 0.15, "x_max": 0.35, "y_min": -0.15, "y_max": 0.15, "z_min": 0.0, "z_max": 0.2}}
MM = ARM | {"robot_type": "mobile_manipulator", "terrain": "flat", "speed_m_s": 0.3, "max_incline_deg": 5,
            "base_payload_bare_kg": 0.0}


# --------------------------------------------------------------------------- 1. hold_still, no payload


@pytest.mark.slow
def test_coffee_button_with_no_payload_holds_still_and_passes(tmp_path):
    """Benchmark: "internal error in TaskBot: SimMeasurementError: hold_still measured exactly zero
    deviation". Pressing a button holds nothing (0 kg): the controller's 20% payload belief error is
    20% of nothing and gravity feedforward is exact, so zero deviation is a real result."""
    code = main(["design", "Press the start button on my coffee machine every morning at 7", "--backend", "replay",
                 "--transcript", str(RUNS / "easy_arm_coffee_button.jsonl"), "--out", str(tmp_path), "--no-record"])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    assert code == 0 and run["status"] == "PASSED"
    assert "No payload to speak of (under 1 g), so this judges the arm alone" in report


def test_an_internal_error_is_never_an_invalid_spec(tmp_path):
    p = by_id("easy_arm_coffee_button")[0]
    rec = Record(p.id, p.category, "main", p.prompt, "done", False)
    req = Requirements(**normalize_requirements(ARM, p.prompt)[0])
    (tmp_path / "out").mkdir()
    result = SimpleNamespace(requirements=req, rejected=False, infeasible=[], out_dir=tmp_path / "out",
                             error="internal error in TaskBot: SimMeasurementError: boom\nTraceback ...", outcome=None)
    bench._score_result(rec, p, result, tmp_path)
    assert rec.internal_error and rec.valid_spec is None
    assert rec.unfixed == [{"check": "internal_error", "cause": "bug",
                            "message": "internal error in TaskBot: SimMeasurementError: boom"}]
    assert bench._outcome(rec) == "internal error (TaskBot bug)"
    text = bench.render_markdown([rec], RunMeta("ollama", "m", "ollama", "real"), "real")
    assert "- Internal errors (TaskBot bugs, not the model's spec): 1 (easy_arm_coffee_button)" in text
    assert "invalid spec" not in text.split("## Every prompt")[1]


# --------------------------------------------------------------------------- 2. time limits


def test_a_deadline_stops_the_next_step():
    token = timing.set_deadline(-1.0)  # already past
    timing.start()
    try:
        with pytest.raises(timing.PromptTimeout, match="passed its -1 s limit"):
            with timing.timed("llm", "requirements attempt 1"):
                pass
    finally:
        timing.clear_deadline(token)
    with timing.timed("llm", "no deadline any more"):
        pass


def test_a_prompt_past_its_limit_is_timed_out_and_rerun_on_resume(tmp_path):
    run_dir = tmp_path / "run"
    meta = RunMeta("fake", "scripted", "fake", "t", design_mode="hybrid")
    (rec,) = run_bench(by_id("mobile_office_box"), run_dir, fake_factory(), meta, concurrency=1,
                       prompt_timeout_s=1e-9)
    assert rec.status == TIMED_OUT and rec.error.startswith("timed out: the prompt passed its")
    assert bench._outcome(rec) == "timed out (not a design failure; rerun with the same --name)"
    assert not rec.scored  # not counted as a design failure
    (again,) = run_bench(by_id("mobile_office_box"), run_dir, fake_factory(), RunMeta(**vars(meta)),
                         concurrency=1, prompt_timeout_s=None)
    assert again.status == "done" and again.passed


def test_bench_cli_has_the_time_limits():
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), pytest.raises(SystemExit):
        bench.main(["--help"])
    assert "--prompt-timeout" in buf.getvalue() and "default 1800" in buf.getvalue()
    assert "--call-timeout" in buf.getvalue()


def test_sleep_warning():
    """easy_arm_lego_sort: 50,350 s on the wall clock for a run whose steps take a few minutes."""
    assert "the computer probably slept" in sleep_warning(50351.0, 140.0)
    assert sleep_warning(130.0, 120.0) is None   # a few seconds of unmeasured work
    assert sleep_warning(500.0, 400.0) is None   # under the 300 s gap


# --------------------------------------------------------------------------- 3. scene rules


@pytest.mark.parametrize("name, kg", [
    ("a plastic cup", 0.015),                         # cafe cups and "a light plastic cup" were 0.3 kg mugs
    ("a light plastic cup", 0.015),
    ("a plastic cup with 250 ml of water", 0.265),   # the water counts
    ("a plastic cookbook page", 0.005),              # the page, not the cookbook
    ("a coffee mug", 0.3),
])
def test_the_held_objects_head_noun_and_contents(name, kg):
    assert scene.object_named(name)[1] == pytest.approx(kg)


def test_heights_the_task_never_names_are_capped():
    """vague_grandma: a desk arm with targets up to 1.2 m; vague_workshop and vague_shop: a mobile
    manipulator up to 1.5 m; mm_cafe_cups: z 0.3-1.0 m for cups on cafe tables."""
    ws = dict(ARM["workspace"], z_min=0.05, z_max=1.2)
    d, notes = normalize_requirements(ARM | {"payload_bare_kg": 0.1, "workspace": ws}, "something to help my grandma")
    assert (d["workspace"]["z_min"], d["workspace"]["z_max"]) == (0.05, 0.5)
    assert any(n.startswith("workspace z 0.05 to 1.2 m became z 0.05 to 0.5 m: the task names no height") for n in notes)
    d, _ = normalize_requirements(ARM | {"payload_bare_kg": 0.4, "workspace": ws},
                                  "get jars of jam down from the top shelf")
    assert d["workspace"]["z_max"] == 1.2  # a height named: kept
    mm_ws = {"x_min": 0.2, "x_max": 0.45, "y_min": -0.2, "y_max": 0.2, "z_min": 0.0, "z_max": 1.5}
    d, _ = normalize_requirements(MM | {"payload_bare_kg": 0.1, "workspace": mm_ws},
                                  "I want a robot to help me in my workshop")
    assert d["workspace"]["z_max"] == 1.0
    cafe = dict(mm_ws, z_min=0.3, z_max=1.0)
    d, notes = normalize_requirements(MM | {"payload_bare_kg": 0.015, "workspace": cafe},
                                      "Clear empty cups from the tables in our cafe and bring them to the counter")
    assert (d["workspace"]["z_min"], d["workspace"]["z_max"]) == (0.65, 1.0)  # tables ~0.75 m, counter ~0.9 m
    d, _ = normalize_requirements(MM | {"payload_bare_kg": 0.06, "workspace": dict(mm_ws, z_max=0.3)},
                                  "pick up the toys from the floor and put them on the table")
    assert d["workspace"]["z_min"] == 0.0  # the floor is named: targets down there stay


def test_a_vague_task_says_what_taskbot_would_need_to_know():
    req = Requirements(**normalize_requirements(ARM | {"payload_bare_kg": 0.3}, "make my mornings easier")[0])
    problems = ["Intent 1 (yaw_pitch_pitch): shoulder_pitch needs 3.23 Nm to hold the sized arm out straight with "
                "0.375 kg (its own links and motors included), but the strongest catalog motor gives 2.83 Nm usable."]
    line = in_short(req, CATALOG, problems, [], "make my mornings easier")
    assert line.endswith("TaskBot needs to know which object, about how heavy it is, and how high and how far from "
                         "the robot it is (e.g. \"bring my 300 g mug from the kitchen counter\").")
    assert "needs to know" not in in_short(req, CATALOG, problems, [], "Lift my 1.5 kg pan onto the stove")
    for task in ("something to help my grandma", "a robot for my kitchen", "make my mornings easier"):
        assert scene.vague(task), task
    for task in ("keep my desk tidy", "water the three plants on my windowsill", "fetch the TV remote"):
        assert not scene.vague(task), task


# --------------------------------------------------------------------------- replays of the infeasible runs


PROMPTS = {p.id: p.prompt for p in bench.load_suite(Path(__file__).resolve().parent.parent / "bench", "main")}


def replay(pid: str, tmp_path, *extra):
    code = main(["design", PROMPTS[pid], "--backend", "replay", "--transcript", str(RUNS / f"{pid}.jsonl"),
                 "--out", str(tmp_path), "--no-record", *extra])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    return code, run, (out / "report.md").read_text(encoding="utf-8")


@pytest.mark.slow
@pytest.mark.parametrize("pid, why", [
    ("mm_tennis_balls", "a hard, level surface like a concrete floor"),  # was outdoor: 0.08 m wheels, slip
    ("mm_cafe_cups", "the gripper holds 1 x 'a plastic cup'"),        # was a 0.3 kg mug at z 0.3-1.0 m
    ("vague_kitchen", "the gripper holds 1 x 'a light plastic cup'"),  # was a 0.3 kg mug: 2.96 Nm
])
def test_wrongly_infeasible_runs_now_pass(pid, why, tmp_path):
    code, run, report = replay(pid, tmp_path)
    assert code == 0 and run["status"] == "PASSED", run["reason"]
    assert why in report


@pytest.mark.slow
def test_a_vague_task_beyond_the_catalog_says_what_to_tell_taskbot(tmp_path):
    """vague_mornings: qwen chose fetching a 0.3 kg coffee mug 0.5 m out (3.2 Nm against 2.83)."""
    code, run, _ = replay("vague_mornings", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert run["reason"].startswith("In short: holding 0.375 kg at 0.52 m needs about 3.2 Nm at the shoulder")
    assert "TaskBot needs to know which object, about how heavy it is, and how high" in run["reason"]
