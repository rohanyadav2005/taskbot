"""Task 12: the first full benchmark with a real model (qwen2.5:7b-instruct via Ollama on a laptop
CPU, 2026-11-20; transcripts in tests/fixtures/real_runs/bench_2026-11-20), replayed through
today's code; the one-line "In short" for an INFEASIBLE run; and the correct outcome rate."""

import json
from pathlib import Path

import pytest

from taskbot import bench
from taskbot.bench import Record, RunMeta, load_suite, outcome_correct
from taskbot.cli import main
from taskbot.feasibility import check_feasibility, in_short
from taskbot.normalize import normalize_requirements
from taskbot.requirements import Requirements
from taskbot.schema import load_motor_catalog

BENCH = Path(__file__).resolve().parent.parent / "bench"
RUNS = Path(__file__).parent / "fixtures" / "real_runs" / "bench_2026-11-20"
CATALOG = load_motor_catalog()
PROMPTS = {p.id: p.prompt for p in load_suite(BENCH, "main")}


def replay(pid: str, tmp_path, *extra):
    code = main(["design", PROMPTS[pid], "--backend", "replay", "--transcript", str(RUNS / f"{pid}.jsonl"),
                 "--out", str(tmp_path), "--no-record", *extra])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    return code, run, report


# --------------------------------------------------------------------------- In short


def test_in_short_says_what_it_needs_what_the_catalog_gives_and_what_would_work():
    req = Requirements(**normalize_requirements({
        "task_summary": "lift a pan", "robot_type": "arm", "payload_bare_kg": 1.5, "gripper_needed": True,
        "gripper_kind": "parallel", "dof_min": 3, "position_tolerance_m": 0.01, "tolerance_reason": "r",
        "workspace": {"x_min": 0.15, "x_max": 0.4, "y_min": -0.1, "y_max": 0.1, "z_min": 0.0, "z_max": 0.2}},
        "Lift my 1.5 kg cast iron pan from the counter onto the stove burner")[0])
    f = check_feasibility(req, CATALOG)
    line = in_short(req, CATALOG, f.problems, f.options)
    assert line.startswith("In short: holding 1.88 kg at ")
    assert "the strongest motor here (xm540_w270) gives 2.83 Nm. That needs an industrial arm, or a load under" in line
    assert "(margin included) at that distance" in line and "\n" not in line


def test_in_short_for_a_sized_arm_uses_its_own_shoulder_torque():
    req = Requirements(**normalize_requirements({
        "task_summary": "t", "robot_type": "arm", "payload_bare_kg": 0.24, "gripper_needed": True,
        "gripper_kind": "parallel", "dof_min": 3, "position_tolerance_m": 0.01, "tolerance_reason": "r",
        "workspace": {"x_min": 0.15, "x_max": 0.45, "y_min": -0.1, "y_max": 0.1, "z_min": 0.0, "z_max": 0.4}}, "x")[0])
    problems = ["Intent 1 (yaw_pitch_pitch): shoulder_pitch needs 4.03 Nm to hold the sized arm out straight with "
                "0.3 kg (its own links and motors included), but the strongest catalog motor gives 2.83 Nm usable."]
    line = in_short(req, CATALOG, problems, [])
    assert "needs about 4 Nm at the shoulder" in line and "gives 2.83 Nm" in line


def test_in_short_for_anything_else_is_the_first_problem_and_option():
    assert in_short(Requirements(**normalize_requirements({
        "task_summary": "t", "robot_type": "mobile", "terrain": "flat", "speed_m_s": 0.5, "max_incline_deg": 5,
        "base_payload_bare_kg": 1, "position_tolerance_m": 0.05, "tolerance_reason": "r"}, "x")[0]), CATALOG,
        ["No wheel motor drives it. Details."], ["A slower speed."]) == \
        "In short: No wheel motor drives it. What would work: A slower speed."


# --------------------------------------------------------------------------- correct outcome rate


def test_every_main_prompt_has_an_expected_outcome():
    main_prompts = load_suite(BENCH, "main")
    outcomes = {p.id: p.expect.expected_outcome for p in main_prompts}
    assert all(p.expect.outcome is not None for p in main_prompts)
    assert {k for k, v in outcomes.items() if v == "infeasible"} == {
        "hard_arm_grandma_shelf", "hard_arm_cast_iron_pan", "hard_arm_jug_pour", "mm_cafe_cups"}
    assert all((v == "out_of_scope") == p.expect.should_reject for p, v in zip(main_prompts, outcomes.values()))


def test_outcome_must_agree_with_should_reject():
    with pytest.raises(ValueError, match="outcome: out_of_scope goes with should_reject"):
        bench.Expectations(outcome="out_of_scope")
    with pytest.raises(ValueError, match="outcome: out_of_scope goes with should_reject"):
        bench.Expectations(should_reject=True, outcome="pass")


@pytest.mark.parametrize("expected, fields, ok", [
    ("pass", {"passed": True}, True),
    ("pass", {"infeasible": True}, False),
    ("infeasible", {"infeasible": True}, True),
    ("infeasible", {"passed": True}, False),  # a pass where none can exist: something is wrong
    ("infeasible", {"infeasible": True, "approximate": "x"}, False),
    ("out_of_scope", {"rejected": True}, True),
    ("pass", {"rejected": True}, False),
])
def test_outcome_correct(expected, fields, ok):
    r = Record("a", "hard_arm", "main", "p", "done", expected == "out_of_scope", expected_outcome=expected, **fields)
    assert outcome_correct(r) is ok


def test_report_shows_the_correct_outcome_rate_next_to_the_pass_rate():
    rows = [Record("p", "easy_arm", "main", "p", "done", False, passed=True, expected_outcome="pass",
                   outcome_correct=True),
            Record("i", "hard_arm", "main", "p", "done", False, infeasible=True, expected_outcome="infeasible",
                   outcome_correct=True),
            Record("w", "hard_arm", "main", "p", "done", False, infeasible=True, expected_outcome="pass",
                   outcome_correct=False),
            Record("o", "out_of_scope", "main", "p", "done", True, rejected=True, passed=True,
                   expected_outcome="out_of_scope", outcome_correct=True)]
    text = bench.render_markdown(rows, RunMeta("openai_compat", "m", "p", "real"), "real")
    assert "- **Overall pass rate (main prompts): 2/4 (50%)**" in text
    assert "- **Correct outcome rate (main prompts): 3/4 (75%)**" in text
    assert "| i | main | hard_arm | infeasible (stopped) | infeasible ✓ |" in text
    assert "| w | main | hard_arm | infeasible (stopped) | pass ✗ |" in text


# --------------------------------------------------------------------------- replays


def test_office_box_regression_passes_without_loosening_ground_clearance(tmp_path):
    """Benchmark: INFEASIBLE. qwen called an office "outdoor", so Task 11's ground_clearance asked for
    0.08 m wheels and no four_wheel base could turn in place. The task names only an indoor place:
    terrain is read as flat (the clearance minimums are unchanged)."""
    code, run, report = replay("mobile_office_box", tmp_path, "--no-sim")
    assert code == 0 and run["status"] == "PASSED"
    assert "terrain 'outdoor' read as 'flat': the task names an indoor place and nowhere outside" in report


def test_chess_is_an_arm_with_a_chess_piece_in_its_gripper(tmp_path):
    """Benchmark: INFEASIBLE at the gate after 2 calls (~160 s): a mobile manipulator holding 0.5 kg."""
    code, run, report = replay("easy_arm_chess", tmp_path, "--no-sim")
    assert code == 0 and run["status"] == "PASSED"
    assert "robot_type 'mobile_manipulator' read as 'arm'" in report
    assert "lowered to 0.03 kg: the task names 'pieces on a chess board'" in report


def test_cast_iron_pan_stays_infeasible_and_says_what_would_work(tmp_path):
    code, run, report = replay("hard_arm_cast_iron_pan", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert run["reason"].startswith("In short: holding 1.88 kg at 0.62 m needs about 12 Nm at the shoulder; "
                                    "the strongest motor here (xm540_w270) gives 2.83 Nm. That needs an industrial arm")
    assert "**In short: holding 1.88 kg" in report


@pytest.mark.slow
@pytest.mark.parametrize("pid", ["easy_arm_cat_feeder", "mm_shelf_restock"])
def test_scene_fixes_pass(pid, tmp_path):
    """Benchmark: INFEASIBLE. Cat feeder: heights from the floor for a bin on the counter. Shelf: the
    bottom shelf up to 1.2 m, 0.25 kg for a packet of crisps, 10 kg on the base."""
    code, run, _ = replay(pid, tmp_path)
    assert code == 0 and run["status"] == "PASSED", run["reason"]


@pytest.mark.slow
def test_laundry_basket_carries_its_stated_5_kg(tmp_path):
    """Benchmark: PASSED, but sized for 0 kg on the base (qwen left the load out and code's default
    was 0). The task carries a stated 5 kg: 6.25 kg with the margin, and it still passes."""
    code, run, report = replay("mobile_laundry_basket", tmp_path)
    assert code == 0 and run["status"] == "PASSED"
    assert "base_payload_bare_kg 0 kg set to 5 kg: the task carries" in report


@pytest.mark.slow
def test_fetching_the_remote_gets_an_arm(tmp_path):
    """Benchmark: PASSED as a "mobile" robot with no arm, for a task that picks up the remote."""
    code, run, report = replay("mm_remote_fetch", tmp_path)
    assert code == 0 and run["status"] == "PASSED"
    assert "robot_type 'mobile' read as 'mobile_manipulator'" in report


@pytest.mark.slow
@pytest.mark.parametrize("pid, words", [
    ("mm_tennis_balls", "no wheel motor, wheel size and chassis from this catalog passes the drive tests on outdoor"),
    ("mobile_garden_cart", "on outdoor ground needs about 3.2 Nm at each of 2 driven wheels"),
])
def test_outdoor_places_are_outdoor_and_say_why_they_are_infeasible(pid, words, tmp_path):
    """A court and a garden path are outdoor ground (Task 12 E): 0.08 m wheels for clearance,
    rolling resistance 0.05. No catalog base passes the drive tests there; the verdict says so in one line."""
    code, run, report = replay(pid, tmp_path)
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert run["reason"].startswith("In short:") and words in run["reason"]
    if pid == "mobile_garden_cart":  # qwen wrote "paved" (read as flat); for the court it wrote outdoor itself
        assert "terrain 'flat' read as 'outdoor': the task drives outdoors" in report


@pytest.mark.slow
def test_lego_sort_passes_with_a_stiffer_base_yaw(tmp_path):
    """Benchmark: INFEASIBLE (a mobile manipulator with 0.25 kg for a 3 g brick). With the requirements
    fixed, the cheapest servos missed 5 mm at two far corners through base_yaw compliance (FAILED);
    autofix now gives that joint the stiffer motors the sim names, as the sim's datasheet gains are."""
    code, run, report = replay("easy_arm_lego_sort", tmp_path)
    assert code == 0 and run["status"] == "PASSED"
    assert "base_yaw: mg996r → ax12a (stiffer position loop" in report
