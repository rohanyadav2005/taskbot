"""Task 10: the second real smoke test (qwen2.5:7b-instruct via Ollama on a laptop CPU, 2026-11-07,
tests/fixtures/real_runs/smoke_2026-11-07) replayed through today's code. A replay answers only
the steps the run recorded; where today's code asks something the run never did, code's
documented fallback stands in (said in the test)."""

import json
from pathlib import Path

import pytest

from taskbot.cli import main

pytestmark = pytest.mark.slow
SMOKE = Path(__file__).parent / "fixtures" / "real_runs" / "smoke_2026-11-07"
TOYS = "a robot that picks toys off the floor and drops them in a basket"


def replay(transcript: Path, prompt: str, tmp_path, *extra):
    code = main(["design", prompt, "--backend", "replay", "--transcript", str(transcript),
                 "--out", str(tmp_path), "--no-record", *extra])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    report = (out / "report.md").read_text(encoding="utf-8")
    return code, out, run, report


def test_1_windowsill_passes(tmp_path):
    code, _, run, _ = replay(SMOKE / "1_windowsill.jsonl", "water the three plants on my windowsill", tmp_path)
    assert code == 0 and run["status"] == "PASSED"


def test_2_m3_screws_pass(tmp_path):
    code, _, run, _ = replay(SMOKE / "2_m3_screws.jsonl",
                             "a desk arm that picks M3 screws out of a tray and drops them into a jar next to it",
                             tmp_path)
    assert code == 0 and run["status"] == "PASSED"


def test_3_box_passes(tmp_path):
    code, _, run, _ = replay(SMOKE / "3_box.jsonl", "a robot that carries a 2 kg box across my office", tmp_path)
    assert code == 0 and run["status"] == "PASSED"


def test_4_toys_passes_once_the_invented_ramp_and_base_load_are_removed(tmp_path):
    """qwen wrote a 15° ramp and 4 kg (x1.25) on the base; the task names no slope and drops the toys
    into a basket, so the normaliser makes them 5° and 0 kg (Task 11). qwen's own first intent then
    passes on the cheapest wheel motor, with every sizer sim inside 70% of its test's limit, so the
    outcome doesn't depend on the platform (on Windows the old sizer passed a different, borderline
    four-wheel design on code's fourth intent; here it failed turn_in_place)."""
    code, _, run, report = replay(SMOKE / "4_toys.jsonl", TOYS, tmp_path)
    assert code == 0 and run["status"] == "PASSED" and run["calls"] == 3 and run["fallbacks"] == []
    assert "max_incline_deg 15° reduced to 5°: the task names no ramp" in report
    assert "base_payload_kg 5 kg set to 0: the task drops the items into a separate container" in report
    assert "wheels: jgb37_520_110" in report and run["reason"] == ""


def test_4_toys_up_a_named_ramp_is_infeasible_and_says_why(tmp_path):
    """The same replies for a task that does name a ramp keep the 15°. No intent can be built, and
    each reason clears its limit by a wide margin (static torque 3.1-3.4 Nm against 2.83 usable;
    four-wheel turning 5.9° off against 5°), so the verdict is the same on every platform."""
    prompt = "a robot that picks toys off the floor, drives up the ramp to the playroom and drops them in a basket"
    code, _, run, report = replay(SMOKE / "4_toys.jsonl", prompt, tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert "**INFEASIBLE: none of the 4 intents TaskBot tried can be built from the catalog" in report
    assert "max_incline_deg 15° reduced" not in report and "Drove up a 15° slope" not in run["reason"]
    assert run["reason"].count("strongest catalog motor gives 2.83 Nm") == 3 and "turn_in_place" in run["reason"]
    sizing = sum(t["time_s"] for t in run["timings"] if t["kind"] == "sizing")
    assert sizing < 120


def test_5_high_shelf_is_infeasible(tmp_path):
    code, _, run, _ = replay(SMOKE / "5_grandmother_shelf.jsonl",
                             "something to help my grandmother reach things on a high shelf", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"


def test_6_drone_is_out_of_scope(tmp_path):
    code, _, run, _ = replay(SMOKE / "6_drone.jsonl", "a flying drone that delivers coffee to my room", tmp_path)
    assert code == 3 and run["status"] == "OUT_OF_SCOPE" and run["calls"] == 0
