"""Task 11: the third real smoke test (qwen2.5:7b-instruct via Ollama on a Windows laptop CPU,
2026-11-12, tests/fixtures/real_runs/smoke_2026-11-12) replayed through today's code."""

import json
from pathlib import Path

import pytest

from taskbot.cli import main

pytestmark = pytest.mark.slow
SMOKE = Path(__file__).parent / "fixtures" / "real_runs" / "smoke_2026-11-12"
RUNS = [
    ("1_windowsill", "water the three plants on my windowsill", 0, "PASSED"),
    ("2_m3_screws", "a desk arm that picks M3 screws out of a tray and drops them into a jar next to it", 0, "PASSED"),
    ("3_box", "a robot that carries a 2 kg box across my office", 0, "PASSED"),
    ("4_toys", "a robot that picks toys off the floor and drops them in a basket", 0, "PASSED"),
    ("5_grandmother_shelf", "something to help my grandmother reach things on a high shelf", 4, "INFEASIBLE"),
    ("6_drone", "a flying drone that delivers coffee to my room", 3, "OUT_OF_SCOPE"),
]


@pytest.mark.parametrize("name, prompt, code, status", RUNS)
def test_smoke_run_replays_to_its_expected_outcome(tmp_path, name, prompt, code, status):
    got = main(["design", prompt, "--backend", "replay", "--transcript", str(SMOKE / f"{name}.jsonl"),
                "--out", str(tmp_path), "--no-record"] + (["--no-sim"] if status != "PASSED" else []))
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert got == code and run["status"] == status
    if name == "4_toys":
        # On the laptop this run passed only on qwen's fourth intent, with a borderline four-wheel
        # base (turn_in_place); without the invented 15° ramp and 4 kg load, its first intent passes.
        assert run["calls"] == 3 and run["reason"] == ""
