"""Task 10: the second real smoke test (qwen2.5:7b-instruct via Ollama on a laptop CPU, 2026-11-07,
tests/fixtures/real_runs/smoke_2026-11-07) replayed through today's code. A replay answers only
the steps the run recorded; where today's code asks something the run never did, code's
documented fallback stands in (said in the test)."""

import json
from pathlib import Path

from taskbot.cli import main

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


def test_4_toys_on_a_15_degree_ramp_is_infeasible_and_says_why(tmp_path):
    """qwen asked for a 15° ramp and 4 kg (x1.25) on the base. Two driven wheels only climb that with
    xm540_w270 on ~0.1 m wheels, which reach 0.28-0.30 m ahead of the chassis centre, so the floor
    targets start 0.33-0.35 m ahead and 0.3 kg out there needs 3.4-3.9 Nm at the shoulder (2.83 usable);
    four-wheel skid steering can't turn in place. The run stopped after the first infeasible intent;
    now every intent is tried (the replay's later ones are code's: the run recorded only two)."""
    code, _, run, report = replay(SMOKE / "4_toys.jsonl", TOYS, tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert "**INFEASIBLE: none of the 4 intents TaskBot tried can be built from the catalog" in report
    assert "0.11 m wheels driven by xm540_w270" in report and "so the targets start 0.327 m ahead instead of 0.2 m" in report
    assert "shoulder_pitch needs 3.43 Nm" in run["reason"] and "four_wheel" in run["reason"]
    assert "a gentler slope than 15°" in report
    sizing = sum(t["time_s"] for t in run["timings"] if t["kind"] == "sizing")
    assert sizing < 120  # four intents; the run spent 110 s on two before the sizer cache and early stops


def test_4_toys_on_a_10_degree_ramp_passes(tmp_path):
    """The fix the report names, checked: the same replies with the ramp at 10° size small wheels
    (pololu_25d_34_hp, 0.03 m) and pass every check and sim with qwen's own first intent."""
    lines = []
    for line in (SMOKE / "4_toys.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        if r["schema"] == "Requirements":
            r["reply"] = json.dumps(json.loads(r["reply"]) | {"max_incline_deg": 10})
        lines.append(json.dumps(r))
    gentler = tmp_path / "toys_10deg.jsonl"
    gentler.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_root = tmp_path / "out"
    out_root.mkdir()
    code, _, run, report = replay(gentler, TOYS, out_root)
    assert code == 0 and run["status"] == "PASSED" and run["fallbacks"] == []
    assert "wheels: pololu_25d_34_hp, radius 0.03 m" in report


def test_5_high_shelf_is_infeasible(tmp_path):
    code, _, run, _ = replay(SMOKE / "5_grandmother_shelf.jsonl",
                             "something to help my grandmother reach things on a high shelf", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"


def test_6_drone_is_out_of_scope(tmp_path):
    code, _, run, _ = replay(SMOKE / "6_drone.jsonl", "a flying drone that delivers coffee to my room", tmp_path)
    assert code == 3 and run["status"] == "OUT_OF_SCOPE" and run["calls"] == 0
