"""Task 14: the two main-set prompts that passed in benchmark v2 (2026-11-22) and were INFEASIBLE in
v3 (2026-11-24). Replaying both runs' transcripts through today's code shows the code did not
regress: the v2 replies still pass, and the v3 replies (same prompt, same system prompt, sampled at
temperature 0.2) ask for requirements no catalog robot meets, and the verdict says why. They also
fail on Task 12's code (checked by hand: commit dd43d95)."""

import json
from pathlib import Path

import pytest

from taskbot.bench import load_suite
from taskbot.cli import main

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "tests" / "fixtures" / "real_runs" / "bench_2026-11-22"
V3 = ROOT / "tests" / "fixtures" / "real_runs" / "bench_2026-11-24"
PROMPTS = {p.id: p.prompt for p in load_suite(ROOT / "bench", "main")}


def replay(folder: Path, pid: str, tmp_path, *extra):
    code = main(["design", PROMPTS[pid], "--backend", "replay", "--transcript", str(folder / f"{pid}.jsonl"),
                 "--out", str(tmp_path), "--no-record", *extra])
    (out,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    return code, run, (out / "report.md").read_text(encoding="utf-8")


def first_reply(folder: Path, pid: str) -> dict:
    line = (folder / f"{pid}.jsonl").read_text(encoding="utf-8").splitlines()[0]
    return json.loads(line)


def test_the_two_runs_saw_the_same_prompt():
    for pid in ("easy_arm_phone_timelapse", "vague_shop"):
        v2, v3 = first_reply(V2, pid), first_reply(V3, pid)
        assert v2["system_sha256"] == v3["system_sha256"] and v2["user"] == v3["user"]
        assert v2["reply"] != v3["reply"]


def test_phone_v3_asks_for_a_wider_pan_and_is_infeasible(tmp_path):
    """v3: the workspace +-0.4 m across the desk (v2: +-0.3 m), so the phone (0.2 kg, 0.25 kg with the
    margin) is held 0.6 m out: 3.0-3.1 Nm at the shoulder against 2.83 Nm usable."""
    code, run, report = replay(V3, "easy_arm_phone_timelapse", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert run["reason"].startswith("In short: holding 0.25 kg at 0.6 m needs about 3.1 Nm at the shoulder")
    assert "the workspace (x -0.4 to 0.4, y -0.3 to 0.3, z 0.05 to 0.5 m) surrounds the robot" in report


def test_shop_v3_chooses_a_4_kg_box_and_says_what_taskbot_needs_to_know(tmp_path):
    code, run, _ = replay(V3, "vague_shop", tmp_path, "--no-sim")
    assert code == 4 and run["status"] == "INFEASIBLE"
    assert run["reason"].startswith("In short: holding 3.75 kg at 0.77 m needs about 29 Nm at the shoulder")
    assert "TaskBot needs to know which object, about how heavy it is" in run["reason"]


@pytest.mark.slow
@pytest.mark.parametrize("pid", ["easy_arm_phone_timelapse", "vague_shop"])
def test_the_v2_replies_still_pass_on_todays_code(pid, tmp_path):
    code, run, _ = replay(V2, pid, tmp_path)
    assert code == 0 and run["status"] == "PASSED", run["reason"]
