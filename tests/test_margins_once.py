"""Each margin is applied exactly once: payload x1.25 and reach x1.10 in the requirements
normaliser, motor x1.25 in the sizer, 80% usable torque in the checks."""

from pathlib import Path

from taskbot import sizing
from taskbot.checks import run_checks, stretch_m
from taskbot.llm import ReplayClient
from taskbot.normalize import default_intent
from taskbot.requirements import extract_requirements, reach_corner
from taskbot.schema import load_motor_catalog
from taskbot.sizing import SizedDesign, size_design

CRASH = Path(__file__).parent / "fixtures" / "real_runs" / "windowsill_reach_crash.jsonl"


def test_windowsill_arm_is_no_longer_than_its_derived_reach():
    req = extract_requirements(ReplayClient(CRASH), "water the three plants on my windowsill")
    _, far, _ = reach_corner("arm", req.workspace)  # from the shoulder the sizer uses, 0.1625 m up
    assert req.reach_m == 0.39 and abs(req.reach_m - far * 1.10) <= 0.001
    sized = size_design(default_intent(req), req, load_motor_catalog())
    assert isinstance(sized, SizedDesign)
    assert stretch_m(sized.spec) <= far * 1.10 + 0.01
    assert [r.check for r in run_checks(sized.spec, req) if not r.passed] == []


def test_the_sizer_has_no_reach_margin_of_its_own():
    assert not hasattr(sizing, "REACH_MARGIN") and sizing.MOTOR_MARGIN == 1.25


def test_readme_says_where_each_margin_lives():
    text = (Path(__file__).parent.parent / "README.md").read_text(encoding="utf-8")
    assert "**Where each margin lives (each applied exactly once):**" in text
    for row in ("| Payload | ×1.25", "| Reach | ×1.10", "| Motor | usable torque ≥ 1.25", "| Usable torque | 80%"):
        assert row in text
