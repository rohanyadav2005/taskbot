"""The hybrid report says who chose every number, lists each sizing and autofix step, and what
the prompt asked for that TaskBot didn't use."""

from conftest import with_motor

from taskbot import hybrid
from taskbot.fake import fake_client
from taskbot.pipeline import run_pipeline
from taskbot.schema import RobotSpec, load_motor_catalog, supported_motors
from taskbot.sizing import SizedDesign, size_design

CATALOG = load_motor_catalog()


def report_for(prompt, tmp_path, design_mode="hybrid"):
    catalog = supported_motors(CATALOG)
    result = run_pipeline(fake_client(catalog), prompt, CATALOG, tmp_path, simulate=False, record=False,
                          design_mode=design_mode)
    return result, (result.out_dir / "report.md").read_text(encoding="utf-8")


def test_every_number_says_who_chose_it(tmp_path):
    result, report = report_for("Water the three potted plants on my windowsill", tmp_path)
    assert "## Who chose what" in report
    assert "| arm_layout | yaw_pitch_pitch_pitch | LLM (intent) |" in report
    spec = result.outcome.final.spec
    upper = next(s.link for s in spec.modules[1].segments if s.link.name == "upper_arm")
    assert f"| upper_arm.length_m | {upper.length_m:.4g} | sizer |" in report
    assert "| shoulder_pitch.motor_id | " in report and "base_yaw.lower" not in report  # continuous
    assert "- sizing: shoulder_pitch: " in report


def test_autofix_steps_and_changed_numbers_are_marked(tmp_path, monkeypatch):
    def weak_shoulder(intent, req, catalog, nearest=False):
        sized = size_design(intent, req, catalog)
        d = with_motor(sized.spec.model_dump(exclude_none=True), "shoulder_pitch", "mg996r")
        return SizedDesign(RobotSpec.model_validate(d), req, sized.steps, sized.provenance)

    monkeypatch.setattr(hybrid, "size_design", weak_shoulder)
    result, report = report_for("Water the three potted plants on my windowsill", tmp_path)
    assert result.passed
    assert "- autofix 1: shoulder_pitch: mg996r → " in report and "(needed " in report
    assert "| shoulder_pitch.motor_id | " in report and "| sizer, then autofix |" in report


def test_prompt_words_taskbot_does_not_model_are_listed(tmp_path):
    _, report = report_for("Water my plants with a red arm that has a camera", tmp_path)
    line = next(ln for ln in report.splitlines() if ln.startswith("- Not applied"))
    assert '"red" (colour)' in line and '"camera" (sensing)' in line
    assert "- Applied: robot type: arm; layout: yaw_pitch_pitch_pitch; gripper: parallel" in report


def test_llm_mode_report_has_no_provenance_section(tmp_path):
    _, report = report_for("Water the three potted plants on my windowsill", tmp_path, design_mode="llm")
    assert "## Who chose what" not in report and "Design mode: llm." in report

