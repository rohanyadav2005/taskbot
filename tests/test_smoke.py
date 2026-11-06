"""The smoke script: six prompts, one line each, transcripts saved (fake backend here)."""

import re
from pathlib import Path

from taskbot import smoke

ROOT = Path(__file__).parent.parent


def test_smoke_prints_one_line_per_prompt_and_saves_transcripts(tmp_path, capsys):
    code = smoke.main(["--backend", "fake", "--model", "scripted", "--no-sim", "--out", str(tmp_path)])
    lines = [ln for ln in capsys.readouterr().out.splitlines() if re.match(r"\d\. ", ln)]
    assert code == 0 and len(lines) == 6
    for line, prompt in zip(lines, smoke.PROMPTS):
        assert re.match(r"\d\. (PASSED|FAILED|APPROXIMATE|INFEASIBLE|OUT OF SCOPE) +[\d.]+ s  \[LLM +[\d.]+ \| size +[\d.]+ "
                        r"\| fix +[\d.]+ \| checks +[\d.]+ \| sims +[\d.]+ \| other +[\d.]+\] +\d+ LLM calls +\d+ corrections  ",
                        line), line
        assert line.endswith(prompt)
    assert lines[-1].startswith("6. OUT OF SCOPE")
    assert len(list((tmp_path / "transcripts").glob("*.jsonl"))) == 6
    assert all((p / "log.txt").exists() for p in (tmp_path / "1").iterdir() if p.is_dir())


def test_wrapper_scripts_call_the_module():
    for name in ("smoke.sh", "smoke.ps1"):
        text = (ROOT / "scripts" / name).read_text(encoding="utf-8")
        assert "taskbot.smoke" in text and "qwen2.5:7b-instruct" in text
