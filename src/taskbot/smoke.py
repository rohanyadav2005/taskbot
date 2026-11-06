"""Smoke test: six everyday prompts through `taskbot design`, one line per result.

    python -m taskbot.smoke --backend ollama --model qwen2.5:7b-instruct
    scripts/smoke.sh ollama qwen2.5:7b-instruct        scripts/smoke.ps1 -Backend ollama -Model qwen2.5:7b-instruct

Each prompt runs in its own process with --no-record; its folder holds report.md, run.json,
llm_transcript.jsonl and log.txt, and the transcripts are also copied to <out>/transcripts/ so
a failing run can be replayed offline (--backend replay --transcript ...).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROMPTS = [
    "water the three plants on my windowsill",
    "a desk arm that picks M3 screws out of a tray and drops them into a jar next to it",
    "a robot that carries a 2 kg box across my office",
    "a robot that picks toys off the floor and drops them in a basket",
    "something to help my grandmother reach things on a high shelf",
    "a flying drone that delivers coffee to my room",
]
LABELS = {"PASSED": "PASSED", "FAILED": "FAILED", "APPROXIMATE": "APPROXIMATE", "DESIGN_FAILED": "FAILED",
          "INTERNAL_ERROR": "FAILED", "INFEASIBLE": "INFEASIBLE", "OUT_OF_SCOPE": "OUT OF SCOPE",
          "LLM_UNUSABLE": "LLM UNUSABLE"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="taskbot.smoke", description=__doc__.split("\n\n")[0])
    ap.add_argument("--backend", default="ollama")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--out", type=Path, default=None, help="default: smoke/<date-time>")
    ap.add_argument("--no-sim", action="store_true", help="static checks only (faster)")
    args = ap.parse_args(argv)
    out = args.out or Path("smoke") / dt.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    (out / "transcripts").mkdir(parents=True, exist_ok=True)
    print(f"Smoke test: {args.backend} / {args.model}, results in {out.as_posix()}")
    print("   status         total    [seconds by step: LLM calls, sizing, autofix, static checks, sims, other]")
    worst = 0
    for i, prompt in enumerate(PROMPTS, 1):
        folder = out / f"{i}"
        cmd = [sys.executable, "-m", "taskbot.cli", "design", prompt, "--backend", args.backend,
               "--model", args.model, "--out", str(folder), "--no-record"] + (["--no-sim"] if args.no_sim else [])
        start = time.monotonic()
        proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        elapsed = time.monotonic() - start
        runs = sorted(folder.glob("*/run.json")) if folder.exists() else []
        if runs:
            run = json.loads(runs[-1].read_text(encoding="utf-8"))
            status = LABELS.get(run.get("status", ""), run.get("status") or "UNKNOWN")
            calls, corrections = run.get("calls", 0), run.get("corrections", 0)
            t = run.get("time_by_kind") or {}
            split = (f"LLM {t.get('llm', 0):6.1f} | size {t.get('sizing', 0):5.1f} | fix {t.get('autofix', 0):5.1f} | "
                     f"checks {t.get('static_checks', 0):5.1f} | sims {t.get('sim', 0):6.1f} | other "
                     f"{t.get('report', 0) + t.get('other', 0):5.1f}")
            (runs[-1].parent / "log.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
            transcript = runs[-1].parent / "llm_transcript.jsonl"
            if transcript.exists():
                shutil.copy(transcript, out / "transcripts" / f"{i}.jsonl")
        else:  # no report: only a bad argument or configuration gets here
            status, calls, corrections, split = f"NO REPORT (exit {proc.returncode})", 0, 0, ""
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "log.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
        worst = max(worst, 0 if status in ("PASSED", "INFEASIBLE", "OUT OF SCOPE") else 1)
        print(f"{i}. {status:<13} {elapsed:6.1f} s  [{split}]  {calls:2d} LLM calls  {corrections:2d} corrections  {prompt}")
    return worst


if __name__ == "__main__":
    sys.exit(main())
