"""The whole run, shared by the CLI and the Gradio app:

    prompt -> requirements -> (stop if out of scope) -> design / check / repair -> outputs

Every run also writes run.json (provider, model, calls, tokens, JSON validation retries,
time) and adds the same to report.md. No API key is ever written.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from taskbot.llm import LLMClient, LLMOutputError
from taskbot.repair import DEFAULT_MAX_ROUNDS, Event, RepairOutcome, repair_loop
from taskbot.report import write_design_failed, write_out_of_scope, write_outputs
from taskbot.requirements import Requirements, extract_requirements
from taskbot.schema import Motor


@dataclass
class PipelineResult:
    requirements: Requirements
    outcome: RepairOutcome | None   # None: out of scope, or no valid design (see error)
    out_dir: Path
    usage: dict = field(default_factory=dict)
    error: str | None = None        # the model never produced a valid initial design

    @property
    def rejected(self) -> bool:
        """Stopped as out of scope (no design was attempted)."""
        return self.outcome is None and self.error is None

    @property
    def out_of_scope(self) -> list[str]:
        return self.requirements.out_of_scope

    @property
    def passed(self) -> bool:
        return self.outcome is not None and self.outcome.passed


def run_pipeline(client: LLMClient, prompt: str, catalog: dict[str, Motor], out_root: Path, *,
                 max_rounds: int = DEFAULT_MAX_ROUNDS, simulate: bool = True, record: bool = True,
                 on_event: Event | None = None) -> PipelineResult:
    emit = on_event or (lambda line: None)
    start = time.monotonic()
    emit("Extracting requirements...")
    req = extract_requirements(client, prompt)
    emit(f"Requirements: {req.robot_type} robot. " + " ".join(f"[{a}]" for a in req.assumptions))
    if req.out_of_scope:
        out = write_out_of_scope(req, out_root)
        emit("Stopped: the task is out of scope, so no design was made. " + " ".join(req.out_of_scope))
        return PipelineResult(req, None, out, record_usage(client, out, start))
    try:
        outcome = repair_loop(client, req, catalog, max_rounds=max_rounds, simulate=simulate,
                              on_event=on_event)
    except LLMOutputError as exc:  # no valid initial design after the JSON retries
        out = write_design_failed(req, out_root, str(exc))
        emit(f"Stopped: the model never produced a valid design. {exc}")
        return PipelineResult(req, None, out, record_usage(client, out, start), error=str(exc))
    if record and outcome.final.sim_runs:
        emit("Recording simulation GIFs...")
    out = write_outputs(outcome, out_root, catalog, record=record)
    usage = record_usage(client, out, start)
    emit(f"Done: {'PASSED' if outcome.passed else 'FAILED'}. Saved to {out}")
    emit(f"LLM: {usage['provider']} / {usage['model']}: {usage['calls']} calls, "
         f"{usage['total_tokens']} tokens, {usage['json_retries']} JSON retries")
    return PipelineResult(req, outcome, out, usage)


def record_usage(client: LLMClient, out: Path, start: float) -> dict:
    """Write run.json and append an 'LLM usage' section to report.md."""
    usage = client.usage.as_dict() | {"time_s": round(time.monotonic() - start, 2)}
    (out / "run.json").write_text(json.dumps(usage, indent=2) + "\n", "utf-8")
    report = out / "report.md"
    if report.exists():
        with report.open("a", encoding="utf-8") as f:
            f.write("\n## LLM usage\n\n"
                    f"Provider {usage['provider']}, model {usage['model']}: {usage['calls']} calls, "
                    f"{usage['input_tokens']} input + {usage['output_tokens']} output tokens, "
                    f"{usage['json_retries']} JSON validation retries, "
                    f"{usage['rate_limit_waits']} rate-limit waits, {usage['time_s']:g} s.\n")
    return usage
