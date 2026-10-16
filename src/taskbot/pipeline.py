"""The whole run, shared by the CLI and the Gradio app:

    prompt -> requirements -> (stop if out of scope) -> design / check / repair -> outputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from taskbot.llm import LLMClient
from taskbot.repair import DEFAULT_MAX_ROUNDS, Event, RepairOutcome, repair_loop
from taskbot.report import write_out_of_scope, write_outputs
from taskbot.requirements import Requirements, extract_requirements
from taskbot.schema import Motor


@dataclass
class PipelineResult:
    requirements: Requirements
    outcome: RepairOutcome | None   # None when the task was out of scope
    out_dir: Path

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
    emit("Extracting requirements...")
    req = extract_requirements(client, prompt)
    emit(f"Requirements: {req.robot_type} robot. " + " ".join(f"[{a}]" for a in req.assumptions))
    if req.out_of_scope:
        out = write_out_of_scope(req, out_root)
        emit("Stopped: the task is out of scope, so no design was made. " + " ".join(req.out_of_scope))
        return PipelineResult(req, None, out)
    outcome = repair_loop(client, req, catalog, max_rounds=max_rounds, simulate=simulate,
                          on_event=on_event)
    if record and outcome.final.sim_runs:
        emit("Recording simulation GIFs...")
    out = write_outputs(outcome, out_root, catalog, record=record)
    emit(f"Done: {'PASSED' if outcome.passed else 'FAILED'}. Saved to {out}")
    return PipelineResult(req, outcome, out)
