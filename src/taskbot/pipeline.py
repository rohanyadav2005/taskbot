"""The whole run, shared by the CLI and the Gradio app:

    prompt -> requirements -> (stop if out of scope) -> feasibility gate -> design -> outputs

Design modes: "hybrid" (default; taskbot.hybrid: the LLM chooses a DesignIntent, code sizes
every number, autofix, then a revised intent only if needed) and "llm" (repair.repair_loop:
the LLM writes the whole spec and repairs it; kept for comparison).

The feasibility gate (taskbot.feasibility) sends infeasible requirements back to the
requirements step once with the numbers; if they're still infeasible, the run stops with
no design (PipelineResult.infeasible). The repair loop can also stop as infeasible.

Every run also writes run.json (provider, model, calls, tokens, JSON validation retries,
time) and adds the same to report.md. No API key is ever written.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import traceback

from taskbot.llm import LLMClient, LLMError, LLMOutputError, LLMRateLimitError
from taskbot.repair import DEFAULT_MAX_ROUNDS, Event, RepairOutcome, repair_loop

DESIGN_MODES = ("hybrid", "llm")
from taskbot.feasibility import check_feasibility, retry_message
from taskbot.report import write_design_failed, write_infeasible, write_out_of_scope, write_outputs
from taskbot.requirements import Requirements, extract_requirements, requirements_system_prompt
from taskbot.schema import Motor


@dataclass
class PipelineResult:
    requirements: Requirements
    outcome: RepairOutcome | None   # None: out of scope, or no valid design (see error)
    out_dir: Path
    usage: dict = field(default_factory=dict)
    error: str | None = None        # the model never produced a valid initial design
    infeasible: list[str] = field(default_factory=list)  # why no design can meet the requirements
    design_mode: str = "hybrid"
    llm_unusable: str | None = None  # the LLM couldn't be reached: no design was attempted
    llm_unusable_kind: str = ""      # "rate_limited" or "error"
    fallbacks: list[str] = field(default_factory=list)  # where code stood in for an unusable reply

    @property
    def approximate(self) -> str | None:
        """The requirement relaxed for a nearest-feasible design (hybrid --nearest), else None."""
        return getattr(self.outcome, "approximate", None)

    @property
    def rejected(self) -> bool:
        """Stopped as out of scope (no design was attempted)."""
        return self.outcome is None and self.error is None and not self.infeasible and self.llm_unusable is None

    @property
    def out_of_scope(self) -> list[str]:
        return self.requirements.out_of_scope

    @property
    def passed(self) -> bool:
        return self.outcome is not None and self.outcome.passed and self.approximate is None


def run_pipeline(client: LLMClient, prompt: str, catalog: dict[str, Motor], out_root: Path, *,
                 max_rounds: int = DEFAULT_MAX_ROUNDS, simulate: bool = True, record: bool = True,
                 on_event: Event | None = None, design_mode: str = "hybrid",
                 nearest: bool = False) -> PipelineResult:
    """`max_rounds` applies to the llm mode; `nearest` to the hybrid mode."""
    if design_mode not in DESIGN_MODES:
        raise ValueError(f"design_mode must be one of {DESIGN_MODES}")
    emit = on_event or (lambda line: None)
    start = time.monotonic()
    fallbacks: list[str] = []

    def usage_for(out, status, reason=""):
        return record_usage(client, out, start, design_mode, status, reason, fallbacks)

    def fall_back(step: str, what: str) -> None:
        fallbacks.append(f"{step}: {what}")
        client.corrections.append(("FALLBACK", f"{step}: {what}"))
        emit(f"Fallback ({step}): {what}")

    emit("Extracting requirements...")
    try:
        req = extract_requirements(client, prompt)
    except LLMOutputError as exc:
        from taskbot.fallback import requirements_fallback

        req, how, notes = requirements_fallback(prompt, exc.last_data)
        client.corrections += [("requirements_fallback", n) for n in notes]
        fall_back("requirements", f"no reply passed validation ({exc.last_error.splitlines()[0]}); used {how}")
    except LLMError as exc:
        if getattr(exc, "propagate", False):
            raise
        return _llm_unusable(client, prompt, out_root, usage_for, emit, design_mode, exc, "requirements")
    emit(f"Requirements: {req.robot_type} robot. " + " ".join(f"[{a}]" for a in req.assumptions))
    if req.out_of_scope:
        return _out_of_scope(client, req, out_root, usage_for, emit, design_mode, fallbacks)
    feasible = check_feasibility(req, catalog)
    if not feasible.ok:
        first = feasible
        emit("Feasibility: no design can meet these requirements. " + " ".join(feasible.problems))
        emit("Sending the requirements back once with these numbers...")
        from taskbot.normalize import normalize_requirements

        try:
            req = client.complete_json(requirements_system_prompt(), retry_message(prompt, req, feasible),
                                       Requirements, step="feasibility_retry",
                                       normalize=lambda d: normalize_requirements(d, prompt))
        except LLMError as exc:
            if getattr(exc, "propagate", False):
                raise
            fall_back("feasibility_retry", f"the model gave no usable second version ({str(exc).splitlines()[0]}); "
                      "kept the first requirements")
        emit(f"Requirements (second try): {req.robot_type} robot. " + " ".join(f"[{a}]" for a in req.assumptions))
        if req.out_of_scope:
            return _out_of_scope(client, req, out_root, usage_for, emit, design_mode, fallbacks)
        feasible = check_feasibility(req, catalog)
        if not feasible.ok:
            out = write_infeasible(req, out_root, feasible, first)
            emit("Stopped: the requirements are still infeasible, so no design was made. "
                 + " ".join(feasible.problems))
            return PipelineResult(req, None, out, usage_for(out, "INFEASIBLE", " ".join(feasible.problems)),
                                  infeasible=feasible.problems, design_mode=design_mode, fallbacks=fallbacks)
        emit("Feasibility: the second version can be built.")
    try:
        if design_mode == "hybrid":
            from taskbot.hybrid import hybrid_loop

            outcome = hybrid_loop(client, req, catalog, simulate=simulate, nearest=nearest, on_event=on_event,
                                  on_fallback=fall_back)
            outcome.prompt = prompt
        else:
            outcome = repair_loop(client, req, catalog, max_rounds=max_rounds, simulate=simulate,
                                  on_event=on_event)
    except LLMOutputError as exc:  # llm mode: no valid initial design after the JSON retries
        out = write_design_failed(req, out_root, str(exc))
        emit(f"Stopped: the model never produced a valid design. {exc}")
        return PipelineResult(req, None, out, usage_for(out, "DESIGN_FAILED", str(exc)), error=str(exc),
                              design_mode=design_mode, fallbacks=fallbacks)
    except LLMError as exc:  # llm mode: the backend stopped answering at the design step
        if getattr(exc, "propagate", False):
            raise
        return _llm_unusable(client, prompt, out_root, usage_for, emit, design_mode, exc, "design", req)
    except Exception as exc:  # a bug in TaskBot: report it rather than crash
        detail = f"internal error in TaskBot: {type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}"
        out = write_design_failed(req, out_root, detail)
        emit(f"Stopped: {detail.splitlines()[0]}")
        return PipelineResult(req, None, out, usage_for(out, "INTERNAL_ERROR", detail), error=detail,
                              design_mode=design_mode, fallbacks=fallbacks)
    if not outcome.rounds:  # hybrid: nothing could be sized, so nothing was checked
        reason = outcome.stop_reason or "no design could be sized"
        if outcome.infeasible:
            from taskbot.feasibility import Feasibility

            out = write_infeasible(req, out_root, Feasibility(list(outcome.infeasible),
                                                              list(getattr(outcome, "infeasible_options", []))))
            emit(f"Stopped: {' '.join(outcome.infeasible)}")
            return PipelineResult(req, None, out, usage_for(out, "INFEASIBLE", " ".join(outcome.infeasible)),
                                  infeasible=list(outcome.infeasible), design_mode=design_mode, fallbacks=fallbacks)
        out = write_design_failed(req, out_root, reason)
        emit(f"Stopped: {reason}")
        return PipelineResult(req, None, out, usage_for(out, "DESIGN_FAILED", reason), error=reason,
                              design_mode=design_mode, fallbacks=fallbacks)
    if record and outcome.final.sim_runs:
        emit("Recording simulation GIFs...")
    out = write_outputs(outcome, out_root, catalog, record=record)
    approx = getattr(outcome, "approximate", None)
    state = ("INFEASIBLE" if outcome.infeasible else "APPROXIMATE" if approx
             else ("PASSED" if outcome.passed else "FAILED"))
    usage = usage_for(out, state, outcome.stop_reason or "")
    emit(f"Done: {state}. Saved to {out}")
    emit(f"LLM: {usage['provider']} / {usage['model']}: {usage['calls']} calls, "
         f"{usage['total_tokens']} tokens, {usage['json_retries']} JSON retries")
    return PipelineResult(req, outcome, out, usage, infeasible=list(outcome.infeasible), design_mode=design_mode,
                          fallbacks=fallbacks)


def _out_of_scope(client, req, out_root, usage_for, emit, design_mode, fallbacks) -> PipelineResult:
    out = write_out_of_scope(req, out_root)
    emit("Stopped: the task is out of scope, so no design was made. " + " ".join(req.out_of_scope))
    return PipelineResult(req, None, out, usage_for(out, "OUT_OF_SCOPE", " ".join(req.out_of_scope)),
                          design_mode=design_mode, fallbacks=fallbacks)


def next_step(client: LLMClient) -> str:
    """What to try when the backend can't be reached."""
    provider = client.usage.provider
    base = getattr(client, "base_url", "") or getattr(client, "host", "")
    if provider == "ollama" or "11434" in str(base):
        return f"is Ollama running? try: ollama ps (and ollama pull {client.usage.model})"
    if provider.startswith("replay"):
        return "the transcript ran out of replies for this step; record a longer run"
    return "check the API key, the model name and the network, then run again"


def _llm_unusable(client, prompt, out_root, usage_for, emit, design_mode, exc, step, req=None) -> PipelineResult:
    """The backend couldn't be reached (or kept refusing): write a report saying so, no design."""
    from taskbot.fallback import rule_based
    from taskbot.normalize import normalize_requirements
    from taskbot.report import write_llm_unusable

    reason = f"the LLM backend failed at the {step} step: {str(exc).splitlines()[0]}"
    hint = next_step(client)
    if req is None:
        req = Requirements(**normalize_requirements(rule_based(prompt), prompt)[0])
    out = write_llm_unusable(req, out_root, reason, hint)
    emit(f"Stopped: {reason}. {hint}")
    return PipelineResult(req, None, out, usage_for(out, "LLM_UNUSABLE", f"{reason}. {hint}"), design_mode=design_mode,
                          llm_unusable=f"{reason}. {hint}",
                          llm_unusable_kind="rate_limited" if isinstance(exc, LLMRateLimitError) else "error")


def corrections_section(corrections: list[tuple[str, str]]) -> str:
    lines = ["", "## Corrections made by code", "",
             "What the LLM wrote that code fixed or filled in before validating it (taskbot.normalize). "
             "The checks and their limits are unchanged.", ""]
    lines += [f"- {step}: {note}" for step, note in corrections] or ["None: every reply was used as written."]
    return "\n".join(lines) + "\n"


def record_usage(client: LLMClient, out: Path, start: float, design_mode: str = "hybrid", status: str = "",
                 reason: str = "", fallbacks: list[str] | None = None) -> dict:
    """Write run.json (with the run's status and why) and the transcript, and append the
    corrections and LLM usage to report.md."""
    usage = client.usage.as_dict() | {"time_s": round(time.monotonic() - start, 2), "design_mode": design_mode,
                                      "corrections": len(client.corrections), "status": status,
                                      "reason": reason, "fallbacks": list(fallbacks or [])}
    (out / "run.json").write_text(json.dumps(usage, indent=2) + "\n", "utf-8")
    (out / "llm_transcript.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in client.transcript), "utf-8")
    report = out / "report.md"
    if report.exists():
        with report.open("a", encoding="utf-8") as f:
            f.write(corrections_section(client.corrections))
            f.write("\n## LLM usage\n\n"
                    f"Provider {usage['provider']}, model {usage['model']}: {usage['calls']} calls, "
                    f"{usage['input_tokens']} input + {usage['output_tokens']} output tokens, "
                    f"{usage['json_retries']} JSON validation retries, "
                    f"{usage['rate_limit_waits']} rate-limit waits, {usage['time_s']:g} s. "
                    f"Design mode: {design_mode}.\n")
    return usage
