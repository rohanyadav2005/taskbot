"""Time per step of a run, for run.json ("timings", "time_by_kind"), report.md and the smoke summary.

    timeline = start()                       # once per run (taskbot.pipeline)
    with timed("sizing", "intent 1"): ...    # anywhere in the run

Kinds: llm (each call, as the backend took it), sizing, autofix, static_checks, sim (each sim
test), report (writing outputs and GIFs). Steps nest (autofix re-runs the checks and sims): each
entry has its total time_s and its self_s without the steps inside it, and time_by_kind sums
self_s, so nothing is counted twice; "other" is the rest of the run (feasibility gate,
normalising, file writes). Outside a run (start() not called) timed() does nothing.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

KINDS = ("llm", "sizing", "autofix", "static_checks", "sim", "report")
_LOG: ContextVar[list | None] = ContextVar("taskbot_timeline", default=None)
_STACK: ContextVar[list | None] = ContextVar("taskbot_timeline_stack", default=None)


def start() -> list[dict]:
    log: list[dict] = []
    _LOG.set(log)
    _STACK.set([])
    return log


@contextmanager
def timed(kind: str, step: str) -> Iterator[dict | None]:
    log, stack = _LOG.get(), _STACK.get()
    if log is None or stack is None:
        yield None
        return
    entry = {"kind": kind, "step": step, "time_s": 0.0, "self_s": 0.0}
    frame = [entry, 0.0]  # [entry, time spent in steps inside it]
    stack.append(frame)
    t0 = time.perf_counter()
    try:
        yield entry
    finally:
        dt = time.perf_counter() - t0
        stack.remove(frame)
        entry["time_s"], entry["self_s"] = round(dt, 3), round(max(dt - frame[1], 0.0), 3)
        if stack:
            stack[-1][1] += dt
        log.append(entry)


def by_kind(log: list[dict], total_s: float) -> dict[str, float]:
    """Seconds per kind (self time, so nested steps aren't counted twice), plus "other"."""
    out = {k: round(sum(e["self_s"] for e in log if e["kind"] == k), 2) for k in KINDS}
    out["other"] = round(max(total_s - sum(out.values()), 0.0), 2)
    return out


def summary(times: dict[str, float]) -> str:
    """One line, e.g. "LLM 1008 s, sizing 0.31 s, ... other 0.1 s"."""
    names = {"llm": "LLM", "static_checks": "static checks", "sim": "sims"}
    return ", ".join(f"{names.get(k, k)} {v:.3g} s" for k, v in times.items())
