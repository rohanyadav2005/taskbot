"""Time per step of a run, for run.json ("timings", "time_by_kind"), report.md and the smoke summary.

    timeline = start()                       # once per run (taskbot.pipeline)
    with timed("sizing", "intent 1"): ...    # anywhere in the run

Kinds: llm, sizing, autofix, static_checks, sim, report. Steps nest, so each entry has time_s and
self_s (without inner steps) and time_by_kind sums self_s; "other" is the rest. Outside a run
timed() does nothing.

Every timed step first checks two things, so the pipeline needs no other hooks:
- a wall-clock deadline (set_deadline; the benchmark's --prompt-timeout), which also catches a
  laptop that slept;
- a stop event (set_stop; the web app's Stop button).
Either raises PromptTimeout, a BaseException so `except Exception` in the pipeline never turns it
into an internal error. set_listener(fn) reports each step's start and end to a live UI. All of
these are per run (ContextVars).
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

KINDS = ("llm", "sizing", "autofix", "static_checks", "sim", "report")
_LOG: ContextVar[list | None] = ContextVar("taskbot_timeline", default=None)
_STACK: ContextVar[list | None] = ContextVar("taskbot_timeline_stack", default=None)
_DEADLINE: ContextVar[tuple[float, float] | None] = ContextVar("taskbot_deadline", default=None)
_LISTENER: ContextVar = ContextVar("taskbot_step_listener", default=None)
_STOP: ContextVar = ContextVar("taskbot_stop", default=None)


class PromptTimeout(BaseException):
    """The run passed its deadline (wall clock)."""


def set_deadline(limit_s: float | None):
    """Deadline limit_s seconds of wall-clock time from now (None: none). Returns a token for
    clear_deadline."""
    return _DEADLINE.set(None if limit_s is None else (time.time() + limit_s, limit_s))


def clear_deadline(token) -> None:
    _DEADLINE.reset(token)


def set_listener(fn):
    """fn(event, kind, step, seconds) for every timed step: event "start" (seconds None) or "end".
    Returns a token for reset_listener."""
    return _LISTENER.set(fn)


def reset_listener(token) -> None:
    _LISTENER.reset(token)


def set_stop(event):
    """A threading.Event: once set, the next timed step raises PromptTimeout. Returns a token."""
    return _STOP.set(event)


def reset_stop(token) -> None:
    _STOP.reset(token)


def _notify(*args) -> None:
    fn = _LISTENER.get()
    if fn is not None:
        try:
            fn(*args)
        except Exception:  # a broken listener must never stop a design run
            pass


def check_deadline(step: str = "") -> None:
    stop = _STOP.get()
    if stop is not None and stop.is_set():
        raise PromptTimeout("stopped by the user" + (f" (at {step})" if step else ""))
    d = _DEADLINE.get()
    if d is not None and time.time() > d[0]:
        raise PromptTimeout(f"the prompt passed its {d[1]:g} s limit" + (f" (at {step})" if step else ""))


def start() -> list[dict]:
    log: list[dict] = []
    _LOG.set(log)
    _STACK.set([])
    return log


@contextmanager
def timed(kind: str, step: str) -> Iterator[dict | None]:
    check_deadline(f"{kind}: {step}")
    _notify("start", kind, step, None)
    log, stack = _LOG.get(), _STACK.get()
    if log is None or stack is None:
        t0 = time.perf_counter()
        try:
            yield None
        finally:
            _notify("end", kind, step, time.perf_counter() - t0)
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
        _notify("end", kind, step, dt)


def by_kind(log: list[dict], total_s: float) -> dict[str, float]:
    """Seconds per kind (self time, so nested steps aren't counted twice), plus "other"."""
    out = {k: round(sum(e["self_s"] for e in log if e["kind"] == k), 2) for k in KINDS}
    out["other"] = round(max(total_s - sum(out.values()), 0.0), 2)
    return out


def summary(times: dict[str, float]) -> str:
    """One line, e.g. "LLM 1008 s, sizing 0.31 s, ... other 0.1 s"."""
    names = {"llm": "LLM", "static_checks": "static checks", "sim": "sims"}
    return ", ".join(f"{names.get(k, k)} {v:.3g} s" for k, v in times.items())
