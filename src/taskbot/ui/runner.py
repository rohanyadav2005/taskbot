"""Run one design in a background thread for the web app, with live progress and a Stop button.

The thread calls taskbot.pipeline.run_pipeline exactly as the CLI does; progress comes from its
log lines (on_event) and the timing listener (taskbot.timing.set_listener), stopping from
taskbot.timing.set_stop, and the time limit from taskbot.timing.set_deadline. Errors become one
friendly sentence; a traceback goes to out/<run>/app_error.txt, never to the page.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from taskbot import timing
from taskbot.llm import LLMClient, LLMError

MAX_PROMPT_CHARS = 500
DEFAULT_TIMEOUT_S = 1800.0
STEP_NAMES = {"llm": "LLM", "sizing": "Sizing", "autofix": "Automatic fixes", "static_checks": "Checks",
              "sim": "Simulation", "report": "Report"}


@dataclass
class Step:
    kind: str
    step: str
    status: str = "running"      # running | done
    seconds: float | None = None

    @property
    def label(self) -> str:
        return f"{STEP_NAMES.get(self.kind, self.kind)}: {self.step}"


@dataclass
class Run:
    prompt: str
    mode: str                    # demo | local | cloud
    label: str                   # e.g. "Demo: Windowsill watering arm" or "Local: qwen2.5:7b-instruct"
    out_root: Path
    record: bool = False
    design_mode: str = "hybrid"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    started: float = field(default_factory=time.time)
    finished: float | None = None
    state: str = "running"       # running | done | stopped | timed_out | error
    message: str = ""            # a friendly sentence when state is not done
    steps: list[Step] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    result: object = None        # taskbot.pipeline.PipelineResult
    stop_event: threading.Event = field(default_factory=threading.Event, repr=False)
    thread: threading.Thread | None = field(default=None, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def running(self) -> bool:
        return self.state == "running"

    @property
    def elapsed(self) -> float:
        return (self.finished or time.time()) - self.started

    def stop(self) -> None:
        self.stop_event.set()

    def snapshot(self) -> tuple[list[Step], list[str]]:
        with self._lock:
            return [Step(s.kind, s.step, s.status, s.seconds) for s in self.steps], list(self.log)

    # called from the worker thread
    def _on_step(self, event: str, kind: str, step: str, seconds: float | None) -> None:
        with self._lock:
            if event == "start":
                self.steps.append(Step(kind, step))
            else:
                for s in reversed(self.steps):
                    if s.kind == kind and s.step == step and s.status == "running":
                        s.status, s.seconds = "done", round(seconds or 0.0, 2)
                        break

    def _on_event(self, line: str) -> None:
        with self._lock:
            self.log.append(line)


def check_prompt(prompt: str) -> str | None:
    """A friendly reason the prompt can't be run, or None."""
    text = (prompt or "").strip()
    if not text:
        return "Describe the job you want a robot to do."
    if len(text) > MAX_PROMPT_CHARS:
        return f"Please keep the task under {MAX_PROMPT_CHARS} characters (it has {len(text)})."
    return None


def friendly_error(exc: BaseException) -> str:
    if isinstance(exc, LLMError):
        text = str(exc).splitlines()[0]
        if "timed out" in text:
            return "The language model didn't answer in time. Try again, or use a faster model."
        if "401" in text or "403" in text or "auth" in text.lower():
            return "The API rejected the key. Check the key and the base URL."
        if "429" in text or "rate" in text.lower():
            return "The API is rate-limiting this key. Wait a minute and try again."
        return "The language model couldn't be used: " + _strip_secrets(text)
    return "TaskBot hit an internal error while designing this robot. The details are saved in the run folder."


def _strip_secrets(text: str) -> str:
    return " ".join(w for w in text.split() if not w.lower().startswith(("bearer", "sk-", "gsk_")))[:300]


def start_run(run: Run, make_client: Callable[[], LLMClient], catalog, *,
              timeout_s: float | None = DEFAULT_TIMEOUT_S, simulate: bool = True) -> Run:
    """Start `run` in a daemon thread and return it at once."""
    from taskbot.pipeline import run_pipeline

    def work() -> None:
        tokens = (timing.set_listener(run._on_step), timing.set_stop(run.stop_event), timing.set_deadline(timeout_s))
        try:
            client = make_client()
            run.result = run_pipeline(client, run.prompt.strip(), catalog, run.out_root, simulate=simulate,
                                      record=run.record, on_event=run._on_event, design_mode=run.design_mode)
            run.state = "done"
        except timing.PromptTimeout as exc:
            stopped = "stopped by the user" in str(exc)
            run.state = "stopped" if stopped else "timed_out"
            run.message = ("Stopped. Nothing was designed." if stopped else
                           f"The run passed its {timeout_s:g} s limit and was stopped.")
        except BaseException as exc:  # noqa: BLE001 - nothing may escape to the page
            run.state = "error"
            run.message = friendly_error(exc)
            if not isinstance(exc, LLMError):
                run.out_root.mkdir(parents=True, exist_ok=True)
                (run.out_root / "app_error.txt").write_text(
                    "".join(traceback.format_exception(exc)), encoding="utf-8")
        finally:
            timing.clear_deadline(tokens[2])
            timing.reset_stop(tokens[1])
            timing.reset_listener(tokens[0])
            run.finished = time.time()
            for s in run.steps:
                if s.status == "running":
                    s.status = "stopped"

    run.thread = threading.Thread(target=work, name=f"taskbot-run-{run.id}", daemon=True)
    run.thread.start()
    return run
