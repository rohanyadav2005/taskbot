"""The two run hooks the web app uses (taskbot.timing): a step listener for live progress
and a stop event for the Stop button. Nothing else in the pipeline changed."""

import threading

import pytest

from taskbot import timing


def test_listener_sees_each_step_start_and_end():
    seen = []
    token = timing.set_listener(lambda *a: seen.append(a))
    try:
        timing.start()
        with timing.timed("sizing", "intent 1"):
            with timing.timed("sim", "reach_targets"):
                pass
    finally:
        timing.reset_listener(token)
    assert [(e, k, s) for e, k, s, _ in seen] == [
        ("start", "sizing", "intent 1"), ("start", "sim", "reach_targets"),
        ("end", "sim", "reach_targets"), ("end", "sizing", "intent 1")]
    assert seen[0][3] is None and seen[-1][3] >= 0


def test_a_broken_listener_never_stops_the_run():
    def boom(*_):
        raise ValueError("listener bug")
    token = timing.set_listener(boom)
    try:
        with timing.timed("llm", "requirements attempt 1"):
            pass
    finally:
        timing.reset_listener(token)


def test_the_stop_event_stops_the_next_step():
    stop = threading.Event()
    token = timing.set_stop(stop)
    try:
        with timing.timed("sizing", "intent 1"):
            pass  # not set yet: runs
        stop.set()
        with pytest.raises(timing.PromptTimeout, match="stopped by the user"):
            with timing.timed("sim", "hold_still"):
                pass
    finally:
        timing.reset_stop(token)
    with timing.timed("sim", "after reset"):
        pass
