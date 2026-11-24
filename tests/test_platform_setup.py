"""MUJOCO_GL is only set on headless Linux (egl/osmesa don't exist on Windows or macOS, where
`import mujoco` fails with "invalid value for environment variable MUJOCO_GL"), the app starts
with a friendly message when an import fails, and Windows' harmless WinError 10054 log noise is
dropped."""

import logging
import sys
from pathlib import Path

import pytest

from taskbot.platform_setup import _DropWindowsConnectionReset, configure_mujoco_gl, quiet_windows_connection_reset

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")


@pytest.mark.parametrize("platform, env, expected", [
    ("win32", {}, None),
    ("darwin", {}, None),
    ("linux", {"DISPLAY": ":0"}, None),
    ("linux", {"WAYLAND_DISPLAY": "wayland-0"}, None),
    ("linux", {"MUJOCO_GL": "glfw"}, None),     # the user's own choice is kept
])
def test_mujoco_gl_is_left_alone(platform, env, expected):
    before = dict(env)
    assert configure_mujoco_gl(env, platform) is expected
    assert env == before


def test_headless_linux_gets_an_offscreen_backend():
    env = {}
    assert configure_mujoco_gl(env, "linux") in ("osmesa", "egl") and env["MUJOCO_GL"] in ("osmesa", "egl")


def test_the_app_starts_on_windows_without_setting_mujoco_gl(monkeypatch, tmp_path):
    """sys.platform == "win32" as platform_setup sees it (patching the real sys.platform would make
    the standard library import Windows-only modules on this machine)."""
    import os
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    import taskbot.platform_setup as ps

    monkeypatch.setattr(ps, "sys", SimpleNamespace(platform="win32"))
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setenv("TASKBOT_OUT", str(tmp_path))
    logger = logging.getLogger("asyncio")
    saved = list(logger.filters)
    try:
        at = AppTest.from_file(APP, default_timeout=60).run()
        assert not at.exception and not at.error
        assert at.title[0].value == "Design a robot"
        assert "MUJOCO_GL" not in os.environ
        assert any(isinstance(f, _DropWindowsConnectionReset) for f in logger.filters)
    finally:
        logger.filters[:] = saved


class _Raises:
    """Stands in for a module whose import fails the way `import mujoco` does with a bad MUJOCO_GL."""

    def __getattr__(self, name):
        if name.startswith("__"):  # pytest and repr look at dunders
            raise AttributeError(name)
        raise RuntimeError("invalid value for environment variable MUJOCO_GL: egl")


@pytest.mark.parametrize("module, starts", [
    (None, "TaskBot couldn't start: import of taskbot.ui.runner halted"),
    (_Raises(), "TaskBot couldn't start: invalid value for environment variable MUJOCO_GL: egl"),
])
def test_a_failed_import_shows_a_friendly_startup_message(monkeypatch, module, starts):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setitem(sys.modules, "taskbot.ui.runner", module)
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    message = at.error[0].value
    assert message.startswith(starts)
    assert "**Fix:**" in message and "Traceback" not in message
    if module is not None:
        assert "Remove the MUJOCO_GL environment variable" in message


def _record(exc) -> logging.LogRecord:
    return logging.LogRecord("asyncio", logging.ERROR, __file__, 1, "Exception in callback", None, (type(exc), exc, None))


def test_only_the_windows_connection_reset_noise_is_dropped():
    reset = ConnectionResetError(10054, "An existing connection was forcibly closed by the remote host")
    reset.winerror = 10054
    f = _DropWindowsConnectionReset()
    assert not f.filter(_record(reset))
    assert f.filter(_record(ConnectionResetError(104, "Connection reset by peer")))  # no winerror: kept
    assert f.filter(_record(ValueError("real problem")))
    assert f.filter(logging.LogRecord("asyncio", logging.ERROR, __file__, 1, "plain", None, None))


def test_the_filter_is_installed_on_windows_only():
    logger = logging.getLogger("asyncio")
    saved = list(logger.filters)
    try:
        assert quiet_windows_connection_reset("linux") is False
        assert quiet_windows_connection_reset("win32") is True
        assert quiet_windows_connection_reset("win32") is True  # idempotent
        assert sum(isinstance(f, _DropWindowsConnectionReset) for f in logger.filters) == 1
    finally:
        logger.filters[:] = saved
