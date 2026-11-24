"""Process setup that must run before MuJoCo is imported. Imports nothing heavy.

MUJOCO_GL picks MuJoCo's OpenGL backend. "egl" and "osmesa" exist only on Linux: on Windows or
macOS `import mujoco` fails with "invalid value for environment variable MUJOCO_GL". So it is only
set on Linux without a display, and never overrides a value the user set.
"""

from __future__ import annotations

import ctypes.util
import logging
import os
import sys
from collections.abc import MutableMapping


def configure_mujoco_gl(env: MutableMapping[str, str] | None = None, platform: str | None = None) -> str | None:
    """Set MUJOCO_GL for headless Linux (osmesa if installed, else egl). Returns the value set, or
    None when it was left alone (not Linux, a display is available, or already set)."""
    env = os.environ if env is None else env
    platform = sys.platform if platform is None else platform
    if not platform.startswith("linux") or env.get("DISPLAY") or env.get("WAYLAND_DISPLAY") or "MUJOCO_GL" in env:
        return None
    env["MUJOCO_GL"] = "osmesa" if ctypes.util.find_library("OSMesa") else "egl"
    return env["MUJOCO_GL"]


class _DropWindowsConnectionReset(logging.Filter):
    """asyncio on Windows logs "ConnectionResetError: [WinError 10054]" when a browser tab closes
    its connection; it is harmless. Everything else passes."""

    def filter(self, record: logging.LogRecord) -> bool:
        exc = record.exc_info[1] if record.exc_info else None
        return not (isinstance(exc, ConnectionResetError) and getattr(exc, "winerror", None) == 10054)


def quiet_windows_connection_reset(platform: str | None = None) -> bool:
    """Install the filter on the asyncio logger (Windows only). Returns True if installed."""
    if not (sys.platform if platform is None else platform).startswith("win"):
        return False
    logger = logging.getLogger("asyncio")
    if not any(isinstance(f, _DropWindowsConnectionReset) for f in logger.filters):
        logger.addFilter(_DropWindowsConnectionReset())
    return True
