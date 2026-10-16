"""Render recorded qpos frames to an animated GIF with MuJoCo's offscreen renderer."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

WIDTH, HEIGHT, FPS = 320, 240, 20


def render_gif(model: mujoco.MjModel, frames: np.ndarray, path: Path, *,
               look_at=(0.0, 0.0, 0.15), distance: float = 1.0,
               width: int = WIDTH, height: int = HEIGHT, fps: int = FPS,
               track_body: int | None = None) -> str | None:
    """Write `path`. Returns None on success, or why rendering was skipped
    (usually: no OpenGL backend; set MUJOCO_GL=egl or osmesa when headless)."""
    try:
        from PIL import Image

        renderer = mujoco.Renderer(model, height, width)
    except Exception as exc:  # missing GL / Pillow: recording is optional
        return f"recording skipped ({type(exc).__name__}: {exc})"
    try:
        cam = mujoco.MjvCamera()
        cam.lookat[:] = look_at
        cam.distance, cam.azimuth, cam.elevation = distance, 135.0, -25.0
        if track_body is not None:  # follow a driving robot
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = track_body
        data = mujoco.MjData(model)
        images = []
        for qpos in frames:
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=cam)
            images.append(Image.fromarray(renderer.render()).quantize(colors=64))
    finally:
        renderer.close()
    path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(path, save_all=True, append_images=images[1:],
                   duration=round(1000 / fps), loop=0, optimize=True)
    return None
