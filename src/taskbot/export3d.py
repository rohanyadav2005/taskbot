"""Export the robot's MuJoCo geometry (at its home pose) as a GLB file for 3D viewers."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from taskbot.build_urdf import build_model
from taskbot.schema import Motor, RobotSpec

VISUAL_GROUP = 1  # URDF <visual> geoms; collision copies (group 0) are skipped when visuals exist


def export_glb(spec: RobotSpec, path: Path, catalog: dict[str, Motor] | None = None) -> Path:
    import trimesh

    model = build_model(spec, catalog)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    has_visual = bool(np.any(model.geom_group == VISUAL_GROUP))
    scene = trimesh.Scene()
    for g in range(model.ngeom):
        if has_visual and model.geom_group[g] != VISUAL_GROUP:
            continue
        kind, size = model.geom_type[g], model.geom_size[g]
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            mesh = trimesh.creation.box(extents=2 * size)
        elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            mesh = trimesh.creation.cylinder(radius=size[0], height=2 * size[1], sections=32)
        elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=size[0])
        elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            mesh = trimesh.creation.capsule(radius=size[0], height=2 * size[1])
        else:
            continue  # planes etc. are not part of the robot
        transform = np.eye(4)
        transform[:3, :3] = data.geom_xmat[g].reshape(3, 3)
        transform[:3, 3] = data.geom_xpos[g]
        mesh.apply_transform(transform)
        rgba = (np.clip(model.geom_rgba[g], 0, 1) * 255).astype(np.uint8)
        mesh.visual = trimesh.visual.ColorVisuals(mesh, vertex_colors=np.tile(rgba, (len(mesh.vertices), 1)))
        scene.add_geometry(mesh, node_name=f"geom_{g}")
    # MuJoCo is z-up; glTF is y-up.
    scene.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(scene.export(file_type="glb"))
    return path
