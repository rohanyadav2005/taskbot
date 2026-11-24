"""What the web app shows for a finished run, read from the PipelineResult and the files
run_pipeline wrote (report.md, run.json, rounds/round_N/checks.json, sim/*.gif, spec.json). Nothing
here re-checks or re-sizes anything."""

from __future__ import annotations

import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from taskbot.schema import ArmChain, Gripper, Motor, RobotSpec

SIM_CHECKS = ("reach_targets", "carry_payload", "hold_still", "drive_straight", "turn_in_place", "incline",
              "drive_with_arm_extended")


@dataclass
class Headline:
    status: str      # PASSED | APPROXIMATE | INFEASIBLE | OUT OF SCOPE | FAILED | LLM UNUSABLE
    reason: str      # one plain line


def run_json(result) -> dict:
    path = Path(result.out_dir) / "run.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def headline(result) -> Headline:
    reason = run_json(result).get("reason", "") or ""
    if result.llm_unusable:
        return Headline("LLM UNUSABLE", result.llm_unusable.splitlines()[0])
    if result.outcome is None and result.out_of_scope:
        return Headline("OUT OF SCOPE", result.out_of_scope[0].removeprefix("OUT OF SCOPE:").strip())
    if result.infeasible:
        return Headline("INFEASIBLE", result.in_short or result.infeasible[0])
    if result.approximate:
        return Headline("APPROXIMATE", f"Not a pass: {result.approximate}")
    if result.passed:
        from taskbot.report import verdict

        return Headline("PASSED", verdict(result.outcome).removeprefix("PASSED: "))
    if result.error:
        return Headline("FAILED", result.error.splitlines()[0])
    return Headline("FAILED", reason.split(". Blocked by:")[0] or "no design passed every check")


def report_md(result) -> str:
    try:
        return (Path(result.out_dir) / "report.md").read_text(encoding="utf-8")
    except OSError:
        return ""


def section(markdown: str, title: str) -> str:
    """The body of one '## title' section of report.md ('' if absent)."""
    m = re.search(rf"^## {re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)", markdown, re.M | re.S)
    return m.group(1).strip() if m else ""


def final_spec(result) -> RobotSpec | None:
    return result.outcome.final.spec if result.outcome is not None else None


def parts(spec: RobotSpec, catalog: dict[str, Motor]) -> list[dict]:
    """Every motor in the design: where it is, which one, its price (the catalog's)."""
    rows = []
    for m in spec.modules:
        if isinstance(m, ArmChain):
            rows += [{"part": f"{m.name}: {s.joint.name}", "motor": s.joint.motor_id} for s in m.segments]
        elif isinstance(m, Gripper) and m.motor_id:
            rows.append({"part": f"{m.name} ({m.gripper_type} gripper)", "motor": m.motor_id})
    if spec.is_mobile:
        rows += [{"part": f"{spec.base.name}: driven wheel {i + 1}", "motor": spec.base.wheel_motor_id}
                 for i in range(spec.base.driven_wheels)]
    for r in rows:
        motor = catalog.get(r["motor"])
        r["name"] = motor.name if motor else r["motor"]
        r["price_usd"] = round(motor.price_usd, 2) if motor else None
    return rows


def round_checks(result) -> list[tuple[int, list[dict]]]:
    """(round number, its checks) for every round run_pipeline wrote."""
    rounds = Path(result.out_dir) / "rounds"
    out = []
    for d in sorted(rounds.glob("round_*"), key=lambda p: int(p.name.split("_")[1])) if rounds.is_dir() else []:
        try:
            out.append((int(d.name.split("_")[1]), json.loads((d / "checks.json").read_text(encoding="utf-8"))))
        except (OSError, ValueError):
            continue
    return out


def final_value(result, check: str) -> float | None:
    rounds = round_checks(result)
    if not rounds:
        return None
    return next((c["value"] for c in rounds[-1][1] if c["check"] == check and c["value"] is not None), None)


def sim_rows(result) -> list[dict]:
    rounds = round_checks(result)
    if not rounds:
        return []
    return [c for c in rounds[-1][1] if c["check"] in SIM_CHECKS]


def gifs(result) -> list[Path]:
    return sorted((Path(result.out_dir) / "sim").glob("*.gif"))


def mesh_figure(spec: RobotSpec, catalog: dict[str, Motor], path: Path):
    """An interactive plotly figure of the robot at its home pose, from export3d's GLB."""
    import numpy as np
    import plotly.graph_objects as go
    import trimesh

    from taskbot.export3d import export_glb

    export_glb(spec, path, catalog)
    scene = trimesh.load(path, force="scene")
    # export_glb turns MuJoCo's z-up into glTF's y-up; turn it back so z is up in the plot.
    scene.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, [1, 0, 0]))
    fig = go.Figure()
    for mesh in scene.dump():  # every part, already moved to its place in the world frame
        name = mesh.metadata.get("name", "") if hasattr(mesh, "metadata") else ""
        v, f = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        color = "#7a7f87"
        try:
            rgba = mesh.visual.face_colors[0]
            color = f"rgb({rgba[0]},{rgba[1]},{rgba[2]})"
        except (AttributeError, IndexError):
            pass
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color=color, flatshading=True, name=name or "part", showscale=False))
    fig.update_layout(scene=dict(aspectmode="data", xaxis_title="x (m)", yaxis_title="y (m)", zaxis_title="z (m)",
                                 camera=dict(eye=dict(x=2.3, y=2.3, z=1.2))),
                      margin=dict(l=0, r=0, t=0, b=0), height=600, showlegend=False)
    return fig


def zip_bytes(out_dir: Path) -> bytes:
    """Every file of the run, zipped, with forward-slash names."""
    buf = io.BytesIO()
    out_dir = Path(out_dir)
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out_dir.rglob("*")):
            if p.is_file():
                z.write(p, (Path(out_dir.name) / p.relative_to(out_dir)).as_posix())
    return buf.getvalue()
