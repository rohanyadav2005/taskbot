"""Gradio web app: describe a job, watch the design loop, inspect the robot.

    pip install -e ".[app]"
    taskbot-app            # http://127.0.0.1:7860

The fake backend works without any API key. Recording GIFs needs an OpenGL backend
(on a headless machine: MUJOCO_GL=egl or osmesa); without one the GIFs are skipped.
"""

from __future__ import annotations

import os
import queue
import shutil
import threading
import traceback
from pathlib import Path

from taskbot.cli import BACKENDS, get_client
from taskbot.export3d import export_glb
from taskbot.pipeline import PipelineResult, run_pipeline
from taskbot.report import controller_table_lines, requirements_rows, verdict
from taskbot.schema import load_motor_catalog

OUT_ROOT = Path(os.environ.get("TASKBOT_OUT", "out"))
EXAMPLES = [
    "Water the three potted plants on my windowsill",
    "a robot that carries a 2 kg box across my office",
    "a robot that picks toys off the floor and drops them in a basket",
    "a drone that flies my mail to the neighbour",
]


def _requirements_md(result: PipelineResult) -> str:
    req = result.requirements
    lines = [f"### Requirements ({req.robot_type})", "", *requirements_rows(req)]
    if result.outcome is not None:
        note = getattr(result.outcome.final.spec.base, "battery_note", None)
        if note:
            lines += [f"- Design assumption: {note}"]
    return "\n".join(lines)


def _summary_md(result: PipelineResult) -> str:
    if result.error is not None:
        return f"### FAILED: the model never produced a valid design\n\n```\n{result.error}\n```"
    if result.outcome is None:
        return ("### OUT OF SCOPE: no design was made\n\n"
                + "\n".join(f"- {r}" for r in result.out_of_scope))
    o = result.outcome
    rounds = " → ".join(f"R{r.number} {'pass' if r.passed else f'{len(r.failures)} fail'}" for r in o.rounds)
    return f"### {verdict(o)}\n\nRounds: {rounds}\n\nSpec: `{o.final.spec.name}` — {o.final.spec.description}"


def _collect(result: PipelineResult) -> dict:
    """Paths and markdown for the result panels."""
    out = result.out_dir
    files = [p for p in (out / "spec.json", out / "robot.urdf", out / "report.md") if p.exists()]
    glb = None
    if result.outcome is not None:
        glb = export_glb(result.outcome.final.spec, out / "robot.glb")
    archive = shutil.make_archive(str(out), "zip", root_dir=out)
    gifs = sorted(str(p) for p in (out / "sim").glob("*.gif")) if (out / "sim").exists() else []
    controller = ("\n".join(controller_table_lines(result.outcome.final.spec))
                  if result.outcome is not None else "No design, so no controller.")
    return {"files": [str(p) for p in files] + [archive], "glb": str(glb) if glb else None,
            "gifs": [(g, Path(g).stem) for g in gifs], "requirements": _requirements_md(result),
            "controller": controller, "summary": _summary_md(result)}


def design(prompt: str, backend: str, record: bool, max_rounds: int = 5):
    """Generator for the Design button: yields (log, summary, model3d, gifs, requirements,
    controller, files) as the run progresses."""
    log: list[str] = []
    empty = ("", None, [], "", "", [])
    if not prompt.strip():
        yield ("Enter what the robot must do.", *empty)
        return
    catalog = load_motor_catalog()
    try:
        client = get_client(backend, None, catalog)
    except Exception as exc:  # e.g. missing API key
        yield (f"Can't start the {backend} backend: {exc}", *empty)
        return
    events: queue.Queue = queue.Queue()
    box: dict = {}

    def work():
        try:
            box["result"] = run_pipeline(client, prompt, catalog, OUT_ROOT, max_rounds=max_rounds,
                                         record=record, on_event=events.put)
        except Exception as exc:  # shown in the log rather than crashing the UI
            box["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}"
        finally:
            events.put(None)

    threading.Thread(target=work, daemon=True).start()
    while True:
        line = events.get()
        if line is None:
            break
        log.append(line)
        yield ("\n".join(log), *empty)
    if "error" in box:
        log.append("ERROR: " + box["error"])
        yield ("\n".join(log), *empty)
        return
    c = _collect(box["result"])
    yield ("\n".join(log), c["summary"], c["glb"], c["gifs"], c["requirements"], c["controller"], c["files"])


def build_app():
    import gradio as gr

    with gr.Blocks(title="TaskBot") as app:
        gr.Markdown("# TaskBot\nDescribe what the robot must do. TaskBot extracts requirements, "
                    "designs a robot from modules, checks and simulates it, and repairs it until it "
                    "passes. The **fake** backend runs offline with canned answers.")
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(label="What must the robot do?", lines=2, value=EXAMPLES[2])
                gr.Examples(EXAMPLES, inputs=prompt)
            with gr.Column(scale=1):
                backend = gr.Radio(list(BACKENDS[::-1]), value="fake", label="Backend",
                                   info="openai_compat reads TASKBOT_BASE_URL, TASKBOT_MODEL and "
                                        "TASKBOT_API_KEY from the environment")
                record = gr.Checkbox(value=False, label="Record simulation GIFs (slow without a GPU)")
                go = gr.Button("Design", variant="primary")
        summary = gr.Markdown()
        log = gr.Textbox(label="Live log: each round's checks, with causes", lines=14, max_lines=30,
                         autoscroll=True)
        with gr.Row():
            model = gr.Model3D(label="Final robot (home pose)", clear_color=(0.95, 0.95, 0.95, 1.0),
                               height=420, elem_id="robot3d")
            gifs = gr.Gallery(label="Simulation recordings", columns=2, height=420)
        requirements = gr.Markdown()
        gr.Markdown("### Controller settings")
        controller = gr.Markdown()
        files = gr.File(label="Downloads: spec.json, robot.urdf, report.md, everything (zip)",
                        file_count="multiple")
        # The GLB goes through a State and is set on the Model3D once, after the run:
        # Gradio 6's Model3D drops its lazily loaded viewer when a generator re-sends the
        # same value on completion, leaving the panel blank.
        glb = gr.State(None)
        go.click(design, inputs=[prompt, backend, record],
                 outputs=[log, summary, glb, gifs, requirements, controller, files]
                 ).then(lambda path: path, inputs=glb, outputs=model)
    return app


def main() -> None:
    build_app().queue().launch(server_name=os.environ.get("TASKBOT_HOST", "127.0.0.1"))


if __name__ == "__main__":
    main()
