"""Helpers for the Streamlit web app (streamlit_app.py at the repo root).

The app only calls existing functions: taskbot.pipeline.run_pipeline with a client from
taskbot.llm, taskbot.export3d.export_glb for the 3D view, and the files run_pipeline writes
(report.md, run.json, requirements.json, spec.json, robot.urdf, rounds/, sim/). Nothing here
sizes, checks or decides anything about a design. These modules import no Streamlit, so they are
tested directly.
"""
