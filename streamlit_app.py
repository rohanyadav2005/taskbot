"""TaskBot web app (Streamlit).

    pip install -e ".[ui]"
    streamlit run streamlit_app.py

Describe a job in plain English; TaskBot designs a robot for it (the LLM chooses the layout, code
sizes every number, MuJoCo verifies). Three modes: Demo replays real recorded runs (no LLM needed),
Local uses Ollama on this computer, Cloud API uses any OpenAI-compatible provider. The app only
calls taskbot's existing functions (see taskbot/ui); runs are written under out/ui/ and nowhere else.
"""

from __future__ import annotations

import ctypes.util
import os
import re
import time
import uuid
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa" if ctypes.util.find_library("OSMesa") else "egl")

import streamlit as st  # noqa: E402

from taskbot.ui import backends, gallery, results  # noqa: E402
from taskbot.ui.runner import MAX_PROMPT_CHARS, Run, check_prompt, start_run  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_ROOT = Path(os.environ.get("TASKBOT_OUT", "out")) / "ui"
RUN_TIMEOUT_S = 900.0
POLL_S = 0.5
MODES = ["Demo", "Local (Ollama)", "Cloud API"]
STATUS_STYLE = {"PASSED": st.success, "APPROXIMATE": st.warning, "INFEASIBLE": st.warning,
                "OUT OF SCOPE": st.info, "FAILED": st.error, "LLM UNUSABLE": st.error}


@st.cache_resource
def catalog():
    from taskbot.schema import load_motor_catalog

    return load_motor_catalog()


def secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "") or "")
    except Exception:  # no secrets file: that's fine
        return ""


def state():
    ss = st.session_state
    ss.setdefault("sid", uuid.uuid4().hex[:8])
    ss.setdefault("runs", [])
    ss.setdefault("current", None)
    ss.setdefault("page", "Design")
    ss.setdefault("prompt", gallery.EXAMPLE_PROMPTS[0])
    return ss


def current_run() -> Run | None:
    ss = st.session_state
    return next((r for r in ss.runs if r.id == ss.current), None)


def active_run() -> Run | None:
    return next((r for r in st.session_state.runs if r.running), None)


# --------------------------------------------------------------------------- sidebar


def sidebar() -> dict:
    ss = st.session_state
    st.sidebar.title("TaskBot")
    st.sidebar.radio("Page", ["Design", "How it works", "History"], key="page")
    cfg: dict = {}
    if ss.page != "Design":
        return cfg
    mode = st.sidebar.radio("Mode", MODES, key="mode",
                            help="Demo replays real recorded runs with no LLM; Local needs Ollama on this "
                                 "computer; Cloud API needs a provider's base URL, model and key.")
    cfg["mode"] = mode
    if mode == "Demo":
        demos = [d for d in gallery.DEMOS if d.available]
        key = st.sidebar.selectbox("Recorded run", [d.key for d in demos], key="demo",
                                   format_func=lambda k: gallery.by_key(k).title)
        cfg["demo"] = gallery.by_key(key)
        st.sidebar.caption("A real run of qwen2.5:7b-instruct, replayed from its recorded transcript: the "
                           "design, checks and simulations run again now; only the model's answers are replayed.")
    elif mode == "Local (Ollama)":
        cfg["url"] = st.sidebar.text_input("Ollama URL", backends.DEFAULT_OLLAMA_URL, key="ollama_url")
        cfg["model"] = st.sidebar.text_input("Model", backends.DEFAULT_OLLAMA_MODEL, key="ollama_model")
        if st.sidebar.button("Check Ollama", key="check_ollama"):
            show_check(backends.check_ollama(cfg["url"], cfg["model"]), st.sidebar)
        st.sidebar.caption("Only works when this app runs on the same computer as Ollama.")
    else:
        cfg["url"] = st.sidebar.text_input("Base URL", secret("TASKBOT_BASE_URL"), key="cloud_url",
                                           placeholder="https://api.groq.com/openai/v1")
        cfg["model"] = st.sidebar.text_input("Model", secret("TASKBOT_MODEL"), key="cloud_model",
                                             placeholder="llama-3.3-70b-versatile")
        typed = st.sidebar.text_input("API key", type="password", key="cloud_key",
                                      help="Sent only to the provider; never shown, logged or saved.")
        cfg["key"] = typed or secret("TASKBOT_API_KEY")
        if not typed and cfg["key"]:
            st.sidebar.caption("Using the API key from the app's secrets.")
    with st.sidebar.expander("Advanced"):
        cfg["design_mode"] = st.selectbox(
            "Design mode", ["hybrid", "llm"], key="design_mode", disabled=mode == "Demo",
            help="hybrid: the LLM chooses the layout, code sizes every number (the default). llm: the model "
                 "writes the whole design itself, kept for comparison (0/10 in the benchmark, ~21 min each).")
        if mode == "Demo":
            cfg["design_mode"] = "hybrid"  # the recorded runs are hybrid runs
    return cfg


def show_check(check: backends.Check, where=st) -> None:
    if check.ok:
        where.success(check.message)
    else:
        where.error(check.message + (f"\n\n**Fix:** {check.fix}" if check.fix else ""))


# --------------------------------------------------------------------------- design page


def design_page(cfg: dict) -> None:
    st.title("Design a robot")
    st.caption("Describe a job in plain English. The LLM chooses the layout, code sizes every number, "
               "MuJoCo verifies.")
    busy = active_run()
    if cfg["mode"] == "Demo":
        demo = cfg["demo"]
        prompt = demo.prompt
        st.text_area("Task", prompt, disabled=True, key=f"demo_prompt_{demo.key}")
        st.caption(f"Recorded outcome: **{demo.expected}**")
    else:
        cols = st.columns(3)
        for i, example in enumerate(gallery.EXAMPLE_PROMPTS):
            cols[i % 3].button(example, key=f"example_{i}", on_click=lambda e=example: st.session_state.update(prompt=e),
                               width="stretch")
        prompt = st.text_area("Task", key="prompt", max_chars=MAX_PROMPT_CHARS, height=90)
    record = st.toggle("Record GIFs", value=False, key="record",
                       help="Renders each simulation as a GIF. Slow on a CPU (minutes per design).")
    if record:
        st.warning("Recording GIFs is slow on a CPU: expect several extra minutes.")
    if st.button("Design", type="primary", key="design", disabled=busy is not None):
        launch(cfg, prompt, record)
        busy = active_run()
    run = current_run()
    if run is not None and run.mode != cfg["mode"] and not run.running:
        run = None  # another mode's run: it stays in History
    if run is None:
        if cfg["mode"] == "Demo" and cfg["demo"].gif and cfg["demo"].gif.is_file():
            st.image(str(cfg["demo"].gif), caption=f"{cfg['demo'].title}: a recording of this design's simulation")
        return
    if run.running:
        progress(run)
        time.sleep(POLL_S)
        st.rerun()
    show_result(run)


def launch(cfg: dict, prompt: str, record: bool) -> None:
    ss = st.session_state
    if active_run() is not None:
        st.warning("A design is already running in this session; wait for it or press Stop.")
        return
    problem = check_prompt(prompt)
    if problem:
        st.warning(problem)
        return
    mode = cfg["mode"]
    if mode == "Demo":
        demo = cfg["demo"]
        make = lambda: backends.replay_client(demo.transcript)  # noqa: E731
        label = f"Demo: {demo.title}"
    elif mode == "Local (Ollama)":
        check = backends.check_ollama(cfg["url"], cfg["model"])
        if not check.ok:
            show_check(check)
            return
        url, model = cfg["url"], cfg["model"]
        make = lambda: backends.ollama_client(url, model)  # noqa: E731
        label = f"Local: {model}"
    else:
        check = backends.cloud_settings_check(cfg["url"], cfg["model"], cfg["key"])
        if not check.ok:
            show_check(check)
            return
        url, model, key = cfg["url"], cfg["model"], cfg["key"]
        make = lambda: backends.cloud_client(url, model, key)  # noqa: E731
        label = f"Cloud: {model}"
    run = Run(prompt=prompt.strip(), mode=mode, label=label, record=record, design_mode=cfg["design_mode"],
              out_root=OUT_ROOT / ss.sid)
    run.out_root = OUT_ROOT / ss.sid / run.id
    start_run(run, make, catalog(), timeout_s=RUN_TIMEOUT_S)
    ss.runs.append(run)
    ss.current = run.id


STEP_ICON = {"running": "⏳", "done": "✅", "stopped": "⏹️"}


def progress(run: Run) -> None:
    steps, log = run.snapshot()
    st.subheader(f"Designing… {run.elapsed:.0f} s")
    if st.button("Stop", key=f"stop_{run.id}"):
        run.stop()
        st.info("Stopping after the current step…")
    gate = next((line for line in log if line.startswith("Scope gate")), None)
    rows = [{"": "✅", "step": "Scope gate", "time (s)": ""}] if not gate and (steps or log) else []
    rows += [{"": STEP_ICON.get(s.status, ""), "step": s.label,
              "time (s)": "" if s.seconds is None else f"{s.seconds:.1f}"} for s in steps[-25:]]
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    if log:
        st.caption(log[-1][:300])


# --------------------------------------------------------------------------- results


def show_result(run: Run) -> None:
    if run.state != "done":
        st.warning(run.message or "The run ended without a result.")
        return
    res = run.result
    head = results.headline(res)
    STATUS_STYLE.get(head.status, st.info)(f"**{head.status}** — {head.reason}")
    st.caption(f"{run.label} · {run.elapsed:.0f} s · {run.prompt}")
    tabs = st.tabs(["Summary", "Requirements", "Checks", "Simulation", "3D view", "Report", "Downloads"])
    report = results.report_md(res)
    spec = results.final_spec(res)
    with tabs[0]:
        summary_tab(res, spec)
    with tabs[1]:
        st.markdown(results.section(report, "Requirements") or "No requirements were extracted.")
        corrections = results.section(report, "Corrections made by code")
        if corrections:
            st.markdown("#### Corrections made by code")
            st.markdown(corrections)
    with tabs[2]:
        checks_tab(res)
    with tabs[3]:
        simulation_tab(run, res)
    with tabs[4]:
        if spec is None:
            st.info("No design was made, so there is nothing to show in 3D.")
        else:
            try:
                st.plotly_chart(results.mesh_figure(spec, catalog(), Path(res.out_dir) / "robot.glb"),
                                width="stretch")
                st.caption("The robot at its home pose (MuJoCo geometry exported as GLB). Drag to rotate, scroll "
                           "to zoom.")
            except Exception:
                st.info("The 3D view needs trimesh and plotly (pip install -e \".[ui]\").")
    with tabs[5]:
        st.markdown(re.sub(r"!\[[^\]]*\]\([^)]*\)", "", report) or "No report.")
    with tabs[6]:
        downloads_tab(res)


def summary_tab(res, spec) -> None:
    req = res.requirements
    c = st.columns(4)
    c[0].metric("Robot type", req.robot_type.replace("_", " "))
    if res.outcome is not None:
        c[1].metric("Rounds", len(res.outcome.rounds))
    cost, mass = results.final_value(res, "budget"), results.final_value(res, "mass")
    if cost is not None:
        c[2].metric("Estimated cost", f"${cost:,.0f}")
    if mass is not None:
        c[3].metric("Mass", f"{mass:.2f} kg")
    if spec is None:
        st.info("No design was made: see the headline above and the Report tab.")
        return
    st.markdown(f"**{spec.name}**: {spec.description}")
    rows = results.parts(spec, catalog())
    if rows:
        st.markdown("#### Motors")
        st.dataframe([{"part": r["part"], "motor": r["name"], "price (USD)": r["price_usd"]} for r in rows],
                     hide_index=True, width="stretch")
        st.caption(f"Motors ${sum(r['price_usd'] or 0 for r in rows):,.2f} (catalog prices); the estimated "
                   "cost above adds a rough material cost.")


def checks_tab(res) -> None:
    rounds = results.round_checks(res)
    if not rounds:
        st.info("No checks ran (no design was made).")
        return
    for number, checks in rounds:
        failed = sum(not c["passed"] for c in checks)
        with st.expander(f"Round {number}: {'all checks pass' if not failed else f'{failed} failed'}",
                         expanded=number == rounds[-1][0]):
            rows = [{"": "🟢" if c["passed"] else "🔴", "check": c["check"],
                     "value": c["value"], "limit": c["limit"], "cause": c["cause"] or "",
                     "message": c["message"]} for c in checks]
            st.dataframe(rows, hide_index=True, width="stretch")


def simulation_tab(run: Run, res) -> None:
    rows = results.sim_rows(res)
    if rows:
        st.dataframe([{"": "🟢" if r["passed"] else "🔴", "test": r["check"], "value": r["value"],
                       "limit": r["limit"], "result": r["message"]} for r in rows],
                     hide_index=True, width="stretch")
    else:
        st.info("No simulation ran (no design was made, or the static checks failed first).")
    gifs = results.gifs(res)
    for g in gifs:
        st.image(str(g), caption=g.stem)
    if not gifs:
        demo = gallery.by_key(st.session_state.get("demo")) if run.mode == "Demo" and st.session_state.get("demo") else None
        if demo and demo.gif and demo.gif.is_file():
            st.image(str(demo.gif), caption="Precomputed recording of this demo's design (docs/media)")
        else:
            st.caption("Turn on 'Record GIFs' to render the simulations (slow on a CPU).")


def downloads_tab(res) -> None:
    out = Path(res.out_dir)
    for name, mime in (("spec.json", "application/json"), ("robot.urdf", "application/xml"),
                       ("report.md", "text/markdown")):
        path = out / name
        if path.is_file():
            st.download_button(f"Download {name}", path.read_bytes(), file_name=name, mime=mime, key=f"dl_{name}")
    st.download_button("Download everything (zip)", results.zip_bytes(out), file_name=f"{out.name}.zip",
                       mime="application/zip", key="dl_zip")


# --------------------------------------------------------------------------- other pages


def how_it_works_page() -> None:
    st.title("How it works")
    st.markdown(
        "You type a task in plain English. A language model reads it and writes down what the job needs "
        "(reach, payload, precision, where it drives), then picks a layout: an arm on a desk, a wheeled robot, "
        "or a wheeled robot with an arm. It never writes a number for the robot itself. Ordinary code then "
        "sizes every part from a catalog of real motors, builds a URDF and checks it: static torque, reach, "
        "collisions, tipping, then MuJoCo simulations. Known numeric problems are fixed by code; only then is "
        "the model asked for a different layout. If no robot from the catalog can do the job, TaskBot says so, "
        "with the numbers and one line on what would work.\n\n"
        "**The LLM chooses the layout, code sizes every number, MuJoCo verifies.**")
    st.graphviz_chart("""
digraph { rankdir=LR; node [shape=box, style=rounded, fontname="sans-serif"];
  task [label="Task (plain English)"]; gate [label="Scope gate\\n(code)"];
  req [label="Requirements\\n(LLM)"]; feas [label="Feasibility gate\\n(code)"];
  intent [label="Layout / intent\\n(LLM, no numbers)"]; size [label="Sizer\\n(code: every number)"];
  check [label="Checks + MuJoCo sims\\n(code)"]; fix [label="Automatic fixes\\n(code)"];
  out [label="PASSED / INFEASIBLE /\\nOUT OF SCOPE + report"];
  task -> gate -> req -> feas -> intent -> size -> check -> out; check -> fix -> check;
  check -> intent [label="still failing", style=dashed]; }""")
    text = _read(ROOT / "docs" / "RESULTS.md")
    st.header("Results")
    st.markdown("Real runs with qwen2.5:7b-instruct via Ollama on a laptop CPU. The holdout prompts were never "
                "used to change code or prompts: 7/10 is the honest score on unseen tasks.")
    prog = re.search(r"## Progression\n(.*?)(?=\n## |\Z)", text, re.S)
    if prog:
        st.markdown(prog.group(1))
    main = re.search(r"## Main set \(40 prompts\)\n\n(.*?)\n\n", text, re.S)
    if main:
        st.markdown(main.group(1))
    st.header("Known limits")
    limits = re.search(r"\*\*What TaskBot is, honestly:\*\*\n(.*?)\n\*\*Technical limits", _read(ROOT / "README.md"), re.S)
    st.markdown(limits.group(1) if limits else "See README.md, Known limits.")


def history_page() -> None:
    st.title("History")
    st.caption("This session's runs (kept in memory only; their files are under out/ui/).")
    runs = st.session_state.runs
    if not runs:
        st.info("No runs yet in this session.")
        return
    for run in reversed(runs):
        head = results.headline(run.result).status if run.state == "done" and run.result is not None else run.state
        c = st.columns([5, 2, 1, 1])
        c[0].markdown(f"**{run.prompt[:80]}**  \n{run.label}")
        c[1].markdown(head)
        c[2].markdown(f"{run.elapsed:.0f} s")
        c[3].button("Open", key=f"open_{run.id}", on_click=_open, args=(run.id,))


def _open(run_id: str) -> None:
    st.session_state.current = run_id
    st.session_state.page = "Design"


def _read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# --------------------------------------------------------------------------- main


def main() -> None:
    st.set_page_config(page_title="TaskBot", page_icon="🤖", layout="wide")
    state()
    cfg = sidebar()
    try:
        page = st.session_state.page
        if page == "How it works":
            how_it_works_page()
        elif page == "History":
            history_page()
        else:
            design_page(cfg)
    except Exception as exc:  # never a traceback on the page
        if type(exc).__name__ in ("RerunException", "StopException"):  # Streamlit's own control flow
            raise
        st.error("Something went wrong showing this page. Try again, or open the run from History.")


main()
