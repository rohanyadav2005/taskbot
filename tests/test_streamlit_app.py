"""The Streamlit web app (streamlit_app.py), driven headless with streamlit.testing.
The app only calls existing functions; these tests check what a user sees."""

import os
import socket
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from taskbot.ui import backends, results  # noqa: E402
from taskbot.ui.runner import MAX_PROMPT_CHARS, check_prompt, friendly_error  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "streamlit_app.py")
SECRET = "sk-test-NEVER-SHOW-THIS-1234567890"


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("TASKBOT_OUT", str(tmp_path / "out"))
    monkeypatch.setenv("MUJOCO_GL", os.environ.get("MUJOCO_GL", "disable"))
    return AppTest.from_file(APP, default_timeout=300)


def texts(at) -> str:
    """Everything the page shows as text, widgets included."""
    parts = []
    for kind in ("markdown", "success", "info", "warning", "error", "caption", "title", "header", "subheader",
                 "text", "code", "metric"):
        for el in getattr(at, kind):
            parts.append(str(getattr(el, "value", "")))
    for w in list(at.text_input) + list(at.text_area) + list(at.sidebar.text_input):
        parts.append(f"{w.label}={w.value}")
    return "\n".join(parts)


def wait_done(at):
    for _ in range(600):
        if not any(s.value.startswith("Designing") for s in at.subheader):
            return at
        at.run()
    raise AssertionError("the run did not finish")


def test_the_app_loads_in_demo_mode(app):
    at = app.run()
    assert not at.exception
    assert at.title[0].value == "Design a robot"
    assert at.sidebar.radio(key="mode").value == "Demo"
    assert at.sidebar.selectbox(key="demo").value == "windowsill"


@pytest.mark.slow
def test_a_demo_replays_end_to_end_and_passes(app):
    at = app.run()
    at.sidebar.selectbox(key="demo").set_value("windowsill").run()
    at = wait_done(at.button(key="design").click().run())
    assert not at.exception and not at.error
    assert at.success[0].value.startswith("**PASSED** — every check passed")
    assert [t.label for t in at.tabs] == ["Summary", "Requirements", "Checks", "Simulation", "3D view", "Report",
                                         "Downloads"]
    assert "Corrections made by code" in texts(at)
    at.sidebar.radio(key="page").set_value("History").run()
    assert "Demo: Windowsill watering arm" in texts(at) and "PASSED" in texts(at)


def test_an_out_of_scope_demo_says_so(app):
    at = app.run()
    at.sidebar.selectbox(key="demo").set_value("drone").run()
    at = wait_done(at.button(key="design").click().run())
    assert not at.exception
    assert at.info[0].value.startswith("**OUT OF SCOPE** — it needs a robot that flies")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_a_missing_ollama_server_gives_a_friendly_message(app):
    at = app.run()
    at.sidebar.radio(key="mode").set_value("Local (Ollama)").run()
    at.sidebar.text_input(key="ollama_url").set_value(f"http://127.0.0.1:{_free_port()}").run()
    at.button(key="design").click().run()
    assert not at.exception
    message = at.error[0].value
    assert message.startswith("Can't reach Ollama at http://127.0.0.1:")
    assert "**Fix:** Start it (open the Ollama app, or run `ollama serve`" in message
    assert "Traceback" not in texts(at)


def test_the_api_key_is_never_rendered(app):
    at = app.run()
    at.sidebar.radio(key="mode").set_value("Cloud API").run()
    at.sidebar.text_input(key="cloud_url").set_value(f"http://127.0.0.1:{_free_port()}/v1").run()
    at.sidebar.text_input(key="cloud_model").set_value("some-model").run()
    at.sidebar.text_input(key="cloud_key").set_value(SECRET).run()
    at.text_area(key="prompt").set_value("water the three plants on my windowsill").run()
    at = wait_done(at.button(key="design").click().run())
    page = "\n".join(str(getattr(el, "value", "")) for kind in ("markdown", "success", "info", "warning", "error",
                                                                "caption", "title", "subheader", "metric")
                     for el in getattr(at, kind))
    assert SECRET not in page and "sk-test" not in page
    from streamlit.proto.TextInput_pb2 import TextInput

    assert at.sidebar.text_input(key="cloud_key").proto.type == TextInput.PASSWORD
    at.sidebar.radio(key="page").set_value("History").run()
    assert SECRET not in texts(at).replace(f"API key={SECRET}", "")


def test_prompt_limits_and_friendly_errors():
    assert check_prompt("") == "Describe the job you want a robot to do."
    assert check_prompt("x" * (MAX_PROMPT_CHARS + 1)).startswith(f"Please keep the task under {MAX_PROMPT_CHARS}")
    assert check_prompt("water my plants") is None
    from taskbot.llm import LLMError

    assert "Bearer" not in friendly_error(LLMError("HTTP 401: bad token Bearer sk-abc"))
    assert friendly_error(ValueError("boom")).startswith("TaskBot hit an internal error")


def test_ollama_check_rejects_bad_urls_without_a_request():
    check = backends.check_ollama("file:///etc/passwd", "m")
    assert not check.ok and "isn't an http(s) URL" in check.message


def test_out_of_scope_headline_from_a_real_result(tmp_path):
    transcript = Path(__file__).parent / "fixtures" / "real_runs" / "smoke_2026-11-07" / "6_drone.jsonl"
    from taskbot.llm import ReplayClient
    from taskbot.pipeline import run_pipeline
    from taskbot.schema import load_motor_catalog

    res = run_pipeline(ReplayClient(transcript), "a flying drone that delivers coffee to my room", load_motor_catalog(),
                       tmp_path, record=False)
    head = results.headline(res)
    assert head.status == "OUT OF SCOPE" and head.reason.startswith("it needs a robot that flies")
    assert results.zip_bytes(res.out_dir)[:2] == b"PK"


def test_cloud_api_defaults_to_openai_and_reads_its_key_without_showing_it(app, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", SECRET)
    at = app.run()
    at.sidebar.radio(key="mode").set_value("Cloud API").run()
    assert at.sidebar.selectbox(key="cloud_provider").value == "OpenAI"
    assert at.sidebar.text_input(key="cloud_url").value == "https://api.openai.com/v1"
    assert at.sidebar.text_input(key="cloud_model").value == "gpt-4o-mini"
    assert at.sidebar.text_input(key="cloud_key").value in ("", None)  # the key is not put in the field
    assert "Using OPENAI_API_KEY from the app's secrets or environment." in texts(at)
    assert SECRET not in texts(at)
    at.sidebar.selectbox(key="cloud_provider").set_value("Groq").run()
    assert at.sidebar.text_input(key="cloud_url").value == "https://api.groq.com/openai/v1"
