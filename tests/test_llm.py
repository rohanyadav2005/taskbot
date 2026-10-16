import io
import json
import urllib.error
from types import SimpleNamespace

import pytest
from lxml import etree
from pydantic import ValidationError

from conftest import EXAMPLES_DIR
from taskbot import cli
from taskbot.design import design_robot, design_system_prompt
from taskbot.llm import (
    AnthropicClient,
    FakeClient,
    LLMError,
    LLMOutputError,
    OllamaClient,
    extract_json,
    make_client,
)
from taskbot.requirements import Requirements, extract_requirements
from taskbot.schema import RobotSpec, load_motor_catalog

CATALOG = load_motor_catalog()
SPEC = json.loads((EXAMPLES_DIR / "arm_2dof.json").read_text())
REQ = {
    "task_summary": "Move a 200 g cup 30 cm across a desk.",
    "robot_type": "arm",
    "reach_m": 0.33,
    "payload_kg": 0.25,
    "workspace": {"x_min": -0.3, "x_max": 0.3, "y_min": -0.3, "y_max": 0.3, "z_min": 0, "z_max": 0.2},
    "dof_min": 4,
    "gripper_needed": True,
    "gripper_kind": "parallel",
    "position_tolerance_m": 0.01,
    "tolerance_reason": "a mug only needs to land on the desk",
    "assumptions": ["a mug weighs about 0.2 kg; +25% margin"],
}


# --------------------------------------------------------------------------- complete_json


def test_valid_reply_returned_first_try():
    fake = FakeClient({"Requirements": [REQ]})
    req = fake.complete_json("sys", "move a cup", Requirements)
    assert req.reach_m == 0.33
    assert len(fake.calls) == 1
    system, messages, schema = fake.calls[0]
    assert system == "sys" and schema == "Requirements"
    assert messages == [{"role": "user", "content": "move a cup"}]


def test_validation_errors_are_sent_back_then_retry_succeeds():
    bad = REQ | {"reach_m": -1, "gripper_kind": None}
    fake = FakeClient({"Requirements": [bad, REQ]})
    req = fake.complete_json("sys", "move a cup", Requirements)
    assert req.payload_kg == 0.25
    assert len(fake.calls) == 2
    retry_messages = fake.calls[1][1]
    assert [m["role"] for m in retry_messages] == ["user", "assistant", "user"]
    assert json.loads(retry_messages[1]["content"]) == bad  # the model sees its own reply
    feedback = retry_messages[2]["content"]
    assert "reach_m: Input should be greater than 0" in feedback
    assert "Requirements schema" in feedback


def test_unparseable_reply_is_retried():
    fake = FakeClient({"Requirements": ["Sure! I think the arm needs...", REQ]})
    fake.complete_json("sys", "u", Requirements)
    assert "reply contains no JSON object" in fake.calls[1][1][2]["content"]


def test_gives_up_after_three_retries():
    fake = FakeClient({"Requirements": [{"nope": 1}] * 10})
    with pytest.raises(LLMOutputError) as info:
        fake.complete_json("sys", "u", Requirements)
    assert len(fake.calls) == 4  # first attempt + 3 retries
    assert info.value.attempts == 4
    assert "nope" in info.value.last_error  # extra field rejected
    assert "task_summary" in info.value.last_error


def test_max_retries_is_configurable():
    fake = FakeClient({"Requirements": [{}] * 3}, max_retries=0)
    with pytest.raises(LLMOutputError):
        fake.complete_json("sys", "u", Requirements)
    assert len(fake.calls) == 1


def test_context_reaches_validators_and_errors_feed_back():
    # sts3215 is not in this catalog, so the first spec fails the motor check.
    small_catalog = {k: v for k, v in CATALOG.items() if k != "sts3215"}
    fixed = json.loads(json.dumps(SPEC).replace("sts3215", "xl430_w250"))
    fake = FakeClient({"RobotSpec": [SPEC, fixed]})
    spec = fake.complete_json("sys", "u", RobotSpec, context={"motor_ids": set(small_catalog)})
    assert spec.name == "arm_2dof"
    assert "unknown motor_id 'sts3215'" in fake.calls[1][1][2]["content"]


@pytest.mark.parametrize(
    "text",
    [
        '{"a": 1}',
        'Here you go:\n```json\n{"a": 1}\n```\nLet me know!',
        'Result: {"a": 1} -- done',
    ],
)
def test_extract_json_tolerates_fences_and_prose(text):
    assert extract_json(text) == {"a": 1}


def test_extract_json_reports_bad_json():
    with pytest.raises(ValueError, match="invalid JSON"):
        extract_json('{"a": 1,}')


def test_fake_client_raises_when_unconfigured_or_exhausted():
    fake = FakeClient({"Requirements": []})
    with pytest.raises(LLMError, match="ran out"):
        fake.complete_json("s", "u", Requirements)
    with pytest.raises(LLMError, match="no replies for RobotSpec"):
        fake.complete_json("s", "u", RobotSpec)


# --------------------------------------------------------------------------- requirements


def test_requirements_gripper_rules():
    with pytest.raises(ValidationError, match="gripper_kind must be 'parallel' or 'suction'"):
        Requirements(**(REQ | {"gripper_kind": None}))
    with pytest.raises(ValidationError, match="gripper_kind must be null"):
        Requirements(**(REQ | {"gripper_needed": False}))
    Requirements(**(REQ | {"gripper_needed": False, "gripper_kind": None}))


def test_workspace_min_must_not_exceed_max():
    ws = REQ["workspace"] | {"z_min": 0.5, "z_max": 0.1}
    with pytest.raises(ValidationError, match="z_min .* must not exceed z_max"):
        Requirements(**(REQ | {"workspace": ws}))


def test_out_of_scope_assumptions_are_flagged():
    req = Requirements(**(REQ | {"assumptions": ["ok", "OUT OF SCOPE: needs to walk upstairs"]}))
    assert req.out_of_scope == ["OUT OF SCOPE: needs to walk upstairs"]


def test_extract_requirements_sends_prompt_and_schema():
    fake = FakeClient({"Requirements": [REQ]})
    extract_requirements(fake, "water my plants")
    system, messages, _ = fake.calls[0]
    assert system.startswith("You are a robotics requirements engineer.")
    assert '"assumptions"' in system and "Schema:" in system
    assert messages[0]["content"] == "water my plants"


# --------------------------------------------------------------------------- design


def test_design_prompt_has_schema_and_catalog_inserted():
    system = design_system_prompt(CATALOG)
    assert "{schema}" not in system and "{motors}" not in system
    assert system.startswith("You are a robot designer.")
    assert '"fixed_base"' in system and '"arm_chain"' in system
    for motor_id in CATALOG:
        assert f'"{motor_id}"' in system
    assert "rated_torque_nm" in system
    assert "sources" not in system  # citations stay out of the prompt


def test_design_robot_sends_requirements_and_validates():
    fake = FakeClient({"RobotSpec": [SPEC]})
    spec = design_robot(fake, Requirements(**REQ), CATALOG)
    assert isinstance(spec, RobotSpec)
    user = fake.calls[0][1][0]["content"]
    assert user.startswith("Requirements:") and '"reach_m": 0.33' in user


# --------------------------------------------------------------------------- backends


class FakeSDK:
    def __init__(self, text):
        self.kwargs = None
        self.messages = SimpleNamespace(create=self._create)
        self._text = text

    def _create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(content=[
            SimpleNamespace(type="thinking", thinking="hmm"),
            SimpleNamespace(type="text", text=self._text[:5]),
            SimpleNamespace(type="text", text=self._text[5:]),
        ])


def test_anthropic_backend_calls_messages_api():
    sdk = FakeSDK(json.dumps(REQ))
    client = AnthropicClient(model="test-model", sdk_client=sdk)
    req = client.complete_json("the system", "the task", Requirements)
    assert req.dof_min == 4
    assert sdk.kwargs["model"] == "test-model"
    assert sdk.kwargs["system"] == "the system"
    assert sdk.kwargs["messages"] == [{"role": "user", "content": "the task"}]


def test_anthropic_backend_needs_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        AnthropicClient()


def test_anthropic_api_errors_become_llm_errors():
    def boom(**_):
        raise RuntimeError("overloaded")

    client = AnthropicClient(sdk_client=SimpleNamespace(messages=SimpleNamespace(create=boom)))
    with pytest.raises(LLMError, match="overloaded"):
        client.complete_json("s", "u", Requirements)


def test_ollama_backend_uses_json_mode(monkeypatch):
    sent = {}

    def fake_urlopen(req, timeout):
        sent["url"] = req.full_url
        sent["body"] = json.loads(req.data)
        reply = {"message": {"role": "assistant", "content": json.dumps(REQ)}}
        return io.BytesIO(json.dumps(reply).encode())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    client = OllamaClient(model="llama-test", host="localhost:11434")
    assert client.complete_json("the system", "the task", Requirements).reach_m == 0.33
    assert sent["url"] == "http://localhost:11434/api/chat"
    body = sent["body"]
    assert body["model"] == "llama-test" and body["format"] == "json" and body["stream"] is False
    assert body["messages"][0] == {"role": "system", "content": "the system"}
    assert body["messages"][1] == {"role": "user", "content": "the task"}


def test_ollama_connection_failure_is_llm_error(monkeypatch):
    def refuse(req, timeout):
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", refuse)
    with pytest.raises(LLMError, match="connection refused"):
        OllamaClient().complete_json("s", "u", Requirements)


def test_make_client_rejects_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        make_client("gpt-nine")


# --------------------------------------------------------------------------- CLI


def test_cli_design_with_fake_backend_writes_outputs(tmp_path, capsys):
    argv = ["design", "pick up a cup", "--backend", "fake", "--out", str(tmp_path),
            "--no-sim", "--no-record"]
    assert cli.main(argv) == 0
    out_dir = tmp_path / "fake_arm_4dof"
    req = Requirements.model_validate_json((out_dir / "requirements.json").read_text())
    assert req.task_summary == "pick up a cup"
    RobotSpec.model_validate_json((out_dir / "spec.json").read_text())
    assert etree.fromstring((out_dir / "robot.urdf").read_bytes()).tag == "robot"
    summary = capsys.readouterr().out
    assert "pick up a cup" in summary and "shoulder_pitch" in summary

    assert cli.main(argv) == 0  # a second run must not overwrite the first
    assert (tmp_path / "fake_arm_4dof_2" / "spec.json").exists()


def test_cli_reports_llm_errors(monkeypatch, tmp_path, capsys):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert cli.main(["design", "x", "--backend", "anthropic", "--out", str(tmp_path)]) == 1
    assert "ANTHROPIC_API_KEY" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())
