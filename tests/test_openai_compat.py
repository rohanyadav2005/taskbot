"""The openai_compat backend against a mocked HTTP server (never a real provider)."""

import json

import pytest
from openai_mock import MockOpenAI
from pydantic import BaseModel

from taskbot.llm import LLMError, LLMRateLimitError, OpenAICompatClient, make_client

KEY = "sk-test-SECRET-123"


class Point(BaseModel):
    x: float
    y: float


def client(url, **kw):
    kw.setdefault("api_key", None)
    c = OpenAICompatClient(model="test-model", base_url=url, **kw)
    c.waits = []
    c._sleep = c.waits.append  # record backoff instead of sleeping
    return c


def test_valid_json_reply_with_json_mode_and_usage():
    with MockOpenAI([{"content": '{"x": 1, "y": 2}', "prompt_tokens": 120, "completion_tokens": 30}]) as m:
        c = client(m.url, api_key=KEY)
        assert c.complete_json("sys", "give a point", Point) == Point(x=1, y=2)
    req = m.requests[0]
    assert req["path"] == "/v1/chat/completions"
    assert req["json"]["model"] == "test-model"
    assert req["json"]["response_format"] == {"type": "json_object"}
    assert req["json"]["messages"][0] == {"role": "system", "content": "sys"}
    assert req["headers"]["Authorization"] == f"Bearer {KEY}"
    u = c.usage
    assert (u.provider, u.model, u.calls, u.input_tokens, u.output_tokens, u.json_retries) == (
        f"openai_compat (127.0.0.1:{m.url.split(':')[2].split('/')[0]})", "test-model", 1, 120, 30, 0)


def test_no_key_sends_no_authorization_header():
    with MockOpenAI([{"content": '{"x": 1, "y": 2}'}]) as m:
        client(m.url).complete_json("sys", "u", Point)
    assert "Authorization" not in m.requests[0]["headers"]


def test_invalid_json_then_valid_retry_counts_one_json_retry():
    with MockOpenAI([{"content": "Sure! here is a point: {x: 1"}, {"content": '{"x": 3, "y": 4}'}]) as m:
        c = client(m.url)
        assert c.complete_json("sys", "u", Point) == Point(x=3, y=4)
    assert c.usage.json_retries == 1 and c.usage.calls == 2
    retry = m.requests[1]["json"]["messages"]
    assert retry[-1]["role"] == "user" and "does not match the Point schema" in retry[-1]["content"]


def test_schema_error_then_valid_retry():
    with MockOpenAI([{"content": '{"x": 1}'}, {"content": '{"x": 1, "y": 0}'}]) as m:
        c = client(m.url)
        c.complete_json("sys", "u", Point)
    assert c.usage.json_retries == 1
    assert "- y: Field required" in m.requests[1]["json"]["messages"][-1]["content"]


def test_timeout_is_an_llm_error():
    with MockOpenAI([{"content": '{"x": 1, "y": 2}', "delay_s": 1.0}]) as m:
        c = client(m.url, timeout_s=0.2)
        with pytest.raises(LLMError, match="timed out after 0.2 s"):
            c.complete_json("sys", "u", Point)


def test_rate_limit_waits_retry_after_then_succeeds():
    with MockOpenAI([{"status": 429, "headers": {"Retry-After": "3"}, "body": {"error": "slow down"}},
                     {"content": '{"x": 1, "y": 2}'}]) as m:
        c = client(m.url)
        assert c.complete_json("sys", "u", Point).x == 1
    assert c.waits == [3.0] and c.usage.rate_limit_waits == 1 and c.usage.json_retries == 0


def test_rate_limit_backs_off_exponentially_then_gives_up():
    with MockOpenAI([{"status": 429, "body": {"error": "rate limited"}}]) as m:
        c = client(m.url, rate_limit_retries=3, backoff_s=2.0)
        with pytest.raises(LLMRateLimitError, match="429"):
            c.complete_json("sys", "u", Point)
    assert c.waits == [2.0, 4.0, 8.0] and len(m.requests) == 4


def test_provider_without_json_mode_falls_back_to_validation():
    def reply(body):
        if "response_format" in body:
            return {"status": 400, "body": {"error": {"message": "response_format is not supported"}}}
        return {"content": '```json\n{"x": 5, "y": 6}\n```'}

    with MockOpenAI([reply]) as m:
        c = client(m.url)
        assert c.complete_json("sys", "u", Point) == Point(x=5, y=6)
        c.complete_json("sys", "u", Point)
    assert c.json_mode is False
    assert ["response_format" in r["json"] for r in m.requests] == [True, False, False]


def test_key_never_appears_in_errors_or_repr():
    def echo_auth(body):  # a hostile/buggy server that echoes the header back
        return {"status": 401, "body": {"error": "bad key Bearer " + KEY}}

    with MockOpenAI([echo_auth]) as m:
        c = client(m.url, api_key=KEY)
        with pytest.raises(LLMError) as err:
            c.complete_json("sys", "u", Point)
    assert KEY not in str(err.value) and "***" in str(err.value)
    assert KEY not in repr(c) and KEY not in json.dumps(c.usage.as_dict())


def test_configuration_from_environment(monkeypatch):
    monkeypatch.setenv("TASKBOT_BASE_URL", "http://localhost:11434/v1/")
    monkeypatch.setenv("TASKBOT_MODEL", "qwen2.5:7b-instruct")
    monkeypatch.delenv("TASKBOT_API_KEY", raising=False)
    c = make_client("openai_compat")
    assert isinstance(c, OpenAICompatClient)
    assert (c.base_url, c.model, c.json_mode) == ("http://localhost:11434/v1", "qwen2.5:7b-instruct", True)
    monkeypatch.setenv("TASKBOT_JSON_MODE", "0")
    assert make_client("openai_compat", model="other").json_mode is False


def test_missing_configuration_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("TASKBOT_BASE_URL", raising=False)
    with pytest.raises(LLMError, match="TASKBOT_BASE_URL is not set"):
        OpenAICompatClient(model="m")
    monkeypatch.setenv("TASKBOT_BASE_URL", "http://x/v1")
    monkeypatch.delenv("TASKBOT_MODEL", raising=False)
    with pytest.raises(LLMError, match="TASKBOT_MODEL is not set"):
        OpenAICompatClient()


def _scripted_pipeline(body):
    """Requirements for the requirements step, the passing mobile design for the design step."""
    from taskbot.fake import _load

    system = body["messages"][0]["content"]
    if "requirements engineer" in system:
        return {"content": json.dumps(_load("mobile_requirements"))}
    return {"content": json.dumps(_load("mobile_design_round2"))}


def test_cli_runs_the_pipeline_on_openai_compat_and_records_usage(tmp_path, monkeypatch, capsys):
    from taskbot.cli import main

    with MockOpenAI([_scripted_pipeline]) as m:
        monkeypatch.setenv("TASKBOT_BASE_URL", m.url)
        monkeypatch.setenv("TASKBOT_MODEL", "mock-model")
        monkeypatch.setenv("TASKBOT_API_KEY", KEY)
        code = main(["design", "a robot that carries a 2 kg box across my office",
                     "--backend", "openai_compat", "--out", str(tmp_path), "--no-sim", "--no-record"])
    assert code == 0
    (out,) = tmp_path.iterdir()
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["model"] == "mock-model" and run["provider"].startswith("openai_compat")
    assert run["calls"] == 2 and run["total_tokens"] == 300 and run["json_retries"] == 0
    assert "## LLM usage" in (out / "report.md").read_text(encoding="utf-8")
    captured = capsys.readouterr()
    for f in out.rglob("*"):
        if f.is_file():
            assert KEY not in f.read_text(encoding="utf-8", errors="ignore")
    assert KEY not in captured.out + captured.err


def test_reply_cut_off_at_the_token_limit_is_a_truncation_error():
    from taskbot.llm import LLMTruncatedError

    def cut(body):
        return {"body": {"choices": [{"message": {"content": '{"x": 1, "y'}, "finish_reason": "length"}],
                         "usage": {"prompt_tokens": 900, "completion_tokens": 4096}}}

    with MockOpenAI([cut]) as m:
        c = client(m.url)
        with pytest.raises(LLMTruncatedError, match="cut off at the token limit after 4096 tokens"):
            c.complete_json("sys", "u", Point)
    assert len(m.requests) == 1 and c.usage.json_retries == 0  # not retried as bad JSON


def test_timeout_defaults_for_local_and_remote_urls(monkeypatch):
    monkeypatch.delenv("TASKBOT_TIMEOUT_S", raising=False)
    assert OpenAICompatClient(model="m", base_url="http://localhost:11434/v1").timeout_s == 600
    assert OpenAICompatClient(model="m", base_url="https://openrouter.ai/api/v1").timeout_s == 120
