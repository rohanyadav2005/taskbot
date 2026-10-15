"""One interface for every LLM backend.

    client = AnthropicClient()            # or OllamaClient(), FakeClient(...)
    spec = client.complete_json(system, user, RobotSpec)

`complete_json` asks for JSON, validates it against the Pydantic model, and on
failure sends the validation errors back to the model and tries again.
Backends only implement `_chat`, which turns a conversation into reply text.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}

DEFAULT_MAX_RETRIES = 3
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


class LLMError(RuntimeError):
    """The backend failed (network, auth, bad response)."""


class LLMOutputError(LLMError):
    """The model never produced output that validates, even after retries."""

    def __init__(self, schema: type[BaseModel], attempts: int, last_error: str, last_reply: str):
        super().__init__(
            f"no valid {schema.__name__} after {attempts} attempts; last error:\n{last_error}"
        )
        self.attempts = attempts
        self.last_error = last_error
        self.last_reply = last_reply


class LLMClient(ABC):
    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        self.max_retries = max_retries

    @abstractmethod
    def _chat(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        """Return the assistant's reply to `messages`."""

    def complete_json(
        self,
        system: str,
        user: str,
        schema: type[T],
        context: Mapping[str, Any] | None = None,
    ) -> T:
        """Get a validated `schema` instance. Makes 1 + max_retries attempts at most.

        `context` is passed to Pydantic validators (see RobotSpec's motor check).
        """
        messages: list[Message] = [{"role": "user", "content": user}]
        attempts = self.max_retries + 1
        for attempt in range(1, attempts + 1):
            reply = self._chat(system, messages, schema)
            try:
                data = extract_json(reply)
                return schema.model_validate(data, context=dict(context or {}))
            except ValueError as exc:  # ValidationError and JSON errors are ValueErrors
                error = describe_error(exc)
            if attempt == attempts:
                raise LLMOutputError(schema, attempts, error, reply)
            messages += [
                {"role": "assistant", "content": reply},
                {"role": "user", "content": retry_message(schema, error)},
            ]
        raise AssertionError("unreachable")


# --------------------------------------------------------------------------- helpers


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse the JSON object in a reply, tolerating ``` fences and stray prose."""
    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end < start:
        raise ValueError("reply contains no JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc


def describe_error(exc: ValueError) -> str:
    if isinstance(exc, ValidationError):
        lines = []
        for e in exc.errors(include_url=False):
            loc = ".".join(str(p) for p in e["loc"]) or "(root)"
            lines.append(f"- {loc}: {e['msg']}")
        return "\n".join(lines)
    return f"- {exc}"


def retry_message(schema: type[BaseModel], error: str) -> str:
    return (
        f"Your JSON does not match the {schema.__name__} schema:\n{error}\n"
        "Fix every error above and reply with the complete corrected JSON only."
    )


# --------------------------------------------------------------------------- backends


class AnthropicClient(LLMClient):
    """Anthropic Messages API. Reads ANTHROPIC_API_KEY from the environment."""

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 8000,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sdk_client: Any = None,
    ):
        super().__init__(max_retries)
        self.model = model or os.environ.get("TASKBOT_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.max_tokens = max_tokens
        if sdk_client is None:
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise LLMError("ANTHROPIC_API_KEY is not set")
            try:
                import anthropic
            except ImportError as exc:
                raise LLMError('install the SDK: pip install -e ".[anthropic]"') from exc
            sdk_client = anthropic.Anthropic()
        self._sdk = sdk_client

    def _chat(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        try:
            resp = self._sdk.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system,
                messages=messages,
            )
        except Exception as exc:  # SDK raises many types; surface them uniformly
            raise LLMError(f"Anthropic API call failed: {exc}") from exc
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class OllamaClient(LLMClient):
    """Local model through Ollama's /api/chat in JSON mode."""

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout_s: float = 300.0,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        super().__init__(max_retries)
        self.model = model or os.environ.get("TASKBOT_MODEL", DEFAULT_OLLAMA_MODEL)
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = (host if "://" in host else f"http://{host}").rstrip("/")
        self.timeout_s = timeout_s

    def _chat(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2},
        }
        req = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                payload = json.loads(resp.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMError(f"Ollama call to {self.host} failed: {exc}") from exc
        try:
            return payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected Ollama response: {payload!r:.200}") from exc


Responder = Callable[[str, list[Message], type[BaseModel]], str]


class FakeClient(LLMClient):
    """Offline backend for tests and demos.

    `replies` maps a schema class name to either a list of replies (returned in
    order, one per call) or a function (system, messages, schema) -> reply.
    Dicts are serialised to JSON. Every call is recorded in `calls`.
    """

    def __init__(
        self,
        replies: Mapping[str, list[str | dict] | Responder],
        max_retries: int = DEFAULT_MAX_RETRIES,
    ):
        super().__init__(max_retries)
        self._replies = {k: (v if callable(v) else list(v)) for k, v in replies.items()}
        self.calls: list[tuple[str, list[Message], str]] = []

    def _chat(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        self.calls.append((system, [dict(m) for m in messages], schema.__name__))
        source = self._replies.get(schema.__name__)
        if source is None:
            raise LLMError(f"FakeClient has no replies for {schema.__name__}")
        if callable(source):
            reply = source(system, messages, schema)
        elif source:
            reply = source.pop(0)
        else:
            raise LLMError(f"FakeClient ran out of replies for {schema.__name__}")
        return reply if isinstance(reply, str) else json.dumps(reply)


def make_client(backend: str, model: str | None = None) -> LLMClient:
    """Build a real backend by name. The fake backend is built by the CLI."""
    if backend == "anthropic":
        return AnthropicClient(model=model)
    if backend == "ollama":
        return OllamaClient(model=model)
    raise ValueError(f"unknown backend '{backend}' (use anthropic or ollama)")
