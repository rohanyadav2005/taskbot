"""One interface for every LLM backend.

    client = AnthropicClient()   # or OpenAICompatClient(), OllamaClient(), FakeClient(...)
    spec = client.complete_json(system, user, RobotSpec)

`complete_json` asks for JSON, validates it against the Pydantic model, and on
failure sends the validation errors back to the model and tries again.
Backends only implement `_chat`, which turns a conversation into reply text.
Rate-limit errors are retried with exponential backoff (honouring Retry-After).
Each client counts its calls, tokens and JSON retries in `client.usage`.
API keys are never put in messages, errors or `usage`.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)
Message = dict[str, str]  # {"role": "user" | "assistant", "content": str}

DEFAULT_MAX_RETRIES = 3          # JSON validation retries per complete_json call
DEFAULT_RATE_LIMIT_RETRIES = 4   # waits after a rate-limit error, per model call
DEFAULT_BACKOFF_S = 2.0          # first wait; doubles each time (Retry-After wins if given)
MAX_BACKOFF_S = 60.0
DEFAULT_TIMEOUT_S = 120.0        # remote APIs
DEFAULT_LOCAL_TIMEOUT_S = 600.0  # a model on this machine (CPU inference can be slow)
DEFAULT_NUM_CTX = 16384          # native Ollama context window (Ollama's own default is small)
CHARS_PER_TOKEN_MIN = 3.0        # conservative: real prompts average more characters per token
DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"


class LLMError(RuntimeError):
    """The backend failed (network, auth, bad response)."""


class LLMRateLimitError(LLMError):
    """The provider said slow down (HTTP 429, or 503 overloaded). Retried with backoff."""

    def __init__(self, message: str, retry_after_s: float | None = None):
        super().__init__(message)
        self.retry_after_s = retry_after_s


class LLMTruncatedError(LLMError):
    """The prompt didn't fit the context window, or the reply was cut off at the token limit."""


class LLMOutputError(LLMError):
    """The model never produced output that validates, even after retries."""

    def __init__(self, schema: type[BaseModel], attempts: int, last_error: str, last_reply: str):
        super().__init__(
            f"no valid {schema.__name__} after {attempts} attempts; last error:\n{last_error}"
        )
        self.schema_name = schema.__name__
        self.attempts = attempts
        self.last_error = last_error
        self.last_reply = last_reply


@dataclass
class Usage:
    """What one client used: recorded per run in run.json and the report."""

    provider: str
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    json_retries: int = 0       # replies that failed JSON/schema validation and were retried
    rate_limit_waits: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def as_dict(self) -> dict:
        return asdict(self) | {"total_tokens": self.total_tokens}


class LLMClient(ABC):
    provider = "unknown"

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES,
                 rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
                 backoff_s: float = DEFAULT_BACKOFF_S):
        self.max_retries = max_retries
        self.rate_limit_retries = rate_limit_retries
        self.backoff_s = backoff_s
        self.usage = Usage(self.provider, "")
        self._sleep = time.sleep  # replaced in tests

    def _call(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        """`_chat`, waiting and retrying on rate-limit errors."""
        for attempt in range(self.rate_limit_retries + 1):
            try:
                self.usage.calls += 1
                return self._chat(system, messages, schema)
            except LLMRateLimitError as exc:
                if attempt == self.rate_limit_retries:
                    raise
                wait = exc.retry_after_s if exc.retry_after_s is not None else self.backoff_s * 2**attempt
                self.usage.rate_limit_waits += 1
                self._sleep(min(max(wait, 0.0), MAX_BACKOFF_S))
        raise AssertionError("unreachable")

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
            reply = self._call(system, messages, schema)
            try:
                data = extract_json(reply)
                return schema.model_validate(data, context=dict(context or {}))
            except ValueError as exc:  # ValidationError and JSON errors are ValueErrors
                error = describe_error(exc)
            if attempt == attempts:
                raise LLMOutputError(schema, attempts, error, reply)
            self.usage.json_retries += 1
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


def is_local_url(url: str) -> bool:
    host = (urllib.parse.urlsplit(url if "://" in url else f"http://{url}").hostname or "").lower()
    return host in ("localhost", "127.0.0.1", "::1", "0.0.0.0") or host.endswith(".local")


def default_timeout_s(url: str) -> float:
    """TASKBOT_TIMEOUT_S if set, else 600 s for a model on this machine, 120 s for a remote API."""
    env = os.environ.get("TASKBOT_TIMEOUT_S")
    if env:
        return float(env)
    return DEFAULT_LOCAL_TIMEOUT_S if is_local_url(url) else DEFAULT_TIMEOUT_S


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

    provider = "anthropic"

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 8000,
        max_retries: int = DEFAULT_MAX_RETRIES,
        sdk_client: Any = None,
    ):
        super().__init__(max_retries)
        self.model = model or os.environ.get("TASKBOT_MODEL", DEFAULT_ANTHROPIC_MODEL)
        self.usage.model = self.model
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
            if getattr(exc, "status_code", None) in (429, 529) or type(exc).__name__ == "RateLimitError":
                raise LLMRateLimitError(f"Anthropic rate limit: {exc}") from exc
            raise LLMError(f"Anthropic API call failed: {exc}") from exc
        u = getattr(resp, "usage", None)
        self.usage.input_tokens += int(getattr(u, "input_tokens", 0) or 0)
        self.usage.output_tokens += int(getattr(u, "output_tokens", 0) or 0)
        return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")


class OllamaClient(LLMClient):
    """Local model through Ollama's /api/chat in JSON mode. (Ollama also serves the
    OpenAI-compatible API at /v1, usable through OpenAICompatClient.)"""

    provider = "ollama"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout_s: float | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        num_ctx: int | None = None,
    ):
        super().__init__(max_retries)
        self.model = model or os.environ.get("TASKBOT_MODEL", DEFAULT_OLLAMA_MODEL)
        host = host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        self.host = (host if "://" in host else f"http://{host}").rstrip("/")
        self.timeout_s = timeout_s or default_timeout_s(self.host)
        self.num_ctx = num_ctx or int(os.environ.get("TASKBOT_NUM_CTX", DEFAULT_NUM_CTX))
        self.usage.model = self.model

    def _chat(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        body = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.2, "num_ctx": self.num_ctx},
        }
        chars = sum(len(m["content"]) for m in body["messages"])
        if chars / CHARS_PER_TOKEN_MIN > self.num_ctx:
            # Ollama would silently drop the start of the prompt: the schema and catalog.
            raise LLMTruncatedError(
                f"the prompt is {chars} characters (up to ~{chars / CHARS_PER_TOKEN_MIN:.0f} tokens) and may "
                f"not fit num_ctx {self.num_ctx}; Ollama would cut its start. Raise TASKBOT_NUM_CTX.")
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
            content = payload["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMError(f"unexpected Ollama response: {payload!r:.200}") from exc
        prompt_n, reply_n = int(payload.get("prompt_eval_count") or 0), int(payload.get("eval_count") or 0)
        self.usage.input_tokens += prompt_n
        self.usage.output_tokens += reply_n
        if payload.get("done_reason") == "length" or prompt_n + reply_n >= self.num_ctx:
            raise LLMTruncatedError(
                f"Ollama filled its {self.num_ctx}-token context ({prompt_n} prompt + {reply_n} reply "
                f"tokens, done_reason {payload.get('done_reason')!r}): the prompt or reply was cut off. "
                "Raise TASKBOT_NUM_CTX.")
        return content


class OpenAICompatClient(LLMClient):
    """Any OpenAI-compatible /chat/completions endpoint: Ollama (/v1), Groq, Gemini,
    OpenRouter, vLLM, ...

    Configured from the environment (arguments override):
        TASKBOT_BASE_URL   e.g. http://localhost:11434/v1
        TASKBOT_MODEL      e.g. qwen2.5:7b-instruct
        TASKBOT_API_KEY    optional (local Ollama needs none); sent as a Bearer token only
        TASKBOT_JSON_MODE  "0" to never ask for JSON mode
        TASKBOT_TIMEOUT_S  per request; default 600 s for a local URL, 120 s otherwise

    JSON mode (response_format json_object) is requested by default; if the provider
    rejects that parameter, it is dropped for the rest of the run and the reply is left
    to Pydantic validation and the usual retries.

    A reply cut off at the token limit (finish_reason "length") raises LLMTruncatedError.
    The prompt's own context window is the server's business: for Ollama's /v1 endpoint
    set OLLAMA_CONTEXT_LENGTH before `ollama serve`, or use the native ollama backend,
    which sets num_ctx itself.
    """

    provider = "openai_compat"

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        json_mode: bool | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        rate_limit_retries: int = DEFAULT_RATE_LIMIT_RETRIES,
        backoff_s: float = DEFAULT_BACKOFF_S,
    ):
        super().__init__(max_retries, rate_limit_retries, backoff_s)
        self.base_url = (base_url or os.environ.get("TASKBOT_BASE_URL") or "").rstrip("/")
        if not self.base_url:
            raise LLMError("TASKBOT_BASE_URL is not set (e.g. http://localhost:11434/v1 for Ollama)")
        self.model = model or os.environ.get("TASKBOT_MODEL") or ""
        if not self.model:
            raise LLMError("TASKBOT_MODEL is not set (the provider's model name)")
        self._api_key = (api_key if api_key is not None else os.environ.get("TASKBOT_API_KEY")) or None
        self.json_mode = (json_mode if json_mode is not None
                          else os.environ.get("TASKBOT_JSON_MODE", "1") != "0")
        self.timeout_s = timeout_s or default_timeout_s(self.base_url)
        host = urllib.parse.urlsplit(self.base_url).netloc or self.base_url
        self.usage = Usage(f"openai_compat ({host})", self.model)

    def __repr__(self) -> str:  # never show the key
        return (f"OpenAICompatClient(base_url={self.base_url!r}, model={self.model!r}, "
                f"api_key={'set' if self._api_key else 'none'})")

    def _redact(self, text: str) -> str:
        return text.replace(self._api_key, "***") if self._api_key else text

    def _chat(self, system: str, messages: list[Message], schema: type[BaseModel]) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": system}, *messages],
            "temperature": 0.2,
        }
        if self.json_mode:
            body["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = resp.read()
        except urllib.error.HTTPError as exc:
            text = self._redact(exc.read().decode("utf-8", "replace"))[:400]
            if exc.code in (429, 503):
                retry = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    retry_s = float(retry) if retry is not None else None
                except ValueError:
                    retry_s = None
                raise LLMRateLimitError(f"{exc.code} from {url}: {text}", retry_s) from None
            if exc.code == 400 and self.json_mode and "response_format" in text.lower():
                self.json_mode = False  # provider doesn't take JSON mode; rely on validation
                return self._chat(system, messages, schema)
            raise LLMError(f"HTTP {exc.code} from {url}: {text}") from None
        except (TimeoutError, socket.timeout) as exc:
            raise LLMError(f"{url} timed out after {self.timeout_s:g} s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise LLMError(f"{url} timed out after {self.timeout_s:g} s") from None
            raise LLMError(f"can't reach {url}: {self._redact(str(exc.reason))}") from None
        try:
            payload = json.loads(raw)
            choice = payload["choices"][0]
            content = choice["message"]["content"] or ""
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            raise LLMError(f"unexpected response from {url}: {self._redact(raw.decode('utf-8', 'replace'))[:200]}") from None
        u = payload.get("usage") or {}
        self.usage.input_tokens += int(u.get("prompt_tokens") or 0)
        self.usage.output_tokens += int(u.get("completion_tokens") or 0)
        if choice.get("finish_reason") == "length":
            raise LLMTruncatedError(
                f"the reply from {self.model} was cut off at the token limit after "
                f"{u.get('completion_tokens', '?')} tokens (finish_reason 'length'), so its JSON is incomplete. "
                "Use a model or provider with a larger output limit.")
        return content


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
        self.usage = Usage("fake", "scripted")
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
    if backend == "openai_compat":
        return OpenAICompatClient(model=model)
    raise ValueError(f"unknown backend '{backend}' (use anthropic, openai_compat or ollama)")
