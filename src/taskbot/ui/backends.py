"""Which LLM the web app talks to, and checks with plain-English fixes.

Three modes: "demo" (the replay backend over a recorded transcript: no LLM), "local" (Ollama's
native API) and "cloud" (any OpenAI-compatible API). The API key is passed straight to
OpenAICompatClient, which sends it only as a Bearer header and never shows it (its __repr__ hides
it); nothing here logs, stores or returns it.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from taskbot.llm import DEFAULT_NUM_CTX, LLMClient, OllamaClient, OpenAICompatClient, ReplayClient

DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:7b-instruct"
CHECK_TIMEOUT_S = 3.0


@dataclass
class Check:
    ok: bool
    message: str
    fix: str = ""


def clean_url(url: str) -> str | None:
    """An http(s) URL with a host, without a trailing slash; None for anything else."""
    url = (url or "").strip().rstrip("/")
    parts = urllib.parse.urlsplit(url)
    if parts.scheme not in ("http", "https") or not parts.netloc:
        return None
    return url


def _get_json(url: str, body: dict | None = None, timeout: float = CHECK_TIMEOUT_S) -> dict:
    data = None if body is None else json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"},
                                 method="POST" if body is not None else "GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_ollama(base_url: str, model: str) -> Check:
    """Is Ollama reachable at base_url, is the model pulled, and is its context long enough for
    TaskBot (which asks for num_ctx = DEFAULT_NUM_CTX tokens)?"""
    url = clean_url(base_url)
    if url is None:
        return Check(False, f"'{base_url}' isn't an http(s) URL.", f"Use something like {DEFAULT_OLLAMA_URL}.")
    try:
        tags = _get_json(f"{url}/api/tags")
    except (urllib.error.URLError, OSError, ValueError):
        return Check(False, f"Can't reach Ollama at {url}.",
                     "Start it (open the Ollama app, or run `ollama serve` in a terminal) and try again. "
                     "Local mode only works when this app runs on the same computer as Ollama; on "
                     "Streamlit Community Cloud use Demo or Cloud API mode.")
    names = {m.get("name", "") for m in tags.get("models", [])}
    if model not in names and f"{model}:latest" not in names:
        return Check(False, f"Ollama is running, but the model '{model}' isn't pulled.",
                     f"Run `ollama pull {model}` in a terminal, then try again.")
    try:
        info = _get_json(f"{url}/api/show", {"model": model}).get("model_info", {})
        ctx = max((int(v) for k, v in info.items() if k.endswith(".context_length")), default=0)
    except (urllib.error.URLError, OSError, ValueError, TypeError):
        ctx = 0
    if ctx and ctx < DEFAULT_NUM_CTX:
        return Check(False, f"'{model}' handles {ctx} tokens of context; TaskBot's prompts need {DEFAULT_NUM_CTX}.",
                     "Use a model with a longer context (e.g. qwen2.5:7b-instruct, 32k).")
    detail = f" (context {ctx} tokens; TaskBot asks for {DEFAULT_NUM_CTX})" if ctx else ""
    return Check(True, f"Ollama is running at {url} with {model}{detail}.")


def cloud_settings_check(base_url: str, model: str, api_key: str) -> Check:
    if clean_url(base_url) is None:
        return Check(False, "The base URL must be an http(s) URL.",
                     "e.g. https://api.groq.com/openai/v1 (see README, OpenAI-compatible backends).")
    if not model.strip():
        return Check(False, "Enter the provider's model name.", "e.g. llama-3.3-70b-versatile on Groq.")
    if not api_key:
        return Check(False, "Enter an API key (or put TASKBOT_API_KEY in the app's secrets).",
                     "The key is only sent to the provider as a Bearer header; it is never shown or saved.")
    return Check(True, f"Ready: {model} at {urllib.parse.urlsplit(base_url).netloc}.")


def replay_client(transcript: Path) -> LLMClient:
    return ReplayClient(transcript)


def ollama_client(base_url: str, model: str) -> LLMClient:
    return OllamaClient(model=model, host=clean_url(base_url) or DEFAULT_OLLAMA_URL)


def cloud_client(base_url: str, model: str, api_key: str) -> LLMClient:
    return OpenAICompatClient(model=model.strip(), base_url=clean_url(base_url), api_key=api_key)
