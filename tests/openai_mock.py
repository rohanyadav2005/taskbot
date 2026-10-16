"""A local stand-in for an OpenAI-compatible /chat/completions endpoint, for tests.

Each request takes the next scripted response; a response is a dict with
`status` (default 200), `headers`, and either `content` (wrapped in a chat completion
with usage) or `body` (sent raw), plus optional `delay_s`. A callable is given the
parsed request and returns such a dict. The last response repeats when the script
runs out. Every request is recorded in `requests` as {"headers", "json"}.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class MockOpenAI:
    def __init__(self, script):
        self.script = list(script)
        self.requests: list[dict] = []
        self._lock = threading.Lock()
        mock = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                with mock._lock:
                    mock.requests.append({"headers": dict(self.headers), "json": body, "path": self.path})
                    step = mock.script.pop(0) if len(mock.script) > 1 else mock.script[0]
                if callable(step):
                    step = step(body)
                time.sleep(step.get("delay_s", 0))
                if "body" in step:
                    raw = step["body"] if isinstance(step["body"], str) else json.dumps(step["body"])
                else:
                    raw = json.dumps({
                        "choices": [{"message": {"role": "assistant", "content": step["content"]}}],
                        "usage": {"prompt_tokens": step.get("prompt_tokens", 100),
                                  "completion_tokens": step.get("completion_tokens", 50)},
                    })
                data = raw.encode()
                try:
                    self.send_response(step.get("status", 200))
                    for k, v in step.get("headers", {}).items():
                        self.send_header(k, v)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):  # client timed out first
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        self._thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> MockOpenAI:
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()
