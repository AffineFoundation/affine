"""Eight fake loopback vLLM servers for CI and the scorer selftest command."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from sn120_v2.envs.math_pool import MANDATE, Task
from sn120_v2.generator.scoring import rollout_seed

PASS = "<think>The answer follows.</think>\\boxed{1}"
FAIL = "<think>The answer follows.</think>\\boxed{2}"
CAPPED = "<think>unfinished"


def passing(task_id: str, sample: int, mode: str = "mixed") -> bool:
    if mode == "pass":
        return True
    if mode == "fail":
        return False
    return int(hashlib.sha256(f"{task_id}:{sample}".encode()).hexdigest()[:8], 16) % 4 in (1, 2)


class FakeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "AffineGeneratorFake/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _json(self, code: int, value: Any) -> None:
        raw = json.dumps(value).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self) -> None:
        if self.path == "/health":
            # Match vLLM 0.32: successful health is HTTP 200 with no body.
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
        elif self.path == "/v1/models":
            self._json(200, {"data": [{"id": "null"}]})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path != "/v1/chat/completions":
            self._json(404, {"error": "not found"})
            return
        try:
            farm = self.server.farm
            length = int(self.headers.get("Content-Length", ""))
            body = json.loads(self.rfile.read(length))
            assert isinstance(body, dict)
            assert body["model"] == "null"
            assert body["n"] == 1
            assert body["temperature"] == 1.0
            assert body["top_p"] == 0.95
            assert body["top_k"] == 20
            assert body["max_tokens"] == 65536
            assert body["skip_special_tokens"] is False
            assert body["chat_template_kwargs"] == {"enable_thinking": True}
            assert type(body["seed"]) is int and 0 <= body["seed"] < 2**48
            assert body["messages"][0] == {"role": "system", "content": MANDATE}
            assert body["messages"][1]["role"] == "user"
            task_id, sample = farm.identify(body["seed"])
            assert isinstance(task_id, str) and type(sample) is int
        except (AssertionError, KeyError, IndexError, TypeError, ValueError):
            self._json(400, {"error": "invalid Generator request"})
            return
        farm.record(body)
        time.sleep(farm.delay_s)
        mode = farm.mode
        if mode == "capped":
            text, reason = CAPPED, "length"
        elif passing(task_id, sample, mode):
            text, reason = PASS, "stop"
        else:
            text, reason = FAIL, "stop"
        self._json(200, {
            "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": reason}],
            "usage": {"prompt_tokens": len(body["messages"][1]["content"]), "completion_tokens": len(text)},
        })


class FakeFarm:
    """One shared request log behind eight independently bound HTTP servers."""

    def __init__(self, *, delay_s: float = 0.002, mode: str = "mixed", first_port: int = 8100) -> None:
        self.delay_s, self.mode, self.first_port = delay_s, mode, first_port
        self.seeds: dict[int, tuple[str, int]] = {}
        self.requests: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self.servers: list[ThreadingHTTPServer] = []
        self.threads: list[threading.Thread] = []

    def start(self) -> "FakeFarm":
        for gpu in range(8):
            port = 0 if self.first_port == 0 else self.first_port + gpu
            server = ThreadingHTTPServer(("127.0.0.1", port), FakeHandler)
            server.daemon_threads = True
            server.farm = self
            thread = threading.Thread(target=server.serve_forever, name=f"fake-vllm-{gpu}", daemon=True)
            thread.start()
            self.servers.append(server)
            self.threads.append(thread)
        return self

    @property
    def urls(self) -> tuple[str, ...]:
        return tuple(f"http://127.0.0.1:{server.server_address[1]}" for server in self.servers)

    def identify(self, seed: int) -> tuple[str, int]:
        try:
            return self.seeds[seed]
        except KeyError:
            raise AssertionError("seed is not from a declared selftest job") from None

    def record(self, body: dict[str, Any]) -> None:
        with self._lock:
            self.requests.append(body)

    def close(self) -> None:
        for server in self.servers:
            server.shutdown()
            server.server_close()
        for thread in self.threads:
            thread.join(timeout=5)
        self.servers.clear()
        self.threads.clear()

    def __enter__(self) -> "FakeFarm":
        return self.start()

    def __exit__(self, *args: Any) -> None:
        self.close()


class FakePool:
    def tasks(self, task_ids: list[str] | tuple[str, ...]) -> tuple[Task, ...]:
        return tuple(Task(task_id=task_id, slice="math", problem="What is 1?", answer="1") for task_id in task_ids)
