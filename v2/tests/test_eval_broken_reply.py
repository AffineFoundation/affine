"""A reply that breaks off while it is read is a failed try, the same as a refused connection.

The server here is real: a local HTTP server on a free port, read by the
evaluator's own ``http_json``. It can stop inside the body of a reply (the
standard library raises ``IncompleteRead``) and it can answer with a first
line that is not HTTP (``BadStatusLine``). Neither is an ``OSError``.
"""

from __future__ import annotations

import dataclasses
import json
import threading
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from sn120_v2 import layout
from sn120_v2.canonical import manifest_digest, sha256_hex
from sn120_v2.errors import InfraError
from sn120_v2.gpu import plan
from sn120_v2.gpu.evaluator import ServeSettings, VllmEvaluator, http_json
from sn120_v2.store import LocalStore
from sn120_v2.types import NULL_MODEL, EvalRequest, HeadVersion

TASKS = ("t0", "t1", "t2", "t3")
CUT, NOT_HTTP = "cut", "not http"  # the two ways a reply breaks
POOL = SimpleNamespace(task=lambda task_id: SimpleNamespace(messages=lambda: [{"role": "user", "content": task_id}]))
GRADER = SimpleNamespace(grade_many=lambda jobs: [SimpleNamespace(code=text, machine_cause=None) for _, text, _ in jobs])


class Handler(BaseHTTPRequestHandler):
    """One vLLM server. ``server.breaks`` maps a path or a ``(model, task)`` to the ways its next replies break."""

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        self.answer(self.path)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path != "/v1/chat/completions":
            return self.answer(self.path)
        key = (body["model"], body["messages"][-1]["content"])
        with self.server.lock:
            self.server.chats.append(key)
        self.answer(key, {"choices": [{"message": {"content": "P"}, "finish_reason": "stop"}], "usage": {"completion_tokens": 1}})

    def answer(self, key, reply: dict | None = None) -> None:
        with self.server.lock:
            waiting = self.server.breaks.get(key)
            how = waiting.pop(0) if waiting else None
        data = json.dumps(reply if reply is not None else {"data": []}).encode("utf-8")
        if how == NOT_HTTP:
            self.wfile.write(b"this is not HTTP\r\n\r\n")
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data[:10] if how == CUT else data)  # cut: the headers promise the whole reply, then the server stops


@pytest.fixture
def served(tmp_path, build):
    """The evaluator of a box with one GPU, a head and one LoRA in the store, and one real local server."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.lock, server.chats, server.breaks = threading.Lock(), [], {}
    threading.Thread(target=server.serve_forever, daemon=True).start()
    store = LocalStore(tmp_path / "store")

    def put(prefix_of, files: dict[str, bytes]) -> str:
        identity = manifest_digest({name: sha256_hex(data) for name, data in files.items()})
        for name, data in files.items():
            store.put(f"{prefix_of(identity)}/{name}", data)
        return identity

    base = put(layout.public_base_prefix, {"config.json": b"{}", "model.safetensors": b"bf16"})
    master = put(layout.public_master_prefix, {"model.part00.safetensors": b"fp32"})
    head = HeadVersion.make(parent_version_id=None, base_hash=base, base_fp32_master_digest=master, loras=(), round_id="genesis", applied_at=1)
    params = build.params(n_gpus=1)
    files = {"adapter_config.json": b"{}", "adapter_model.safetensors": b"lora"}
    identity = put(lambda digest: f"pending/loras/{digest}", files)
    lora = dataclasses.replace(build.artifact(head, build.entry(0), params), lora_hash=identity, prefix=f"pending/loras/{identity}", files={name: sha256_hex(data) for name, data in files.items()})
    settings = ServeSettings(first_port=server.server_address[1], start_timeout_s=10.0, clients_per_server=4)
    process = SimpleNamespace(alive=lambda: True, stop=lambda: None)
    seam = VllmEvaluator(store, plan.BoxConfig(tmp_path / "work", build.image), POOL, GRADER, spawn=lambda argv, env, log: process, gpus_free=lambda limit, timeout: None, settings=settings, sleep=lambda s: None)
    request = EvalRequest("r000001", head, (lora,), TASKS, "seed-0", 0, params, 0, 1)
    yield SimpleNamespace(seam=seam, request=request, server=server, lora=identity, url=f"http://127.0.0.1:{server.server_address[1]}")
    server.shutdown()
    server.server_close()


@pytest.mark.parametrize("how, cause", [(CUT, "IncompleteRead"), (NOT_HTTP, "BadStatusLine")])
def test_a_reply_that_breaks_off_is_a_connection_error(served, how, cause):
    served.server.breaks["/v1/models"] = [how]
    with pytest.raises(ConnectionError, match=f"the reply broke off: {cause}") as caught:
        http_json(served.url + "/v1/models", None, 10.0)
    assert isinstance(caught.value, OSError)
    assert http_json(served.url + "/v1/models", None, 10.0) == {"data": []}


def test_a_rollout_whose_reply_breaks_off_is_tried_again_and_the_look_is_scored(served):
    server, lora = served.server, served.lora
    # One reply of null M stops inside its body. One reply of the LoRA is not HTTP in two tries. The next try of each is whole.
    server.breaks.update({(NULL_MODEL, "t1"): [CUT], (lora, "t2"): [NOT_HTTP, CUT]})
    assert served.seam.evaluate(served.request).outcomes == {NULL_MODEL: ("PPPP",), lora: ("PPPP",)}
    sent = Counter(server.chats)
    assert (sent[NULL_MODEL, "t1"], sent[lora, "t2"], len(server.chats)) == (2, 3, 8 + 3)


def test_a_rollout_whose_reply_breaks_off_in_every_try_is_one_e_as_a_refused_connection_is(served):
    server, lora = served.server, served.lora
    server.breaks[lora, "t3"] = [CUT, NOT_HTTP, CUT]  # all 3 tries (ServeSettings.retries + 1)
    assert served.seam.evaluate(served.request).outcomes == {NULL_MODEL: ("PPPP",), lora: ("PPPE",)}
    assert Counter(server.chats)[lora, "t3"] == 3 and len(server.chats) == 8 + 2


def test_a_broken_reply_to_the_load_of_an_adapter_is_a_failure_of_the_box(served):
    served.server.breaks["/v1/load_lora_adapter"] = [CUT]
    with pytest.raises(InfraError, match="/v1/load_lora_adapter failed: the reply broke off: IncompleteRead"):
        served.seam.evaluate(served.request)
    assert not served.server.chats  # no rollout was sent
