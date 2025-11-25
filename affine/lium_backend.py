"""
Lium-backed model serving for Affine sampling.

This module provisions short-lived GPU pods on Lium, starts a minimal
OpenAI-compatible HTTP server inside the pod for a requested model,
and exposes the server via a local SSH tunnel so existing evaluation
code can use a base_url like http://127.0.0.1:<port>/v1 unchanged.
"""
from __future__ import annotations

import atexit
import os
import random
import shlex
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

from lium_sdk import Lium, ExecutorInfo, PodInfo


_LOCK = threading.Lock()
_MODEL_TO_SERVER: Dict[str, "ModelServer"] = {}
_SSH_TUNNELS: Dict[int, subprocess.Popen] = {}
_LIUM = Lium()


def _pick_free_port(start: int = 30000, end: int = 45000) -> int:
    """Return an available localhost TCP port."""
    for _ in range(50):
        port = random.randint(start, end)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    # last resort: let the OS pick an ephemeral port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _choose_executor() -> ExecutorInfo:
    """Choose a Lium executor, preferring GPU types from AFFINE_LIUM_GPU."""
    prefs = (os.getenv("AFFINE_LIUM_GPU") or "").split(",")
    prefs = [p.strip().upper() for p in prefs if p.strip()]
    exs = _LIUM.ls()
    if prefs:
        for p in prefs:
            for e in exs:
                if e.gpu_type.upper().startswith(p):
                    return e
    # Prefer docker-in-docker for easier server setups
    for e in exs:
        if getattr(e, "docker_in_docker", False):
            return e
    return exs[0]


def _build_server_py() -> str:
    """Inline FastAPI server exposing minimal OpenAI-compatible endpoints."""
    return r'''
import os, uvicorn, torch
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM

MODEL_ID = os.environ.get("AFFINE_MODEL_ID") or "google/flan-t5-small"
MAX_TOK = int(os.environ.get("AFFINE_MAX_NEW_TOKENS", "64"))
TEMPERATURE = float(os.environ.get("AFFINE_TEMPERATURE", "0.7"))

tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
model = None
try:
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, trust_remote_code=True)
except Exception:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

app = FastAPI()

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = None
    messages: List[ChatMessage]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

class CompRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None

def _gen(prompt: str, max_new: int, temp: float) -> str:
    encoded = tok(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(
            **encoded,
            do_sample=(temp > 0),
            temperature=max(1e-5, temp),
            max_new_tokens=max_new,
            pad_token_id=tok.pad_token_id,
        )
    return tok.decode(out[0], skip_special_tokens=True)

@app.post("/v1/chat/completions")
def chat(req: ChatRequest) -> Dict[str, Any]:
    prompt = "\n".join(m.content for m in req.messages if m.content)
    ans = _gen(prompt, req.max_tokens or MAX_TOK, req.temperature or TEMPERATURE)
    return {
        "id": "chatcmpl",
        "object": "chat.completion",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": ans}}],
        "model": req.model or MODEL_ID,
    }

@app.post("/v1/completions")
def completions(req: CompRequest) -> Dict[str, Any]:
    ans = _gen(req.prompt, req.max_tokens or MAX_TOK, req.temperature or TEMPERATURE)
    return {
        "id": "cmpl",
        "object": "text_completion",
        "choices": [{"index": 0, "text": ans}],
        "model": req.model or MODEL_ID,
    }

def main():
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()
'''


@dataclass
class ModelServer:
    """Represents a running Lium pod hosting a model HTTP server."""
    model: str
    pod: PodInfo
    local_port: Optional[int] = None
    public_url: Optional[str] = None

    @property
    def base_url(self) -> str:
        """Prefer public URL (reachable from containers) else local tunnel."""
        if self.public_url:
            return self.public_url.rstrip("/") + "/v1"
        if self.local_port:
            return f"http://127.0.0.1:{self.local_port}/v1"
        raise RuntimeError("Model server has no reachable URL")


def _try_public_url(pod: PodInfo) -> Optional[str]:
    """Attempt to derive a public HTTP URL from pod port mappings."""
    try:
        # Refresh pod info to get latest ports mapping
        for p in _LIUM.ps():
            if p.id == pod.id:
                pod = p
                break
        # Common shapes: {"8000": 30123} or {"http":30123}
        ports = pod.ports or {}
        port = None
        if "8000" in ports:
            port = ports.get("8000")
        elif "http" in ports:
            port = ports.get("http")
        elif isinstance(ports, dict):
            # Fallback to any int value
            for v in ports.values():
                if isinstance(v, int):
                    port = v
                    break
        host = pod.host
        if host and port:
            return f"http://{host}:{port}"
    except Exception:
        return None
    return None


def _start_server_on_pod(model: str) -> ModelServer:
    """Provision a pod, start server, and open local SSH tunnel."""
    ex = _choose_executor()
    pod_dict = _LIUM.up(executor_id=ex.id, pod_name=f"affine-{int(time.time())}", initial_port_count=1)
    pod = _LIUM.wait_ready(pod_dict, timeout=900)
    if not pod:
        raise RuntimeError("Lium pod failed to become ready")

    # Write server file and start it in background
    env = {
        "AFFINE_MODEL_ID": model,
        "AFFINE_MAX_NEW_TOKENS": os.getenv("AFFINE_MAX_NEW_TOKENS", "64"),
        "AFFINE_TEMPERATURE": os.getenv("AFFINE_TEMPERATURE", "0.7"),
    }
    py_code = _build_server_py()
    cmd = f"""bash -lc 'python3 -m venv /tmp/venv && . /tmp/venv/bin/activate && pip -q install --upgrade pip && pip -q install torch --index-url https://download.pytorch.org/whl/cu121 && pip -q install transformers sentencepiece tiktoken fastapi uvicorn && cat > /tmp/affine_serv.py <<PY
{py_code}
PY
nohup bash -lc "(. /tmp/venv/bin/activate; python /tmp/affine_serv.py)" >/tmp/affine_serv.log 2>&1 & disown
sleep 3
curl -sSf localhost:8000/docs >/dev/null || (tail -n+1 /tmp/affine_serv.log; exit 1)
'"""
    res = _LIUM.exec(pod, command=cmd, env=env)
    if not res.get("success"):
        raise RuntimeError(f"Failed to start server: {res.get('stderr')}")

    # Prefer public URL if available; otherwise, open SSH tunnel -> local_port
    public = _try_public_url(pod)
    if public:
        return ModelServer(model=model, pod=pod, public_url=public)
    local_port = _pick_free_port()
    user = pod.username or "root"
    host = pod.host or "localhost"
    port = pod.ssh_port
    key = str(_LIUM.config.ssh_key_path)
    ssh_cmd = [
        "ssh",
        "-i", key,
        "-N",
        "-o", "StrictHostKeyChecking=no",
        "-p", str(port),
        "-L", f"{local_port}:127.0.0.1:8000",
        f"{user}@{host}",
    ]
    proc = subprocess.Popen(ssh_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _SSH_TUNNELS[local_port] = proc
    return ModelServer(model=model, pod=pod, local_port=local_port)


def ensure_model_server(model: str) -> ModelServer:
    """Return a live server for model, provisioning if necessary."""
    with _LOCK:
        srv = _MODEL_TO_SERVER.get(model)
        if srv:
            return srv
        srv = _start_server_on_pod(model)
        _MODEL_TO_SERVER[model] = srv
        return srv


def stop_all() -> None:
    """Terminate SSH tunnels and pods."""
    with _LOCK:
        for port, proc in list(_SSH_TUNNELS.items()):
            try:
                proc.terminate()
            except Exception:
                pass
            _SSH_TUNNELS.pop(port, None)
        for model, srv in list(_MODEL_TO_SERVER.items()):
            try:
                _LIUM._request("DELETE", f"/pods/{srv.pod.id}")
            except Exception:
                pass
            _MODEL_TO_SERVER.pop(model, None)


@atexit.register
def _cleanup() -> None:
    stop_all()


