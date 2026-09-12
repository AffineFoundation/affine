"""Model clients for the probe: the research teacher boxes and the king box.

Both wrap affine.evalsrv.vllm_client.VllmModel so sampling and echoes are
rendered byte-for-byte the way the duel renders them (gen_prompt /
force_text / thought_text through the model's own chat template, direct
/v1/completions, add_special_tokens=False, echo tail xarg for the cache
plugin). The only additions are the bearer header and the hint placement.
"""
from __future__ import annotations

import asyncio
import copy
import json
import os
import sys
import urllib.request
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))

from evalsrv.chat import gen_prompt  # noqa: E402
from evalsrv.vllm_client import Served, VllmModel  # noqa: E402

TEACHER_REPO = "Qwen/Qwen3.8-27B"
KING_TOK_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                  "special_tokens_map.json", "vocab.json", "merges.txt",
                  "added_tokens.json", "generation_config.json", "config.json")
HINT_HEADER = "\n\n[Reviewer note]\n"


def with_hint(prefix: list[dict], hint: str | None, placement: str = "last_user") -> list[dict]:
    """x + h: the hint appended to the last user message (default) or to the
    first system message. The miner never sees this; only teacher sampling."""
    if not hint:
        return prefix
    msgs = copy.deepcopy(prefix)
    if placement == "system" and msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = msgs[0]["content"] + HINT_HEADER + hint
        return msgs
    for m in reversed(msgs):
        if m["role"] == "user":
            m["content"] = m["content"] + HINT_HEADER + hint
            break
    return msgs


class Box:
    """One vLLM replica serving one model behind a bearer."""

    def __init__(self, name: str, base_url: str, key: str, repo: str,
                 model_name: str | None = None, concurrency: int = 48,
                 require_think_close: bool = False):
        self.name = name
        self.base_url = base_url
        self.http = httpx.AsyncClient(headers={"Authorization": f"Bearer {key}"},
                                      timeout=httpx.Timeout(600.0, connect=15.0),
                                      limits=httpx.Limits(max_connections=concurrency + 8))
        self.sem = asyncio.Semaphore(concurrency)
        self.model = VllmModel(Served(name=name, repo=repo, revision=None, port=0,
                                      base_url=base_url, model_name=model_name),
                               self.http, self.sem,
                               require_think_close=require_think_close)
        self.n_calls = 0

    async def sample_raw(self, prefix: list[dict], temperature: float, max_tokens: int) -> str:
        """Natural rollout text (starts inside <think>), unsplit — recorded verbatim."""
        d = await self.model._post({
            "model": self.model.cfg.request_model,
            "prompt": gen_prompt(self.model.cfg.repo, self.model.cfg.revision, prefix),
            "max_tokens": max_tokens, "temperature": temperature,
            "add_special_tokens": False,
        })
        self.n_calls += 1
        return d["choices"][0]["text"]

    async def score_action(self, prefix, thoughts, action) -> dict:
        self.n_calls += 1
        return await self.model.score_action(prefix, thoughts, action)

    async def score_thought(self, prefix, thoughts) -> dict:
        self.n_calls += 1
        return await self.model.score_thought(prefix, thoughts)

    async def aclose(self):
        await self.http.aclose()


def load_boxes(state_json: Path, concurrency: int = 48) -> list[Box]:
    st = json.loads(Path(state_json).read_text())
    boxes = []
    for name, mem in st["pods"].items():
        if mem.get("base_url") and mem.get("ready_at"):
            boxes.append(Box(name, mem["base_url"], mem["key"], TEACHER_REPO,
                             concurrency=concurrency))
    if not boxes:
        raise SystemExit("no ready teacher box in pods.json")
    return boxes


def king_tokenizer_dir(digest: str, dst: Path) -> Path:
    """Fetch the public king's tokenizer files (models.affine.io) once."""
    dst.mkdir(parents=True, exist_ok=True)
    for fname in KING_TOK_FILES:
        p = dst / fname
        if p.exists():
            continue
        url = f"https://models.affine.io/models/sha256/{digest}/{fname}"
        req = urllib.request.Request(url, headers={"User-Agent": "affine-hints/1"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                p.write_bytes(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 404:
                raise
    return dst


def load_king(dst: Path, concurrency: int = 4) -> Box | None:
    base = os.environ.get("KING_BASE_URL")
    key = os.environ.get("KING_KEY")
    model = os.environ.get("KING_MODEL")
    digest = os.environ.get("KING_DIGEST")
    if not (base and key and model and digest):
        return None
    tok_dir = king_tokenizer_dir(digest, dst)
    return Box("king", base, key, str(tok_dir), model_name=model,
               concurrency=concurrency, require_think_close=True)
