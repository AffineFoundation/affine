"""Shared plumbing for the frontier-arbiter probes (2026-09-20).

Frontier arbiter = the idea that a stronger model F (GLM / DeepSeek / Kimi via
Engy) samples the same turn prefix as the teacher C; where F agrees with C the
turn keeps the live teacher score; where F disagrees the miner wins the turn by
being closer to F than C was.

Everything here is API-only (no GPU):
  * `Engy` — async chat sampling for any Engy model, with cost accounting
    (`x_engy.charged_micro` = micro-USD per call) and retries.
  * `TeacherEcho` — teacher-forced logprobs of a span under the frozen teacher
    `qwen3.8-27b` through Engy's `/completions` echo path. Engy caps `echo` at
    1024 prompt tokens unless the prompt is sent as TOKEN IDS with
    `logprob_start_len`; then only the span is scored (verified 2026-09-20:
    4.4k-token prompt, 16-token span, $0.0002, ~1 s, bit-identical on repeat).
    Rendering follows the live evalsrv (`thought_rendering = "as_generated"`,
    wvk 22): assistant body = latent + "\\n</think>" [+ "\\n\\n" + visible]
    [+ "\\n\\n" + action]; scored spans = latent (+ visible) for thoughts, the
    action bytes for actions. The prompt is the teacher's chat template with
    add_generation_prompt=True ending inside the open <think>.
  * dialect-aware action normalisation + token-Jaccard (ported from
    research/scripts/frontier_rule_probe.py so numbers stay comparable).
  * verdict / corpus access: stored duel records live on the validator box
    (`~/subnet120/affine/state/evals/chal-XXXXX.json.gz`); prefixes are
    re-materialised from the public corpus (data.affine.io) by the record's
    pinned manifest through evalsrv.corpus.CorpusSync.

Secrets: ENGY_2 from the environment, else `op read` from the Arbos vault
(service account token OP_SERVICE_ACCOUNT_TOKEN). SSH to the box uses
~/.ssh/arbos_box (op://Arbos/nlijfp36ed4aqbkp2svh2lefbi).
"""

from __future__ import annotations

import asyncio
import gzip
import json
import math
import os
import random
import re
import statistics as st
import subprocess
import sys
import time
from functools import lru_cache
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402
from evalsrv.chat import split_rollout  # noqa: E402

ENGY_BASE = "https://api.engy.ai/v1"
TEACHER_REPO = "Qwen/Qwen3.8-27B"
TEACHER_ENGY = "qwen3.8-27b"
GENESIS_ENGY = "qwen3.6-35b-a3b"
FRONTIER_MODELS = ("glm-5.3", "glm-5.3-flash", "deepseek-v4.1-flash", "kimi-k3")
# USD per 1M prompt tokens, measured 2026-09-20 from x_engy.charged_micro on a
# 4k-token prompt (completion share negligible). Completion prices are higher;
# read the real charge from `Engy.cost_usd`.
PRICE_PER_M = {"glm-5.3": 0.70, "glm-5.3-flash": 0.10, "deepseek-v4.1-flash": 0.03,
               "kimi-k3": 1.40, "qwen3.8-27b": 0.03, "qwen3.6-35b-a3b": 0.05}

BOX = "const@204.12.171.6"
BOX_KEY = Path.home() / ".ssh" / "arbos_box"
BOX_REPO = "~/subnet120"
DATA = REPO / "research" / "data" / "frontier_arbiter"
EVALS_CACHE = DATA / "evals"

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
Z_SPLIT = "\n" + THINK_CLOSE + "\n"
UNCOND_PREFIX: list[dict] = [{"role": "user", "content": ""}]
DUEL_MAX_TOKENS = 2048 + 768      # wvk 18+: miner thought cap 2048 + action 768


# ------------------------------------------------------------------ secrets
def op_read(ref: str) -> str:
    out = subprocess.run(["op", "read", ref], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"op read failed for {ref}: {out.stderr.strip()[:200]}")
    return out.stdout.strip()


def engy_key() -> str:
    k = os.environ.get("ENGY_2") or os.environ.get("ENGY")
    if not k:
        k = op_read("op://Arbos/hbmcxmszp7fthxom3kcsvd52la/credential")
        os.environ["ENGY_2"] = k
    return k


def hf_token() -> str | None:
    k = os.environ.get("HF_TOKEN")
    if not k:
        try:
            k = op_read("op://Arbos/vxynhbmikcllahzj3vnswggzsq/credential")
            os.environ["HF_TOKEN"] = k
        except RuntimeError:
            return None
    return k


# ------------------------------------------------------------------ io
def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def write_jsonl(path: Path, recs: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


# ------------------------------------------------------------------ Engy
class Engy:
    """Async Engy client. One completion per call (Engy ignores `n`).

    chat(model, messages, temperature, max_tokens) -> dict:
        reasoning, content, tool_calls, finish, usage, cost_usd, worker
    echo_ids(ids, start) -> list[float | None] logprobs of ids[start:], under
        the teacher (first entry is None: Engy gives no logprob for the token at
        `start` itself, so callers pass start-1 and drop it).
    Cost is tallied in `self.cost_usd` / `self.usage` per model.
    """

    def __init__(self, key: str | None = None, concurrency: int = 16,
                 timeout: float = 900.0, retries: int = 6):
        self.cli = httpx.AsyncClient(base_url=ENGY_BASE, timeout=timeout,
                                     headers={"Authorization": f"Bearer {key or engy_key()}"})
        self.sem = asyncio.Semaphore(concurrency)
        self.retries = retries
        self.usage: dict[str, dict] = {}
        self.cost_usd = 0.0

    def _tally(self, model: str, d: dict) -> float:
        u = d.get("usage") or {}
        x = d.get("x_engy") or {}
        cost = float(x.get("charged_micro") or 0) / 1e6
        m = self.usage.setdefault(model, {"calls": 0, "prompt_tokens": 0,
                                          "completion_tokens": 0, "cost_usd": 0.0})
        m["calls"] += 1
        m["prompt_tokens"] += int(u.get("prompt_tokens") or 0)
        m["completion_tokens"] += int(u.get("completion_tokens") or 0)
        m["cost_usd"] += cost
        self.cost_usd += cost
        return cost

    async def _post(self, path: str, payload: dict, model: str) -> dict:
        last = None
        for attempt in range(self.retries):
            async with self.sem:
                try:
                    r = await self.cli.post(path, json=payload)
                    if r.status_code == 200:
                        d = r.json()
                        if d.get("choices"):
                            return d
                        last = f"200 without choices: {r.text[:200]}"
                    elif r.status_code in (400, 404, 413, 422):
                        raise RuntimeError(f"engy {r.status_code}: {r.text[:300]}")
                    else:
                        last = f"HTTP {r.status_code}: {r.text[:200]}"
                except (httpx.HTTPError, ValueError) as e:
                    last = repr(e)[:200]
            await asyncio.sleep(3 * (attempt + 1) + random.random() * 2)
        raise RuntimeError(f"engy {model} failed after {self.retries} tries: {last}")

    async def chat(self, model: str, messages: list[dict], temperature: float = 0.8,
                   max_tokens: int = DUEL_MAX_TOKENS, **extra) -> dict:
        payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, **extra}
        d = await self._post("/chat/completions", payload, model)
        cost = self._tally(model, d)
        ch = d["choices"][0]
        msg = ch.get("message") or {}
        return {"reasoning": msg.get("reasoning_content") or "",
                "content": msg.get("content") or "",
                "tool_calls": msg.get("tool_calls") or [],
                "finish": ch.get("finish_reason"), "usage": d.get("usage"),
                "cost_usd": cost, "worker": (d.get("x_engy") or {}).get("worker"),
                "model": d.get("model") or model}

    ECHO_SPAN_CAP = 1024      # Engy backend limit per request (verified 2026-09-20)

    async def echo_ids(self, ids: list[int], start: int,
                       model: str = TEACHER_ENGY) -> list[float | None]:
        """Logprobs of ids[start:] under `model`. Spans longer than the cap are
        scored in chunks: chunk (s, e) sends prompt=ids[:e] with
        logprob_start_len=s, whose first entry (token s) is None, so chunks
        overlap by one token and the None is dropped for every chunk but the
        first."""
        out: list[float | None] = []
        s = start
        while s < len(ids):
            e = min(len(ids), s + self.ECHO_SPAN_CAP)
            payload = {"model": model, "prompt": ids[:e], "max_tokens": 1, "echo": True,
                       "logprobs": 1, "temperature": 0.0, "logprob_start_len": s}
            d = await self._post("/completions", payload, model)
            self._tally(model, d)
            lp = ((d["choices"][0].get("logprobs") or {}).get("token_logprobs") or [])[: e - s]
            out.extend(lp if s == start else lp[1:])
            s = e - 1 if e < len(ids) else e
        return out


# ------------------------------------------------------------------ rendering (live evalsrv, as_generated)
@lru_cache(maxsize=2)
def teacher_tokenizer():
    from transformers import AutoTokenizer  # heavy import kept lazy for CLI help
    return AutoTokenizer.from_pretrained(TEACHER_REPO, token=hf_token())


def split_z(z: str) -> tuple[str, str]:
    if Z_SPLIT in z:
        latent, _, visible = z.partition(Z_SPLIT)
        return latent, visible
    if z.startswith(THINK_CLOSE + "\n"):
        return "", z[len(THINK_CLOSE) + 1:]
    return z, ""


def gen_prompt(prefix_messages: list[dict]) -> str:
    tok = teacher_tokenizer()
    p = tok.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)
    if not p.rstrip().endswith(THINK_OPEN):
        p = p + THINK_OPEN
    return p


def thought_body(thoughts: str, rendering: str = "as_generated") -> tuple[str, list[tuple[int, int]]]:
    if rendering == "canonical":
        head = THINK_CLOSE + "\nTHOUGHT: "
        return head + thoughts, [(len(head), len(head) + len(thoughts))]
    latent, visible = split_z(thoughts)
    body = latent + "\n" + THINK_CLOSE
    spans = [(0, len(latent))] if latent else []
    if visible:
        body += "\n\n"
        spans.append((len(body), len(body) + len(visible)))
        body += visible
    return body, spans


def force_text(prefix_messages: list[dict], thoughts: str, action: str,
               rendering: str = "as_generated") -> str:
    body, _ = thought_body(thoughts, rendering)
    return gen_prompt(prefix_messages) + body + "\n\n" + action


def thought_text(prefix_messages: list[dict], thoughts: str,
                 rendering: str = "as_generated") -> tuple[str, list[tuple[int, int]]]:
    gp = gen_prompt(prefix_messages)
    body, spans = thought_body(thoughts, rendering)
    return gp + body, [(len(gp) + a, len(gp) + b) for a, b in spans]


class TeacherEcho:
    """lpC(span | context) per byte under the frozen teacher, via Engy echo.

    Returns {"sum_lp", "n_tokens", "n_bytes", "lp_per_byte"} (+ "tokens" =
    [(start, end, lp)] relative to the first span start when tokens=True) —
    the same fields evalsrv's VllmModel echoes return. Span tokens = those
    whose start offset lies inside any scored span.
    """

    def __init__(self, engy: Engy):
        self.engy = engy
        self.tok = teacher_tokenizer()
        self.n_calls = 0

    def _encode(self, full: str) -> tuple[list[int], list[tuple[int, int]]]:
        enc = self.tok(full, add_special_tokens=False, return_offsets_mapping=True)
        return enc["input_ids"], [tuple(o) for o in enc["offset_mapping"]]

    async def _echo(self, full: str, spans: list[tuple[int, int]], tokens: bool = False) -> dict:
        n_bytes = sum(len(full[a:b].encode()) for a, b in spans) or 1
        if not spans:
            return {"sum_lp": 0.0, "n_tokens": 0, "n_bytes": 1, "lp_per_byte": 0.0,
                    **({"tokens": []} if tokens else {})}
        ids, offs = self._encode(full)
        in_span = [any(a <= s < b for a, b in spans) for s, _ in offs]
        first = next((i for i, v in enumerate(in_span) if v), None)
        if first is None:
            raise ValueError("span not found in tokenization")
        start = max(first - 1, 0)
        lps = await self.engy.echo_ids(ids, start)
        self.n_calls += 1
        # lps[i] is the logprob of ids[start + i]; the first (at `start`) is None.
        scored = []
        for i, lp in enumerate(lps):
            j = start + i
            if j < len(in_span) and in_span[j] and lp is not None:
                scored.append((offs[j][0], offs[j][1], lp))
        s = sum(lp for _, _, lp in scored)
        out = {"sum_lp": s, "n_tokens": len(scored), "n_bytes": n_bytes, "lp_per_byte": s / n_bytes}
        if tokens:
            base = spans[0][0]
            out["tokens"] = [(a - base, b - base, lp) for a, b, lp in scored]
        return out

    async def lp_action(self, prefix: list[dict], z: str, y: str,
                        rendering: str = "as_generated") -> dict:
        full = force_text(prefix, z, y, rendering)
        return await self._echo(full, [(len(full) - len(y), len(full))])

    async def lp_thought(self, prefix: list[dict], z: str, tokens: bool = False,
                         rendering: str = "as_generated") -> dict:
        full, spans = thought_text(prefix, z, rendering)
        return await self._echo(full, spans, tokens)

    async def lp_thought_uncond(self, z: str, tokens: bool = True,
                                rendering: str = "as_generated") -> dict:
        return await self.lp_thought(UNCOND_PREFIX, z, tokens, rendering)


# ------------------------------------------------------------------ rollouts
FUNC_OPEN_RE = re.compile(r"(?m)^function=")


def render_tool_call(call: dict) -> str:
    fn = call.get("function") or {}
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except ValueError:
            pass
    body = json.dumps({"name": fn.get("name"), "arguments": args}, ensure_ascii=False)
    return f"<tool_call>\n{body}\n</tool_call>"


def repair_tool_call(content: str) -> tuple[str, bool]:
    """Undo Engy's tool-parser mangling of GLM XML tool calls (2026-09-06)."""
    fixed = FUNC_OPEN_RE.sub("<function=", content)
    if fixed.count("<tool_call>") > fixed.count("</tool_call>") and \
            fixed.rstrip().endswith("</function>"):
        fixed = fixed.rstrip() + "\n</tool_call>"
    return fixed, fixed != content


def reply_to_rollout(reply: dict, kind: str) -> dict:
    """(z, y) of one Engy chat reply in the turn's dialect, duel-style:
    reasoning_content is the latent thought, content the visible reply."""
    content = reply.get("content") or ""
    if reply.get("tool_calls"):
        content = content.rstrip() + "\n" + "\n".join(render_tool_call(c) for c in reply["tool_calls"])
    repaired = False
    if kind == "tool_call":
        content, repaired = repair_tool_call(content)
    text = (reply.get("reasoning") or "") + "\n" + THINK_CLOSE + "\n" + content
    z, y = split_rollout(text, kind)
    return {"z": z, "y": y, "parsed": bool(y), "repaired": repaired,
            "finish": reply.get("finish"), "reasoning_chars": len(reply.get("reasoning") or ""),
            "content_chars": len(content), "cost_usd": reply.get("cost_usd"),
            "model": reply.get("model")}


# ------------------------------------------------------------------ action similarity
_TOKEN_RE = re.compile(r"[A-Za-z0-9_./-]+|[^\sA-Za-z0-9_./-]")


def strip_fence(y: str) -> str:
    y = y.strip()
    m = re.match(r"^```[a-zA-Z_]*\n(.*?)\n?```$", y, re.S)
    return m.group(1).strip() if m else y


def norm_action(y: str, kind: str | None = "bash") -> str:
    """Dialect-aware canonical form (whitespace collapsed, quotes unified)."""
    kind = kind or "bash"
    if kind == "bash":
        body = strip_fence(y)
    elif kind == "tool_call":
        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", y, re.S)
        body = m.group(1) if m else y
        try:
            d = json.loads(body)
            body = json.dumps({"name": d.get("name"), "arguments": d.get("arguments")},
                              sort_keys=True, ensure_ascii=False)
        except ValueError:
            pass
    elif kind == "boxed":
        m = re.search(r"\\boxed\{(.*)\}", y, re.S)
        body = m.group(1) if m else y
    elif kind == "terminus_json":
        try:
            d = json.loads(y)
            body = json.dumps({"commands": d.get("commands"), "task_complete": d.get("task_complete")},
                              sort_keys=True, ensure_ascii=False)
        except ValueError:
            body = y
    else:
        body = y
    return re.sub(r"\s+", " ", body.replace('"', "'")).strip()


def action_tokens(y: str) -> set[str]:
    return set(_TOKEN_RE.findall(strip_fence(y)))


def jaccard(a: str, b: str) -> float:
    ta, tb = action_tokens(a), action_tokens(b)
    if not ta and not tb:
        return 1.0
    return len(ta & tb) / max(len(ta | tb), 1)


def agree(y: str, targets: list[str]) -> float:
    """Best token-Jaccard of y against any target (the prior probes' metric)."""
    return max((jaccard(y, t) for t in targets), default=0.0)


def exact(y: str, targets: list[str], kind: str | None = "bash") -> bool:
    ny = norm_action(y, kind)
    return any(ny == norm_action(t, kind) for t in targets)


# ------------------------------------------------------------------ scoring helpers (live rule pieces)
def lme(vals: list[float], tau: float) -> float:
    if not vals:
        return float("nan")
    if len(vals) == 1 or tau <= 0:
        return st.mean(vals)
    m = max(vals)
    return m + tau * math.log(st.mean(math.exp((v - m) / tau) for v in vals))


def clme(vals: list[float], tau: float) -> float:
    """Centered tempered LME = affine.score.centered_reason's rule."""
    return lme(vals, tau) - st.mean(vals) if vals else float("nan")


# ------------------------------------------------------------------ verdicts + corpus
def box_ssh(cmd: str, timeout: int = 600) -> str:
    out = subprocess.run(["ssh", "-i", str(BOX_KEY), "-o", "StrictHostKeyChecking=accept-new",
                          BOX, cmd], capture_output=True, text=True, timeout=timeout)
    if out.returncode != 0:
        raise RuntimeError(f"ssh failed: {out.stderr[:300]}")
    return out.stdout


def fetch_verdict(rec: str) -> Path:
    """scp one stored duel record (chal-XXXXX) into the local cache."""
    EVALS_CACHE.mkdir(parents=True, exist_ok=True)
    dst = EVALS_CACHE / f"{rec}.json.gz"
    if not dst.exists():
        subprocess.run(["scp", "-q", "-i", str(BOX_KEY), "-o", "StrictHostKeyChecking=accept-new",
                        f"{BOX}:{BOX_REPO}/affine/state/evals/{rec}.json.gz", str(dst)],
                       check=True, timeout=600)
    return dst


def load_verdict(rec: str) -> dict:
    return json.load(gzip.open(fetch_verdict(rec)))


def verdict_index() -> list[dict]:
    """The box's evals/index.jsonl (one line per stored duel)."""
    return [json.loads(l) for l in box_ssh(f"cat {BOX_REPO}/affine/state/evals/index.jsonl").splitlines() if l.strip()]


def corpus_for(manifest_sha: str, corpus_base_url: str | None = None):
    """CorpusSync on the record's pinned manifest (public data.affine.io)."""
    from affine.config import load_config
    from evalsrv.corpus import CorpusSync
    cfg = load_config()
    base = corpus_base_url or cfg.dataset.corpus_base_url
    scratch = Path(f"/tmp/frontier_arbiter_corpus/{manifest_sha[:12]}")
    corpus = CorpusSync(base, f"corpus/manifests/{manifest_sha}.json", scratch, lazy_chunks=True)
    if not corpus.ready:
        corpus.refresh()
    if not corpus.ready:
        raise RuntimeError(f"pinned manifest {manifest_sha[:12]} not syncable from {base}")
    return corpus


def materialize(rec_or_verdict: str | dict, turn_ids: list[str]) -> dict[str, dict]:
    """turn_id -> {prefix, action_kind, source, ...} for stored duel turns."""
    d = load_verdict(rec_or_verdict) if isinstance(rec_or_verdict, str) else rec_or_verdict
    sl = d["verdict"]["slice"]
    corpus = corpus_for(sl["manifest_sha256"], sl.get("corpus_base_url"))
    rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
    picked = [rows[t] for t in turn_ids if t in rows]
    turns = corpus.materialize_turns(picked)
    return {r["turn_id"]: t for r, t in zip(picked, turns)}


def not_forfeit(row: dict | None) -> bool:
    return bool(row and row.get("valid") and "pairs" in row)


def now() -> float:
    return time.time()
