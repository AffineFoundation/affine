"""Stage 2: one LLM call per post -> a task specification as strict JSON.

Reads `<out>/e<epoch>/posts.jsonl.gz`, writes `<out>/e<epoch>/specs.jsonl`
(one row per post: the post, the spec, or a `reject` reason). Every model
reply is cached in `<out>/e<epoch>/cache/synth.jsonl` keyed by
(PROMPT_VERSION, model, post hash), so a re-run is free.

The endpoint is OpenAI-compatible (`/chat/completions`). Default = the duel
teacher on Engy (`qwen3.8-27b`): cheapest route we run, and tasks the teacher
can phrase are tasks near its own competence, which is the band the duel
needs. `--model / --base-url / --key-env` switch to a stronger model.

Spec schema (all keys required):
  instruction  what the agent must achieve, verifiable end state, no solution
  dockerfile   FROM one of BASE_IMAGES; fixture files via COPY _fixtures/...
  fixtures     {relative path under _fixtures/: file content}
  solve_sh     reference solution (bash, runs as root in the container)
  test_py      pytest module checking the container's FINAL STATE only
  metadata     {domain, language, difficulty, commands}

    ENGY=... python synth.py --epoch 63 --out ~/terminal_gen/out --concurrency 8 --max-usd 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

import httpx

from common import out_dir, read_jsonl, sha256_text, write_jsonl

log = logging.getLogger("terminal_gen.synth")

PROMPT_VERSION = "tg-v1"
DEFAULT_MODEL = "qwen3.8-27b"
DEFAULT_BASE_URL = "https://api.engy.ai/v1"
DEFAULT_KEY_ENV = "ENGY"
# $ per 1M tokens, Engy qwen3.8-27b (rollouts/policies.toml [pricing.engy]).
DEFAULT_PRICE_IN = 0.045
DEFAULT_PRICE_OUT = 0.32

BASE_IMAGES = ("ubuntu:22.04", "ubuntu:24.04", "debian:bookworm-slim", "python:3.12-slim")
SPEC_KEYS = ("instruction", "dockerfile", "fixtures", "solve_sh", "test_py", "metadata")
DIFFICULTIES = ("easy", "medium", "hard")
# The pod runs task containers without internet (task.toml allow_internet = false):
# solutions and tests must not fetch anything. Dockerfiles may (build time).
_NET_RE = re.compile(r"\b(curl|wget|apt-get|apt|pip3?\s+install|npm\s+install|git\s+clone)\b")
_FROM_RE = re.compile(r"(?im)^\s*FROM\s+(\S+)")
_JSON_RE = re.compile(r"\{.*\}", re.S)

SYSTEM_PROMPT = """You turn a Stack Exchange question and its accepted answer into a self-contained TERMINAL TASK for an autonomous agent working in a Linux container as root. The task must be reproducible and machine-verifiable. Reply with ONE JSON object and nothing else."""

USER_TEMPLATE = """## Source post ({site}, score {score})
### Title
{title}

### Question
{question}

### Accepted answer
{answer}

## What to produce
Design a task in the spirit of the post: same skill, but a concrete scenario with files, services or state that you create in the image, so that the agent has to do the work, not recall the post. Requirements:

1. `instruction`: 80-400 words, plain text/markdown. Describe the situation, the exact goal and the exact end state that will be checked (paths, names, formats). Do NOT include the solution or hints about commands. Do not mention Stack Exchange.
2. `dockerfile`: starts with `FROM <image>` where <image> is one of: {images}. Install packages with apt-get (network is available at build time only). Create the starting state (files, users, services, broken configs) with RUN lines and `COPY _fixtures/<path> <dest>` for any file you list in `fixtures`. Work dir `/app` unless the scenario needs another. Do not run long-lived services in the Dockerfile.
3. `fixtures`: object mapping a relative path (no leading slash, no `..`) to the full text content of a fixture file the Dockerfile copies. Use `{{}}` if none. Keep each file under 20 KB and deterministic.
4. `solve_sh`: a bash script that, run inside a fresh container built from the Dockerfile, reaches the end state. It must not download anything.
5. `test_py`: a pytest module (functions `test_*`) that inspects ONLY the container's final state (files, permissions, process output of local commands, database contents) and passes iff the goal is met. It must fail on the untouched image. No network, no randomness, no timing dependence, no imports beyond the standard library and pytest. Tests run from `/tests` as root.
6. `metadata`: {{"domain": short lower-case category (e.g. "text-processing", "permissions", "networking", "shell-scripting", "databases", "packaging", "filesystems", "processes", "git"), "language": "shell" or the main language involved, "difficulty": one of {difficulties}, "commands": list of 2-6 key commands the reference solution relies on}}.

Return exactly: {{"instruction": ..., "dockerfile": ..., "fixtures": {{...}}, "solve_sh": ..., "test_py": ..., "metadata": {{...}}}}"""


def build_messages(post: dict) -> list[dict]:
    user = USER_TEMPLATE.format(
        site=post["site"], score=post["score"], title=post["title"],
        question=post["question"][:5000], answer=post["answer"][:5000],
        images=", ".join(BASE_IMAGES), difficulties=", ".join(DIFFICULTIES))
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]


def parse_spec(text: str) -> tuple[dict | None, str]:
    """(spec, '') or (None, reject reason)."""
    m = _JSON_RE.search(text or "")
    if not m:
        return None, "no_json"
    try:
        spec = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, "bad_json"
    if not isinstance(spec, dict) or any(k not in spec for k in SPEC_KEYS):
        return None, "missing_keys"
    for k in ("instruction", "dockerfile", "solve_sh", "test_py"):
        if not isinstance(spec[k], str) or not spec[k].strip():
            return None, f"empty_{k}"
    fm = _FROM_RE.search(spec["dockerfile"])
    if not fm or fm.group(1) not in BASE_IMAGES:
        return None, "base_image_not_allowed"
    if not isinstance(spec["fixtures"], dict):
        return None, "fixtures_not_object"
    for path, content in spec["fixtures"].items():
        if not isinstance(path, str) or not isinstance(content, str):
            return None, "fixture_type"
        if path.startswith("/") or ".." in Path(path).parts or len(content) > 20_000:
            return None, "fixture_path"
    if _NET_RE.search(spec["solve_sh"]) or _NET_RE.search(spec["test_py"]):
        return None, "network_in_solution_or_test"
    if "def test_" not in spec["test_py"]:
        return None, "no_pytest_functions"
    if len(spec["instruction"].split()) < 40:
        return None, "instruction_too_short"
    meta = spec["metadata"]
    if not isinstance(meta, dict):
        return None, "metadata_type"
    meta["domain"] = str(meta.get("domain") or "shell").lower()[:40]
    meta["language"] = str(meta.get("language") or "shell").lower()[:20]
    meta["difficulty"] = meta.get("difficulty") if meta.get("difficulty") in DIFFICULTIES else "medium"
    cmds = meta.get("commands") or []
    meta["commands"] = [str(c)[:60] for c in cmds][:8] if isinstance(cmds, list) else []
    return spec, ""


class Cache:
    def __init__(self, path: Path):
        self.path = path
        self.rows: dict[str, dict] = {}
        for r in read_jsonl(path):
            self.rows[r["key"]] = r
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def get(self, key: str) -> dict | None:
        return self.rows.get(key)

    def put(self, key: str, row: dict) -> None:
        row = {"key": key, **row}
        self.rows[key] = row
        self._fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._fh.flush()


def post_key(model: str, post: dict) -> str:
    return sha256_text(f"{PROMPT_VERSION}|{model}|{post['site']}|{post['qid']}|{sha256_text(post['answer'])}")


async def call_model(client: httpx.AsyncClient, base_url: str, key: str, model: str,
                     messages: list[dict], max_tokens: int, temperature: float,
                     reasoning_effort: str = "low") -> tuple[str, dict]:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "response_format": {"type": "json_object"}}
    # Qwen3.8 on Engy thinks before it answers; with the old max_tokens 6000
    # every reply of the first run (2,953 / 3,000) was 6,000 reasoning tokens
    # and an EMPTY visible text -> "no_json". Cap the thinking (Engy's
    # reasoning_effort tiers: none/minimal = off) and leave room for the spec.
    if reasoning_effort and reasoning_effort != "default":
        body["reasoning_effort"] = reasoning_effort
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    for attempt in range(4):
        try:
            resp = await client.post(f"{base_url.rstrip('/')}/chat/completions", json=body,
                                     headers=headers, timeout=600)
            if resp.status_code == 400 and "response_format" in body:
                body.pop("response_format")
                continue
            resp.raise_for_status()
            data = resp.json()
            text = data["choices"][0]["message"].get("content") or ""
            return text, data.get("usage") or {}
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            if attempt == 3:
                raise
            await asyncio.sleep(2 ** attempt * 3)
            log.warning("retry %d after %s", attempt + 1, exc)
    raise RuntimeError("unreachable")


async def run(args: argparse.Namespace) -> None:
    out = out_dir(args.out, args.epoch)
    posts = list(read_jsonl(out / "posts.jsonl.gz"))
    if not posts:
        raise SystemExit(f"no posts at {out / 'posts.jsonl.gz'} (run posts.py first)")
    key = os.environ.get(args.key_env, "")
    cache = Cache(out / "cache" / "synth.jsonl")
    spent = {"usd": 0.0, "in": 0, "out": 0, "calls": 0}
    sem = asyncio.Semaphore(args.concurrency)
    results: dict[str, dict] = {}

    async def one(client: httpx.AsyncClient, post: dict) -> None:
        k = post_key(args.model, post)
        cached = cache.get(k)
        if cached is None:
            if not key:
                raise SystemExit(f"{args.key_env} is not set and post {post['qid']} is not cached")
            async with sem:
                if spent["usd"] >= args.max_usd:
                    results[k] = {"post": post, "reject": "budget_exhausted"}
                    return
                t0 = time.time()
                text, usage = await call_model(client, args.base_url, key, args.model,
                                               build_messages(post), args.max_tokens, args.temperature,
                                               args.reasoning_effort)
                n_in = int(usage.get("prompt_tokens") or 0)
                n_out = int(usage.get("completion_tokens") or 0)
                cost = n_in * args.price_in / 1e6 + n_out * args.price_out / 1e6
                spent["usd"] += cost
                spent["in"] += n_in
                spent["out"] += n_out
                spent["calls"] += 1
                cached = {"model": args.model, "prompt_version": PROMPT_VERSION, "text": text,
                          "usage": usage, "cost_usd": cost, "seconds": round(time.time() - t0, 1)}
                cache.put(k, cached)
                if spent["calls"] % 10 == 0:
                    log.info("%d calls, $%.2f", spent["calls"], spent["usd"])
        spec, reason = parse_spec(cached["text"])
        row = {"post": post, "model": cached["model"], "prompt_version": cached["prompt_version"]}
        if spec is None:
            row["reject"] = reason
        else:
            row["spec"] = spec
        results[k] = row

    async with httpx.AsyncClient() as client:
        await asyncio.gather(*(one(client, p) for p in posts))

    rows = [results[post_key(args.model, p)] for p in posts]
    n = write_jsonl(out / "specs.jsonl", rows)
    rejects: dict[str, int] = {}
    for r in rows:
        if "reject" in r:
            rejects[r["reject"]] = rejects.get(r["reject"], 0) + 1
    summary = {"posts": n, "specs": sum("spec" in r for r in rows), "rejects": rejects,
               "model": args.model, "prompt_version": PROMPT_VERSION, **spent}
    (out / "synth_summary.json").write_text(json.dumps(summary, indent=1))
    log.info("synth: %s", json.dumps(summary))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out", default="~/terminal_gen/out")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--key-env", default=DEFAULT_KEY_ENV)
    ap.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN, help="$ per 1M prompt tokens")
    ap.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT, help="$ per 1M completion tokens")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--reasoning-effort", default="low",
                    help="Engy/OpenAI reasoning_effort tier (none|minimal|low|medium|high; 'default' = do not send)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-usd", type=float, default=20.0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
