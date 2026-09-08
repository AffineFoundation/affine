#!/usr/bin/env python3
"""Canary: cached-prefix teacher span vs today's echo=True path.

Hits one teacher replica over /v1/completions. Never sends
skip_reading_prefix_cache=false.

Uses the live injection boundaries (thought after `THOUGHT: `, action
after `\\n\\n`) so BPE-at-the-cut matches production.

Pass only if, for thought and action, at ~2k / ~8k / ~14k prefixes:
  generated text equals the scored remainder
  n_tokens matches the old echo span
  |Δ sum_lp| < 1e-5
  second new-path request shows local_cache_hit ≈ floor(n_prompt/784)*784

If string-split fails, retry with prompt = full_ids[:n_prompt] and
choice=[span]. If that still drifts, do not ship.

Observed 2026-08-30 (do not ship):
  - APC works: local_cache_hit lands on 784-token blocks.
  - structured_outputs.choice can emit the span bytes.
  - When the grammar completes, generated logprobs are often all 0.0
    (constrained / jump-forward path). That is not log P(span|prefix).
  - Thought boundary merges ` The` into the prefix token, so
    choice=[thought] retokenizes and n_tokens / sum_lp drift.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import uuid
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))

from evalsrv.chat import force_text, get_tokenizer, thought_text  # noqa: E402

ENV_FILE = HERE / ".swarm_env"
STATE_JSON = HERE / "state" / "state.json"
MODEL = "Qwen/Qwen3.8-27B"
LP_TOL = 1e-5
BLOCK = 784
SPECIALS = {"<|im_end|>", "<|endoftext|>"}

THOUGHT = (
    "The failing test expects `parse_header` to keep the original casing of "
    "the first token and to treat a missing colon as a soft error, not a "
    "hard reject. I will open the module, find the split, and check the "
    "early-return path before changing anything. The fixture in test_headers "
    "uses a tab after the name, so whitespace around the separator has to "
    "stay. I could either verify the current state or proceed with the change; "
    "the usual choice is to check first. After the read I will patch only the "
    "branch that drops the token and re-run the single test."
)
ACTION = (
    "```bash\n"
    "sed -n '1,80p' src/http/headers.py && "
    "pytest -q tests/test_headers.py::test_parse_header_keeps_case\n"
    "```"
)


def swarm_key() -> str:
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("SWARM_KEY="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"SWARM_KEY missing from {ENV_FILE}")


def backends() -> list[dict]:
    return list(json.loads(STATE_JSON.read_text()).get("backends") or [])


def scrape_cache(client: httpx.Client, origin: str, headers: dict) -> dict[str, float]:
    r = client.get(f"{origin}/metrics", headers=headers, timeout=15.0)
    r.raise_for_status()
    out: dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#") or "prompt_tokens_by_source_total" not in line:
            continue
        m = re.search(r'source="([^"]+)".*?\s([0-9.eE+-]+)\s*$', line)
        if m:
            out[m.group(1)] = float(m.group(2))
    return out


def cache_delta(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
    keys = set(before) | set(after)
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in sorted(keys)}


def pad_messages(n_tokens: int, salt: str, tok) -> list[dict]:
    # Overshoot then rely on chat template; ~4 chars/token is plenty.
    pad = (salt + " ") * (n_tokens * 2)
    return [
        {"role": "system", "content": "You are a coding agent in a SWE arena."},
        {"role": "user", "content": "Fix the header parser.\n\n" + pad},
    ]


def echo_span(tok, full: str, span_start: int, d: dict) -> tuple[int, list[int], float, int]:
    enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
    n_prompt = sum(1 for s, _ in enc["offset_mapping"] if s < span_start)
    lp = d["choices"][0]["logprobs"]["token_logprobs"]
    span = [x for x in lp[n_prompt:-1] if x is not None]
    return n_prompt, enc["input_ids"], float(sum(span)), len(span)


def gen_span(d: dict) -> tuple[str, float, int, int]:
    ch = d["choices"][0]
    toks = list((ch.get("logprobs") or {}).get("tokens") or [])
    lps = list((ch.get("logprobs") or {}).get("token_logprobs") or [])
    while toks and toks[-1] in SPECIALS:
        toks.pop()
        if lps:
            lps.pop()
    n = min(len(toks), len(lps))
    lps = lps[:n]
    zeros = sum(1 for x in lps if x == 0.0)
    return ch.get("text") or "", float(sum(x or 0.0 for x in lps)), n, zeros


def post(client: httpx.Client, base: str, headers: dict, payload: dict) -> dict:
    payload = {k: v for k, v in payload.items()
               if k != "skip_reading_prefix_cache"}
    t0 = time.perf_counter()
    r = client.post(f"{base}/completions", headers=headers, json=payload,
                    timeout=240.0)
    wall = time.perf_counter() - t0
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code} {r.text[:800]}")
    body = r.json()
    body["_wall"] = wall
    return body


def pick_backend(client: httpx.Client, headers: dict, prefer: str | None) -> tuple[str, str]:
    cands = []
    for b in backends():
        url = b["url"].rstrip("/")
        if url.endswith(":20015/v1") or url.endswith(":20015"):
            continue
        cands.append((url, b.get("pod", "?")))
    if prefer:
        cands = [c for c in cands if prefer in c[0]] + [
            c for c in cands if prefer not in c[0]
        ]
    last = ""
    for base, pod in cands:
        try:
            r = client.get(f"{base}/models", headers=headers, timeout=8.0)
            if r.status_code == 200:
                origin = base[:-3] if base.endswith("/v1") else base
                print(f"backend {base} pod={pod} ok", flush=True)
                return base, origin
            last = f"{base} status={r.status_code}"
        except httpx.HTTPError as e:
            last = f"{base} {type(e).__name__}:{e}"
    raise SystemExit(f"no healthy backend (last: {last})")


def run_case(client, tok, base, origin, headers, full: str, span: str,
             label: str) -> dict:
    span_start = len(full) - len(span)
    prefix = full[:span_start]
    print(f"\n--- {label} prefix_tail={prefix[-24:]!r} span_chars={len(span)} ---",
          flush=True)

    before = scrape_cache(client, origin, headers)
    warm = post(client, base, headers, {
        "model": MODEL, "prompt": prefix, "max_tokens": 1,
        "temperature": 0, "add_special_tokens": False,
    })
    after = scrape_cache(client, origin, headers)
    print(f"warm wall={warm['_wall']:.2f}s cacheΔ={cache_delta(before, after)}",
          flush=True)

    before = scrape_cache(client, origin, headers)
    old = post(client, base, headers, {
        "model": MODEL, "prompt": full, "max_tokens": 1,
        "temperature": 0, "echo": True, "logprobs": 0,
        "add_special_tokens": False,
    })
    after = scrape_cache(client, origin, headers)
    n_prompt, full_ids, old_sum, old_n = echo_span(tok, full, span_start, old)
    print(f"old  wall={old['_wall']:.2f}s sum={old_sum:.8f} n={old_n} "
          f"cacheΔ={cache_delta(before, after)}", flush=True)

    max_new = max(old_n + 16, len(tok.encode(span, add_special_tokens=False)) + 16)

    def new_payload(prompt, choice: str) -> dict:
        return {
            "model": MODEL,
            "prompt": prompt,
            "max_tokens": max_new,
            "temperature": 0,
            "echo": False,
            "logprobs": 1,
            "add_special_tokens": False,
            "structured_outputs": {"choice": [choice]},
        }

    def once(prompt, choice: str, tag: str) -> dict:
        before = scrape_cache(client, origin, headers)
        d = post(client, base, headers, new_payload(prompt, choice))
        after = scrape_cache(client, origin, headers)
        text, s, n, zeros = gen_span(d)
        hit = cache_delta(before, after)
        print(f"{tag} wall={d['_wall']:.2f}s sum={s:.8f} n={n} zeros={zeros} "
              f"text_eq={text == choice} cacheΔ={hit}", flush=True)
        return {
            "sum": s, "n": n, "zeros": zeros, "text_eq": text == choice,
            "wall": d["_wall"], "cache": hit,
            "ok": text == choice and n == old_n and abs(s - old_sum) < LP_TOL,
        }

    string = once(prefix, span, "new1")
    string2 = once(prefix, span, "new2")
    token = None
    if not string["ok"]:
        print("string-split drifted — retrying token-id prompt", flush=True)
        token = once(full_ids[:n_prompt], span, "tok ")

    exp_hit = (n_prompt // BLOCK) * BLOCK
    hit2 = string2["cache"].get("local_cache_hit", 0.0)
    cache_ok = exp_hit == 0 or hit2 >= 0.8 * exp_hit
    row = {
        "label": label,
        "n_prompt": n_prompt,
        "old_sum": old_sum,
        "old_n": old_n,
        "string": string,
        "string2": string2,
        "token": token,
        "expected_hit": exp_hit,
        "cache_ok": cache_ok,
        "pass": (string["ok"] or (token or {}).get("ok")) and cache_ok,
    }
    print(f"result pass={row['pass']} cache_ok={cache_ok} "
          f"Δsum={string['sum'] - old_sum:.3e} hit2={hit2} expected={exp_hit}",
          flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefer", default="")
    ap.add_argument("--lengths", default="2000,8000,14000")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {swarm_key()}"}
    salt = f"cached-span-canary-{uuid.uuid4().hex[:12]}"
    print(f"salt={salt}", flush=True)
    tok = get_tokenizer(MODEL, None)
    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    rows = []
    with httpx.Client() as client:
        base, origin = pick_backend(client, headers, args.prefer or None)
        for n in lengths:
            msgs = pad_messages(n, salt, tok)
            thought_full = thought_text(MODEL, None, msgs, THOUGHT)
            action_full = force_text(MODEL, None, msgs, THOUGHT, ACTION)
            rows.append(run_case(client, tok, base, origin, headers,
                                 thought_full, THOUGHT, f"thought~{n}"))
            rows.append(run_case(client, tok, base, origin, headers,
                                 action_full, ACTION, f"action~{n}"))
    print("\n=== SUMMARY ===")
    print(json.dumps(rows, indent=2))
    if all(r["pass"] for r in rows):
        print("PASS — ship cached_span")
        return 0
    print("FAIL — do not ship")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
