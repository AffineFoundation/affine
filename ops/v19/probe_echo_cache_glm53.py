"""Echo-cache plugin port probe for a GLM-5.3-Flash replica (run ON a swarm pod).

The plugin (ops/teacher-swarm/echo_cache_plugin) patches three vLLM 0.28
internals so echo requests reuse the prefix KV and recompute only the scored
tail. GLM-5.3-Flash needs vLLM >= 0.29 (glm53 build, FlashInfer >= 0.6.17)
and mixes KDA linear-attention layers with sparse MLA, so two things must be
re-verified before the swarm serves it:

  1. register() still finds every hooked attribute on the new vLLM (it logs
     "NOT patching" and falls back to uncached echoes otherwise — slow, not
     wrong);
  2. cached-tail echoes are numerically identical to uncached ones on this
     architecture (hybrid models align prefix-cache hits to larger block
     boundaries; the plugin caps the hit at num_tokens-1-T and the client
     falls back when the +1.0 sentinel reaches the span — both paths must
     agree).

Usage (pod, venv with the plugin installed, replica on :8000):
  python probe_echo_cache_glm53.py --base http://127.0.0.1:8000/v1 \
      --model zai-org/GLM-5.3-Flash --n 24 --prefix-tokens 8000 20000 60000
Pass = max |Δ sum_lp| < 1e-3 over every probe AND median cached latency
< 0.5 × uncached at 20k+ prefixes. Writes probe_echo_cache_glm53.json.
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import time

import httpx
from transformers import AutoTokenizer

XARG = "affine_echo_tail"


def build(tok, n_prefix: int, salt: int) -> tuple[str, int]:
    unit = f"[{salt}] The quick brown fox jumps over the lazy dog; "
    body = unit * (n_prefix * 5 // len(unit) + 1)
    ids = tok(body, add_special_tokens=False)["input_ids"][:n_prefix]
    prefix = tok.decode(ids)
    thought = ("Let me think about the repository layout and the failing test "
               "before I run anything. " * 6).strip()
    action = "\n\n```bash\nls -la && pytest -x tests/test_core.py\n```"
    full = prefix + "<|assistant|><think>" + thought + "\n</think>\n\n" + action
    return full, len(prefix) + len("<|assistant|><think>")


def echo(client: httpx.Client, base: str, model: str, full: str, span_start: int,
         tok, cached: bool) -> tuple[float, int, float, bool]:
    enc = tok([full], add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = enc["input_ids"][0], enc["offset_mapping"][0]
    n_prompt = sum(1 for s, _ in offsets if s < span_start)
    tail = len(ids) - n_prompt + 1 + 8
    payload = {"model": model, "prompt": full, "max_tokens": 1, "temperature": 0,
               "echo": True, "logprobs": 0, "add_special_tokens": False}
    if cached:
        payload["vllm_xargs"] = {XARG: tail}
    t0 = time.perf_counter()
    r = client.post(f"{base}/completions", json=payload, timeout=600)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    lp = r.json()["choices"][0]["logprobs"]["token_logprobs"]
    span = lp[n_prompt:-1]
    touched = any(x is None or x > 0 for x in span)
    vals = [x for x in span if x is not None and x <= 0]
    return sum(vals), len(vals), dt, touched


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--prefix-tokens", type=int, nargs="+", default=[8000, 20000, 60000])
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    out = []
    with httpx.Client() as client:
        for n_prefix in args.prefix_tokens:
            for i in range(args.n):
                full, span_start = build(tok, n_prefix, i)
                s_u, n_u, t_u, _ = echo(client, args.base, args.model, full, span_start, tok, cached=False)
                # Warm the prefix, then the cached-tail request.
                echo(client, args.base, args.model, full, span_start, tok, cached=True)
                s_c, n_c, t_c, touched = echo(client, args.base, args.model, full, span_start, tok, cached=True)
                out.append({"prefix_tokens": n_prefix, "i": i, "sum_uncached": s_u, "sum_cached": s_c,
                            "n_uncached": n_u, "n_cached": n_c, "delta": abs(s_u - s_c),
                            "t_uncached": t_u, "t_cached": t_c, "span_touched_cache": touched})
                print(f"{n_prefix:>6} #{i:02d} Δ={abs(s_u - s_c):.2e} n {n_u}/{n_c} "
                      f"t {t_u:.2f}s -> {t_c:.2f}s touched={touched}", flush=True)
    by = {}
    for r in out:
        by.setdefault(r["prefix_tokens"], []).append(r)
    summary = {k: {"max_delta": max(r["delta"] for r in v),
                   "n_mismatch": sum(1 for r in v if r["n_uncached"] != r["n_cached"]),
                   "n_touched": sum(1 for r in v if r["span_touched_cache"]),
                   "speedup_median": st.median(r["t_uncached"] / max(r["t_cached"], 1e-6) for r in v)}
               for k, v in by.items()}
    ok = all(s["max_delta"] < 1e-3 and s["n_mismatch"] == 0 for s in summary.values()) and all(
        s["speedup_median"] > 2.0 for k, s in summary.items() if k >= 20000)
    json.dump({"model": args.model, "ok": ok, "summary": summary, "rows": out},
              open("probe_echo_cache_glm53.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print("PASS" if ok else "FAIL — do not enable the plugin on the GLM swarm")


if __name__ == "__main__":
    main()
