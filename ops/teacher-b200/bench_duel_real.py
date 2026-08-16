#!/usr/bin/env python3
"""Duel-faithful teacher load: per turn 1 sample + N echoes sharing the prefix.

reason_only teacher work per scored pair:
  ref sample (z_C, y_C)            -> 1 sample from prefix x   (cached per turn)
  lpC(y_C | z_A), lpC(y_C | <empty>) -> echoes over x + suffix
  B: lpC(y_A | z_A), lpC(y_A | <empty>) -> echoes over x + suffix
All four echoes share the turn prefix x, so with sticky routing + automatic
prefix caching the engine prefills x once. bench_mixed.py's unique prompts
measure the no-sharing worst case; this measures the real duel shape.

Turns are sticky-routed by hash to one base (mirrors ModelPool._pick).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import zlib
from pathlib import Path

import httpx

try:
    from transformers import AutoTokenizer
except ImportError:  # pods without transformers can still run string-echo mode
    AutoTokenizer = None

ROOT = Path("/home/const/subnet120")


def make_prefix(n_tokens: int, salt: int) -> str:
    unit = "The quick brown fox jumps over the lazy dog. "
    body = (unit * ((n_tokens * 4) // len(unit) + 1))[: max(0, n_tokens * 4 - 48)]
    return f"TURN={salt};" + body


def make_suffix(n_tokens: int, salt: int) -> str:
    unit = "meanwhile the reasoning continues step by step. "
    body = (unit * ((n_tokens * 4) // len(unit) + 1))[: max(0, n_tokens * 4 - 32)]
    return f"\nTHOUGHT={salt};" + body


RETRIES = 0


async def post(client: httpx.AsyncClient, base: str, model: str,
               payload: dict, max_retries: int = 0) -> float:
    global RETRIES
    t0 = time.perf_counter()
    for attempt in range(max_retries + 1):
        try:
            r = await client.post(f"{base}/completions",
                                  json={"model": model, **payload})
            r.raise_for_status()
            return time.perf_counter() - t0
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            retryable = (isinstance(e, httpx.TransportError)
                         or e.response.status_code in (429, 500, 502, 503, 504))
            if not retryable or attempt == max_retries:
                if isinstance(e, httpx.HTTPStatusError):
                    print(f"FAIL {e.response.status_code}: "
                          f"{e.response.text[:300]}", flush=True)
                raise
            RETRIES += 1
            await asyncio.sleep(2.0 * (attempt + 1))
    raise RuntimeError("unreachable")


async def amain(args: argparse.Namespace) -> int:
    bases = [b.rstrip("/") for b in args.base_url]
    timeout = httpx.Timeout(args.timeout_s)
    limits = httpx.Limits(max_connections=args.concurrency + 16,
                          max_keepalive_connections=args.concurrency + 16)
    headers = ({"Authorization": f"Bearer {args.api_key}"}
               if args.api_key else {})
    async with httpx.AsyncClient(timeout=timeout, limits=limits,
                                 headers=headers) as client:
        model = args.model
        for b in bases:
            r = await client.get(f"{b}/models")
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", [])]
            if model not in ids and ids:
                model = ids[0]
            print(f"ready {b} model={model}", flush=True)

        await post(client, bases[0], model, {
            "prompt": make_prefix(2048, -1), "max_tokens": 16,
            "temperature": 0.8}, args.retries)

        tokenizer = None
        if args.span_tokenizer:
            if AutoTokenizer is None:
                raise SystemExit("--span-tokenizer needs transformers installed")
            tokenizer = AutoTokenizer.from_pretrained(args.span_tokenizer)

        def echo_payload(x: str, suffix: str) -> dict:
            # Span mode (engy): token-id prompt + logprob_start_len scores only
            # the suffix, mirroring what the duel actually consumes.
            if tokenizer is not None:
                x_ids = tokenizer(x, add_special_tokens=False)["input_ids"]
                s_ids = tokenizer(suffix, add_special_tokens=False)["input_ids"]
                return {"prompt": x_ids + s_ids,
                        "logprob_start_len": len(x_ids),
                        "max_tokens": 1, "temperature": 0,
                        "echo": True, "logprobs": args.logprobs}
            return {"prompt": x + suffix, "max_tokens": 1, "temperature": 0,
                    "echo": True, "logprobs": args.logprobs}

        sem = asyncio.Semaphore(args.concurrency)
        sample_t: list[float] = []
        echo_t: list[float] = []

        async def turn(idx: int) -> None:
            async with sem:
                base = bases[zlib.adler32(f"turn-{idx}".encode()) % len(bases)]
                x = make_prefix(args.prefix_tokens, idx)
                sample_t.append(await post(client, base, model, {
                    "prompt": x, "max_tokens": args.sample_max_tokens,
                    "temperature": 0.8}, args.retries))
                echoes = [
                    post(client, base, model, echo_payload(
                        x, make_suffix(args.suffix_tokens, idx * 10 + k)),
                        args.retries)
                    for k in range(args.echoes_per_turn)
                ]
                if args.serial_echoes:
                    for e in echoes:
                        echo_t.append(await e)
                else:
                    echo_t.extend(await asyncio.gather(*echoes))

        t0 = time.perf_counter()
        await asyncio.gather(*[turn(i) for i in range(args.n_turns)])
        wall = time.perf_counter() - t0

        def pack(times: list[float]) -> dict:
            return {
                "n": len(times),
                "rps": round(len(times) / wall, 3) if wall else 0,
                "p50_s": round(statistics.median(times), 3),
                "p95_s": round(sorted(times)[max(0, int(0.95 * len(times)) - 1)], 3),
            }

        out = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_urls": bases,
            "model": model,
            "concurrency": args.concurrency,
            "n_turns": args.n_turns,
            "prefix_tokens": args.prefix_tokens,
            "suffix_tokens": args.suffix_tokens,
            "echoes_per_turn": args.echoes_per_turn,
            "serial_echoes": bool(args.serial_echoes),
            "span_tokenizer": args.span_tokenizer,
            "retries": RETRIES,
            "wall_s": round(wall, 3),
            "turns_per_s": round(args.n_turns / wall, 3) if wall else 0,
            "proj_4000_min": (round(4000 / (args.n_turns / wall) / 60, 1)
                              if wall else None),
            "sample": pack(sample_t),
            "echo": pack(echo_t),
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({k: out[k] for k in (
            "turns_per_s", "proj_4000_min", "wall_s")}
            | {"echo_p50": out["echo"]["p50_s"],
               "echo_p95": out["echo"]["p95_s"]}), flush=True)
        print("WROTE", path, flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", action="append", required=True)
    p.add_argument("--model", default="zai-org/GLM-4.5-Air-FP8")
    p.add_argument("--concurrency", type=int, default=24,
                   help="concurrent turns (each turn = 1 sample + N echoes)")
    p.add_argument("--n-turns", type=int, default=96)
    p.add_argument("--prefix-tokens", type=int, default=8192)
    p.add_argument("--suffix-tokens", type=int, default=1024)
    p.add_argument("--echoes-per-turn", type=int, default=4)
    p.add_argument("--sample-max-tokens", type=int, default=256)
    p.add_argument("--serial-echoes", action="store_true",
                   help="issue echoes one-by-one (guaranteed cache reuse) "
                        "instead of concurrently")
    p.add_argument("--logprobs", type=int, default=0,
                   help="logprobs value for echo requests (engy needs >=1)")
    p.add_argument("--retries", type=int, default=0,
                   help="retries per request on 429/5xx/transport errors")
    p.add_argument("--span-tokenizer", default="",
                   help="HF tokenizer repo; when set, echoes send token ids "
                        "+ logprob_start_len so only the suffix is scored "
                        "(required by engy for prompts > 1024 tokens)")
    p.add_argument("--timeout-s", type=float, default=900.0)
    p.add_argument("--api-key", default="",
                   help="optional bearer token (swarm pods enforce one)")
    p.add_argument("--out",
                   default=str(ROOT / "ops/teacher-b200/artifacts/"
                                      "bench_duel_real_latest.json"))
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
