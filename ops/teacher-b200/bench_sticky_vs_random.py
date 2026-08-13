#!/usr/bin/env python3
"""Compare sticky vs random teacher replica routing (prefix-cache affinity).

Mimics reason_only teacher work per turn on co-located :8000/:8003:
  1) sample (short) from shared prefix x
  2) echo score with planted thoughts (same x stem)
  3) echo score with different thoughts (same x stem)  # king + chall

Run on the eval pod (or any host that can reach both bases).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
import zlib

import httpx


def make_prefix(n_tokens: int, salt: int) -> str:
    unit = "The quick brown fox jumps over the lazy dog. "
    body = (unit * ((n_tokens * 4) // len(unit) + 1))[: n_tokens * 4 - 48]
    return f"TURN={salt};" + body


async def post(client: httpx.AsyncClient, base: str, model: str, payload: dict) -> float:
    t0 = time.perf_counter()
    r = await client.post(f"{base}/completions", json={"model": model, **payload})
    r.raise_for_status()
    return time.perf_counter() - t0


async def turn(client, bases, model, salt, mode: str, conc_sem, sample_tok, echo_tok):
    async with conc_sem:
        x = make_prefix(echo_tok, salt)
        if mode == "sticky":
            idx = zlib.adler32(f"turn-{salt}".encode()) % len(bases)
            bases_for_turn = [bases[idx]] * 3
        else:
            # Old-ish bias: prefer first idle → simulated as always pick bases[salt%2]
            # for call 0, then alternate — actually pure random per call:
            bases_for_turn = [bases[(salt + k) % len(bases)] for k in range(3)]
            # stronger anti-sticky: force different replica for echoes when possible
            if len(bases) > 1:
                bases_for_turn = [bases[0], bases[1], bases[0]]

        # 1) sample
        await post(client, bases_for_turn[0], model, {
            "prompt": x, "max_tokens": sample_tok, "temperature": 0.8,
        })
        # 2-3) two echoes sharing prefix x (different suffixes)
        for i, b in enumerate(bases_for_turn[1:]):
            await post(client, b, model, {
                "prompt": x + f"\nTHOUGHTS_{i}_" + ("z" * 200),
                "max_tokens": 1, "temperature": 0, "echo": True, "logprobs": 0,
            })


async def run_mode(client, bases, model, mode, n_turns, concurrency, sample_tok, echo_tok):
    sem = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    await asyncio.gather(*[
        turn(client, bases, model, i, mode, sem, sample_tok, echo_tok)
        for i in range(n_turns)
    ])
    wall = time.perf_counter() - t0
    return {
        "mode": mode,
        "n_turns": n_turns,
        "wall_s": round(wall, 3),
        "turns_per_s": round(n_turns / wall, 3),
        "proj_2080_min": round(2080 / (n_turns / wall) / 60, 1),
    }


async def amain(args):
    bases = [b.rstrip("/") for b in args.base_url]
    timeout = httpx.Timeout(300.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        model = args.model
        for b in bases:
            r = await client.get(f"{b}/models")
            r.raise_for_status()
            ids = [m["id"] for m in r.json()["data"]]
            if model not in ids:
                model = ids[0]
            print("ready", b, model)

        out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "bases": bases, "results": []}
        for mode in ("random_split", "sticky"):
            # remap
            m = "sticky" if mode == "sticky" else "random"
            print("running", mode)
            res = await run_mode(
                client, bases, model, m, args.n_turns, args.concurrency,
                args.sample_tokens, args.echo_tokens,
            )
            res["mode"] = mode
            print(json.dumps(res))
            out["results"].append(res)
        path = args.out
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
            f.write("\n")
        print("WROTE", path)
        if len(out["results"]) == 2:
            a, b = out["results"]
            speedup = a["wall_s"] / b["wall_s"] if b["wall_s"] else None
            print(json.dumps({"sticky_speedup_vs_split": round(speedup, 3) if speedup else None}))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", action="append", required=True)
    p.add_argument("--model", default="zai-org/GLM-4.5-Air-FP8")
    p.add_argument("--n-turns", type=int, default=24)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--sample-tokens", type=int, default=64)
    p.add_argument("--echo-tokens", type=int, default=4096)
    p.add_argument("--out", default="/tmp/bench_sticky_vs_random.json")
    asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    main()
