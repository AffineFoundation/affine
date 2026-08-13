#!/usr/bin/env python3
"""Duel-shaped mixed load: concurrent sample + echo against one or more teacher bases.

Mirrors reason_only teacher work per turn ≈ 1 ref-sample + 1 lpC echo.
Optionally round-robins across multiple base URLs (dual replica).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path

import httpx

ROOT = Path("/home/const/subnet120")


def make_prompt(n_tokens_approx: int, salt: int = 0) -> str:
    """Unique-enough prompt so prefix cache cannot collapse the wave.

    Real duel turns have distinct prefixes; reusing one string makes vLLM
    prefix caching invent fake throughput.
    """
    unit = "The quick brown fox jumps over the lazy dog. "
    reps = max(1, (n_tokens_approx * 4) // len(unit))
    body = (unit * reps)[: max(0, n_tokens_approx * 4 - 64)]
    return f"TURN_SALT={salt};" + body


async def one_sample(client: httpx.AsyncClient, base: str, model: str,
                     prompt: str, max_tokens: int) -> float:
    t0 = time.perf_counter()
    r = await client.post(
        f"{base}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.8,
        },
    )
    r.raise_for_status()
    return time.perf_counter() - t0


async def one_echo(client: httpx.AsyncClient, base: str, model: str,
                   prompt: str) -> float:
    t0 = time.perf_counter()
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 1,
        "temperature": 0,
        "echo": True,
        "logprobs": 1,
        "prompt_logprobs": 1,
    }
    r = await client.post(f"{base}/completions", json=payload)
    if r.status_code >= 400:
        payload.pop("prompt_logprobs", None)
        r = await client.post(f"{base}/completions", json=payload)
    r.raise_for_status()
    return time.perf_counter() - t0


async def amain(args: argparse.Namespace) -> int:
    bases = [b.rstrip("/") for b in args.base_url]
    timeout = httpx.Timeout(args.timeout_s)
    limits = httpx.Limits(
        max_connections=args.concurrency + 16,
        max_keepalive_connections=args.concurrency + 16,
    )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        model = args.model
        for b in bases:
            r = await client.get(f"{b}/models")
            r.raise_for_status()
            ids = [m["id"] for m in r.json().get("data", [])]
            if model not in ids and ids:
                model = ids[0]
            print(f"ready {b} model={model}")

        # Warmup (unique salts)
        await one_sample(
            client, bases[0], model, make_prompt(args.sample_prompt_tokens, -1), 32
        )
        await one_echo(
            client, bases[0], model, make_prompt(min(4000, args.echo_prompt_tokens), -2)
        )

        sem = asyncio.Semaphore(args.concurrency)
        sample_t: list[float] = []
        echo_t: list[float] = []
        rr = 0
        turn_i = 0

        async def turn():
            nonlocal rr, turn_i
            async with sem:
                idx = turn_i
                turn_i += 1
                base = bases[rr % len(bases)]
                rr += 1
                sample_prompt = make_prompt(args.sample_prompt_tokens, idx * 2)
                echo_prompt = make_prompt(args.echo_prompt_tokens, idx * 2 + 1)
                sample_t.append(
                    await one_sample(
                        client, base, model, sample_prompt, args.sample_max_tokens
                    )
                )
                echo_t.append(await one_echo(client, base, model, echo_prompt))

        t0 = time.perf_counter()
        await asyncio.gather(*[turn() for _ in range(args.n_turns)])
        wall = time.perf_counter() - t0

        def pack(name: str, times: list[float]) -> dict:
            return {
                "name": name,
                "n": len(times),
                "rps": round(len(times) / wall, 3) if wall else 0,
                "p50_s": round(statistics.median(times), 3),
                "p95_s": round(sorted(times)[max(0, int(0.95 * len(times)) - 1)], 3),
                "mean_s": round(statistics.mean(times), 3),
            }

        out = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_urls": bases,
            "model": model,
            "concurrency": args.concurrency,
            "n_turns": args.n_turns,
            "wall_s": round(wall, 3),
            "turns_per_s": round(args.n_turns / wall, 3) if wall else 0,
            "proj_4000_min": round(4000 / (args.n_turns / wall) / 60, 1) if wall else None,
            "sample_prompt_tokens": args.sample_prompt_tokens,
            "echo_prompt_tokens": args.echo_prompt_tokens,
            "sample": pack("sample", sample_t),
            "echo": pack("echo", echo_t),
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps({
            "turns_per_s": out["turns_per_s"],
            "proj_4000_min": out["proj_4000_min"],
            "wall_s": out["wall_s"],
            "sample_rps": out["sample"]["rps"],
            "echo_rps": out["echo"]["rps"],
            "echo_p50": out["echo"]["p50_s"],
        }))
        print("WROTE", path)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", action="append", required=True)
    p.add_argument("--model", default="zai-org/GLM-4.5-Air-FP8")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--n-turns", type=int, default=48)
    p.add_argument("--sample-prompt-tokens", type=int, default=2048)
    p.add_argument("--echo-prompt-tokens", type=int, default=8192)
    p.add_argument("--sample-max-tokens", type=int, default=256)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument(
        "--out",
        default=str(ROOT / "ops/teacher-b200/artifacts/bench_mixed_latest.json"),
    )
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
