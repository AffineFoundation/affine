#!/usr/bin/env python3
"""Throughput bench for a remote (or local) GLM teacher endpoint.

Measures two duel-relevant workloads:
  1) sample  — short generation (teacher ref z/y style)
  2) echo    — forced logprobs over a long prompt (lpC path)

Usage:
  source .venv/bin/activate
  python ops/teacher-b200/bench_teacher.py --base-url http://HOST:PORT/v1
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


async def wait_ready(client: httpx.AsyncClient, base: str, model: str) -> str:
    r = await client.get(f"{base}/models")
    r.raise_for_status()
    data = r.json()
    ids = [m["id"] for m in data.get("data", [])]
    if model not in ids and ids:
        print(f"warn: expected {model}, got {ids}")
        return ids[0]
    return model


def make_prompt(n_tokens_approx: int, salt: int = 0) -> str:
    # Rough: ~1 token ≈ 4 chars for English filler.
    # Salt breaks vLLM prefix-cache so waves measure real prefill cost.
    unit = "The quick brown fox jumps over the lazy dog. "
    reps = max(1, (n_tokens_approx * 4) // len(unit))
    body = (unit * reps)[: max(0, n_tokens_approx * 4 - 64)]
    return f"TURN_SALT={salt};" + body


async def one_sample(client: httpx.AsyncClient, base: str, model: str,
                     prompt: str, max_tokens: int, temperature: float) -> float:
    t0 = time.perf_counter()
    r = await client.post(
        f"{base}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
    )
    r.raise_for_status()
    return time.perf_counter() - t0


async def one_echo(client: httpx.AsyncClient, base: str, model: str,
                   prompt: str) -> float:
    # Teacher-force style: echo prompt tokens with logprobs (temp 0).
    t0 = time.perf_counter()
    r = await client.post(
        f"{base}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0,
            "echo": True,
            "logprobs": 1,
            "prompt_logprobs": 1,
        },
    )
    # Some vLLM builds want prompt_logprobs differently; fall back.
    if r.status_code >= 400:
        r = await client.post(
            f"{base}/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": 1,
                "temperature": 0,
                "echo": True,
                "logprobs": 1,
            },
        )
    r.raise_for_status()
    return time.perf_counter() - t0


async def run_wave(coro_factory, n: int, concurrency: int) -> list[float]:
    sem = asyncio.Semaphore(concurrency)
    out: list[float] = []

    async def wrap():
        async with sem:
            out.append(await coro_factory())

    await asyncio.gather(*[wrap() for _ in range(n)])
    return out


def summarize(name: str, times: list[float], n: int, wall_s: float) -> dict:
    d = {
        "name": name,
        "n": n,
        "wall_s": round(wall_s, 3),
        "sum_latency_s": round(sum(times), 3),
        "rps": round(n / wall_s, 3) if wall_s else 0,
        "p50_s": round(statistics.median(times), 3),
        "p95_s": round(sorted(times)[max(0, int(0.95 * len(times)) - 1)], 3),
        "mean_s": round(statistics.mean(times), 3),
    }
    print(json.dumps(d))
    return d


async def amain(args: argparse.Namespace) -> int:
    base = args.base_url.rstrip("/")
    timeout = httpx.Timeout(args.timeout_s)
    limits = httpx.Limits(max_connections=args.concurrency + 8,
                          max_keepalive_connections=args.concurrency + 8)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        model = await wait_ready(client, base, args.model)
        print(f"ready model={model} base={base}")

        results = []
        # Warmup
        await one_sample(
            client, base, model, make_prompt(args.sample_prompt_tokens, -1), 32, 0.8
        )
        await one_echo(
            client, base, model, make_prompt(min(2000, args.echo_prompt_tokens), -2)
        )

        sample_i = 0

        async def sample_once():
            nonlocal sample_i
            i = sample_i
            sample_i += 1
            return await one_sample(
                client,
                base,
                model,
                make_prompt(args.sample_prompt_tokens, i),
                args.sample_max_tokens,
                0.8,
            )

        echo_i = 0

        async def echo_once():
            nonlocal echo_i
            i = echo_i
            echo_i += 1
            return await one_echo(
                client, base, model, make_prompt(args.echo_prompt_tokens, 10_000 + i)
            )

        t0 = time.perf_counter()
        times = await run_wave(sample_once, args.n_sample, args.concurrency)
        results.append(summarize("sample", times, args.n_sample, time.perf_counter()-t0))

        t0 = time.perf_counter()
        times = await run_wave(echo_once, args.n_echo, args.concurrency)
        results.append(summarize("echo", times, args.n_echo, time.perf_counter()-t0))

        out = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "base_url": base,
            "model": model,
            "concurrency": args.concurrency,
            "sample_prompt_tokens": args.sample_prompt_tokens,
            "echo_prompt_tokens": args.echo_prompt_tokens,
            "results": results,
        }
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out, indent=2) + "\n")
        print("WROTE", path)
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", default="zai-org/GLM-4.5-Air-FP8")
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--n-sample", type=int, default=48)
    p.add_argument("--n-echo", type=int, default=48)
    p.add_argument("--sample-prompt-tokens", type=int, default=2048)
    p.add_argument("--echo-prompt-tokens", type=int, default=8192)
    p.add_argument("--sample-max-tokens", type=int, default=256)
    p.add_argument("--timeout-s", type=float, default=600.0)
    p.add_argument(
        "--out",
        default=str(ROOT / "ops/teacher-b200/artifacts/bench_latest.json"),
    )
    return asyncio.run(amain(p.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
