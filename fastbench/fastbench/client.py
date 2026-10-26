"""Blocking client: bench(n, seed, model) -> result dict. Also a CLI.

    from fastbench.client import bench
    r = bench(25, 0, "Qwen/Qwen3.6-35B-A3B")
    print(r["score"])          # 0.16
    print(r["per_instance"])   # {instance_id: resolved bool}

    $ fastbench --model Qwen/Qwen3.6-35B-A3B --n 25 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

DEFAULT_BASE = os.environ.get("FASTBENCH_URL", "http://127.0.0.1:9100")


def _headers(token: str | None) -> dict:
    tok = token or os.environ.get("FASTBENCH_TOKEN", "")
    return {"X-Fastbench-Token": tok} if tok else {}


def bench(n: int, seed: int, model: str, *,
          base_url: str = DEFAULT_BASE, token: str | None = None,
          poll_s: float = 10.0, timeout_s: float = 10800.0,
          quiet: bool = True) -> dict:
    """Submit a bench and block until it finishes, then return its result:
    score, n_resolved, per_instance, timings, plus job_id.

    Raises RuntimeError if the run failed or was aborted, so a broken box can
    never be mistaken for a low score."""
    headers = _headers(token)
    r = httpx.post(f"{base_url}/bench",
                   json={"model": model, "n": n, "seed": seed},
                   headers=headers, timeout=60)
    r.raise_for_status()
    job_id = r.json()["job_id"]
    t0 = time.time()
    last_phase = ""
    while True:
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"bench {job_id} still running after {timeout_s}s")
        j = httpx.get(f"{base_url}/bench/{job_id}", headers=headers,
                      timeout=60)
        j.raise_for_status()
        job = j.json()
        state = job["state"]
        if not quiet:
            phase = f"{state}/{job.get('phase', '')} {job.get('progress', '')}"
            if phase != last_phase:
                print(f"[{job_id}] {phase}", file=sys.stderr)
                last_phase = phase
        if state == "completed":
            result = dict(job.get("result") or {})
            result["job_id"] = job_id
            return result
        if state in ("failed", "aborted"):
            raise RuntimeError(f"bench {job_id} {state}: {job.get('error')}")
        time.sleep(poll_s)


def preload(model: str, *, base_url: str = DEFAULT_BASE,
            token: str | None = None) -> dict:
    """Start downloading a model's weights and return immediately.

    Call this for the next model while the current bench runs, so the switch
    pays only the engine boot instead of boot plus download."""
    r = httpx.post(f"{base_url}/preload", json={"model": model},
                   headers=_headers(token), timeout=60)
    r.raise_for_status()
    return r.json()


def main():
    ap = argparse.ArgumentParser(description="bench(n, seed, model)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-url", default=DEFAULT_BASE)
    ap.add_argument("--token", default=None)
    ap.add_argument("--timeout", type=float, default=10800)
    args = ap.parse_args()

    try:
        result = bench(args.n, args.seed, args.model, base_url=args.base_url,
                       token=args.token, timeout_s=args.timeout, quiet=False)
    except RuntimeError as exc:
        print(f"\n{exc}", file=sys.stderr)
        sys.exit(1)
    print(json.dumps(result, indent=1))
    print(f"\nscore={result.get('score')} "
          f"({result.get('n_resolved')}/{result.get('n')}) "
          f"wall={result.get('wall_time_s')}s "
          f"model_warm={result.get('model_warm')}", file=sys.stderr)


if __name__ == "__main__":
    main()
