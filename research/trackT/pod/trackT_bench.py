#!/usr/bin/env python3
"""Track T panel bench: runs a SWE panel (proxy-16 or full-150) against the
DEDICATED bench server (:8006, GPUs 6,7) so the GAN loop never pauses.

Responsibilities: hot-load the checkpoint adapter on the bench server,
assert the adapter actually changes generations (the silent no-op class),
run the panel via Track A's swerunner, persist per-instance results for
McNemar pairing, compute p_vs_baseline against the raw-student control run,
and append the [bench] status line. Clears bench.lock on exit.

Runs under /root/benchenv/bin/python (no torch/transformers needed here).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, "/root/work")

# swerunner shells out to `mini-extra` and `python -m swebench...`; both must
# resolve to the benchenv. The HF cache must be the /dshare copy (see driver).
os.environ["PATH"] = "/root/benchenv/bin:" + os.environ.get("PATH", "")
os.environ["HF_HOME"] = "/dshare/hf"

STUDENT = "dendriteholdings/albedo-qwen3.6-35b-king-genesis"
PORT = 8006
URL = f"http://127.0.0.1:{PORT}"


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def mcnemar_exact(base_map, new_map):
    """Exact two-sided McNemar on the intersection of instances."""
    common = set(base_map) & set(new_map)
    b = sum(1 for i in common if base_map[i] and not new_map[i])
    c = sum(1 for i in common if new_map[i] and not base_map[i])
    n = b + c
    if n == 0:
        return 1.0, b, c
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail), b, c


def resolved_map(art):
    return {i: bool(v.get("resolved"))
            for i, v in (art.get("instances") or {}).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="proxy_rN | full_rN | full_baseline")
    ap.add_argument("--model", required=True, help="served name (adapter or base)")
    ap.add_argument("--adapter", default="", help="adapter dir ('' = base model)")
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--work", default="/dshare/trackT")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=14400)
    args = ap.parse_args()

    W = args.work
    kind = args.tag.split("_")[0]  # proxy | full
    pin_path = f"/root/work/panels/panel_{'proxy' if kind == 'proxy' else 'full'}.json"

    def status(line):
        with open(f"{W}/status.log", "a") as fh:
            fh.write(f"{now()} [bench] {line}\n")
        print(line, flush=True)

    def gpu_hours():
        try:
            t0 = int(open(f"{W}/pod_t0").read().strip())
        except Exception:
            t0 = int(time.time())
        return round((time.time() - t0) / 3600 * 8, 1)

    try:
        if args.adapter:
            # hot-load the checkpoint adapter on the bench server
            bname = f"bench_{args.model}"
            for _ in range(3):
                try:
                    requests.post(f"{URL}/v1/unload_lora_adapter",
                                  json={"lora_name": bname}, timeout=60)
                except Exception:
                    pass
                try:
                    r = requests.post(f"{URL}/v1/load_lora_adapter",
                                      json={"lora_name": bname,
                                            "lora_path": args.adapter},
                                      timeout=600)
                    r.raise_for_status()
                    break
                except Exception:
                    time.sleep(20)
            else:
                status(f"FAILED tag={args.tag}: adapter load error")
                return
            serve_name = bname

            # adapter-effect gate: greedy sample must differ from base
            def sample(model):
                r = requests.post(
                    f"{URL}/v1/chat/completions",
                    json={"model": model, "temperature": 0.0,
                          "max_tokens": 256,
                          "messages": [{"role": "user", "content":
                                        "Explain, step by step, how you "
                                        "would locate a bug in a large "
                                        "Python repo."}]},
                    timeout=600)
                r.raise_for_status()
                return r.json()["choices"][0]["message"]["content"] or ""
            if sample(serve_name) == sample(STUDENT):
                status(f"ABORTED tag={args.tag}: adapter-effect gate failed "
                       f"({serve_name} output identical to base -- silent "
                       f"no-op adapter?)")
                return
        else:
            serve_name = args.model  # raw base (the control run)

        os.environ["AFFINE_BENCH_DIR"] = f"/dshare/bench/{kind}"
        from evalsrv import swerunner
        swerunner.IDS_PATH = Path(pin_path)
        swerunner.BENCH_DATA_DIR = Path(os.environ["AFFINE_BENCH_DIR"])

        n = len(json.load(open(pin_path))["instance_ids"])
        status(f"START tag={args.tag} ckpt={args.model} n={n} "
               f"gpu_hours={gpu_hours()}")
        res = swerunner.run_swe_lite(serve_name, PORT,
                                     workers=args.workers,
                                     timeout_s=args.timeout)
        art = res.pop("_artifact", None) or {}
        if art.get("instances"):
            json.dump(art, open(f"{W}/bench_{args.tag}_instances.json", "w"))
            res["per_instance"] = {
                iid: {"resolved": bool(v.get("resolved")),
                      "exit_status": v.get("exit_status"),
                      "patch_len": len(v.get("model_patch") or "")}
                for iid, v in art["instances"].items()}
        json.dump(res, open(f"{W}/bench_{args.tag}.json", "w"))

        if not res.get("ok"):
            status(f"FAILED tag={args.tag}: {str(res.get('error'))[:200]}")
            return

        if kind == "proxy":
            status(f"PROXY round={args.round} ckpt={args.model} "
                   f"score={res['n_resolved']}/{res['n_instances']} "
                   f"gpu_hours={gpu_hours()} wall={res['wall_time_s']}s")
        else:
            p_str = "n/a"
            base_path = f"{W}/bench_full_baseline_instances.json"
            if args.tag != "full_baseline" and os.path.exists(base_path):
                base = resolved_map(json.load(open(base_path)))
                p, b, c = mcnemar_exact(base, resolved_map(art))
                p_str = f"{p:.4f}(b={b},c={c})"
            status(f"FULL ckpt={args.model} score={res['score']:.4f} "
                   f"resolved={res['n_resolved']}/{res['n_instances']} "
                   f"p_vs_baseline={p_str} gpu_hours={gpu_hours()} "
                   f"round={args.round} wall={res['wall_time_s']}s "
                   f"(refs: teacher27B=0.3133 stock35B=0.1333)")
    finally:
        try:
            os.remove(f"{W}/bench.lock")
        except Exception:
            pass


if __name__ == "__main__":
    main()
