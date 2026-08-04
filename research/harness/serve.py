"""Launch vLLM servers for kings on the pod (teacher assumed already up on :8000).

Usage: python -m harness.serve --kings genesis:8001:2 I:8002:3 XCIX:8003:4 ...
Each spec is suffix:port:gpu. Prints readiness; run under nohup.
"""

import argparse
import os
import subprocess
import time

import httpx

ENV_BASE = {
    "HF_HOME": "/root/hf",
    "CUDA_HOME": "/usr/local/cuda",
    "VLLM_ATTENTION_BACKEND": "FLASH_ATTN",
    "VLLM_USE_FLASHINFER_SAMPLER": "0",
}


def launch(name: str, repo: str, port: int, gpu: str):
    env = dict(os.environ, **ENV_BASE,
               CUDA_VISIBLE_DEVICES=gpu,
               PATH="/usr/local/cuda/bin:" + os.environ["PATH"])
    log = open(f"/root/logs/serve_{name}.log", "w")
    # Teacher-forcing sends echo+prompt_logprobs requests; vLLM materializes a
    # fp32 log_softmax over (prefill_chunk x vocab) per request (~4GB at 4k
    # chunk with a 250k vocab), so keep chunks small and leave real headroom
    # or the engine OOMs and dies (observed 2026-08-02).
    subprocess.Popen(
        ["vllm", "serve", repo, "--port", str(port),
         "--max-model-len", "32768", "--gpu-memory-utilization", "0.85",
         "--max-num-batched-tokens", "2048"],
        env=env, stdout=log, stderr=log, stdin=subprocess.DEVNULL,
        start_new_session=True)
    print(f"launched {name} port {port} gpu {gpu}")


def wait_ready(ports: list[int], timeout_s: int = 1800):
    t0 = time.time()
    pending = set(ports)
    while pending and time.time() - t0 < timeout_s:
        for p in list(pending):
            try:
                httpx.get(f"http://localhost:{p}/v1/models", timeout=3)
                pending.discard(p)
                print(f"port {p} READY ({time.time() - t0:.0f}s)")
            except httpx.HTTPError:
                pass
        time.sleep(10)
    if pending:
        raise RuntimeError(f"not ready after {timeout_s}s: {sorted(pending)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kings", nargs="*", default=[], help="suffix:port:gpu")
    ap.add_argument("--models", nargs="*", default=[], help="name=repo:port:gpu")
    args = ap.parse_args()
    ports = []
    for spec in args.kings:
        suffix, port, gpu = spec.split(":")
        launch(f"king-{suffix}",
               f"dendriteholdings/albedo-qwen3.6-35b-king-{suffix}",
               int(port), gpu)
        ports.append(int(port))
        time.sleep(5)
    for spec in args.models:
        name, rest = spec.split("=", 1)
        repo, port, gpu = rest.rsplit(":", 2)
        launch(name, repo, int(port), gpu)
        ports.append(int(port))
        time.sleep(5)
    wait_ready(ports)


if __name__ == "__main__":
    main()
