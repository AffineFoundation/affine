# Redeploy the benchmark node from scratch

Everything needed is in this directory. Secrets stay on the host (`affinevalidator`): the Lium API key (`~/.lium`), the R2
presign credentials used by the gate's `presign.py`. The node gets only public code, public HF weights and short-lived
presigned GET URLs (deleted right after each king download).

1. **Rent** (host): automatic since 2026-11-02. The daemon (`affine_bench/pod.py`) rents a pod when a king is queued and none
   exists (`bench/deploy/rent.sh`: cheapest secure 4-8x B300/B200/H200 by $/GPU-h, then most GPUs, <= `BENCH_MAX_PRICE`, default
   $30/h; never a second pod named `affine-bench-1`; at most 8 rents/24 h), bootstraps it (step 2), and releases it (`lium rm`
   on its exact id, only if still named `affine-bench-1`) after 15 min with an empty queue. A pod Lium reclaims is detected
   (`node-gone`, via `lium describe` after 10 min unreachable), its run is requeued without a failure, and a new pod is rented.
   The live pod is in `state/bench-results/pod.json`; `bench/deploy/pod.json` is only the seed. Manual: `bash bench/deploy/rent.sh [NODE_HUID]`.
   Queue: only GREEN kings with epoch >= `BENCH_MIN_EPOCH` (544), newest first, plus manual `affine-bench run` requests; a king's
   weights are deleted from the pod once all its results are collected (host results are never deleted).
2. **Bootstrap** (node, from the repo itself):
   ```
   ~/bench-work/bp.sh 'mkdir -p /data/bench && git clone https://github.com/AffineFoundation/affine.git /data/bench/repo && bash /data/bench/repo/bench/deploy/bootstrap.sh'
   ```
   Installs uv, the vLLM venv (exact SN120 probe closure, `deploy/requirements-vllm.txt`), the harness venv (verifiers 0.4.0,
   prime 0.19.0, `deploy/requirements-harness.txt`), the pinned Hub envs (`deploy/requirements-envs.lock`, sha256-checked)
   and the repo envs (`bench/environments/*`). Ends with `BOOTSTRAP_OK`. Plain disk only (`/data/bench`), never `/root`.
3. **Smoke** (node): `bash /data/bench/repo/bench/deploy/serve.sh /data/bench/models/ref/<some model> qwen3` then the
   `vf-eval local-math500 ... -n 20` line from README.md; `bash bench/deploy/stop.sh` stops the server (exact PID).
4. **Host code**: `git clone https://github.com/AffineFoundation/affine.git ~/bench-work/affine` (then
   `ln -sfn ~/bench-work/affine/bench ~/subnet120-rewrite/bench`). Paths are in `bench/affine_bench/config.py` (override
   with `BENCH_*` env vars).
5. **Daemon** (host): `install -m 644 bench/systemd/affine-bench.service ~/.config/systemd/user/ && systemctl --user daemon-reload &&
   systemctl --user enable --now affine-bench`. State/results: `~/subnet120-rewrite/state/bench-results/` (results/ public
   aggregates; private/ per-task, 0600, never published). Restarting is safe: finished benchmarks are never re-run, an
   interrupted model is resumed (`vf-eval --resume`).
6. **Site** (host): `bash bench/deploy/activate_site.sh` (backs up index.html; adds `bench.js`/`bench.css`); undo with
   `bench/deploy/revert_site.sh`. Data file: `affine.io/benchmarks.json`, written by the daemon.
7. A new node scores the same way (same pins, same configs); `serving.toml` `version` stays unless settings change. A
   different GPU type moves numbers by about their SE (vLLM batching is not bit-reproducible), so re-run Genesis + one
   reference first and compare.
