# Redeploy the benchmark node from scratch

Everything needed is in this directory. Secrets stay on the host (`affinevalidator`): the Lium API key (`~/.lium`), the R2
presign credentials used by the gate's `presign.py`. The node gets only public code, public HF weights and short-lived
presigned GET URLs (deleted right after each king download).

1. **Rent** (host): `bash bench/deploy/rent.sh` picks 4xB300, else the cheapest secure 4-8x B300/B200/H200 node <= `MAX_PRICE`
   (default $30/h, the owner's bench-pod cap; never above $35/h), or pass a node huid; names it `affine-bench-1`, and writes the
   ssh wrapper `~/bench-work/bp.sh`. Record the pod in `bench/deploy/pod.json` (name, huid, lium id, gpu, $/h), then restart the
   daemon: a run that belonged to the old pod is requeued (`pod-changed`), not counted as a failure.
   Lium can reclaim a pod (`executor_reclaimed`); the daemon then logs `node-gone` once (it checks `lium describe` after 10 min
   unreachable) and benchmarks.json shows `node.pod_gone`. That is the signal to redo steps 1-2.
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
