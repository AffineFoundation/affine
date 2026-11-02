# SN120 Lium pods: watch and one-command replacement

Lium pods get reclaimed often, so every SN120 pod role has a tested way to re-rent and re-provision it.
The tool runs on the validator host (`affinevalidator`), which holds the Lium key. The key never leaves the host.

Code is in `ops/pods/` (`replace_pod.py`, `roles.json`, `provision/vllm_venv.sh`).
It is deployed to `~/subnet120-rewrite/ops/pods/`. State is in `~/subnet120-rewrite/state/pods/`:

| file | what |
|---|---|
| `status.json`, `status.md` | per-role liveness: `up` / `unreachable` / `dead` (unreachable for 20 min or more) / `down` (Lium status not RUNNING) / `gone` (no longer listed), plus the action |
| `events.jsonl` | one line per state change (node-gone, node-back) |
| `rents.jsonl` | rent ledger. It enforces the caps: `max_rents_24h` per role and 6 in total |
| `golden/<role>/` | the role's code and runtime, copied from the live pod with no secrets (`snapshot`) |
| `runs/<ts>-<role>/` | backups of every host file a run changed, plus `revert.sh` |
| `ssh/<role>.sh` | ssh wrapper for a newly rented pod |

## Commands (on the host)

```
cd ~/subnet120-rewrite/ops/pods
python3 replace_pod.py status                     # every role; exit 3 if a required role is gone/down/dead
python3 replace_pod.py replace ROLE --dry-run     # full plan: candidates, provisioning steps, host-config diff, rm; changes nothing
python3 replace_pod.py replace ROLE               # rent + provision + health + repoint + lium rm of the exact dead id
python3 replace_pod.py replace ROLE --resume HUID # re-run the idempotent provisioning on a pod an earlier run already rented
python3 replace_pod.py snapshot ROLE              # refresh the golden copy from the live pod (read-only on the pod, nice/ionice)
python3 replace_pod.py verify ROLE                # read-only: venv pins, files and golden listing vs the live pod, then the health check
python3 replace_pod.py repoint ROLE --to HUID [--dry-run] [--rm-old]   # host-side repoint only (after a manual ROOT step)
```

Watch timer, every 5 min. It only runs `status` and never rents:
`install -m 644 systemd/sn120-pod-watch.* ~/.config/systemd/user/ && systemctl --user daemon-reload && systemctl --user enable --now sn120-pod-watch.timer`.

## Guards

- **Excluded pods are never touched:** Casper `swift-wolf-bd`, Tobias `zesty-orbit-fe` / `affine-tobias-4xh200`, and mech-1 (`roles.json _excluded`).
- **Live pods are never replaced.** `replace` refuses in any of these cases:
  - the role's current pod is `up`;
  - the pod has been unreachable for less than 20 min;
  - a reachable pod already carries the role's name.
- **Rent caps:** the per-role cap and 6 rents per 24 h in total.
- **Removal:** only `lium rm <exact old id>`. It runs only if Lium still lists that id, the pod carries one of the role's names, and it is unreachable.
- **No secrets on pods:**
  - Code comes from GitHub (`git clone` on the pod) or from the host's golden copy. The golden copy excludes `*.seed`, `r2.env`, `private/` and `id_*`, and the file list is checked again before copying.
  - Checkpoints come from presigned R2 GETs (the gate's `presign.py`, 2 h). The URL file is deleted right after `gate_fetch.py`.
  - Pods are rented with `--no-volume-encryption`. Everything goes on plain disk, per STANDING-RULES.
- **Revert:** every edited host file is backed up first. Run `bash ~/subnet120-rewrite/state/pods/runs/<ts>-<role>/revert.sh` (the new pod is kept).

## Hourly health routine (Jacob 2026-11-02 10:34 CR: rent replacements without asking)

1. Read `~/subnet120-rewrite/state/pods/status.json`. Run `replace_pod.py status` first if it is older than 10 min.
2. For each role whose `state` is `gone`, `down` or `dead`, follow `action`:
   - `auto-replace` (`prior`, `perf`, `probe`): run `python3 replace_pod.py replace ROLE`, then report the new pod, its $/h and `revert.sh`.
     - **Probe:** if the output says `no node offered` and shows a `--allow-gpu-change` alternative (8xB300 or 8xH200), rerun with `--allow-gpu-change`. Tell Jacob, because gate probe numerics move slightly. Both sides of a gate comparison still run on the same pod.
   - `delegated` (bench): the affine-bench daemon replaces it by itself. Only check `journalctl --user -u affine-bench`.
   - `manual` (trainer, evaluator): rent and restore the public runtime with `replace ROLE --allow-manual`, then stop. A ROOT step is needed (below), so ask Jacob.
   - `optional` (perf2): do nothing unless asked.
3. If `replace` refuses (caps, unreachable for under 20 min, no offer), report why and leave things as they are.

## Roles

| role | pod now | spec (fallbacks in order, cheapest node first) | host config repointed | automatic |
|---|---|---|---|---|
| bench | lunar-matrix-98 8xH200 | `bench/deploy/rent.sh` | `state/bench-results/pod.json` (daemon) | yes, by the affine-bench daemon |
| prior | cosmic-raven-ac 8xH100 | 8xH100, 8xH200, 8xB200 whole node, <= $32/h, secure or spot | `RA/20261101-prior-table/pp.sh`, `controller.json` (`pod_lium_id`), `pull.json` offset -> 0; restart `affine-prior-table` | yes |
| perf | calm-orbit-ef 2xH200 | 2xH200, 2xB200, 2xH100, 1xB200, <= $13/h | `RA/20261031-performance-eval/pp.sh`, `perf_service.py` PODS (lium_id, gpu, cost), `revert_perf.sh`; restart `affine-perf-eval` | yes |
| perf2 | eager-wolf-98 1xB200 | 1xB200, 1xH200, <= $7/h | `pp2.sh`, `perf_service.py`, `revert_perf.sh` (unit only if enabled) | on request (pod 2 tears itself down) |
| probe | noble-shark-d6 8xB200 | 8xB200 whole node secure, <= $80/h; 8xB300 / 8xH200 only with `--allow-gpu-change` | `gate_service.py` PROBE_SSH, `probe_daemon.py` POD, `khalf_build.py`, `percache_watch.py`, `disk_guard.py`, plain-disk `pod.sh`/`podin.sh`; restart `affine-promotion-gate`, `affine-env-probe` | yes |
| trainer | golden-raven-7a 8xH200 | 8xH200, 8xB200 whole node secure, <= $45/h | `rssh.sh`, last trainer-supervised and tunnel drop-ins, `disk_guard.py`, `fp32_prune.py`, `watchdog_ts.py`, `watchdog_ff.py`, `pod.sh`/`podin.sh` | public part only, then ROOT |
| evaluator | gentle-lion-27 1xH200 | 1xH200, 1xB200 secure, <= $7/h | last evaluator-supervised drop-in, evaluator-tunnel unit, `disk_guard.py`, `pod.sh`/`podin.sh` | public part only, then ROOT |

### prior

Provisioning:

1. Clone GitHub.
2. Create `/opt/prior/venv` with `provision/vllm_venv.sh`: `bench/deploy/requirements-vllm.txt` (the probe pins) plus `math-verify==0.9.0`.
3. Copy the host's `~/subnet120-rewrite/priors` -> `/opt/prior/code/priors`, plus `pod/serve.sh`, `gate_fetch.py`, `sampler.json`, and the two sha256-checked task files.
4. Copy `RA/20261101-prior-table/serve.env` -> `/opt/prior/serve/serve.env` if that file exists on the host.
5. Fetch the gate head: host `presign.py`, then pod `gate_fetch.py`.
6. Run `serve.sh <head> 0.90` (8 replicas) and start the worker.
7. Health check: `SERVING.json` replicas == GPUs, and the worker pid is alive.

Then the tool stops `affine-prior-table`, repoints the host config, resets the `pull.json` offset to 0 and restarts the controller.
The controller's head reconcile (da8649a) takes over from there.

`verify prior` on cosmic-raven-ac (2026-11-02): all 200 venv pins match, and `serve.sh`, `gate_fetch.py`, `worker.py`, `grader.py` and `sampler.json` match the host copies.

### perf / perf2

Provisioning:

1. Clone GitHub.
2. Create `/data/venv` (the same pins plus math-verify; the live pod matches all 200 pins).
3. Restore the golden copy into `/data/perf/`: `perf_eval.py`, `run_king.sh`, `allan_src.py`, `gate_fetch.py`, `env.sh` and `tasks.private.json`. This is the live pod's copy, because the pod's `perf_eval.py` carries an `except BaseException` fix that the host and GitHub copies lack.
4. Health check: `import vllm, math_verify` and GPUs visible.

`perf_service.py` fetches kings itself. The PODS key (`calm-orbit-ef`, `eager-wolf-98`) is kept as the slot name, because the unit's `PERF_POD` and the watchdog use it. Only `lium_id`, `gpu` and `cost` change.

### probe

Provisioning:

1. Restore the golden copy of `/opt/plain/probe` (see the excludes in `roles.json`): `affine-env-probe` code/envs/prepcache/extra-site, the `affine-mimo-20261013` venv, `affine-mimo-public-evaluator-20261013` without `private/`, `.local`, and `.cache/uv`. Checkpoints, runs, logs, compile caches and secrets are left out.
2. Recreate the `/root/...` symlinks.
3. Fetch the gate head into `ckpt/` (the gate's delta source).
4. Health check: the venv imports vLLM with 8 GPUs, and `podctl.py status` works.

The gate serves candidates itself (`podctl serve`). Restarting `affine-promotion-gate` and `affine-env-probe` changes no gate or pay logic. Only the ssh target moves.

### trainer and evaluator: ROOT step required

Both pods hold role signing seeds by design: the trainer has `private/trainer.seed`, `publisher.seed` and `r2.env`; the evaluator has `private/evaluator.seed`.
This tool never copies those files. `replace ROLE --allow-manual` rents the pod, restores the public runtime (golden copy: `runtime-exact` venv, `host-v8*`, `owned-worker-*.json`, `source-final-*`, or the evaluator venv and sources), runs the health check, and stops.

ROOT then follows the 2026-10-22 migration (`RA/20261022-trainer-pod-migration/ROLLBACK.md`, `RECOVERY.md`):

1. Stage the private files with the `stage_pod.py` pattern.
2. Rebuild the trainer state from the published head with `rebuild_trainer_state_pod.py`. This gives fresh Adam moments, step = CURRENT.
3. If a job expired, apply the `recover_e88.sh` precedent.
4. Run `replace_pod.py repoint trainer --to HUID --rm-old`. It rewrites the active drop-ins, `rssh.sh` and the watchdogs, then restarts the tunnel, the supervised worker and `ts-fp32-prune`.

The trainer's optimizer state (`trainer-state`, `fp32-master`) exists only on the pod. A lost trainer pod always means a fresh-moment rebuild.

## Keeping the golden copies current

A golden copy is what a dead pod gets rebuilt from, so refresh it after any change on the pod:
`replace_pod.py snapshot probe|trainer|evaluator|perf`.
`verify ROLE` reports drift (new or changed files since the copy). Run it from the hourly routine and re-snapshot on drift.

## Tests

`python3 -m pytest -q ops/pods/test_replace_pod.py` runs offline, with no Lium or ssh. It covers:

- endpoint substitution on every config format;
- the shared node IP case: several pods share one node IP and only the right line changes;
- the exclusions;
- node choice and fallbacks;
- the rent caps.
