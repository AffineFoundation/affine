# Affine (SN120) validator architecture

One root machine runs the whole control plane; GPU work happens on a single
auto-provisioned eval machine. One validator, no consensus — trust is earned
through a replayable audit trail, not multi-validator voting.

```
┌─ root machine (coldkeypub + validator hotkey) ────────────────────────────┐
│  affine-validator (pm2, doppler-injected secrets)                         │
│    ├─ chain.py        scan commit-reveals, set rolling-king weights       │
│    ├─ model_store.py  HF revision pinning, hygiene, copy detection        │
│    ├─ state.py        king/queue/history (local JSON+JSONL, crash-safe)   │
│    ├─ provisioner.py  keep 1 eval pod alive (Lium → Targon fallback)      │
│    ├─ eval_client.py  duel dispatch + SSE stream + idle watchdog          │
│    ├─ bench.py        tau2 queue (runs when no duel is pending)           │
│    └─ dashboard.py    data/*.json + website → Hippius S3                  │
│         │ ssh tunnel (lium port-forward / ssh -N -L) :9000                │
└─────────┼─────────────────────────────────────────────────────────────────┘
          ▼
┌─ eval machine (8×H200 pod, reprovisioned automatically) ──────────────────┐
│  evalsrv (FastAPI, self-restarting; self-kills on fatal CUDA errors)      │
│    ├─ engine.py       vLLM slots: teacher (warm) · king (warm) · chall.   │
│    ├─ dueling.py      seeded slice → probe → Reason scoring → verdict     │
│    └─ benchrunner.py  tau2 airline/retail/telecom vs served model         │
└────────────────────────────────────────────────────────────────────────────┘
          ▼
   Hippius S3 bucket: index.html + data/{dashboard,history,benchmarks,contract}.json
```

## Submission → verdict flow

1. Miner pushes a checkpoint to HF and runs `scripts/submit.py`, which
   commits `affine1|repo|revision|hotkey` on-chain (reveal after 3 blocks).
2. Validator scans reveals each tick. Intake burns the hotkey's one eval
   slot at enqueue and dedups revisions globally.
3. Metadata-only validation: repo pattern + coldkey token, safetensors
   layout, no `*.py` / `auto_map`, size cap, per-blob copy check against the
   king (identical weights with an earlier commit date crown the original
   author without an eval; otherwise reject).
4. Duel on the eval machine: slice of `n_turns` drawn from public corpus D
   seeded by `blake2b(reveal_block_hash ‖ hotkey)` (block hash fetch is
   fail-closed: chain hiccup ⇒ requeue, never a predictable fallback slice);
   injectability probe; full Reason instrumentation against the warm teacher;
   verdict = `paired mean(Reason_c − Reason_k) > k_sigma·SE` — purely relative,
   no gates and no absolute floors (Reason v3, 2026-08-10). Everything the
   retired S* v2 gates measured (causality/leakage, bank frac, calibration r,
   baseline magnitude, L1lift, lengths) is published as verdict telemetry.
   Frozen constants in `affine.toml [duel]` (`n_turns`, `k_sigma`).
5. Win ⇒ crown, immediate weight set. Either way, tau2 suites are enqueued
   (advisory only — never part of the score; a running bench is aborted server-side
   the moment a duel arrives, then requeued). Weights go to the rolling king
   chain (current + up to 4 prior kings, equal share, burn fallback), gated
   on metagraph freshness and stamped with `weight_version_key`.

Hygiene caps both the `.safetensors` bytes and the TOTAL repo bytes (plus
file-count and config.json-size guards), so junk files can't disk-DoS the
pod. Copy detection compares blob-digest multisets, so renaming shards or
padding the repo with an extra safetensors doesn't evade it.

## Fault tolerance

- **Tick watchdog**: wedged main loop self-exits; pm2 restarts; state
  reloads from disk and the king chain reconciles from `history.jsonl`.
- **Duels run as a background task**: ticks keep their cadence during a
  multi-hour eval, so weight-setting, dashboard flushes, reveal scanning,
  and the provisioner are never starved by a duel in flight.
- **Eval machine**: health-checked every 60s through the tunnel; N
  consecutive failures ⇒ terminate + reprovision (Lium first, Targon REST
  fallback), rsync code (excluding venvs/state/logs/.env), bootstrap, wait
  healthy. Providers are behind a `Provider` interface (`LiumProvider`,
  `TargonProvider`); a price cap is enforced pre-rent (cheapest executor
  under `max_price_per_hour` via `lium ls`, Targon inventory cost check).
  Provisioning runs in a background thread — ticks, reveal scanning,
  weight-setting, and the watchdog never block on it — and failed attempts
  back off exponentially (60s → 15 min) so a provider outage isn't hammered.
  A validator restart adopts the existing pod (tunnel rebuilt from state,
  stale tunnel pid killed) instead of tearing down a healthy machine.
  Termination is verified: if the provider doesn't confirm release, the pod
  record is kept and termination retried rather than double-renting.
- **Miner fairness under flaky infra**: duels interrupted by infra failures
  requeue at the front; the bounded retry budget is only spent when the
  failure was plausibly the challenger's. Server-busy contention, a dead
  machine, and self-diagnosed pod trouble (low disk, dead teacher — the
  server tags these `infra:`) requeue for free. A popped challenge is never
  silently dropped: unexpected exceptions requeue (bounded) or record an
  `internal_error` verdict.
- **Stream idle watchdog**: evalsrv heartbeats every 30s; silence past the
  timeout is classified transient. If the stream is lost, the verdict is
  recovered from the job record (`GET /duel/{id}`) before declaring the
  eval transient — a finished duel is never re-run because of a dropped
  connection.
- **Eval server auth**: every evalsrv endpoint requires the shared
  `X-Affine-Token` (from `AFFINE_EVAL_TOKEN`); Targon pods expose only
  DIRECT ssh, never a public proxied port. Secrets reach the pod as a 0600
  env file written over stdin — never on a command line — and are redacted
  from provisioner logs.
- **Fail-closed inputs**: the turn corpus syncs from sharded objects + an
  immutable-manifest pointer on the public bucket (`turns/manifest.json`);
  the pod verifies the manifest hash against its published immutable copy
  and every shard sha256, and bootstrap refuses to start without a verified
  corpus. `min_submission_block` must be set (>-1) or the validator refuses
  to boot; a stale metagraph refuses weight-setting.
- **Hippius**: all dashboard writes fail open with a cooldown; history tails
  are read from the end of the file, not by scanning it.

## Trust & audit

Every verdict records the reveal-block hash, slice seed + digest, duel
params, and per-side Reason + telemetry summaries. Anyone with the two
checkpoints, the public corpus, and `affine/score.py` can recompute the
verdict. There is no validator-private data (explicitly rejected 2026-08-03)
— unpredictability comes from the reveal-block hash, and memorization
attacks are handled by fresh per-duel teacher refs + seed-shuffled slices
(pre-fork verdicts additionally carry the retired S* v2 gate constants).

## Running it

```bash
cd affine && uv venv && source .venv/bin/activate && uv pip install -e .
doppler run -- pm2 start scripts/ecosystem.config.js
```

Secrets expected in the environment: `HF_TOKEN`, `HIPPIUS_ACCESS_KEY`,
`HIPPIUS_SECRET_KEY`, `LIUM_API_KEY` (CLI must be `lium init`-ed),
`TARGON_API_KEY`, `AFFINE_EVAL_TOKEN` (shared secret for the eval server),
optional `OPENROUTER_API_KEY` (bench user-sim fallback).
The bittensor wallet (`affine`/`validator`) must exist on this machine.

Offline verification: `scripts/smoke_test.py` (39 checks over config,
scoring floors, state invariants, hygiene, copy detection, chain decode,
tail reads, price parsing, and the eval-server auth surface).
