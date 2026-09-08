# Affine control-plane map

Source-grounded map, 2026-09-06. `affine/affine.toml` is the contract SSOT;
monitoring does not change it. Operational endpoints and identities are deliberately
omitted. Historical module docstrings sometimes describe retired scoring rules;
the executable score path and current TOML take precedence.

## Admission → queue → verdict → weights

- **Validator:** `affine/affine/validator.py` runs asynchronous ticks with one
  outstanding duel task. Metagraph refresh, intake, provisioner checks, benchmark
  pumping, weight-setting and state/dashboard publication continue during a duel.
  `TickWatchdog` detects stuck ticks; the log heartbeat is a separate 600s signal.
- **Admission:** `affine/affine/chain.py` scans chain reveals;
  `registrations.py` + `r2protocol.py` implement signed private-model activation,
  sealed upload access, ready/revocation, inventory/signature verification and
  admission. `model_store.py` checks model hygiene and architecture. Admission
  consumes a hotkey slot; historical registration `queued` is not queue membership.
- **Durability and dispatch:** `affine/affine/state.py` owns `state.json`, king
  lineage, waiting entries, persisted `in_flight`, and append-only `history.jsonl`.
  Startup reconciles crowns and recovers unresolved work. Queue order follows
  `QueueEntry.order_key` (submission sequence plus explicit deferral), not retry
  timestamps. Validator failure classification distinguishes miner faults from
  infra requeues; repeated entry-specific infra failures can defer an entry.
- **Weights:** `Validator._maybe_set_weights` sweeps model accessibility, then
  `State.king_lineage_members` selects current king followed by newest prior
  distinct hotkeys, up to `king_chain_size`. Empty-hotkey genesis takes no seat;
  inaccessible members are skipped and deeper kings backfill. Unknown accessibility
  probes retain prior status. `chain.set_rolling_weights` subsequently filters
  unregistered hotkeys, deduplicates UIDs and assigns equal shares to remaining
  UIDs (no further lineage backfill at this step); none means all weight to burn.
  A stale metagraph prevents the write. Only a successful inclusion/dispatch result
  leads to `State.record_weights_set`. `last_weights_at` is this validator's recorded
  success, **not an independent chain confirmation by the monitor**. The configured
  interval is an attempt cadence; rate limits/ticks/probes can delay success.
- **Publication:** `affine/affine/dashboard.py` writes `state/public/snapshot.json`
  and mirrored presentation data; `affine/affine/dash/` serves the web/API/SSE view.
  Snapshot phase/current evaluation are live-memory projections, not fields in
  durable `state.json`. Public `earning`/rounded `weight_bps` describe intended
  window candidates before chain registration filtering, not proven payments.

## Evaluation and teacher serving

- **Wire protocol:** `affine/affine/eval_client.py` dispatches jobs to
  `affine/evalsrv/server.py`, consuming SSE progress, 30s heartbeats and terminal
  verdict/error events with an idle watchdog and poll fallback. Evalsrv serializes
  GPU work; duels can preempt benches. Artifacts retain sampled rollouts/logprobs.
- **Slots:** `affine/evalsrv/engine.py` manages local or remote teacher, warm king,
  per-duel challenger and optional miner replicas, plus prefetch/cache/pruning.
  `r2store.py` verifies downloaded file hashes before loading private checkpoints;
  `vllm_client.py` provides sampling/echo clients and model pools.
- **Measurement:** `affine/evalsrv/dueling.py` pins the corpus slice and runs both
  sides; its per-duel reference cache shares fresh teacher references between them.
  `terms.py` samples thoughts/actions and teacher-forces Reason, Grounding and B
  echoes. `affine/affine/dialects.py` parses per-turn actions; `score.py` computes
  the contract score and crown test. Current `min_rg` uses centered R, banded G,
  a forfeit floor and crown gates; the staged `min_rga` action leg is not live.
- **Teacher swarm:** `ops/teacher-swarm/manager.py` reconciles rental/bootstrap/
  probes/healing and publishes eligible replicas. `router.py` exposes one
  OpenAI-compatible teacher service, rendezvous-hashing the rendered turn prefix
  for cache locality, with overload spill, circuit breakers and retry. Router
  eligibility is not an independent GPU health assertion. Endpoint configuration
  in `engine.py` selects remote serving without a local teacher process.

## Traces → corpus view → immutable duel slice

- `rollouts/rollouts/run.py`, `scheduler.py`, `sources.toml` and `policies.toml`
  schedule tasks and harnesses. `runners/` executes them; `schema.py`/`store.py`
  retain canonical trace envelopes/message graphs, including tools and reasoning.
- `rollouts/rollouts/r2mirror.py` publishes immutable trace chunks and manifests
  with the latest pointer last. Traces, not pod-derived flat turns, are canonical.
- `ops/corpus_build.py` derives `duel_turns@v4` using
  `affine/affine/corpus/trace.py` and `view.py`: bake tool dialects, preserve the
  sampled node's actual ancestor path, validate turns, exclude benchmark tasks,
  enforce dialect allowlists and deduplicate. Mix caps target **slice strata** at
  rollout granularity; excess rollouts defer rather than splitting trajectories.
- `corpus/viewpack.py` and `publish.py` publish view chunks/index before the
  immutable schema-3 manifest and pointer. `affine/evalsrv/corpus.py` syncs it;
  the duel samples seed-shuffled strata and materializes prefixes from the pinned
  view. Verdicts stamp manifest/slice provenance. Corpus refresh alone is a data
  event, not permission to change scoring or admitted dialects.

## Advisory, audit and infrastructure branches

- **Bench:** `affine/affine/bench.py` dispatches pending suites to the dedicated
  bench role; `affine/evalsrv/benchrunner.py` / `swerunner.py` run them. Results and
  trajectories persist under `affine/state/` and enter public benchmark views.
  These are advisory, never duel/crown inputs. RT-7 remains open: score/crown does
  not establish coding capability; small task panels are noisy.
- **Chat:** `affine/evalsrv/chatsrv.py` polls the public king snapshot and reloads
  a single engine slot, exposed through `affine/affine/dash/app.py` chat proxies.
  Chat has no scoring/chain role. Its health may remain `ok` while loading, so
  service liveness must be distinguished from generation readiness.
- **Audits:** `ops/exploit-audit/auditd.py` watches crowns and sitting-king audit
  coverage, builds hash-pinned evidence workspaces and publishes verdicts. With
  enforcement enabled, a valid exploit verdict can revert a still-matching king
  and requeue eclipsed challengers. Invalid/missing judge output does not revert.
  This is a separate post-crown enforcement path, not the numeric score.
- **Infrastructure:** `affine/affine/provisioner.py` manages duel/bench/chat rentals,
  tunnels, bootstrap and health/replacement; `affine/evalsrv/bootstrap.sh` starts
  pod services. The teacher swarm has its own manager. Root process supervision
  is configured in `affine/scripts/ecosystem.config.js`; monitoring never runs provision,
  restart, mutation, upload or weight-setting actions.

## Read-only monitoring boundary

`ops/monitoring/validator.py::collect()` returns `validator` and `weights` through
`common.panel`, reading only local `affine/state/state.json`,
`affine/state/public/snapshot.json` and `affine/affine.toml`. It never instantiates
`State`, loads production configuration/secrets, performs RPCs or reconciles files.
Output is an allowlist of bounded numeric aggregates, normalized timestamps,
known phases and static explanatory text; raw identities, references, URLs,
credentials and exception strings are excluded even from public input files.

File publication ages are **heartbeat proxies**, not watchdog beats or proof of
GPU/chain health. Both ages must be fresh against
`max(tick_warn_after_s, 3 × poll_interval_s, 3 × dashboard_flush_min_interval_s)`.
Missing/malformed inputs and future timestamps become Unknown with warnings;
zero is reserved for an observed empty count. Separate file reads are not atomic
across files. Weight age is compared to the configured interval but never promoted
from a recorded success to a monitor-confirmed on-chain payment.
