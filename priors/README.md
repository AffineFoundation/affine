# SN120 prior table + prior-weighted baseline (live as the gate baseline prior-v1 from 2026-11-01 17:14 CR, epoch 497 (first round r20261101T231445Z-fb048812))

Jacob pt988u / pt989u, 2026-11-01. A dedicated pod keeps sampling the **current head** on an *active set* of math environments and
keeps a per-task prior table. From that table the held-out baseline can be drawn by weighted random sampling (no hard filters),
refined with 12-16 head rollouts per drawn task, and used as the gate bar for miners. **The live gate/probe is unchanged**:
nothing here is imported by `gate_service.py`, the probe, the trainer, `llms.txt` or `mining.json`. Switching waits for Jacob.

Privacy (Jacob pt1019u, 2026-11-01): the gate pool (affine_math + dapo) is **public** (`miner/data/gate-pool-v1.jsonl`), and so is
all code, including how the prior is formed. The prior table VALUES (the sqlite, per-task counts, the private status) and the
private perf-eval holdout stay private: they live under `state/root-audits/20261101-prior-table/private/` (0700/0600) and on the
prior pod only. Never commit or publish them.

## Pieces

| file | runs on | what |
|---|---|---|
| `pod/serve.sh` | prior pod | one vLLM 0.32 replica per GPU (ports 8100+g), same flags as the live probe `serve.sh` (bf16, max-model-len 16384 via serve.env since the 12K switch, prefix caching, max-num-seqs 512, gmu 0.90); kills only PIDs it recorded |
| `worker.py` | prior pod | continuous sampler: pass after pass over the active set, `k` (4) rollouts per task per pass, private shuffle per pass, appends one line per task to `grades.jsonl` |
| `grader.py`, `allan_src.py` | prior pod | the live affine_math grade: native final-channel projection + math_verify, then Allan's lenient fixes (`allan-lenient-v1-786cdb9`, byte-identical to the live probe worker's `_ALLAN_SRC`, sha256 pinned); capped replies never count |
| `tasks.py` | both | active-set loader; ids are env-prefixed (`math:5442`, `dapo:123`); both envs use the affine_math system prompt + bare problem |
| `controller.py` | validator host (systemd --user `affine-prior-table.service`) | follows the gate head (`promotion-gate/state/status.json`), presigns R2 GETs on the host and materializes the checkpoint on the pod (`gate_fetch.py`, sha256-checked), re-serves, discounts the table, pulls grades into the table, writes a private status |
| `table.py` | host | SQLite prior table (schema below) |
| `selection.py` | host | Beta-posterior weights, per-env priors, stratified PPS draw with inclusion probabilities, seed + commitment |
| `refine.py` | host | baseline plan (blocks x k), merge, write-back, scoring with the live `heldout.score_ivw` (+ Hajek gain) |
| `dryrun.py` | host | draws a set from the table and simulates what the gate would look like vs today's |
| `dapo/build_dapo.py` | box/host | builds the pinned, decontaminated DAPO pool |
| `baseline_v1.py`, `gate/` | host | the prior-v1 gate baseline (staged, not live): see `gate/README.md` |

Sampling settings = the live probe harness for affine_math: temperature 1.0, top_p 0.95, top_k 20, thinking on, checkpoint
`chat_template.jinja`, max output 12288, context 16384 (= the gate task cap; 8192 / 12288 before the 2026-11-02 12K switch; pod: serve.env MAX_MODEL_LEN=16384), EOS ids 248046/248044, one request per sample with its own seed.

## Prior table

One row per task: `task_id, env, s, n, b` (effective successes / trials / cap hits, floats), `head` (head the fresh counts come from),
`s_head, n_head, b_head` (raw counts from that head only), `disc_from, discounts, last_seen`; plus a raw `cells` ledger of every ingested
record per head. On a head change (GREEN or rollback): `s, n, b *= discount` (default 0.5) and the per-head raw counts restart at 0, so
old-head evidence still informs the draw but with wider uncertainty, and the new head overwrites it within one pass (the worker also
starts a fresh pass on a new head). Records arriving late from an old head are added with the discount.

## Selection weight (accounts for the sample count)

Per task, Beta posterior `a = s + a0_env`, `b = (n - s) + b0_env`; weight = expected Bernoulli variance

    w = E[p(1-p)] = a*b / ((a+b) * (a+b+1))

Under Beta(1,1): 0/0 -> 0.167, **1/1 -> 0.167** (not p=1), 8/16 -> 0.235, **16/16 -> 0.050**, **64/64 -> 0.015**, 0/16 -> 0.050.
Per-env prior `(a0, b0)`: Beta(1,1) by default (`prior="uniform"`), or empirical Bayes (`prior="eb"`, default): method of moments on
the env's tasks with n >= 4, `m = mean(s/n)`, `var_p = var(s/n) - mean(p(1-p)/n)`, `kappa = m(1-m)/var_p - 1` clipped to
[0.5, 4] (shrunk: 4 own trials already outweigh it), `a0 = m*kappa`, `b0 = (1-m)*kappa`. A new task then starts at its env's mean.

## Draw

Fixed size `n` (default 1000), stratified by env: each env gets at least `min_share*n` (default 0.3 each), the rest in proportion to
the env's weight mass. Inside an env: probability-proportional-to-w without replacement (systematic PPS on a seeded random order;
`pi_i = min(1, c*w_i)`, `sum pi = n_env` exactly). No hard filter: every task has `pi_i > 0`; `pi_i` is stored per drawn task, so
Horvitz-Thompson / Hajek (`1/pi`) weighting gives population (uniform-over-tasks) estimates. Seed = `os.urandom(32).hex()`
(secret), commitment = `sha256("sn120-prior-select-v1|<seed>|<comma-joined sorted task ids>")` (gate_core rule, string ids).
Record the table digest (`PriorTable.digest()`) privately with the draw.

## Refine + gate bar

The head runs S `blocks x k_block` times (default 4 x 4 = 16; plan seed per block `sha256("<seed>:prior-baseline:<b>")`). The merged
cells are (a) written back into the prior table (`source='baseline'`) and (b) the bar: candidate C runs `k_cand` per task on S minus its
training batch with block-1 seeds (CRN) and is scored by the live `heldout.score_ivw` unchanged, with the draw-time prior cell of
each task in the role of today's selection block (`q = (c+1)/(n+2)`, independent of the baseline runs) and no `0<c<n` filter.

## Ops

    # host
    systemctl --user status affine-prior-table.service
    cat state/root-audits/20261101-prior-table/private/table/STATUS.private.json     # throughput + per-env table stats
    .venv/bin/python -B priors/dryrun.py <table.sqlite copy> <out.private.json> --reps 200
    # pod (state/root-audits/20261101-prior-table/pp.sh)
    cat /opt/prior/runs/sampler/progress.json
    touch /opt/prior/runs/sampler/STOP   # graceful stop; remove STOP only after the worker exited

Tests: `python -m pytest priors/tests` (grader test uses the private pool if `PRIORS_DAPO_POOL` is set).

## Switching live (needs Jacob's word)

1. In the gate's round open, replace `G.draw_sample(seed, pool, n)` by `selection.draw(table.rows(active_envs), seed, cfg)` and the
   commitment by `selection.commitment`; store `cells` (pi, prior s/n) and the table digest in `ROUND.private.json`.
2. Baseline: run `refine.plan(...)` blocks on the probe pod (probe must load the dapo env: tasks file + the same grader), merge,
   `refine.writeback` into the table, skip today's selection block / kept filter.
3. Turns: `refine.score(draw, base, cand, batch_ids, ...)` instead of `heldout.score_ivw(sel, ...)`; re-pool `sigma_run` on the new
   scale (the weighted mean now covers all drawn tasks, so today's 0.022 floor is too large for it).
4. Publish rules: reveal the seed, ids, `pi`s and prior cells with the round reveal (same lag rule as today).
