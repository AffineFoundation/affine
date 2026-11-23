# KL update gate: operator deployment and rollback

This kit does **not** deploy anything. It documents an operator-authorized,
between-round change to the existing v2 runner. It must not change the sole
hourly best-assessment writer, its transaction cursor or UID mapping, its
six-hour contribution half-life, or its last-valid-assessment behavior. Do not
start an archived service, enable burn/owner-only fallback weights, or rewrite
journal, manifest, result or optimizer history.

## Reviewed identities

| item | value |
|---|---|
| base / rollback commit | `917ad5484ff38ba96924fb7631fcad79720151f2` |
| old params hash | `073ef680cf338de5e085100704e60f80ba96657d6d4e19d12995e4cb971ef6f8` |
| new params hash | `98d2e1eeaa4106a1e59a21a2f733242fde30a2dd90caaf7d5fa892c002259814` |
| old GPU code digest | `22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1` |
| new GPU code digest | `5e3aef5849ef6903c9fc1ab67f89eddbd2ea2cac9748621fc42b75224bf58419` |
| new recipe id | `b1f7cda18368657aebf33ee5097244ca41cccd6bcee6713459799680a6ef2d44` |
| unchanged image digest | `9a4b16ff282b21baf44350e54c852ed59586b9efb6fe09e5cd2f1e288f4c108a` |

The complete effective diff against the live params is:

```text
min_kl_seq: 0.0 -> 0.000225
recipe.code_digest: 22b2d191b32e394d4621ce477e702b50410a6ef7706cce6cfd20c60ff64e42e1
                 -> 5e3aef5849ef6903c9fc1ab67f89eddbd2ea2cac9748621fc42b75224bf58419
```

`kl_rollouts` is explicitly `128` in the new raw file but is already the default
on both sides. `kl_max_tokens=0` retains full completions; setting it above zero
scores only that many leading completion tokens, without adding an end token at
the cut. `kl_time_limit_s=1200` is an independent hard cap. All four fields are
omitted from canonical JSON at their defaults, preserving the exact old live hash
and old sealed-record compatibility. This params file changes only the floor and
code digest; it does **not** turn on a completion-token cap.

## Preconditions and copied-state rehearsal

Use the **actual operator-selected state/work/Python/environment**, not guessed
paths or a second writer. Preserve the existing R2, model, systemd and loop
settings. Read the current unit to obtain `STATE`, `WORK`, `PY` and `STACK`; the
production stack is `sn120_v2.assembly:stack`. The new checkout must be the
reviewed commit on `bb/kl-gate`, with a clean source tree. The deployment
operator must independently authorize the merge/checkout and go-live boundary;
this branch does not authorize automatic deployment.

```bash
# Set these from the operator environment/current unit, without printing secrets.
NEW_V2=/OPERATOR-SELECTED/reviewed-kl-checkout/v2
STATE=/OPERATOR-SELECTED/validator-state
WORK=/OPERATOR-SELECTED/run-work
PY=/OPERATOR-SELECTED/persistent-python
STACK=sn120_v2.assembly:stack
KIT="$NEW_V2/tools/kl-gate-deploy"

"$PY" -I -m sn120_v2.gpu.plan  # code digest must equal the new value above
"$PY" -I -m sn120_v2 status --state-dir "$STATE"  # inspect; no writer takeover
```

The local state directory alone is not necessarily a complete store copy. For
a disconnected simulation, copy **both** its local journal and the needed local
store data, or use a purpose-built local simulation store. Never point a
simulation stack at the live store. Never copy live state while its writer is
active. The following verifies the new file without accessing a journal:

```bash
PYTHONPATH="$NEW_V2" "$PY" -I - <<'PY'
import json
from pathlib import Path
from sn120_v2.params import load_params
p = load_params(Path('/OPERATOR-SELECTED/reviewed-kl-checkout/v2/ops/train64k_params_t3600_prune_open_kl.json'))
assert p.params_hash == '98d2e1eeaa4106a1e59a21a2f733242fde30a2dd90caaf7d5fa892c002259814'
assert p.recipe_id == 'b1f7cda18368657aebf33ee5097244ca41cccd6bcee6713459799680a6ef2d44'
print(json.dumps({'params_hash': p.params_hash, 'recipe_id': p.recipe_id}))
PY
```

With the loop stopped at the closed-round boundary below, create a new isolated
state directory containing a **copy** of the local journal and local store, and
rehearse the exact real CLI against that copy only:

```bash
COPY=/OPERATOR-SCRATCH/kl-copied-state       # new operator-owned copy
"$PY" -I -m sn120_v2 set-params --state-dir "$COPY" --stack "$STACK" \
  --params "$NEW_V2/ops/train64k_params_t3600_prune_open_kl.json"
```

Require `params_hash` to be the new hash and `changed` to contain exactly the
two effective fields above. If the copied stack has remote dependencies, use an
isolated local store and an explicitly non-live chain/entropy fixture; do not
substitute a remote production stack for a simulation. Then independently
replay the copy and inspect the original journal hash to prove it did not
change. The included `qualify.py` exercises the old-code compatibility behavior
on synthetic state, not this production-copy rehearsal.

## Exact between-round go-live

1. Confirm the existing runner is healthy and there is exactly one loop. Leave
   the existing weights writer untouched.
2. Request the real stop-after-round and wait for the runner's own success
   message (`STOPPED after the round, as asked`). Do not kill it during a seal
   or promotion:

   ```bash
   touch "$WORK/logs/run-loop.stop-after-round"
   # Monitor $WORK/logs/run.log and run-loop.status until stopped.
   ```

3. Confirm status has no open round and the stop request was removed. Ensure no
   other controller takes the journal while stopped.
4. Check out the reviewed KL commit and confirm the GPU code digest exactly.
   Do not install dependencies or replace the image.
5. Run the copied-state `set-params` simulation above and verify the exact
   expected hash and two-field effective diff.
6. Only after explicit go-live authorization, journal the same complete file
   once through the real CLI:

   ```bash
   "$PY" -I -m sn120_v2 set-params --state-dir "$STATE" --stack "$STACK" \
     --params "$NEW_V2/ops/train64k_params_t3600_prune_open_kl.json"
   ```

7. Verify `status` reports the new params hash and no open round. Start exactly
   one runner using its existing unit/arguments and environment. Do not invent
   a second service or change the hourly writer.
8. Watch the first complete gated round. Check each entry's sealed/public
   `kl_seq`, `kl_seq_se`, `kl_tok`, `kl_k`, round-level `null_kl_*`, charged
   below-floor `null_update`, uncharged measurement-failure `no_decision`, and
   the `kl` and `kl_gpu` records in `phases.jsonl`. Confirm all eight GPUs carry
   candidate shards, and inspect accepted tokens/s against the backtest above.
   A measurement failure or timeout must never win. Inspect the published
   `llms.txt` and `mining.json` rule before announcing activation.

## Timing and warm-up

The operator's H100 backtest, with the causal-conv1d fast path missing, measured
about 2.7k model-prefill tokens/s per H100 for two models on roughly 30k-token
rollouts. That implies roughly 15–20 GPU-minutes per entry for 128 rollouts on
B300 if the slow path remains. Entries and their rollout shards now run in
parallel across all eight GPUs. Each model load handles at most 16 rollouts;
the log-only null calibration runs in its own worker with a resident head.

The hard cap is the earlier of `kl_time_limit_s` (1,200 seconds) and the
promotion deadline minus 60 seconds for sealing. With a fully used round, the
current 10,800-second limit and 600-second reserve leave only about 540 seconds
after scoring. A timeout stops every child; any incomplete entry becomes
uncharged `NO_DECISION` and cannot win. Neither cap is a measured runtime
guarantee. No B300 KL measurement has been run, so the ten-minute target remains
unqualified. If the first timing is too slow, change `kl_max_tokens` through a
reviewed params update rather than another code change; that changes what the
floor means and must be announced accordingly.

`phases.jsonl` includes an overall `kl` wall-time record and a `kl_gpu` record
per worker with `wall_s`, accepted completion `tokens`, `tokens_s`, and
`model_tokens` (two forwards per accepted completion token). Timed-out work is
not counted as accepted throughput. `kl_entry` and `kl_null` include unrounded
Python-float64 metrics. Sequence sums and shard aggregation use float64;
log-softmax/probabilities stay fp32 as specified. Invalid negative or non-finite
values fail closed rather than being clamped.

The serving replies do not preserve exact token IDs. Measurement therefore
uses `head chat-template(prompt) + head-tokenizer(completion_text)`, restoring
`<|im_end|>` for uncapped replies, and teacher-forces the exact tested merged
candidate and resident head. Output-layer/log-softmax chunks bound vocabulary
memory; the trunk still handles the full teacher-forced sequence.

No model or training-data warm-up is needed for the rule itself. Trainer FLA
warm-up already belongs to the current 64k deployment. The KL path reuses that
loading environment, but inference-only kernels have not been independently
benchmarked or pre-tuned. Do not run a GPU warm-up while eval/training owns the
cards; if a cold first round shows compilation overhead, schedule an isolated
follow-up benchmark rather than delaying this deployment automatically.

## Rollback: finish KL work first, then exact old code and params

Bare `917ad548` cannot read KL params or sealed KL fields. Use the verified
compatibility loader. It validates the manifest and exact old GPU digest,
replays the exact KL pin into old effective behavior **in memory**, and never
rewrites historical journal/result bytes. Unknown newer params are refused.

1. With the reviewed **new** checkout, stop after the round and let it finish
   every open/decided KL round. Do not discard a sealed result to force a
   rollback. An open KL round is always refused by old-code commands before a
   leader record can be appended.
2. Create an exact clean checkout of `917ad548`; set `KL_OLD_V2` to its `v2`
   directory. Preserve the kit separately. `KL_ROLLBACK_PY` must be the same
   persistent interpreter/environment used by the old runner, not a temporary
   `uv run` interpreter.
3. Verify the old file, then journal the complete old pin through this loader:

   ```bash
   OLD_V2=/OPERATOR-SELECTED/exact-917ad548/v2
   export KL_OLD_V2="$OLD_V2"
   export PYTHONPATH="$OLD_V2"
   export KL_ROLLBACK_PY="$PY"

   "$PY" -I "$KIT/rollback.py" --kit "$KIT" verify-params \
     "$NEW_V2/ops/train64k_params_t3600_prune_open.json"
   "$PY" -I "$KIT/rollback.py" --kit "$KIT" set-params \
     --state-dir "$STATE" --stack "$STACK" \
     --params "$NEW_V2/ops/train64k_params_t3600_prune_open.json"
   ```

   Require old hash `073ef680…` and only the reverse two-field diff. Only the
   explicit `set-params` command may cross a closed KL state to this pin. An
   ordinary old `run`, `end-round`, or loop remains refused while KL is pinned.
4. Confirm `rollback.py status --state-dir "$STATE"` reports the old hash. Start
   exactly one old loop through this kit:

   ```bash
   SN120_WORK_DIR="$WORK" KL_OLD_V2="$OLD_V2" KL_ROLLBACK_PY="$PY" \
     bash "$KIT/rollback-loop.sh" --state-dir "$STATE" --stack "$STACK"
   ```

   Use the operator's normal process supervision and full existing environment;
   this example is not a request to launch it from this checkout. The loop is
   the exact old loop except that `status`, `end-round`, and `run` all use the
   compatibility loader. (The original loop passed extra arguments only to
   `run`.)
5. Restore/read through the operator's existing validator/weights package
   procedure before restarting the original hourly writer. Do not create a new
   writer or change its state, cadence, cursor, UID mapping or fallback policy.

`rollback.py` covers command-line replay only; it is not a physical-package
replacement for a `python -m` systemd weights unit. If the validator/weights
reader shares the changed source package, restore that complete package using
the operator's reviewed package-restoration route as well. The loader must not
be treated as authorization to finish an open KL ticket. `rollback-loop.sh`
exports `KL_OLD_V2` on `PYTHONPATH` for its children; preserve that line.

## Disposable qualification

The included test uses only synthetic CPU state. It proves closed-KL replay,
active-pin loop refusal, exact old-pin transition and repeat no-op behavior,
unchanged historical journal/mirror bytes, old-loop stop-after-round, and
open-KL-round refusal with no appended leader. It does not prove a production
store, chain interaction, service restart, or GPU runtime.

```bash
export KL_OLD_V2=/OPERATOR-SELECTED/exact-917ad548/v2
KL_QUALIFY_PY=/OPERATOR-SELECTED/persistent-python \
  uv run --project v2 --python 3.11 --with boto3 \
  python -I v2/tools/kl-gate-deploy/qualify.py
```

`manifest.json` pins every compatibility overlay file. Regenerate it only after
reviewed overlay changes, then rerun this qualification and the targeted tests.
