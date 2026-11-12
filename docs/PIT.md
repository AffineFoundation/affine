# SN120 per-item train+eval (PIT) — serverless

PIT rents one GPU box **on demand** per queued registration. That box trains the
miner's batch with trainer-v834e from the bound head + fp32 master, then runs
the live-gate probe on the same box. The host scores the probe and records
incentives. Only the validator host stays always-on; baseline, bench and prior
are also short rent-per-job boxes (SPEC).

## Hard rule: on-demand rent only

- No warm pool, no pre-rented or idle PIT boxes.
- Empty queue ⇒ **zero** item rentals.
- B queued items with `max_boxes` N ⇒ at most **min(B, N)** concurrent item rentals.
- A box is released as soon as the item's result is stored or the item fails.
- Baseline / bench / prior jobs do **not** count toward the item cap.
- Controller default, not a config option (`warm_pool` / `pre_rent` refused).

## Live defaults (SPEC)

| Knob | Value |
|------|-------|
| `mode` | `shadow` (observe) or `live` (cutover) |
| Item `max_boxes` / `box_cap` | **5** live, **2** shadow |
| GPU | **B300 pinned** for item + baseline (no B200/H200 fallback) |
| Price ceiling | **$56/h** |
| Master upload | **after GREEN only** |
| Lium auto-backup | **disabled** on every rental |
| Trainer fallback | kept **24 h** after live cutover, then release with Jacob OK |
| REDs | **earn nothing** (last-10-GREEN rule) |
| ROOT path | **delegated host key only** (`root_mode=delegated_host`) |

## Dispatch model (Jacob 2026-11-09)

PIT is **decoupled from the lifecycle one-open-epoch model**. Do not hold
lifecycle / controller dispatch waiting for PIT decisions.

- Queued batches go **straight to the host PIT queue** (global box cap N; **5** live).
- Each box downloads the **current head** from the host at spawn, trains the
  batch, and runs the full equivalent eval.
- The **first GREEN** from a given head uploads the new head and becomes **HEAD**;
  boxes spawned after that use it.
- A later GREEN from the old head still uploads, is **paid**, and is marked
  **`stranded` / `merge_blocked`**. It never becomes HEAD.
- Everything else (RED / no_decision / fail-closed) **wipes and is destroyed**.
- Lifecycle only offers a ready verified fresh GREEN via `ready_apply`, or
  returns immediate `no-update` (`pit-decoupled-eN-kept-H`) so weight nonces keep
  moving. k1 / weights **read PIT DECISION.json / HEAD-CHAIN** independently.
- Fail-closed apply, gate math, and draw secrecy are unchanged.

## Rules (short)

1. On dequeue/spawn, bind the item to the **current head** (and its expected fp32 master digest).
2. If `M(H)` is unpublished, dequeue **pauses**.
3. The **first fresh GREEN** from a head becomes the new head (uploads bf16 `C` + `M(C)`, host applies via `pe_apply` / HEAD-CHAIN).
4. Later GREENs bound to an older head are **paid** but **`stranded` / `merge_blocked`**. Non-GREEN boxes destroy their state.
5. Shadow mode never writes `DECISION.json`, weights, `HEAD-CHAIN`, or R2 head pointers.
6. Never promote on a timeout, partial probe, or fail-closed error.
7. Box lost after SEALED ⇒ **full redo** with the same binding (SPEC D3).
8. Lifecycle must **not** claim/wait on per-epoch PIT decisions.

## Gate math (unchanged)

k_cand 8, head rollouts 12, strict z>1, max output 65536, bf16, no fp8, N=1000.
fp32 master checks stay fail-closed. PIT never passes `--fp32-bootstrap-override`.

## ROOT: delegated host key (only live path)

Owner decision (2026-11-09): ROOT signs a **host admission key once**. The host
issues short-lived per-box certs signed by that host key. Verifier chain:

`box payload → host pubkey (admission) → host-key cert → ROOT`

`root_mode=per_round` is kept in code as an **unused alternative** and is
**refused** when `mode=live`. There is **no bootstrap override**.

Never commit or embed ROOT seeds, host private keys, or cert private material.
Operator files stay mode `0600` on the host / air-gapped ROOT machine.

### One-time ceremony (operator)

1. On the validator host, generate a host key (once):

```bash
python3 - <<'PY'
from nacl.signing import SigningKey
from pathlib import Path
sk = SigningKey.generate()
p = Path('/secure/pit-host.key')  # operator-chosen path; never in git
p.write_bytes(bytes(sk)); p.chmod(0o600)
print(bytes(sk.verify_key).hex())
PY
```

2. On the ROOT machine (ROOT seed out-of-band only):

```bash
AFFINE_PIT_ROOT_SEED_FILE=/secure/root.seed \
  PYTHONPATH=/path/to/repo python3 -m ops.parallel_eval.pit.ceremony_sign_host_key \
    --host-pubkey-hex <hex from step 1> \
    --out /secure/pit-host.cert.json \
    --ttl-days 365
```

3. Install on the validator (cert is public; host key stays `0600`):

```bash
export AFFINE_PIT_HOST_KEY_FILE=/secure/pit-host.key
export AFFINE_PIT_HOST_CERT_FILE=/secure/pit-host.cert.json
export AFFINE_PIT_ROOT_PUBKEY_HEX=<root verify key hex>
```

Script: `ops/parallel_eval/pit/ceremony_sign_host_key.py`.

## Sibling jobs

| Job | Trigger | Cap | Notes |
|-----|---------|-----|-------|
| Baseline | new bindable head / round rollover | 2 | Secret draw stays host-private until decision; B300 |
| Bench | each GREEN_FRESH | 1 | Profile `qwen3.5-family-pp0`; merge-only history; coalesce to newest on backlog |
| Prior | each GREEN_FRESH | 1 | Cheap per-head pass; does not block scoring |

## Modules

| File | Role |
|------|------|
| `binding.py` | Dequeue + bind; pause on unpublished master |
| `expected_master.py` | Chain override → manifest → index → TRAINING-RESULT |
| `box.py` / `train_box.py` | Box phases + on-box train asserts |
| `controller.py` | On-demand rent, live/shadow decide+apply |
| `run_host.py` | Live host daemon: batch queue + scheduler, PIT-CLAIM skip marker, DECISION.json |
| `queue_ingest.py` | Inbox / K1 / pending → enqueue; refresh head from HEAD-CHAIN |
| `durable.py` | Item + DecisionLog persistence under `PIT_STATE` |
| `decision_emit.py` | Gate-schema `DECISION.json` for k1 / ff_launcher / lifecycle |
| `apply_live.py` | GREEN_FRESH → verify uploads (fail closed) → pe_apply; stranded paid; else destroy |
| `cutover.py` | Live cutover / revert plan (trainer, warm, lanes) |
| `scripts/cutover_enable.sh` | Flocked enable; TOCTOU trainer-idle; `WARM-OFF`; flag backup |
| `scripts/cutover_revert.sh` | Flocked revert; clear `PIT-CLAIM-e*`; release `affine-pit-*`; restore bak |
| `scripts/affine-pit-host.service` | **Template** systemd user unit (`REPLACE_ME` placeholders only) |
| `cutover_ops.py` | Python helpers used by cutover scripts and tests |
| `ceremony_sign_host_key.py` | One-time ROOT→host-key ceremony (no embedded keys) |
| `baseline_job.py` / `bench_job.py` / `prior_job.py` | Sibling rent-per-job services |
| `decisions.py` | Hash-chained decision log |
| `signer.py` | Ephemeral key + `DelegatedHostSigner` (live) / `FixtureRootSigner` |
| `pe_rent.py` / `pe_lane.py` / `pe_score.py` | Git ports; host RA modules when present |
| `ops/parallel_eval/live/*` | Deploy SOT for live pe_* (hash-sync + early_baseline + PIT hooks) |

`pe_score.score_epoch_shadow` always records the **box** candidate hash, never
the live epoch's `candidate`.

## Shadow note: E1041 (H200 vs B300; noise-tolerant pass)

On E1041 the box produced candidate `36aabc9a…` while live produced
`57033dce…` (z −0.48 vs +0.64). All inputs matched (dispatch, job, config,
fp32 master start, lr, v834e checksums, torch). The remaining difference is
**GPU**: live trains on H200, PIT boxes on B300, with nondeterministic
all-reduce. The z gap sits inside same-model eval noise (per-task
disagreement ratio ≈ 0.94).

Fail-closed wiring still required so a box cannot silently train a different
object:

1. **Batch handle** — live `pe_box` refuses synthetic `batch-handle:<item_id>`;
   ingest must stamp `meta.batch_path`.
2. **fp32 master vs bf16 upcast** — no `--fp32-bootstrap-override`; expected
   master hash must match the bound head.
3. **Trainer binary / hyperparams** — `trainer-v834e`, same public config.
4. **Training report copy-back** — before release the box returns its report
   (source file hashes, batching, tokens, update size, gradient norm); the
   host stores it on the item (`meta.training_report`) and under
   `PE_STATE/pit/training-reports/`.
5. **GPU stamps** — box result and SCORE-vs-live record `train_gpu` /
   `eval_gpu` (and live counterparts).

**Shadow pass criterion** (`ops.parallel_eval.pit.shadow_compare`, also used
by live `pe_score.score_epoch_shadow`): equal input hashes; exact rollouts and
tokens; update size within 1e-3 relative; gradient norm within 5%; same draw,
seed and head baseline; per-task disagreement ratio in [0.8, 1.25];
`|Δz| < 2.0`. Equal candidate hashes / equal z are **not** required.

## Artifact transfer (box → R2 → live prefix)

After training, one box job sends the fp32 master and the bf16 candidate to the
private pending prefix. The host downloads nothing to check them.

1. **Upload.** The host signs one URL for each part of each file (a part is a
   byte range; 256 MiB). The box tool `pe_r2_box_upload.py` sends 16 parts at a
   time, straight from disk, and hashes every file. It returns a manifest: the
   sha256 of each file and the MD5 of each part.
2. **Check.** The host reads the size and the ETag of each object. The ETag of
   R2 is a function of the MD5 of the parts, so an equal ETag shows that R2
   holds the bytes that the box hashed. Each bf16 sha256 must also be the one
   in the checkpoint descriptor. A difference fails the upload (3 tries, then
   the box is kept and the item fails closed).
3. **Promote (APPLY).** The host copies each pending object to the live prefix
   inside R2, in parallel part copies (`UploadPartCopy`), on the condition that
   the pending object still has the ETag of step 2. The live object must have
   that ETag too. Only then is the public index written, with the `meta_json`
   pointer.

When R2 returns an ETag that the host cannot tie to the sha256 (an index
written by the serial transport, or a store that forms ETags in another way),
the host hashes that object as a stream. This is slow and still fail closed;
`downloads` in the upload record counts these objects.

| Variable | Default | Meaning |
|----------|---------|---------|
| `PIT_R2_PART_BYTES` | 268435456 | bytes in one part (5 MiB to 5 GiB) |
| `PIT_R2_WORKERS` | 16 | parts in flight at one time |
| `PIT_R2_READ_TIMEOUT_S` | 900 | seconds one request waits for data |

`PIT_R2_LOCAL_MIRROR` (tests, dry runs) keeps the serial transport.

## Deploy file placement (RA snapshot; never `git checkout` live)

Services do **not** run from the git branch. Deploy by copying reviewed files into
an RA release snapshot (example: `RA/20261109-serverless/release/`), then point a
systemd user unit / drop-in at that snapshot. Keep `~/subnet120-rewrite/.venv` as
the interpreter.

**Authoritative file list:** `ops/parallel_eval/DEPLOY-MANIFEST.txt` — copy every
path listed there. The bundle **must** include PR `live/pe_apply.py`,
`pe_lifecycle.py`, `pe_warm.py`, `pe_controller.py` **and** the new host seams
`pe_box.py`, `pe_verify.py` / `pe_upload_verify.py`, `pe_round_seed.py` alongside
`ops/parallel_eval/pit/**`. Host `code/` copies without these hooks are not
cutover-ready.

| Repo path | Live destination | Notes |
|-----------|------------------|-------|
| `ops/parallel_eval/live/pe_apply.py` | Prefer **additive** `accept_pit_decision` onto the live RA `pe_apply.py` (backup first). Do **not** replace the live file blindly — it must keep `_sync_expected_hash_after_head_apply`. | E1028 fix |
| `ops/parallel_eval/live/pe_gate_hook.py` | RA PE code dir (must keep `early_baseline`) | throughput-v4 |
| `ops/parallel_eval/live/pe_live.py` | RA PE code dir | PIT dispatch noop + PIT-CLAIM skip (no PE-lane double-dispatch) |
| `ops/parallel_eval/live/pe_lifecycle.py` | RA PE code dir | `pit_cutover()`: never holds; ready_apply or immediate `pit-decoupled` no-update |
| `ops/parallel_eval/pit/**` | `RA/20261109-serverless/release/ops/parallel_eval/pit/` | host library + daemon |
| `docs/PIT.md` | same snapshot `docs/PIT.md` | operator runbook |

Do **not** commit systemd units with host IPs / `known_hosts`. Template units with
placeholders are OK. Do **not** commit key material.

Shared state (do not migrate):

- `PE_STATE` (`RA/20261106-parallel-cutover/state`) — `GREENS.json`, `HEAD-CHAIN.json`, `apply.lock`
- Gate epochs — `RA/20261026-promotion-gate/state/epochs/<e>/DECISION.json` (PIT emits gate schema)
- `PIT_STATE` — default `$PE_STATE/pit` (items, claims, DECISIONS.json)

## Operator: shadow mode

```json
"parallel_eval": {
  "pit": {
    "enabled": true,
    "mode": "shadow",
    "max_boxes": 2
  }
}
```

Confirm: decision log under staging; **no** `DECISION.json` / `HEAD-CHAIN` / R2
head writes; rentals drop to 0 when the queue is empty.

## Operator: live cutover

Preconditions: trainer **idle** (no `TRAINING-JOB` without `TRAINING-RESULT`);
Lium runway ≥ 24 h; ROOT ceremony done; `verify_uploads` wired fail-closed.

1. Shadow pass criteria met (DESIGN §10 / SPEC §7 step 0).
2. ROOT ceremony complete; env vars `AFFINE_PIT_HOST_KEY_FILE`,
   `AFFINE_PIT_HOST_CERT_FILE`, `AFFINE_PIT_ROOT_PUBKEY_HEX` set on the host.
3. Enable at an epoch boundary (trainer idle re-checked **after** acquiring
   `$PE_FLAG.lock` or `$PE_STATE/cutover.lock`):

```bash
export PE_FLAG=/path/to/PROMOTION-GATE-or-PE-flag.json
export LIFE_ROOT=/path/to/training-lifecycle-root  # CURRENT.json + epochs/ (required)
export PE_LIFE=$LIFE_ROOT                          # alias accepted by queue_ingest
export PE_LIFE_STATE=/path/to/pe-life-state        # pending/ + apply/ (distinct; required)
export PE_STATE=/path/to/parallel-cutover/state    # WARM-OFF + GREENS/HEAD-CHAIN
# pe_apply hash-sync after HEAD apply (required at cutover; unset ⇒ hook no-op in tests only):
export PE_SYNC_EXPECTED_HASH=/path/to/sync_expected_hash.py
# optional: PE_SYNC_EXPECTED_HASH_PYTHON / PE_SYNC_EXPECTED_HASH_LOG / PE_SYNC_EXPECTED_HASH_ALERT
export AFFINE_PIT_HOST_KEY_FILE=/secure/pit-host.key
export AFFINE_PIT_HOST_CERT_FILE=/secure/pit-host.cert.json
export AFFINE_PIT_ROOT_PUBKEY_HEX=<hex>
# On-box train + GREEN R2 upload (never commit these host paths):
export PIT_BUNDLE_JSON=/path/to/BUNDLE5090.json          # must include trainer {key,sha256,bytes}
export PIT_FP32_MASTER_INDEX=/path/to/k1/fp32-master     # published M(H) index (head_set)
export PIT_LIFE_CFG=/path/to/lifecycle.ACTIVE.private.json  # bucket for scoped put_object
export PIT_V834E_RELEASE=/path/to/release-v834e-signed
# Exact pe_controller PID only (no cmdline pattern kill):
export PE_CONTROLLER_PIDFILE=/path/to/pe-controller.pid
# PYTHONPATH must resolve `subnet` AND the release `ops/` — do not shadow ops/:
export PYTHONPATH=/path/to/release:/path/to/subnet120-rewrite
bash ops/parallel_eval/pit/scripts/cutover_enable.sh
# Preflight: PE_FLAG + LIFE_ROOT + PE_LIFE_STATE + PE_STATE + PE_SYNC_EXPECTED_HASH*;
# import subnet + ops; refuse mid-probe lanes; copies $PE_FLAG -> bak;
# writes PE_STATE/WARM-OFF ({}); stops pe_controller by exact PIDFILE only.
# then install scripts/affine-pit-host.service (fill REPLACE_ME) and start the unit.
```

**Required live env (host-local; never commit paths):**

| Variable | Purpose |
|----------|---------|
| `PE_FLAG` | promotion / PE flag JSON (lifecycle + cutover) |
| `LIFE_ROOT` / `PE_LIFE` | training lifecycle root (`CURRENT.json` + `epochs/`); **required** — queue_ingest floor; never `PE_LIFE_STATE` |
| `PE_LIFE_STATE` | PE life-state dir (`pending/` + `apply/`); **required** — distinct from `LIFE_ROOT` |
| `PE_STATE` | PE state (GREENS / HEAD-CHAIN / pit uploads); **required** — no host default |
| `PE_SYNC_EXPECTED_HASH` | `sync_expected_hash.py` for pe_apply head-apply hook (+ optional `*_PYTHON` / `*_LOG` / `*_ALERT`) |
| `PIT_BUNDLE_JSON` | train-cap-64k `BUNDLE5090.json` with `trainer` entry |
| `PIT_FP32_MASTER_INDEX` | public `k1/fp32-master/<head>.json` (queue_ingest `head_set`) |
| `PIT_LIFE_CFG` | lifecycle ACTIVE.private.json (`bucket`) for GREEN R2 PUT |
| `PE_CONTROLLER_PIDFILE` | exact pe_controller PID (refuse cmdline pattern kill) |
| `PYTHONPATH` | resolves `subnet` + release `ops/` (do not shadow `ops/`) |

Equivalent flag:

```json
"parallel_eval": {
  "enabled": true,
  "pit": {
    "enabled": true,
    "mode": "live",
    "cutover": true,
    "max_boxes": 5,
    "master_upload": "after_green",
    "root_mode": "delegated_host"
  }
}
```

4. Effect: trainer dequeue **off** (pod kept 24 h as hot fallback); warm pool
   **off** (`WARM-OFF` + `pe_warm` / `pe_controller` no-op during PIT cutover);
   standing PE lanes **off**; host PIT queue/scheduler owns batches (cap 5);
   lifecycle never waits on PIT; first fresh GREEN uploads and becomes HEAD;
   later old-head GREEN paid+stranded; REDs earn nothing; `DECISION.json`
   emitted for k1 / website / weights.
5. After 24 h of healthy live: release the trainer pod with Jacob OK.

## Operator: revert (one command)

```bash
export PE_FLAG=/path/to/PROMOTION-GATE-or-PE-flag.json
export PE_STATE=/path/to/parallel-cutover/state
export PE_LIFE_STATE=...   # optional; claims mirrored here
export PIT_STATE=$PE_STATE/pit
# RESTORE_FLAG_BACKUP=1 restores latest $PE_FLAG.bak-pre-pit-* when present
bash ops/parallel_eval/pit/scripts/cutover_revert.sh
```

Stops `affine-pit-host` if active; clears all `PIT-CLAIM-e*.json` under configured
state roots; best-effort `release` of `affine-pit-*` via `pe_rent` / `lium`;
removes `WARM-OFF`; sets `pit.revert=true`, `enabled=false`, `cutover=false`.
Restores trainer dequeue, warm pool and standing lanes. Re-rent standing pods
with `replace_pod.py` if they were released (≤ $35/h or prior price + 10%).

## Tests

```bash
PYTHONPATH=. python3 -m unittest \
  ops.parallel_eval.pit.test_state_machine \
  ops.parallel_eval.pit.test_binding \
  ops.parallel_eval.pit.test_decisions \
  ops.parallel_eval.pit.test_signer \
  ops.parallel_eval.pit.test_ceremony \
  ops.parallel_eval.pit.test_decision_emit \
  ops.parallel_eval.pit.test_durable \
  ops.parallel_eval.pit.test_expected_master \
  ops.parallel_eval.pit.test_ondemand_rent \
  ops.parallel_eval.pit.test_pe_score \
  ops.parallel_eval.pit.test_shadow_controller \
  ops.parallel_eval.pit.test_box_fake \
  ops.parallel_eval.pit.test_live_controller \
  ops.parallel_eval.pit.test_apply_live \
  ops.parallel_eval.pit.test_cutover \
  ops.parallel_eval.pit.test_cutover_scripts \
  ops.parallel_eval.pit.test_baseline_job \
  ops.parallel_eval.pit.test_jobs \
  ops.parallel_eval.pit.test_lifecycle_inflight \
  ops.parallel_eval.pit.test_queue_ingest
```
