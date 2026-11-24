# Offline re-check of an infra-failed crown confirmation

This is an **offline diagnostic**, not a settlement or deployment tool. It does
not write the journal, change weights, publish a result, or start/stop a
production service. A `confirmed: true` verdict is evidence for an operator; it
does not itself pay the crown.

Approved re-check pairs:

| UID | Crown | Confirmation |
| --- | --- | --- |
| 738 | `r000114` | `r000115` |
| 922 | `r000115` | `r000116` |
| 112 | `r000116` | `r000117` |
| 876 | `r000118` | `r000119` |

The confirmation rounds' historical parent prefixes include `8a7f7e55...`,
`64a25282...`, `17a1ec50...`, and the parent sealed into `r000119`. Do not
choose a base from these abbreviated names: the command uses the full parent
`HeadManifest` in the selected round's sealed ticket.

## What is preserved

* Export replays a **temporary copy** of the journal. It never constructs a
  journal writer, takes `journal.lock`, pulls the store, or writes beneath
  `--state-dir`. It reads the sealed ticket/result objects, including the
  settled result digest, and the winner's own training batch.
* The bundle contains the exact post-exclusion draw, its weights and snapshot,
  all new-head null outcome strings, full parent manifest, crown params,
  winner-file bytes/prompts/rollouts, original infra failure and confirmation
  resolution, and a SHA-256 of its canonical body. Loading verifies those
  duplicated identities against the sealed ticket/result bytes; recomputing the
  outer hash does not make a modified outcome valid.
* Scoring constructs the real `VllmEvaluator`, prepares the verified historical
  parent base with the live `FoldInputs` compatibility path, applies the live
  solo client budget, and calls `VllmEvaluator.score_parent(ticket, deadline=0)`.
  Parent sample **k** uses the seed of new-head null sample **k** for every
  task. The sealed draw is never redrawn.
* The live `judging.score_confirmation(weights, new_outcomes, parent_outcomes)`
  decides the result. **`delta > TIE_TOLERANCE` confirms; an exact tie fails;
  z and SE are diagnostic only.** All null outcome strings are used.
* UID 738's historical one-parent-sample crown is deliberately scored under
  the live two-sample contract. Its verdict includes:
  `crown round used 1-sample design; scored under live 2-sample contract`.
  The verdict separately reports the number of sample strings and the number
  of task outcomes in each string. It records the historical and checkout GPU
  code identities; an earlier crown pin is not misrepresented as the scorer's
  code identity.

## 1. Install

Run all commands from the repository root. Use the commit under review for
both the production-side read-only commands and the scoring host.

Production export only needs the control plane and the R2 reader if there is
no complete local object mirror:

```bash
uv venv v2/.venv --python 3.12
uv sync --project v2 --extra dev --extra r2
```

The scoring host needs the complete pinned worker dependencies. The project
pins `vllm==0.32.0`; do not substitute the CPU-only torch used for tests:

```bash
uv venv v2/.venv --python 3.12
uv sync --project v2 --extra dev --extra gpu --extra r2 --extra envs
uv run --project v2 --no-sync python -m sn120_v2.gpu.plan
```

For the reviewed base the `code_digest` must be:

```text
d46b54e9891355fef71becb3b860c0e355804cb414353a11e66966ac2454c5d8
```

The scoring host needs eight compatible GPUs, free ports 8100--8107, enough
RAM/disk for the parent model, and no other process using its GPU memory. The
existing v2 smoke guide's 300 GB disk / 64 GB RAM starting point applies;
model download time and peak disk use depend on the sealed parent. No store
credential, wallet, chain endpoint, or production state directory is needed on
this host when `--links` is supplied. Older MiMo parents may also need the
pinned small Hugging Face metadata files through `FoldInputs`; no extra weight
files are downloaded by that compatibility step.

## 2. Read-only export on the production box

Set `STATE` to the validator state directory and `ROUND` to a **confirmation**
round, not its crown. Supply the ordinary production store reader environment
if `$STATE/store` is not a complete mirror. Do not paste credentials into a
command, shell history, output, bundle, log, commit, or PR.

```bash
STATE=/path/to/validator-state
ROUND=r000115
OUT=$PWD/recheck-$ROUND
mkdir -p "$OUT"
python -m sn120_v2.recheck export \
  --state-dir "$STATE" --round "$ROUND" --out "$OUT/bundle.json"
```

The console reports only the round, bundle hash and output path. A bundle can
contain training prompts and rollout text; treat it as operator evidence, not
as a public link file. Repeat for `r000116`, `r000117`, and `r000119`, using a
separate output directory for each.

Export is deliberately supported only for a matching unresolved,
infrastructure-failed parent confirmation with sealed two-sample null strings.
It refuses a wrong round, changed seal, missing parent manifest, incomplete
null evidence, or malformed bundle rather than substituting other evidence.
The original winner batch is included and hash-verified when available. The
sealed ticket's draw remains authoritative even when that optional best-effort
exclusion record is unavailable.

## 3. Issue historical parent links on the production box

The public `https://affine.io/public/read_links.json` is **not sufficient**:
it covers the current published head, not an arbitrary historical parent.
Issue a private links document for the selected round using the production
store's normal environment credentials:

```bash
python -m sn120_v2.recheck links \
  --state-dir "$STATE" --round "$ROUND" --out "$OUT/links.json" --ttl 21600
```

This reuses `headchain.file_set`, the builder behind the site's read-links
machinery. It uses the authoritative R2 store (never a possibly incomplete
local cache), first checks every required base object, signs only that parent's
bf16 base, and writes a `sn120-v2-read-links-1` document. `master` is `null` and
`loras` is empty: neither is needed for parent confirmation.

**Every presigned URL is a secret.** The command never prints, logs or embeds
one in its console output. It writes URLs only to the links file, created with
mode `0600`. Missing objects are reported by manifest file name only. Other
store failures use a generic message so an exception cannot disclose a signed
URL or credential. Never paste the links JSON into a ticket, chat, terminal,
CI log or PR.

Copy the bundle and links to the scoring host with an authenticated encrypted
channel. For example, from the scoring host (replace `production-host` and the
operator paths):

```bash
mkdir -p "$OUT"
scp production-host:"$OUT/bundle.json" "$OUT/bundle.json"
scp production-host:"$OUT/links.json" "$OUT/links.json"
chmod 600 "$OUT/links.json"
```

Do not use a public artifact bucket, an unencrypted copy service, or a shared
world-readable directory for `links.json`. Issue a fresh file after its six-hour
TTL rather than extending or publishing the old one.

## 4. Small sanity run, then full score on eight GPUs

A prefix run tests the installation, exact draw prefix, parent loading and
scoring plumbing. It uses the first N tasks, the corresponding prefix of every
null outcome string, and the matching prefix of every weight:

```bash
python -m sn120_v2.recheck score \
  --bundle "$OUT/bundle.json" --work-dir "$OUT/work" \
  --n-tasks 8 --gpus 8 --links "$OUT/links.json" --out "$OUT/sanity.json"
```

A sanity verdict always has `"sanity": true` and
`sanity prefix only; not a full-draw confirmation`. It is **not** payment
evidence. Check the model files, vLLM logs, `parent_eval` statistics and code
identity before launching the full run.

For the full re-check, omit `--n-tasks`:

```bash
python -m sn120_v2.recheck score \
  --bundle "$OUT/bundle.json" --work-dir "$OUT/work" \
  --gpus 8 --links "$OUT/links.json" --out "$OUT/verdict.json"
```

`--gpus K` changes only local serving concurrency; it does not reduce the
sample count or draw. Omitting it uses the round's pinned GPU count. Every
parent file's SHA-256 and size is checked against `manifest.base_files`.
The live evaluator verifies the local parent cache again before serving.
The verdict records `delta`, `se`, `z`, both pass rates, confirmation/reason,
sample strings and counts, bundle/code/Git identities, vLLM/torch versions,
wall time, parent-eval statistics, and local overrides.

Without `--links`, scoring uses anonymous object GETs beneath
`https://affine.io/public/bases/<full-base-hash>/` (or `SN120_PUBLIC_URL`), with
an unsigned HEAD for each size and the same hash-verified downloader. This
fallback needs no credentials but succeeds only if that historical parent is
still publicly available. The current-head `read_links.json` is not a fallback
for an old parent.

For the remaining approved pairs, repeat the **entire** export/links/score
sequence with a separate `$OUT`: UID 922 / `r000116`, UID 112 / `r000117`, and
UID 876 / `r000119`. UID 738's pair is `r000114` / `r000115`. Verify that the
printed UID and `crown_round_id` match the table before using any verdict.
Remove the private links copy after the verified base cache and verdict are
retained; do not put links in the evidence archive.

## Runtime estimate and qualification status

**No GPU run was performed for this change.** No target production bundle was
exported, no historical parent object was checked on the production store,
and no target verdict has been produced.

There is no measured parent-only full-draw runtime in this checkout. The
closest recorded serving evidence is an operator-supplied **full multi-model**
round on 8x B300 using `r000057`: 5,562 seconds (92.7 minutes) for the shadow
evaluation, versus 9,114 seconds for live, documented in
`v2/SOLO_SERVING_CHANGES.md`. It is not a parent-only benchmark and was not
reproduced here. The current parent evaluator instead launches one merged-model
solo server per available GPU and shares only the parent jobs across them.

The live params can request 2,000 tasks, two parent samples, up to 65,536
output tokens per rollout, and a 3,600-second timeout for an individual model
request. Offline scoring passes `deadline=0`, so there is **no whole-round
wall-clock cutoff**. Plan for hours, not minutes, for a full eight-GPU run;
provide a multi-hour maintenance window rather than treating 92.7 minutes as a
promise. The six-hour link TTL is not a guarantee that scoring will finish in
six hours. A valid verified base cache avoids reusing signed URLs on retry.
Use the resulting verdict's `parent_eval.wall_s` and `parent_eval.serve_s` as
the first actual parent-only timing evidence.

CPU tests cover export determinism/seals, read-only journal/lock behavior,
affine training-file evidence, exact ties, N-task prefixes, matched live
seeds, legacy history, evaluator setup, private links, missing/corrupt objects,
public fallback, and console secret containment. They do not qualify CUDA,
vLLM serving, the historical parent objects, or a production payout.
