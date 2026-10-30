# SN120 K=1: persistent fp32 head master (trainer v8.3.3f) + gate verifier fix (2026-10-30)

Approved by Jacob (t821u–t823u, t829u, t831u "push the changes ahorita"). Both fixes are Allan's (catoneone/sn120-k1-fixes).
Live state, timings, gate results and revert log: validator RA `state/root-audits/20261030-fp32-verifier/`
(PROGRESS.md, REVERT.md).

## 1. fp32 head master — `trainer/` (launcher-only change on signed v8.3.3u)
- New engine key `sgd_fp32_master` (OFF when absent; the off path is the exact v8.3.3 expression, bit-identical).
- The step is unchanged: one fresh AdamW step (zero moments, lr 5e-7, per-tensor relative cap 2e-5, eps 1e-8, wd 0.01).
  With the flag on, it runs on the head's fp32 master M_H instead of fp32(bf16 head):
  `M_C = M_H - fp32(s*r_t) * d_t(M_H, g)`, and the candidate is `bf16_rne(M_C)`. RED discards M_C. GREEN makes M_C the head master.
  Optimizer moments are still not persisted.
- Master source per turn:
  - `memory`: the trainer's own M_C, when its bf16 projection equals the input head.
  - `disk`: the store entry `<bf16 sha256>`, verified by meta, master digest and bf16 projection.
  - `bootstrap-upcast`: fp32(bf16 head) exactly, used for the first turn.
- Store: the trainer pod at `/workspace/affine-mimo-trainer-prod/fp32-master/<bf16 sha256>/`, as safetensors shards (≤4 GiB) plus meta.json.
  The save runs in the background. GC keeps 2 entries plus the head's entry plus any PINned entry.
- turn.json `training.optimizer.sgd.fp32_master` commits the input/output master digests (sha256 of the canonical JSON
  `{name: sha256(fp32 LE bytes)}`) and the bf16 sha256s.
- Publication (`release/fp32_publish.py`, host): on each GREEN, R2 `checkpoints/<C>/fp32-master/*` (presigned PUTs from the pod),
  a size read-back, then `https://affine.io/k1/fp32-master/<C>.json` and `index.json`.
- Signed with `release/sign_v833f.py`: the v8.3.3u release with only the launcher and the engine flag changed.

## 2. Verifier fix — `verifier/` (gate probe worker)
- `probe_worker.py` embeds Allan's `code/lenient_grader.py` (786cdb9) verbatim (P1/P2 projection, E1–E7 equivalence).
- It runs only when the live native grade is not positive and the reply ended with EOS; the reply then becomes `positive`
  if the lenient rules match the gold. A live positive is never changed, and capped replies stay unresolved.
- Regression (`tests/regress_verifier.py`, 4,073 finished K=1 rollouts e196–e260):
  - the embedded copy matches Allan's module on every rollout
  - 481 recovered (290 negative→positive, 191 neutral→positive), 0 positive lost
- `tests/test_verifier_wrong.py`: 19 deliberately wrong and right variants, all pass.
- math-verify's timeout needs the main thread, and probe grading runs in multiprocessing Pool workers, so this is fine
  (in a thread only 98 of the 481 are recovered).
- The trainer reward and the miner capture path are NOT changed: miner claims must equal the validator's native grade,
  so that change needs a coordinated client release.
- Going live closes the open held-out round (no promotion), so the next epoch opens a fresh round and baseline on the same head.

## Operations
- `release/activate_fp32v.sh`
  - `MODE=switch WAIT_EPOCH=E`: after the gate DECISION of E it installs the verifier and closes the round, then switches the
    trainer at the E/E+1 job boundary, then starts the publisher and the watchdog.
  - `MODE=revert-trainer | revert-verifier [NOW=1] | revert`.
- `release/watchdog_fp32v.py` auto-reverts the change it blames when:
  - an epoch runs longer than 36 min (45 min for a round-opening epoch)
  - training takes longer than 13 min
  - the gate fails closed
- `tests/test_v833f.py` (13 tests): flag-off identity, bootstrap identity, memory/disk/RED paths, digest chain, tamper/stale rejection, GC.
- `docs/docs_patch.py`: the llms.txt and step-recipe.json sections.
