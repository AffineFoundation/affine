# Changing your batch: replace until seal, and head-bound batches (v8.2.3)

These rules apply to epochs whose signed opening carries
`batch_lifetime_policy = {"version": "replace-until-seal-head-bound-v1", "upload_slots": S}`.
An epoch without that key follows the previous rules unchanged.

## What changed for miners

1. **You can change your batch at any time until the seal.** Each epoch you get
   `S` write-once upload slots (S = 2 × `max_batches` at launch). A commitment
   names which slots are your batch. You can write a new commitment as often as
   you want before the signed deadline. The seal takes the **newest commitment
   that landed before the deadline**. Anything written after the deadline is
   rejected, and the commitment the seal already took stays as it was.
2. **A batch stays valid while the head checkpoint does not change.** Token
   draws are bound to (head checkpoint, your hotkey), not to the epoch. If the
   epoch rolls over and the head is unchanged, the client carries your groups
   forward. It re-labels them for the new epoch, re-uploads them and re-commits
   them, without sampling again. When the head changes (a GREEN gate), old
   batches are invalid and the client starts fresh.
3. **Unchanged:**
   - Epoch timing.
   - Caps (`max_batches`, the K=1 rollout cap).
   - One K=1 turn per hotkey, forever. A served hotkey gets no upload grant.
   - Grading and pay.

## Commands

All of these use the normal `affine-miner` invocation, plus:

| Goal | Flag |
| --- | --- |
| Add groups (same as before) | no flag. Each upload re-commits the full current set. |
| Drop groups from your committed set | `--replace-drop math:17 math:42` |
| Withdraw (commit an empty set) | `--withdraw` |
| Do not carry last epoch's groups on an unchanged head | `--no-carry` |

`--replace-drop` and `--withdraw` run once when the client opens the epoch. Use
them with `--once`, or remove them before the next epoch. A dropped task cannot
be sampled again for credit: its draws are fixed, so sampling it again would
give the same output. Pick other tasks instead.

When all `S` slots are used, an upload fails with `all write-once upload slots
for this epoch are used`. Your newest commitment stays in force.

## What the validator publishes

- Each K=1 turn record (`K1-TURN.json`, `K1-TURNS.jsonl`, and the public
  `k1.json` row) has `sealed_commitment_sha256`. This is the SHA-256 of the exact
  commitment bytes the seal used.
- The admission ledger keeps every commitment that replaced an earlier one,
  append-only and hash-chained: a sequence number, the SHA-256, the upload time,
  the previous SHA-256 and the group count.
- After the seal, upload slots that the sealed commitment does not name are
  deleted. Each epoch's `V823-SUPERSEDED-CLEANUP.json` records which slots were
  kept and which were deleted. Commitment snapshots are kept for audit.

## Before the seal you get structural checks only

Capture runs during collection. It returns only structural eligibility
(`assurance: structurally-eligible-unaudited`): signatures, scope, sizes and
upload windows. Native grades are computed on the host and kept in host memory
only, so collection-time pre-grading leaves no public record. Grades are used
only after the seal freezes the population. Replacing a batch therefore tells
you nothing about how a group would grade.

## Why head-bound draws are safe (no grinding)

- **How draws are made.** Each draw is
  `sha256(DOMAIN, draw_context, env, task, nonce, turn, prompt, position)`.
  - On an unchanged head, the draw context's only per-epoch field is the epoch
    label. We checked this on live openings e196–e204: the checkpoint id, the
    checkpoint files, the source sha, the run id and the runtime are all the same.
  - Under this policy the epoch label is replaced by a constant.
  - The seed is the existing `sampling_contract.randomness`. The host keeps it
    fixed per head: it is generated fresh and kept secret until the first
    opening after a head change, then reused until the next head change.
- **Seed-fixed, not epoch-fresh.** Draws are a deterministic function of
  (head, seed, hotkey, task, nonce), so a miner cannot re-roll them.
  Per-epoch randomness gave everyone a fresh roll each epoch, so it was more
  grindable, not less. What a miner can choose (which tasks to submit) is the
  same as before. They just get more wall-clock time to choose.
- **One valid derivation.** Each epoch has exactly one valid draw derivation.
  The signed policy key decides which, so nobody can pick the better of two.
- **Bound to the miner.** The hotkey is in the draw context, so copying another
  miner's batch fails.
- **A head change resets everything.** A new checkpoint gets a new seed, so old
  draws, and outputs from the old model, no longer verify.
