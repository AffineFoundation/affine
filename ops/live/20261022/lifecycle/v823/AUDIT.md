# 20261016 SN120 MiMo v8.2.3: overlong soft start 7000 -> 6000 (lifecycle-config succession)

Operator approval: Jacob (owner), 2026-10-16 evening ("If the parser fix and the length penalty will help learning we should add them now go").
Signed with the on-box controller authority 3301134b (seed file named by the live lifecycle config; never printed or copied).
SUBNETOWNER mnemonic not used. No chain transactions. No Discord or Telegram posts.

## What v8.2.3 contains
- **Exactly one change.** `trainer_config.overlong_soft_start` goes from 7000 to 6000. `overlong_max_tokens` stays 8192 and `overlong_penalty` stays 0.5. KL (0.02), LR (1e-6) and max_tasks (128) are unchanged.
  - Shaped reward for a correct answer: 1.0 up to 6000 tokens, then linear down to 0.5 at 8192. Examples: 0.886 at 6500, 0.772 at 7000, 0.658 at 7500.
- **Unchanged.** The public release policy stays `dfe0cbf7` and the release chain stays `460a1f5e`. The source archive (`a2a03cb3`), the worker closure (`d0b0475d`, 773 files) and `archive_execution` are the same, and so is the writer. No verifier, trainer or evaluator host files change.
  - So `auditable_archives` is unchanged and e40-e45 stay auditable.
- **Why no new policy hash.** trainer_config is not part of the public scope. It reaches the trainer inside the authority-signed TRAINING-JOB, and `validate_job` re-derives TrainerConfig from it. Miners see no change.

## Files (release-v823-signed/, all 0600, SIGNED-MANIFEST.sha256)
- `lifecycle.v8.2.3.ACTIVE.private.json`, sha256 `b8c1a318…`. This is v8.2.2 ACTIVE `943ed077…` with only `/trainer_config/overlong_soft_start` changed, which `sign_v823.py` asserts.
- `capture-operator.v8.2.3.ROOT-ADMITTED.private.json`, sha256 `d07b9218…`, and `uid-operator.v8.2.3.ROOT-ADMITTED.private.json`, sha256 `65155f17…`. These carry new ROOT admissions pinning the new lifecycle config. The entry and source_root are unchanged.
- `observer.v823.private.json` is the mining.json observer rebound to the v8.2.3 lifecycle, capture and uid pins.
- `activation-v823/` holds the lifecycle drop-in `zzzz~~~~~~~~v823-release.conf` and the observer drop-in `zzzz~~~mimo-observer-v823.conf`.

## Tests before signing
- Unit tests (worktree a7bc7f77 = v8.2.2 source): 73 passed. That covers grpo_overlong, grpo, grpo_worker, lifecycle, lifecycle_adapters, archive_execution and audit_archive_scope. See unit-tests.log.
- `sign_v823.py` checks:
  - the v8.2.2 signed manifest
  - the policy, chain and closure
  - the admission formula, which reproduces the v8.2.2 admissions
  - the trainer document delta, which is only settings.overlong_soft_start
  - the shaped-reward values
  - `mimo_uid_capacity_launcher --check`, which reports operator_admission_passed and original_source_guard_passed
- Trainer round trip: the e45 TRAINING-JOB config equals the v8.2.2 TrainerConfig document, and the v8.2.3 document re-validates.
- Observer dry run: on a simulated v8.2.3 lifecycle argv, the v8.2.3 observer accepts and the v8.2.2 observer refuses.

## Activation
`activate_v823.sh --arm` runs as the transient unit `mimo-v823-arm`. It waits for round 235 (e45 finalized; e46 is the first epoch to train with soft start 6000). It aborts with nothing changed if epochs/46 already has a TRAINING-JOB. Otherwise it:
1. Backs up the unit drop-in dirs, CURRENT and mining.json to `backup-<TS>/`.
2. Installs the two drop-ins (the v8.2.2 drop-ins stay underneath) and runs daemon-reload.
3. Restarts only affine-mimo-lifecycle and affine-mining-discovery.

## Rollback (one line)
U=/home/const/.config/systemd/user; rm -f "$U/affine-mimo-lifecycle.service.d/zzzz~~~~~~~~v823-release.conf" "$U/affine-mining-discovery.service.d/zzzz~~~mimo-observer-v823.conf" && systemctl --user daemon-reload && systemctl --user restart affine-mimo-lifecycle.service && sleep 20 && systemctl --user restart affine-mining-discovery.service
(The same steps are in rollback_v823.sh. This is safe at any time: an existing TRAINING-JOB reloads from disk with the config it was signed with.)

## Not in v8.2.3 (deferred; see the report)
- **Parser fix `mimo-final-channel-math-v2`.** `subnet/mimo_answer_projection.py` is in the worker closure, and the projection policy is in the public immutable_manifest.
  - Shipping it changes the closure. `source_guard` needs an exact closure match, so the e40-e45 audit jobs would become unauditable on v8.2.3 workers.
  - It also changes the scope, so the native-grading qualification has to be redone.
  - It needs `_validate_public_protocol` to accept v1 for historical chain members. As written, it requires `== POLICY`.
- **Verifier speedup (L1, L2, epoch-affine).** It changes the worker closure too, with the same auditability conflict, and it needs per-host worker configs plus MPS.
# Incident 2026-10-16: e46 training failed after the v8.2.3 activation (fixed by recovery, v8.2.3 kept)

## What happened
- e46 job mimo-train-e5dcf2bf (soft_start 6000) was dispatched at 03:06Z (22:06 CT).
- Attempt 1 failed at 03:08Z with ValueError('original healthy optimizer step and resolved config') in subnet/mimo_grpo/training.py:66 (`optimizer.config != config`).
- Attempts 2 and 3 failed with RecoveryRequired('original update interrupted; restore original evidence explicitly'). The transaction had been left in `updating`.
- After that the queue row was failed 3/3, and the lifecycle wrote RECOVERY-REQUIRED every tick.

## Root cause (confirmed in the source)
- `subnet/mimo_training_worker.py` keeps the warm `context.runtime` and `context.optimizer` between jobs. It rebuilds them only when the checkpoint changes or `optimizer.failed` is set, never when the resolved TrainerConfig changes.
- The trainer worker had been up since 14:14Z, so its CPUAdamW held the v8.2.2 config (soft_start 7000). `train_update` correctly refused to run it against the v8.2.3 job config.
- My v8.2.3 activation restarted only the lifecycle and the observer, not the trainer worker. That is the gap.

## Recovery (03:22Z, recover_e46.sh). It follows precedent v821-trainer-override/cutover_v821_trainer.sh plus reset_e39_job.py from 10-16 06:48Z.
1. **Pre-checks.**
   - RECOVERY-REQUIRED is for this job and shows failed.
   - The e46 TRANSACTION.json is a bare never-started journal: keys job, state=updating and version only, with no artifacts and no report, so no optimizer step ever ran.
   - Trainer CURRENT is at the e45 output efd2eef3, step 46.
2. **Drain.** Stopped the lifecycle and the trainer watcher, then drained the warm trainer worker (owned-worker v822s config, `--stop-owned-after-drain`). Its GPU memory was released.
3. **Backups and queue reset.**
   - Arbos `recovery-e46-20261017T032202Z/` holds roles.sqlite3, TRAINING-JOB.json and RECOVERY-REQUIRED.json, which were moved aside.
   - Deleted only the failed e46 train row (asserted status failed, attempt 3, role train).
4. **Trainer files moved, nothing deleted.** On the trainer, moved the job's attempt workspace and the bare `updating` journal (history/update-6e817cba…) to `/workspace/affine-mimo-trainer-prod/backup-e46-recovery-20261017T032202Z/`.
5. **Restart.**
   - Started the trainer watcher first. It came up as a cold worker that restores the e45 optimizer from the published journal under the v8.2.3 config (`restorable_config`: Adam settings identical).
   - Then started the lifecycle, which re-issued the same job id with the same 128 documents and soft_start 6000. created 03:23:56Z, expires 05:23:56Z (00:23 CT). The cold worker leased it as attempt 1.
- No check was weakened, no signature was needed, and there were no chain transactions.

## Undo
Stop the lifecycle and trainer watcher. Restore roles.sqlite3 and the moved files from recovery-e46-*/. Move the trainer backup-e46-recovery-* items back.

## Fix for future config changes
- **Procedure, required now.** Any lifecycle `trainer_config` change must drain-restart the trainer worker at activation, after the previous epoch's job has finished and before the next TRAINING-JOB is created. The step is the `--stop-owned-after-drain` and watcher restart from activate_v822.sh.
  - activate_v823.sh lacked this step, and it is added to the v8.2.4 plan.
- **Code, v8.2.4, not shipped tonight.** In `mimo_training_worker.execute`, also close the warm context when `context.optimizer.config != config` (one line), so the cold-restore path runs. Add a unit test.
