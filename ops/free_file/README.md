# SN120 free-file protocol: validator side

Jacob 2026-10-30 (pt898u). Miner-facing contract: [docs/MINER_QUICKSTART.md](../../docs/MINER_QUICKSTART.md)
and https://affine.io/mining.json → `batch_file`. Host work dir: `RA/20261030-free-file`
(`PROGRESS.md`, `DECISIONS.tsv`).

## What changed

| Removed | Kept |
| --- | --- |
| mining client, TOPLOC, hashes/proofs/seals for miner batches, epoch commitments, dual-accept, regrading, mixed-outcome rule | activation commitment (Ed25519), SealedBox delivery, hotkey-registration-once queue (UID 0 excluded), private held-out probe, bar z > 1, baseline revealed after GREEN, weights / flip / nonzero floor, bf16 + fp32 master publication on GREEN, nonces, batch cap file |

The only gate is model improvement. The probe no longer excludes the turn batch.

## Pieces

- `release/ff_format.py`: the `affine-batch-v1` parser and tokenizer check (shared by the lifecycle
  and `ff_service`). Limits: 64 MiB, output ≤ 8192 tokens (8192 = length-capped, no EOS), prompt +
  output ≤ 12288, no control tokens, finite rewards, distinct prompts. Cap = whole prompts in file order.
- `release/ff_service.py` (`affine-ff.service`): every 30 min re-issues each eligible hotkey's sealed
  link (`/ff/links/<hk>.json`: presigned PUT/GET for `private/ff/v1/miners/<hk>/batch.json`, 24 h);
  every 20 s format-checks changed files → private `FF-READY.json` + public `/ff/status.json`.
- `release/ff_lifecycle.py` + `ff_launcher.py`: wraps the admitted v8.2.9x launcher. At the turn seal it walks
  the K1 queue: first member with a valid file → copy (evidence, `private/ff/v1/turns/<epoch>/`), build a
  signed dispatch (`sn120-ff-dispatch-v1`) and the training job; until the cutoff a member with a ready
  old-format batch is trained the old way; nobody ready → no-update epoch. Missing/invalid file = skipped,
  not charged. A selection error before the cutoff falls back to the old path.
- `trainer-v833h/`: trainer host v8.3.3h = v8.3.3g + `ff_trainer.py` (rewarded trajectories → the same
  GRPO rows; reward used as given, overlong penalty applied). Native rows are byte-identical to v8.3.3g.
- `site/`: the new https://affine.io/llms.txt (old one kept at /llms-legacy.txt) and the dashboard JS diff (k1 tiles: mining line, queue readiness, held-out z text).
- `services/sidecars.diff`: gate (probe counts the turn batch from `turn_batch_counted_from_epoch`),
  k1 transparency (GREEN publishes the file, RED does not), k1 queue (preview 8, file readiness),
  training params (rule texts), mining.json publisher (`batch_file`, `old_format_cutoff`, FF-PINS).
- Old-format cutoff: 2026-10-31 09:00 UTC = 03:00 CR (unix 1793437200), in `staged/FF.json` and mining.json.

## Operate

```bash
MODE=stage  bash RA/20261030-free-file/activation/activate_ff.sh   # inert staging + --check
MODE=switch bash RA/20261030-free-file/activation/activate_ff.sh   # after a DECISION, trainer idle
MODE=revert bash /home/const/subnet120-rewrite/state/root-audits/20261030-free-file/activation/activate_ff.sh
```

`ff-watchdog` reverts automatically on: lifecycle down 90 s, FF training job > 25 min, two selection
errors, job build > 300 s, seal → training result > 30 min, or no finalized epoch for 45 min.

## Tests (`tests/`)

`test_ff_format.py` (parser/tokenizer rules), `test_trainer_ff.py` (native rows identical to v8.3.3g;
synthetic file → trainer rows), `test_r2_roundtrip.py` (sealed link → PUT/GET, wrong content type 403,
wrong key refused), `test_lifecycle_sim.py` (queue skip of invalid/missing files, legacy before the cutoff,
FF after it, job validates with the worker's `validate_job` + `prepare`).

## Fresh turns (K1-GRANTS, Jacob 2026-10-30 19:30 CR)

`staged/K1-GRANTS.json` (copy: `release/K1-GRANTS.json`): one fresh free-file turn for each hotkey whose turn since
the held-out gate went live (E271+) was RED with 0 < weighted_z ≤ 1.00 (10 hotkeys; e286 with z = 1e-16 and the
fail-closed e274 are excluded). `ff_lifecycle.open_grants/grant_rows` put them ahead of the unchanged K1 order
(positions −k…−1, oldest RED first). A grant is used only by a real turn (a K1-TURNS row for the hotkey after its
RED epoch); a missing or invalid file is a skip and keeps it open. Granted hotkeys get links (`ff_service.eligible`)
and show in `training.k1.queue_grants`; `queue_next` lists them while their file is valid. Off switch:
`enabled: false` (read on every call). Tests: `tests/test_grants.py`. Text diffs for the sidecars and the
held-out gate wording: `services/grants-and-gate-text.diff`. `activate_ff.sh MODE=restart` reloads
`release/*.py` into the lifecycle at an idle trainer (re-pin `FF-PINS.json` first, or mining.json reports
`discovery_binding_error`).

## Legacy removal (after the cutoff)

Branch `legacy-client-removal-20261031` removes the old client (`subnet/cli.py`, the `affine-miner` entry point,
client-only tests); merge after 2026-10-31 09:00 UTC. The validator-side legacy paths are switched off by
their existing no-restart flags with `activation/legacy_off.sh` (MODE=check|apply|revert: `FF.json legacy_capture=false`;
`EPOCH-TIME-LIFECYCLE.json` seal_pregrade / seal_prefetch / capture_put_fanout / dual_extended = false), only after
the cutoff + 30 min and two clean post-cutoff epochs. Code removal from the live closure needs a new signed source
release, because workers hash `subnet/*.py` (`source_files`); it is not done by editing files in place.
Plan: `RA/20261030-free-file/PROGRESS.md`.
