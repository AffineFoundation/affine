# fp32 head master publisher (Jacob 2026-11-04 05:42 CR: "Publish the fp32 master tensors per GREEN head")

Every GREEN head is published twice: the bf16 checkpoint (lifecycle, signed current.json -> opening.json read_urls) and the fp32
master M_C the trainer stepped (bf16 head = round-to-nearest-even(M_C)). This sidecar does the second part.

- Trigger: a GREEN line in the gate ledger (applied == decision == green) + the turn's public turn.json
  (`training.optimizer.sgd.fp32_master.output_store_entry / output_master_digest`).
- Source: the trainer pod's fp32 store entry (pinned while uploading). Refused unless meta.json's master_digest equals the
  digest the trainer committed in turn.json, and every shard holds only F32 tensors named model.language_model.* / lm_head.*
  (427 tensors; no optimizer state exists: the step is stateless).
- Storage: the same R2 bucket as the bf16 head, `mimo/affine-mimo9b-grpo-20261013/checkpoints/<C>/fp32-master/` (9 shards + meta.json).
  Only missing or wrong-size objects are uploaded (idempotent). `pod_upload.py` is reinstalled on the trainer pod when missing.
- Public: https://affine.io/k1/fp32-master/<C>.json (sha256 + bytes per file, tensor_sha256, master_digest, `download` =
  presigned R2 GET links valid 7 days, re-signed when 3 days are left, for the newest 3 GREEN heads), index.json, current.json.
- Failure: logged + state/ALERT.json; retried every 5 min up to 12 times; never blocks, delays or reverts the gate, the king or
  weights. A store entry that is already gone is recorded as `store-entry-gone`.
- Verify as a miner: `python3 miner/kit/fetch_fp32_master.py --out ./head-fp32-master` (sha256 per file + master digest).
- Revert (host): `bash ~/subnet120-rewrite/state/root-audits/20261104-fp32-master-publish/revert.sh` (stops this unit, removes the
  download links, restores llms.txt, restarts the v1 publisher, which uploads without public links). Public repo: git revert.
