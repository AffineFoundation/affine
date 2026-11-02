# Kings / Queue / History tables (affine.io)

Jacob pt1100u (2026-11-02 11:24 CR): the Next in line, Payees (last 10 GREEN), Head lineage and Turns tiles are combined
into three full-width tables stacked Kings, Queue, History. They share one column schema and one renderer (k1-tables.js).
Layout after teutonic.ai (shared identity columns first, keys clipped head…tail with the full value on hover/copy, a sticky
header in a fixed-height scroll box, relative times with the exact time in the tooltip, a count beside each title).

- Shared columns: UID, Hotkey, Coldkey (clipped 6…4, click to copy, full value in the tooltip), Batch (tasks × mean
  rollouts per task), Hparams (only the train knobs that differ from mining.json batch_file.train.fields defaults, full set
  in the tooltip; generic over keys, so new optional knobs show up when they ship), File (batch.json link, GREEN only).
- Kings: # · shared · GREEN at (epoch + relative time) · z · Gain (pp) · Weight (scores.json share) · Head hash (+ "head" tag).
- Queue: # (line) · shared · Reg block · Order · Est. eval (rest of the current epoch + one median epoch per place; median
  of the last 20 finalized epochs) · Submitted (time since the current file was first checked, ff/status.json).
- History: Epoch (links turn.json) · shared · Result · Gain · z · vs bar (z − 1) · Cap-hit · Evaluated.
- Privacy: train params and the batch file are shown for GREEN turns only. RED rows show "private" (red dashed, like the LR
  chart); pending and queued files show "private" (neutral); affine-batch-v1 queued files show "defaults". No priors, selection
  probabilities, IRT parameters or holdout data are read or published.
- Data: existing /k1.json, /scores.json, /mining.json + new /k1/tables.json from services/build_k1_tables.py (user timer
  affine-site-k1-tables, every 60 s): per decided turn gain/SE/decided_at, GREEN-only train params + file link (from the
  published turn.json), coldkeys (chain SubtensorModule.Owner, cached 6 h in site-tools/state), queue file checks
  (ff/status.json), epoch timing (gate.json), train defaults (mining.json). Charts are unchanged.
- Load: frames and headers are static HTML (afx placeholder rows), rows fade in once (200 ms, reduced-motion honoured).
  ?v=pt1100u-tables-20261102 on k1.js, k1-tables.js, k1-tables.css; /k1/tables.json prefetched in <head>.
- Deployed 20261102T173422Z. Backup + revert: ~/subnet120-rewrite/state/bench-results/site-backup/20261102T173422Z/revert.sh (copy in this dir).

## Queue count fix (20261102T173805Z, Jacob follow-up 11:36 CR)
- The Queue title read 64 over 8 rows. k1.json queue.queue_length is the number of unserved members with a valid file now;
  only the next 8 (QUEUE_N) are published. The title now reads "8 of 64 ready", and a muted last row says
  "+N more with a valid file now · M unserved registrations without a valid file are skipped and keep their place"
  (M = registrations_unserved - queue_length). Option 1 (list every unserved entry) was not used: the k1 service publishes
  only the next 8, and rebuilding the full line outside it could drift from the gate's real order.
- Order verified against ff_lifecycle.queue/select: open grants first, then (registration block, UID) ascending; at each
  seal the first member with a valid file takes the turn. k1.json next[].line is the same order filtered to valid files.
  UID 371 (order 131) sat at line 0 because its file was first checked at 11:25:26 CR, after the E563 seal (11:19:31 CR,
  which went to UID 456 at order 147); E564 then went to UID 371 as listed. Est. eval = rest of the current epoch + (line + 1)
  median epochs, consistent with that order.
- ?v=pt1100u-queue-20261102 on k1-tables.js / k1-tables.css. Revert this step: revert-queue.sh
  (~/subnet120-rewrite/state/bench-results/site-backup/20261102T173805Z/revert.sh); full revert: revert.sh.
