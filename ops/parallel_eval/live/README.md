# Parallel evaluation: live wiring (phase 1, fresh-only)

Live on SN120 from epoch 843 (2026-11-07 08:00 CR, Jacob). These are the exact host files that decide and apply
PE epochs (copied from the validator host; host paths kept, no secrets). Pod rental/lane/podctl helpers are host-only.

- `pe_lifecycle.py`: non-blocking trainer. Each epoch trains one candidate from the current head; the result is recorded
  as pending and the epoch returns `no-update`, so the next queued turn trains while the candidate is evaluated. Before a
  new job the trainer must be back on the head (restore request). Epochs before `parallel_eval.from_epoch` stay on the
  serial gate path. A fresh GREEN ready on base H replaces that epoch's result with the candidate's own signed checkpoint.
- `pe_launcher.py`: composes the above outermost over the live lifecycle launcher (timeout = no_decision, never promotes).
- `pe_live.py`: restore consumer, dispatch of each pending candidate as an exact replica of the gate probe (the epoch's
  committed round draw, plan seed `cand_seed(round seed, epoch)`, k_cand 8, 65,536 output, max-num-seqs 512), GREEN queue
  in decision-time order, apply records.
- `pe_gate_hook.py`: runs inside the gate process. Same head baseline (once per round), same estimator and thresholds,
  same DECISION fields as the serial gate plus a `parallel_eval` block. Eval gave up / round closed before judged /
  lifecycle timeout -> no_decision (turn not served, key stays queued).
- `pe_apply.py`: staleness = GREENs applied since the candidate's training head. Phase 1: `APPLY_STALENESS_MAX = 0` —
  only fresh GREENs are applied; staleness 1-2 is `merge_blocked` (DECISION green, credited and paid, head unchanged)
  until the trainer-signed fp32-delta merge publication exists; >2 or unknown is `stale_overflow`.

Tests: `python3 test_pe_lifecycle.py`, `python3 test_pe_lifecycle_ntp.py`, `python3 test_pe_apply.py`.
