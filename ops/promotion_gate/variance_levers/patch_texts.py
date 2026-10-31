#!/usr/bin/env python3
"""Variance levers (Jacob 2026-10-31 14:07 CR): public gate-math text. usage: patch_texts.py apply|revert "<switch time CR>" [file ...]
Anchored, idempotent; revert = exact reverse with the same switch-time string (kept in state/SWITCH_CR).
Files: training_params.py (mining.json / training.json heldout), gate_ledger.py (gate.json rule), k1_transparency.py (green.json note),
llms.txt (section 6/7)."""
import os, sys
mode, T = sys.argv[1], sys.argv[2]
RA = "/home/const/subnet120-rewrite/state/root-audits"
FILES = sys.argv[3:] or [RA + "/20261026-promotion-gate/code/training_params.py", RA + "/20261026-promotion-gate/code/gate_ledger.py",
                         RA + "/20261027-k1-per-uid/code/k1_transparency.py", "/home/const/subnet120/affine/website/llms.txt"]
TP = [
 ("""                sigma_run='run-to-run noise of the head, measured each round from its 3 replicate head runs on the kept tasks '
                          '(between-run variance of the weighted mean minus within-run sampling variance); used as max(sigma_run_floor, measured)',
                sigma_run_floor=h.get('sigma_run', 0.022), sigma_run_history=h.get('sigma_run_history'))
""",
  """                sigma_run=('run-to-run noise of the head. Rounds opened after %s (variance levers, Jacob 2026-10-31): POOLED over '
                           'the last %%d finished rounds (each round: between-run variance of the 3 replicate head runs\\' weighted means minus '
                           'their within-run sampling variance; df-weighted mean over rounds, floored at 0), frozen when the round opens and '
                           'published as sigma_run in the round record (current.json); used as max(sigma_run_floor, pooled). Older rounds: '
                           'max(sigma_run_floor, that round\\'s own 3-run estimate)' %% int(h.get('sigma_pool_rounds', 20))),
                sigma_run_policy=h.get('sigma_run_policy', 'per-round'), sigma_pool_rounds=h.get('sigma_pool_rounds', 20),
                sigma_run_floor=h.get('sigma_run', 0.022), sigma_run_history=h.get('sigma_run_history'))
""" % T),
 ("    return dict(from_epoch=h.get('from_epoch'), bar_z=h.get('bar_z', 1.0), z_inclusive=h.get('z_inclusive', False), se=heldout_se(h),\n",
  "    return dict(from_epoch=h.get('from_epoch'), bar_z=h.get('bar_z', 1.0), z_inclusive=h.get('z_inclusive', False), se=heldout_se(h),\n"
  "                set_size_next_round=h.get('n', 1000), k_cand_next_round=h.get('k_cand', 4), head_rollouts_per_kept_task=12,\n"),
 ("            'bar_z', 'z_inclusive', 'status', 'rule')\n", "            'bar_z', 'z_inclusive', 'sigma_run', 'status', 'rule')\n"),
 ("'candidate of the round is scored on the kept tasks minus its own batch tasks (4 rollouts per task, the same request seeds '",
  "'candidate of the round is scored on the kept tasks minus its own batch tasks (k_cand rollouts per task: 8 in rounds opened after "
  + T + ", 4 before; for j < 4 the same request seeds '"),
]
GL = [
 ("'per head, 4 seeded samples per task with shared seeds on both sides); gain",
  "'per head; set size and candidate samples are fixed per round: 1000 tasks and 4 candidate samples per task before, 2000 tasks and 8 "
  "candidate samples per task in rounds opened after " + T + ", the first 4 with the same request seeds as head block 1; the head gets 4 "
  "samples on every task + 12 per kept task); gain"),
 ("from E354 no fixed floor, the measured value is used)",
  "from E354 no fixed floor, the measured value is used; rounds opened after " + T + ": the pooled value over the last 20 finished rounds, "
  "frozen when the round opens, published as sigma_run in current.json)"),
]
K1 = [
 ("GREEN only: the round's 1000 sampled tasks (id", "GREEN only: the round's sampled tasks, set_size of them (id"),
 ("selection run (k=4 on all 1000), blocks 1-3", "selection run (k=4 on all set_size tasks), blocks 1-3"),
]
LL_OLD = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ll_old.txt")).read()
LL_NEW = """- A round: the gate draws n task ids from the pool with gate_core.draw_sample(seed, 7496, n) from a secret os.urandom(32)
  seed: n = 2000 in rounds opened after %(T)s (1000 before). Only the commitment is published
  (https://affine.io/gate/heldout/current.json): sha256(gate_version | seed | sorted ids). n, k_cand and sigma_run are fixed
  when a round opens and published there (set_size, k_cand, sigma_run).
- Block 0 = selection: the head runs 4 samples on all n tasks, and the **kept** tasks are those with 0 < c < 4 (about 13 %%,
  so about 270 at n = 2000). Blocks 1-3 = three more head runs (4 samples each) on the kept tasks (12 head samples per kept
  task). Every candidate of the round runs k_cand = 8 samples per kept task (4 before %(T)s); samples j = 0-3 use **the same
  request seeds as block 1** (common random numbers), j = 4-7 follow the same seed rule. Head and candidate are graded by the
  same grader on exactly the same kept tasks. The turn batch is NOT excluded.
- Score (heldout.score_ivw): w_t = 1/(q(1-q)) with q = (c_sel + 1)/(4 + 2); d_t = candidate rate (k_cand samples) - head rate
  (12 head samples); gain = sum w d / sum w; se_task = sqrt(sum w^2 d^2) / sum w; SE = sqrt(se_task^2 + sigma_run^2 x (1 + 1/3)).
  sigma_run (rounds opened after %(T)s) = the head's run-to-run spread POOLED over the last 20 finished rounds
  (heldout.pooled_sigma_run: per round, the between-run variance of the 3 head blocks' weighted means minus their within-run
  sampling variance; df-weighted mean over rounds, floored at 0), frozen when the round opens. At the switch it was 0.000
  (20 rounds, 40 df, 95%% upper bound 0.011). Before: each round's own 3-block estimate (heldout.run_sigma), floor 0 since E354
  (E271-E353: floor 0.022). z = gain / SE.
- **GREEN iff gain > 0 AND z > 1.00 (strict) AND estimated whole-set cap-hit < 0.20 AND >= 100 kept tasks.** At n = 2000 and
  k_cand = 8 the bar (= SE) is expected near +1.3 points of weighted gain (it was about +1.9 to +3.4, median 2.3, before).
- GREEN: the candidate becomes the next head (bf16 + fp32 master), and the turn is published. RED: the head is kept.
""" % dict(T=T)
LL = [(LL_OLD, LL_NEW),
      ("  - the tasks: all 1000 sampled tasks with id", "  - the tasks: all sampled tasks of the round (1000; 2000 in rounds opened after %s) with id" % T),
      ("  (gate_eval.py uses its own random seed; expect a few points of difference from the gate's hardware).",
       "  (gate_eval.py uses its own random seed and the live n / k_cand / sigma_run from gate-config.public.json; expect a few points\n"
       "  of difference from the gate's hardware).")]
PAIRS = {"training_params.py": TP, "gate_ledger.py": GL, "k1_transparency.py": K1, "llms.txt": LL}
rc = 0
for p in FILES:
    s = open(p).read(); o = s; bad = 0
    for old, new in PAIRS[os.path.basename(p)]:
        a, b = (old, new) if mode == "apply" else (new, old)
        if b in s and (a not in s or a in b):
            continue
        if s.count(a) != 1:
            print("ANCHOR %s: %r found %d times" % (p, a[:70], s.count(a))); rc = bad = 2; continue
        s = s.replace(a, b)
    if bad: print("NOT WRITTEN %s" % p); continue
    if s != o:
        t = p + ".tmp-vl"; open(t, "w").write(s); os.chmod(t, os.stat(p).st_mode & 0o777)
        if p.endswith(".py"):
            import py_compile; py_compile.compile(t, doraise=True, cfile=t + "c"); os.unlink(t + "c")
        os.replace(t, p); print("%s %s" % (mode, p))
    else:
        print("unchanged %s" % p)
sys.exit(rc)
