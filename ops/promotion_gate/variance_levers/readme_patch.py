import sys
p = sys.argv[1] + "/README.md"; T = sys.argv[2]; s = open(p).read()
R = [
("""- **A round.** The validator draws n = 1000 task ids with `gate_core.draw_sample(seed, 7496, 1000)` from a secret
  `os.urandom(32)` seed and publishes only a commitment (https://affine.io/gate/heldout/current.json). Block 0
  (the selection) runs the head k=4 on all 1000, and the **kept** tasks are those with 0 < c < 4. Blocks 1–3 are
  three more head runs on the kept tasks. Every candidate of the round runs k=4 on the kept tasks with **the same
  seeds as block 1**. The turn batch is not excluded.""",
"""- **A round.** The validator draws n task ids with `gate_core.draw_sample(seed, 7496, n)` from a secret
  `os.urandom(32)` seed and publishes only a commitment (https://affine.io/gate/heldout/current.json). n = 2000 in rounds
  opened after %s (1000 before); n, k_cand and sigma_run are frozen when a round opens and published in current.json.
  Block 0 (the selection) runs the head k=4 on all n tasks, and the **kept** tasks are those with 0 < c < 4 (about 270 at
  n = 2000). Blocks 1–3 are three more head runs (k=4) on the kept tasks. Every candidate of the round runs k_cand = 8
  (4 before the switch) on the kept tasks; samples j < 4 use **the same seeds as block 1**. The turn batch is not excluded.""" % T),
("""head samples; gain = Σw·d/Σw; se_task = √(Σw²d²)/Σw; SE = √(se_task² + σ_run²·(1+1/3)), where σ_run = the run-to-run
  spread of the 3 head blocks (`heldout.run_sigma`; there's no floor since E354, and it was 0.022 for E271–E353); z = gain/SE.""",
"""head samples (candidate rate over its k_cand samples); gain = Σw·d/Σw; se_task = √(Σw²d²)/Σw; SE = √(se_task² + σ_run²·(1+1/3)).
  σ_run in rounds opened after %s = the head's run-to-run spread POOLED over the last 20 finished rounds
  (`heldout.pooled_sigma_run`, frozen at round open; 0.000 at the switch, 95%% upper bound 0.011); before, each round's own
  3-block spread (`heldout.run_sigma`; no floor since E354, 0.022 for E271–E353); z = gain/SE.""" % T),
("""  In practice that bar is about +1.5 to +2 points of weighted gain (GREEN e356 passed at gain 0.0198, z 1.027).""",
"""  At n = 2000, k_cand = 8 and pooled σ_run the bar (= SE) is expected near +1.3 points of weighted gain (it was about
  +1.9 to +3.4, median 2.3, before; GREEN e356 passed at gain 0.0198, z 1.027)."""),
("""`gate_eval.py` draws its own local seed, so it can't know a live round's tasks.""",
"""`gate_eval.py` draws its own local seed, so it can't know a live round's tasks. Its defaults (n, k_cand, σ_run policy and the
pooled σ_run value) come from `validator/gate/gate-config.public.json`, a copy of the live flag."""),
]
for a, b in R:
    assert s.count(a) == 1, a[:60]; s = s.replace(a, b)
open(p, "w").write(s); print("ok")
