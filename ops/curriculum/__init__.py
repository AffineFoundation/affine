"""Adaptive, auditable curriculum for the duel corpus D (stages 1 + 2).

ledger.py          king performance per (verdict, side, turn) from published
                   duel records; deterministic rebuild; content sha
rule.py            rule v1 math (shrinkage, weights, floors, clamp, multiplicity)
weights.py         rollup + live index -> shadow weight vector + audit files
counterfactual.py  stored verdicts re-scored under the shadow group vector
diff.py            one-page diff vs the previous fold
publish.py         data.affine.io/curriculum/** + private Discord line
run.py             the pm2 job: everything above + the stage-3 criterion
"""
