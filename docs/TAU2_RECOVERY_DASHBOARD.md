# Tau2 evaluation projection after explicit recovery

The public dashboard compares the original sixteen-task baseline with the
completed after-checkpoint population assembled from eleven originally verified
tasks and five separately regenerated and independently verified recovery tasks.
The original after-evaluation remains a separate partial eleven-task record with
five errors. Recovery did not perform another optimizer update. Both complete
population means are zero; this is completion evidence, not a quality gain.

`ops.export_tau2_recovery_evaluations` authenticates the original signed completion,
its exact evidence inventory and signed envelopes, and a separate signed metadata
approval binding the unchanged completion and both heldout contracts. It checks
paired task identities, trajectory attempts, measured rewards, explicit recovery
origins and full independent role/native verification before producing records.
The approval adds publication metadata; it does not replace or resign the original
failed evaluation. The new comparison cohort digest includes the authenticated
agent seeds omitted by the historical dataset digest.

Reproduce from operator evidence (no model generation or training is performed):

```sh
PYTHONPATH=. .venv/bin/python -m ops.export_tau2_recovery_evaluations \
  --repository . \
  --epoch-folder state/native-tau2-common-live/epoch-1790869388 \
  --authority d54a3a345d0de3e2c7898f30c0942d78f931f8c4b8036ffdc6adffcd2525062f
.venv/bin/python -m unittest discover -s tests -p test_tau2_recovery_evaluation_export.py
.venv/bin/python -m unittest dashboard.test_server
```

The three derived files go to `state/evaluations`, where the existing dashboard
projects only allowlisted public metadata. Free-text worker failures, episode
paths, private storage capabilities and token traces are excluded. Only completed
records enter the performance curve. Public recovery metadata preserves the
original error count and separately recovered count. Tau2's different model,
harness and seed-bound cohort remain separate from the wide-loop curves.
