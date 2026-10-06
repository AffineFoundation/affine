# Complete heldout128 dashboard evidence

The default-off dashboard hook reads `dashboard/heldout128-sources.ROOT-SIGNED.json` only. The signed payload version is `heldout128-dashboard-sources-v1`, with exact source SHA, canonical runtime-map SHA, ordered cohort SHA, original fixed32 excluded indices, and an `evaluations` list. Each entry pins a signed summary path and byte SHA, plus each group's full archive path and full-readback receipt path/SHA. Paths must be absolute, canonical and not symlinks. The deployment scope stays private.

This supports the original paired CP11/CP12 summary and the completed single CP13 summary. A missing summary yields no row. An existing malformed or incomplete summary fails closed. It authenticates all four ROOT-signed original job/report/terminal ACKs per checkpoint from the hash-bound complete archive, checks each full-object readback receipt, exact runtime/source/checkpoint/task/seed/harness bindings, mining and old32 exclusion, distinct original jobs, the full ordered128 cohort and actual model-retirement completion. No network access, model load, dispatch, report modification or signature generation occurs in the projection.

Public rows contain native outcomes, the128 denominator, cached1024 budget and authenticated production-checkpoint association. They never contain private paths, GET URLs, bucket object keys, proof-validity credit or a partial aggregate. Repeated scoped summaries cannot duplicate a row. The existing fixed32 projection and both charts remain unchanged. This is independent evaluation evidence, not a training admission barrier or convergence assertion.

Current actual preparation is `state/root-audits/heldout128-dashboard-projection-preparation-20261006-v1/source-pointer.UNSIGNED.private.json`. ROOT must inspect and sign that exact payload before deploying the additive module and seven-line server hook. Do not replace the live server with the repository baseline: port this hook into its existing evaluation-input collection, preserving its outcome/incentive additions.

Validation:

```
/home/const/subnet120-rewrite/.venv/bin/python -B -m unittest dashboard.test_heldout128_projection dashboard.test_cached_evaluator_projection dashboard.test_server
```

A read-only check on actual authenticated original archives reconstructs CP11=77/128, CP12=86/128, CP13=84/128. CP14 has no result in this scope. The private CP13 full archive was independently fetched from R2 and matched906660 bytes and the ROOT-signed summary digest before preparing the unsigned dashboard scope.

Comparability is mandatory across every32-task chunk and every checkpoint in the signed dashboard scope: common runtime package versions, signed model runtime revision/backend/numerical profiles, native environment revision and harness-source hash, normalized6746-task mining set, complete native specification (snapshot, native source hash, grader dependencies, turn/token/reward settings), and cached1024 harness. Reports must match their original signed manifest execution/generation revisions and profiles. The same index+seed must retain the same native task hash across checkpoints. Epoch/request clocks and model checkpoint identity may differ; they do not enter the comparability digest. Signed-but-different runtime/profile/grader data cannot be combined into a displayed comparable series.
