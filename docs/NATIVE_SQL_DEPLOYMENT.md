# Controlled original Spider deployment

`native_sql_deployment.py` binds each public descriptor and operator-private fixture to the signed task commitments. The private commitment covers every fixture field except its machine-specific database path; database bytes have their own SHA-256. Changing the question, reference SQL, ordering rule, database, public descriptor, or original task identity fails before execution. The reference SQL remains in the operator-private collection and is never present in a model prompt or source archive.

The isolated SQL dispatcher runs the approved public bash actor and the original grader in separate bounded Docker containers. This is a controlled cohost experiment; it does not grant arbitrary external miners access to operator fixtures, and it does not establish isolation against an operator with host-root access. The active general environment dispatcher remains unchanged.

The initial public-only curated policy explores the original database using two bash calls and chooses a final query derived from the original question. A standalone honest model search found both outcomes in two seeds, and a separate model process verified full float32 output probabilities, strict TOPLOC, and fresh original environment replay. Those controls are distinct from the subsequently launched epoch service.

The `affine-native-sql-common.service` unit uses an isolated, immutable source tree and `state/native-sql-common-config.json`. It preserves the full published ten-minute window, freezes direct-R2 submissions, fully audits before training, proposes weights without chain transactions, and evaluates sixteen original held-out tasks separately from the training task. Every CUDA role waits for measured free VRAM; the owned pod stays running. Stop only this trial with `systemctl --user stop affine-native-sql-common.service`. Its private configuration and artifacts are excluded from Git.

The first full common epoch, `nonpayable-native-sql-common-v1-1790846858-0`,
completed one audited K1/L1 pair, one proposed point/normalized weight, one
full-model optimizer update and checkpoint
`fd8ad8cc7975eb313696fdb52185aff38a397bbc945841baa342f5dfc928bc24`.
All six published checkpoint objects were independently streamed and hashed
(272,585,280 bytes total). Sixteen fixed original held-out tasks, disjoint
from mining, ran with the same autoregressive harness and 128-token budget
before and after: mean reward 0 in both. The following epoch consumes the
new checkpoint. These are controlled common-pipeline measurements, not a
claim of generalization, unrestricted successful mining, or weight submission.

The following epoch closed without a complete K1/L1 batch. Independent checks
of signed R2 manifests, audit challenge and scores confirmed zero receipts,
points and proposed weights, no optimizer run, and the unchanged checkpoint
in the subsequent published epoch. Both retained the full ten-minute window.
Subsequent independent checks established successful recovery: epochs
`nonpayable-native-sql-common-v1-1790849077-2` and
`nonpayable-native-sql-common-v1-1790850079-3` each accepted and audited a
K1/L1 pair, proposed one point and normalized weight, and completed a
full-model optimizer update. Their checkpoints are respectively
`dc1e1672b8d1125b271050b913fe44c75d29b84efedf8bce64fd4d7a8210f639` and
`52a46cf0107e95b52ea9fdf4e79413d97508e85e65f915ed3270e2d3f1ef2347`.
All six objects of each checkpoint were independently streamed and hashed
(272,585,280 bytes per checkpoint). Thus the recorded sequence includes
three trained epochs and an intervening empty epoch, followed by resumed
auditing and training. The sixteen fixed held-out tasks still scored zero
before and after every update; recovery does not establish learning gains.

At the completed third training boundary, the controller adopted a new
immutable worker version for bounded empty searches. A search that finds no
complete K/L pair now reports its observed outcome counts and exits normally
without uploading a submission. Actual inference, environment and transport
errors still fail. This avoids repeatedly executing a successful search merely
because it could not assemble a qualifying pair. The new version's next
epoch was initially a separate ongoing trial. It and its successor have now
completed auditing, training and paired evaluation under that new source:

| Epoch suffix | Successor checkpoint |
| --- | --- |
| `v2-empty-1790851130-4` | `c34b4f717e4baf75dd77438ef9025775779d74c7bc5cc7e4cf93a856ca48d193` |
| `v2-empty-1790852134-5` | `d5347dc1c9f59da0ec574fba01314ff6c637a12126b441fd27f0bbd4af481d4b` |

This brings independently checked common training to five epochs. Each new
checkpoint's six R2 objects were independently streamed and hashed, and the
public dashboard matches all ten paired evaluation records. The fixed sixteen
held-out tasks still score zero before and after; training remains confined to
the controlled original task zero. Wider public-question-derived SQL policies
are prospective until their model/proof admission gates pass.
