# Controlled native Agent common pipeline

`affine-native-agent-common.service` runs the standard signed GPU miner, R2
freeze, full audit, proposed scoring, optimizer, and held-out evaluation loop
under a separate permanently nonpayable namespace. Its operator configuration
is `state/native-agent-common-config.json`; no blockchain weights are written.

The adapter is currently enabled only in the isolated reviewed source fork
`state/prospective-native-agent-common-v2`, not the main environment registry.
Four distinct original `3d_print_shop_t0..t3` fixtures have separate immutable
public actor and private grader images. The actor executes original database
and tool code through MCP argument validation. The private grader executes the
original checks and solved predicate. This is controlled original-tool/DB/grader
coverage, **not the full original Verifiers orchestrator**. Private grader
images stay on the operator-owned retained machine; this pilot does not distribute
private grading data to external miners.

Training searches task 0 using a public-input curated candidate policy. The two
otherwise identical submit-print-job candidates differ in material ID, providing
success and tool-validation failure paths. The model samples those candidates;
this is not evidence of autonomous solving or unbiased policy sampling. Original
tasks 2 and 3 are fixed, disjoint, autoregressive held-out evaluations. All runs
use a separate pinned BF16 CUDA profile; historical CPU proofs are not reused.

Before each new CUDA role, the dispatcher measures actual free VRAM and waits
for at least 2 GiB, or 4 GiB for training. Existing wide-model jobs are preserved.
Every complete batch uploads cumulatively to private R2, then freezes at the
published deadline. Only fully audited pairs can train. Status and signed role
receipts are in `state/native-agent-common`; a started service alone does not
establish successful mining, training, or learning improvement.

Stop with `systemctl --user stop affine-native-agent-common.service`. The pod is
retained. Restart with `systemctl --user start affine-native-agent-common.service`
to resume recorded phases and existing role IDs rather than launch duplicate jobs.
