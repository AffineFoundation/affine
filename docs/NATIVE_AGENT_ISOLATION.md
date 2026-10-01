# Controlled original Agent fixture pilot

`subnet/native_agent_isolation.py` separates original mutable task tools from
the private original grader. The current measured fixture is GeneralAgent
`3d_print_shop_t0`. Its unmodified `TaskDB` and `TaskTools` execute inside an
operator-approved immutable actor image containing public initial database,
tool implementation, instruction and checked worker. The distinct grader image
contains the original `checks` and `solved` method bodies and private gold.
The actor receives no gold, grader mount, network, Docker socket or credentials.
Both containers run nonroot with read-only filesystems, dropped capabilities,
bounded memory/processes and a temporary filesystem. Fixture files are copied
into separate images, so execution does not depend on host bind-mount paths.

Actual isolated native controls return success **1** after public-instruction
selection of red PLA and an idle compatible printer followed by
`submit_print_job`; stopping without changing state returns **0**. The native
`verify` and `db_hash` metrics, original solved reward and mutable state replay
are preserved. This is a controlled fixture adapter, not a claim that the full
original verifiers orchestrator, arbitrary corpus or transitive provider/native
dependencies have been integrated. Its declared dependency scope is
`immutable-controlled-images-not-full-upstream-closure`.

`ops/probe_native_agent.py` takes an independently approved plan containing
checkpoint file hashes, runtime and source pins, public instruction, actor and
grader image IDs, and fixture inventories. Generation records every complete
model token context, full output log-probability distributions and TOPLOC
fingerprints; a separate process reloads the same independently approved
weights, recomputes all distributions/fingerprints and replays original tools
and private grading. The policy is curated from public instruction and observed
tools. It establishes target-model computation on selected trajectories;
it does not claim unbiased autoregressive sampling. Original private gold
never participates in selecting actions.

The probe is a standalone, permanently nonpayable CPU pilot. It does not call
blockchain APIs, perform optimizer updates or register this adapter in the
production environment registry. The genuine model/proof outcome is recorded
separately from the native controls. The fixed current task is not a broad
heldout evaluation or evidence of training improvement. Before production
integration, the common signed environment/resource contract must select the
approved images and private grading policy, and real proof-bearing positive
and negative traces must pass independent verification and mutation controls.

Operator evidence is retained under `state/native-agent-isolation/` (ignored
by Git): `image-descriptor.json`, `module-native-controls.json`, independently
approved `model-plan.json` and versioned genuine-pair directories. The probe
CLI runs `generate` first and then `verify` in a fresh process with identical
`--plan` and `--output`; it validates source hashes before approved model
loading. An initial verifier API typo was fixed in a new plan/artifact version;
its failure and earlier generation remain historical evidence, not relabeled
as successful verification.

The version-two genuine pilot passed a fresh verification process: positive
four turns/three actual tools/reward one, negative one turn/no tools/reward
zero. Every output includes approved 135M model full log-probabilities and
TOPLOC fingerprints, recomputed at absolute probability tolerance `1e-5`,
relative tolerance zero and TOPLOC error zero. Four actual fresh-process
mutations (probabilities, fingerprints, tool observation, claimed reward) were
rejected by their expected checks. The operator-signed audit hashes the exact
raw files and verification evidence. This is K1/L1 **controlled experience**;
no optimizer or production epoch admission has occurred.

The plan identity in generation, verification and the signed audit is SHA-256
of canonical JSON. The mutation runner's historical `approved_plan_sha256`
field is SHA-256 of the exact pretty-printed plan **file bytes**; those two
representations intentionally have different hashes of the same plan. An
independent checker must verify both against its separately trusted plan and
must not treat the raw-file hash as the canonical artifact identity.
