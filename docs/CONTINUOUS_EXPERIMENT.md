# Nonpayable multi-environment experiment

## Verified milestones as of 2026-10-01

The direct-R2 service has completed three trained epochs, including original
Oolong, and recovered two empty epochs without inventing training steps.
A subsequent controlled-resource Verbatim epoch is running; its contract pins
provider namespace bytes and explicitly discloses incomplete native dependency
closure. These trials remain nonpayable.

An isolated seven-job GPU control completed mining, full independent verification,
real head training, complete R2 checkpoint publication and a fresh verified batch
from the new weights. A separate full-model 1.7B optimizer measurement also passed
on the retained RTX 3090, with about 14.16 GB peak allocated GPU memory. Its fixed
128-token Oolong held-out results remained zero before and after training. This
is evidence of execution and changed weights, not demonstrated learning gains.
The continuous GPU controller and its finalized score ledger are being integrated.

The ten-environment CPU evaluator is active with refreshed source pins. Earlier
records remain immutable, and source/harness/runtime changes create separate
comparison groups. The public dashboard projects completed measurements only.

## Historical direct transport and research runner

The research runner below describes the earlier seven-epoch experiment. It is
paused at a configuration boundary. The production-capable `subnet.service` pilot
subsequently completed two consecutive direct-R2 epochs with the same continuous
remote miner, accepting two batches then one. Its trial tunnel is stopped. Only
the dedicated owned UID131 test seed was provisioned remotely; operator coldkeys,
validator keys and R2 credentials remain local. The earlier service was paused after its
completed checkpoint a88f352f for broader environment configuration; later
Oolong and controlled-resource trials are described above.
See LIVE_SUBNET.md for direct reads/uploads, atomic deadline snapshots and renewal.

## Historical research runner

`affine-multi-environment.service` supervises the owned chain-registered UID131
miner on the retained Lium RTX3090 machine. The operator controller, independent
audit subprocess and trainer are in this checkout. Public HTTPS uses the retained
registered-test gateway on port8792; production port8790 remains separate.
The rental is retained at $0.22/hour. No weight extrinsic or payout cutover is part
of this experiment. Every epoch is prefixed `nonpayable-` and permanently excluded
from hourly payout accounting.

The installed unit is versioned in `systemd/affine-multi-environment.service`.
Start/restart with `systemctl --user restart affine-multi-environment.service`;
stop the experiment with `systemctl --user stop affine-multi-environment.service`.
Stopping this service does not terminate the rental. The existing tunnel continues
running; the stopped service's public gateway becomes unavailable until restart.
Unit environment controls pin compatible MKL/default ATen/SSE41, four Torch/BLAS
threads, and tokenizer parallelism off. `subnet.affinity` constrains this process
and children to four permitted CPUs without using machine-specific CPU IDs.
No system-wide CPU settings are changed.

The controller publishes a signed manifest, delegates only an encrypted mailbox's
single-object expiring upload capability, and invokes the miner over SSH. The
coldkey and hotkey signing seed stay on the operator host. The miner fetches the
actual HTTPS signed manifest and hash-pinned weights, produces positive/negative
batches and continuously overwrites its private epoch object. Freeze closes writes;
separate full-audit execution replays original tasks and checks complete probability
distributions and strict TOPLOC fingerprints. Proposed normalized weights are
recorded locally. Only fully audited pairs train; real optimizer steps publish a
new immutable checkpoint before the next epoch opens.

Per-epoch `epoch-stage-N.json` records the exact manifest and starting checkpoint
before submission. Restart reuses that stage; it never reopens or changes an already
journaled manifest. Frozen score reports are cached, and completed trainer metrics
and a complete saved checkpoint can be reused. A finalized epoch with no accepted
pairs is recorded as a failed attempt and skipped; it is never counted as a
successful training epoch. Accepted epoch progress is persisted in `progress.json`.
Records from interrupted attempts are retained with distinct epoch identifiers.

The initial training schedules use Mastermind, original Prime Verbatim, and original
reasoning-gym count_bits. These are curated research policies: candidate sequences
are chosen with target-model likelihoods; visible-copy candidates read only the
public prompt. They demonstrate computation/proof/replay/training plumbing, not
unconstrained problem-solving ability. `text-tools-v1` uses a tokenizer chat template
plus a declared text tool protocol; `plain-transcript-v1` uses a versioned explicit
role transcript. The same model/proof code supports arbitrary trusted adapters,
structured tool observations, and unconstrained autoregressive sampling. Original
language-task remote coverage uses genuinely unconstrained generation and records
honest failures rather than inventing successes.

Each environment has distinct fixed training and held-out indices. Comparable
before/after evaluations use identical dataset hashes, task identities, seeds,
harness configuration and CPU profile. `state/evaluations/*.json` contains real
completed counts, requested counts, reward/success statistics, uncertainty and
explicit infrastructure errors. A missing task is an error record rather than a
smaller successful evaluation population. `policy_kind` distinguishes curated
controls from autoregressive evaluations. Root's separate held-out suite worker
tracks newly published checkpoints for broader original-environment coverage.

`state/multi-environment/` contains signed-manifest source files, audit challenges,
frozen submissions, independent reports, proposed weights, trainer metrics,
checkpoint directories, public identity mapping, status and experiment history.
Private capability files and authority seeds must remain private and uncommitted.
The public dashboard exposes allowlisted metrics only.

Full audit is the pilot baseline. Sampled audits select batch indices using a
cryptographically generated seed recorded after freeze and bound to receipt hashes.
Reports state coverage and exact hypergeometric detection under a stated assumed
bad fraction. Unchecked batches never enter training. Scores from sampled subsets
are provisional, and duplicate coverage is incomplete until collision-related data
is fully audited; they are not complete correctness certificates or payout inputs.

The transport currently supports at most32 turns,512 output tokens per turn,
100MB compressed and500MB uncompressed, with bounded model context. These are
pilot resource limits, not claims of incompatibility with original long-context
benchmarks. Native TOPLOC inputs are checked for exact framing and a valid modulus
range before extension calls; no broad numerical tolerance is deployed. CPU135M
portability has been exercised with the common profile; arbitrary GPU portability
and production sandbox/image/network hardening remain separate work.

At 21:52 UTC on 2026-09-30, the first three registered remote epochs completed
with accepted positive/negative batches, full independent audits and three actual
training steps. Their final checkpoint is
`d53bb84e8eddd39b4168800e6744359ae4b399a13e5c8c6676292b1c12b365ef`.
The full signed history is under `state/multi-environment/report.json`; these
remain permanently nonpayable. Three completed epochs establish plumbing, not
training improvement. Compare fixed held-out records for that separate claim.

Numerical runtime revision `cpu-float32-eager-v2-bounded-toploc` explicitly passes
four threads to TOPLOC's native activation bit extraction. The upstream default
calls `omp_set_num_threads(hardware_concurrency())` for every verification, which
otherwise overrides the declared Torch/OpenMP budget and greatly slows subsequent
model evaluation. The revision changes resource scheduling, keeps strict proof
and probability comparisons, and separates evaluation dataset fingerprints from
the older unbounded runtime. Portable four-core affinity is applied before imports.

The exact credential-free implementation used for those three epochs is retained
as the SHA-256-addressed archive
`state/source-bundles/26b36fb35ced8cb3ef09be2d491dbad4dcd6eefdb14a0619099733701c19bc5a.tar.gz`.
Its authority-signed descriptor links the epoch identifiers and the public archive
path under `/public/source-bundles/`. Preserve this version when adapters change;
old manifests must be checked using their pinned trusted implementation rather
than silently substituting a newer adapter or accepting a mismatched source hash.

The next source revision makes the harness registry dispatch actual implementations
for rendering, action parsing, observation handling and policy sampling. Operator-
signed `turn_overrides` permit a bounded first-turn tool choice followed by a
separate final-answer policy without adding environment names to model code.
Overrides cannot read hidden task state, select uploaded code, change harness
version or exceed the signed token budget. An optional separately signed
`evaluation` contract pins held-out indices, seeds and harness: tool-conformance
training controls can therefore coexist with honest free-autoregressive evaluation.
Historical curated records and checkpoint histories remain immutable.
