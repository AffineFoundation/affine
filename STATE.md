# Affine rewrite status

## Separate H200 deployment — October 2, 2026

The operator's Arbos.life name refers to this current code/validator host.
The original five single-H200 nodes passed pinned CUDA BF16, Qwen import,
TOPLOC generation/replay and mutation controls. However, Lium account audit
records now show all five were deleted through an API key on October 2:
delete requests occurred at 20:57:38–20:57:48 UTC and were marked
`user_initiated`. This is distinct from the original miner's earlier
`REBOOT_FAILED` event. The deletion client was the local `affine-pod-reaper`: its logs and ledger
match all five account API requests. The nodes lacked required ownership
registration, so its 90-minute unregistered-pod rule deleted them. The surviving
replacement miner now has explicit retained ownership with no lifetime ceiling.
Four replacement roles were registered before rental; the shared reaper
remains running.
The replacement miner and four restored H200 nodes are now present as RUNNING
in independently queried Lium inventory, and all five have retained ownership
records. All five passed fresh source/model/native-grader admission. The four restored
nodes additionally passed strict independent replay, tampering, weight-restoration
and corrected-source production initialization controls. The retained miner also
passed corrected-source initialization. The restored fleet's listed hourly rate
totals $15.35; node inventory and worker liveness are checked separately.

The recovery epoch
`nonpayable-separated-hopper-original-math-recovery-v1-1790975553-0`
opened at 21:12:38 UTC and kept its signed 21:42:38 UTC deadline. The replacement
miner genuinely searched 20 samples and uploaded one success/failure batch
before the deadline. Its R2 artifact is 318,236,285 bytes with SHA-256
`676815117403b201844f7e5318fdc671d17d030b6feb5b2819bb303d49ce1013`.
The genuine frozen batch passed the real audit and earned one nonpayable
point (proposed weight 1.0). Two additional audit jobs ran concurrently on the
two distinct verifier workers; both accepted the same batch, and the actual
score-file hash stayed unchanged. The recovery companion completed the original
fixed eight-task untouched-model baseline: eight correct out of eight, with no
training. That saturated small cohort does not measure performance on all MATH
problems. The companion and its original workers are terminal. The original
manifest, submission and signed deadline remain unchanged.

Review found that the frozen training guard assumes a single weight file and
rejects Qwen's sharded checkpoint. Commit
`745cee2a2fd040ee601ba4fb8c5e5defa6235c8d`, pushed to main, compares all
weight shards and verifies actual parameter-value hashes before and after
optimization. Relevant verification passed 105 tests. That fix requires a
new signed source and fresh epoch; the original source and submission cannot
be silently relabeled. The corrected v2 source was signed and strictly admitted
on all five nodes, and its full 15,242,788,091-byte initial checkpoint publication
completed. On October 2 at 22:41:46 UTC, its first mining epoch opened:
`nonpayable-separated-hopper-original-math-v2-1790980906-0`. The exact remote
mining job `...-mine-598fa5ce-00339d36` completed successfully. It searched
20 genuine samples and uploaded one success/failure batch for task 1278 before
the original deadline. Root independently fetched and hashed the 318,236,285-byte
R2 submission: `edc79ec48ff59c884809c6161d3c123de0966fa0c5e1e43b0e7bb2698836b689`.
The epoch froze after its original signed 23:11:51 UTC deadline. Its actual
independent audit accepted the batch, yielding one nonpayable point and proposed
weight 1.0. The full-model training job completed one AdamW update across
7,615,616,512 parameters and all 339 parameter tensors. Actual parameter-value
hashes changed, independently of serialization layout; the exported successor
is `253f921e60baee63d803e69c78408e3ae5b96b088b7e93c2d288c7ea571d7c29`.
Evidence: `state/prospective-separated-hopper-math-v2/actual-training-root-review.private.json`.

Checkpoint publication then failed: Transformers exported a single roughly
15 GB weight object, exceeding R2's 5 GiB single-PUT limit. The once-controller
is terminal and retains phase `train`; its committed step counter is still zero
because publication and handover have not completed. Recovering the exact
trained bytes through multipart upload must not repeat the optimizer update.
Future source exports now explicitly request 4 GB shards and preflight every
object against the single-PUT limit before starting an upload. Those changes
do not alter the frozen live source or its existing checkpoint.

Before v2 activation, a new fixed heldout cohort of 32 tasks was selected without
consulting outputs: sixteen Level 4 and sixteen Level 5 problems, all reserved
and disjoint from mining. Before/after evaluations use that identical cohort
and numerical profile. The baseline attempted all 32 tasks, but one task
(reserved index 6665) failed strict TOPLOC verification. Its status is `error`,
with 31 verified results and no mean reward or confidence interval. It is not
a completed baseline or evidence of improvement. The original failed record
is preserved; a same-model, same-task diagnostic is being prepared without
changing thresholds, seeds or cohort. A separate same-source, same-base
one-task diagnostic reproduced the TOPLOC failure with index 6665 and seed
26926002; the original baseline and score hash stayed unchanged. A read-only
diagnostic then confirmed identical hidden bytes and zero full-probability
error. Only prefill block zero failed, including against the original hidden
tensor used to build its proof. Root reproduced the omitted index remapping
in TOPLOC's Python list verifier; its native C++ verifier already applies that
mapping. The future-source correction follows those native semantics and
retains exact zero-error thresholds; see `docs/TOPLOC_INDEX_MAPPING.md`.
The old eight-task baseline remains a separate record.

Public miner fixes now preserve the complete delegated epoch upload capability
and honor the signed long-proof artifact budget in packing, restoration and
upload. Twenty-six focused tests pass; commit
`57fa14c8700e1d03177f744141e158b5076e557e` is verified on GitHub main.
A separately signed prospective v3 source will qualify that public client on
real H200 outputs. The running v2 source remains unchanged. Prospective v3 source
`c20009a33302e8268407160a63668d9f4f5c718c9d4a61d42aa1203e688e2a47`
contains only the six reviewed public-client changes. Its 1,438-member archive
passed strict installation and source/runtime/native-grader admission on all
five H200 nodes. Root separately verified its signed descriptor and exact R2
bytes. This admission uses the initial checkpoint as a placeholder; v3 remains
preparation-only until the actual trained successor and handover are reviewed.
Internal mining is disabled in v3 so the delegated public CLI can be tested
without competing internal submissions. Successful checkpoint publication,
paired heldout evaluation, public-client GPU qualification and consecutive
handovers are still required.

The future corpus source v5 has 1,443 strictly admitted files and passed 109
tests in its frozen tree. Its startup guard refuses preparation-only plans
before opening network or controller resources. Source
`9160d4786c5c74237ad093205343f79e18568e86e65e7684c28c7605acd2e8b7`
is signed and independently read back from R2, but has no activated controller
or GPU jobs. Its older export code still needs the subsequent shard fix before
it can be used for a new production training release. The larger DeepMath and
Numina tasksets remain CPU-qualified preparation, not proven GPU training.

The prospective v6 source includes the explicit export shard setting and
single-PUT size guard. Its 1,444-file archive
`b5cac29975cc2a01d5db17d47b5c964a697232cde0efddca4d448144ebcea781`
passed 128 tests in the frozen source and strict post-test membership checks.
It is signed and independently read back from R2 under authority
`861b6ba38150b0f17097abe1918f8d74d9ad932eab42d3828887203bbad43897`.
The next external-client pilot now targets a fresh v7 source containing both
the export fix and index-map verification correction. v3 and v6 remain
preparation history. The next pilot still requires actual trained-checkpoint
and verification admission. No v6 epoch has opened or proposed score been created.
The dashboard now explicitly allows its eventual public measurements; fifteen
projection privacy tests and the deployed affine.io JavaScript readback pass.

Root independently streamed all six recovered successor objects from R2 and
verified 15,242,698,782 bytes against the original trained file map. The failed
single-PUT job remains failed. The reviewed CPU transport-recovery adapter
resumes only that exact publication, rechecks current R2 bytes and signs the
normal checkpoint descriptor before continuing the original once-controller.
It reuses the original completed training report and does not repeat the update.
The resumed controller is confirmed live; after-evaluation and final handover
are still pending.

Corrected proof source v7
`175cd629e397f3b1a32d3066fcb673dddacfe674fb39682f117ca1af2ff27ce9`
has 1,446 strictly admitted files and passed 132 tests in its frozen tree.
Its signed archive was independently read back from R2 under authority
`d372183790955364d0cd7758d1d64d71200b8d6d30ad1e39d05c4f446dc0a3f2`.
Separate fleet source installation and runtime admission are in progress;
the known-task GPU qualification and fresh comparable evaluations are pending.
No v7 mining epoch is active. affine.io now allows that future namespace, keeps
the two-chart layout, and explains the distinct Qwen cohort in `/llms.txt`.
Sixteen privacy projection tests and actual HTTPS readbacks pass.
Chain submissions remain disabled. Production, the retained-3090 pilot and
independent services are preserved.

DeepMath and NuminaMath-CoT catalog/native-adapter qualification now yields
832,335 training tasks plus separate heldouts, represented by 106 bounded
assets. Their GPU mining/proof/training qualification is still pending; see
`docs/MATH_CORPORA_QUALIFICATION.md`. Resources are separate signed, hashed
data assets outside immutable executable source caches.

affine.io now has exactly two charts: evaluation performance over time and
batches per finalized epoch, with a linked `/llms.txt`. Live desktop/mobile
browser checks pass. No Arbos website or DNS was changed.

## Earlier retained-3090 qualification — October 2, 2026

That earlier pilot is restricted to the original MATH environment and the untouched
upstream SmolLM2-1.7B-Instruct base. Its six model files have been downloaded to
the retained GPU host and checked against pinned upstream metadata. The original
loader provides 7,496 distinct tasks: 6,746 authorized mining tasks and 750
reserved evaluation tasks, with an initial fixed cohort of 32. All prompts fit
the pinned tokenizer/context budget. Grader controls and portable source,
bootstrap, cumulative-upload and rotating-search controls pass.

The untouched base's 32-task GPU baseline completed with 6 correct answers,
zero verification failures and no optimizer update, inside its original signed
lease. Its six R2 objects were independently streamed and hashed. The registered
UID 131 miner has now completed the real external bootstrap path: signed source
admission, isolated CLI execution, checkpoint download, autoregressive sampling,
and a private R2 cumulative upload. The independently hashed 32,373,517-byte
artifact contains one batch for original task 503, with two rollouts and full
49,152-vocabulary float32 probabilities. The epoch closed at its original deadline,
passed the normal inference verification and native grading, earned one point,
and completed eight full-model updates. All 218 parameter tensors received
gradients under one persistent AdamW optimizer with an immutable epoch reference.
The actual published successor checkpoint bytes were independently streamed and
hashed, and the epoch's test score was excluded from payout aggregation.

The same 32 held-out tasks improved from 6 correct (0.1875) to 7 correct (0.21875),
with zero verification failures. Three tasks improved, two declined and 27 were
unchanged. Full tokenizer semantics, special tokens, chat template and all 32
prompt token sequences are unchanged despite export serialization differences.
Actual affine.io browser checks include the new math cohort and completed batch.

The second epoch exposed an owned-worker admission ordering bug: resolving the
task subset imported pinned environment modules before installing the fresh
source loader. Its four failed jobs, original manifest and deadline are preserved.
The registered external miner recovered the window with two genuine batches,
for tasks 1235 and 4454. Both passed full verification and earned two nonpayable
test points. The controller resumed at the original deadline, completed another
eight full-model updates and exited normally. All six successor checkpoint
objects were independently streamed and hashed; only the model-weight file
changed. The pre-training evaluation reproduced the previous 7/32 result exactly.
After training, the same 32 tasks reached 9/32: two improvements, no declines and
30 unchanged. The untouched-base series is therefore 6, 7 and 9 correct on this
small fixed cohort, with 16 actual optimizer updates across two completed epochs.
These measurements do not establish broader improvement.

The worker fix defers source-dependent
subset and replay checks until after source/runtime admission, still before
creating job artifacts or loading a checkpoint. Fresh-process controls confirm
valid admission and rejection of held-out indices, altered source and preloaded
runtime modules. The corrected immutable source bundle has been published and
all 1,731 actual remote files checked against the reviewed bytes. Fresh-process
admission controls also pass on the retained GPU host. A continuous controller
has resumed from the twice-trained checkpoint with the same task pool, held-out
cohort and training policy. Its first owned GPU mining job completed successfully
and uploaded three cumulative batches for tasks 1862, 7197 and 3422. The actual
62,607,619-byte R2 artifact and remote terminal report were independently checked.
The third epoch closed at its original deadline and all three batches passed
full inference/native verification. It completed eight more full-model updates;
every update's positive/negative hashes match the verified pairs. All six actual
R2 checkpoint objects were independently hashed. The fixed cohort remained 9/32,
with one improvement, one regression and 30 unchanged tasks. The independently
inspected ledger now contains three completed epochs, no pending records and no
chain writes: 24 actual updates and a base-to-successor series of 6, 7, 9, 9.
The same continuous controller is proceeding to the next owned-worker handover;
that next epoch's genuine contributions remain to be qualified.

The GPU policy field is now bound when opening and signing an epoch. The generic
bootstrap requests unencoded R2 responses while retaining its encoding refusal,
byte bounds, archive hashes and signature checks. The successful loader recovery
admitted the unchanged deployed source; it did not rewrite the epoch manifest or
extend its deadline. The pilot is nonpayable and does not submit chain weights.
The wider experiment's latest paired evaluation completed with two improving,
four declining and ten unchanged metrics. Its next opening is interrupted by a
truncated local manifest; that historical artifact remains preserved. The older
details below describe earlier milestones and must not be read as current live
job status. See [the math pilot requirements](docs/MATH_PILOT.md).

## Earlier verified state — October 1, 2026

GitHub main now contains the inference-verification rewrite. The deployed public
dashboard is affine.io; Arbos domains are unchanged. New-pipeline test epochs are
nonpayable and do not submit chain weights. Older operational entries below are
historical records, rather than instructions to re-enable their services.

The earlier small-held-out GPU series has five completed epochs with independently checked frozen submissions,
original-environment audits, 13 real optimizer updates, checkpoint publication,
and held-out evaluation records. Those updates consumed verified pairs from
Math, Verbatim, Reasoning Gym, When2Call, IFEval, and Oolong. These measurements
do not establish improvement across all environments.

The wider series has eleven independently checked completed epochs and 41
full-model updates. The latest completed epoch used fourteen original families,
sixteen fixed held-out tasks per family, and eleven balanced fresh/replay updates
with immutable reference probabilities and one persistent AdamW optimizer.
Its successor is checkpoint
`0081b0698c0ccc103edfca0506a2c60aeaf9d0a521716889e6a51fdef0a6513b`.
Independent HTTPS checks matched all 254 public evaluation records and the
checkpoint/256-UID grids. The latest comparison has three improved metrics,
two declines and nine unchanged metrics; Math fell from 0.5 to 0.375 and Trivia
from 0.5 to 0.4375. These measurements do not establish improvement across all
environments. See [paired results](docs/WIDE_EVALUATION_RESULTS.md).

MRCR and TriviaAbstain have now completed genuine common training epochs.
MRCR's public-shell retrieval policy remains a curated control over two disjoint
conversation groups. TriviaAbstain's original grader rewards abstention, so its
successful curated abstention batches do not establish factual knowledge gains.
The balanced replay controller has consumed independently reverified historical
pairs with exact optimizer attribution. Historical standalone qualifications and
failed candidate searches remain preserved separately.

The first sixteen-family controlled empty window honored its full published
900-second deadline and froze zero submissions with no optimizer update. Its
subsequent 256-task evaluation completed 658.770 seconds after the original
signed one-hour lease. That report is preserved as diagnostic evidence and is
rejected for epoch admission; the signed epoch status is `aborted_evaluation`.
A fresh immutable recovery source preserved all sixteen environment contracts
and completed a new empty window using an original signed three-hour evaluation
budget. Both 256-task evaluations passed with no failures, no submissions or
optimizer updates, and an unchanged checkpoint. Independent HTTPS checks matched
its 32 public evaluation records. The subsequent normal epoch has a genuine
Numina index-13 batch with fresh full model/native audit, immutable private and
public R2 copies, and an independently recomputed signed normalized score of 1.
Its before-training evaluation completed with all 256 tasks across sixteen
environments verified and no failures, inside its original signed lease. The
results exactly match the empty epoch's after-evaluation on the unchanged
checkpoint, including fixed task IDs, seeds, dataset identities and rewards.
The signed twelve-step training job completed with one fresh Numina submission
and approved historical replay. Independent source/archive and attribution checks
bind all twelve full-gradient updates, immutable input references and persistent
AdamW counters 1–12. The uploaded successor is
`9d558f36c595bf7f894b820daed11f18f9331ccbe7827ef455ed30381f1c34b6`.
Its paired after-evaluation is running; full epoch completion and the subsequent
rollover remain unproven. The completed-epoch ledger remains eleven epochs and
41 updates until those gates pass. Pydantic index-0 training follows that
recovery; standalone model qualification is insufficient.

The separate mixed-runtime Tau2 common trial completed frozen proof auditing
and an agent-only full optimizer update. Its original after-evaluation retained
11 verified tasks and five infrastructure errors. A separately signed recovery
attempt now verifies all five failed tasks under the same checkpoints, task
seeds and fixed auxiliary user model; the original failed report is unchanged.
A fresh successor contribution under the trained checkpoint passed six-role
independent model verification, original native replay and exact private proof
storage checks. These
separate attempts must not be presented as an error-free original evaluation
or as evidence of broad performance improvement.

The controlled Tau2 tool-use positive/negative pair now passes independent
model verification and replay through the original environment grader. The
positive trajectory has six agent/user responses and reward 1; the negative has
eleven responses and reward 0. Both use the same task and checkpoint, and
auxiliary user tokens are excluded from training targets. Its agent-only
full-model optimizer completed one agent-only update. All six successor
checkpoint objects were independently hashed in R2, and a fresh six-role
rollout passed independent probability/TOPLOC verification and original
tool/grader replay. This is a controlled outcome-conditioned training example;
the separate common mixed-runtime trial and explicit recovery are recorded above.
Held-out quality improvement remains unproven.

The controlled original Agent fixtures now run through the shared GPU epoch
pipeline. Eleven full epochs passed independent artifact, audit, score,
optimizer attribution, checkpoint-publication and held-out evidence checks.
One full-model update per epoch consumed its verified positive/negative pair; all six new
checkpoint objects were independently hashed (272,585,280 bytes). Two fixed
held-out tasks scored 0 before and after training, so no improvement is claimed.
The same miner has uploaded a new pair using the trained checkpoint. The
public actor/private grader images preserve original tools, mutable state and
grading; the full upstream Verifiers orchestrator remains unproven. See
[the shared native Agent scope](docs/NATIVE_AGENT_COMMON.md).

Original Spider SQL now has thirty-two pinned tasks, sixteen for mining and
sixteen disjoint held-out tasks. Public actors and private original graders
passed fresh remote controls. A controlled target-model candidate-policy
positive/negative pair passed independent full probability/TOPLOC checks and
native replay. Fourteen common training epochs now passed independent evidence checks,
including resumed submission, auditing and training after an intervening empty
epoch. Each published six independently hashed checkpoint files and completed
sixteen fixed autoregressive held-out evaluations (0 before and 0 after).
The controller adopted bounded empty-search reporting at a completed epoch boundary.
These checks confirm the controlled loop, without
establishing held-out quality improvement. Calendar's curated positive/negative
six-turn model artifacts also pass signed model-audit binding and separate
original native replay. A new length-balanced Calendar pair additionally passes
seven model turns, including complete-context terminal proofs, and separate
fresh native admission with original rewards 1 and 0. See
[terminal control scope](docs/NATIVE_EOG_BALANCED_TERMINAL.md). These controls do not establish original sampling,
general quality gains; the separate common-pipeline results are recorded below.

Calendar's common long-context controller has three independently checked
completed epochs and three full-model agent-only updates. Signed manifests,
frozen complete-context batches, full inference/native audits, normalized
proposed weights and optimizer attribution were checked. Each successor's six
R2 checkpoint objects were independently streamed and hashed (999,524,442 bytes).
Two fixed, disjoint original held-out tasks scored zero before and after every
update, so no quality gain is claimed. The controller progressed past two empty
search epochs without treating them as negative samples. Live affine.io JSON
matched all six paired evaluation records and the 256-UID grids; earlier
Calendar desktop/phone chart checks remain recorded separately. The auditor
accepts larger bundles only under the exact approved numerical/transport profile.
The active trial uses one step and its frozen source remains unchanged. The
public repo includes a guarded utility that reproduces its exact portable source
variant in an independent checkout; actual Git-archive reproduction passed
without accessing fixtures/models or starting services.

The source-level coverage matrix records 45 imports and 18 trained
original families, with versioned evidence distinguishing native controls,
model/proof qualification and completed common training. Numina and Pydantic
now have qualifying target-model positive/negative controls, but their new
common epochs remain pending. Prolog has a shared-model positive/negative proof
batch independently verified against its original native grader, over the
qualified NQueens fixture; common training remains pending. Its three fixtures
represent two problem geometries. RCore's 64 original operator grader controls
and public arithmetic controls passed, with pinned isolated dependencies and
resource bytes; remote model/proof qualification remains pending.
The public coverage snapshot reflects these milestones, including Tau2's
completion with explicitly recorded evaluation recovery.
These flags describe demonstrated scopes, rather than complete support or broad
task mastery across each environment family.

See [environment coverage](docs/environment-coverage.json) for the source-level
snapshot and [wide tasksets](docs/WIDE_TASKSETS.md) for measurement limitations.
Private operator evidence and the full completion audit remain under `state/`;
the full multi-environment goal is not complete.

## Historical implementation record

## Objective
Rewrite Affine for inference verification through sampling. The previously planned Wednesday teacher-model switch was publicly postponed.

## Workspaces
- New development workspace: /home/const/subnet120-rewrite
- Branch: rewrite/inference-verification
- Legacy running workspace: /home/const/subnet120
- Full legacy archive: /home/const/subnet120-archive/20260930T144912Z/repo
- Archive inventory and restore metadata: /home/const/subnet120-archive/20260930T144912Z/metadata

The full archive copy completed. All 1,396 tracked entries matched recorded checksums, and archived Git connectivity passed. Runtime logs/state were copied live and are not an atomic database snapshot.
Legacy source removals are staged only on the rewrite branch. No rewrite commit or remote push has been made.

## Established access
GitHub PAT authenticates as unarbos, with access to AffineFoundation/affine.
Arbos Discord bot can post in Bittensor SN120 channel 1381987595881414656.
Validator wallet: /home/const/.bittensor/wallets/default/hotkeys/default (Finney netuid 120). Private keys remain outside this checkout.
1Password vault: Arbos. Credentials are not copied into this workspace.

## Existing operations
Public chat was persistently disabled and its pod removed.
The legacy validator and support services remain active. The operator deferred the shutdown decision and requested a standalone hourly owner-weight burn script; that implementation is underway in a separate sub-agent.
No GPU pod is terminated by workspace preparation.

## Next design decisions
1. Define the work miners perform and the inference-verification protocol.
2. Define sampling, scoring, fraud handling, and incentive rules.
3. Specify validator/miner interfaces and compute/storage requirements.
4. Plan migration, rollout, and restoration of the previous system if needed.

## Hourly owner-weight heartbeat
Installed independent owner-only SN120 weight job: `ops/hourly_burn.py`; operator reference `ops/HOURLY_BURN.md`. Runtime is `~/.local/share/affine-burn/` with dedicated Bittensor 11.0.2 venv; user-systemd timer `affine-hourly-burn.timer` is enabled and active at each UTC hour, with user lingering enabled. Owner resolved live to hotkey `5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8`, UID 0. Commit/reveal is enabled and automatically handled by SDK. The first manual invocation was deferred because the live validator had recently set weights (100-block limit); no extrinsic was sent by the new job. Legacy services remain untouched. The old validator can overwrite these weights or keep the hourly job rate-limited until the operator decides its transition.

## Verification experiment (2026-09-30)

Operator delegated an end-to-end GPU prototype and explicitly requested retaining its machine for follow-up experiments. New experimental Lium pod `affine-proof-experiment` / HUID `golden-comet-16`, ID `8c416640-e9dd-453a-b78f-ce1d9a866df8`, single RTX3090 24GB, $0.22/hour, SSH `root@90.95.12.246` port20062; created15:35:12UTC, no removal schedule. At15:52UTC Lium reported $0.06 spent. This is separate from all production pods.

`prototype/pipeline.py` ran pinned SmolLM2-1.7B-Instruct through upstream Prime Intellect Mastermind, generated a genuine3-turn trajectory, full-vocabulary log-probabilities and12 TOPLOC fingerprints, then independently verified it in a fresh process. Positive curated control accepted as intended. Final report:2 accepted controls and14 rejected resealed adversarial artifacts, including actual changed-layer computation and isolated changed-weight fingerprints. Full warm verification about0.47seconds; same GPU/backend/bfloat16 only. Trusted weights policy is outside miner ZIPs. See `prototype/README.md`, `prototype/pod.json`, and local/remote `prototype/artifacts/report.json`; full weights+trajectory are in `complete-rollout.tar`. Pod source is `/root/affine-proof`; Python `/root/proof-venv/bin/python`. TOPLOC0.1.6 had to be rebuilt for torch2.12.0 ABI. Production environment remains unchanged. Next: repeated/cross-hardware trials and random partial-audit design.

## Equal registration transition — 2026-09-30
User replaced owner-only burn with 1/N for model registrations activated from now onward. Burn timer disabled; affine-transition-weights.timer active every minute. Cutoff Finney block 9181759, 15:54:34 UTC. Verified new uploaded R2 models only; no current eligible miners at deployment. Original validator restarted online with persistent marker payout guard, intake/evaluation retained. Submission window ends October 2 15:54:34 UTC; no automatic old payout restoration. See ops/EQUAL_REGISTRATION_WEIGHTS.md.

Experiment artifacts are fully retrieved locally. All16 attack/control ZIP SHA256 values match the report; the3,438,131,200-byte full-state bundle matches its remote SHA256 `430d9d27540a6eb77cb9daafdf4cbb6d23a5f123a9522c73d614f673ea3ecec5`. Billing ledger through15:55:01UTC:1,189.86seconds, $0.0727136884. The experimental pod remains running at $0.22/hour.

## Extended inference checks (2026-09-30)

Local isolated TOPLOC0.1.6 extension rebuilt for torch2.14.0+cu130 using privately extracted Python3.12 development headers; `.venv/bin/python` now imports and verifies native proofs. Transformers5.14 BatchEncoding chat-template return handled in prototype context helper. Experimental GPU was externally removed: Lium ledger `user_initiated`17:20:20UTC, final$0.382979/6,266.93seconds; this agent did not terminate or re-rent it. Full approved1.7B state is preserved locally.

CPU-vs-originalGPU **cross-runtime** test (hardware and software versions both differ) rejects honest GPU proofs at exact settings. Worst honest exponent mismatches3/128, mean mantissa error1.76, full logprob delta0.62977. One post-hoc tolerant point admits altered-weight computations with layer scales1.003/1.01, despite rejecting the original four numeric attacks. These tolerances are diagnostics only, not deployed.55frozen-artifact random audit configurations/1.1million sampled challenges match hypergeometric detection predictions; cached reference computations mean this measures detection rather than savings. See prototype/EXTENDED_RESULTS.md and prototype/artifacts/extended JSON reports. Production services untouched; no additional rental.

## Complete synchronous pipeline — 2026-09-30

Implemented `subnet/` controller, private cumulative R2 upload gateway, signed epoch
manifests, Ed25519 sealed capabilities, miner search/upload client, bounded ZIP
transport, independent full-audit verifier, uniqueness scoring, separate trainer
subprocess, checkpoint publication/next-epoch barrier, fresh-chain registration
adapter and hourly SDK payout worker. See IMPLEMENTATION_PLAN.md and
[architecture](docs/ARCHITECTURE.md).

Actual dedicated R2 mock completed epoch `mock-1790789692`: three local miner keys,
real135M target-model sampled guesses, full-vocabulary probabilities/TOPLOC proofs,
independent verifier processes and environment replay. Two miners collided on
index0 and each had one unique valid index, producing points1/1/0 and weights.5/.5/0.
Third miner's corrupted reward rejected. Private access403, wrong-key decryption
failed, late overwrite403, frozen public SHA verified. Real preference training
changed weights; checkpoint `97f3406807b61835a497eeba9ff82b58244b722632065d2a8786b230f0b1e3d8`
was uploaded and its next-model rollout verified. No blockchain transactions.
Private evidence `state/e2e/report.json`; later separate trainer-process proof in
`state/e2e/separate-trainer-report.json`. Two protocol and nine chain-adapter checks
pass. This is conservative full-audit smoke, not a finalized research objective.

Live read-only chain inspection found212 current signed Ed25519 registrations at
block9182279; actual participation in the NEW pipeline not yet demonstrated.
Payout cutover and reachable HTTPS controller deployment remain to be verified.
Original transition writer still active as of this record; no new chain extrinsic.
Cross-backend numerical compatibility is not proven and tolerances remain strict.

Final current-source mock also completed `mock-1790790431` with separate trainer
subprocess and invalid miner intentionally colliding with valid index1. Scores stayed
1/1/0, verifying invalid data cannot cancel valid competitor. Evidence
`state/e2e-final/report.json` success=true. Cached finalization was separately checked
idempotent, retaining the exact finalized timestamp/report. Service units verify.

Live controller and dedicated temporary Cloudflare quick tunnel were started,
without altering old production ingress. Independently fetched/verified signed
manifest over HTTPS: `live-1790790840-0`,212 registered miner mailboxes,8000 indices,
deadline2026-09-30T18:04:00UTC. Temporary gateway
https://takes-binding-pastor-biblical.trycloudflare.com; authority public key
`cebf616ae4cdd1a78a4441040046a1714522dfb6d4e8ecedbdd3974f66a09646`.
Local evidence `state/live/readiness.json`. New controller is collecting, not proof
of actual miners or accepted live weights. Transition payout timer remains active
until root cutover; no new mechanism extrinsic. Quick-tunnel hostname changes on
restart: replace it with a durable named ingress before claiming durable operation.
Full-audit CPU tiny Mastermind/135M checkpoint are explicit smoke settings.

## Registered remote trial steering

Operator now requests remote owned-registered-miner trial **without chain payouts**.
No payout cutover occurred; new payout timer/service inactive. Added permanent
nonpayable report flag and payout filtering of `nonpayable-`, `test-`, `mock-` epochs;
ten chain checks (including nonpayable exclusion) plus two protocol checks pass.
`subnet.registered_test` isolates test uploads/epochs and never writes payout ledger.
Remote miner may receive only an expiring scoped capability after local mailbox
decryption, so the operator wallet need not be transferred.

Rented and retained separate RTX3090 pod `affine-registered-miner-test`,
`816e4656-05f5-4b11-afee-cd823068ef17` / `gentle-fox-99`, $0.22/hour,
SSH root@90.95.12.246:20062; new container at prior experiment's endpoint, with a
separate explicitly accepted host-key file. No scheduled removal. Exact CPU
runtime installation is underway in `/root/miner-venv`; global image torch2.12 pip
constraints were overridden for isolated torch2.14 install.

Identity discovery is blocked:17 original miner wallets and validator are sr25519;
no Ed25519 seed/public matches in backups. Relevant authorized vault mining and
validator mnemonic records likewise correspond to sr25519. Their Ed25519 derived
alternatives have no SN120 UID at fresh block9182516. See public-only evidence
`state/registered-test-discovery.json`. Transient private vault copies were deleted.
An already-registered owned Ed25519 private-key source is needed; no replacement
identity or new on-chain registration is fabricated. Existing transition writer
may independently transact; this trial has submitted no chain extrinsic.

## Owned registered remote trial (2026-09-30)

One explicitly authorized paid Ed25519 miner registration succeeded on Finney SN120 at block 9182681, UID131, hotkey `5E68nqmVj1gSjoJHbusG2o4SiJ1dq17M7QG5PVzFKK2ic49u`. Actual funded coldkey balance decrease including burn+fee:1.061166556TAO; remaining95.669061463TAO. Registration evidence `state/registered-test-registration/receipt.json`; activation commitment confirmed and visible at block9182704. Coldkey and Ed25519 private key remain on operator host.

Retained Lium miner pod `816e4656-05f5-4b11-afee-cd823068ef17` (`gentle-fox-99`, affine-registered-miner-test), RTX3090, $0.22/hour, SSH root@90.95.12.246:20062; only expiring encrypted-mailbox-derived upload capability delegated to host, no coldkey exported. Pinned CPU inference runtime installation in progress. Isolated nonpayable epoch job is `state/registered-test/job.json`, temporary HTTPS ingress `https://the-organization-initial-latex.trycloudflare.com` forwarding8792. Port8790 was already occupied by production uvicorn and was left untouched; discarded first trial tunnel stopped.

New payout writer and cutover remain DISABLED. Trial epoch prefix `nonpayable-` and `payable:false` are excluded independently from hourly points. No payout ledger inputs are written by trial runner; registration+activation were setup transactions, no chain weights submitted by this experiment. Existing unrelated transition writer remains active.

### Registered trial progress19:36UTC

Aligned CPU execution profile (MKL_CBWR=COMPATIBLE, ATEN_CPU_CAPABILITY=default, ONEDNN_MAX_CPU_ISA=SSE41) allowed the honest remote batch to pass unchanged strict full-logprob/TOPLOC checks. Independent audit accepted index2; frozen SHA d8cf3fb11d00fc537efdc02a3376509ca6fc67c2a74a45b76257c32356d12168. Real trainer step completed with loss0.6931471824645996 and changed weights. Initial tokenizer publication omitted Transformers5 chat_template.jinja; model_files now includes it. Repaired immutable checkpoint2b80bfabb0b54a409d8fb4df832112d208773d40391a76e5e2475d3306f31166 is public in nonpayable-registered-1790795876-next-complete.

Unexpected external interruption: Lium pod816e4656-05f5-4b11-afee-cd823068ef17 was deleted19:34:48→19:35:27UTC, authoritative reasonuser_initiated. This agent did not issue deletion/reboot/TTL. Saved describe evidence state/registered-miner-pod-current.json. Final next-round remote sampling/verification has NOT completed. Health source state/registered-test-compatible/health.json truthfully blocked_remote_pod_deleted; prior accepted batch and training remain valid. Root was notified to reconcile possible user cancellation before any replacement rental. No weights/cutover were executed.

## Public dashboard — September 30
User mockup implemented and deployed to affine.io: monochrome Affine wordmark, hero batches-per-epoch chart,256-cell UID grid. SQLite public projection refreshed every15seconds by affine-network-dashboard.service; exported network-data.json uses only allowlisted public metrics. Pilot/nonpayable mode separate from live; UID131 recorded batch1. Desktop/mobile and public HTTPS checks passed. No Arbos/DNS edits. Dashboard README has backup/rollback paths.

### Registered remote trial COMPLETE (20:07 UTC)

The root cause of the original rental removal was the existing `affine-pod-reaper`: the rental's required ownership entry was omitted, so it removed the pod after 90 minutes. Replacement pod `0a67adc9-b484-4172-96c1-073efec7d8be` (`zesty-comet-b2`, `affine-registered-miner-retained`) is RUNNING, RTX3090 at $0.22/hour, SSH `root@90.95.12.246:20059`. It is registered explicitly as `manual:user-retained-verification` with expected_hours=0 (indefinite); global production cleanup was not disabled. Exact known tested runtime was copied as a portable package bundle. No coldkey/private hotkey was exported.

The remote replacement downloaded the complete trained checkpoint, generated and uploaded another positive/negative batch, and passed the independent full audit at environment index 1. Frozen artifact SHA `c1f3a9aadb97438fd235c15cd64c1a57cc7a41780b5287ba24ad83f611ed50e6`. Final epoch `nonpayable-registered-1790795876-next-retained`. The initial accepted epoch index 2, genuine separate-process training step, publication of changed weights, and fresh next-epoch remote verification are all complete. First failed unconstrained CPU audit and incomplete tokenizer-packaging attempt remain historical evidence; thresholds were not loosened.

Final report and health `state/registered-test-compatible/report.json` / `health.json` have complete=true / status=complete / next_remote_rollout_verified=true. UID131 Ed25519 hotkey ownership rechecked at block9183026. All test scores remain nonpayable, future payout points empty, no chain weight submission or writer cutover occurred. Existing unrelated transition writer remains active; new payout worker inactive.

13 protocol/chain tests pass; four trained-checkpoint attack controls (forged reward, forged feedback, forged full log-probs, substituted TOPLOC proof) reject with expected reasons. `trained-checkpoint-tampering-controls.json` records them. Final public experiment, health, chain-activity and tampering-controls reports are signed under first accepted epoch at the retained temporary HTTPS gateway. Gateway service retains audit access; remote pod stays running indefinitely as user requested.

## Latest nonpayable multi-environment milestone — 2026-09-30 21:52 UTC

Three actual epochs completed on the retained remote registered UID131 miner:
Prime Verbatim, Prime Mastermind, and original reasoning-gym count_bits. Each
provided an accepted positive/negative batch, full independent target-model and
environment replay audit, local proposed weights, one genuine training step and
publication of the next immutable checkpoint. Final checkpoint:
`d53bb84e8eddd39b4168800e6744359ae4b399a13e5c8c6676292b1c12b365ef`.
All reports are permanently nonpayable; this experiment made no weight extrinsic.
The older transition service is separate and may still operate independently.

Current evidence: `state/multi-environment/report.json`, `progress.json`,
`status.json`, and cumulative `state/evaluations/*.json` (operator-owned metrics
consumed by affine.io). Same fixed-task before/after means are Verbatim1→1,
Mastermind1→1, count_bits0→0. Those are curated control policies; no training
improvement or autonomous-solving claim follows. A separate fixed original-task
suite is continuously evaluating baseline and newly trained checkpoints, with
honest unsuccessful autoregressive samples retained as valid evaluation evidence.

The active nonpayable service is `affine-multi-environment.service`; stop/start
instructions and exact numerical profile are in `docs/CONTINUOUS_EXPERIMENT.md`.
The immutable source implementation for these three epochs is retained under
`state/source-bundles/`, with an authority-signed public descriptor. Runtime v2
explicitly bounds TOPLOC native bit-extraction threads to avoid its upstream
hardware-concurrency override while retaining strict numerical comparisons.

An additional real 1.7B GPU original-Math inference/replay pilot accepted its honest
trace and rejected five tampering controls, reward0, with exact same-GPU backend
weights reload. Evidence `state/multi-environment/gpu-pilot/report.json`. That is
larger-model proof evidence, not training or cross-hardware compatibility.

## Independently checked wide/replay milestones — 2026-10-01

The root full-ledger auditor now verifies nine completed nonpayable wide epochs
and 27 actual full-model updates, including the first original MRCR common epoch
`nonpayable-gpu-wide-v7-mrcr-1790865913-5`. Its checkpoint is
`aaac517b5a1a39f3fdd78cf2c73adbad62f9f8b94f00793a95f7f8bcf6d3739d`.
All six checkpoint objects (3,426,302,727 bytes) were independently streamed and
hashed from R2. Actual affine.io HTTPS export includes the matching nine epochs
and 200 evaluation records. MRCR heldout mean declined; original correlated
conversation coverage and curated public-shell controls remain disclosed.

The isolated nine-family balanced replay experiment also completed. Independent
root checks authenticated both before/after model jobs and their 384 task rows,
fixed identities, and all 24 public dashboard aggregates. Math improved, Trivia
and Reasoning Gym declined, and nine other families were flat. The replay-aware
continuous auditor preserves fresh-only checks and requires signed current-model
replay provenance with exact consumed target and request bindings. Deployment of
that source is a separate completed-boundary gate.

The versioned equal-token TriviaAbstain candidate probe obtained K1L1 for original
indices 0 and 1 after 4 and 9 attempts. All four retained traces passed separate
fresh model/TOPLOC/original-native verification. Root checked actual source,
snapshot and ZIP bytes. This clears a prerequisite for that new harness, not a
continuous-training or factual-knowledge improvement claim. Original failed
all-positive probe remains preserved. See docs/TRIVIA_ABSTAIN_BALANCED_PROBE.md.

The fresh controlled native Tau2 window has positive/negative full-role proofs,
private R2 cumulative submission and signed final boundary selection. Before
heldout evaluation is in progress. It has no completed optimizer claim yet.
Signing keys and R2 account credentials remain on the operator. All these new
experiments remain nonpayable and make no chain weight submissions.

## 2026-10-01: verified common adapter expansion, staged only

The root independent wide-loop audit confirms ten completed nonpayable epochs and thirty optimizer updates, with two authenticated aborted openings and no demonstrated empty wide epoch yet. The live v7f controller has fourteen environments and is collecting epoch `nonpayable-gpu-wide-v7f-balanced-replay-1790874860-7`. Actual affine.io HTTPS projection matches all ten completed epochs, their checkpoints/UID131 points, and 226 paired evaluation records. These checks do not establish improvement across all environments.

Numina qualified its original index13 public-starter positive/negative pair under the shared adapter, with direct R2 hydration and separately reloaded full-logit/TOPLOC/native verification (`state/numina-model-control/1790874912/root-evidence-check.json`). Pydantic's preserved v1 found only positives; a new public-schema wrong-type v2 qualified original index0 K1/L1 (`state/pydantic-type-model-control/1790875093/root-evidence-check.json`). Neither family has completed a common training epoch yet.

Separate undeployed expansion archives preserve every existing environment/harness and heldout identity: Numina15 archive `2e9a6e6481c8dd0040dfb794a6eef3c584279fdf0c1d2fa9a9c72c73eecb518b`, then Pydantic16 archive `211bdd956d83230c00c8251b4031af7f593f585ee1e7a7e81aa5612d62ed6fa3`. Both preserve existing compute source bytes. Their configs/preparation records are in `state/gpu-wide/numina-v7g-1790875313-*` and `state/gpu-wide/pydantic-v7i-1790875568-*`; subsequent signed empty-recovery control requires a new reviewed source, not modification of either frozen archive.

The separate original native Tau2 common-service epoch completed all sixteen baseline tasks. Root checked all 148 signed role receipts against the exact approved role/task seed formulas (`root-before-seed-receipt-check.json`). Its optimizer, after-evaluation and successor remain pending; the historical heldout digest alone omits agent seeds, so completion still requires explicit before/after signed seed comparison.

## 2026-10-01: current balanced epoch optimizer and independent publication

The current fourteen-environment epoch `nonpayable-gpu-wide-v7f-balanced-replay-1790874860-7` completed eleven actual full-model updates across eleven contributing source families. Its authority-approved remote training job produced checkpoint `0081b0698c0ccc103edfca0506a2c60aeaf9d0a521716889e6a51fdef0a6513b`. Root independently authenticated the job/report and streamed all six checkpoint objects from R2 (3,426,302,727 bytes), checking every file digest without storing weights on the operator disk. Evidence: `state/gpu-wide/nonpayable-gpu-wide-v7f-balanced-replay-1790874860-7-root-checkpoint-publication.json`.

The controller is performing post-training evaluation. Its earlier 224 baseline rows across fourteen environments were independently checked; paired after results and full epoch completion remain pending. The completed ledger remains ten epochs/thirty updates until those gates pass; the current optimizer would bring the cumulative total to forty-one. This publication alone establishes neither improvement nor the prospective sixteen-environment empty/recovery sequence. Native Tau2's separate after-evaluation is also still running. No chain weights were submitted.

## 2026-10-01: balanced epoch finalized; sixteen-family controlled window opened

Independent full-ledger verification now accepts eleven completed wide epochs and forty-one actual optimizer updates. The first balanced epoch consumed eleven fresh/replay family pairs; the auditor reconstructs their signed provenance and binds each update to the positive/negative rollout's environment index. Historical replay carries an environment definition rather than a fresh batch, so index validation now also checks agreement between both rollouts and any fresh-batch index. Four mutation controls plus sixteen replay and ten ledger controls pass; the actual ledger rerun passes.

The fourteen fixed heldout suites (sixteen tasks each, independently checked before and after) show three improved means, two declines and nine unchanged. Math fell from 0.5 to 0.375; Trivia fell from 0.5 to 0.4375. This does not establish improvement across all environments. Actual affine.io HTTPS data matches all eleven completed epochs and 254 paired evaluation records.

At the completed idle boundary, the service activated source `6c9df7739a910c494eff3b2c8fc7996b3277aea967b6e1573f5f2fe14bf03be9` and opened `nonpayable-gpu-wide-v7j-replay-recovery-1790878859-8`. Root authenticated the actual R2 first manifest: sixteen environments, exact deadline 1790879759, signed mining disabled, no mining jobs at inspection. Real empty completion and its subsequent trained Numina/Pydantic recovery remain pending. Evidence: `state/gpu-wide/root-sixteen-empty-opening-check.json`.

## Durable replay-source reproduction

The independent auditor now tries the immutable `public/source-bundles/{signed-sha256}.tar.gz` locator when an archived manifest's key hint is stale or missing, before relying on an expiring signed URL. It retains exact signed size/digest and complete module-inventory checks; access-denied errors remain failures. Root removed the ephemeral URL from a copy of the authenticated completed v7f manifest and successfully fetched its 9,149,057-byte archive with all forty-seven source digests matching the signed job. Evidence: `state/gpu-wide/root-v7f-durable-source-check.json`. Seven source-route controls, ten ledger controls and the actual eleven-epoch ledger pass. No frozen manifest or active compute source was modified.

## 2026-10-01: Tau2 recovered heldouts published without rewriting failures

The separate native Tau2 common epoch completed one full-model update and all
sixteen after-checkpoint heldouts with five explicitly separate recovery attempts.
The original eleven-success/five-error evaluation remains unchanged. Its signed
completion and all inventoried evidence were independently checked; the fresh
successor also passed independent native/model verification without another
optimizer step. Both baseline and completed after means are zero.

The dashboard now exposes three authenticated derived records: sixteen baseline,
eleven original partial, and sixteen completed with explicit recovery. A separate
signed approval binds the unchanged completion and paired metadata contracts;
the derived cohort includes the actual agent seeds omitted by the historical
dataset digest. Seven export mutation/privacy controls and ten dashboard checks
pass. Actual affine.io HTTPS JSON returned all three exact records (HTTP200),
including original error and recovery counts. Evidence:
`state/native-tau2-common-live/epoch-1790869388/root-public-dashboard-recovery-check.json`.
This does not establish quality improvement, wide empty-epoch completion, or
chain payouts. The sixteen-family wide evaluation remains a separate live job.

## 2026-10-01: original Wikispeedia native controls qualified

Four original Wikispeedia tasks now have real positive/negative tool trajectories
and fresh native replay through the shared environment adapter. Thirty-two
mutation controls rejected forged observations, rewards, task identities and
source hashes. Root independently reran all eight retained trajectories and
checked 4,610 extracted SNAP graph/article files against the recorded archives.
The initial tool-prefix mismatch attempt is preserved; the successful v3 records
bind the full environment definition. No model/TOPLOC/training/remote epoch claim
is made. Evidence and reproduction: `docs/WIKISPEEDIA_NATIVE_CONTROLS.md`.

## 2026-10-01: original standard UUIDCTF native controls qualified

Two original standard-difficulty UUIDCTF tasks have successful and unsuccessful
real sandbox trajectories using a public-file forensic solver. The unchanged
original grader returned 1/0 for each pair. Eight forged observation/reward
mutations were rejected; a separate root process reran all four native
trajectories and recorded the actual image digest and observed owned-container
teardown. Hidden expected answers are not inputs to the solver. No model,
TOPLOC, remote epoch or common training is claimed. Evidence and reproduction:
`docs/UUIDCTF_NATIVE_CONTROLS.md`.

## 2026-10-01: Wikispeedia model-cohort prerequisites

The original Wikispeedia snapshot now has twenty native-qualified tasks, scoped
for four mining indices and sixteen separate heldouts. Forty positive/negative
native controls and 160 mutations passed; both shared candidate branches also
ran through the common tool dialect on all twenty tasks. Hash-checked R2
tokenizer/configuration files show both registered harnesses fit the retained
control paths within the pinned model context, with no model weights downloaded
or model executed. Evidence: `state/native-wikispeedia-candidate-cohort20-v1/root-model-budget-preflight.json`.
Mixed-choice context coverage, remote model/TOPLOC qualification and common
training remain pending. No active signed GPU source bundle was changed.

## 2026-10-01: Wikispeedia mixed-path context qualification

Exhaustive original native replay now covers all 384 public good/bad choice paths
across the twenty-task cohort. Both legacy harnesses overflow the pinned context
on eight turn contexts. The separately versioned `text-tools-window-v1` retains
the initial messages and latest complete messages in model context while keeping
full history for replay verification; all 384 paths fit with zero context
failures. Five focused controls pass, including rejection of an altered old
observation omitted from the current model prompt. The reproducible public CLI
was rerun against original tools and grader, and root independently checked its
entire retained trace population. This establishes native/context prerequisites,
not remote model/TOPLOC or training coverage. See `docs/WINDOWED_HARNESS.md` and
`docs/WIKISPEEDIA_NATIVE_CONTROLS.md`.

## 2026-10-01: RCore prospective package review

The isolated v2 RCore model resource guard passes eleven controls. Root's archive
review found that the preparation tarball predates final path-dependent
`environment.json` and the signed model plan: it has 8,797 files, one older source
hash and no plan, whereas the final signed profile protects 8,798 files. This is
a bootstrap/finalization distinction, not evidence of model execution. The old
tarball remains preserved; a separate final sealed packaging artifact and an
independent actual remote inventory check are required before approval. No RCore
GPU inference or training has been launched. The live wide evaluation and its
normal Numina/Pydantic recovery keep priority.

Root subsequently compared a fresh SSH inventory of all 8,798 remote files with
the signed profile: exact membership, sizes and hashes match. The complete
bootstrap-plus-finalized-fixtures audit also passes, including both authority
signatures, the original snapshot, approved checkpoint, all 49 source pins and
48 unchanged v7k modules. The original bootstrap discrepancy remains recorded;
a final self-contained archive is still being prepared. GPU launch remains
withheld while the actual wide evaluation uses the retained GPU.

## 2026-10-01: Wikispeedia prospective model contract

A separate model search probe now binds the original twenty-task snapshot,
four mining indices, per-task public graph candidates, signed windowed harness,
checkpoint and numerical/source policies. Five contract controls pass, including
refusal of heldouts, payable scope, altered context budgets, preloaded providers
and an unapproved cache. Root ran a fresh original native preflight across all
four mining tasks and 4,610 original resource files, then confirmed rejection of
a substituted candidate. Remote resources-only staging also completed; model
plan/resource guard preparation is pending. No GPU model, TOPLOC, training or
new coverage claim is made. See `docs/WIKISPEEDIA_MODEL_QUALIFICATION.md`.

## 2026-10-01: real wide empty epoch completed

The sixteen-environment controlled empty epoch completed naturally. Independent
full-ledger verification now accepts eleven trained wide epochs, three preserved
aborts and one completed empty epoch. Root authenticated both original evaluation
jobs and manifests, their 10,800-second leases and all 512 model/native-verified
heldouts. All sixteen before/after cohorts have identical indices, seeds, task
hashes, runtime, harness and means at unchanged checkpoint `0081b069...`.
There are no mine/train jobs, optimizer metrics, accepted batches or proposed
payouts for the empty epoch. Evidence:
`state/gpu-wide/root-v7k-completed-empty-pair-check.json`.

The same controller subsequently opened normal round ten and a fresh remote
mining worker is confirmed live. This proves natural continuation after the
empty window; completed normal training recovery remains pending. An existing
separate native SQL evaluation also uses the retained host and is preserved.
No new qualification GPU job or chain submission was started.

## 2026-10-01: Wikispeedia signed remote stage reviewed

Root authenticated the separate model plan and resource profile, checked every
one of the 1,391 regular source-archive files and all 108 current compute-module
pins, and independently hashed the actual 6,364 protected remote files via SSH.
Exact membership, sizes and bytes match. The original twenty-task snapshot,
checkpoint, native preparation and imported ops helper are pinned. Seven
resource-guard controls pass. The Verifiers namespace is controlled; full
transitive dependency closure is explicitly not claimed. Evidence:
`state/wikispeedia-window-model-stage-v1/root-signed-stage-actual-byte-check.json`.
This qualifies preparation only. Model K/L, TOPLOC, numerical replay and common
training remain pending behind the normal wide recovery priority.

## 2026-10-01: post-empty private R2 submission checked

The first normal post-empty mining job completed successfully under its original
signed job and source/runtime/lease bindings. Root independently read the actual
46,230,234-byte R2 staging object and checked its SHA, atomic completion metadata
and arrival within the signed window. Bounded decoding confirms one Numina index
13 batch with K1/L1, 484 float32 probability rows over all 49,152 vocabulary
entries, finite normalized log probabilities and proof framing. This checks
transport and artifact structure; it is not fresh inference verification or
optimizer recovery. The actual controller remains alive and honors the original
deadline before freeze. Evidence:
`state/gpu-wide/root-post-empty-private-R2-batch-check.json`.
