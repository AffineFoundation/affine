# Original environment compatibility

The active source list is the actual live datagen `sources.toml`, not the count of package directories: 45 active sources, 9 disabled sources. Live datagen2 source hash: `ce522ed366a2187c1d2327171f2a77fb5ee933d2f83bbd74ce6f6a338ace5df9`.

Trusted original wrappers are vendored under `subnet/vendor/legacy/rollouts/envs`; upstream research environments are vendored at revision `b10db7640be3051650eef759e6ed80ddcadae220` under `subnet/vendor/research`. Original live verifiers checkout was `a298bcfe4a3a410b7287254d61a65947906c6a89`; the installed portable package exposes `verifiers.v1`. Each spec pins the adapter, bundled wrapper/upstream source bytes, config and recorded dependency versions. Production datasets must additionally be materialized with `snapshot_spec` into operator-trusted immutable task snapshots; mutable upstream dataset names alone are not a production pin. Snapshot paths are relative and portable.

## Current recorded coverage

[environment-coverage.json](environment-coverage.json) is the latest public snapshot of the operator execution matrix. It reports each source’s milestone or blocker code separately from imports, reset, original rewards, native tools, remote proofs, replay and training. Rebuild it with `.venv/bin/python -m ops.export_environment_coverage`; its source-matrix SHA binds the exact underlying evidence inventory. The exporter does not rerun the experiments or certify production readiness. Raw exceptions, upload capabilities and private nested probe metadata are excluded.

The table below preserves the initial import probe. Some of its missing-package errors were resolved later; use the JSON snapshot for current flags. The controlled native Tau2 negative-only experiment is described in [NATIVE_TAU2_MODEL.md](NATIVE_TAU2_MODEL.md) and does not imply production K/L batches or training coverage.

## Actual conformance evidence

`tests/test_environments.py`: 5 passing tests. Original Verbatim task reset, public-copy reward 1 and invalid-copy reward 0; original reasoning-gym `count_bits` reset and correct public-input-derived answer reward 1; original When2Call native tool execution, tool observation, final reply and reward 1. Source fingerprint corruption is rejected. No autonomous model success is implied by these oracle conformance tests. Independent remote proof and training evidence belongs to the controller's remote epoch reports, not this import matrix.

When2Call uses the original `nvidia/When2Call` revision `0582f7749df63a96fdc3070932e83e72396ace53`, train split, 9000 original rows. Config is `build_spec('affine_when2call', {}, num_samples=2, max_turns=4)`; sample limit belongs to adapter, not dataset taskset config. Four tau2 sources need the original user-simulator/orchestrator bridge, not a single assistant-turn substitute, and the generic direct-tool adapter explicitly refuses to substitute for that native loop.

## Historical initial source import probe

Import success only establishes module/class availability; it does not establish dataset download, reset, tool execution, scoring, inference proof or training compatibility. Resource-heavy repository and terminal tasks still need isolated image/task/runtime validation.

| Source | Module/class | Harness category | Import result |
|---|---|---|---|
| scaleswe | `scaleswe_v1.ScaleSWETaskset` | multiturn_sandbox | imported |
| swerebench_v2 | `swerebench_v2_v1.SWERebenchV2Taskset` | multiturn_sandbox | imported |
| r2e_gym | `r2e_gym_v1.R2EGymTaskset` | multiturn_sandbox | imported |
| multiswe | `multiswe_v1.MultiSWETaskset` | multiturn_sandbox | imported |
| swesmith | `swesmith_v1.SWESmithTaskset` | multiturn_sandbox | imported |
| swelego | `swelego_v1.SWELegoTaskset` | multiturn_sandbox | imported |
| terminal_lego | `terminal_lego_v1.TerminalLegoTaskset` | multiturn_sandbox | imported |
| terminal_bench_2 | `terminal_bench_2_v1.TerminalBench2Taskset` | multiturn_sandbox | imported |
| nl2repobench | `nl2repobench_v1.NL2RepoTaskset` | multiturn_sandbox | imported |
| affine_nl2lib | `affine_nl2lib_v1.NL2LibTaskset` | multiturn_sandbox | imported |
| affine_math | `affine_math_v1.MathTaskset` | singleturn_text | imported |
| affine_wiki | `affine_wiki_v1.WikiTaskset` | tool | blocked — No module named 'wiki_search_v1' |
| affine_agent | `affine_agent_v1.AffineAgentTaskset` | tool | imported |
| affine_when2call | `affine_when2call_v1.When2CallTaskset` | tool | imported |
| affine_tau2 | `affine_tau2_v1.AffineTau2Taskset` | multiturn_tool_simulator | blocked — No module named 'tau2' |
| affine_tau2_synth | `affine_tau2_synth_v1.AffineTau2SynthTaskset` | multiturn_tool_simulator | blocked — No module named 'tau2' |
| affine_tau2_gen | `affine_tau2_gen_v1.AffineTau2GenTaskset` | multiturn_tool_simulator | blocked — No module named 'tau2' |
| affine_kb_synth | `affine_kb_synth_v1.AffineKBSynthTaskset` | multiturn_tool_simulator | blocked — No module named 'tau2' |
| affine_logic | `affine_logic_v1.LogicTaskset` | singleturn_text | imported |
| affine_trivia | `affine_trivia_v1.TriviaTaskset` | singleturn_text | imported |
| affine_trivia_abstain | `affine_trivia_abstain_v1.TriviaAbstainTaskset` | singleturn_text | imported |
| affine_popqa_abstain | `affine_popqa_abstain_v1.PopQAAbstainTaskset` | singleturn_text | imported |
| affine_ifeval | `affine_ifeval_v1.IFEvalTaskset` | singleturn_text | imported |
| affine_science | `affine_science_v1.ScienceTaskset` | singleturn_text | imported |
| affine_scitext | `affine_scitext_v1.SciTextTaskset` | singleturn_text | imported |
| affine_unscramble | `affine_unscramble_v1.UnscrambleTaskset` | singleturn_text | imported |
| affine_prolog | `affine_prolog_v1.PrologTaskset` | multiturn_sandbox | imported |
| affine_wikispeedia | `affine_wikispeedia_v1.WikispeediaTaskset` | tool | imported |
| affine_tmax | `affine_tmax_v1.TMaxTaskset` | multiturn_sandbox | imported |
| affine_eog | `affine_eog_v1.AffineEnterpriseOpsTaskset` | tool | imported |
| affine_numina | `affine_numina_v1.AffineNuminaTaskset` | multiturn_sandbox | imported |
| affine_sql | `affine_sql_v1.SqlTaskset` | multiturn_sandbox | imported |
| affine_autobench | `affine_autobench_v1.AffineAutomationBenchTaskset` | tool | blocked — No module named 'automationbench' |
| affine_uuidctf | `affine_uuidctf_v1.AffineUUIDCTFTaskset` | multiturn_sandbox | imported |
| affine_i3code | `affine_i3code_v1.I3CodeTaskset` | singleturn_code_sandbox | imported |
| affine_scicomp | `affine_scicomp_v1.SciCompTaskset` | singleturn_code_sandbox | imported |
| affine_i3math | `affine_i3math_v1.I3MathTaskset` | singleturn_text | imported |
| affine_deshuffle | `affine_deshuffle_v1.DeshuffleTaskset` | multiturn_sandbox | blocked — No module named 'deshuffle_papers' |
| affine_rgym | `affine_rgym_v1.RGymTaskset` | singleturn_text | imported |
| affine_rcore | `affine_rcore_v1.RCoreTaskset` | singleturn_text | blocked — No module named 'reasoning_core' |
| affine_pydantic | `affine_pydantic_v1.PydanticTaskset` | singleturn_text | imported |
| affine_verbatim | `affine_verbatim_v1.VerbatimTaskset` | singleturn_text | imported |
| affine_oolong | `affine_oolong_v1.OolongTaskset` | multiturn_sandbox | imported |
| affine_mrcr | `affine_mrcr_v1.MRCRTaskset` | multiturn_sandbox | imported |
| affine_docqa | `affine_docqa_v1.DocQATaskset` | singleturn_text | imported |

## Dependency and resource blockers

37/45 taskset classes import. Missing packages: wiki_search_v1; tau2 for all four tau2-family sources; automationbench; deshuffle_papers; reasoning_core. Original package/deployment declarations supply their Git revisions; installing an unrelated similarly named module is not acceptable. Heavy SWE/terminal/SQL/prolog tasks require original datasets and Docker images, disk and bounded sandbox execution. Gated HF datasets require authorized credentials. Agent/eog/wiki environments may require original search/API services.

The generic adapter currently keeps model-generated shell commands inside Docker, with bounded commands, execution timeout and output. Native When2Call is a safe simulated trusted Python tool. General sandbox images are currently trusted task image names, not immutable image digests; explicit network restrictions and production runtime hardening remain outstanding. Dynamic dataset-provided Python graders must not be executed on the host without independent trust/sandbox review.

## Callable adapter boundary

`EnvironmentSpec.from_dict`, `build_spec`, `snapshot_spec`, `create_session`; spec exposes `id`, `version`, `config`, `max_turns`, `max_output_tokens`, `num_samples`, `source_hash`. `session.reset(index,seed)` returns original messages, original tool schemas, task hash and task name. `session.step({'text': ..., 'tool_calls': [...]})` returns observations, done, reward and positive/negative/neutral classification. `session.close()` releases tools/runtimes. Uploaded miner artifacts never select import paths, reward hooks or datasets. Legacy Mastermind stays a separate compatibility adapter, preserving old observation/reward semantics.

Disabled sources: swerebench_main, swebench_multilingual, swebench_verified, swebench_pro, affine_notool, affine_needle, affine_terminal_gen, affine_longcot, affine_gdpval.

## Expanded original dataset probes

Eight additional original sources reset and execute their original reward hooks with an invalid negative control (reward 0): trivia, unscramble, logic, science, scitext, ifeval, i3math, docqa. `state/environment-reset-matrix.json` records task names and task hashes. All eight original first-task snapshots were materialized into `state/original-task-snapshots`, content-pinned, reloaded and regraded with matching task identities. No positive miner solving claim is made from these tests. The comprehensive 45-source status matrix is `state/multi-environment/environment-execution-matrix.json`; proof/training entries require actual independent run evidence.

Four more original reset/reward controls pass: trivia_abstain, popqa_abstain, math and MRCR. MRCR used actual local Docker runtime and original reward, with bash tool schema. Oolong exceeded the bounded50second reset budget and needs a larger resource trial. Six remote original sources currently pass genuine unconstrained target-model generation, TOPLOC/full probability capture, separately reloaded checkpoint verification and original reward replay: i3math, logic, science, scitext, trivia, unscramble. These produce genuine negative rollouts (reward0); no successful solving or training is claimed. The checkpoint was retained earlier Mastermind-trained `2b80bfabb0b54a409d8fb4df832112d208773d40391a76e5e2475d3306f31166`, not a new initial model. Original DocQA first task is154715prompt tokens and exceeds this model8192context budget; it was not shortened or replaced. IFEval has an explicit heterogeneous typed task snapshot blocker after runtime dependencies were installed; unpinned original reset/reward was successful.

## Fixed independent evaluation suite

`state/multi-environment/fixed-heldout-environments.json` lists10 supported original sources with4 genuine task rows each. All40 original task reset/reload probes pass. Fixed heldout ordinals are2 and3, separate from training ordinals0 and1. Direct `EnvironmentSpec` JSON files are `state/original-task-snapshots/fixed4-SOURCE.spec.json`; their referenced task snapshots have mode0400 because they contain private reward labels. Public metrics must expose only task IDs/hashes and metrics. Snapshot task data must not be published as metrics.

## Remote coverage evidence

Ten distinct original sources now have genuine unconstrained target-model16token rollout plus TOPLOC/full probabilities and separately reloaded-checkpoint/environment verification: math, MRCR, popqa_abstain, trivia_abstain, i3math, logic, science, scitext, trivia and unscramble. These are genuine negative examples with zero task reward, not successful solving claims. In addition MRCR has a real2turn Docker bash trace with model-weighted curated safe candidates (`printf conformance`, `pwd`) and full proof/replay; label this curated policy, not autonomous problem solving. Artifacts are stored in `state/multi-environment/raw-proof-artifacts`; evidence envelopes bind exact checkpoint file hashes, environment spec and rollout/probability file hashes.

Checkpoint/filemap digest is2b80bfabb0b54a409d8fb4df832112d208773d40391a76e5e2475d3306f31166, the retained earlier Mastermind-trained135M checkpoint. The remote coverage runs did not use the newly-trained233e checkpoint. Upstream initial model revision was not independently recovered by this probe; exact retained weights are bound by the full allowlist. Runtime: CPU,float32,eager attention,torch2.14.0,transformers5.14.1,verifiers0.3.1,torch/native pools2; MKL compatible mode,ATEN default,ONEDNN SSE41. The retained RTX3090 GPU is available (CUDA13.0,driver595.58.03) but these coverage runs did not change or migrate GPU jobs. No abandoned manifest CLI existed during cleanup; no process was killed.

CPU coverage profile correction: these earlier runs initialize Torch and native environment pools to2, but TOPLOC v1 native bit extraction used hardware_concurrency and could override later OpenMP/Torch thread counts. The actual later count was not recorded; these reports are explicitly tagged cpu-float32-eager-v1-unbounded-toploc. Successful full probability/inference/environment replay evidence remains genuine, but does not establish a continuously bounded thread profile. Subsequent controller revisionv2 explicitly pins TOPLOC parts threads. The final standalone1.7B CUDA pilot explicitly pins parts threads2 and passes its independent reload and tamper controls.

Latest evidence (2026-09-30 22:30 UTC): the source-specific IFEval RLVRTask/IFEvalRowTask snapshot configuration bridge is fixed, with unchanged active source pins since the coordinated maintenance window. The actual original lowercase IPv6 public-constraint control scores 1 and its uppercase variant scores 0. `state/multi-environment/progress.json` records five fully accepted, audited and trained mock epochs: Verbatim, legacy Mastermind, count_bits, native When2Call and IFEval. These are curated conformance controls, with independently reported heldout results, rather than claims of general autonomous task solving.

`state/multi-environment/scaleswe-proof-pilot/` now contains a real original ScaleSWE task, two target-model-weighted safe bash actions, complete TOPLOC fingerprints and probability arrays, and successful separately reloaded checkpoint plus Docker environment replay. This negative-reward trace proves sandbox/tool/proof compatibility; it does not solve the issue or train that source. The probe used checkpoint 2b80bf... and the explicitly bounded CPU v2/native-TOPLOC-two-thread profile.

The former 55-second resource probes were insufficient to establish persistent external blockers. Longer actual retries now load/reset MultiSWE, SWElego, NL2Lib, and (after installing its declared Harbor 0.21.0 extra without changing verifiers) TerminalBench2. Longer retry evidence is in `state/multi-environment/original-resource-retries.json`. Original TMax Git 8b38d35b53271a5f955dfc5dd8197d562cebf46e has been checked out into the operator cache; task-specific Docker image building remains separate. Dataset-provided graders/tool modules are not executed on the host during resource inspection.

All five previously timed-out sources now have actual reset evidence after longer budgets or exact task snapshot loading. UUIDCTF took 141.8 seconds; its procedural enumeration produces different randomized TaskData on separate loads, so the manifest must bind an operator snapshot before replay. Oolong's original 16k context belongs in the sandbox document file, not the model prompt: the first original test16k row is index200, loaded byte-for-byte from cached parquet revision f0d59eaf0febf130664cfceb710436c8e3216b2b. It now passes two-turn native bash/TOPLOC/probability/reloaded-verifier replay on the retained pod; model prompt276tokens, reward0. Evidence: `state/multi-environment/oolong-proof-pilot/`.

DocQA's first twelve actual epoch1 task indices were scanned without truncation or substitution. All require56,192–154,622 target-tokenizer prompttokens, beyond the current8192-context pilot. Per-index public IDs/counts are in `state/multi-environment/docqa-original-context-scan.json`; this establishes a bounded-scan context gap, not universal impossibility of the entire source.

Previously missing external original packages were installed privately without changing verifiers/model dependencies. Exact Git resolutions, versions and installed Python-file hashes are recorded in `state/multi-environment/original-upstream-dependency-pins.json`. AutomationBench, Deshuffle and ReasoningCore now load/reset original tasks. All four Tau2 families load actual original task data after installing pinned upstream tau2-synth798589e02ca91ea61e85557eb672be0a915592eb, but require the faithful user-simulator/orchestrator harness; direct replay is deliberately rejected. GeneralAgent and EnterpriseOps remain subject to isolated tool-server execution, because task-provided Python must not execute on the host. External installed libraries are recorded in evidence but are not yet included in the adapter's signed source-footprint calculation; portability and footprint integration require the next coordinated source-pin maintenance window.

Latest original taskset import retry passes45/45 (`original-import-retry-matrix.json`). The original TMax first task now passes a complete native sandbox/TOPLOC/probability/reloaded-environment verification trace. Its unmodified original Git8b38d35... Dockerfile was built on the retained pod, image digest10c96a0cf76e4b242dbfa9894c099f6b9fe6d39142a099f5ad50cbd22dd49514 (3,721,483,326bytes). The negative two-tool-turn trace uses497prompttokens and took23.09seconds after warm build. `tmax-proof-pilot/evidence-envelope.json` binds the snapshot, checkpoint and image. Harbor snapshot data embeds an absolute host grader/tests path; we reproduced that path with original taskfiles unchanged for this conformance run. Production needs a signed portable task-resource mapping.

Wiki now resets its genuine original corpus and Chroma-based native search/read/view tools. SWE-smith's exact original first Python task was extracted from its pinned parquet at revision77cab9055d42ab4a5c25c89a8f937096db13558e, same original taskhash55f063f85fd855b6a6dceecb6949a34ac38f86457e59dd7f7532495689842066. Its remote original image and setup reset pass; grader dependency resolution and full trace verification are separate statuses. The default all-language dataset staging exceeded local headroom, so only owned disposable Arrow caches were removed (filelist/bytes in owned-probe-cache-cleanup.json); original parquet sources and all production data were preserved. Resource retry code now aborts its own child if free disk falls below2GiB.

SWE-smith full remote proof verification now passes31.11seconds after resolving trusted upstream grader dependencies: genuine first oauthlib task, two native bash turns, original grader outcome0, full TOPLOC/full probabilities, separately reloaded approved checkpoint, and original environment replay. Image digest21e580c0d78b89e4d2f47ed79ca97d6e111ce5399069a735048f8457876eb5d3,3,223,249,275bytes. Raw evidence and original dataset/parquet/source/checkpoint pins are in `state/multi-environment/swesmith-proof-pilot/`. Current stage totals are45original imports,36genuine reset sources,18sources with real proof or accepted/trained mock pipeline. Counts include negative conformance traces; they do not imply eighteen solved environments.

Final bounded coverage summary:45/45 actual original taskset imports,37/45 sources with genuine reset,19/45 with full remote proof verification or actual accepted/trained mock epoch. Numina now also passes its original Mathlib sandbox setup, two safe native bash actions, full TOPLOC/probability verification and original Lean grader replay in173seconds on the retained pod. Original image digest4e256bfd5056d7b82aaf2c39322b42e0a6e7a0b4c3965e22a2ba6041c3546abc (8,257,048,426bytes). Its outcome0 remains an unsolved negative conformance trace. Raw signed-input/weight/source/image evidence is in `numina-proof-pilot/`. Remaining eight reset gaps are explicit: two unavailable Docker source images, four native Tau2 simulator-harness requirements, and two isolated dataset-authored tool-server requirements. No arbitrary55second timeout remains the sole reason for a gap. `environment-coverage-summary.json` separates actual stages and remaining production pin/portability work.

Oolong now additionally has verified positive and negative public-algorithm rollouts. Fixed original test16k global rows215/214/216/217 are pinned in `state/original-task-snapshots/oolong-date-fixed4.tasks.json`; their distinct original questions share the dataset's original context. Training candidate ordinal0 asks how many dates occur exactly twice. Both candidates read the complete sandbox context and run a date Counter; one writes the exact count, the other count+1. The target model chose each across six seeds. All six two-turn native-tool traces passed full TOPLOC/probability verification with a separately reloaded approved2b80 checkpoint and original Docker replay, and rejected changed claimed rewards. Official scores were1.0 positive and0.75 negative under success threshold1.0. This is a curated public-algorithm control, not autonomous semantic solving or a training result. Raw artifacts and byte receipts: `state/multi-environment/oolong-date-proof-pilot/`; tested policy: `state/multi-environment/oolong-date-harness.json`. Original216/217 remain separate heldout questions, not yet solved by this date-frequency policy.
