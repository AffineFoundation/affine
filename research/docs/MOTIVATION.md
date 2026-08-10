# Motivation — Distillation-KL Scoring for Albedo (SN97)

_Last updated: 2026-08-03. This file is the fixed point of the research program. Re-read it
whenever resuming work. Results go to RESEARCH_LOG.md; this file holds the goal._

> **Status note (2026-08-10).** Two things changed after this file froze.
> (1) Production scoring forked to **Reason v3** (`weight_version_key=3`):
> score = raw Λ2, crown = paired 3σ — the S\* v2 mix/gates/δ described in
> "Current production candidate" below are **retired** (kept as telemetry in
> every verdict). Current contract: `affine/affine.toml [duel]`, AGENTS.md §2.
> (2) The isomorphism claim **inverted on the live SN120 board** (RT-7/A12 in
> REDTEAM.md: ρ=−0.42, all three S-crowned kings 0/25) — the public claim is
> now a **distillation meter**, not coding isomorphism; the +0.758/+0.844
> results below remain true of the Albedo panel only. Asymptote/equilibrium
> framing and the assumption ledger live in `EQUILIBRIUM.md`.

## The problem, observed in production

Albedo (SN97, albedo.tech) is a king-of-the-hill Bittensor subnet: miners fine-tune
Qwen3.6-35B-A3B, challengers duel the king on coding-trajectory turns, winner takes emissions.
The current scoring is an LLM-judge checklist (GLM-5.2 writes 50 yes/no questions anchored on a
GLM-5.2 reference trajectory; judge answers per side; 3% win margin).

**The judge score is being Goodharted.** Real benchmark scores (swe-rebench, 500 tasks) of the
crowned kings, from albedo.tech/data/model-scores.json (2026-08-02):

| model | swe-rebench |
|---|---|
| GLM-5.2 (reference/teacher) | **73.8** |
| genesis (Qwen3.6-35B-A3B) | **58.2** |
| king-I (first dethroner) | 38.4 |
| kings II–L (typical) | 26–37 |
| worst reigns (XLVI, LI, XC, CI) | 11.6–16.0 |

Every single king that won the duel is far below genesis on the real benchmark. The scoring
function crowns models that game the checklist, not models that code better.

## The proposed replacement: non-overfittable distillation scoring

Score miners by how well they *distill the teacher* (GLM-5.2), using thought-injection
cross-evaluation instead of a scalar judge. For a turn context x, each model M factors into
thoughts z ~ M_z(·|x) and action y ~ M_y(·|x, z). With teacher C and miner A define:

- π_M(y|x) natural policy; π_{M←N}(y|x) = M's decoder driven by N's thoughts ("A#C", "C#A").
- Six divergence terms (KL where density access exists, else sample-based MMD):
  L1 KL(C‖A) behavioral; L2 KL(C‖C←A) thought sufficiency (A's thoughts must make C act like C);
  L3 KL(C‖A←C) decoder fidelity; L4 KL(A‖C←A) faithfulness; L5 KL(A‖A←C) receptivity;
  L6 KL(C←A‖A←C) swap symmetry. Score S(A) = −Σ wᵢ Lᵢ averaged over thought-causal turns.

Key properties (established in design discussions):
- L2 cannot be maximized by style: A controls only z, C's frozen decoder produces the output.
- The objective identifies A up to reader-equivalent thoughts (z ~_C z' iff C acts identically).
- Turns must be **thought-causal** (KL(C*‖C#∅) large) or terms lose discrimination — filter D.
- Dense per-token signal via teacher forcing (self-hosted C gives exact byte-level KL).
- Copier protection: score vs fixed C ⇒ a king-copy draws from the null of the paired test.

## The claim to prove (the "final point")

**S is benchmark-isomorphic under Goodhart pressure, on the capability axis exercised by the
turn set D**: for models below the teacher's benchmark level — including adversarially trained
ones — higher S implies higher downstream score on the benchmark that measures the *same
capability D probes*, even though S never touches benchmark tasks. D is chosen; S inherits its
target axis from D. Our D is SWE-style coding trajectories, so the operative benchmark is
swe-rebench.

The decisive experiment exists because production already produced the adversarial artifacts:
~31+ king checkpoints with measured swe-rebench scores (11.6–58.2 span), all crowned by the
gameable judge. If our S ranks them in benchmark order (high Spearman) — while the judge score
that crowned them does not — the claim is proven on real Goodharted models, not simulations.

**Domain-selectivity is a feature, not a caveat.** tau2 measures a different axis (tool-use /
customer-service dialog). On the 11-king overlap, swe-rebench and tau2 are themselves nearly
orthogonal (ρ=−0.08), so *no* single scorer can track both; S tracks the one its turns exercise
(S vs swe +0.93, S vs tau2 −0.06). The claim is therefore not "S measures general capability"
but "S is a **programmable** capability meter: pick D, get isomorphism to the matching benchmark."
Decisive falsification test (not yet run): build D_tau2 from tool-use turns and predict
S_tau2 ≅ tau2 and S_tau2 ⊥ swe-rebench.

## Current production candidate (2026-08-03)

After E-KINGS (10 kings × 200 turns) + RT-1/2/2c/4 + bank + soft-idents red-team:

**Hybrid S\*** — rank by teacher-anchored mix; invalidate with two miner-level gates; duel at 3σ.

- Ranking term: S = E[Λ2 + w · L1lift] with **w=1.0**,
  Λ2 = lpC(y_C|z_A) − lpC(y_C|∅), L1lift = lpA(y_C|z_A) − lpA(y_C|∅).
  Full KING_BENCH (**n=30**, S\* v2): ungated Spearman **+0.758** (p=1.2e-6; was
  +0.862@19 / +0.844@15; judge ≈ **+0.3** ns). Hybrid gates **+0.799**@15 (bank@80;
  many mid kings at γ_bank knife-edge 0.075). Under clip, genesis #2 vs II by 0.002 —
  δ=0.05 duel floor is load-bearing. Live duels: VII/XCIX/LI none dethrone; I/II
  live z=+0.20/−0.03 under S\* v2.
  **tau2 not tracked (by design)**: S vs tau2 ρ=−0.06 @11, but swe vs tau2 is itself ρ=−0.08
  on this population — the benchmarks disagree, so S faithfully tracks its own axis (D=coding).
  RT-2c para win ~21% before bank gate.
- Causality+leakage gate γ=0.30: closes exact stuffing (0%) and silence (≤5%).
- Prior-bank positivity gate γ_bank=0.08: closes paraphrase stuffing (frac_bank=0).
- Duel: challenger dethrones iff paired mean(S_c − S_k) > 3·SE (RT-4: copier |z|≈1.5).

Abandoned: soft-idents (pad-gameable). Prefer higher L1 weight over z-text regularizers.

Open risks:
- Short-style I/II: bank-valid with higher mean S on 80-turn slices (z=1.35/1.72);
  bank gate does not reject; 3σ duel is load-bearing. Mid/high/weak checks all
  subnet-correct (VII/XCIX/LI).
- **S* v2 SHIPPED** (2026-08-03): clip0.1 + r∈[1,4] + δ=0.05 effect floor in harness + affine.
  Offline re-freeze: ungated ρ=+0.862@19; full n=30 → **+0.758**. Hybrid +0.799@15 with bank@80.
- **A11**: effect-floor δ=0.05 closes n*-scaling; 3σ alone was insufficient (II n*≈244).
- tau2 is a **different axis, not a failure**: swe⊥tau2 (ρ=−0.08) on the overlap, so S tracks
  the capability D exercises (coding). Reframed as domain-selectivity, above.
- **RT-3b LIVE CLOSED**: I z=+0.20 / II z=−0.03; live r∈[1.14,1.42].
- **Second-teacher CONFIRMED**: Qwen3-32B vs GLM-Air rank ρ=+0.943 (n=6). Teacher-stable.
- **More kings CLOSED**: n=19→30; XC mid-S/low-swe soft outlier. e6 terminated.
- **D_tau2 probes (three negative)**: bash −0.881; native −0.738; **force-only −0.257** (n=6,
  still not +tau2; S∽swe +0.71). Programmability stays an interpretation.
- **RT-6 MITIGATED (ops)**: dated leave-repo-out STABLE (ρ_late=+0.845); `corpus_epoch` refresh.
- **Affine gaps CLOSED for go-live blockers**: `Wejh/affine-turns` pinned+uploaded; e10 evalsrv
  live smoke PASS; residual = AffineFoundation org mirror + n=80 burn-in.

## Resources

- Turn dataset D: four pooled duel corpora (albedo.tech/llms.txt; manifest
  s3.hippius.com/albedo/datasets/manifest.json): **mini-coder** (python,
  ricdomolm/mini-coder-trajs-400k), **mini-coder-rs** (rust,
  AlienKevin/SWE-smith-rs-gpt-5-mini-trajectories), **open-swe-traces** (4 arms,
  nvidia/Open-SWE-Traces), **swe-hero** (OpenHands,
  nvidia/SWE-Hero-openhands-trajectories). One rollout per instance, stratified by phase × bug
  family; prefix ends on a user turn. Native observation simulators: `<returncode>` /
  `OBSERVATION:` (SWE-agent) / OpenHands trailer.
- Benchmarks (albedo.tech/data/benchmarks.json): **swe_rebench_2026_03** (500 tasks;
  king scores in albedo.tech/data/model-scores.json), **tau2_airline** (50),
  **tau2_retail** / **tau2_telecom** (114 each). Scores for kings already measured — no
  re-running needed.
- Teacher: GLM-5.2 (prod reference, 73.8). Pilot teacher: GLM-4.5-Air-FP8 (already serving).
- Compute: Lium pods (affv-e1 terminated after freeze). Rent on demand as needed.
- APIs: Engy (GLM-5.2 top-20 logprobs on own tokens), OpenRouter ($20k/mo), Chutes (raw
  completions injection). Verified clean injection routes recorded in RESEARCH_LOG.md.

## End state

1. **Evidence**: E-KINGS isomorphism result + Qwen-ladder sanity + ablations (which terms matter)
   + causality-filter yield stats.
2. **Red team**: attack the score as a miner would (style mimic, steering/steganographic thoughts,
   empty-thought, memorizer, king copier) and measure exploitability vs the checklist judge.
3. **Subnet integration**: scoring service replacing judge_core (drop-in for the duel pipeline).
4. **Paper**: formalization, mechanism, experiments, red-team results.

## Standing instructions (from operator)

- Record all experiments in RESEARCH_LOG.md. Never lose the motivation (this file).
- Parallelize aggressively: subagents + as many Lium rentals as useful. Don't block on the
  operator; think around obstacles and keep moving toward the end state.
