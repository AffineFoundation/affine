# Proposal: min(R, G) — a two-leg scoring rule for SN120

**Status:** SHIPPED 2026-08-27 as `weight_version_key = 10` (explicit dated
operator directive; genesis reset to untouched `Qwen/Qwen3.6-35B-A3B`).
Live knobs: `score_mode = "min_rg"`, `band_c = 2.0`, `band_floor = 0.002`
in `affine/affine.toml`. This document is the design rationale as proposed.
**Date:** 2026-08-26 (proposal); shipped 2026-08-27
**Artifacts:** `research/results/centered_reason_replay.{json,txt}`,
`research/results/min_rg_replay.{json,txt}`, `research/scripts/min_rg_*.py`

---

## 1. Why we need a change

The current rule (Reason v4) scores one thing: how much a miner's thought
`z_A` helps the teacher predict the teacher's own next action.

```
R_i     = lpC(y_i | z_A) − lpC(y_i | ∅)          # per byte, i = 1..k refs
Reason  = tau · log( (1/k) Σ_i exp(R_i / tau) )   # tempered log-mean-exp
```

Terms, defined once:

- `lpC(a | b)` — log-probability per byte that the teacher model C
  (GLM-4.5-Air) assigns to text `a` when text `b` is in its context.
- `y_i` — the teacher's own sampled action for this turn (a reference).
  We sample `k = 3` references per turn.
- `z_A` — the miner's thought (its reasoning text).
- `∅` — the empty thought. `lpC(y_i | ∅)` is the baseline difficulty.
- `tau` — the tempering constant (0.03). The log-mean-exp is dominated by
  the best-matched reference, so committing to one valid teacher action
  beats hedging across all of them.

**The exploit.** The current king (reign41, `tojointhecommunity`) appends a
fixed filler suffix to every thought. The suffix raises `lpC(y_i | z_A)` by
a roughly constant amount on every reference — a flat "insurance" floor. It
adds no information about *this* turn, but under the current rule a flat
floor is worth as much as real prediction. The king crowned on it (margin
0.0022, z = 2.7) and benches 0/25 on swe-rebench.

The general lesson: any rule with a single scored quantity can be maximized
along its cheapest direction. The cheapest direction of "help the teacher
predict its action" turned out to be constant filler, not reasoning.

## 2. The proposed rule

Score each turn as the **weaker of two legs**. Both are per-byte
log-ratios (nats/byte), so they share one currency and need no weighting
coefficient.

```
Leg 1 — Reason, centered across the k references:
    R_i     = lpC(y_i | z_A) − lpC(y_i | ∅)
    R_turn  = tau·log( (1/k) Σ_i exp(R_i/tau) )  −  (1/k) Σ_i R_i
              └────── tempered best match ──────┘   └── mean (centering) ──┘

Leg 2 — Grounding, thought judged against the teacher's own thoughts:
    t_i     = lpC(z_C^i | x) / |z_C^i|      # teacher's own thought i, per byte
    mu, sd  = mean_i(t_i), stdev_i(t_i)     # per-turn typicality band
    w       = max(2·sd, 0.002)              # band half-width, floored
    m       = lpC(z_A | x) / |z_A|          # miner's thought, per byte
    G       = min( m − (mu − w), (mu + w) − m )
              # positive inside the band, negative (nats/byte) outside

Turn score:
    turn    = min( R_turn , G )

Crown (form unchanged):
    paired mean(turn_c − turn_k) > max(k_sigma·SE, δ)
    AND median stripped |z_A| ≥ min_thought_chars     # length floor license
    AND B pass rate ≥ causality_gamma                 # B license
```

New symbols:

- `x` — the turn prefix (the task context the miner sees).
- `z_C^i` — the teacher's own thought from reference rollout `i`. We
  already sample these for Leg 1; today we throw them away.
- `B = lpC(y_A|z_A) − lpC(y_A|∅)` — the existing causality license
  (the miner's thought must cause its own action). Unchanged, stays a
  gate, not a scored term.
- `δ` — the crown margin floor. Must be recalibrated (see §5).

### What each piece does

- **Centering (Leg 1).** Subtracting the mean over references removes any
  flat, reference-independent lift. The filler suffix centers to ~0. What
  survives is *spread*: how much the best-matched reference beats the
  average one — which requires actually predicting the right action.
  Algebraic note: because log-mean-exp is shift-equivariant, "center then
  temper" equals "temper minus mean"; that is the closed form above.
- **Grounding (Leg 2).** The teacher's own k thoughts define, per turn,
  what a normal thought looks like in likelihood terms. The miner's
  thought must sit inside that band. Below the band = weird/adversarial
  text (filler, gibberish). Above the band = suspiciously predictable text
  (copying the task, boilerplate). Both fail.
- **The `min`.** A weakest-link ratchet. A miner cannot coast on one leg;
  optimization pressure always lands on whichever leg lags. Because the
  `min` is nonlinear, shared baselines do NOT cancel between the two sides
  of a duel (they would under the current linear rule), so both nulls stay
  load-bearing.
- **Teacher-side only.** No miner-logprob (`lpA`) terms anywhere. The
  entire RT-3 attack family (lm_head manipulation) stays retired by
  construction.

## 3. Evidence so far

### 3a. Centered-Reason replay — 208 live v4 duels (free re-aggregation)

- **All 7 crowns awarded under the current rule dissolve**, including the
  exploit crown (chal-01093: old z = +2.71 WIN → centered z = −2.80).
  The other six were near-noise crowns (old z 2.1–4.5 → centered |z| < 2.1).
- The filler king **cannot defend**: challengers that lost to it live now
  beat it at centered z = 3–15.
- Centering surfaces ~9 would-be winners that the old rule suppressed.
  We inspected their thoughts (Tok331102, afgod1079, michael-chan-000):
  task-specific, committed reasoning followed by exactly the planned
  action. No pasted commands, no filler. The old rule was ranking the
  filler king's flat lift above these models' genuine commitment.

### 3b. Grounding band — adversary panel (1,261 turns, GPU echoes)

Band = teacher's 3 reference thoughts, mu ± max(2sd, 0.002 nats/byte).

| thought | median (m − mu) | % inside band |
|---|---|---|
| teacher's own thought (leave-one-out) | +0.001 | 66% |
| filler king (`tojointhecommunity`) | −0.24 | 9% |
| previous king (`unconstai`) | −0.37 | 3% |
| parrot (restatement of the task) | **+0.09 (above)** | 30% |
| generic boilerplate | −0.27 | 10% |
| style-swap (real teacher thought, wrong turn) | −0.36 | 7% |

Every adversary class separates in the predicted direction. The style-swap
row matters most: a *genuine* teacher thought transplanted from another
turn fails the band, so the band is content-specific — style cannot fake it.
The parrot lands *above* the band (copying is too predictable), which
closes the main hole of a simpler `lpC(z_A|x) − lpC(z_A|∅)` grounding term.

### 3c. Full min(R, G) duel replays (4 duels echoed)

| duel | stored verdict | min(R, G) verdict |
|---|---|---|
| chal-01093 — filler king's crowning | WIN (z 2.7) | **loses, z = −55** |
| chal-01113 — afgod1079 vs filler king | lose | **wins, z = +78** |
| chal-01074 — active-king vs prior king | lose | wins, z = +10 |
| chal-01115 — isomsom vs filler king | lose | lose |

The exploit crown reverses; the exploit king is immediately dethronable.
Note the honest-vs-honest flip in chal-01074: ordering among non-exploit
models does change — this is a different metric, and that needs the wider
panel check (§6).

## 4. Cost

- Leg 1 centering: free (re-aggregation of numbers we already compute).
- Leg 2: the ref thoughts `z_C^i` are already sampled. New echo work per
  turn: 3 ref-thought echoes (shared by both sides via the ref cache) +
  1 miner-thought echo per side. Roughly **+40% teacher echo work**; all
  new echoes share the `x` prefix, so prefix caching absorbs much of it.
  Mitigations if duel wall-time threatens the 1-hour budget: teacher
  swarm, or rebalance `n_turns`.

## 5. Open problems (honest list)

1. **δ / tau / band-width calibration.** min(R, G) margins in replay run
   0.03–0.29 nats/byte — ~100x the current δ = 0.002. Every current miner
   is G-bound (grounding far below reason), so day-one competition becomes
   "produce grounded thoughts". Intended, but δ must be re-derived from
   the new margin/SE distribution, with the ε-copy simulation re-run.
2. **Thin grounding coverage.** 4 duels echoed, one exploit case. The
   honest-vs-honest ordering behavior is measured on essentially one pair.
3. **Band noise at k=3.** The teacher's own thoughts only land inside
   their own leave-one-out band 66% of the time. Paired duels absorb this
   (both sides face the same band), but a wider floor deserves a sweep.
4. **Residual surfaces.** (a) Sitting mid-band with content-free text —
   guarded by the reason leg via the `min`, unverified under direct
   optimization. (b) "Next command:"-style cue tails appended to real
   reasoning survive both legs (they are small per-byte perturbations).
   (c) Betting on a single reference to manufacture centered spread —
   grounding is the intended guard, unverified.
5. **Teacher-clone equilibrium.** GLM-4.5-Air is public. A miner can run
   the teacher and copy its thought; grounding passes by construction, and
   reason then measures the quality of teacher imitation. Under the
   equilibrium framing (EQUILIBRIUM.md) this is the *intended* asymptote —
   the score is a distillation meter — but it should be stated, not
   discovered.
6. **No coding-isomorphism claim.** Same discipline as RT-7: this fixes
   the *meter*, it does not make crowns imply benchmark capability.

## 6. Validation program

Done:
- 208-duel centered-Reason replay (3a).
- 27,260 thought echoes → adversary panel + 4 full replays (3b, 3c).
- Thought inspection of the new centered winners (3a).

Done — round 2 (2026-08-27, full detail in
`research/results/minrg_round2/FINDINGS.txt`):
- **Model-free suffix attack + band sweep + parrot probe.** The live
  filler suffix barely dents v4 lme but collapses min(R,G) (honest wins
  ~97% of turns); the defense holds across band widths 1x–4x sd; parroting
  is caught above-band 95%. Not knife-edge calibrated.
- **Adversarial GRPO, white-box.** Aggressive (lr 2e-5, r32, 200 steps)
  against min(R,G): reward pinned at ~0 the whole run; final adapter
  indistinguishable from base on every metric. Same against the G leg
  alone: flat (thoughts padded longer; the band did not inflate).
  Interpretation caveat: an identical GRPO arm against the v4 reward also
  failed to climb, so these are "no cheap gradient in 200 steps" nulls —
  the live v4 exploit was found by SFT, not GRPO.
- **Positive control (discriminative power).** Paired leave-one-out on 100
  held-out turns: a genuine held-out teacher thought beats the base
  model's thought under min(R,G) at **z = +2.56** (63% wins), while v4 lme
  cannot tell them apart on the same pairs (**z = +0.11**). The new score
  detects teacher-likeness the live rule is blind to.
- **Capability benches (swe_rebench_lite).** Filler king: **0/25 twice**
  (0/50 pooled). Sampled live v4-board models: 0.00–0.08. Untouched base
  Qwen3.6-35B-A3B: 0.20. Model trained 300 GRPO steps *against min(R,G)*:
  0.24 and 0.28 (13/50 = 0.26) — capability preserved.
- **Crown-test SE calibration** (`duel_se_calibration.txt`): paired
  min(R,G) noise between ε-copies is *lower* than v4 lme noise;
  2·SE@1300 ≈ 0.002, so keep δ=0.002 only with ~2x n_turns, else raise δ
  to ≈ 0.003 to preserve today's safety ratio.

In flight:
- **Trainable positive control.** Properly-powered SFT distill of teacher
  thoughts (4,383 examples, 2 epochs, two lr arms) — must move the meter
  up where the underpowered 94-step pilot could not move the model at all.

Remaining before any fork:
- Honest-panel echoes (6–10 pre-exploit v4 duels) → ordering confirmation.
- δ / band-width / tau finalization on live artifacts + ε-copy simulation.
- Burn-in duel through the real evalsrv path; legacy replay parity check.

## 7. Rollout sketch (post-validation)

1. Implement behind the existing legacy-replay pattern (`score.py`,
   `terms.py` + one new thought-echo primitive, `affine.toml` knobs,
   verdict telemetry for both legs and the band).
2. Fork = `weight_version_key` bump with explicit dated operator
   directive. Never bumped autonomously.
3. Incumbent handling: recommend passive — the filler king loses its
   first defense under the new rule (replays at z ≈ +78 against it).
4. Post-fork watch: which leg binds per duel, crown churn at near-δ
   margins, duel wall-time; recalibrate δ once against measured SE
   (expect one touch-up, as with the 2026-08-22 δ revert).
5. Rollback = config + wvk bump back; the legacy scorer stays in the
   code path.

## 8. One-paragraph summary

Reason v4 ranks miners by a single teacher-side quantity, and its cheapest
maximization turned out to be a constant filler suffix — the current king
crowned on it and resolves 0/25. We propose scoring each turn as
`min(centered Reason, banded Grounding)`: centering subtracts the mean
over the k teacher references, which zeroes any flat reference-independent
lift and leaves only genuine action prediction; grounding requires the
miner's thought to sit inside the per-turn likelihood band defined by the
teacher's own thoughts, which rejects filler (below band) and parroting
(above band) while costing one extra echo per side. Both legs are
nats/byte, so no weighting constant. In replay over 208 live duels the
exploit crown reverses, all near-noise crowns dissolve, and the models
promoted instead show genuine committed reasoning on inspection. Open:
calibration at the new scale, thin grounding coverage, and an adversarial
GRPO run now in flight that attacks the proposed rule directly before any
contract change.
