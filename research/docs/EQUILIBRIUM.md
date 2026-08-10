# Equilibrium — why S-maximization graduates to a correct distill

_Written 2026-08-10 from an operator design discussion. This file records the
alignment argument for the asymptote of the S-duel game, the assumptions it
hangs on, and the operator policy decisions that frame it. It is the intuition
companion to REDTEAM.md (which tracks the intermediate states) and
MOTIVATION.md (which states the empirical claim)._

## 0. The frame (operator policy, 2026-08-10)

**We care about alignment of the asymptote, not intermediate states.** The
contract is judged on one question: *if the duel game runs long enough, do
miners eventually graduate to correct distills of the teacher?* Intermediate
reigns that are gate-valid but capability-free (the RT-7 0/25 kings) are
acceptable, provided they are transient. RT-7 is therefore evidence about
*where on the slope* the live board sits, not about *where the slope goes*.

Explicitly out of scope (operator decision, do not re-litigate here):

- **Teacher ceiling.** A miner better than GLM gets no extra credit and can be
  penalized where its (correct) answer diverges from `y_C`. Known, accepted,
  deferred — can be addressed later (teacher upgrades, ensemble teachers).
- **Intermediate Goodhart reigns.** Priced in; see §2.

## 1. Leaking is knowing

Λ2 = lpC(y_C | z_A) − lpC(y_C | ∅). The miner's thought z_A is graded by how
much it lifts the frozen teacher toward the teacher's own fresh answer.

In the fresh-D regime — `y_C` sampled fresh per duel, slice seeded by reveal
block hash, strata order shuffled, corpus refreshed by epoch (the RT-6 code
fixes) — a thought containing "the answer is obviously X" scores maximally
**only if the miner computed X**. Computing GLM's output on arbitrary unseen
prefixes *is* having distilled GLM. The "degenerate" argmax of Λ2 and the
desired equilibrium are the same point:

- The optimal thought is a **sufficient statistic** for `y_C` given x, in a
  form GLM's decoder can read.
- Word-for-word reproduction of GLM's own thinking is not an exploit; it is
  the perfect distill (REDTEAM A5, "accepted outcome").
- Λ2 is teacher-forced distillation log-loss with the miner as encoder and
  GLM as decoder. If miners degenerate to memorizing a finite D, the answer
  is **more data** (corpus_epoch refresh), not a new gate. Pretraining-style
  regime assumed: at sufficient scale and freshness, memorizing the teacher's
  outputs is indistinguishable from learning the teacher's function.

## 2. Hedging is not knowing — what the gates actually buy

There is one shortcut family that leaking-is-fine does **not** cover:
enumeration. Λ2 measures reduction of GLM's uncertainty about `y_C`, and a
miner can reduce it without knowing the answer by dumping top-k candidates
into z — recall instead of precision, log(k) bits cheaper than commitment.

That is RT-2 (action stuffing) and RT-2c (paraphrase stuffing). So in the
equilibrium picture the leakage and bank gates are not anti-cheat bolt-ons:
they **force commitment**, converting the objective from "reduce GLM's
entropy by any means" into "reduce it with one committed line of reasoning" —
which is the distill objective. "The answer is obviously X" is fine; "the
answer is one of X₁…X₁₀" is the thing the gates correctly tax.

## 3. The graduation argument (δ as a ratchet)

Each crowning requires paired mean(S_c − S_k) > 3·SE **and** > δ = 0.02, so
noise cannot crown and every reign raises the champion's true S by ≥ δ.
S is bounded above (per turn, by the teacher's baseline uncertainty,
−lpC(y_C|∅)). A strictly increasing, bounded sequence with a minimum step
size terminates:

> **There can only be finitely many reigns before the only S left to mine is
> S that requires actually predicting `y_C`.**

Every capability-free exploit channel is therefore a **finite prepaid
budget**, not a refutation. It buys some number of cheap crownings (each
costing ≥ δ of headroom) and is spent. The red-team table is a ledger of
channel budgets against δ:

| channel | measured budget | vs δ=0.02 |
|---|---|---|
| lm_head sharpening (RT-3 residual) | ≤ +0.012 | < 1 crowning |
| baseline-band minting (RT-3d, at band edge) | ≤ +0.015 | < 1 crowning |
| style similarity to teacher (RT-7 mechanism) | crowned 3 kings so far | finite but not yet sized |

Technical caveat, stated so the argument is not overclaimed: the potential
function here is **Λ2**. The L1lift term is miner-referenced, so S is not a
single global potential; but L1lift is clipped ±0.1, causality-gated, and
band-gated, so any cycling it permits is confined to a bounded slab and
cannot sustain an infinite reign sequence on its own.

## 4. The style channel's own asymptote is the distill

The RT-7 reading of Λ2 as "similarity-to-incumbent, not capability" describes
the shallow part of the slope, not a different slope. Matching GLM's
phrasing, format priors, and reasoning register **on fresh, unpredictable
prefixes** is matching GLM's conditional distribution — which is the
definition of the distillation. Style vs capability is a distinction about
intermediate states only: low-order statistics (register, format) are matched
long before high-order ones (which action to take), and the low-order gains
don't transfer to swe-bench. But it is all one slope toward one fixed point.
RT-7 measured that the live board is on the shallow prefix of it.

Teacher swaps partially refill the style budget — expect a fresh round of
cheap style reigns after each swap. (A 2026-08-10 Air → GLM-5.2 cutover was
attempted then torn down before live scoring changed; teacher remains Air.)

## 5. The single load-bearing condition

The whole claim "run long enough ⇒ correct distills" reduces to exactly one
condition:

> **No capability-free channel is unbounded.**

This is a sharper red-team target than "no exploits exist." A finite-margin
exploit is annoying but digested by the ratchet. The only fatal object is a
channel extracting > δ per reign *forever* without ever matching GLM's
conditional. When triaging a new attack, the first question is not "can it
win a duel" but "**is its budget bounded, and by how much**."

Watch-class: **decoder manipulation** — thoughts that make GLM generically
more confident rather than better-informed (sharpening family). It is the one
family that lifts lpC without carrying information about `y_C`. Bounded by
construction: `y_C` is *sampled*, not greedy, so blanket sharpening helps on
modal turns and hurts on off-modal ones (hence the small measured residual).
Any change that makes `y_C` more deterministic (temperature, greedy refs)
would widen this channel — do not make one without re-sizing it.

## 6. Free consequence: auto-curriculum

Per-turn Λ2 is capped at −lpC(y_C|∅), GLM's own uncertainty on that turn.
Turns where GLM is confident are worth ~nothing; score mass concentrates on
turns where GLM genuinely had to think. D auto-weights toward hard turns
without any explicit difficulty labeling.

## 7. Assumption ledger

The graduation argument holds iff **all** of these stay true:

1. **Fresh `y_C` per duel** — RefCache scoped to one duel (RT-6 fix #1).
2. **Unpredictable slices** — reveal-block-hash seeding incl. strata order
   (RT-6 fix #2). Miner cannot precompute D_t.
3. **D refresh outpaces memorization** — corpus_epoch refreshes keep the
   target the *function*, not the corpus. "If we degenerate, we generate more
   data" is a standing operational commitment, not a one-off.
4. **δ enforced and > every residual channel's per-reign mint** — δ=0.02 vs
   ledger in §3. Lowering δ or raising a channel bound re-opens the game.
5. **Commitment gates live** — leakage + bank gates (else hedging, §2).
6. **`y_C` stays sampled** — bounds the decoder-manipulation class (§5).
7. **Contract stationarity per ratchet** — each teacher/scoring change resets
   budgets; graduation is per-era. Fine, but expect cheap reigns after swaps.

Falsifier for the whole file: a demonstrated **unbounded** capability-free
channel (> δ per reign, sustained across reigns, without conditional
matching). That would break the asymptote claim, not just an intermediate
state, and would demand a scoring change, not a data refresh.

---

_Cross-refs: REDTEAM.md A12/RT-7 (the slope measurement), A5 (distill-shortcut
accepted), A2/A2c (commitment gates); MOTIVATION.md (empirical isomorphism
claim); AGENTS.md §3b (claim caveat)._
