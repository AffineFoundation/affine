# GAN-Style Distillation for SN120 — Open Questions

Updated: 2026-08-23 ~05:20 UTC. Live experiments: Track M (KOTH mock), Track S (self-play control).

## Context in one page

We are testing a new incentive mechanism for subnet 120: **king-of-the-hill adversarial distillation**.

- A fixed public **teacher** (Qwen3.8-27B).
- A **judge** (discriminator D): the same 27B base plus a LoRA adapter. It sees
  (prefix, rollout A, rollout B) — one rollout from a miner, one from the teacher,
  random order — and predicts which is the teacher. Thoughts are normalized before judging.
- **Miner score = fool rate**: how often the judge mistakes the miner for the teacher.
- **King of the hill**: a challenger dethrones the king only if it beats the king's
  fool rate by >= 3 points AND the paired per-turn difference is significant (z >= 2.5),
  tested at n = 400/800/1200/1600 paired turns.
- **Crown event**: evals pause; a new judge LoRA is trained **from scratch** on the full
  eval archive (so it specializes in the new king's tells); it is published openly
  (public D, no lag); evals resume. The judge is **frozen for the whole reign**.
- Key theoretical property: a king whose outputs truly match the teacher's distribution
  is immune to judge retraining (accuracy pinned at 0.5). The only stable throne is
  real distillation.

### What we have already established (evidence, not theory)

1. **An always-online judge crushes the generator.** Tracks A/B (overnight): judge accuracy
   pinned 0.95+, generator reward ~0, ~100 rounds with zero real SWE movement
   (all full-panel scores in the 13-15% noise band; baseline 13.33%, teacher 31.33%).
   Throttling the judge (train every 4th round) reopened rewards but only bought a
   modestly higher plateau — the judge re-adapts within a few updates.
   → This motivated the frozen-per-reign design.
2. **The teacher-with-LoRA judge works and lands in the useful band.** Track M judge v0:
   zero-shot 0.575 → 0.865 held-out after one from-scratch LoRA train (~90 min original,
   ~27 min with DDP + grad checkpointing).
3. **The crown gate separates signal from noise.** First duel (round-10 checkpoint):
   +1.1pp, z=2.25 → rejected, correctly. Second duel (round-20): +4.2pp, z=3.94 → crowned.
   Raw 3pp-over-400-turns would have had a ~10% false-crown rate per attempt;
   the added z >= 2.5 gate brings it under 1%.
4. **The miner learns against a frozen judge.** Local fool rate across submissions:
   0.21 → 0.29 → 0.38 (round 55). Two crowns so far.
5. **The scoring pipeline is leak-free.** Track S round 0 (generator = exact teacher copy,
   fresh teacher samples, identical serving): judge trained on truth-0.5 pairs stays at
   chance on held-out (probabilities 0.48-0.52). No template/sampling/caching artifact.
6. **Self-play does not (so far) beat itself.** Track S rounds 1-6: fool rate pinned at
   0.50 +/- 0.02, judge training accuracy at/below chance, no typicality-sharpening drift.
   Early evidence that perfect distillation is a **stable equilibrium**.
7. **Silent failure modes are real.** A trained LoRA served as a no-op due to a key-prefix
   mismatch (model.layers.* vs language_model.*) — caught only by an adapter-effect
   assertion, which is now mandatory at every publish. Bench runs corrupted each other
   via shared container/port — benches now serialized.
8. **Fool-rate SFT caused a format-compliance collapse (Goodhart, observed).** Track M's
   crowned kings score ~0 on SWE bench NOT from lost ability but because training burned
   in the arena's ```bash action fence; the kings now ignore the bench harness's
   instruction to use its own fence tag, so every action is rejected. The untrained base
   follows the instruction fine (temp-0 replay proof). FIXED: fence-randomized SFT
   targets + tolerant validity gate + judge-input canonicalization, live from miner
   round 78. king_r3 (reign 3) is the first king trained format-robust; its bench is
   the pending Q1 datapoint.
9. **Judge scoring must be forced-token, and online D updates must be from-scratch.**
   Two independent measurement bugs found in both tracks: (a) top-k readout misses the
   A/B answer on 15-35% of calls, silently scoring 0.5 and biasing accuracy toward/below
   chance — fixed by forced-token logprob scoring (miss rate now structurally 0);
   (b) warm-start LoRA updates at lr 1e-5/150 steps are too weak to learn even a
   trivially separable gap (train_acc 0.51 vs 0.687 length-probe) — from-scratch
   retrains on the same pairs reach 0.79-0.87. Also: D training loss must be full-vocab
   CE on the label token, not 2-way restricted CE, or answer-first behavior drifts
   (caught by the new >=99% answer-compliance publish gate on its first firing).
10. **The Track S "sharpening drift" was a winners-only selection artifact.** Fresh G
   outputs stayed ~teacher-like (think length 660 vs teacher 720 chars); only the
   best-of-6 WINNERS collapsed to short thoughts, because a mushy judge let them win.
   With a properly trained judge (vf13, 0.87 on drifted winners), winner think length
   snapped back 332 -> 666 in ONE round — the arms race self-corrects selection drift
   when the judge actually works. Round-0 leak-free verdict survives corrected scoring.
11. **Fool-rate SFT destroys task termination — real capability loss, not just format
   (Goodhart round 2).** With finding 8's format fix in place (tolerant harness that
   accepts the kings' fence), base 35B scores 5/16 on the proxy panel but king_r4
   scores 0/16 on a clean serialized run (adapter effect proven by logprob probe on
   the serving endpoint), and king_r5 reproduces it (all agent rollouts end with empty
   patches). The kings' rollouts are coherent, on-task, correctly fenced multi-step
   debugging — but they NEVER submit: 0/16 submission attempts vs 8/16 for base,
   mean 44 agent turns vs 33, every episode dying on the context window (9/16) or step
   limit (7/16) in an endless verify-more loop. Duel SFT on single-turn rollouts
   appears to erode the multi-turn episode-termination behavior the bench requires.
   Audit note: one earlier 0/16 (129s wall) was a serving artifact — the qwen3
   reasoning parser blanked the non-thinking kings' outputs (16/16
   RepeatedFormatError) — quarantined; the finding rests on gate-verified runs only.
   Full 150-instance panels (king_r4, king_r5, tolerant base control) in flight.
   **Mechanism-design lesson (fix2, 2026-08-23):** the arena's turn distribution
   must match the deployment turn distribution, or miners silently lose the
   behaviors the arena under-samples. Terminal (submission) turns were only 1.4%
   of the trainable duel corpus (102/7174 turns whose cached teacher rollout ends
   in the submit action — one per ~4 rounds at 16 turns/round), so 200+ rounds of
   fool-rate SFT diluted termination to zero: winners r199–r228 contained 0/480
   submission actions, and the kings' submission rate on terminal-context
   prefixes decayed base 87% → r2 77% → r3 24% → r5 7%. Corpus-level fix, same
   shape as finding 8: oversample terminal turns to ~20% of every miner training
   round AND every duel batch (crowns must measure the distribution miners train
   on); the teacher refs on those turns end in submissions, so D-scored duels
   reward submitting and SFT-on-winners re-teaches it — no reward shaping needed.
   This generalizes: any behavior class the duel corpus under-represents (error
   recovery, rare tool use, …) will erode the same way under prolonged fool-rate
   SFT, and the fix is always distribution matching, not a new reward term.
   Caveat from the live fix (fix2 verification, miner r260 after 28 oversampled
   rounds): once erosion is TOTAL, oversampling alone cannot re-seed the behavior
   (best-of-k winners can never contain what the model never samples — 0/72
   sampled submissions pre-fix), so terminal turns needed teacher-rollout SFT
   targets as fallback. That restored the marginal probability (0% → ~3% at
   temp 0.9; submission winners reappeared by round 253; three fix2-lineage
   kings crowned under the rebalanced duels, term_sub_k 1-2/5 vs ~0 for
   king_r5) but the bench proxy still read 0/16 submissions: the bench decodes
   greedily (temp 0), so submission must become the MODAL action, and 23
   corrective rounds against 200+ rounds of erosion is not there yet. Lesson
   sharpened: catch distribution mismatch BEFORE erosion completes, or pay a
   re-training bill comparable to the erosion length.

## Open questions

### Q1. Does climbing the fool-rate landscape move real SWE ability? (THE question)
The mechanism pays for "judged teacher-like". We need proof this transfers to task success.
Track A said no under a dominant online judge. The frozen-per-reign regime's first
CLEAN reading (post format-fix, finding 11) is NEGATIVE: on the format-tolerant
harness with verified adapters, kings r4/r5 score ~0/16 vs base 5/16 — and the
failure is lost episode termination (the kings never submit), a real agentic
capability regression, not a fence artifact. Pending: full 150-instance panels
(in flight) + McNemar pairing for the definitive number. Open: whether termination
behavior can be preserved the same way finding 8 fixed fences (e.g. include
episode-final/submission turns in SFT targets, or gate duels on end-to-end task
completion), and whether the current SFT recipe can climb fool rate at all without
cannibalizing multi-turn behavior.

### Q2. Does the ratchet rise across reigns, or reset?
Each crown trains a fresh judge specialized in the new king. If the king's fool rate
against each *fresh* judge (reign-start reading) climbs across reigns, miners are
genuinely absorbing the teacher. If it resets every reign, the loop only ever pays for
judge-specific tricks that the next retrain erases (style cycling). Two reigns exist;
we need ~5+ for a trend.

### Q3. Is the perfect-distillation equilibrium stable long-run? (Track S)
Twelve rounds + corrected scoring say yes so far: fresh-G distribution stayed
~teacher-like, held-out judge accuracy at chance in both directions, and the one
observed pathology (winners-only think-length collapse under a weak judge) was
self-corrected within one round once the judge was properly trained (finding 10).
Open: does this hold over 30+ rounds with the fixed pipeline (from-scratch D every
3rd round)? Watch: fool_win, winner think length, typicality of FRESH samples (not
winners), and the round-15 SWE proxy (drift benign vs harmful).

### Q4. Can a miner sustain fool rate > 0.5 ("more teacher than the teacher")?
Theoretical route: typicality sharpening (concentrating on the judge's idea of the
teacher's most characteristic behavior). Track S shows no sign of it yet. If it appears:
does the online judge learn to detect over-typicality, and does the resulting arms race
help or hurt SWE?

### Q5. What judge strength makes the landscape climbable?
Evidence so far: accuracy 0.95+ = flat landscape (nobody learns); 0.865 = works.
Open: where is the optimum, and should the from-scratch retrain *target* a band
(e.g. early-stop at 0.8) rather than train to convergence? Related: how should judge
training steps scale as the archive grows across reigns?

### Q6. Does from-scratch-per-crown forget old exploits?
The judge retrains from scratch on the full archive, so memory lives in the *data*,
not the weights. Untested: whether a trick from reign k re-emerges profitably at
reign k+3 if archive sampling under-weights old rounds. The replay buffer of
high-fool-rate rollouts is designed to prevent this; needs an adversarial test
(deliberately resurrect an old exploit and check it stays unpaid).

### Q7. Multi-miner dynamics (nothing tested yet)
The mock has one miner. Production has a population:
- False-crown rate under many simultaneous challengers (multiple-comparisons problem —
  does the z-gate need tightening when 50 miners attempt duels?).
- The copy-the-king meta: if challengers submit near-copies of the king, duels become
  self-play (Track S says: stalemate at 0.5, throne held — likely fine, but untested
  with margin economics).
- Challenger-tail data in judge training: how much weight, and does it blind-side
  the judge less at dethronement?

### Q8. White-box exploit cycling under public-D-no-lag
Miners get the judge weights and can attack them directly. Defense = continuous
retraining at crowns + validity gate + replay buffer. Open: the empirical half-life of
a deliberate adversarial exploit, and whether the signature (fool-rate spike that dies
at the next judge version, with flat teacher-agreement/SWE) is detectable automatically.
Fallback if cycling pays: score with a time-ensemble of the last k adapters (no lag,
no secrecy — just ensemble).

### Q9. Miner capacity cap
Chosen: Qwen3.6-35B-A3B (MoE, ~3B active). Open: can 35B total params "pack" the 27B
teacher rather than compress it (weakening the anti-memorization argument)? Detector:
teacher-verbatim n-gram rates in miner rollouts. Also open: the right cap policy for
production (dense-equivalent? active-param cap?).

### Q10. Benchmark trustworthiness and power
The proxy panel (16 instances) has ~+/-12pp noise — fine for direction, useless for
verdicts. Full panel (150) has ~+/-2.8pp. Open: right bench cadence per reign given
~75 min per full panel; whether the panel needs refreshing against contamination as
miners train on SE-corpus turns from the same repos.

### Q11. Judge self-recognition bias
The judge is initialized from the teacher. Position bias and matched-pair accuracy are
logged and look sane so far; open question is whether bias grows across reigns as the
archive becomes dominated by high-quality (near-teacher) kings — i.e. does the judge
drift toward calling everything "teacher"?

### Q12. Production economics (design, not experiment)
- Dethrone margin: is 3pp the right economic floor once real money is on it?
- Crown cadence: retrain time is the only eval downtime; 27 min is acceptable,
  6-9 min likely achievable — matters if crowns become frequent.
- Emission curve during a reign: winner-take-all vs decaying share for the king
  (affects whether challengers keep trying at 0.5-stalemate endgame).
- On-chain publication: adapter hash + activation block per crown; all validators
  must swap atomically.

## Current infrastructure

- **Track M** (KOTH mock): eval box eager-comet-81 (8xH200), miner box zesty-orbit-c3
  (8xB300). Status: reign 2, miner round ~55, duel for reign-3 in progress.
  Log: research/logs/trackM_status.log.
- **Track S** (self-play): cosmic-eagle-0c (8xB200). Status: round ~6, verdict-so-far
  STABLE/LEAK-FREE. Log: research/logs/trackS_status.log.
- **Dashboard**: https://sheer-teachers-win-marriage.trycloudflare.com (live data,
  2-min refresh; tunnel URL is ephemeral — rerun trackm-site/serve.sh if dead).
- **Harvested history** (Tracks A/B/C, torn down): research/data/harvest/.
- Baselines on the 150-instance SWE panel: base 35B-A3B **0.1333**, teacher 27B **0.3133**.
