# Affine / SN120 — project context

Affine is a Bittensor subnet (netuid **120**) that crowns miners by a single
**teacher-anchored distillation score — Reason (Λ2)**, not an LLM judge. This monorepo is the public command center:
validator + evalsrv (`affine/`), research harness and freeze artifacts (`research/`), and
lightweight ops helpers (`ops/`).

Thesis: `research/docs/MOTIVATION.md` · Red-team: `research/docs/REDTEAM.md` ·
Equilibrium: `research/docs/EQUILIBRIUM.md` · Paper draft: `research/docs/PAPER_DRAFT.md` ·
Contract SSOT: `affine/affine.toml`

---

## 1. Goal and claim

**Why:** Albedo (SN97) uses a GLM judge checklist that has been Goodharted. Every crowned
king sits far below genesis on swe-rebench (genesis 58.2 → typical kings 26–38 → worst ~12).
We want a score that stays **benchmark-isomorphic under adversarial pressure** on the
capability axis that the turn set D exercises.

**Claim (precise):** higher S ⇒ higher swe-rebench for models below the teacher, even though
S never touches benchmark tasks. ⚠️ **This holds only on the Albedo panel. On the live
SN120 board it inverts (ρ=−0.42, p=0.024; all three S-crowned kings resolve 0/25) — see
§3b RT-7 before repeating the claim.** D is SWE-style coding trajectories ⇒ target axis is coding.
“Programmable capability meter” (pick D → get matching benchmark) is an *interpretation*;
D_tau2 tests did **not** demonstrate it yet.

**Lineage:** research started against Albedo kings
(`dendriteholdings/albedo-qwen3.6-35b-king-*`), then productized into the Affine validator +
evalsrv package under `affine/`.

---

## 2. Frozen production scoring — min(R,G) v5: centered Reason + banded Grounding + δ + length floor + B gate (2026-08-27, `weight_version_key = 10`, genesis reset)

Implemented in `affine/affine/score.py` (research twin `research/harness/score.py`,
which also keeps v4 under `turn_reason` / `score_mode="reason"` and v2 under
`legacy_*` for pre-fork replay). Contract knobs in `affine/affine.toml`
`[duel]` + `[dataset]`. Design doc: `research/docs/MIN_RG_PROPOSAL.md` (shipped).

### The whole contract
```
a_i (per teacher ref) = lpC(y_i | z_A) − lpC(y_i | ∅)          # i = 1..k, k = 3
R (per turn) = tau·log((1/k)·Σ_i exp(a_i/tau)) − (1/k)·Σ_i a_i  # centered tempered LME, tau = 0.03
m   = lpC(z_A | x)          # miner-thought grounding echo (per-byte)
t_i = lpC(z_C^i | x)        # same echo for each teacher reference thought
mu = mean(t_i), w = max(band_c · sd(t_i), band_floor)           # band_c = 2, band_floor = 0.002
G (per turn) = min(m − (mu − w), (mu + w) − m)                  # positive iff m inside the band
Turn score  = min(R, G)              # if the side produced a parseable action
            = forfeit_turn_score      # = −0.1 otherwise (wvk 12, 2026-09-05); both forfeit → tie at 0
Miner score = mean(turn) over all turns, forfeits included
Crown = paired mean(turn_c − turn_k) > max(k_sigma · SE, min_margin)
        AND median(len(z_A.strip())) ≥ min_thought_chars
        AND B pass rate ≥ causality_gamma
```
Scoring hyperparameters: `n_turns = 1300`, `k_sigma = 2.0`,
`min_margin = 0.002` (churn risk at v4/v5 noise scale explicitly accepted
2026-08-27), `min_thought_chars = 80`, `causality_gate = true`,
`n_teacher_samples = k = 3`, `tau = 0.03`, `n_miner_samples = 1`,
`score_mode = "min_rg"`, `band_c = 2.0`, `band_floor = 0.002`,
`forfeit_turn_score = -0.1` (wvk 12, 2026-09-05).
No mix, no clip, no lpA gates, no `min_se` floor.

**Why v5 (2026-08-27):** the reign-41 king gamed v4 with a fixed filler
suffix appended to every thought — a reference-independent lift that raises
all k `a_i` by the same constant, which the tempered LME happily keeps.
Two orthogonal fixes, combined with `min`:
- **Centering** subtracts the plain mean of the `a_i` before the LME, so any
  flat (task/ref-independent) lift cancels *exactly*; only the spread across
  refs — actually predicting which action the teacher takes — scores.
- **Grounding** echoes the miner's thought itself under the teacher,
  `m = lpC(z_A|x)`, and requires it to land inside the likelihood band the
  teacher's own k reference thoughts define. Filler sits far below the band
  (G ≪ 0); pasting/parroting the teacher's literal thought sits above it
  (also negative) — the "parroting hole" is closed by the two-sided band.
- **min(R, G)** makes the lagging leg binding: a miner cannot buy score on
  one leg while zeroing the other.
Positive control (n=100 held-out turns, paired): a genuine held-out teacher
thought beats base-Qwen output at z = +2.56 under min(R,G) vs z = +0.11
under v4 — the new meter detects real thought quality; the old one did not.
Adversarial GRPO/SFT arms attacking min(R,G) directly did not find a
capability-free channel (see `research/docs/MIN_RG_PROPOSAL.md`). Cost:
~+40% teacher echo work (k=3 `t_i` echoes shared both sides via RefCache
+ 1 `m` echo per side per turn). wvk-9 artifacts replay bit-identically
through `score_mode="reason"` (parity-verified on stored duels). The fork
came with a **throne reset**: reign 0 re-seeded from untouched
`Qwen/Qwen3.6-35B-A3B` (unpaid genesis), old-era reveals invalidated via
`min_submission_block`.

### v6: forfeit floor LIVE (wvk 12, 2026-09-05); action leg A staged, NOT live
Operator directive 2026-09-04 ("implement immediately") after the reign-5
bench post-mortem; **forfeit floor flipped 2026-09-05 on explicit operator
directive ("yes add this forfeit"), wvk 11→12** via
`ops/v6/v6_toml_edits.py --apply 2026-09-05 --wvk-to 12 --forfeit-only`,
then eval-pod redeploy + validator restart + llms.txt "Fork history: wvk
12" section. Code is in `affine/affine/score.py` (+ research twin),
`evalsrv/terms.py` / `dueling.py`, `config.py`. The A leg stays behind
`score_mode` (still `min_rg`); pre-wvk-12 verdicts (no
`duel_params.forfeit_turn_score`) replay bit-identically through the
legacy drop-from-pairing path (210/210 checked).
- **A leg** (`score_mode = "min_rga"`): `b_i = lpC(y_A|z_C^i) − lpC(y_A|∅)`,
  `A = tau·log(mean_i exp(b_i/tau))`, `turn = min(R, G, A)`. The dual of R
  — "would the teacher, thinking its own thought, take the miner's action?"
  Teacher-side only. **Not centered**: centering rewards spread, and an
  action all refs license is the target; a generic action earns less lift
  than the right one, so the R-leg flat-lift attack has no analog. Why:
  actions were only licensed (B), never ranked — bench transcripts of
  reigns 1–5 show commands shrinking 426→150–250 chars while thoughts grow
  and steps multiply. Cost +k echoes/turn/side (`lpC_ya_zc`). Telemetry
  `mean_a_leg` / `a_bind_frac`.
  **Probe result (2026-09-04, `research/results/v6_action_leg_probe.*`,
  1,222 turns of chal-00248 re-echoed on the teacher swarm) — DO NOT FLIP
  A AS BUILT:** ordering passes (generic `ls -la` A = −0.18, negative on
  86% of turns; teacher-own action 0.009 > king 0.0063 ≈ challenger
  0.0066), and A would bind ~36% of turns (R 40%, G 22%), but (i) it is
  noisy at the teacher's own level — even the teacher's own action gets
  A < 0 on 30% of turns (multi-modal refs); (ii) it did not separate the
  two live miners and raised the paired SE 0.00057→0.00086, turning this
  crown's z from +3.5 to +0.5; (iii) **per-byte A rewards short actions
  ~10×** (king short 0.021 vs long 0.002; challenger 0.019 vs 0.0025) —
 the fence bytes dominate a short span — which is the opposite of the
 directive's intent.
 **Redesign staged 2026-09-10 (`research/results/v6_action_leg_norm.txt`,
 offline re-normalization of the same 1,222-turn echoes):** the summed
 lift of the teacher's own actions is `S = 0.61 + 0.00007·len` nats
 (Spearman(S, len) = +0.05) — a thought lifts an action by a near-constant
 ~0.6 nats concentrated on its decision tokens, so dividing by the real
 length *was* the length bias, and a body-only/fence-subtracted proxy
 makes it worse (ratio 11× → 30×). Fix: `b_i = S_i / action_norm_bytes`
 (fixed byte count, new `[duel].action_norm_bytes = 128`, inert under
 `min_rg`; pairs now carry `n_bytes_ya`). At 128: short/long ratio 0.77,
 `ls -la` negative on 89 %, A binds 37 % (R 40 / G 23), paired z on the
 probed duel +0.48 → +2.32 (min(R,G) alone +3.53; L0=192 gives +2.93 but
 A binds more as it shrinks toward 0). Teacher-own `A<0` stays ~30 % under
 every normalization (refs disagree by 1.5 nats median; symmetric noise).
 `v6_action_leg_probe.py` now reports both forms.
 **Isomorphism check before any flip (2026-09-10, same day):**
 (a) `research/results/v6_bench_renorm.txt` — the kings' real bench steps
 (1,348 + 1,600 steps, reigns 0–5) re-scored with fixed-byte A: length
 bias gone on-bench (5.4× → 0.90); A alone sees repeat steps (z +3.0)
 but `min(R,G,A)` does not (z −0.3) because A's large negatives (~30 %
 of steps, refs disagree) dominate the min; a downside floor
 `max(A, −0.01)` recovers it (bench repeat z +1.9; duel-probe z +2.32 →
 +3.94 with SE 0.00053 < min(R,G)'s 0.00057). Run-outcome AUC is
 untestable there (7 mixed tasks, sign flips between step samples).
 A_fixed rises monotonically over reigns 0→5 while their bench was
 flat/down — RT-7 shape. (b) `research/results/v6_action_leg_panel.txt`
 (`v6_action_leg_panel.py`, 11 crown-chain duels × 400 turns re-echoed
 on the swarm, both sides benched on `swe_rebench_lite_300`): the rules
 make the same crown decisions (10/11 identical; `chal-00169` differs on
 the subsample only), Spearman(margin, Δbench) ≈ 0 for every rule
 (min(R,G) +0.21, A +0.01, A floor 0.01 +0.14; n=11 CI ≈ ±0.6), and the
 per-duel A difference agrees with the bench sign on 3/7 nonzero moves
 (the three large moves ≥ 0.08 all agree, p = 0.125). Caveat on the
 bench itself: `swe_rebench_lite_300` is the pinned **25-task** panel at a
 300-step budget (not 300 tasks) → SE ≈ 0.10 per model, Δbench SE ≈ 0.14;
 only moves ≥ 0.2 are clear. **The panel is winners-only, so it cannot
 discriminate rules.** The decisive test needs benched LOSERS.
 **Step 1 done 2026-09-10 (`research/scripts/v6_loser_panel.py` →
 `research/results/v6_loser_panel.{txt,json}`):** 12 rejected challengers,
 12 distinct hotkeys, all Sep 9–10 (schema-3 D, kings 4f7dea97 0.56 /
 93b1f299 0.60 / 0ce59769 0.68), all weights verified fetchable
 (private R2), stratified by min(R,G) margin: near_miss 3 (+0.0011…
 +0.0018, z 1.4–2.4), tie 3 (−0.0010…−0.0023), mid 3 (−0.0034…−0.0038),
 far 3 (−0.0129…−0.0257; B pass ≥ 0.45, forfeit < 0.15). Launched the same
 day: the 12 benches sequentially on the always-on bench pod via
 `affine/scripts/bench_run.py --label loser-<id>` (~0.3–1 h each; rows land
 in `bench_history.jsonl` / `benches/index.jsonl` with hotkey ""), and the
 A echoes for their 12 duels (`v6_action_leg_panel.py --records … --out
 research/results/v6_action_leg_losers`). When both finish, re-run the
 panel script with `--resume` to fold the new bench scores into the
 report, then compute Spearman(margin, Δbench) over winners + losers
 (n = 23). Until then A is not shown isomorphic with performance; the
 floor is the recommended form if it ships (needs a staged `action_floor`
 knob, not yet added).
  `ops/v6/v6_toml_edits.py --forfeit-only` flips the floor alone.
- **Forfeit floor** (`forfeit_turn_score = -0.1`): a turn with no parseable
  action scores the floor instead of being dropped from pairing
  (one-sided = loss, two-sided = tie at 0, kept in n). Old rule made
  runaway thinking free and let a miner pick which turns entered its mean.
  Calibrated on 40 live duels / 100k turns: p1 of valid turn scores −0.087,
  p5 −0.041, live forfeit rates 1.7% median / 5.7% max per side → −0.1 is
  under p1 and a 2% forfeit rate costs one δ. Counterfactual on the 210
  stored verdicts: median margin shift +0.00003, 5 flips (4 king-forfeit
  losses → wins, 1 crown `chal-00169` → below δ); forward-only, so none
  re-verdict. Telemetry `n_forfeits` / `forfeit_rate` / `n_forfeit_turns`.
- **Bench post-mortem facts that drove this** (`affine/state/benches/`):
  57% of 150 king agent runs never submitted; kings 1–3 hit the 8,192-token
  reply cap with no command 53–63×/run (genesis 17); one command repeated
  ×32–36 (genesis ×5); thought chars/reply 660→1,200–1,370, command chars
  426→150–250, steps 26→44. On D the same kings are surface-identical to
  the teacher (thought 332–347 vs 352 chars, same 53% two-paragraph
  restatement shape). D is 100% the bench's mini-swe scaffold already, so
  the gap is not prompt format: the bench conditions on the model's OWN
  trajectory and decodes greedily (T=0; duel samples at 0.8). Multi-step
  control failures stay a blind spot of the single-turn duel by design.
- **Data pipeline (item 2 of the directive):** three harnesses already run
  teacher-only (mini-swe textbased, verifiers `bash` tool, `pi`), staged
  to T0. A fourth (`hermes_agent`, tool_call) exists only as a commented
  policy and needs a datagen-pod probe before enabling — deferred past T0.

### v7: "models must work as chat models" — LIVE 2026-09-09 (wvk 12→13, notice 2026-09-07/08)
Trigger: SWE-bench Pro under the Claude Code harness (full 731 tasks, king
reign-8 vs teacher, 2026-09-07/08). The king answers Claude Code's
compaction ("summarize, TEXT ONLY") and final-report prompts with reasoning
only and no visible text → compaction fails silently → "Prompt is too
long"; it also never closes `</think>` under IDE-shaped prompts. Root cause
is the contract: min(R,G) never scored the visible message and treated
`</think>` as optional. Operator order 2026-09-08 ("lets do this"): probe
to shadow now, the rest bundled into one fork with one notice. **Flipped
2026-09-09 ~02:00 UTC on explicit operator directive ("make all the changes
and flip the bit lets go to main", 2026-09-08 22:30 UTC-3), after the
noticed queue (through `chal-00359`) had been judged under wvk 12
(`chal-00366` = last pre-fork duel; `chal-00349` crowned reign 9 under wvk
12 on 2026-09-08). Ran as `ops/v7/v7_toml_edits.py --apply 2026-09-09`
(+ `[protocol_probe].mode = "enforce"` by hand), commit `fc09cc5`, then
`/tmp/v7flip/deploy.sh` (wait for pod idle → pm2 stop validator →
`redeploy_pods.py` → pm2 start) and one `ops/corpus_build.py --rederive`
(new flag: re-derive every trace chunk so the `text` turns of trajectories
folded under wvk 11/12 back-fill; already-published turn ids are skipped,
the deferred carryover is regenerated) → epoch 20: +6,045 `text` turns (of
+14,075 new; D = 283,828 turns, text ≈ 2.1%), zero not-admitted drops, 12 min
wall. Eval pod redeployed 02:05–02:10 UTC at the duel boundary (pod toml +
code verified); `chal-00367` = first wvk-13 duel. Discord: epoch-20 post by
the fold + fork notice `…/1547069605703454831`. Pitfall hit: `pm2 stop` 10 s
after the pod went idle left `in_flight = chal-00366` in state.json although
its verdict was already in history — `State.load` would have requeued it
(no verdict check); cleared by hand before the restart. **First wvk-13
verdicts:** `chal-00367` probe-rejected (0.75 < 0.90, 12 min); `chal-00368`
/ `00369` passed the probe at 0.95 but closed `</think>` on only 80 / 77 %
of duel rollouts (long SWE prefixes at the 1792-token cap) → 20 / 23 %
forfeits, z −8.6 / −9.4; king think-close 0.96, forfeits 3 %. Slices carry
21–25 `text` turns (1.6–1.9 %); teacher text refs 3.0/3, zero empty. Stamps
verified: `duel_params.require_think_close`, `allowed_action_kinds`,
`ranking_formula` suffix, `by_dialect.text` on both sides + teacher.
Forward-only: reign 9 stands, no re-verdicts, `min_submission_block`
unchanged.**
- **Protocol probe — `shadow` 2026-09-08 13:58 UTC → `enforce` 2026-09-09** (toml
  swapped on the eval pod at a duel boundary; `[protocol_probe].mode`).
  Ten Cursor/IDE-shaped prompts through the challenger's own template; pass
  = closes `</think>` + visible answer. Result published on every verdict
  (`verdict.protocol_probe`). Shadow read, 18 verdicts `chal-00348`…`365`:
  pass_rate 0.00–1.00, median 0.60, **1/18 at ≥ 0.90**; every failure was
  `no_think_close` (the reign-9 king's own lineage sat at 0.70). Enforce
  (`min_pass_rate = 0.90`) therefore rejects most of the current queue on
  sight — that is the intent (a model that does not close `</think>` also
  forfeits under `require_think_close`), and `min_pass_rate` is the dial if
  the operator wants a softer bar. Admission rule, no wvk.
- **`text` dialect — admitted at the flip (`allowed_action_kinds` += text).**
  `affine/dialects.py` `text`: action = the whole visible reply after
  `</think>` (stripped), no system marker (`Dialect.system_marker = ""`
  → `system_ok` vacuous), `ends_in_text` marks the dialects whose rollouts
  legitimately end on prose (bash, tool_call; not boxed). Fold rule
  (`affine/corpus/view.py::deliberate_final_reply` + `datagen/slicer.py`
  `text_final`): only the FINAL sampled reply of a rollout with
  `stop_condition == agent_completed` whose last call did not finish
  `length`, and only when it has no action in the policy dialect. Existing
  records unchanged by construction (fallback fires only where the old rule
  yielded nothing). Measured on real traces (3,485 envelopes): +89 text
  turns next to 3,902 bash-tool `tool_call` turns, +105 / 3,896 pi, +46 /
  187 wiki — ~1 text turn per 35–45 tool turns; 100% of claude_code, 99%
  pi, 98% bash-tool teacher rollouts end on such a reply (p50 1.6–2.2k
  chars). Before the flip the fold dropped them as
  `action_kind_not_admitted:text`; the `--rederive` fold at the flip
  back-filled them from the already-published traces. No per-kind token
  cap needed: a truncated text reply still parses (shorter report), it
  does not forfeit.
- **`require_think_close = true`** — knob since 2026-09-07 (a rollout that
  never emits `</think>` scores `forfeit_turn_score`), live with the flip.
- **Visible-span G — decided NO** (`research/results/visible_span_g_decision.txt`):
  the teacher's own tool-loop replies carry no visible prose on 20–33%
  (pi 33 / bash 24 / claude_code 20 %; GLM-era 61–63 %), so a band on the
  visible span has no support there and would punish the teacher's own
  behaviour; the spans are 15–35 tokens (per-byte noise, the A-leg
  lesson). The visible channel that matters — the reply that ends a
  trajectory — is the `text` dialect. Uncovered: mid-trajectory compaction
  summaries (not final replies; rare in datagen) — extend the fallback if
  they become common.
- **`teacher_claude_code` datagen policy** — live on datagen-1/2 since
  2026-09-07 (~130 kept tool_call turns per 24-rollout batch); tomls
  synced to datagen-3 and its supervisor bounced 2026-09-08.
- **Flip tooling (used):** `ops/v7/v7_toml_edits.py --apply 2026-09-09`
  (require_think_close true, allowed_action_kinds += text, wvk 12→13 +
  history paragraph). Notice as published: effective once the queue as of
  2026-09-07 (through `chal-00359`) drained, projected 2026-09-09 — met.
  llms.txt "Upcoming changes" → "Fork history: wvk 13".

### v9: window-best crown — LIVE 2026-09-12 17:01 UTC (wvk 14→15)
Operator directive 2026-09-12 16:39 UTC (Jacob Steeves: "best positive
margin of the last 12 hours"; confirmed 16:43 UTC "Yes I want to make this
update … implement the full design and push the updated code"). It
superseded, the same afternoon, the decaying-δ proposal approved at 15:27
(`MarginSchedule` stays in code, `min_margin_mode = "fixed"`, never
flipped). Contract (`[duel]`): `crown_mode = "window_best"`,
`crown_window_blocks = 3600` (12 h at 12 s/block; window id =
`decision_block // 3600`, aligned on the block number),
`crown_confirm_slice = true`, `crown_confirm_max = 2`,
`crown_one_entry_per_hotkey = true`. Rule: the king is FROZEN for a
window; every challenger dispatched inside window N duels window N's king
(a duel that crosses the boundary is still N's candidate — the close waits
for it); at the close the verdicts with a finite margin > 0 and no gate
rejection are ranked by margin (ties: higher z, earlier id), one per
hotkey; the best gets ONE fresh 1,300-turn slice vs the frozen king (the
near-miss draw: seed `block_hash ‖ hotkey ‖ "|slice<k>"`, k = the number
of slices its duel scored, turns disjoint) and is crowned iff the exact
pooled margin over both samples is > 0 (`score.pooled_margin_stats`);
else the next-best, up to 2; nobody confirms → king stays. The
confirmation runs at the start of N+1 before N+1's first duel and is not
an N+1 verdict. `max(k_sigma·SE, δ)` + gates are still computed and
stamped (`duel_rule_wins`) but do not decide; `challenger_wins` is false
on every duel row. Reign chain / payouts unchanged. Code: PR #16 branch
(`score.py`: `window_id_of`, `window_candidate_reason`,
`rank_window_candidates`, `pooled_margin_stats`; `validator.py`:
`_stamp_window_verdict`, `_window_due`, `_close_window[_safely]`,
`_finalize_window`, `_confirm_candidate`, `WINDOW_CLOSE_MAX_ATTEMPTS = 6`
→ after 6 failed infra attempts the window closes with
`king_stays_confirmation_unavailable`; `state.py`: `crown_window` in
state.json, `record_window_verdict`, `record_window_close`;
`evalsrv/dueling.py` + `server.py` + `eval_client.py`: `confirm`
request → `confirmation_stamp`, probes skipped, near-miss off on a
confirmation; artifact `evals/<cid>-confirm.json.gz`). Stamps: verdict
`crown_mode` / `window_id` / `window_blocks` / `decision_block` /
`duel_rule_wins` / `crown_decision`; history `window_close` rows
(`verdicts_considered`, `candidates`, `dropped`, `confirmations`,
`winner`, `outcome`, `crown_block`); `crowned` row `via = "window_best"`.
Replay (`affine/scripts/replay_window_best.py`, 364 scored duels /
16 days / 11 real crowns; wall time → blocks from anchor 9052470 @
15:45 UTC, dispatch = at − duration): 12 h windows → 26/32 windows with a
positive candidate, ~22–23 crowns with confirmation (1.4 kings/day), 8
winners at z < 2, 10/11 real crowns are window winners (`chal-00407` is
not: same window as `chal-00409`); 24 h → 14/17, 0.8 kings/day, 4 at z <
2. Confirmation modelled (no second-slice records exist yet): z ≥ 1
pass, z < 1 coin. Flip: `ops/v9/wvk15_toml_edits.py --apply 2026-09-12
--wvk-to 15 --mode window_best` (+ website mirror), `build_llms_txt.py`
("Fork history: wvk 15 — window-best crown", `_margin_subs` renders the
crown rule from the toml), box commit `2ebb3dc`, `/tmp/v9flip/deploy.sh`
(keepalive ralph off → pod idle + no in_flight → pm2 stop → env from
`/proc/<pm2 pid>/environ` → `redeploy_pods.py` → pod toml/code verified →
pm2 start + `affine-dash` restart → keepalive on). Queue was empty; no
duel interrupted. First window opened: 2514 at block 9052869 (17:03:05 UTC; window 2514 = blocks 9050400–9053999, closes ~20:50 UTC). First wvk-15
verdict: none yet at flip time (queue empty since 15:30 UTC) — `bash /tmp/v9flip/verify_first_verdict.sh` prints the stamps of every verdict / window_close since the flip. Discord notice `…/1548378529534840892` (17:03:29 UTC). Forward-only:
reign 11 stands, `min_submission_block` unchanged; pre-flip rows carry no
`crown_mode` and replay bit-identically. Known gaps: tensor-level copy
detection still not built (file-hash `check_model_copy` only); a window
with zero verdicts closes with `king_stays_no_candidates`; the frozen king
means a challenger that lands in the last minutes of a window is compared
with the same king as the first — by design.

### The king seat — king-failure datagen (LIVE 2026-09-10, data event, no wvk)
Operator directive 2026-09-10 ("do the simplest thing first: trigger on
the new king, spin up the new king on our fleet, sample envs from Prime
env from it, add only the failures to the dataset"). Motivation:
`research/results/reign9_vs_teacher_diagnosis.txt` — kings fail at depth
in their OWN trajectory context (loops, shrinking commands) while D held
only teacher-trajectory prefixes (covariate shift; DAgger fix). Pieces:
- **Controller** `ops/king-datagen/kingctl.py` (pm2 `affine-king-datagen`,
  60 s ticks; config `king.toml`, state `state/state.json` 0600 with the
  per-box bearer). Reads `affine/state/state.json` → king; rents a Lium
  pod `king-dg-<digest12>-<hex4>` (first `[[types]]` with stock under cap:
  h200-2x, b200-2x, pro6000-4x/8x, …; budget $25/h = two boxes during a
  swap/rotation), pushes `bootstrap_king.sh` (vLLM 0.28, replicas TP per
  type behind nginx on the first mapped data port, `--served-model-name
  king-<digest12>`, qwen3_xml tool parser + qwen3 reasoning parser,
  max_model_len 262144, per-replica compile cache), waits for `/v1/models`
  + a chat canary, then writes `/root/rollouts/.king_env` (KING_BASE_URL /
  KING_MODEL / KING_KEY / KING_DIGEST / KING_REIGN) on every
  `affine-datagen*` pod and releases the previous king's box. R2 kings come
  from the public `models.affine.io` copy; an HF genesis king from its
  pinned revision. `kingctl.py status` / `unpublish` / `pods`.
  **Fault tolerance (release 2026-09-10 16:00 UTC, drilled live + in
  simulation `/tmp/kd_sim.py`-style: rent→ready→publish, listing flake,
  wedged engine, dark→re-rent, crown change, rotation, stale state.json):**
  lost controller state → memory RECOVERED from the box's `/root/king/env`
  + `ready` marker (never released; verified live by moving state.json
  aside); a pod missing from the Lium listing is forgotten only after
  `pod_forget_ticks = 5` consecutive misses; an unreadable state.json keeps
  the last king `state_stale_min = 30`; after READY a canary completion
  runs every `canary_every_min = 10`, 3 misses = dark even while `/models`
  answers; dark past `unreachable_grace_min = 20` → remove → re-rent (the
  accepted ~45 min gap; teacher datagen continues); bootstrap failure /
  timeout → remove + executor strike (`state/blacklist.txt`); Lium TTL
  (72 h) → replacement rented `rotate_before_ttl_hours = 2` before
  `removal_scheduled_at`, published once it serves, old box removed (zero
  gap); on a crown the old king's PUBLISHED box stays until the new one
  serves, any never-published box of another king is removed at once; no
  serving box → `.king_env` emptied so the king policies idle; datagen-pod
  watchdog every 5 min (`pgrep -f -x` on the supervisor + bootstrap loop;
  both gone `watchdog_relaunch_min = 10` → relaunch `bootstrap.sh`; loop
  alive but supervisor gone → crash-loop alert; ssh unreachable → alert);
  every state change is one Discord line (`[discord]`, channel
  1381987595881414656, token `DISCORD_BOT_TOKEN_ARBOS_BITTENSOR`).
  Deploy to the pods with `ops/king-datagen/deploy_pods.sh [--restart]
  --all` (scp + registry import check + `/root/rollouts/RESTART` flag —
  the supervisor exits at its next cycle boundary and the bootstrap loop
  relaunches it; nothing is killed).
- **Rollouts** (`rollouts/`): `Endpoint.model_env` / `base_url_env`
  (resolved per pick against a shared env dict; unset = unavailable, like
  a missing key); `rollouts/king.py` re-reads `.king_env` every supervisor
  cycle (no restart on a crown). `policies.toml` `king_textbased /
  king_bashtool / king_pi / king_claude_code / king_boxed / king_toolcall`
  (share 1.0 vs teacher 2.0 → ~1/3 of new rollouts per source while the
  seat is up; T=0.8). `sources.toml`: king policies on `[defaults]` +
  math/wiki/agent; `[mix]` scaled ×0.9 + `king_fail = 0.10`; `[king_fail]
  strata_buckets = 1000, policy_prefix = "king_"`. Deployed to the three
  datagen pods 2026-09-10 13:20 UTC (rollouts files only — the pods'
  `/root/affine` tree is pre-wvk-13 and used for yield accounting only;
  datagen-3's bootstrap loop had been dead since its 2026-09-09 06:51
  reboot and was relaunched). **Dead-endpoint guards (16:00 UTC
  release):** before any batch the supervisor runs `EndpointHealth.
  preflight` — `GET /models` on every DYNAMIC endpoint (`base_url_env`
  set); a miss strikes it (exponential cooldown 60 s → 15 min) and the
  cycle is skipped without a batch failure or a zero-yield strike on the
  source; `Scheduler.pick_policy` skips policies whose endpoints are ALL
  cooling (falls back to every usable policy so a source is never left
  unpicked). One endpoint name `king` is shared by all `king_*` policies,
  so one strike idles them all for the cooldown. `/root/rollouts/RESTART`
  makes the supervisor exit between batches (graceful redeploy).
- **Fold** (`ops/corpus_build.py`): view records now carry `outcome`
  (`affine.corpus.view.rollout_outcome`: `errors` non-empty or
  `stop_condition` ∉ {agent_completed, max_turns} → **errored** (harness /
  API failure the env still graded 0 — 3 of 48 king rollouts in the first
  two batches); else `rewards.solved.score` → solved / failed / unscored).
  `route_king_fail` (after `assign_bucket_strata`, so it wins on
  math/tool_use too): `king_*` records with `outcome == failed` →
  `fold_group = king_fail`, `stratum = king_fail:<sha256(instance_id) %
  1000>` (own namespace — sharing the teacher's `repo|phase` strata would
  add within-stratum variety and move no share under `cap_fill`);
  `solved` → drop `king_not_failed`, `errored` → `king_errored`, else
  `king_unscored`; `group_of` honours `fold_group`. Announce `by_group`
  shows `king_fail`. First real routing (dry run 2026-09-10 15:22 UTC):
  32 failed king rollouts → 1,303 turns, 36 successes dropped.
  **Prefix cap (data event, operator decision 2026-09-10):**
  `datagen/slicer.py MAX_PREFIX_CHARS` 120_000 → **300_000** (median king
  failure trajectory is 193k chars; the deep turns are the point) plus a
  tokenizer-measured guard in `derive_chunk` — prefixes > 120k chars are
  tokenized with the teacher tokenizer and dropped past `MAX_PREFIX_TOKENS
  = 110_000` (`prefix_too_many_tokens`; serving window 131072 − 1792 gen).
  No wvk change: the duel scores whatever prefix D carries.
- **Known gaps (by design, "simplest first"):** the failure label is a
  noisy proxy (a failed run has good turns too; only the outcome filters);
  teacher refs hit the 1792-token cap more at depth (refs<2 → turn
  dropped) — `[duel] max_thought_tokens` 1024→4096 is a contract change
  and is NOT applied; no paired filter, replay buffer or decay yet; king
  successes are discarded rather than used as a control; no hot standby
  (a dead box means ~45 min without king rollouts, by operator choice).
- **Breadth release (2026-09-10 evening, operator: "data broad enough for
  a really good agent — Claude Code, Hermes, terminal-bench agent, …"):**
  (a) **Claude Code 401 fixed** — Claude Code speaks Anthropic Messages
  and sends the key as `x-api-key`; vLLM `--api-key` reads only
  `Authorization: Bearer`, so every `king_claude_code` call got
  `upstream 401` (2 batches, 48 errored rollouts). The king box's nginx
  now maps `x-api-key` → bearer (`bootstrap_king.sh`; patched live).
  (b) **Seats** (`rollouts/scheduler.py`): "done" is per (source, seat) —
  teacher seat = all non-king policies (unchanged), each king =
  `king:<served model>`. The king replays tasks the teacher finished, and
  a new king starts over. Before this every pool the teacher had exhausted
  (terminal, math, tool_use, nl2repo) was closed to the king: epoch 22's
  1,994 `king_fail` turns were 100 % `king_textbased` from scaleswe /
  swesmith / terminal_lego. (c) **Three more harnesses**, teacher + king
  pairs in `[defaults].policies`: `kimi_code` (tool_call; king probe 27
  turns / 0 drops), `hermes_agent` (tool_call), `terminus_2` — harbor's
  terminal-bench reference agent, reply = one JSON object
  `{"analysis","plan","commands"}` → **new dialect `terminus_json`**
  (`affine/dialects.py`; `Dialect.marker_roles` lets its mandate sit in
  the first *user* message, where Terminus states its format; every other
  dialect keeps the historical first-system-message check bit-for-bit).
  King probe: 29 turns / 0 drops, one terminal-bench task solved.
  **ADMITTED 2026-09-10 ~21:00 UTC — wvk 13→14** on the explicit operator
  directive "do it now yes" (same day; it had been staged behind
  `[dataset].allowed_action_kinds` for ~2 h, so no `--rederive` was
  needed: no fold had run since the terminus policies went live).
  Commit `fcb2354` (toml: allowed_action_kinds += terminus_json, wvk 14 +
  history paragraph), `8ef06b2` (llms.txt "Fork history: wvk 14", dialect
  table row, marker-rule note). Deploy = `/tmp/v8flip/deploy.sh` (the v7
  script re-pointed): pod was idle (queue 0, no in_flight) → pm2 stop →
  `redeploy_pods.py` (dialects.py with terminus_json + toml) → pm2 start.
  Forward-only; reign 11 stands; `min_submission_block` unchanged. The
  next fold (pm2 cron 16:00 UTC) admits the staged terminus turns; the
  duel tripwire accepts slices carrying them once the pod is on wvk 14.
  Discord notice `…/1547714360326094858`; llms.txt live with the section.
  (d) `EndpointHealth.preflight` probes a cooling king endpoint instead of
  skipping it (a source whose every policy cooled spun the cycle loop at
  5 s while the fallback pick was refused). Pods get
  `affine/dialects.py` + `affine/corpus/trace.py` via `deploy_pods.sh`
  (`AFFINE_FILES`). Codex and OpenClaw stay out: both speak OpenAI
  Responses with two leading system messages the teacher template refuses
  (see `policies.toml`). (e) **Turn-cap artifact** (`corpus/trace.py
  is_turn_cap_artifact`): ACP harnesses raise interception's refusal past
  `max_turns` as a `HarnessError` ("rollout stopped: max_turns") and
  verifiers then skips grading (`rewards == {}`); `trace_error_type` /
  `rollout_outcome` no longer count it as an error, and an ungraded
  `max_turns` rollout is `failed` (the agent did not finish). Before: the
  first post-401 `king_claude_code` batch — 24/24 rollouts at the 80-turn
  cap, 653 tool_call turns, prefixes p50 111k / max 223k chars — folded to
  zero as `king_errored`, and 15–20 % of the teacher's Claude Code
  rollouts had been dropped the same way since 2026-09-07. Forward-only
  (the fold re-derives only unfolded chunks); `--rederive` would back-fill.
  (f) `rollout_outcome` reads the env's primary grade in order `solved` /
  `correct` / `passed_fraction` (`PRIMARY_REWARD_KEYS`): affine-math grades
  into `correct`, so every king math rollout had been `king_unscored`
  (120/120 in the 20:05 UTC dry fold); affine-wiki grades nothing and stays
  unscored. (g) `king_boxed` / `king_toolcall` carry `max_tokens = 16384`:
  the null harness sets no cap and the king box serves 262k context, so one
  looping math reply held a 24-task batch for the full 3600 s rollout
  timeout (datagen-2 19:54, datagen-3 20:11 UTC). Once the teacher's pool
  is done the seat scheduler drains the king's math pool first (math is the
  cheapest deficit: 1 turn per rollout, ~1–4 min per batch, ~1 h per pod).

### History — Reason v4 (wvk 7–9, 2026-08-17 → 2026-08-27)
v4 was the uncentered tempered LME, `Reason = tau·log((1/k)·Σ exp(a_i/tau))`,
same B gate and length floor, δ = 0.002 (0.001 experiment 2026-08-21 reverted
2026-08-22 after winner's-curse crown churn: 4 near-noise crowns in 18h with
the winners' measured Reason drifting down 0.01954→0.01809). Retired because
the flat-lift channel (filler suffix) was live and won the board.

**Why v4 (2026-08-17):** the teacher's next-action distribution is
multi-modal; v3 scored the thought against ONE sampled ref and averaged in
log space, punishing a missed mode without bound (teacher-vs-own-thought
blind ≈ −0.010/byte, n=509). Equilibrium was non-committal filler
("hedge"). The tempered log-mean-exp is dominated by the best-matched ref:
a miss zeroes its own share but cannot drag the turn below the credit from
a hit, so committing to the teacher's dominant next action becomes optimal.
`k=1` reduces exactly to v3 (archived rows replay bit-identically through
the same code path); `tau→∞` recovers the broken mean. `tau=0.03`
calibrated externally (AIIan, n=100 turns: flip from hedge-wins to
commit-wins near τ=0.1, decisive at 0.03). Cold-τ failure modes —
mode-guessing (hit 1-of-k by guessing the modal action) and leakage
amplification (a pasted action's ref term dominates exponentially) — are
why τ is warm, plus the length floor, B leakage check, and post-crown audit.

B = `lpC(y_A|z_A) − lpC(y_A|∅)`; a rollout passes if B ≥ 0.02 and no
leakage; the miner is licensed if pass rate ≥ 0.30. The length floor evicts
empty / cue-thought kings; B is the license against padding. The z-test is
relative to the slice's own noise (bare `SE = stdev/√n`; false-crown ≈
2.28%/duel ≈ 1 in 44 at 2σ for a *distinct* zero-edge model). δ exists
because that test tracks the challenger's own variance: an ε-copy of the
king would crown on pure noise at the same 1-in-44 — under the floor its
tiny SE turns δ into a ~z≥6 bar, and no SE-compression (A11) can pull the
crown bar below δ. Scale history: v3-era live 2·SE ≈ 0.0035 with δ=0.002;
the v4 tempered score compressed margins and noise ~2x (82 live v4 duels:
median 2·SE ≈ 0.0013). On 2026-08-21 δ was lowered to 0.001 (wvk 7→8) to
restore the v3 δ/SE ratio; 18 hours of live data reverted it (wvk 8→9,
2026-08-22): with δ at the noise floor, 4 crowns landed at z≈2.1–2.7 and
margins 0.00116–0.00153, and the winners' duel-measured Reason fell
monotonically-ish across the chain (0.01954→0.01898→0.01847→0.01766→
0.01809). That is the winner's curse: near-threshold crowns are selected
for lucky slices, so each reign change re-baselines the king slightly
lower and the ratchet leaks. δ=0.002 (~1.5x the 2σ bar at v4 scale) is
deliberately ABOVE the statistical bar — it trades a blocked marginal
true-improver (chal-00967-class, z=2.77) for immunity to noise churn
(v3 calibration: `research/results/delta_calibration.{json,txt}`;
merges / fresh parity models crowning stays policy-accepted, 2026-08-12). The ranked quantity
lives entirely on the teacher side, which retires the whole lpA attack
surface by construction.

### Live instrumentation (min(R,G), v5 2026-08-27)
GPU work per turn is 1 miner sample + k=3 Reason echoes `lpC(y_i|z_A)` (one
per teacher ref) + B echoes `lpC(y_A|z_A)` / `lpC(y_A|∅)` once per rollout
(B is ref-independent; deduped in `evalsrv/terms.py`) + the grounding
echoes: `m = lpC(z_A|x)` once per distinct miner rollout (`lpC_za_x` on
pairs) and `t_i = lpC(z_C^i|x)` once per ref (`lp_thought` on ref records,
shared both sides via RefCache). Refs supply `lpC(y_i|z_C)` / `lpC(y_i|∅)`.
Prior-bank and retired lpA / extra lpC echoes are **off** (`reason_only`,
`score_bank=false`). Still published: η (sufficiency), B mean/pass rate,
thought/action lengths + teacher deltas, `duel_seconds`, and new per-side
leg telemetry `mean_r_leg` / `mean_g_leg` / `g_bind_frac` (which leg binds);
verdict `duel_params` stamp `tau`, `n_teacher_samples`, `score_mode`,
`band_c`, `band_floor`. Pre-fork / full-telemetry verdicts may still carry
miner-side causality, bank, r, baseline, L1lift.

### History — S\* v2 (retired 2026-08-10)
v2 was `S = mean(Λ2 + w·clip(L1lift, ±0.1))` behind 4 gates (causality γ=0.30,
bank γ_bank=0.08, calibration r∈[0.3,4], baseline band 1.25×) and a duel floor
δ=0.02 + min_se=0.005. Retired because raw Λ2 correlates with swe-rebench as well
as the mix (+0.847@15 vs +0.844) while the L1 term dragged in most of the contract's
complexity (RT-3 family + three defensive gates), and A11 was already policy-dead.
Pre-fork verdicts stamp the old formula/`gates` block and remain replayable
(`legacy_duel` in `research/harness/score.py`).

### Headline correlations (coding D, measured under v2-era freezes)
| set | Spearman(S, swe) | notes |
|---|---|---|
| ungated @ n=15 | +0.844 (mix) / **+0.847 raw Λ2** | early freeze; basis for v3 |
| ungated @ n=19 | +0.856 / +0.862 under clip | |
| ungated @ n=30 | **+0.758** (p≈1.2e-6) | wave-5; XC soft outlier |
| hybrid @ n=15 | +0.799 | many mid kings at γ_bank knife-edge ~0.075 |
| LLM judge (same turns) | ~+0.31 | S ≫ judge; prompt variants checked |

Second teacher (Qwen3-32B vs GLM-Air), n=6 kings: Spearman(S_T1, S_T2) = **+0.943**.

---

## 3. Red-team status (load-bearing)

Statuses restated 2026-08-27 for min(R,G) v5 (centered R + banded G;
gates removed; L1 channel unscored):

| ID | Attack | Status | Defense under min(R,G) v5 |
|---|---|---|---|
| **RT-11** | **filler-suffix flat lift** (fixed generic suffix on every thought inflates all a_i equally — the reign-41 exploit that won the v4 board) | **CLOSED (v5, 2026-08-27)** | centering cancels any ref-independent lift exactly; the filler thought also lands far below the grounding band, so G ≪ 0 binds the turn |
| **RT-12** | **thought parroting** (paste the teacher's own thought / gold action to max grounding) | **CLOSED (v5)** | band is two-sided: m above `mu + w` is negative too; per-duel-fresh refs mean the parrot target is unknown; B leakage check still applies |
| RT-1 / A1 | fixed thought payloads | CLOSED | Reason ≈ 0 loses the relative duel; fixed thoughts also fail the grounding band |
| RT-2 / A2 | action stuffing into z | CLOSED | must beat incumbent at 2σ; y_i fresh per duel (leakage fails B) |
| RT-2 / A9 | silent / cue-thought / hedge-filler miner | **CLOSED (v4, 2026-08-17)** | length floor + B license + tempered LME: filler earns ≈ 0 per turn and loses duels — hedging is no longer the optimum, committing is |
| RT-2c / A2c | paraphrase stuffing | MITIGATED | ties genesis on raw Reason ⇒ cannot dethrone; **bank telemetry monitored** (residual watch item) |
| RT-soft-pad / A10 | soft-idents pad | CLOSED | abandoned by attacker; single-term score |
| RT-4 / A4 | king copy | CLOSED | δ floor (0.002 2026-08-12; 0.001 experiment 2026-08-21 reverted 2026-08-22 after winner's-curse crown churn): at 0.002 an ε-copy needs ~z≥6 to luck past δ |
| RT-3 / A3 | L1lift / overconfidence | **DEAD CHANNEL** | L1lift is not scored; lpA never enters the ranked quantity |
| A11 | short-style I/II FP | MOOT | policy-accepted 2026-08-05; SE-compression variant capped since 2026-08-12 — the crown bar never drops below δ |
| RT-6 / A6 | dataset sniping | **CLOSED (code, 2026-08-06)** | seed-shuffled strata + per-duel fresh y_i — see §5 |
| **RT-9** | **mode-guessing** (hit 1-of-k refs by guessing the modal action from surface cues) | **WATCH (v4→v5)** | τ=0.03 kept warm (one hit does not dominate); centering means a guess only pays via cross-ref spread; length floor + B + post-crown audit; monitor per-turn a_i spread telemetry |
| **RT-10** | **leakage amplification** (pasted action's ref term explodes under LME) | **WATCH (v4→v5)** | B leakage check fails the license; a pasted action also pushes m above the grounding band; audit; monitor |
| **RT-13** | **in-band mimicry** (train thoughts that sit inside the grounding band while flat on R) | **WATCH (new with v5)** | R leg still binds under min (centered R ≈ 0 for content-free in-band prose); monitor `g_bind_frac` and first-crown thought audits |
| **RT-7 / A12** | **isomorphism inverts on the live panel** | **OPEN — no defense** | see §3b |
| D_tau2 | programmability falsifier | **NOT demonstrated** | see §6 |

### 3b. RT-7 — the coding claim does not survive the live board (2026-08-09)

Every panel behind +0.758 is Albedo kings, optimised against a **GLM judge** —
adversarial to SN97, not to us. On the live SN120 board, where every submission
was made by someone maximising S, the sign flips:

| statistic | value | p |
|---|---|---|
| Spearman(duel margin, swe_lite), n=29 | **−0.421** | 0.024 |
| Spearman(S, swe_lite), n=29 | −0.371 | 0.049 |
| freeze (Albedo, n=30) | +0.758 | — |

- **All three S-crowned kings resolve 0/25** (kevin954, TalentPigs, Tok331102) —
  pooled **0/75**, binomial p=**3.7e-4** even against a conservative 0.10 null.
  Only genesis (0.20) is non-zero and it was seeded, never won a duel.
- **Untouched `Qwen/Qwen3.6-35B-A3B` scores 0.24 — best of 51 benched models.**
- **Same-miner control:** Tok `af5` swe 0.16 *lost* (S=−0.014); `af10` swe **0.00**
  *crowned* (S=+0.0446). Goodhart with confounds held fixed.
- **Mechanism:** raw genesis loses to the king by **−0.055** (n=80, z=−6.05, all
  gates clear) via **Λ2**, and 45 structurally distinct families fail identically
  (λ2_c −0.017…−0.029 vs king +0.005). Λ2 rewards thoughts that help the teacher,
  which the incumbent maximises by construction, so it acts as a
  **similarity-to-incumbent term rather than a capability term.**

The v2 gates were validity checks (causality, leakage, bank, r, band); none asked
whether the winner can write code — a model could be gate-valid, crown, resolve 0/25.
Reason v3 removes them outright: the public claim is a **distillation meter**, not a
coding meter, and crowns do not imply benchmark capability.

**Bench repeatability:** genesis scored 0/25 then 5/25 on the *same revision*, so
single 25-task scores are not per-model evidence — hence the pooled binomial test.
Outcome noise attenuates Spearman toward zero, so −0.42 is a conservative floor.

**Do not claim coding isomorphism without this caveat.** Artifacts:
`research/results/rt7_live_isomorphism.{json,txt}`, `research/scripts/rt7_live_isomorphism.py`.

**Equilibrium framing (policy 2026-08-10, `research/docs/EQUILIBRIUM.md`):** we
care about alignment of the *asymptote*, not intermediate states. Leaking-is-
knowing in the fresh-D regime (perfect thought = "the answer is X", which
requires computing X = distilling GLM). Reason v3 takes this to its limit:
gates and the δ ratchet are gone; the only ratchet left is the incumbent
itself — every crown raises the raw-Reason bar the next challenger must beat
at 3σ, so capability-free channels stay finite budgets unless an **unbounded**
one exists (none demonstrated — that is the red-team target). RT-7 under this
frame = live board is on the shallow style prefix of the slope; kings at 0/25
are the budget being spent.

Full writeups: `research/docs/REDTEAM.md`.

---

## 4. Contract snapshot (`affine/affine.toml`)

- netuid **120**, finney
- official site: **https://affine.io** (dashboard + llms.txt; Cloudflare-proxied
  to the validator box — sn120.arbos.life is a legacy alias via the CF tunnel)
- `weight_version_key = 15` (2026-09-12 ~17:01 UTC, explicit operator directive
 16:39/16:43 UTC: `crown_mode = "window_best"` — the crown is decided per 12 h
 block window, best positive margin + confirmation slice, see §v9; forward-only,
 reign 11 stands; 14 = 2026-09-10 ~21:00 UTC, explicit operator
  directive "do it now yes": `allowed_action_kinds` += `terminus_json`, the
  Terminus 2 / terminal-bench agent JSON command batch; forward-only, reign
  11 stands; 13 = 2026-09-09 ~02:00 UTC, explicit operator
  directive "make all the changes and flip the bit": `require_think_close
  = true`, `allowed_action_kinds` += `text`, protocol probe enforced;
  forward-only, reign 9 stands; 12 = forfeit floor `forfeit_turn_score = -0.1`,
  2026-09-05 ~16:30 UTC, explicit operator directive "yes add this
  forfeit" after the staged-flip review; forward-only, reign 5 stands;
  11 = action dialects + schema-3 trace-first D,
  T0 ran 2026-09-05 12:12 UTC on explicit operator directive "Run the cut
  over" — a day after the noticed 2026-09-04 18:00 slot, which passed
  unrun; commit `45f3466`; scoring rule unchanged = min(R,G) v5, forward-
  only, reign stands; min(R,G) v5 fork + genesis reset was 10, 2026-08-27,
  explicit dated operator directive; δ revert was 9, δ=0.001 was 8,
  Reason v4 was 7, B gate 6, thought-length floor 5, δ floor 4,
  Reason v3 was 3). **Do not bump** without an explicit dated operator
  directive — not for teacher-host moves, serving knobs, corpus refresh,
  or agent “cleanup.” Leave the integer alone.
- teacher: `Qwen/Qwen3.8-27B` (co-located on eval; swapped from
  `zai-org/GLM-4.5-Air-FP8` 2026-08-27, explicit dated operator directive,
  bundled into the wvk-10 fork — GLM served eras wvk ≤ 9. Requires
  vLLM ≥ 0.28 (GDN kernels ICE cutlass JIT on 0.22.x); echo chunk 8192 /
  util ≤ 0.75 for the 248k-vocab fp32 logprob spike. 2026-08-10 GLM-5.2
  remote-teacher push torn down, never cut over). **2026-09-07 serving
  changes, all ops-only (no wvk):** `max_model_len` 65536→131072 on teacher
  swarm + miner slots (a 63,745-token schema-3 prefix + 1792 gen overflowed
  by one token; `Fault.CONTEXT_LIMIT` requeued the same entry 25× for 14.5 h);
  second live teacher box (`eval-b200-8x` target=1, 16 replicas); **echo
  prefix caching** — vLLM plugin `ops/teacher-swarm/echo_cache_plugin`
  lets `echo=True` requests reuse the cached prefix KV and recompute only
  the scored tail (`vllm_xargs.affine_echo_tail`; cached rows marked with
  logprob +1.0, client falls back to uncached on any span overlap; parity
  on 584 stored echoes inside batch nondeterminism, 3.5–4.4× per replica);
  `[duel].concurrency` 64→192; echo tokenization in a thread pool + orjson;
  router passes bodies through. Scoring 52 → 26.6 min/duel. Prefetch
  stall watchdog now reads the r2store child's heartbeat (every earlier
  "stall" was a metric artefact). **Challenger warm-swap LIVE (2026-09-08
  00:12 UTC):** challenger engines stay up between duels and the next
  checkpoint's weights are loaded in place (`evalsrv.vllm_ext.WeightTools.
  affine_direct_load` over vLLM's dev-mode `/collective_rpc`, then
  `/reset_prefix_cache`, then a count-continuation probe; ~136 s vs ~8.5
  min cold). Verified on 1×B200 (TP1) and 2×H200 (TP2): swapped weights
  bit-identical to a fresh load on every rank (948/948 tensors), logprobs
  Δ=0.0 to 39k tokens, 10-min duel-like soak clean — for both the raw path
  and vLLM's layerwise `reload_weights`. vLLM's "weights were not loaded"
  warning on fused MoE experts is a return-value artefact, not data loss
  (the same-day scare + two engine deaths were the operator's own redeploy
  pkill). Guards: `_swap_compatible` (config.json/tokenizer/generation_
  config/tensor-set identical, else cold relaunch) + per-rank loader-report
  invariant (906 reported / 80 unreported fused names) + probe. Total per
  verdict 35 → ~29 min. `AFFINE_CHALLENGER_WARM_SWAP=0` disables.
  **Redeploy pitfall:** `scripts/redeploy_pods.py` writes the pod's
  `.eval_env` from the *calling* process's environment — export from
  `/proc/$(pm2 pid affine-validator)/environ`, never from a `pgrep -f`
  match (2026-09-07 23:07: a shell wrapper matched, the pod got no R2/HF
  creds, and the king snapshot was pruned; guard `_is_public_king_cache`
  committed in `725240e`, deploys with the next redeploy).
- **architecture pin (2026-08-28, explicit operator directive):** submissions
 must be genesis-family fine-tunes — `config.json` must match
 `[submission.pinned_arch]` (Qwen3.6-35B-A3B shape: qwen3_5_moe, 40 layers,
 256 experts, vocab 248320, …) on every pinned key; dtype/rope/token ids free.
 Enforced pre-download at dispatch + prefetch (`validate_repo_arch`). Closes
 teacher-upload: the frozen teacher tops min(R,G) by construction (its thoughts
 are in-band on G and best-predict its own actions on R), so an open board
 converges to "first teacher uploader holds the throne". Found live: 4 of 28
 queued entries on 2026-08-28 were Qwen3.8-27B-shaped. Admission rule, not a
 scoring change — no wvk bump; verdicts/replays untouched.
 **2026-09-04 (explicit operator directive, "allow it"):** the text-only
 extraction of the genesis — `Qwen3_5MoeForCausalLM`, vision tower dropped,
 `text_config` flattened to the root, `model_type = qwen3_5_moe_text` — is
 admitted via `[[submission.pinned_arch_alt]]` (match primary OR any alt
 profile; `validate_repo_arch(..., alternatives)`; `submit.py
 PINNED_ARCH_ALT` mirrors it). Trigger: chal-00251 rejected on the class
 name alone. vLLM 0.28 serves the class. Teacher still matches neither.
 Live since the 2026-09-05 11:57 UTC validator restart.
- **release stage 1 (2026-09-03 15:47 UTC, explicit dated operator
 directive; commit `11c4806`):** (a) private R2 mining live —
 `[submission.r2].enabled = true`, `hf_cutover_block = 8987674`; `affine1`
 HF reveals above that block are dropped at intake, queued ones still duel;
 miners activate with an **Ed25519** hotkey (`affine2|activate`), get a
 sealed credential from `dash.affine.io/mailbox/…`, upload to the private
 bucket, `affine2|ready`; only crowned models are copied to
 `models.affine.io`. Validator identity
 `5Ch9qcQ1X4QBPSVEXaSVmCJY9mPAdj4vRHfaK5opW7DSXucs`. (b) corpus D served
 from `https://data.affine.io` (`[dataset].corpus_base_url`): same schema-2
 manifest, same sha, byte-identical `turns/**`; Hippius `turns/**` read-only.
 Admission + data location only — **no scoring change, wvk stays 10**. The
 wvk 11 fork (dialects + schema-3 view) is stage 2, 2026-09-04 18:00 UTC
 (moved forward from 2026-09-09 on 2026-09-03, operator directive; re-noticed same day)
 (`ops/t0/t0_cutover.sh`). Ran via `ops/t0/stage1_go_live.sh`.
- **action dialects (2026-09-01, mechanism only — no contract change):**
  where a turn's action span starts/ends is now a per-turn `action_kind`
  resolved through `affine/dialects.py` (`bash` = one closed ```bash block;
  `tool_call` = `<tool_call>…</tool_call>`; `boxed` = `\boxed{…}`) instead
  of a hardcoded bash regex. R/G/B math, centering, band, and the canonical
  thought rendering are untouched; only the action *parser* is pluggable.
  Bash replays bit-identically (2.6M texts from 80 stored duels). Live D is
  gated by `[dataset].allowed_action_kinds = ["bash"]` — the fold refuses
  other kinds, the duel tripwires on a slice containing one. **Admitting a
  new dialect is a contract event** (miners must then emit it on those
  turns or forfeit them): explicit dated operator directive + a wvk
  decision, not an agent edit. Rollouts policies declare their harness's
  dialect via `action_kind` in `policies.toml` (default bash).
- **wvk 11 fork LIVE 2026-09-05 12:12 UTC (notice posted 2026-09-02 as
  "not before 2026-09-09"; T0 moved to 2026-09-04 18:00 UTC by operator
  directive on 2026-09-03, re-noticed the same evening; that slot passed
  unrun and the operator directed "Run the cut over" on 2026-09-05; ran as
  `AFFINE_T0_DIRECTIVE=2026-09-05 ops/t0/t0_cutover.sh`, epoch 15 folded
  and announced, eval pod on schema 3, late notice posted to Discord):**
  admit
  `boxed` (math, `affine-math-v1`, MATH train) and `tool_call` (wiki
  search, `affine-wiki-v1`) turns to D at target shares math 0.10 /
  tool_use 0.10 (coding 0.50 / terminal 0.25 / nl2repo 0.05). Forward-only:
  reign stands, no genesis reset, `min_submission_block` unchanged. Notice
  is live in `llms.txt` ("Upcoming fork" section), the dashboard banner
  (`affine/website/index.html` `#fork-notice`, remove at T0) and Discord.
  Until T0 the gate stays `["bash"]`; non-bash turns accumulate on the HF
  staging dataset (rollouts views admit any *registered* dialect; fold,
  `corpus_push`, and the duel tripwire still enforce the allowlist; a
  temporary `[fold_mix]` in `rollouts/sources.toml` keeps the fold on the
  old mix until T0 — delete it then). Gate-closed dry run (2026-09-02, 20
  turns/dialect, teacher-as-challenger vs the bash king): pipeline sound
  (finite lp*, refs 2.55–2.8/3, bash replay parity 6/6 exact) with three
  findings for the T0 decision — (1) **forfeits are dropped from pairing,
  not lost**: `score.duel` pairs only turns valid on both sides, so a
  bash-only king loses no margin on dialect turns it cannot answer (already
  ~5% of bash turns today); (2) **boxed hits the token cap**: the king
  reached `\boxed{}` on 3/18 math turns — every miss was `finish=length`
  at `max_thought_tokens+max_action_tokens = 1792`, and the teacher's own
  ref yield was 2.55/3 for the same reason; a per-dialect or larger cap is
  a `[duel]` knob = contract change; (3) **tool_call carries little R and
  fails B**: 16/19 turns had all 3 teacher refs identical (next tool call is
  near-deterministic) so centered R ≡ 0, and per-byte B ≈ 0.005–0.010 <
  0.02 because the `<tool_call>` XML is boilerplate (teacher B pass 6%,
  king 15%, vs ~60% on bash). Math R is strong (teacher mean_r_leg 0.054 vs
  0.012 on bash) and B ≈ 0.84. Verdicts now stamp `duel_params.
  allowed_action_kinds`, `slice.dialects`, and per-side/teacher
  `by_dialect` telemetry (additive; single `bash` entry pre-fork).
  **T0 data decisions (2026-09-03 night, from staging rehearsals):**
  (a) D restarts from the traces — `ops/corpus_build.py --init --no-legacy`;
  the v2 epochs 1–13 are not imported (importing them froze coding: the
  waterfill counted 60k legacy turns at coding 67% and admitted zero new
  coding/terminal rollouts). (b) The fold enforces `[mix]` in **slice
  strata**, not turns (`cap_fill`): one turn per stratum per duel means a
  group's slice share is its strata share; the turn-count waterfill had
  made coding 4% of the slice. Rule: every key takes all it has except the
  single most over-supplied one, capped where the runner-up is on target.
  (c) The fold assigns math/tool_use strata itself from `strata_buckets`
  (math 470→830; tool_use stays 470 = its 478 tasks). Epoch 14 (pre-published
  2026-09-04 03:20 UTC, not read until the toml flips): 189k turns / 14.2k
  rollouts / 7.4k strata; slice ≈ coding 56 / terminal 27 / math 11 /
  tool_use 4 / nl2repo 1 %; dialects bash 64 / tool_call 24 / boxed 11 %;
  coding languages py 32 / go 26 / java 18 / ts-js 17 / rs 7 %. (d) No
  per-dialect token cap at T0: teacher math completions finish ≤1792 tokens
  in 78% of trace calls (p50 557); the mechanism exists
  (`[duel.max_tokens_by_kind.<kind>]`, empty = unchanged) if wanted later.
  **Second harness (2026-09-02):** coding + terminal sources also run
  under verifiers' `bash` harness (native `bash` + `edit` tools via tool
  calls; policies `teacher_bashtool` / `glm_bashtool`, `action_kind =
  tool_call`, shares mirror the textbased pair → ~half of new shell turns).
  Probe on 2 terminal-bench tasks: 28 turns derived, baked parity held,
  system marker "tool" present. Staged until T0 like every non-bash turn.
  Same-task/two-prompt-styles is the anti-scaffold-overfit lever. Third
  harness the same day: `pi` (pi-coding-agent over pi-acp; own 2.5k-char
  system prompt, tools read/bash/edit/write; policies `teacher_pi` /
  `glm_pi`, equal shares → three harnesses split shell turns ~evenly;
  probe 16 turns derived, parity held, 1/2 rollouts HarnessError "no
  visible reply" = teacher reasoning-only answer). hermes_agent / rlm are
  `tool_call` too and need no new dialect — terminus_2 (JSON command
  blocks) would.
  **Prefix = what the model saw (2026-09-02, mechanism only):** turn
  prefixes are now built per reply from the verifiers message *graph*
  (`rollouts/schema.py::sampled_paths`: root→node path of each sampled
  assistant node; `slice_messages(..., turn=(i, n))` slices that path for
  its last message only) instead of flattening `nodes` in order. The path
  is by construction the exact prompt the model was sent, for any harness
  — echo re-serialization (pi), dropped turns, compaction, subagent
  branches — with no per-harness pattern. Found while checking: mini-swe
  parses a reply *before* adding it to history, so a FormatError /
  token-limit reply never entered the model's context, but the flattened
  walk kept it: 3,595/13,310 stored mini-swe trajectories (27%) and
  17.9% of turns had a prefix with a phantom assistant turn. Parity on
  1,110 linear traces: records identical; 270 forked traces now slice to
  `…user, user(nudge)` prefixes (+6% turns: shorter prefixes clear the
  char cap, phantom-turn false "leaks" gone). Not a contract change; the
  duel scores whatever prefix D carries. Prior-turn `reasoning_content`
  (Qwen3.8 template renders it in tool loops when the client sends it
  back; pi / bash / null harnesses do) is still dropped from prefixes —
  open decision, RT-13 surface.
  **Trace-first corpus on data.affine.io (built + gated 2026-09-02, cuts
  over at the wvk 11 T0 with `ops/t0/t0_cutover.sh` — same directive, one
  event):** the canonical dataset object is now the full rollout **trace**
  (envelope: task + policy + verifiers message graph incl.
  `reasoning_content` + tool schemas), published by the datagen pod
  straight to Cloudflare R2 bucket `affine-data` = `https://data.affine.io`
  (`traces/chunks/*.jsonl.gz`, sha-named immutable; `traces/manifest.json`
  → `traces/manifests/{sha}.json`; `rollouts/r2mirror.py`, live since
  13:05 UTC, 1045 chunks / 23,655 rollouts; HF `affine-rollout-traces` is a
  cold copy behind `ROLLOUTS_HF_TRACE_MIRROR`, drop after 30 d; turn staging
  to HF retired). D is a versioned **view** `duel_turns@v4`
  (`affine/corpus/view.py`: one record per rollout = baked plain-text node
  graph + turn metas; prefix = root→parent path of `node_id`;
  `materialize_turn` handles v2 `msg_pos` and v4 `node_id`). Fold:
  `ops/corpus_build.py` (replaces `ops/datagen_refresh.py`; pm2 entry
  repointed, start only after T0 `--init`) → `views/duel_turns@v4/{chunks,
  index}` + `corpus/manifest.json` (schema_version **3**, `view_spec`,
  `traces_manifest_sha256`, `allowed_action_kinds`; first revision's
  `prev_manifest` chains into `turns/manifests/`). Hippius `turns/**`
  copied byte-identical to `data.affine.io/turns/**` (39 objects, sha
  verified). Evalsrv `CorpusSync` has the schema-3 branch; verdict `slice`
  gains `view_spec` + `corpus_base_url`. **Gates passed 2026-09-02:**
  derivation parity 23,655/23,655 envelopes identical (v3 `derive_turns`
  vs v4 view, all 11 sources); legacy import 600/600 sampled v2 turns
  materialize byte-identical (prefix, reference, strata); legacy replay 8
  verdicts × 4 manifests (epochs 10–13) reproduce `turn_ids` + `slice.
  digest` from `data.affine.io/turns/manifests/{sha}`; golden slices 3
  seeds × 1300 on the staged schema-3 manifest (bash ~1060 / boxed ~145 /
  tool_call ~85, `check_dialects` passes with all three, trips on
  `["bash"]`); 60-turn gate-closed mini-duel (20/dialect, teacher vs bash
  king, v4-materialized prefixes): finite lp*, refs 2.75–2.95/3; pod dry
  run to `staging/traces/` then promoted; size 0.86 GB traces + 60 MB
  views (≈$0.03/mo). Staging rehearsal of the fold published epoch 14
  (legacy 59,745 + 105) and 15 (+4,289 boxed, +1,923 tool_call) under
  `data.affine.io/staging/`. `[fold_mix]` still holds the fold on the
  bash-only mix; `--ignore-fold-mix` is the T0 rehearsal flag. Contract
  patch for T0: `ops/t0/wvk11_T0.patch` (wvk 11, `allowed_action_kinds`,
  `corpus_base_url`/`manifest_key` → data.affine.io; `min_submission_block`
  unchanged). Notice extended in `llms.txt` (§ Upcoming fork "Bundled with
  the fork" + § Turn corpus D schema_version 3 block). Not in scope:
  reasoning in prefixes (`duel_turns@v5` later, same traces).
- **private R2 submissions (`affine2`, built 2026-09-02, go-live directive
 2026-09-03 — admission mechanism, no wvk bump):** miners no longer publish
 to HF. Flow: `affine2|activate|<hotkey>|<ed25519 sig>` → validator
 (`affine/registrations.py`) mints a prefix-scoped temporary R2 credential
 from a per-registration parent Cloudflare token, seals it (NaCl sealed box)
 to the miner's **Ed25519** hotkey and posts it at
 `https://dash.affine.io/mailbox/v1/<registration_id>/generations/…bin` →
 miner uploads the checkpoint + signed `manifest.json` to
 `affine-private-models/models/registrations/<registration_id>/` →
 `affine2|ready|<registration_id>|<sha256(manifest)>` → validator revokes
 the credential, verifies inventory/signature/hygiene/arch, enqueues with
 `repo = r2://<bucket>/<prefix>`, `revision = model_digest`. Eval pods fetch
 with a read-only key and verify every sha256 before vLLM loads
 (`evalsrv/r2store.py`). Only a **crowned** model is copied to the public
 bucket (`https://models.affine.io/models/sha256/<digest>/`); private
 prefixes expire after `private_retention_days` (60; a private-ref king is
 re-promoted on every weight sweep). Wire contract in `affine/r2protocol.py`; miner client
 `affine/scripts/submit.py` (`hotkey/check/register/auth/upload/ready/
 status/submit`). Toggle `[submission.r2].enabled` + `hf_cutover_block`
 in `affine.toml`; secrets in `~/.affine-validator.env` (the validator's
 env snapshot — the repo `.env` is NOT read by pm2) (`CLOUDFLARE_*`, `R2_*`,
 `AFFINE_MAILBOX_SIGNING_SEED`, `AFFINE_EVAL_R2_*`); infra bootstrap
 `affine/scripts/r2_setup.py`. sr25519 hotkeys are rejected at intake
 (`rejected_not_ed25519`). Validator state for registrations is
 `state/registrations.json` mirrored to the private bucket — no database.
- seed king: `Qwen/Qwen3.6-35B-A3B` @ `995ad96e` (min(R,G)-era genesis,
  unpaid — emissions burn until a registered miner crowns; the Albedo
  genesis `dendriteholdings/albedo-qwen3.6-35b-king-genesis` seeded eras
  wvk ≤ 9)
- turns: sharded corpus with immutable manifest; sha-pinned (see toml `[dataset]`)
- duel: n_turns=1300, k_sigma=2, min_margin=0.002 (0.001 experiment
  2026-08-21→22 reverted), min_thought_chars=80, causality_gate=true, causality_tau=0.02,
  causality_gamma=0.30, n_teacher_samples=3, n_miner_samples=1, tau=0.03,
  score_mode="min_rg", band_c=2.0, band_floor=0.002, reason_only
 (v2 knobs deleted from `[duel]` 2026-08-10; bank/lpA echoes off 2026-08-11;
 δ floor restored 2026-08-12; thought-length floor + B gate 2026-08-13;
 tempered k=3 / n_turns 2080→1300 2026-08-17; min(R,G) + genesis reset
 2026-08-27)

### Evalsrv roles
- `AFFINE_ROLE=duel` — teacher + king + challenger; Reason + B echoes
- `AFFINE_ROLE=bench` — SWE advisory (never part of the score)
- Bootstrap: `affine/evalsrv/bootstrap.sh` (fail-closed on empty sha / missing sources)

### Smoke
```bash
./setup.sh
source .venv/bin/activate && source .env
python affine/scripts/smoke_test.py
```

Secrets (`HF_TOKEN`, `AFFINE_EVAL_TOKEN`, chain wallet material, cloud keys) live in the
operator environment / secret manager — never commit `.env` or `.eval_env`.

---

## 5. RT-6 (dataset sniping)

Offline detectors **fail** (concentration/Gini and artifact-string probes are useless —
scaffold tokens are in 100% of turns; memorizers look *more* uniform).

Dated leave-repo-out on 20 commit-pinned repos (early/late 10+10), from stored pairs:
- ρ(S, swe) early +0.846 / late **+0.845** / early↔late **+0.947**
- Not carried by a few early trajectories.

**2026-08-06 incident: both intended mitigations were broken in code, found via a
miner report ("SFT to memorize the result").**
- `sample_slice` round-robined strata in *sorted* order and stopped at n=80; with
  417 strata the reachable pool was the ~267 turns of the alphabetically-first 80
  strata (96–99% slice recurrence, fully predictable). Fixed: strata order is now
  shuffled by the duel seed → pool = full corpus, cross-duel overlap ~5/80.
- `RefCache` persisted `y_C` across duels while artifacts publish them — recurring
  turns were frozen, known targets. Fixed: refs are per-duel now; the cache only
  dedupes teacher sampling between the two sides within one duel.
- Interaction with r_lo (v2-era analysis): at r_lo=1.0 mean L1lift ≤ 0 was forced,
  which accidentally neutralized memorization-minted L1; r_lo=0.3 made it live.
  With both fixes the channel couldn't be targeted; baseline inflation at the band
  edge minted at most +0.015 < δ=0.02 (simulated at king parity). Under Reason v3
  the whole lpA/L1 channel is unscored, so this interaction is moot — the two code
  fixes above are the load-bearing part.

**Mitigations (now enforced in code, not a new score gate):**
1. Fresh teacher `y_C` sampled per duel (`RefCache` scoped to one duel)
2. Reveal-block-hash slice seeding incl. strata order (miner can’t precompute D_t)
3. Corpus refresh via manifest `corpus_epoch` increment — a **data event, not a
   fork** (per the toml comment + published llms.txt; `weight_version_key` is
   reserved for scoring-rule changes). First refresh: epoch 2, 2026-08-07, +327
   SWE-verified datagen turns (`turns_epoch_0002.jsonl.gz`); verdicts stamp the
   manifest they were scored against.

Private 50/50 holdout pool: **REJECTED** (breaks external replayability).

Artifacts: `research/results/rt6_temporal_holdout.{json,txt}`,
`research/scripts/rt6_temporal_holdout.py`.

---

## 6. D_tau2 — three negative probes

Attempted to show S_tau2 ≅ tau2 and ⊥ swe. Same coding-king panel.

| probe | S vs tau2 | notes |
|---|---|---|
| bash-remap of tool calls | **−0.881** | short-style wins |
| native tool contract (`action_kind=tool`) | **−0.738** | gate 1–13%; kings don’t emit valid tools |
| force-only (`--force-y-from-ref`) | **−0.257** | gate 0–3%; S still ~swe (+0.714) |

**Do not claim demonstrated programmability.** Closing it needs tool-capable miners and/or a
tau2-strong teacher C — not another remap of coding kings.

Code: `research/harness/runner.py --force-y-from-ref`, `research/harness/chat.py`
`action_kind` contextvar. Data: `research/data/turns_tau2_native.jsonl`.
Results: `research/results/tau2_prog_force_table.txt`.

---

## 7. Repo layout (uv workspace)

```
.
  AGENTS.md           this file
  START_HERE.txt      handoff pointer (read first)
  LAYOUT.txt          short directory index
  pyproject.toml      uv workspace root (members: affine, research, ops)
  uv.lock             single lockfile — commit it
  setup.sh            uv sync --all-packages → one root .venv; writes .env
  .env                PYTHONPATH=research/ (gitignored; no secrets required for layout)
  affine/             SN120 validator + evalsrv + datagen + website + affine.toml (editable)
  research/           deps-only member (harness, scripts, data, results, docs, chart, e1)
  ops/                deps-only member — burn/legacy weight helpers + datagen refresh
  mining/             operator's own SN120 mining effort (GOAL/LESSONS/experiments/wallets;
                      driven by ralph loops — not a workspace member)
  ralphs/             cursor-agent loop runners (ralph.sh / ralphctl.sh + per-loop prompts:
                      keepalive, discord, bench-sentinel, king-analysis, …)
```

Imports assume `PYTHONPATH=<repo>/research` (set by `.env`; `harness/` and `scripts/` are
imported from there, not installed). Research scripts run from `research/` (relative
`results/`, `data/` paths).

Production corpus `research/data/turns_minicoder.jsonl` is **gitignored** (GitHub 100M
cap). Canonical copy lives on the public Hippius bucket (`s3.hippius.com/affine-sn120`):
schema_version **2** uses trajectory chunks (`turns/chunks/`) + a Parquet turn index
(`turns/index/`) behind an immutable manifest (see toml `[dataset]`); evalsrv samples
the index and materializes prefixes on demand. Legacy per-turn shards remain for
historical replay. The datagen loop stages raw turn-flat shards on HF
(`unconst/affine-datagen-turns`, private) until a corpus refresh packs and folds them
in. Headline freeze tables under `research/results/` are committed; bulky intermediate
pair dumps are not.

---

## 8. How scoring runs (mental model)

1. Turn prefix x from D.
2. Teacher C samples reference rollouts (z_C, y_C); cached per turn in ref jsonl.
3. Miner A samples (z_A, y_A) [or force-y pins y to gold].
4. Teacher-force echo+logprobs for the component lp\* fields (stored in pair records).
5. Offline or online: mean Reason per side → paired kσ duel (telemetry recorded
   alongside; pre-fork replays use the legacy gates→mix→δ path).

Key modules:
- `research/harness/terms.py` — Δ terms + pair components
- `research/harness/score.py` — Reason v3 + duel (legacy v2 kept for replay)
- `research/harness/runner.py` — E-KINGS batch scorer
- `affine/evalsrv/dueling.py` — live duel (slice seed, probe, score, verdict)
- `affine/evalsrv/engine.py` — vLLM slot lifecycle (teacher/king/challenger)

Kings HF pattern: `dendriteholdings/albedo-qwen3.6-35b-king-{ROMAN|genesis}`.
Bench map: `research/harness/config.py` `KING_BENCH` (swe-rebench scores).

---

## 9. Operator notes (eval pods)

- Prefer **tar + direct SSH/SCP** to upload `affine/` onto GPU pods; some cloud rsync paths
  omit `*.py`.
- Prefer direct SSH over sticky control-plane exec/scp wrappers that can hang; if a machine
  won’t answer, tear it down and rent another.
- Host:port reuse across pod swaps ⇒ SSH host-key churn; tunnels should use
  `StrictHostKeyChecking=accept-new` (and a disposable known_hosts file if needed).
- Don’t `pkill -f` patterns that match the SSH command line (kills your session).
- Don’t run two teacher-heavy jobs concurrently on one pod.
- Chain scripts: wait on **result line counts**, not “process absent”.
- Validator/CPU host needs no GPU; rent GPUs only for evalsrv / scoring.

---

## 10. What to do next

### Launch
1. Rent GPUs, tar-upload `affine/`, write pod `.eval_env` (`HF_TOKEN`, `AFFINE_EVAL_TOKEN`).
2. Bootstrap → `/health` ok=true → full **n=80** genesis-vs-challenger burn-in.
3. Bump `min_submission_block` to current finney tip at go-live.
4. Mirror corpus to an AffineFoundation HF dataset when org write exists; retarget toml.

### R2 cutover go-live (2026-09-03, order matters)
1. Cloudflare account API-token quota is 500; each `activate` creates one
   token (deleted at `ready`, GC'd every 30 min). 2026-09-02: 480 unused
   `arbos-r2-s3-<hex>` leftovers (2026-03-25/26) were deleted on operator
   directive → 16/500 in use. If creation ever hits the quota, activations
   defer (not burn) but nobody can upload — check `list_tokens` first.
2. `affine/affine.toml`: `[submission.r2] enabled = true`,
   `hf_cutover_block = <finney tip>`. No `weight_version_key` change.
3. `pm2 restart affine-validator` — reads `~/.affine-validator.env` (R2 keys
   synced there 2026-09-02; fail-closed `SystemExit` if any is missing).
4. `cd affine && python scripts/redeploy_pods.py --all` — live pods get the
   r2store code + `AFFINE_EVAL_R2_*` without a re-rent (kills the in-flight
   duel; it requeues as infra). Until this runs, an R2 challenger fails with
   `FetchError` → infra requeue, never a burn.
5. `python scripts/build_llms_txt.py` (already run) — then watch
   `api/v1/contract` shows `submission_r2.validator_identity`, and the first
   `activate` produces a 200 on `https://dash.affine.io/mailbox/v1/<id>/…`.
6. Remove the `#r2-notice` banner text "tomorrow" wording once live.

### Research / paper
1. Public claim = **distillation meter** (teacher-anchored Reason) + teacher
   robustness; **not** coding isomorphism (RT-7 open) and **not** D_tau2
   programmability.
2. Optional: tool-capable miner panel or tau2-strong teacher for a real D_tau2 test.
3. Keep refreshing D as the RT-6 residual defense; watch bank telemetry for
   adaptive paraphrase priors (the residual channel under gateless Reason).

---

## 11. Key artifact index

| Path | What |
|---|---|
| `research/docs/MOTIVATION.md` | Thesis / fixed point of the program |
| `research/docs/REDTEAM.md` | Attack table + statuses |
| `research/docs/EQUILIBRIUM.md` | Asymptote alignment argument + assumption ledger |
| `research/docs/PAPER_DRAFT.md` | Draft paper text |
| `research/results/hybrid_w5_table.txt` / `_meta.json` | n=30 freeze |
| `research/results/hybrid_sstar_v2_*` | S\* v2 re-freeze |
| `research/results/rt6_temporal_holdout.*` | RT-6b leave-repo |
| `research/results/tau2_prog_*.txt` | D_tau2 probe tables |
| `affine/affine.toml` | chain contract SSOT |

---

## 12. One-paragraph resume

> Affine SN120: teacher-anchored thought-injection duels. Since 2026-08-27
> (`weight_version_key=15` since the 2026-09-12 window-best crown fork —
> crown decided per 12 h window, best positive margin, confirmed on a
> fresh slice; the
> scoring rule itself dates from wvk 10) the contract is **min(R,G) v5: centered Reason
> + banded Grounding + δ floor + thought-length floor + B gate**: per turn
> the teacher samples k=3 refs, a_i = lpC(y_i|z_A) − lpC(y_i|∅);
> R = 0.03·log(mean_i exp(a_i/0.03)) − mean_i a_i (centering cancels any
> flat, ref-independent lift — the reign-41 filler-suffix exploit that won
> the v4 board); G checks the miner's thought's own teacher likelihood
> m = lpC(z_A|x) against the two-sided band mu ± max(2·sd, 0.002) built
> from the teacher's k reference-thought echoes t_i = lpC(z_C^i|x) (filler
> falls below the band, parroting lands above it); turn score = min(R, G);
> score = mean over 1300 turns, crown = paired mean > max(2·SE, 0.002)
> **and** median stripped `|z| ≥ 80` **and** B pass ≥ 0.30 — no lpA gates,
> no mix. B = lpC(y_A|z_A) − lpC(y_A|∅) is a license. δ kills ε-copies /
> SE-compression and winner's-curse churn. Positive control: held-out
> teacher thoughts beat base-model output at z=+2.56 under min(R,G) vs
> z=+0.11 under v4. The fork reset the throne: reign 0 = untouched
> `Qwen/Qwen3.6-35B-A3B` (unpaid genesis, emissions burn until a real
> crown); wvk-9 rows replay bit-identically via `score_mode="reason"`.
> Watch items: RT-9/RT-10 (mode-guessing, leakage amplification) carry
> over; RT-13 in-band mimicry is new. Everything v2 gated on plus
> lengths/timing and the new leg telemetry (mean_r_leg/mean_g_leg/
> g_bind_frac) is published on verdicts. Teacher C is
> `Qwen/Qwen3.8-27B` co-located on the eval box (swapped from
> GLM-4.5-Air-FP8 at the same fork; needs vLLM ≥ 0.28 and echo chunk 8192
> for its 248k vocab — the min(R,G) calibration numbers were measured under
> GLM and carry by operator decision, not re-measurement). Public claim is
> a distillation meter: RT-7 live-board inversion stays open — do not claim
> coding isomorphism. Second teacher +0.943; D_tau2 programmability not
> demonstrated. Corpus sha-pinned; uv-workspace monorepo
> (`affine` + `research` + `ops`).

---

_Update this file when a decision changes the frozen contract or the paper’s claim boundary._
