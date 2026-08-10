# Affine / SN120 — project context

Affine is a Bittensor subnet (netuid **120**) that crowns miners by a **teacher-anchored
distillation score S\***, not an LLM judge. This monorepo is the public command center:
validator + evalsrv (`affine/`), research harness and freeze artifacts (`research/`), and
lightweight ops helpers (`ops/`).

Thesis: `research/docs/MOTIVATION.md` · Red-team: `research/docs/REDTEAM.md` ·
Paper draft: `research/docs/PAPER_DRAFT.md` · Contract SSOT: `affine/affine.toml`

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

## 2. Frozen production scoring — S\* v2

Implemented in `research/harness/score.py` and ported to `affine/affine/score.py`. Contract
knobs in `affine/affine.toml` `[duel]` + `[dataset]`.

### Per-pair / miner gates
1. **Causality + leakage** — pair passes if no fuzzy z⊃y leakage and
   `lpA(y_A|z_A) − lpA(y_A|∅) ≥ τ` (τ=0.02). Miner INVALID if pass_rate < γ=0.30.
2. **Prior-bank positivity** — `frac_bank` = share of pairs with Λ2_bank > 0 over published
   priors. INVALID if frac_bank < γ_bank=0.08. Closes paraphrase stuffing (RT-2c).
3. **Calibration ratio** — `r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|`. INVALID if
 r ∉ [0.3, 4.0]. r_lo was 1.0 at launch; lowered 2026-08-06 — r < 1 is exactly mean
 L1lift > 0, the natural signature of a faithful distill (live distills at r 0.72–0.81;
 teacher-self ≈ 0.35). r_lo=1.0 was invalidating every genuine winner.
3b. **Empty-baseline band** (challenger only, paired, added 2026-08-06) — challenger
 mean|lpA(y_C|∅)| ≤ 1.25× king's on the same slice (honest max observed 1.14×). Closes
 RT-3d free-L1lift via baseline sabotage, which r_lo=1.0 used to cover.

### Ranking term
```
S = mean( Λ2 + w · clip(L1lift, ±0.1) )   with w = 1.0
Λ2     = lpC(y_C|z_A) − lpC(y_C|∅)
L1lift = lpA(y_C|z_A) − lpA(y_C|∅)
```

### Duel crowning rule
Challenger wins iff **all** of:
- paired mean(S_c − S_k) > 3 · SE
- mean margin > δ = **0.02** (`min_margin`, updated 2026-08-05)
- SE floored by `min_se = 0.005`
- both sides gate-valid

δ is a **noise floor, not an effect floor** (policy 2026-08-05): it covers the RT-4
copy null (3·SE≈0.0195), the measured lm_head-sharpening residual (≤ +0.012), and the
min_se degeneracy (3·min_se=0.015). Any challenger statistically above the king crowns.
The former δ=0.05 effect floor (A11 defense) was dropped by explicit decision: same-tier
short-style winners (king-II persistent margin +0.034) are accepted as kings — S is the
metric.

### Headline correlations (coding D)
| set | Spearman(S, swe) | notes |
|---|---|---|
| ungated @ n=15 | +0.844 | early freeze |
| ungated @ n=19 | +0.856 / +0.862 under clip | |
| ungated @ n=30 | **+0.758** (p≈1.2e-6) | wave-5; XC soft outlier |
| hybrid @ n=15 | +0.799 | many mid kings at γ_bank knife-edge ~0.075 |
| LLM judge (same turns) | ~+0.31 | S ≫ judge; prompt variants checked |

Second teacher (Qwen3-32B vs GLM-Air), n=6 kings: Spearman(S_T1, S_T2) = **+0.943**.

---

## 3. Red-team status (load-bearing)

| ID | Attack | Status | Defense |
|---|---|---|---|
| RT-1 / A1 | fixed thought payloads | CLOSED | lose to genesis; empty → causality |
| RT-2 / A2 | action stuffing into z | CLOSED | leakage gate |
| RT-2 / A9 | silent miner | CLOSED | causality gate |
| RT-2c / A2c | paraphrase stuffing | CLOSED | bank gate |
| RT-soft-pad / A10 | soft-idents pad | CLOSED | abandoned; use mix w=1 |
| RT-4 / A4 | king copy | CLOSED | 3σ null |
| RT-3 / A3 | L1lift / overconfidence | CLOSED live | clip0.1 + r∈[0.3,4] + baseline band 1.25×; sharpening residual ≤ +0.012 < 3·SE floor |
| A11 | short-style I/II FP | **ACCEPTED (policy 2026-08-05)** | δ→0.02 noise floor; same-tier S winners may crown |
| RT-6 / A6 | dataset sniping | **CLOSED (code, 2026-08-06)** | seed-shuffled strata + per-duel fresh y_C — see §5 |
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

The gates are validity checks (causality, leakage, bank, r, band); none asks
whether the winner can write code. A model can be gate-valid, crown, resolve 0/25.

**Bench repeatability:** genesis scored 0/25 then 5/25 on the *same revision*, so
single 25-task scores are not per-model evidence — hence the pooled binomial test.
Outcome noise attenuates Spearman toward zero, so −0.42 is a conservative floor.

**Do not claim coding isomorphism without this caveat.** Artifacts:
`research/results/rt7_live_isomorphism.{json,txt}`, `research/scripts/rt7_live_isomorphism.py`.

Full writeups: `research/docs/REDTEAM.md`.

---

## 4. Contract snapshot (`affine/affine.toml`)

- netuid **120**, finney
- official site: **https://affine.io** (dashboard + llms.txt; Cloudflare-proxied
  to the validator box — sn120.arbos.life is a legacy alias via the CF tunnel)
- `weight_version_key = 2` (bumped 2026-08-10 on teacher C swap Air→GLM-5.2-FP8;
  earlier min_margin 0.05→0.02 on 2026-08-05 shipped WITHOUT a version bump)
- teacher: `zai-org/GLM-5.2-FP8` (dedicated `affine-teacher` B300 box; evalsrv
  remote `base_url`)
- seed king: `dendriteholdings/albedo-qwen3.6-35b-king-genesis`
- turns: sharded corpus with immutable manifest; sha-pinned (see toml `[dataset]`)
- duel: n_turns=80, clip=0.1, r∈[0.3,4], baseline_band=1.25, δ=0.02, k_sigma=3

### Evalsrv roles
- `AFFINE_ROLE=duel` — teacher + king + challenger; S\* only
- `AFFINE_ROLE=bench` — SWE advisory (never part of S\*)
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
- Interaction with r_lo: at r_lo=1.0 mean L1lift ≤ 0 was forced, which accidentally
  neutralized memorization-minted L1; r_lo=0.3 made it live. With both fixes the
  channel can't be targeted; pure baseline inflation at the band edge mints at most
  +0.015 < δ=0.02 (simulated at king parity), so r_lo=0.3 + band 1.25 stand.

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
  LAYOUT.txt          short directory index
  pyproject.toml      uv workspace root (members: affine, research, ops)
  uv.lock             single lockfile — commit it
  setup.sh            uv sync --all-packages → one root .venv; writes .env
  .env                PYTHONPATH=research/ (gitignored; no secrets required for layout)
  affine/             SN120 validator + evalsrv + website + affine.toml (installed editable)
  research/           deps-only member (harness, scripts, data, results, docs, chart, e1)
  ops/                deps-only member — weight helpers; discord/twitter planned
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
5. Offline or online: gates → mix S → duel paired test.

Key modules:
- `research/harness/terms.py` — Δ terms + pair components
- `research/harness/score.py` — production S\* + duel
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

### Research / paper
1. Claim coding isomorphism + teacher robustness + closed RT suite; **not** D_tau2
   programmability.
2. Optional: tool-capable miner panel or tau2-strong teacher for a real D_tau2 test.
3. Keep refreshing D as the RT-6 residual defense.

---

## 11. Key artifact index

| Path | What |
|---|---|
| `research/docs/MOTIVATION.md` | Thesis / fixed point of the program |
| `research/docs/REDTEAM.md` | Attack table + statuses |
| `research/docs/PAPER_DRAFT.md` | Draft paper text |
| `research/results/hybrid_w5_table.txt` / `_meta.json` | n=30 freeze |
| `research/results/hybrid_sstar_v2_*` | S\* v2 re-freeze |
| `research/results/rt6_temporal_holdout.*` | RT-6b leave-repo |
| `research/results/tau2_prog_*.txt` | D_tau2 probe tables |
| `affine/affine.toml` | chain contract SSOT |

---

## 12. One-paragraph resume

> Affine SN120: teacher-anchored thought-injection duels (S\* v2 = clip0.1 mix +
> causality/leakage + bank + r∈[0.3,4] + 1.25× baseline band + 3σ∧δ=0.02 noise floor).
> Teacher C is `zai-org/GLM-5.2-FP8` on a dedicated B300 box (`weight_version_key=2`,
> 2026-08-10). Reigns 1–2 (pandora-box ckpt300-m4, kevin954 sft) were crowned
> retroactively on 2026-08-06 from their published genesis duels when r_lo 1.0→0.3
> shipped (operator decision, no re-eval; margins +0.061/z=5.7 and +0.070/z=6.3 vs
> genesis). Coding isomorphism holds at +0.758@30 ungated on the Albedo panel;
> live-board RT-7 inversion open. Second teacher +0.943; RT suite closed/mitigated
> except D_tau2 programmability (three negative probes). Corpus sha-pinned;
> uv-workspace monorepo (`affine` + `research` + `ops`).

---

_Update this file when a decision changes the frozen contract or the paper’s claim boundary._
