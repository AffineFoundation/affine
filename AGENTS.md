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
S never touches benchmark tasks. D is SWE-style coding trajectories ⇒ target axis is coding.
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
   r ∉ [1.0, 4.0]. Honest live band ≈ [1.07, 3.56] / measured live ≈ [1.14, 1.42].

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
| RT-3 / A3 | L1lift / overconfidence | CLOSED live | clip0.1 + r∈[1,4]; sharpening residual ≤ +0.012 < 3·SE floor |
| A11 | short-style I/II FP | **ACCEPTED (policy 2026-08-05)** | δ→0.02 noise floor; same-tier S winners may crown |
| RT-6 / A6 | dataset sniping | **MITIGATED (ops)** | not a score gate — see §5 |
| D_tau2 | programmability falsifier | **NOT demonstrated** | see §6 |

Full writeups: `research/docs/REDTEAM.md`.

---

## 4. Contract snapshot (`affine/affine.toml`)

- netuid **120**, finney
- official site: **https://affine.io** (dashboard + llms.txt; Cloudflare-proxied
  to the validator box — sn120.arbos.life is a legacy alias via the CF tunnel)
- `weight_version_key = 1` (min_margin 0.05 → 0.02 on 2026-08-05 shipped WITHOUT a
  version bump — operator decision)
- teacher: `zai-org/GLM-4.5-Air-FP8`
- seed king: `dendriteholdings/albedo-qwen3.6-35b-king-genesis`
- turns: sharded corpus with immutable manifest; sha-pinned (see toml `[dataset]`)
- duel: n_turns=80, clip=0.1, r∈[1,4], δ=0.02, k_sigma=3

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

**Mitigations (ops, not a new score gate):**
1. Fresh teacher `y_C` sampled at duel time (not a frozen published target)
2. Reveal-block-hash slice seeding (miner can’t precompute D_t)
3. Corpus refresh with `corpus_epoch` + `weight_version_key` bump

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
cap). Canonical copy is on Hugging Face, sha-pinned in `affine/affine.toml`. Headline
freeze tables under `research/results/` are committed; bulky intermediate pair dumps are not.

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
> causality/leakage + bank + r-gate + 3σ∧δ=0.02 noise floor). Coding isomorphism holds at +0.758@30
> ungated; second teacher +0.943; RT suite closed/mitigated except D_tau2 programmability
> (three negative probes). Corpus sha-pinned on HF; uv-workspace monorepo
> (`affine` + `research` + `ops`). Next: production n=80 burn-in and go-live ops, or paper
> writeup without claiming D_tau2.

---

_Update this file when a decision changes the frozen contract or the paper’s claim boundary._
