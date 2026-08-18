"""Frozen production scoring rule: Reason v4 — tempered multi-sample
(2026-08-17, weight_version_key=7).

Shared between root validator and eval server. Any change here is a chain fork
(bump [subnet].weight_version_key).

Reason(A; C, D):
  Per ref i:  a_i  = lpC(y_i | z_A) − lpC(y_i | ∅)     (per-byte, k refs/turn)
  Per turn:   Reason = tau · log( (1/k) · Σ_i exp(a_i / tau) )
  Per miner:  score  = mean(turn Reason) over turns
  Duel:       challenger wins iff
              paired mean(Reason_c − Reason_k) > max(k_sigma · SE, min_margin)
              AND median(len(z_A.strip())) ≥ min_thought_chars
              AND (if causality_gamma > 0) B pass rate ≥ causality_gamma
              with SE = stdev(diffs) / sqrt(n) over paired turns.

Scoring hyperparameters: n_turns, k_sigma, min_margin (δ), tau,
n_teacher_samples (k), min_thought_chars, and the B license
(causality_gamma). There is no mix, no clip, and no lpA gates.

Why tempered (v4, 2026-08-17): the teacher's action distribution is
multi-modal — resampling a turn yields different, equally valid actions.
Averaging per-ref Reason in log space punishes a miss without bound, so a
thought that commits to one valid mode has negative expected score whenever
the reference lands on another mode (measured: the teacher's own thought,
unpaired from its rollout, scores ≈ −0.010/byte, n=509). The equilibrium
under that rule is non-committal filler. The tempered log-mean-exp is
dominated by the best-matched reference instead of the worst: a missed mode
zeroes its own share but cannot drag the turn below the credit from a hit,
so committing to the teacher's dominant next action becomes the optimum.
k=1 reduces to the v3 per-pair Reason exactly (any tau); tau→∞ recovers the
plain mean. tau = 0.03 calibrated externally (AIIan, n=100 turns): the flip
from hedge-wins to commit-wins happens near tau=0.1 and is decisive by 0.03,
while cold tau amplifies mode-guessing/leakage — both monitored via
telemetry. min_margin = 0.002 is provisional at the new score scale and is
recalibrated from the first ~20 live v4 duels.

The length floor (2026-08-13, weight_version_key=5) evicts empty /
cue-thought kings. Teacher-side B (2026-08-13, weight_version_key=6) is the
license to play: thoughts must cause the miner's own action as judged by the
teacher. Live dueling.py passes causality_gamma from toml when
causality_gate is on.

δ (min_margin = 0.002, added 2026-08-12, weight_version_key=4) exists for one
reason: the z-test is relative to the challenger's own noise, so an ε-copy of
the king (per-duel SE ≈ 0.0003, ~6× below a distinct model's) crowns at the
same 1-in-44 as anyone else on ±0.0006 noise margins. The absolute floor makes
that ~z=6.7 (≈1-in-7.6e10) while staying below the live 2·SE bar (~0.0035),
so honest duels never touch it. It also caps SE-compression strategies: the
crown bar never drops below δ no matter how consistent a challenger's thoughts
are. Calibration: research/results/delta_calibration.{json,txt}.

Reason (formerly Λ2) is computed entirely on the teacher side: it asks how much
the miner's thought z_A raises the frozen teacher's likelihood of reproducing
its own action y_C. The miner's weights never touch the ranked quantity, which
retires the whole lpA attack surface (RT-3 family: lm_head sharpening,
water-filling, empty-baseline sabotage).

Everything the retired S* v2 gates measured is still computed and published as
TELEMETRY — recorded for study and monitoring, never affecting score or
validity:
  - causality/leakage pass rate  (τ/fuzzy are telemetry constants, not
    consensus knobs)
  - prior-bank positivity frac   (bank_frac; watched for adaptive paraphrase
    priors, which tie genesis on raw Reason but must still beat the sitting
    king at k_sigma·SE)
  - calibration ratio r and empty-baseline magnitude (lpA channel diagnostics)
  - raw L1lift mean (unclipped — safe to publish now that it is not scored)
  - η (eta): sufficiency fraction Λ2(z_A)/Λ2(z_C) — how much of the teacher's
    own thinking the miner's thought replaces (needs lpC(y_C|z_C) from refs)
  - thought/action character lengths (miner-vs-teacher length deltas are
    assembled in evalsrv where teacher refs are in scope)

History: S* v2 (mix + 5 gates + δ floor) was retired 2026-08-10. Raw Λ2
correlates with swe-rebench as well as the mix did on the Albedo panel
(+0.847@15 vs +0.844); the L1 term and its defensive gates were complexity
without signal. The A11 short-style objection to Λ2-only ranking was already
policy-dead (2026-08-05: same-tier S winners may crown). Pre-fork verdicts
stamp the old formula and remain replayable under their stored `gates` block.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from dataclasses import dataclass


DEFAULT_K_SIGMA = 2.0
DEFAULT_MIN_MARGIN = 0.002
DEFAULT_MIN_THOUGHT_CHARS = 80
# Tempering temperature for the per-turn log-mean-exp over k references
# (v4, 2026-08-17). tau <= 0 or None means plain mean (v3 replay).
DEFAULT_TEMPER_TAU = 0.03
# Teacher-side causality gate B (off unless causality_gamma > 0).
# B = lpC(y_A|z_A) − lpC(y_A|∅). Same τ/γ as retired v2 miner-side A9.
DEFAULT_CAUSALITY_TAU = 0.02
DEFAULT_CAUSALITY_GAMMA = 0.0

# Telemetry constants (non-consensus): thresholds used only to report the
# legacy causality/leakage pass rate. Changing them is NOT a chain fork.
TELEMETRY_TAU = 0.02
TELEMETRY_FUZZY = 0.6


def _cmd(y: str) -> str:
    """Normalize action text for leakage telemetry (bash fence or tool JSON)."""
    y = y.strip()
    if y.startswith("```bash\n") and y.endswith("\n```"):
        return y.removeprefix("```bash\n").removesuffix("\n```").strip()
    return y


def leakage(z: str, y: str, fuzzy: float = TELEMETRY_FUZZY) -> bool:
    """Telemetry: fuzzy z⊃y containment (legacy gate 1 component)."""
    c = _cmd(y)
    if not c:
        return False
    if c in z:
        return True
    if c.startswith("{") and '"name"' in c:
        try:
            name = json.loads(c).get("name") or ""
        except json.JSONDecodeError:
            name = ""
        return bool(name) and name in z
    toks = [t for t in re.split(r"\s+", c) if len(t) >= 3]
    if not toks:
        return False
    return sum(1 for t in toks if t in z) / len(toks) >= fuzzy


def gate_pass(pair: dict, tau: float = TELEMETRY_TAU,
              fuzzy: float = TELEMETRY_FUZZY) -> bool | None:
    """Telemetry: legacy miner-side causality+leakage pass (not scored).

    None when the pair was scored Reason-only (lpA echoes omitted).
    """
    try:
        ya_za, ya_e = pair["lpA_ya_za"], pair["lpA_ya_e"]
    except KeyError:
        return None
    if leakage(pair.get("z_a", ""), pair.get("y_a", ""), fuzzy=fuzzy):
        return False
    return (ya_za - ya_e) >= tau


def teacher_causality(pair: dict) -> float | None:
    """B = lpC(y_A|z_A) − lpC(y_A|∅). Teacher: did z cause the miner's own y?

    None when those echoes were omitted (live reason_only without the B pair).
    """
    try:
        return pair["lpC_ya_za"] - pair["lpC_ya_e"]
    except (KeyError, TypeError):
        return None


def b_gate_pass(pair: dict, tau: float = DEFAULT_CAUSALITY_TAU,
                fuzzy: float = TELEMETRY_FUZZY) -> bool | None:
    """Teacher-side causality+leakage pass (design B).

    None when B echoes are missing. False on leakage or B < tau.
    """
    b = teacher_causality(pair)
    if b is None:
        return None
    if leakage(pair.get("z_a", "") or "", pair.get("y_a", "") or "",
               fuzzy=fuzzy):
        return False
    return b >= tau


def reason(pair: dict) -> float:
    """Per-reference Reason a_i = lpC(y_i|z_A) − lpC(y_i|∅) (per byte)."""
    return pair["lpC_yc_za"] - pair["lpC_yc_e"]


# Historical name (Λ2) kept for research scripts and old artifact replay.
lambda2 = reason


def turn_reason(pairs: list[dict],
                tau: float | None = DEFAULT_TEMPER_TAU) -> float:
    """Turn score: tempered log-mean-exp of per-ref Reason (v4, 2026-08-17).

        turn = tau · log( (1/k) · Σ_i exp(a_i / tau) )

    Dominated by the best-matched reference, so a thought that commits to one
    valid teacher mode is not dragged negative by references that landed on
    another mode. Exact identities: k=1 returns a_1 for any tau; tau <= 0 or
    None returns the plain mean (v3 replay rule). Computed max-shifted for
    numerical stability.
    """
    a = [reason(p) for p in pairs]
    if not a:
        return float("-inf")
    if tau is None or tau <= 0:
        return st.mean(a)
    if len(a) == 1:
        return a[0]
    m = max(a)
    return m + tau * math.log(
        st.mean(math.exp((ai - m) / tau) for ai in a))


def l1_lift(pair: dict) -> float | None:
    """Telemetry: miner-side lift lpA(y_C|z_A) − lpA(y_C|∅) (not scored).

    None when the pair was scored Reason-only (lpA echoes omitted).
    """
    try:
        return pair["lpA_yc_za"] - pair["lpA_yc_e"]
    except KeyError:
        return None


# Floor under |Λ2(z_C)| below which η is undefined (teacher own-lift ~0).
ETA_DENOM_EPS = 1e-9


def eta(pair: dict) -> float | None:
    """Telemetry: η = Λ2(z_A) / Λ2(z_C) = Reason / (lpC(y_C|z_C) − lpC(y_C|∅)).

    How much of the teacher's own thinking the miner's thought replaces on
    this pair. Denominator comes from the teacher reference (`lpC_yc_zc` /
    `lp_own`); no extra GPU echo is required. Undefined when |Λ2(z_C)| is
    below ETA_DENOM_EPS. Not scored.
    """
    try:
        num = reason(pair)
        den = pair["lpC_yc_zc"] - pair["lpC_yc_e"]
    except (KeyError, TypeError):
        return None
    if not (math.isfinite(num) and math.isfinite(den)):
        return None
    if abs(den) < ETA_DENOM_EPS:
        return None
    v = num / den
    return v if math.isfinite(v) else None


def mean_eta(pairs: list[dict]) -> float | None:
    """Mean η over pairs where the ratio is defined."""
    vals = [e for p in pairs if (e := eta(p)) is not None]
    return st.mean(vals) if vals else None


def calibration_ratio(pairs: list[dict]) -> float | None:
    """Telemetry: r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)| (not scored)."""
    if not pairs:
        return None
    try:
        num = st.mean(abs(p["lpA_yc_za"]) for p in pairs)
        den = st.mean(abs(p["lpA_yc_e"]) for p in pairs)
    except KeyError:
        return None
    if den <= 0:
        return None
    return num / den


def _mean_optional(vals: list[float | None]) -> float | None:
    have = [v for v in vals if v is not None and math.isfinite(v)]
    return st.mean(have) if have else None


@dataclass
class MinerScore:
    miner: str
    reason: float                     # the score: mean per-turn tempered Reason
    n_pairs: int
    n_turns: int
    # -- telemetry (measured, never scored) --
    gate_pass_rate: float = 0.0
    bank_frac: float | None = None
    calib_ratio: float | None = None
    baseline_abs: float | None = None  # mean|lpA(y_C|∅)|
    mean_l1lift: float | None = None
    mean_eta: float | None = None      # sufficiency: mean Λ2(z_A)/Λ2(z_C)
    mean_len_z: float | None = None    # chars of z_A
    median_len_z: float | None = None  # median stripped chars of z_A
    mean_len_y: float | None = None    # chars of y_A
    mean_b: float | None = None        # mean teacher-side B (not scored)
    b_gate_pass_rate: float | None = None  # share of pairs passing B+leakage


def score_miner(rows: list[dict],
                bank_frac: float | None = None,
                tau: float | None = DEFAULT_TEMPER_TAU) -> MinerScore:
    """Score one miner: mean per-turn tempered Reason + telemetry.

    Each row is one turn holding k pairs (one per teacher reference); the
    turn score is turn_reason(pairs, tau) and the miner score is the mean
    over turns. With k=1 (or tau <= 0) this is identical to the v3
    mean-per-pair rule, so pre-fork rows replay unchanged. No gating.
    """
    if not rows:
        return MinerScore("?", float("-inf"), 0, 0)
    valid = [r for r in rows if r.get("valid") and "pairs" in r]
    pairs = [p for r in valid for p in r["pairs"]]
    if not pairs:
        return MinerScore(rows[0].get("miner", "?"), float("-inf"), 0, 0)
    gpass = [gate_pass(p) for p in pairs]
    gpass_f = [1.0 if g else 0.0 for g in gpass if g is not None]
    bflags = [b_gate_pass(p) for p in pairs]
    b_have = [1.0 if g else 0.0 for g in bflags if g is not None]
    try:
        baseline_abs = st.mean(abs(p["lpA_yc_e"]) for p in pairs)
    except KeyError:
        baseline_abs = None
    z_lens = [len((p.get("z_a") or "").strip()) for p in pairs]
    return MinerScore(
        miner=rows[0].get("miner", "?"),
        reason=st.mean(turn_reason(r["pairs"], tau) for r in valid),
        n_pairs=len(pairs),
        n_turns=len({r["turn_id"] for r in rows}),
        gate_pass_rate=(st.mean(gpass_f) if gpass_f else 0.0),
        bank_frac=bank_frac,
        calib_ratio=calibration_ratio(pairs),
        baseline_abs=baseline_abs,
        mean_l1lift=_mean_optional([l1_lift(p) for p in pairs]),
        mean_eta=mean_eta(pairs),
        mean_len_z=st.mean(float(n) for n in z_lens),
        median_len_z=float(st.median(z_lens)),
        mean_len_y=st.mean(float(len(p.get("y_a", ""))) for p in pairs),
        mean_b=_mean_optional([teacher_causality(p) for p in pairs]),
        b_gate_pass_rate=(st.mean(b_have) if b_have else None),
    )


@dataclass
class DuelResult:
    challenger: str
    king: str
    margin: float
    se: float
    z: float
    k_sigma: float
    challenger_wins: bool
    n_paired_turns: int
    min_margin: float = DEFAULT_MIN_MARGIN
    min_thought_chars: int = DEFAULT_MIN_THOUGHT_CHARS
    thought_floor_blocked: bool = False
    causality_gamma: float = DEFAULT_CAUSALITY_GAMMA
    causality_blocked: bool = False
    tau: float | None = DEFAULT_TEMPER_TAU


def duel(challenger_rows: list[dict], king_rows: list[dict],
         k_sigma: float = DEFAULT_K_SIGMA,
         min_margin: float = DEFAULT_MIN_MARGIN,
         min_thought_chars: int = DEFAULT_MIN_THOUGHT_CHARS,
         causality_gamma: float = DEFAULT_CAUSALITY_GAMMA,
         challenger_bank_frac: float | None = None,
         king_bank_frac: float | None = None,
         tau: float | None = DEFAULT_TEMPER_TAU) -> DuelResult:
    """Paired duel on per-turn tempered Reason: wins iff
    mean > max(k_sigma·SE, min_margin) AND the challenger's median stripped
    thought length is ≥ min_thought_chars AND (if causality_gamma > 0) the
    challenger's teacher-side B pass rate is ≥ causality_gamma.

    Each turn's score on each side is turn_reason(pairs, tau) — tempered
    log-mean-exp over the k teacher references (k=1 or tau <= 0 recovers the
    v3 plain mean for pre-fork replay). The δ floor stops ε-copies /
    SE-compression; the length floor evicts empty/cue thoughts (A9). B is
    the starting-line license: thoughts must cause the miner's own action,
    as judged by the teacher. Bank fracs are accepted only to thread
    telemetry. min_thought_chars ≤ 0 disables the length floor;
    causality_gamma ≤ 0 disables B (pre-fork replay / try)."""
    cs = score_miner(challenger_rows, challenger_bank_frac, tau=tau)
    ks = score_miner(king_rows, king_bank_frac, tau=tau)
    c_by = {r["turn_id"]: r for r in challenger_rows if r.get("valid") and "pairs" in r}
    k_by = {r["turn_id"]: r for r in king_rows if r.get("valid") and "pairs" in r}
    diffs = []
    for tid in sorted(set(c_by) & set(k_by)):
        rc = turn_reason(c_by[tid]["pairs"], tau)
        rk = turn_reason(k_by[tid]["pairs"], tau)
        diffs.append(rc - rk)
    n = len(diffs)
    if n < 2:
        return DuelResult(cs.miner, ks.miner, 0.0, float("inf"), 0.0,
                          k_sigma, False, n, min_margin, min_thought_chars,
                          False, causality_gamma)
    mean = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(n)
    z = mean / se if se > 0 else (math.inf if mean > 0 else 0.0)
    wins = mean > max(k_sigma * se, min_margin)
    blocked = False
    if min_thought_chars > 0 and (
            cs.median_len_z is None or cs.median_len_z < min_thought_chars):
        wins = False
        blocked = True
    causality_blocked = False
    if causality_gamma > 0:
        rate = cs.b_gate_pass_rate
        if rate is None or rate < causality_gamma:
            wins = False
            causality_blocked = True
    return DuelResult(
        challenger=cs.miner, king=ks.miner, margin=mean, se=se, z=z,
        k_sigma=k_sigma, challenger_wins=wins, n_paired_turns=n,
        min_margin=min_margin, min_thought_chars=min_thought_chars,
        thought_floor_blocked=blocked,
        causality_gamma=causality_gamma,
        causality_blocked=causality_blocked,
        tau=tau,
    )
