"""Frozen production scoring rule: Reason (v3, 2026-08-10).

Shared between root validator and eval server. Any change here is a chain fork
(bump [subnet].weight_version_key).

Reason(A; C, D):
  Per pair:   Reason = lpC(y_C | z_A) − lpC(y_C | ∅)
  Per miner:  score  = mean(Reason) over all pairs
  Duel:       challenger wins iff
              paired mean(Reason_c − Reason_k) > max(k_sigma · SE, min_margin)
              AND median(len(z_A.strip())) ≥ min_thought_chars
              AND (if causality_gamma > 0) B pass rate ≥ causality_gamma
              with SE = stdev(diffs) / sqrt(n) over paired turns.

Scoring hyperparameters: n_turns, k_sigma, min_margin (δ),
min_thought_chars, and the B license (causality_gamma). There is no mix,
no clip, and no lpA gates. The length floor (2026-08-13, weight_version_key=5)
evicts empty / cue-thought kings. Teacher-side B (2026-08-13,
weight_version_key=6) is the license to play: thoughts must cause the
miner's own action as judged by the teacher. Live dueling.py passes
causality_gamma from toml when causality_gate is on.

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
    """Reason = lpC(y_C|z_A) − lpC(y_C|∅). The score."""
    return pair["lpC_yc_za"] - pair["lpC_yc_e"]


# Historical name (Λ2) kept for research scripts and old artifact replay.
lambda2 = reason


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
    reason: float                     # the score: mean per-pair Reason
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
                bank_frac: float | None = None) -> MinerScore:
    """Score one miner: mean Reason + telemetry. No gating of any kind."""
    if not rows:
        return MinerScore("?", float("-inf"), 0, 0)
    pairs = [p for r in rows if r.get("valid") and "pairs" in r for p in r["pairs"]]
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
        reason=st.mean(reason(p) for p in pairs),
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


def duel(challenger_rows: list[dict], king_rows: list[dict],
         k_sigma: float = DEFAULT_K_SIGMA,
         min_margin: float = DEFAULT_MIN_MARGIN,
         min_thought_chars: int = DEFAULT_MIN_THOUGHT_CHARS,
         causality_gamma: float = DEFAULT_CAUSALITY_GAMMA,
         challenger_bank_frac: float | None = None,
         king_bank_frac: float | None = None) -> DuelResult:
    """Paired duel on per-turn mean Reason: wins iff
    mean > max(k_sigma·SE, min_margin) AND the challenger's median stripped
    thought length is ≥ min_thought_chars AND (if causality_gamma > 0) the
    challenger's teacher-side B pass rate is ≥ causality_gamma.

    The δ floor stops ε-copies / SE-compression; the length floor evicts
    empty/cue thoughts (A9). B is the starting-line license: thoughts must
    cause the miner's own action, as judged by the teacher. Bank fracs are
    accepted only to thread telemetry. min_thought_chars ≤ 0 disables the
    length floor; causality_gamma ≤ 0 disables B (pre-fork replay / try)."""
    cs = score_miner(challenger_rows, challenger_bank_frac)
    ks = score_miner(king_rows, king_bank_frac)
    c_by = {r["turn_id"]: r for r in challenger_rows if r.get("valid") and "pairs" in r}
    k_by = {r["turn_id"]: r for r in king_rows if r.get("valid") and "pairs" in r}
    diffs = []
    for tid in sorted(set(c_by) & set(k_by)):
        rc = st.mean(reason(p) for p in c_by[tid]["pairs"])
        rk = st.mean(reason(p) for p in k_by[tid]["pairs"])
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
    )
