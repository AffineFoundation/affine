"""Scoring rule: Reason (v3, 2026-08-10) — research twin of affine/affine/score.py.

  Per pair:   Reason = lpC(y_C | z_A) − lpC(y_C | ∅)
  Per miner:  score  = mean(Reason)
  Duel:       challenger wins iff paired mean(Reason_c − Reason_k) > k_sigma · SE

No mix, no clip, no gates, no absolute margin floor. Everything the retired
S* v2 measured (causality, bank, calibration r, baseline, L1lift) is telemetry.

IMPORTANT — pre-fork freezes: every result frozen before 2026-08-10
(hybrid_w5, sstar_v2, RT tables) was produced under S* v2. To replay those,
use the LEGACY section at the bottom (`legacy_score_miner` / `legacy_duel`
and the DEFAULT_* v2 constants). `score_miner`/`duel` here are the v3 rule
and intentionally reject the old keyword arguments.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from dataclasses import dataclass


DEFAULT_K_SIGMA = 3.0

# Telemetry constants (non-consensus).
TELEMETRY_TAU = 0.02
TELEMETRY_FUZZY = 0.6


def _cmd(y: str) -> str:
    """Normalize action text for leakage telemetry (bash fence or tool JSON)."""
    y = y.strip()
    if y.startswith("```bash\n") and y.endswith("\n```"):
        return y.removeprefix("```bash\n").removesuffix("\n```").strip()
    return y


def leakage(z: str, y: str, fuzzy: float = TELEMETRY_FUZZY) -> bool:
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
              fuzzy: float = TELEMETRY_FUZZY) -> bool:
    """Telemetry: legacy causality+leakage pass (no longer affects score)."""
    if leakage(pair.get("z_a", ""), pair.get("y_a", ""), fuzzy=fuzzy):
        return False
    return (pair["lpA_ya_za"] - pair["lpA_ya_e"]) >= tau


def reason(pair: dict) -> float:
    """Reason = lpC(y_C|z_A) − lpC(y_C|∅). The score."""
    return pair["lpC_yc_za"] - pair["lpC_yc_e"]


lambda2 = reason  # historical name (Λ2)


def l1_lift(pair: dict) -> float:
    """Telemetry: lpA(y_C|z_A) − lpA(y_C|∅) (not scored)."""
    return pair["lpA_yc_za"] - pair["lpA_yc_e"]


def calibration_ratio(pairs: list[dict]) -> float | None:
    """Telemetry: r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)| (not scored)."""
    if not pairs:
        return None
    num = st.mean(abs(p["lpA_yc_za"]) for p in pairs)
    den = st.mean(abs(p["lpA_yc_e"]) for p in pairs)
    if den <= 0:
        return None
    return num / den


@dataclass
class MinerScore:
    miner: str
    reason: float
    n_pairs: int
    n_turns: int
    gate_pass_rate: float = 0.0
    bank_frac: float | None = None
    calib_ratio: float | None = None
    baseline_abs: float | None = None
    mean_l1lift: float | None = None
    mean_len_z: float | None = None
    mean_len_y: float | None = None


def score_miner(rows: list[dict],
                bank_frac: float | None = None) -> MinerScore:
    """v3: mean Reason + telemetry. No gating."""
    if not rows:
        return MinerScore("?", float("-inf"), 0, 0)
    pairs = [p for r in rows if r.get("valid") and "pairs" in r for p in r["pairs"]]
    if not pairs:
        return MinerScore(rows[0].get("miner", "?"), float("-inf"), 0, 0)
    return MinerScore(
        miner=rows[0].get("miner", "?"),
        reason=st.mean(reason(p) for p in pairs),
        n_pairs=len(pairs),
        n_turns=len({r["turn_id"] for r in rows}),
        gate_pass_rate=st.mean(1.0 if gate_pass(p) else 0.0 for p in pairs),
        bank_frac=bank_frac,
        calib_ratio=calibration_ratio(pairs),
        baseline_abs=st.mean(abs(p["lpA_yc_e"]) for p in pairs),
        mean_l1lift=st.mean(l1_lift(p) for p in pairs),
        mean_len_z=st.mean(float(len(p.get("z_a", ""))) for p in pairs),
        mean_len_y=st.mean(float(len(p.get("y_a", ""))) for p in pairs),
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


def duel(challenger_rows: list[dict], king_rows: list[dict],
         k_sigma: float = DEFAULT_K_SIGMA,
         challenger_bank_frac: float | None = None,
         king_bank_frac: float | None = None) -> DuelResult:
    """v3 paired duel on per-turn mean Reason: wins iff mean > k_sigma·SE."""
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
                          k_sigma, False, n)
    mean = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(n)
    z = mean / se if se > 0 else (math.inf if mean > 0 else 0.0)
    return DuelResult(
        challenger=cs.miner, king=ks.miner, margin=mean, se=se, z=z,
        k_sigma=k_sigma, challenger_wins=mean > k_sigma * se, n_paired_turns=n,
    )


# ---------------------------------------------------------------------------
# LEGACY (S* v2, retired 2026-08-10) — kept ONLY so pre-fork freeze scripts
# and REDTEAM replays keep working. Not the production rule. The constants
# reflect the final live v2 contract (r_lo=0.3, min_margin=0.02, band=1.25).
# ---------------------------------------------------------------------------

DEFAULT_TAU = 0.02
DEFAULT_GAMMA = 0.30
DEFAULT_GAMMA_BANK = 0.08
DEFAULT_FUZZY = 0.6
DEFAULT_L1_WEIGHT = 1.0
DEFAULT_L1_CLIP = 0.1
DEFAULT_R_LO = 0.3
DEFAULT_R_HI = 4.0
DEFAULT_BASELINE_BAND = 1.25
DEFAULT_MIN_MARGIN = 0.02
DEFAULT_MIN_SE = 0.005


def clipped_l1_lift(pair: dict, clip: float = DEFAULT_L1_CLIP) -> float:
    lift = l1_lift(pair)
    if clip is None or clip <= 0:
        return lift
    return max(-clip, min(lift, clip))


def rank_term(pair: dict, l1_weight: float = DEFAULT_L1_WEIGHT,
              l1_clip: float = DEFAULT_L1_CLIP) -> float:
    """LEGACY v2 mix: Λ2 + w·clip(L1lift, ±clip)."""
    return lambda2(pair) + l1_weight * clipped_l1_lift(pair, l1_clip)


soft_lambda2 = rank_term  # oldest alias


@dataclass
class LegacyMinerScore:
    miner: str
    valid: bool
    gate_pass_rate: float
    bank_frac: float
    S: float
    n_pairs: int
    n_turns: int
    calib_ratio: float | None = None
    baseline_abs: float | None = None


def legacy_score_miner(rows: list[dict], tau: float = DEFAULT_TAU,
                       gamma: float = DEFAULT_GAMMA,
                       bank_frac: float | None = None,
                       gamma_bank: float = DEFAULT_GAMMA_BANK,
                       l1_weight: float = DEFAULT_L1_WEIGHT,
                       l1_clip: float = DEFAULT_L1_CLIP,
                       r_lo: float = DEFAULT_R_LO,
                       r_hi: float = DEFAULT_R_HI,
                       check_calib: bool = True) -> LegacyMinerScore:
    """LEGACY v2 gated scoring — for replaying pre-fork freezes only."""
    if not rows:
        return LegacyMinerScore("?", False, 0.0, 0.0, float("-inf"), 0, 0, None)
    pairs = [p for r in rows if r.get("valid") and "pairs" in r for p in r["pairs"]]
    if not pairs:
        return LegacyMinerScore(rows[0].get("miner", "?"), False, 0.0, 0.0,
                                float("-inf"), 0, 0, None)
    rate = st.mean(1.0 if gate_pass(p, tau) else 0.0 for p in pairs)
    s = st.mean(rank_term(p, l1_weight, l1_clip) for p in pairs)
    bf = 1.0 if bank_frac is None else bank_frac
    r = calibration_ratio(pairs)
    calib_ok = (not check_calib) or (r is not None and r_lo <= r <= r_hi)
    valid = rate >= gamma and bf >= gamma_bank and calib_ok
    return LegacyMinerScore(
        miner=rows[0].get("miner", "?"),
        valid=valid,
        gate_pass_rate=rate,
        bank_frac=bf,
        S=s if valid else float("-inf"),
        n_pairs=len(pairs),
        n_turns=len({r["turn_id"] for r in rows}),
        calib_ratio=r,
        baseline_abs=st.mean(abs(p["lpA_yc_e"]) for p in pairs),
    )


@dataclass
class LegacyDuelResult:
    challenger: str
    king: str
    margin: float
    se: float
    z: float
    k_sigma: float
    challenger_wins: bool
    n_paired_turns: int
    min_margin: float = DEFAULT_MIN_MARGIN


def legacy_duel(challenger_rows: list[dict], king_rows: list[dict],
                k_sigma: float = DEFAULT_K_SIGMA,
                tau: float = DEFAULT_TAU, gamma: float = DEFAULT_GAMMA,
                challenger_bank_frac: float | None = None,
                king_bank_frac: float | None = None,
                gamma_bank: float = DEFAULT_GAMMA_BANK,
                l1_weight: float = DEFAULT_L1_WEIGHT,
                l1_clip: float = DEFAULT_L1_CLIP,
                r_lo: float = DEFAULT_R_LO,
                r_hi: float = DEFAULT_R_HI,
                check_calib: bool = True,
                min_se: float = DEFAULT_MIN_SE,
                min_margin: float = DEFAULT_MIN_MARGIN,
                baseline_band: float | None = DEFAULT_BASELINE_BAND) -> LegacyDuelResult:
    """LEGACY v2 duel (gates + δ floor + SE floor) — for replay only."""
    cs = legacy_score_miner(challenger_rows, tau, gamma, challenger_bank_frac,
                            gamma_bank, l1_weight, l1_clip, r_lo, r_hi, check_calib)
    ks = legacy_score_miner(king_rows, tau, gamma, king_bank_frac,
                            gamma_bank, l1_weight, l1_clip, r_lo, r_hi, check_calib)
    if (baseline_band and cs.baseline_abs and ks.baseline_abs
            and cs.baseline_abs > baseline_band * ks.baseline_abs):
        cs.valid = False
        cs.S = float("-inf")
    c_by = {r["turn_id"]: r for r in challenger_rows if r.get("valid") and "pairs" in r}
    k_by = {r["turn_id"]: r for r in king_rows if r.get("valid") and "pairs" in r}
    diffs = []
    for tid in sorted(set(c_by) & set(k_by)):
        lc = st.mean(rank_term(p, l1_weight, l1_clip) for p in c_by[tid]["pairs"])
        lk = st.mean(rank_term(p, l1_weight, l1_clip) for p in k_by[tid]["pairs"])
        diffs.append(lc - lk)
    n = len(diffs)
    if n < 2 or not cs.valid or not ks.valid:
        return LegacyDuelResult(cs.miner, ks.miner, 0.0, float("inf"), 0.0,
                                k_sigma, False, n, min_margin)
    mean = st.mean(diffs)
    se = max(st.stdev(diffs) / math.sqrt(n), min_se)
    z = mean / se if se > 0 else 0.0
    wins = (mean > k_sigma * se) and (mean > min_margin)
    return LegacyDuelResult(
        challenger=cs.miner, king=ks.miner, margin=mean, se=se, z=z,
        k_sigma=k_sigma, challenger_wins=wins, n_paired_turns=n,
        min_margin=min_margin,
    )
