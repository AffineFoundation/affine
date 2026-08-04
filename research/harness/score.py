"""Production scoring rule S* (post E-KINGS + RT-1/2/2c/3b/4 + A11 floor).

S*(A; C, D):
  1. Causality+leakage gate (per pair, miner-level aggregate):
       pass if (no fuzzy z⊃y leakage) ∧ (lpA(y_A|z_A)−lpA(y_A|∅) ≥ τ)
       Miner INVALID if pass_rate < γ (default 0.30).
       Closes exact stuffing (0%) and silence (≤5%).
  2. Prior-bank positivity gate (miner-level):
       frac_bank = fraction of pairs with Λ2_bank > 0, where
         Λ2_bank = lpC(y_C|z_A) − max_k lpC(y_C|prior_k)
       over published bank {∅, ls, cat, grep, find, test, para_ls}.
       Miner INVALID if frac_bank < γ_bank (default 0.08).
       Closes paraphrase stuffing (RT-2c: frac_bank=0).
  3. Serve-time calibration-ratio gate (miner-level, RT-3b/3d):
       r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|
       Miner INVALID if r ∉ [r_lo, r_hi] (default [1.0, 4.0]).
       Honest fleet band ≈ [1.07, 3.56]; RT-3b must leave band to breach duel.
  4. Ranking term (teacher-anchored mix, L1lift clipped):
       S = mean(Λ2 + w · clip(L1lift, ±clip)) with w=1.0, clip=0.1
         Λ2     = lpC(y_C|z_A) − lpC(y_C|∅)
         L1lift = lpA(y_C|z_A) − lpA(y_C|∅)
       clip0.1 keeps ρ≈+0.862@19 and caps RT-3b per-turn inflation.
  5. Duel: challenger wins iff paired mean(S_c − S_k) > 3·SE
       AND mean margin > δ (default 0.05). Effect-size floor closes A11
       short-style n*-scaling (II would cross pure 3σ at ~244 turns).

`score_miner` needs pair components from v2 instrumentation. Bank positivity is
passed in via `bank_frac` (from `harness.rescore_bank` / online bank scoring)
because prior-bank logprobs are not in the base pair record.
"""

from __future__ import annotations

import json
import math
import re
import statistics as st
from dataclasses import dataclass


DEFAULT_TAU = 0.02
DEFAULT_GAMMA = 0.30
DEFAULT_GAMMA_BANK = 0.08
DEFAULT_K_SIGMA = 3.0
DEFAULT_FUZZY = 0.6
DEFAULT_L1_WEIGHT = 1.0
DEFAULT_L1_CLIP = 0.1
DEFAULT_R_LO = 1.0
DEFAULT_R_HI = 4.0
DEFAULT_MIN_MARGIN = 0.05


def _cmd(y: str) -> str:
    """Normalize action text for leakage checks (bash fence or raw tool JSON)."""
    y = y.strip()
    if y.startswith("```bash\n") and y.endswith("\n```"):
        return y.removeprefix("```bash\n").removesuffix("\n```").strip()
    return y


def leakage(z: str, y: str, fuzzy: float = DEFAULT_FUZZY) -> bool:
    c = _cmd(y)
    if not c:
        return False
    if c in z:
        return True
    # Tool JSON: leak if the tool name appears verbatim in the thought.
    if c.startswith("{") and '"name"' in c:
        try:
            name = json.loads(c).get("name") or ""
        except json.JSONDecodeError:
            name = ""
        if name and name in z:
            return True
        return False
    toks = [t for t in re.split(r"\s+", c) if len(t) >= 3]
    if not toks:
        return False
    return sum(1 for t in toks if t in z) / len(toks) >= fuzzy


def gate_pass(pair: dict, tau: float = DEFAULT_TAU,
              fuzzy: float = DEFAULT_FUZZY) -> bool:
    if leakage(pair.get("z_a", ""), pair.get("y_a", ""), fuzzy=fuzzy):
        return False
    return (pair["lpA_ya_za"] - pair["lpA_ya_e"]) >= tau


def lambda2(pair: dict) -> float:
    return pair["lpC_yc_za"] - pair["lpC_yc_e"]


def l1_lift(pair: dict) -> float:
    return pair["lpA_yc_za"] - pair["lpA_yc_e"]


def clipped_l1_lift(pair: dict, clip: float = DEFAULT_L1_CLIP) -> float:
    lift = l1_lift(pair)
    if clip is None or clip <= 0:
        return lift
    return max(-clip, min(lift, clip))


def rank_term(pair: dict, l1_weight: float = DEFAULT_L1_WEIGHT,
              l1_clip: float = DEFAULT_L1_CLIP) -> float:
    """Teacher-anchored mix: Λ2 + w·clip(L1lift, ±clip)."""
    return lambda2(pair) + l1_weight * clipped_l1_lift(pair, l1_clip)


def calibration_ratio(pairs: list[dict]) -> float | None:
    """r = mean|lpA(y_C|z_A)| / mean|lpA(y_C|∅)|. None if empty baseline is 0."""
    if not pairs:
        return None
    num = st.mean(abs(p["lpA_yc_za"]) for p in pairs)
    den = st.mean(abs(p["lpA_yc_e"]) for p in pairs)
    if den <= 0:
        return None
    return num / den


# Back-compat aliases used by older call sites / notebooks.
soft_lambda2 = rank_term


@dataclass
class MinerScore:
    miner: str
    valid: bool
    gate_pass_rate: float
    bank_frac: float
    S: float
    n_pairs: int
    n_turns: int
    calib_ratio: float | None = None


def score_miner(rows: list[dict], tau: float = DEFAULT_TAU,
                gamma: float = DEFAULT_GAMMA,
                bank_frac: float | None = None,
                gamma_bank: float = DEFAULT_GAMMA_BANK,
                l1_weight: float = DEFAULT_L1_WEIGHT,
                l1_clip: float = DEFAULT_L1_CLIP,
                r_lo: float = DEFAULT_R_LO,
                r_hi: float = DEFAULT_R_HI,
                check_calib: bool = True) -> MinerScore:
    """Score one miner. Pass `bank_frac` from prior-bank scoring when available."""
    if not rows:
        return MinerScore("?", False, 0.0, 0.0, float("-inf"), 0, 0, None)
    pairs = [p for r in rows if r.get("valid") and "pairs" in r for p in r["pairs"]]
    if not pairs:
        return MinerScore(rows[0].get("miner", "?"), False, 0.0, 0.0,
                          float("-inf"), 0, 0, None)
    rate = st.mean(1.0 if gate_pass(p, tau) else 0.0 for p in pairs)
    s = st.mean(rank_term(p, l1_weight, l1_clip) for p in pairs)
    bf = 1.0 if bank_frac is None else bank_frac
    r = calibration_ratio(pairs)
    calib_ok = (not check_calib) or (r is not None and r_lo <= r <= r_hi)
    valid = rate >= gamma and bf >= gamma_bank and calib_ok
    return MinerScore(
        miner=rows[0].get("miner", "?"),
        valid=valid,
        gate_pass_rate=rate,
        bank_frac=bf,
        S=s if valid else float("-inf"),
        n_pairs=len(pairs),
        n_turns=len({r["turn_id"] for r in rows}),
        calib_ratio=r,
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


def duel(challenger_rows: list[dict], king_rows: list[dict],
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
         min_margin: float = DEFAULT_MIN_MARGIN) -> DuelResult:
    cs = score_miner(challenger_rows, tau, gamma, challenger_bank_frac,
                     gamma_bank, l1_weight, l1_clip, r_lo, r_hi, check_calib)
    ks = score_miner(king_rows, tau, gamma, king_bank_frac,
                     gamma_bank, l1_weight, l1_clip, r_lo, r_hi, check_calib)
    c_by = {r["turn_id"]: r for r in challenger_rows if r.get("valid") and "pairs" in r}
    k_by = {r["turn_id"]: r for r in king_rows if r.get("valid") and "pairs" in r}
    diffs = []
    for tid in sorted(set(c_by) & set(k_by)):
        lc = st.mean(rank_term(p, l1_weight, l1_clip) for p in c_by[tid]["pairs"])
        lk = st.mean(rank_term(p, l1_weight, l1_clip) for p in k_by[tid]["pairs"])
        diffs.append(lc - lk)
    n = len(diffs)
    if n < 2 or not cs.valid or not ks.valid:
        return DuelResult(cs.miner, ks.miner, 0.0, float("inf"), 0.0, k_sigma,
                          False, n, min_margin)
    mean = st.mean(diffs)
    se = st.stdev(diffs) / math.sqrt(n)
    z = mean / se if se > 0 else 0.0
    wins = (mean > k_sigma * se) and (mean > min_margin)
    return DuelResult(
        challenger=cs.miner, king=ks.miner, margin=mean, se=se, z=z,
        k_sigma=k_sigma, challenger_wins=wins, n_paired_turns=n,
        min_margin=min_margin,
    )
