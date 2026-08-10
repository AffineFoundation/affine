"""Frozen production scoring rule: Reason (v3, 2026-08-10).

Shared between root validator and eval server. Any change here is a chain fork
(bump [subnet].weight_version_key).

Reason(A; C, D):
  Per pair:   Reason = lpC(y_C | z_A) − lpC(y_C | ∅)
  Per miner:  score  = mean(Reason) over all pairs
  Duel:       challenger wins iff paired mean(Reason_c − Reason_k) > k_sigma · SE
              with SE = stdev(diffs) / sqrt(n) over paired turns.

That is the whole contract. Scoring hyperparameters: n_turns and k_sigma.
There is no mix, no clip, no gates, and no absolute margin floor.

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


DEFAULT_K_SIGMA = 3.0

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
              fuzzy: float = TELEMETRY_FUZZY) -> bool:
    """Telemetry: legacy causality+leakage pass (no longer affects score)."""
    if leakage(pair.get("z_a", ""), pair.get("y_a", ""), fuzzy=fuzzy):
        return False
    return (pair["lpA_ya_za"] - pair["lpA_ya_e"]) >= tau


def reason(pair: dict) -> float:
    """Reason = lpC(y_C|z_A) − lpC(y_C|∅). The score."""
    return pair["lpC_yc_za"] - pair["lpC_yc_e"]


# Historical name (Λ2) kept for research scripts and old artifact replay.
lambda2 = reason


def l1_lift(pair: dict) -> float:
    """Telemetry: miner-side lift lpA(y_C|z_A) − lpA(y_C|∅) (not scored)."""
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
    reason: float                     # the score: mean per-pair Reason
    n_pairs: int
    n_turns: int
    # -- telemetry (measured, never scored) --
    gate_pass_rate: float = 0.0
    bank_frac: float | None = None
    calib_ratio: float | None = None
    baseline_abs: float | None = None  # mean|lpA(y_C|∅)|
    mean_l1lift: float | None = None
    mean_len_z: float | None = None    # chars of z_A
    mean_len_y: float | None = None    # chars of y_A


def score_miner(rows: list[dict],
                bank_frac: float | None = None) -> MinerScore:
    """Score one miner: mean Reason + telemetry. No gating of any kind."""
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
    """Paired duel on per-turn mean Reason. Purely relative: wins iff
    mean > k_sigma·SE. Bank fracs are accepted only to thread telemetry."""
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
    wins = mean > k_sigma * se
    return DuelResult(
        challenger=cs.miner, king=ks.miner, margin=mean, se=se, z=z,
        k_sigma=k_sigma, challenger_wins=wins, n_paired_turns=n,
    )
