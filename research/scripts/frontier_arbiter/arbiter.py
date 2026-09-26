"""Frontier-arbiter rule family — pure per-turn functions (2026-09-20).

Idea under test: a stronger "frontier" model F samples the same turn prefix
as the frozen teacher C. Where F AGREES with the teacher's k reference
actions the turn keeps today's teacher score; where F DISAGREES the miner
is paid for being closer to F than the teacher was. Variant B instead
re-weights the k teacher refs by their closeness to F.

Everything here is a pure function of numbers/strings already on hand; the
loaders live in rule_sim.py. Notation (all logprobs per byte under the
teacher C, echoed with the canonical rendering of this stored data):

    a_i  = lpC(y_C^i | x, z)  − lpC(y_C^i | x, ∅)     per-teacher-ref Reason of thought z
    f_j  = lpC(y_F^j | x, z)  − lpC(y_F^j | x, ∅)     same against frontier action j
    b_j  = lpC(y     | x, z_F^j) − lpC(y | x, ∅)     frontier thought j licensing action y
    m    = lpC(z | x)                                  thought grounding echo
    t_i  = lpC(z_C^i | x)                              teacher reference thought echoes

Gate D (F disagrees with the teacher): one frontier action differs from ALL
k refs under `method`; with several frontier actions a strict majority must
disagree.

Agree branch: live min(R, G) — R = centered tempered LME over a_i
(affine.score.centered_reason), G = distance into the band mu ± max(band_c ·
sd(t_i), band_floor) (affine.score.grounding).

Disagree branch ("closer to F than the teacher was"):
    act   s_A = mean_j [ sim(y, y_F^j) − mean_i sim(y_C^i, y_F^j) ]   (sim = jaccard | exact),
          scaled to score units by × scale × sd(live turn scores of the dialect)
    RF    centered tempered LME over f_j                 (frontier-target Reason)
    V     mean_j f_j − mean_i a_i                        (signed contrast: does the thought
                                                          favour F's action over the teacher's)
    AF    tempered LME over b_j                          (frontier-thought licence of the action)
    minRFG = min(RF, G), minVG = min(V, G)              (hybrids: G still bounds)

Variant B: w_i ∝ softmax(sim(y_C^i, F)/T) or hard 1[sim ≥ θ] (uniform when
all zero); R_w = weighted centered LME (wclme), G_w from the weighted mu/sd.

sd units: every leg standardised per dialect by (mu, sigma) of the TEACHER's
own leave-one-out values (`Standardizer`), the thought leg replaced by the
typicality `width − |m − mean(t_i)| / sigma_t` with sigma_t the pooled
within-turn sd of the t_i per dialect (a per-byte stand-in for the live
wvk-22 content-masked typ_c; the stored data has no per-token echoes).
"""

from __future__ import annotations

import math
import statistics as st
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontier_arbiter.common import clme, jaccard, lme, norm_action  # noqa: E402

TAU = 0.03
BAND_FLOOR = 0.002
BAND_C_LIVE = 4.0        # wvk 17+ (band_c 2.0 = the era of the stored duels)
SURPRISE_NATS = 0.02     # per-byte teacher-surprise threshold of the "surprise" gate
TYP_WIDTH = 2.0          # live [duel.sd_meter].typicality_width
GATE_METHODS = ("exact", "jaccard", "surprise")
SIM_NAMES = ("jaccard", "exact")


# ------------------------------------------------------------------ similarity
@lru_cache(maxsize=1 << 18)
def sim_exact(y: str, t: str, kind: str | None) -> float:
    return 1.0 if norm_action(y, kind) == norm_action(t, kind) else 0.0


@lru_cache(maxsize=1 << 18)
def sim_jaccard(y: str, t: str, kind: str | None) -> float:
    del kind
    return jaccard(y, t)


SIMS = {"jaccard": sim_jaccard, "exact": sim_exact}


def best_sim(y: str, targets: list[str], kind: str | None, sim: str = "jaccard") -> float:
    fn = SIMS[sim]
    return max((fn(y, t, kind) for t in targets), default=0.0)


def mean_sim(y: str, targets: list[str], kind: str | None, sim: str = "jaccard") -> float:
    fn = SIMS[sim]
    return st.mean(fn(y, t, kind) for t in targets) if targets else 0.0


# ------------------------------------------------------------------ gate
def f_disagrees(yF: str, yC: list[str], kind: str | None, method: str,
                theta: float = 0.5, lpF_e: float | None = None,
                lpC_e: list[float] | None = None,
                surprise_nats: float = SURPRISE_NATS) -> bool:
    """Does ONE frontier action differ from ALL k teacher refs?

    exact    — no ref equals it after norm_action
    jaccard  — best token-Jaccard against the refs < theta
    surprise — lpC(y_F|x,∅) < min_i lpC(y_C^i|x,∅) − surprise_nats  (the teacher
               finds F's action less natural than any of its own)
    """
    if method == "exact":
        return not any(sim_exact(yF, yc, kind) >= 1.0 for yc in yC)
    if method == "jaccard":
        return best_sim(yF, yC, kind, "jaccard") < theta
    if method == "surprise":
        if lpF_e is None or not lpC_e:
            raise ValueError("surprise gate needs lpC(y_F|∅) and lpC(y_C^i|∅)")
        return lpF_e < min(lpC_e) - surprise_nats
    raise ValueError(f"unknown gate method {method!r}")


def gate_disagree(yF: list[str], yC: list[str], kind: str | None, method: str,
                  theta: float = 0.5, lpF_e: list[float] | None = None,
                  lpC_e: list[float] | None = None,
                  surprise_nats: float = SURPRISE_NATS) -> bool:
    """Majority rule over the frontier actions (one action = itself; two =
    both; three = at least two)."""
    if not yF:
        return False
    flags = [f_disagrees(y, yC, kind, method, theta,
                         None if lpF_e is None else lpF_e[j], lpC_e, surprise_nats)
             for j, y in enumerate(yF)]
    return 2 * sum(flags) > len(flags)


# ------------------------------------------------------------------ live legs
def reason_leg(a: list[float], tau: float = TAU) -> float:
    """R = centered tempered LME (affine.score.centered_reason). k=1 → 0."""
    return clme(a, tau)


def band(t: list[float], band_c: float, band_floor: float = BAND_FLOOR) -> tuple[float, float]:
    mu = st.mean(t)
    sd = st.stdev(t) if len(t) >= 2 else 0.0
    return mu, max(band_c * sd, band_floor)


def grounding_leg(m: float, t: list[float], band_c: float,
                  band_floor: float = BAND_FLOOR) -> float:
    """G = min(m − (mu − w), (mu + w) − m): positive iff m is inside the band."""
    mu, w = band(t, band_c, band_floor)
    return min(m - (mu - w), (mu + w) - m)


def live_min_rg(a: list[float], m: float, t: list[float], tau: float = TAU,
                band_c: float = BAND_C_LIVE, band_floor: float = BAND_FLOOR) -> float:
    return min(reason_leg(a, tau), grounding_leg(m, t, band_c, band_floor))


# ------------------------------------------------------------------ disagree-branch closeness
def action_closeness(y: str, yF: list[str], yC: list[str], kind: str | None,
                     sim: str = "jaccard") -> float:
    """s_A = mean_j [ sim(y, y_F^j) − mean_i sim(y_C^i, y_F^j) ] ∈ [−1, 1].

    Positive iff the action is closer to the frontier's than the teacher's
    own refs were, on average over the frontier samples."""
    if not yF or not yC:
        return 0.0
    fn = SIMS[sim]
    return st.mean(fn(y, yf, kind) - st.mean(fn(yc, yf, kind) for yc in yC) for yf in yF)


def frontier_reason(f: list[float], tau: float = TAU) -> float:
    """R_F = centered tempered LME over the frontier refs (0 when they coincide
    or when there is a single frontier action)."""
    return clme(f, tau)


def frontier_contrast(f: list[float], a: list[float]) -> float:
    """V = mean_j f_j − mean_i a_i (signed; centering-free, so a flat lift cancels
    between the two terms but a thought that names F's action over the
    teacher's scores positive)."""
    return st.mean(f) - st.mean(a)


def frontier_action_leg(b: list[float], tau: float = TAU) -> float:
    """A_F = tempered LME over b_j = lpC(y|z_F^j) − lpC(y|∅) (NOT centered, as
    affine.score.action_leg)."""
    return lme(b, tau)


# ------------------------------------------------------------------ combination
def arbiter_turn(disagree: bool, live: float, alt: float) -> float:
    """Turn score: today's teacher score where F agrees, the closeness-to-F
    score where F disagrees."""
    return alt if disagree else live


# ------------------------------------------------------------------ variant B (ref weighting)
def ref_weights(sims: list[float], mode: str = "softmax", temperature: float = 0.1,
                theta: float = 0.5) -> list[float]:
    """w_i over the teacher refs from sim(y_C^i, F) ∈ [0,1]; sums to 1.
    softmax: w ∝ exp(sim/T); hard: 1[sim ≥ θ] (uniform when none qualifies)."""
    if not sims:
        return []
    if mode == "softmax":
        mx = max(sims)
        w = [math.exp((s - mx) / temperature) for s in sims]
    elif mode == "hard":
        w = [1.0 if s >= theta else 0.0 for s in sims]
        if not any(w):
            w = [1.0] * len(sims)
    else:
        raise ValueError(f"unknown weight mode {mode!r}")
    z = sum(w)
    return [x / z for x in w]


def wclme(a: list[float], w: list[float], tau: float = TAU) -> float:
    """Weighted centered tempered LME: tau·log(Σ w_i e^{a_i/tau}) − Σ w_i a_i.
    Uniform weights recover clme; a single non-zero weight gives 0."""
    if not a:
        return float("nan")
    z = sum(w)
    w = [x / z for x in w]
    m = max(a)
    return m + tau * math.log(sum(wi * math.exp((ai - m) / tau) for ai, wi in zip(a, w))) \
        - sum(wi * ai for ai, wi in zip(a, w))


def weighted_band(t: list[float], w: list[float], band_c: float,
                  band_floor: float = BAND_FLOOR) -> tuple[float, float]:
    z = sum(w)
    w = [x / z for x in w]
    mu = sum(wi * ti for ti, wi in zip(t, w))
    var = sum(wi * (ti - mu) ** 2 for ti, wi in zip(t, w))
    # unbiased-ish correction so uniform weights recover stdev(t)
    n_eff = 1.0 / sum(wi * wi for wi in w)
    sd = math.sqrt(var * n_eff / (n_eff - 1)) if n_eff > 1.0 + 1e-9 else 0.0
    return mu, max(band_c * sd, band_floor)


def weighted_min_rg(a: list[float], m: float, t: list[float], w: list[float],
                    tau: float = TAU, band_c: float = BAND_C_LIVE,
                    band_floor: float = BAND_FLOOR) -> float:
    mu, wd = weighted_band(t, w, band_c, band_floor)
    return min(wclme(a, w, tau), min(m - (mu - wd), (mu + wd) - m))


def weights_collapsed(w: list[float], thresh: float = 0.9) -> bool:
    return bool(w) and max(w) >= thresh


# ------------------------------------------------------------------ sd units
def pooled_within_sd(groups: list[list[float]]) -> float:
    """sqrt(mean within-group variance) — the sd-meter's sigma rule."""
    vs = [st.variance(g) for g in groups if len(g) >= 2]
    return math.sqrt(st.mean(vs)) if vs else float("nan")


def typicality(m: float, t: list[float], sigma_t: float, width: float = TYP_WIDTH) -> float:
    """width − |m − mean(t)| / sigma_t: positive iff the thought's grounding
    echo sits within `width` teacher-sd of the refs' mean."""
    if not sigma_t or not math.isfinite(sigma_t):
        return float("nan")
    return width - abs(m - st.mean(t)) / sigma_t


@dataclass
class Standardizer:
    """Per-(dialect, leg) anchors (mu, sigma) from the teacher's own values."""
    mu: dict[tuple[str, str], float] = field(default_factory=dict)
    sigma: dict[tuple[str, str], float] = field(default_factory=dict)
    n: dict[tuple[str, str], int] = field(default_factory=dict)

    def fit(self, kind: str, leg: str, vals: list[float]) -> None:
        vals = [v for v in vals if v is not None and math.isfinite(v)]
        self.n[(kind, leg)] = len(vals)
        if len(vals) >= 3:
            self.mu[(kind, leg)] = st.mean(vals)
            self.sigma[(kind, leg)] = st.stdev(vals) or float("nan")

    def z(self, kind: str, leg: str, x: float) -> float:
        key = (kind, leg)
        if key not in self.sigma or not math.isfinite(x):
            return float("nan")
        return (x - self.mu[key]) / self.sigma[key]
