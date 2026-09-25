"""sd-meter: ``min(z_R, typ_c, z_A)`` in teacher-sd units (shadow, 2026-09-18).

The god-equation finalist (docs/god-equation-search.md §9), computed on
every live duel next to min(R,G) and published under ``verdict.shadow.
sd_meter`` — telemetry only until an operator flips ``score_mode`` to
``"sd_min_rga"`` (a weight_version_key event).

Per turn x with teacher refs (z_C^i, y_C^i), i = 1..k, and a miner reply
(z_A, y_A):

  R      = live centred Reason (affine.score.centered_reason)
  A      = summed action leg: b_i = [lpC(y_A|z_C^i) − lpC(y_A|∅)] · bytes(y_A)
           / a_norm_bytes, A = τ·log mean_i exp(b_i/τ)   (nats when a_norm_bytes = 1)
  m_c    = mean per-token teacher logprob of z_A over its CONTENT tokens:
           tokens with |lpC(tok|x) − lpC(tok|∅)| > content_lift_nats
  μ_c    = mean over the k refs of the same masked statistic of z_C^i
  z_R    = (R − μ_R) / σ_R,  z_A = (A − μ_A) / σ_A
  typ_c  = typicality_width − |m_c − μ_c| / σ_c
  turn   = min(z_R, typ_c, z_A);  forfeit → forfeit_sd;  fewer than
           content_min_tokens content tokens → typ_c = forfeit_sd

Anchors (μ_R, μ_A per turn; σ per dialect):
  "loo"    — leave-one-out over the k refs: ref j plays miner against the
             other k−1 refs (cross echoes lpC(y_C^i|z_C^j), i ≠ j, six per
             turn, shared by both sides). μ = mean_j of the k values, σ =
             pooled within-turn sd per dialect over the duel's turns.
  "frozen" — per-dialect constants (μ, σ) from ``frozen`` in the config
             (phase-2 slice values, refreshed from published verdicts with
             ops/sd-meter/refresh_frozen.py). Zero extra echoes.
  Both are computed when the cross echoes exist; ``anchor`` names the one
  whose decision is headlined.

Crown (shadow "would-crown"): paired mean over turns > max(k_sigma·SE,
min_margin_sd), plus the live contract gates (thought floor, B licence)
read from the live result.
"""

from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field

from affine import dialects
from affine.score import action_leg, centered_reason, is_forfeit

DIALECTS = ("bash", "tool_call", "text", "boxed", "terminus_json")

DEFAULTS = {
    "shadow": False,
    "anchor": "loo",
    "cross_echo": True,
    # wvk 23 (2026-09-22): "none" = every content token of the miner's thought
    # is scored (wvk 22); "refs_max" = only the first K content tokens, K =
    # the largest content-token count among the turn's k references — a
    # thought longer / more deliberate than the teacher's is not penalised
    # for the extra; the two-sided band still applies to the scored prefix.
    "content_prefix": "none",
    # wvk 24 (2026-09-23): reference thoughts with fewer than ref_min_content
    # content tokens are EXCLUDED from the typicality anchor (μ_c, σ_c) and
    # from the teacher control instead of entering it with a noisy mean;
    # a turn with fewer than typ_min_refs content-bearing references scores
    # min(z_R, z_A) (typicality leg dropped). ref_min_content = 0 = the
    # wvk 22/23 rule (every reference enters the anchor).
    "ref_min_content": 0,
    "typ_min_refs": 2,
    "content_lift_nats": 1.0,
    "content_min_tokens": 10,
    "typicality_width": 2.0,
    "a_norm_bytes": 1.0,
    "forfeit_sd": -2.4,
    "k_sigma": 2.0,
    "min_margin_sd": 0.0,
    "frozen": {},
}


def settings(duel_cfg: dict) -> dict:
    """``[duel.sd_meter]`` with defaults filled in."""
    raw = dict(duel_cfg.get("sd_meter") or {})
    out = dict(DEFAULTS)
    out.update({k: raw[k] for k in raw if k in DEFAULTS})
    out["anchor"] = str(out["anchor"])
    if out["anchor"] not in ("loo", "frozen"):
        raise ValueError(f"[duel.sd_meter] anchor must be loo|frozen, got {out['anchor']!r}")
    for k in ("content_lift_nats", "typicality_width", "a_norm_bytes",
              "forfeit_sd", "k_sigma", "min_margin_sd"):
        out[k] = float(out[k])
    out["content_min_tokens"] = int(out["content_min_tokens"])
    out["shadow"] = bool(out["shadow"])
    out["cross_echo"] = bool(out["cross_echo"])
    out["content_prefix"] = str(out["content_prefix"])
    out["ref_min_content"] = int(out["ref_min_content"])
    out["typ_min_refs"] = int(out["typ_min_refs"])
    if out["content_prefix"] not in ("none", "refs_max"):
        raise ValueError(f"[duel.sd_meter] content_prefix must be none|refs_max, got {out['content_prefix']!r}")
    out["frozen"] = dict(out["frozen"] or {})
    return out


# -- content mask ---------------------------------------------------------------

def content_stats(tok_x: list[tuple], tok_e: list[tuple], theta: float) -> dict:
    """Content-masked thought statistic from two per-token echoes of the
    same thought: with the task (``tok_x``) and without it (``tok_e``).
    Tokens are (start, end, lp) with offsets relative to the thought; the
    two tokenizations are aligned on offsets (identical except, rarely, at
    the first token). Returns mean lp over the kept tokens and counts."""
    e_by = {(s, e): lp for s, e, lp in tok_e}
    kept: list[float] = []
    lifts: list[float] = []
    n_aligned = 0
    for s, e, lp in tok_x:
        lp_e = e_by.get((s, e))
        if lp_e is None:
            continue
        n_aligned += 1
        lift = lp - lp_e
        lifts.append(lift)
        if abs(lift) > theta:
            kept.append(lp)
    return {
        "mc": (sum(kept) / len(kept)) if kept else None,
        "n_content": len(kept),
        # kept content-token logprobs in thought order (for the wvk-23
        # content_prefix truncation; not stored on rows)
        "kept": kept,
        "n_tokens": len(tok_x),
        "n_aligned": n_aligned,
        "mean_lift": (sum(lifts) / len(lifts)) if lifts else None,
    }


# -- per-turn legs ------------------------------------------------------------------

def _lme(vals: list[float], tau: float | None) -> float:
    if tau is None or tau <= 0 or len(vals) == 1:
        return st.mean(vals)
    m = max(vals)
    return m + tau * math.log(st.mean(math.exp((v - m) / tau) for v in vals))


def ref_loo_terms(refs: list[dict], tau: float | None, a_norm_bytes: float,
                  ref_min_content: int = 0) -> dict | None:
    """Leave-one-out teacher values on one turn from the ref records.

    refs[i] carries lp_empty (lpC(y_C^i|∅)), n_bytes_y, and lp_cross:
    list over j of lpC(y_C^i | z_C^j) (None at j == i). Ref j as miner:
      R^(j) = LME_i≠j(a_i^(j)) − mean_i≠j(a_i^(j)),  a_i^(j) = lp_cross[i][j] − lp_empty[i]
      A^(j) = LME_i≠j(b_i^(j)),  b_i^(j) = (lp_cross[j][i] − lp_empty[j]) · bytes(y_C^j) / a_norm
    Returns per-ref lists (or None when the cross echoes are missing)."""
    k = len(refs)
    if k < 2 or any(r.get("lp_cross") is None for r in refs):
        return None
    R, A, M = [], [], []
    for j in range(k):
        a = []
        b = []
        for i in range(k):
            if i == j:
                continue
            c_ij = refs[i]["lp_cross"][j]      # lpC(y_C^i | z_C^j)
            c_ji = refs[j]["lp_cross"][i]      # lpC(y_C^j | z_C^i)
            if c_ij is None or c_ji is None:
                return None
            a.append(c_ij - refs[i]["lp_empty"])
            nb = refs[j].get("n_bytes_y") or 1
            b.append((c_ji - refs[j]["lp_empty"]) * nb / a_norm_bytes)
        R.append(_lme(a, tau) - st.mean(a))
        A.append(_lme(b, tau))
        mc = refs[j].get("mc_thought")
        if ref_min_content and (refs[j].get("n_content_thought") or 0) < ref_min_content:
            mc = None      # wvk 24: an (almost) empty reference thought does not anchor typicality
        M.append(mc)
    return {"R": R, "A": A, "Mc": M}


def side_legs_subset(row: dict, tau: float | None, a_norm_bytes: float,
                     idx: list[int]) -> dict | None:
    """Raw legs of one side computed over a SUBSET of its reference pairs
    (pairs are in reference order): R and A as tempered LMEs over the pairs
    in ``idx`` only. Used by the fully matched control, where the king must
    be scored over the same k−1 references as the held-out teacher
    reference. None for a forfeit row or when the subset is incomplete."""
    if is_forfeit(row):
        return None
    pairs = row["pairs"]
    if any(i >= len(pairs) for i in idx) or not idx:
        return None
    sub = [pairs[i] for i in idx]
    R = centered_reason(sub, tau)
    A = action_leg(sub, tau, a_norm_bytes)
    p0 = pairs[0]
    return {"R": R, "A": A, "mc": p0.get("mc_za"),
            "n_content": p0.get("n_content_za"), "n_tokens": p0.get("n_tokens_za")}


def side_legs(row: dict, tau: float | None, a_norm_bytes: float) -> dict | None:
    """Raw legs of one side on one turn: R, A (summed), m_c + content counts.
    None for a forfeit row."""
    if is_forfeit(row):
        return None
    pairs = row["pairs"]
    R = centered_reason(pairs, tau)
    A = action_leg(pairs, tau, a_norm_bytes)
    p0 = pairs[0]
    return {"R": R, "A": A, "mc": p0.get("mc_za"),
            "n_content": p0.get("n_content_za"), "n_tokens": p0.get("n_tokens_za")}


def _var(vals: list[float]) -> float | None:
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return st.variance(vals) if len(vals) >= 2 else None


@dataclass
class Anchors:
    """Per-turn μ (LOO) and per-dialect σ, plus what went into them."""
    mu: dict[str, dict[str, float]] = field(default_factory=dict)       # tid -> {R, A, Mc}
    sigma: dict[str, dict[str, float | None]] = field(default_factory=dict)  # dialect -> {R, A, Mc}
    n_turns: dict[str, int] = field(default_factory=dict)               # dialect -> turns with LOO
    mu_mean: dict[str, dict[str, float | None]] = field(default_factory=dict)


def loo_anchors(turn_refs: dict[str, list[dict]], kind_by_tid: dict[str, str],
                tau: float | None, a_norm_bytes: float,
                ref_min_content: int = 0, typ_min_refs: int = 2) -> Anchors:
    """μ per turn from the leave-one-out values; σ = sqrt(mean within-turn
    variance) pooled per dialect over the turns that have all cross echoes.
    wvk 24: references with < ref_min_content content tokens are left out of
    the Mc anchor; fewer than typ_min_refs content-bearing references → no
    Mc anchor for the turn (typicality leg dropped there)."""
    out = Anchors()
    var_by: dict[str, dict[str, list[float]]] = {}
    mu_by: dict[str, dict[str, list[float]]] = {}
    for tid, refs in turn_refs.items():
        t = ref_loo_terms(refs, tau, a_norm_bytes, ref_min_content)
        if t is None:
            continue
        kind = kind_by_tid.get(tid) or dialects.DEFAULT_KIND
        mu = {}
        for leg in ("R", "A", "Mc"):
            vals = [v for v in t[leg] if v is not None and math.isfinite(v)]
            need = max(2, typ_min_refs) if leg == "Mc" else 2
            if len(vals) >= need:
                mu[leg] = st.mean(vals)
                var_by.setdefault(kind, {}).setdefault(leg, []).append(st.variance(vals))
                mu_by.setdefault(kind, {}).setdefault(leg, []).append(mu[leg])
        if mu:
            out.mu[tid] = mu
            out.n_turns[kind] = out.n_turns.get(kind, 0) + 1
    for kind, legs in var_by.items():
        out.sigma[kind] = {leg: math.sqrt(st.mean(v)) if v else None
                           for leg, v in legs.items()}
        out.mu_mean[kind] = {leg: st.mean(v) if v else None
                             for leg, v in mu_by.get(kind, {}).items()}
    return out


def frozen_table(cfg: dict) -> dict[str, dict[str, float | None]]:
    """``[duel.sd_meter.frozen.<dialect>]`` → dialect -> {R_mu, R_sigma, A_mu, A_sigma, Mc_mu, Mc_sigma}."""
    out = {}
    for kind, d in (cfg.get("frozen") or {}).items():
        out[str(kind)] = {k: (float(v) if v is not None else None)
                          for k, v in dict(d).items()}
    return out


# -- turn score -----------------------------------------------------------------------

def _z(x: float | None, mu: float | None, sigma: float | None) -> float | None:
    if x is None or mu is None or not sigma or not math.isfinite(x):
        return None
    return (x - mu) / sigma


def turn_score(legs: dict | None, mu: dict | None, sigma: dict | None,
               cfg: dict) -> dict:
    """One side's sd-meter turn score with the binding leg.

    legs: side_legs() output (None = forfeit). mu/sigma: {R, A, Mc} for the
    turn / dialect (a missing entry drops that leg from the min and is
    counted). Returns {score, bind, z_R, typ_c, z_A, dropped}."""
    floor = cfg["forfeit_sd"]
    if legs is None:
        return {"score": floor, "bind": "forfeit", "z_R": None, "typ_c": None,
                "z_A": None, "dropped": []}
    mu = mu or {}
    sigma = sigma or {}
    zr = _z(legs["R"], mu.get("R"), sigma.get("R"))
    za = _z(legs["A"], mu.get("A"), sigma.get("A"))
    typ = None
    n_c = legs.get("n_content")
    if n_c is not None and n_c < cfg["content_min_tokens"]:
        typ = floor
    else:
        d = _z(legs.get("mc"), mu.get("Mc"), sigma.get("Mc"))
        if d is not None:
            typ = cfg["typicality_width"] - abs(d)
    cands = {"R": zr, "Gc": typ, "A": za}
    dropped = [k for k, v in cands.items() if v is None]
    live = {k: v for k, v in cands.items() if v is not None}
    if not live:
        return {"score": None, "bind": None, "z_R": zr, "typ_c": typ, "z_A": za,
                "dropped": dropped}
    bind = min(live, key=live.get)
    return {"score": live[bind], "bind": bind, "z_R": zr, "typ_c": typ,
            "z_A": za, "dropped": dropped}


# -- duel -----------------------------------------------------------------------------------

def _mean(vals):
    vals = [v for v in vals if v is not None and math.isfinite(v)]
    return st.mean(vals) if vals else None


def _side_summary(scores: dict[str, dict], legs: dict[str, dict | None]) -> dict:
    valid = [s for s in scores.values() if s["bind"] != "forfeit" and s["score"] is not None]
    n_valid = len(valid)
    binds = {k: sum(1 for s in valid if s["bind"] == k) / n_valid if n_valid else None
             for k in ("R", "Gc", "A")}
    all_scores = [s["score"] for s in scores.values() if s["score"] is not None]
    content = [l for l in legs.values() if l and l.get("n_tokens")]
    return {
        "mean": _mean(all_scores),
        "mean_valid": _mean([s["score"] for s in valid]),
        "sd_valid": (st.stdev([s["score"] for s in valid]) if n_valid >= 2 else None),
        "n_turns": len(scores),
        "n_valid": n_valid,
        "n_forfeits": sum(1 for s in scores.values() if s["bind"] == "forfeit"),
        "n_unscorable": sum(1 for s in scores.values() if s["score"] is None),
        "mean_z_R": _mean([s["z_R"] for s in valid]),
        "mean_typ_c": _mean([s["typ_c"] for s in valid]),
        "mean_z_A": _mean([s["z_A"] for s in valid]),
        "bind_frac": binds,
        "n_leg_dropped": {k: sum(1 for s in valid if k in s["dropped"]) for k in ("R", "Gc", "A")},
        "n_content_floor": sum(1 for l in content
                               if l.get("n_content") is not None and l["n_content"] < 1),
        "mean_content_share": _mean([l["n_content"] / l["n_tokens"] for l in content
                                     if l.get("n_content") is not None]),
        "mean_R": _mean([l["R"] for l in legs.values() if l]),
        "mean_A": _mean([l["A"] for l in legs.values() if l]),
        "mean_mc": _mean([l["mc"] for l in legs.values() if l]),
    }


def _paired(c_scores: dict[str, dict], k_scores: dict[str, dict], cfg: dict) -> dict:
    diffs = []
    n_forfeit = 0
    for tid in sorted(set(c_scores) & set(k_scores)):
        c, k = c_scores[tid], k_scores[tid]
        if c["score"] is None or k["score"] is None:
            continue
        if c["bind"] == "forfeit" and k["bind"] == "forfeit":
            diffs.append(0.0)      # two-sided forfeit: tie, kept in n (live rule)
            n_forfeit += 1
            continue
        if c["bind"] == "forfeit" or k["bind"] == "forfeit":
            n_forfeit += 1
        diffs.append(c["score"] - k["score"])
    n = len(diffs)
    if n < 2:
        return {"n_paired_turns": n, "n_forfeit_turns": n_forfeit, "margin": None,
                "se": None, "z": None, "sd_diff": None, "rule_passes": False}
    mean = st.mean(diffs)
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    z = mean / se if se > 0 else (math.inf if mean > 0 else 0.0)
    bar = max(cfg["k_sigma"] * se, cfg["min_margin_sd"])
    return {"n_paired_turns": n, "n_forfeit_turns": n_forfeit, "margin": mean,
            "se": se, "z": z if math.isfinite(z) else None, "sd_diff": sd,
            "bar": bar, "rule_passes": mean > bar}


def live_floor_in_sd(rows: list[dict], live_turn_score, forfeit_turn_score: float | None) -> dict:
    """Where the LIVE forfeit floor sits in sd of the live valid turn scores
    (the calibration the forfeit_sd knob is set against)."""
    vals = []
    for r in rows:
        if is_forfeit(r):
            continue
        try:
            v = live_turn_score(r["pairs"])
        except Exception:
            continue
        if v is not None and math.isfinite(v):
            vals.append(v)
    if len(vals) < 2 or forfeit_turn_score is None:
        return {"n": len(vals), "floor_sd": None}
    mu, sd = st.mean(vals), st.stdev(vals)
    srt = sorted(vals)
    p1 = srt[max(0, int(0.01 * len(srt)) - 1)]
    return {"n": len(vals), "mean": mu, "sd": sd,
            "floor_sd": (forfeit_turn_score - mu) / sd if sd > 0 else None,
            "p1_sd": (p1 - mu) / sd if sd > 0 else None}


def shadow_verdict(chall_rows: list[dict], king_rows: list[dict],
                   turn_refs: dict[str, list[dict]], kind_by_tid: dict[str, str],
                   tau: float | None, cfg: dict, *,
                   live_gates_pass: bool | None = None,
                   live_turn_score=None,
                   forfeit_turn_score: float | None = None) -> dict:
    """The whole ``shadow.sd_meter`` block for one duel."""
    a_norm = cfg["a_norm_bytes"]
    c_legs = {r["turn_id"]: side_legs(r, tau, a_norm) for r in chall_rows}
    k_legs = {r["turn_id"]: side_legs(r, tau, a_norm) for r in king_rows}
    loo = loo_anchors(turn_refs, kind_by_tid, tau, a_norm,
                      cfg["ref_min_content"], cfg["typ_min_refs"])
    frozen = frozen_table(cfg)

    def anchors_for(mode: str, tid: str) -> tuple[dict | None, dict | None]:
        kind = kind_by_tid.get(tid) or dialects.DEFAULT_KIND
        if mode == "loo":
            return loo.mu.get(tid), loo.sigma.get(kind)
        f = frozen.get(kind) or {}
        mu = {leg: f.get(f"{leg}_mu") for leg in ("R", "A", "Mc")}
        sig = {leg: f.get(f"{leg}_sigma") for leg in ("R", "A", "Mc")}
        # Content-G constants are slice-pooled when the table has none.
        if sig.get("Mc") is None:
            sig["Mc"] = (loo.sigma.get(kind) or {}).get("Mc")
        if mu.get("Mc") is None and tid in loo.mu:
            mu["Mc"] = loo.mu[tid].get("Mc")
        return mu, sig

    def teacher_scores(mode: str) -> dict[str, dict]:
        """Positive control: each ref j scored as a miner against the other
        k−1 refs (its own value left out of μ), averaged over j per turn."""
        out = {}
        for tid, refs in turn_refs.items():
            t = ref_loo_terms(refs, tau, a_norm, cfg["ref_min_content"])
            if t is None:
                continue
            _, sig = anchors_for(mode, tid)
            per = []
            for j in range(len(refs)):
                legs = {"R": t["R"][j], "A": t["A"][j], "mc": t["Mc"][j],
                        "n_content": refs[j].get("n_content_thought"),
                        "n_tokens": refs[j].get("n_tokens_thought")}
                if mode == "loo":
                    others = {leg: [v for i, v in enumerate(t[leg]) if i != j and v is not None]
                              for leg in ("R", "A", "Mc")}
                    mu = {leg: st.mean(v) for leg, v in others.items() if v}
                else:
                    mu, _ = anchors_for(mode, tid)
                s = turn_score(legs, mu, sig, cfg)
                if s["score"] is not None:
                    per.append(s)
            if per:
                out[tid] = {"score": st.mean(p["score"] for p in per),
                            "bind": max(("R", "Gc", "A"), key=lambda b: sum(1 for p in per if p["bind"] == b)),
                            "z_R": _mean([p["z_R"] for p in per]),
                            "typ_c": _mean([p["typ_c"] for p in per]),
                            "z_A": _mean([p["z_A"] for p in per]), "dropped": []}
        return out

    def kmatched_control(sig_of) -> dict:
        """wvk 24 control: the king scored the same way as the held-out teacher
        reference — against the mean of k−1 references, averaged over the
        left-out j — with forfeits and content-floor turns of the king dropped
        (the teacher never forfeits, so the floor only ever enters one side).
        Overall = min over legs; per-leg = each standardised leg alone."""
        out = {leg: [] for leg in ("all", "R", "Gc", "A")}
        for tid, refs in turn_refs.items():
            t = ref_loo_terms(refs, tau, a_norm, cfg["ref_min_content"])
            kl = k_legs.get(tid)
            if t is None or kl is None:
                continue
            sig = sig_of(tid)
            if kl.get("n_content") is not None and kl["n_content"] < cfg["content_min_tokens"]:
                continue   # floor-dropped: the king's content-floor turns are excluded
            k = len(refs)
            per_t, per_k = {leg: [] for leg in ("all", "R", "Gc", "A")}, {leg: [] for leg in ("all", "R", "Gc", "A")}
            for j in range(k):
                others = {leg: [v for i, v in enumerate(t[leg]) if i != j and v is not None] for leg in ("R", "A", "Mc")}
                mu_j = {leg: st.mean(v) for leg, v in others.items() if v}
                tj = turn_score({"R": t["R"][j], "A": t["A"][j], "mc": t["Mc"][j],
                                 "n_content": refs[j].get("n_content_thought"), "n_tokens": None}, mu_j, sig, cfg)
                kj = turn_score(kl, mu_j, sig, cfg)          # king vs the SAME k−1 anchor
                if tj["score"] is None or kj["score"] is None or kj["bind"] == "forfeit":
                    continue
                for leg, key in (("R", "z_R"), ("Gc", "typ_c"), ("A", "z_A")):
                    if tj.get(key) is not None and kj.get(key) is not None:
                        per_t[leg].append(tj[key]); per_k[leg].append(kj[key])
                per_t["all"].append(tj["score"]); per_k["all"].append(kj["score"])
            for leg in out:
                if per_t[leg]:
                    out[leg].append(st.mean(per_t[leg]) - st.mean(per_k[leg]))
        res = {}
        for leg, d in out.items():
            if len(d) >= 2:
                m = st.mean(d); se = st.stdev(d) / math.sqrt(len(d))
                res[leg] = {"margin": m, "se": se, "z": (m / se) if se > 0 else None, "n": len(d)}
            else:
                res[leg] = {"margin": None, "se": None, "z": None, "n": len(d)}
        return res

    def matched_control(sig_of) -> dict:
        """Fully matched control (2026-09-25): for each turn and each left-out
        reference j, the held-out reference AND the king are both scored over
        the same k−1 references — the king's R and A recomputed as LMEs over
        those k−1 pairs (the k-matched form kept the king's k-reference LMEs,
        which are larger by construction) — against the mean of those k−1
        references; king forfeits / content-floor turns dropped. Overall =
        min over legs; per leg = each standardised leg alone. This is the
        rollback signal from 2026-09-25 on."""
        out = {leg: [] for leg in ("all", "R", "Gc", "A")}
        for tid, refs in turn_refs.items():
            t = ref_loo_terms(refs, tau, a_norm, cfg["ref_min_content"])
            krow = k_rows_by.get(tid)
            if t is None or krow is None or is_forfeit(krow):
                continue
            sig = sig_of(tid)
            k = len(refs)
            per_t, per_k = {leg: [] for leg in out}, {leg: [] for leg in out}
            for j in range(k):
                oth = [i for i in range(k) if i != j]
                kl = side_legs_subset(krow, tau, a_norm, oth)
                if kl is None or (kl.get("n_content") is not None and kl["n_content"] < cfg["content_min_tokens"]):
                    continue
                others = {leg: [v for i, v in enumerate(t[leg]) if i != j and v is not None] for leg in ("R", "A", "Mc")}
                mu_j = {leg: st.mean(v) for leg, v in others.items() if v}
                tj = turn_score({"R": t["R"][j], "A": t["A"][j], "mc": t["Mc"][j],
                                 "n_content": refs[j].get("n_content_thought"), "n_tokens": None}, mu_j, sig, cfg)
                kj = turn_score(kl, mu_j, sig, cfg)
                if tj["score"] is None or kj["score"] is None:
                    continue
                for leg, key in (("R", "z_R"), ("Gc", "typ_c"), ("A", "z_A")):
                    if tj.get(key) is not None and kj.get(key) is not None:
                        per_t[leg].append(tj[key]); per_k[leg].append(kj[key])
                per_t["all"].append(tj["score"]); per_k["all"].append(kj["score"])
            for leg in out:
                if per_t[leg]:
                    out[leg].append(st.mean(per_t[leg]) - st.mean(per_k[leg]))
        res = {}
        for leg, d in out.items():
            if len(d) >= 2:
                m = st.mean(d); se = st.stdev(d) / math.sqrt(len(d))
                res[leg] = {"margin": m, "se": se, "z": (m / se) if se > 0 else None, "n": len(d)}
            else:
                res[leg] = {"margin": None, "se": None, "z": None, "n": len(d)}
        return res

    k_rows_by = {r["turn_id"]: r for r in king_rows}
    by_anchor = {}
    for mode in ("loo", "frozen"):
        if mode == "frozen" and not frozen:
            by_anchor[mode] = {"available": False, "reason": "no [duel.sd_meter.frozen] table"}
            continue
        if mode == "loo" and not loo.mu:
            by_anchor[mode] = {"available": False, "reason": "no cross echoes (lp_cross)"}
            continue
        c_scores = {tid: turn_score(l, *anchors_for(mode, tid), cfg) for tid, l in c_legs.items()}
        k_scores = {tid: turn_score(l, *anchors_for(mode, tid), cfg) for tid, l in k_legs.items()}
        paired = _paired(c_scores, k_scores, cfg)
        would = paired["rule_passes"] and (live_gates_pass is not False)
        t_scores = teacher_scores(mode)
        t_vs_k = _paired(t_scores, {t: s for t, s in k_scores.items() if t in t_scores}, cfg)
        by_anchor[mode] = {
            "available": True,
            "challenger": _side_summary(c_scores, c_legs),
            "king": _side_summary(k_scores, k_legs),
            # Positive control: the teacher's own held-out replies vs the king.
            "teacher": _side_summary(t_scores, {}),
            # legacy construction (wvk 22/23): LOO teacher vs the king scored against all k refs
            "teacher_vs_king": {k: t_vs_k.get(k) for k in ("margin", "se", "z", "n_paired_turns")},
            # wvk 24 control: k-matched anchors, king forfeits / content-floor turns dropped
            "control_kmatched": kmatched_control(lambda tid: anchors_for(mode, tid)[1]),
            # 2026-09-25 fully matched control (2-ref both sides) — the rollback signal
            "control_matched": matched_control(lambda tid: anchors_for(mode, tid)[1]),
            **paired,
            "would_crown": would,
            "would_crown_rule_only": paired["rule_passes"],
        }
    out = {
        "formula": ("turn = min(z_R, typ_c, z_A); z_R = (R − μ_R)/σ_R; "
                    "z_A = (A_sum − μ_A)/σ_A, A_sum = τ·log mean_i exp(b_i/τ), "
                    f"b_i = Σ_bytes[lpC(y_A|z_C^i) − lpC(y_A|∅)] / {a_norm:g}; "
                    f"typ_c = {cfg['typicality_width']:g} − |m_c − μ_c|/σ_c, m_c = mean lpC(tok|x) over "
                    f"tokens with |lpC(tok|x) − lpC(tok|∅)| > {cfg['content_lift_nats']:g} nat "
                    f"(< {cfg['content_min_tokens']} such tokens → typ_c = floor"
                    + ("; only the first K content tokens of the miner's thought are scored, "
                       "K = max_i n_content(z_C^i)" if cfg["content_prefix"] == "refs_max" else "")
                    + (f"; references with < {cfg['ref_min_content']} content tokens do not anchor "
                       f"typicality and a turn with < {cfg['typ_min_refs']} content-bearing references "
                       "scores min(z_R, z_A)" if cfg["ref_min_content"] else "")
                    + "); "
                    f"forfeit = {cfg['forfeit_sd']:g} sd; μ per turn = mean of the k refs' "
                    "leave-one-out values (anchor loo) or per-dialect constants (anchor frozen); "
                    "σ = pooled within-turn sd of the refs per dialect; crown iff paired mean > "
                    f"max({cfg['k_sigma']:g}·SE, {cfg['min_margin_sd']:g})"),
        "anchor": cfg["anchor"],
        "knobs": {k: cfg[k] for k in ("content_lift_nats", "content_min_tokens",
                                       "typicality_width", "a_norm_bytes", "forfeit_sd",
                                       "k_sigma", "min_margin_sd", "cross_echo",
                                       "content_prefix", "ref_min_content", "typ_min_refs")},
        "tau": tau,
        "sigma_by_dialect": loo.sigma,
        "mu_mean_by_dialect": loo.mu_mean,
        "n_loo_turns_by_dialect": loo.n_turns,
        "frozen_dialects": sorted(frozen),
        "by_anchor": by_anchor,
        "live_gates_pass": live_gates_pass,
    }
    head = by_anchor.get(cfg["anchor"]) or {}
    for k in ("margin", "se", "z", "sd_diff", "n_paired_turns", "n_forfeit_turns",
              "would_crown", "would_crown_rule_only"):
        out[k] = head.get(k)
    out["challenger"] = head.get("challenger")
    out["king"] = head.get("king")
    if live_turn_score is not None:
        out["live_floor_calibration"] = {
            "challenger": live_floor_in_sd(chall_rows, live_turn_score, forfeit_turn_score),
            "king": live_floor_in_sd(king_rows, live_turn_score, forfeit_turn_score),
        }
    return out


def cost_block(teacher_stats: dict[str, dict], duel_seconds: float | None) -> dict:
    """Echo work per tag (base = live min(R,G) echoes, sd_meter = shadow-only
    echoes) plus the extra share; the wall-time comparison is duel_seconds
    against the previous verdicts on the same stack."""
    base = teacher_stats.get("base") or {}
    extra = teacher_stats.get("sd_meter") or {}

    def ratio(k):
        b = base.get(k) or 0
        return (extra.get(k, 0) / b) if b else None
    return {
        "by_tag": teacher_stats,
        "extra_ratio": {k: ratio(k) for k in ("requests", "prompt_tokens",
                                                "tail_tokens", "computed_tokens", "seconds")},
        "duel_seconds": duel_seconds,
    }
