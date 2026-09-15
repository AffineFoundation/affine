"""Rule v1 of the adaptive curriculum -- pure functions, no I/O.

    w_s      = (M~_s + eps)^gamma * S~_s
    share_g  = sum_{s in g} w_s / sum_all w
    share_g  = max(share_g, floor_g)  ; coding + terminal >= floor_ct ; <= cap
    share_g  = clamp(share_g, current_g +- max_shift)          (then renormalise)
    m_s      = 1 + round(2 * rank_pct_within_group(w_s)), <= n_turns_s, <= m_max

M~ / S~ are the king's miss rate and the live-reference rate shrunk
stratum -> cell -> group -> corpus with prior count n_0. Everything here is
deterministic given its inputs; ties are broken by key order.
"""

from __future__ import annotations

import math

SLICE_N = 1300
CORPUS_PRIOR_M = 0.25      # theta is the 25th percentile by construction
EPS_SUM = 1e-12


# -- shrinkage ------------------------------------------------------------------
def shrink(n: float, value: float | None, prior: float, n0: float) -> float:
    """(n * value + n0 * prior) / (n + n0); the prior alone when n == 0."""
    if n <= 0 or value is None:
        return float(prior)
    return (n * float(value) + n0 * prior) / (n + n0)


def shrunk_rates(strata: dict[str, dict], cells: dict[str, dict], groups: dict[str, dict],
                 *, n0: float, corpus_m: float = CORPUS_PRIOR_M, corpus_s: float,
                 probe_s: dict[str, float] | None = None) -> None:
    """Fill `M_t` / `S_t` on every stratum record in place.

    strata[s]: {group, cell, n_w, M, S}; cells[c]: {group, n_w, M, S};
    groups[g]: {n_w, M, S}. Missing cells / groups fall back one level.
    `probe_s[s]`: teacher-probe yield used as the S prior of an UNSCORED
    stratum (n_w == 0) that has probe rows."""
    probe_s = probe_s or {}
    g_m: dict[str, float] = {}
    g_s: dict[str, float] = {}
    for g, rec in groups.items():
        g_m[g] = shrink(rec.get("n_w") or 0.0, rec.get("M"), corpus_m, n0)
        g_s[g] = shrink(rec.get("n_w") or 0.0, rec.get("S"), corpus_s, n0)
    c_m: dict[str, float] = {}
    c_s: dict[str, float] = {}
    for c, rec in cells.items():
        gm = g_m.get(rec.get("group"), corpus_m)
        gs = g_s.get(rec.get("group"), corpus_s)
        c_m[c] = shrink(rec.get("n_w") or 0.0, rec.get("M"), gm, n0)
        c_s[c] = shrink(rec.get("n_w") or 0.0, rec.get("S"), gs, n0)
    for s, rec in strata.items():
        n = rec.get("n_w") or 0.0
        pm = c_m.get(rec.get("cell"), g_m.get(rec.get("group"), corpus_m))
        ps = c_s.get(rec.get("cell"), g_s.get(rec.get("group"), corpus_s))
        if n <= 0 and s in probe_s:
            ps = float(probe_s[s])
        rec["M_t"] = shrink(n, rec.get("M"), pm, n0)
        rec["S_t"] = shrink(n, rec.get("S"), ps, n0)
        rec["prior_M"] = pm
        rec["prior_S"] = ps


def shrink_field(strata: dict[str, dict], cells: dict[str, dict], groups: dict[str, dict], *,
                 field: str, n_field: str, out: str, corpus_prior: float, n0: float) -> None:
    """Generic stratum -> cell -> group -> corpus shrinkage of `field`
    (observation mass in `n_field`), written to strata[s][out]."""
    g_v = {g: shrink(rec.get(n_field) or 0.0, rec.get(field), corpus_prior, n0) for g, rec in groups.items()}
    c_v = {c: shrink(rec.get(n_field) or 0.0, rec.get(field), g_v.get(rec.get("group"), corpus_prior), n0)
           for c, rec in cells.items()}
    for s, rec in strata.items():
        prior = c_v.get(rec.get("cell"), g_v.get(rec.get("group"), corpus_prior))
        rec[out] = shrink(rec.get(n_field) or 0.0, rec.get(field), prior, n0)


def stratum_weight_v12(f_t: float, dplus_t: float, s_t: float, *, dplus_scale: float, eps: float,
                       gamma: float, s_gate: float) -> tuple[float, float]:
    """Rule v1.2 (coordinator 2026-09-15 00:55 UTC; informational, candidate
    for the fold-3 apply): the deficit is what the meter CAN see on king
    failure states -- "the king cannot answer here" (forfeit rate) plus "a
    challenger can do better here" (mean positive paired gap, scaled so its
    corpus mean equals the corpus mean forfeit rate). S~ is a gate.
    Returns (M12~, w)."""
    m12 = max(f_t, 0.0) + dplus_scale * max(dplus_t, 0.0)
    return m12, ((m12 + eps) ** gamma if s_t >= s_gate else 0.0)


def stratum_weight(m_t: float, s_t: float, *, eps: float, gamma: float) -> float:
    return (max(m_t, 0.0) + eps) ** gamma * max(s_t, 0.0)


def stratum_weight_v11(m_t: float, s_t: float, *, eps: float, gamma: float, s_gate: float) -> float:
    """Rule v1.1 (informational, proposed 2026-09-15): S~ is a GATE, not a
    multiplier -- a stratum is eligible when its live share is at least
    s_gate, and then weighs the king's miss rate alone."""
    return (max(m_t, 0.0) + eps) ** gamma if s_t >= s_gate else 0.0


def shares_by_slice_key(strata: dict[str, dict], index_rows: list[dict], field: str) -> dict[str, float]:
    """Σ over slice keys of the bucket-mean of `field`, normalised per group."""
    from_key: dict[str, set[str]] = {}
    for r in index_rows:
        stratum = str(r.get("stratum") or "")
        src = r.get("stratum_src")
        from_key.setdefault(stratum, set()).add(str(src) if src else stratum)
    acc: dict[str, float] = {}
    for bases in from_key.values():
        vals = [float(strata[b][field]) for b in bases if b in strata]
        if not vals:
            continue
        g = strata[next(iter(b for b in bases if b in strata))]["group"]
        acc[g] = acc.get(g, 0.0) + sum(vals) / len(vals)
    return normalise(acc)


def group_means_by_slice_key(strata: dict[str, dict], index_rows: list[dict],
                             fields: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Per group: mean over its slice keys of the bucket-mean of each field."""
    from_key: dict[str, set[str]] = {}
    for r in index_rows:
        stratum = str(r.get("stratum") or "")
        src = r.get("stratum_src")
        from_key.setdefault(stratum, set()).add(str(src) if src else stratum)
    acc: dict[str, dict[str, list[float]]] = {}
    for bases in from_key.values():
        bs = [b for b in bases if b in strata]
        if not bs:
            continue
        g = strata[bs[0]]["group"]
        slot = acc.setdefault(g, {f: [] for f in fields})
        for f in fields:
            slot[f].append(sum(float(strata[b][f]) for b in bs) / len(bs))
    return {g: {f: (sum(v) / len(v) if v else 0.0) for f, v in d.items()} for g, d in acc.items()}


# -- group shares ---------------------------------------------------------------
def normalise(x: dict[str, float]) -> dict[str, float]:
    tot = sum(x.values())
    if tot <= 0:
        return {g: 0.0 for g in x}
    return {g: v / tot for g, v in x.items()}


def raw_group_shares(strata: dict[str, dict]) -> dict[str, float]:
    acc: dict[str, float] = {}
    for rec in strata.values():
        acc[rec["group"]] = acc.get(rec["group"], 0.0) + float(rec["w"])
    return normalise(acc)


def raw_group_shares_by_slice_key(strata: dict[str, dict], index_rows: list[dict]) -> dict[str, float]:
    """Σ w over slice keys (what the sampler draws), a merged bucket weighing
    the mean w of the base strata inside it. Diagnostic next to
    raw_group_shares (the plan's Σ over base strata)."""
    from_key: dict[str, set[str]] = {}
    for r in index_rows:
        stratum = str(r.get("stratum") or "")
        src = r.get("stratum_src")
        base = str(src) if src else stratum
        from_key.setdefault(stratum, set()).add(base)
    acc: dict[str, float] = {}
    for key, bases in from_key.items():
        ws = [float(strata[b]["w"]) for b in bases if b in strata]
        if not ws:
            continue
        g = strata[next(iter(b for b in bases if b in strata))]["group"]
        acc[g] = acc.get(g, 0.0) + sum(ws) / len(ws)
    return normalise(acc)


def constrained_fill(target: dict[str, float], lo: dict[str, float], hi: dict[str, float],
                     max_iter: int = 200) -> dict[str, float]:
    """The share vector closest (proportionally) to `target` with
    lo <= s <= hi and sum s = 1. Water-filling: clip, spread the remaining
    gap over the groups that can still move, repeat."""
    keys = sorted(target)
    s = {g: min(max(target[g], lo[g]), hi[g]) for g in keys}
    for _ in range(max_iter):
        gap = 1.0 - sum(s.values())
        if abs(gap) < EPS_SUM:
            break
        if gap > 0:
            movable = [g for g in keys if s[g] < hi[g] - EPS_SUM]
        else:
            movable = [g for g in keys if s[g] > lo[g] + EPS_SUM]
        if not movable:
            break
        base = {g: (target[g] if target[g] > 0 else 1e-9) for g in movable}
        tot = sum(base.values())
        for g in movable:
            s[g] = min(max(s[g] + gap * base[g] / tot, lo[g]), hi[g])
    return s


def group_vector(raw: dict[str, float], static: dict[str, float], current: dict[str, float],
                 *, floor_frac: float, floor_ct: float, cap: float, max_shift: float,
                 ct_groups: tuple[str, str] = ("coding", "terminal")) -> dict:
    """raw -> after_floor -> after_clamp, with a reason code per group.

    static: the [mix] table (floor reference); current: the live slice share
    (strata share of the live index; clamp reference). A group with no
    supply in the live index (current == 0 and raw == 0) stays at 0."""
    keys = sorted(set(raw) | set(static) | set(current))
    raw = {g: float(raw.get(g, 0.0)) for g in keys}
    static = {g: float(static.get(g, 0.0)) for g in keys}
    current = {g: float(current.get(g, 0.0)) for g in keys}
    supply = {g: (raw[g] > 0 or current[g] > 0) for g in keys}
    floor = {g: (floor_frac * static[g] if supply[g] else 0.0) for g in keys}
    lo_f = dict(floor)
    hi_f = {g: (cap if supply[g] else 0.0) for g in keys}
    after_floor = constrained_fill(raw, lo_f, hi_f)
    joint = False
    ct = sum(after_floor.get(g, 0.0) for g in ct_groups if supply.get(g))
    if 0 < ct < floor_ct:
        # raise the two floors proportionally so the pair sums to floor_ct
        for g in ct_groups:
            if supply.get(g):
                lo_f[g] = max(lo_f[g], after_floor[g] * floor_ct / ct)
        after_floor = constrained_fill(raw, lo_f, hi_f)
        joint = True
    lo_c = {g: max(lo_f[g], current[g] - max_shift) if supply[g] else 0.0 for g in keys}
    hi_c = {g: min(hi_f[g], current[g] + max_shift) if supply[g] else 0.0 for g in keys}
    for g in keys:
        hi_c[g] = max(hi_c[g], lo_c[g])
    after_clamp = constrained_fill(after_floor, lo_c, hi_c)
    reasons: dict[str, str] = {}
    tol = 1e-9
    for g in keys:
        s = after_clamp[g]
        if not supply[g]:
            reasons[g] = "no_supply"
        elif abs(s - lo_c[g]) < tol and raw[g] < s - tol:
            # the floored target wanted lower: the lower bound is binding
            if lo_f[g] >= current[g] - max_shift - tol:
                reasons[g] = "joint_floor" if (joint and g in ct_groups and lo_f[g] > floor[g] + tol) else "floor"
            else:
                reasons[g] = "clamped_down"
        elif abs(s - hi_c[g]) < tol and raw[g] > s + tol:
            reasons[g] = "capped" if hi_f[g] <= current[g] + max_shift + tol else "clamped_up"
        elif abs(s - lo_f[g]) < tol and lo_f[g] > 0:
            reasons[g] = "joint_floor" if (joint and g in ct_groups and lo_f[g] > floor[g] + tol) else "floor"
        else:
            reasons[g] = "free"
    return {"raw": raw, "after_floor": after_floor, "after_clamp": after_clamp,
            "floor": floor, "current": current, "static": static, "reasons": reasons,
            "joint_floor_applied": joint,
            "bounds": {"lo": lo_c, "hi": hi_c}}


# -- multiplicity ---------------------------------------------------------------
def multiplicity(strata: dict[str, dict], *, m_max: int, field: str = "w") -> None:
    """Fill `rank_pct` and `m` per stratum: 1 + round(2 * rank_pct) inside
    the group (rank by `field`), at most n_turns and m_max. Ties broken by
    key order."""
    by_group: dict[str, list[str]] = {}
    for s, rec in strata.items():
        by_group.setdefault(rec["group"], []).append(s)
    for g, keys in by_group.items():
        keys.sort(key=lambda k: (float(strata[k][field]), k))
        n = len(keys)
        for i, k in enumerate(keys):
            pct = 0.5 if n == 1 else i / (n - 1)
            m = 1 + int(round(2 * pct))
            m = min(m, m_max, max(1, int(strata[k].get("n_turns") or 1)))
            strata[k]["rank_pct"] = pct
            strata[k]["m"] = m


# -- recurrence projection ------------------------------------------------------
def recurrence_projection(strata: dict[str, dict], shares: dict[str, float], *,
                          slice_n: int = SLICE_N) -> dict:
    """Expected draws per turn per duel under a group share vector: a group
    receives slice_n * share_g slots spread over its strata in proportion to
    m_s; a stratum's slots are spread over its turns."""
    by_group: dict[str, list[str]] = {}
    for s, rec in strata.items():
        by_group.setdefault(rec["group"], []).append(s)
    groups_out: dict[str, dict] = {}
    max_turn = (0.0, "")
    per_stratum: dict[str, float] = {}
    for g, keys in sorted(by_group.items()):
        share = float(shares.get(g, 0.0))
        slots = slice_n * share
        m_tot = sum(float(strata[k].get("m") or 1) for k in keys) or 1.0
        turns = sum(int(strata[k].get("n_turns") or 1) for k in keys) or 1
        worst = (0.0, "")
        for k in keys:
            per_turn = slots * float(strata[k].get("m") or 1) / m_tot / max(1, int(strata[k].get("n_turns") or 1))
            per_stratum[k] = per_turn
            if per_turn > worst[0]:
                worst = (per_turn, k)
        groups_out[g] = {"share": share, "slots_per_duel": slots, "n_strata": len(keys),
                         "n_turns": turns, "expected_draws_per_turn_per_duel": slots / turns,
                         "max_turn_draws_per_duel": worst[0], "max_turn_stratum": worst[1]}
        if worst[0] > max_turn[0]:
            max_turn = worst
    return {"groups": groups_out, "max_turn_draws_per_duel": max_turn[0],
            "max_turn_stratum": max_turn[1], "per_stratum": per_stratum}


def check_floors(shares: dict[str, float], static: dict[str, float], *, floor_frac: float,
                 floor_ct: float, cap: float, supply: dict[str, bool] | None = None,
                 tol: float = 1e-6) -> dict:
    """Stage-3 item 6: floors and cap hold on a published vector."""
    supply = supply or {g: True for g in shares}
    bad_floor = [g for g in shares if supply.get(g, True) and static.get(g, 0) > 0
                 and shares[g] < floor_frac * static[g] - tol]
    bad_cap = [g for g in shares if shares[g] > cap + tol]
    ct = shares.get("coding", 0.0) + shares.get("terminal", 0.0)
    return {"ok": not bad_floor and not bad_cap and ct >= floor_ct - tol,
            "below_floor": bad_floor, "above_cap": bad_cap, "coding_plus_terminal": ct,
            "sum": sum(shares.values())}


def is_close_sum_one(shares: dict[str, float], tol: float = 1e-6) -> bool:
    return math.isclose(sum(shares.values()), 1.0, abs_tol=tol)
