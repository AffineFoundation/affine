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


def divergence_v2(comps: dict[str, float | None], corpus_means: dict[str, float],
                  weights: dict[str, float]) -> float:
    """Rule v2 divergence D_s: weighted mix of the king-vs-teacher distances,
    each divided by its corpus mean so no unit dominates. A component
    without a value (e.g. action divergence on `text` turns) is dropped and
    the remaining weights renormalised."""
    num = 0.0
    wsum = 0.0
    for k, w in weights.items():
        v = comps.get(k)
        mean = corpus_means.get(k) or 0.0
        if v is None or mean <= 0:
            continue
        num += w * max(float(v), 0.0) / mean
        wsum += w
    return num / wsum if wsum > 0 else 0.0


def weights_v2(strata: dict[str, dict], *, eps: float, gamma: float, field_d: str = "D_v2",
               out: str = "w_v2") -> None:
    """w_s = eps / N + (1 − eps) · D_s^gamma / Σ D^gamma over ALL strata: the
    uniform floor keeps every turn's draw probability non-zero (nothing can
    be forgotten); the rest follows divergence."""
    n = len(strata) or 1
    pw = {s: max(float(r.get(field_d) or 0.0), 0.0) ** gamma for s, r in strata.items()}
    tot = sum(pw.values())
    for s, r in strata.items():
        r[out] = eps / n + ((1.0 - eps) * pw[s] / tot if tot > 0 else (1.0 - eps) / n)


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
                 ct_groups: tuple[str, str] = ("coding", "terminal"),
                 block_floors: dict[str, tuple[tuple[str, ...], float]] | None = None) -> dict:
    """raw -> after_floor -> after_clamp, with a reason code per group.

    static: the [mix] table (floor reference); current: the live slice share
    (strata share of the live index; clamp reference). A group with no
    supply in the live index (current == 0 and raw == 0) stays at 0.
    Block floors: `coding + terminal >= floor_ct` (always) plus any named
    block in `block_floors` (phase 10: the stop-state groups >= 0.25) --
    when a block sums under its floor, its members' lower bounds are raised
    proportionally so the block lands on the floor; the same bounds hold
    through the clamp stage, so the published vector satisfies them."""
    keys = sorted(set(raw) | set(static) | set(current))
    raw = {g: float(raw.get(g, 0.0)) for g in keys}
    static = {g: float(static.get(g, 0.0)) for g in keys}
    current = {g: float(current.get(g, 0.0)) for g in keys}
    supply = {g: (raw[g] > 0 or current[g] > 0) for g in keys}
    floor = {g: (floor_frac * static[g] if supply[g] else 0.0) for g in keys}
    lo_f = dict(floor)
    hi_f = {g: (cap if supply[g] else 0.0) for g in keys}
    blocks = {"coding_terminal": (tuple(ct_groups), float(floor_ct))}
    blocks.update({k: (tuple(v[0]), float(v[1])) for k, v in (block_floors or {}).items()})
    after_floor = constrained_fill(raw, lo_f, hi_f)
    block_hits: dict[str, bool] = {}
    for _ in range(4):
        changed = False
        for name, (members, bfloor) in blocks.items():
            live = [g for g in members if supply.get(g)]
            tot = sum(after_floor.get(g, 0.0) for g in live)
            if live and 0 < tot < bfloor - EPS_SUM:
                for g in live:
                    lo_f[g] = max(lo_f[g], after_floor[g] * bfloor / tot)
                block_hits[name] = True
                changed = True
        if not changed:
            break
        after_floor = constrained_fill(raw, lo_f, hi_f)
    joint = bool(block_hits.get("coding_terminal"))
    lo_c = {g: max(lo_f[g], current[g] - max_shift) if supply[g] else 0.0 for g in keys}
    hi_c = {g: min(hi_f[g], current[g] + max_shift) if supply[g] else 0.0 for g in keys}
    for g in keys:
        hi_c[g] = max(hi_c[g], lo_c[g])
    after_clamp = constrained_fill(after_floor, lo_c, hi_c)
    member_of = {g: name for name, (members, _) in blocks.items() for g in members if block_hits.get(name)}
    reasons: dict[str, str] = {}
    tol = 1e-9
    for g in keys:
        s = after_clamp[g]
        if not supply[g]:
            reasons[g] = "no_supply"
        elif abs(s - lo_c[g]) < tol and raw[g] < s - tol:
            # the floored target wanted lower: the lower bound is binding
            if lo_f[g] >= current[g] - max_shift - tol:
                reasons[g] = (("joint_floor" if member_of.get(g) == "coding_terminal" else f"block_floor:{member_of[g]}")
                              if g in member_of and lo_f[g] > floor[g] + tol else "floor")
            else:
                reasons[g] = "clamped_down"
        elif abs(s - hi_c[g]) < tol and raw[g] > s + tol:
            reasons[g] = "capped" if hi_f[g] <= current[g] + max_shift + tol else "clamped_up"
        elif abs(s - lo_f[g]) < tol and lo_f[g] > 0:
            reasons[g] = (("joint_floor" if member_of.get(g) == "coding_terminal" else f"block_floor:{member_of[g]}")
                          if g in member_of and lo_f[g] > floor[g] + tol else "floor")
        else:
            reasons[g] = "free"
        if reasons[g] == "free" and g in member_of and s > raw[g] + tol:
            # lifted by a raised block floor (the fill lands a hair above the bound)
            reasons[g] = "joint_floor" if member_of[g] == "coding_terminal" else f"block_floor:{member_of[g]}"
    block_sums = {name: {"members": list(members), "floor": bfloor,
                         "after_floor": sum(after_floor.get(g, 0.0) for g in members),
                         "after_clamp": sum(after_clamp.get(g, 0.0) for g in members),
                         "raised": bool(block_hits.get(name))}
                  for name, (members, bfloor) in blocks.items()}
    return {"raw": raw, "after_floor": after_floor, "after_clamp": after_clamp,
            "floor": floor, "current": current, "static": static, "reasons": reasons,
            "joint_floor_applied": joint, "blocks": block_sums,
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
            m_k = float(strata[k].get("m") or 1)
            # a stratum is drawn at most m_k times per duel (one turn per sub-stratum)
            per_turn = min(slots * m_k / m_tot, m_k) / max(1, int(strata[k].get("n_turns") or 1))
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


def recurrence_guard(strata: dict[str, dict], shares: dict[str, float], *, group_cap: float,
                     turn_cap: float, slice_n: int = SLICE_N, max_iter: int = 10) -> dict:
    """Coordinator amendment 2026-09-15 01:05 UTC: the recurrence cap is a
    hard safety item. If the counted vector would exceed it, lower the
    multiplicity m of the offending strata FIRST (3 -> 2 -> 1); only when
    every stratum of a group is at m = 1 and the group's expected draws per
    turn per duel still exceed group_cap is the group's share lowered to
    the cap level (share <= group_cap * n_turns / slice_n) and the freed
    mass spread over the other groups. Mutates `m` in place; returns the
    (possibly lowered) shares and a log of what it did."""
    actions: list[str] = []
    shares = dict(shares)
    for _ in range(max_iter):
        proj = recurrence_projection(strata, shares, slice_n=slice_n)
        changed = False
        # 1. single-turn cap: lower m on the strata over the cap
        for s, per_turn in sorted(proj["per_stratum"].items()):
            if per_turn > turn_cap + 1e-9 and int(strata[s].get("m") or 1) > 1:
                strata[s]["m"] = int(strata[s]["m"]) - 1
                actions.append(f"m {s} -> {strata[s]['m']} (turn draws {per_turn:.3f} > {turn_cap})")
                changed = True
        if changed:
            continue
        # 2. group cap, and single turns still over the cap at m = 1 (a 1-turn
        #    stratum drawn on most duels): only now lower the group's share
        over = {g: d for g, d in proj["groups"].items() if d["expected_draws_per_turn_per_duel"] > group_cap + 1e-9}
        for s, per_turn in proj["per_stratum"].items():
            if per_turn > turn_cap + 1e-9 and int(strata[s].get("m") or 1) <= 1:
                g = strata[s]["group"]
                d = proj["groups"][g]
                m_tot = sum(float(strata[k].get("m") or 1) for k, r in strata.items() if r["group"] == g) or 1.0
                # per_turn = slots / m_tot / n_turns <= turn_cap  ->  share <= turn_cap * n_turns * m_tot / slice_n
                ceiling = turn_cap * max(1, int(strata[s].get("n_turns") or 1)) * m_tot / slice_n
                if ceiling < shares.get(g, 0.0):
                    over[g] = {**d, "_ceiling": min(ceiling, d.get("_ceiling", 1.0))}
        if not over:
            break
        if len(over) == len(shares):
            actions.append("every group over the cap: nothing to move mass to (slice larger than D allows)")
            break
        for g, d in sorted(over.items()):
            ceiling = min(group_cap * d["n_turns"] / slice_n, d.get("_ceiling", 1.0))
            actions.append(f"share {g} {shares[g]:.4f} -> {ceiling:.4f} (group draws {d['expected_draws_per_turn_per_duel']:.3f} > {group_cap})")
            shares[g] = ceiling
        tot_fixed = sum(shares[g] for g in over)
        rest = [g for g in shares if g not in over]
        rest_tot = sum(shares[g] for g in rest) or 1.0
        for g in rest:
            shares[g] = shares[g] * (1.0 - tot_fixed) / rest_tot
    proj = recurrence_projection(strata, shares, slice_n=slice_n)
    cut = sorted({a.split()[1] for a in actions if a.startswith("share ")})
    return {"shares": shares, "actions": actions, "groups_cut": cut,
            "max_turn_draws_per_duel": proj["max_turn_draws_per_duel"],
            "max_group_draws": max((d["expected_draws_per_turn_per_duel"] for d in proj["groups"].values()), default=0.0),
            "ok": proj["max_turn_draws_per_duel"] <= turn_cap + 1e-12
                  and all(d["expected_draws_per_turn_per_duel"] <= group_cap + 1e-12 for d in proj["groups"].values())}


def restore_block_floors(shares: dict[str, float], block_floors: dict[str, tuple[tuple[str, ...], float]],
                         *, fixed: set[str] | frozenset[str], floor: dict[str, float], cap: float,
                         current: dict[str, float], max_shift: float) -> dict[str, float]:
    """After the recurrence guard cut a member's share, a block can sit a
    hair under its floor. Lift the block's OTHER members proportionally
    back to the floor (cut groups stay fixed), taking the mass from the
    groups outside the block, inside the usual per-group bounds."""
    shares = dict(shares)
    for members, bfloor in block_floors.values():
        live = [g for g in members if shares.get(g, 0.0) > 0]
        tot = sum(shares[g] for g in live)
        if not live or tot >= bfloor - EPS_SUM:
            continue
        movable = [g for g in live if g not in fixed]
        if not movable:
            continue
        need = bfloor - tot
        base = sum(shares[g] for g in movable) or 1.0
        lo = {g: shares[g] for g in shares}
        hi = {g: shares[g] for g in shares}
        for g in movable:
            lo[g] = hi[g] = shares[g] + need * shares[g] / base
        for g in shares:
            if g in fixed or g in members:
                continue
            lo[g] = max(floor.get(g, 0.0), current.get(g, 0.0) - max_shift) if shares[g] > 0 else 0.0
            hi[g] = max(lo[g], min(cap, current.get(g, 0.0) + max_shift)) if shares[g] > 0 else 0.0
        shares = constrained_fill(shares, lo, hi)
    return shares


def check_floors(shares: dict[str, float], static: dict[str, float], *, floor_frac: float,
                 floor_ct: float, cap: float, supply: dict[str, bool] | None = None,
                 guard_cut: set[str] | frozenset[str] = frozenset(), tol: float = 1e-6,
                 block_floors: dict[str, tuple[tuple[str, ...], float]] | None = None) -> dict:
    """Stage-3 item 6: floors and cap hold on a published vector. A group
    whose share the recurrence guard lowered (`guard_cut`) is exempt from
    its floor -- the recurrence cap is the harder of the two safety items
    (coordinator amendment 2026-09-15) -- and is listed separately."""
    supply = supply or {g: True for g in shares}
    bad_floor = [g for g in shares if supply.get(g, True) and static.get(g, 0) > 0
                 and g not in guard_cut and shares[g] < floor_frac * static[g] - tol]
    cut_below = [g for g in shares if g in guard_cut and static.get(g, 0) > 0
                 and shares[g] < floor_frac * static[g] - tol]
    bad_cap = [g for g in shares if shares[g] > cap + tol]
    ct = shares.get("coding", 0.0) + shares.get("terminal", 0.0)
    blocks = {}
    for name, (members, bfloor) in (block_floors or {}).items():
        tot = sum(shares.get(g, 0.0) for g in members)
        blocks[name] = {"sum": tot, "floor": bfloor, "ok": tot >= bfloor - tol}
    return {"ok": not bad_floor and not bad_cap and ct >= floor_ct - tol and all(b["ok"] for b in blocks.values()),
            "below_floor": bad_floor, "above_cap": bad_cap, "coding_plus_terminal": ct,
            "below_floor_by_recurrence_guard": cut_below, "block_floors": blocks, "sum": sum(shares.values())}


def is_close_sum_one(shares: dict[str, float], tol: float = 1e-6) -> bool:
    return math.isclose(sum(shares.values()), 1.0, abs_tol=tol)
