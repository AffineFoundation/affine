"""Wire A_match telemetry into evalsrv/dueling.py + the dashboard allowlist
(+ website duel card). Additive: no scoring change. Idempotent."""

from pathlib import Path

REPO = Path("/home/const/subnet120")


def sub(path: Path, old: str, new: str, marker: str) -> None:
    s = path.read_text()
    if marker in s:
        print(f"{path.name}: already has {marker!r}")
        return
    if s.count(old) != 1:
        raise SystemExit(f"{path.name}: anchor not unique/missing: {old[:70]!r}")
    path.write_text(s.replace(old, new))
    print(f"{path.name}: patched ({marker})")


du = REPO / "affine/evalsrv/dueling.py"
sub(du, "from .protocol_probe import probe_settings, rejection_detail, run_probe\n",
    "from . import amatch\nfrom .protocol_probe import probe_settings, rejection_detail, run_probe\n",
    "from . import amatch")
sub(du, '''        t.update({"turn_id": tid, "miner": miner.cfg.name})
        rows.append(t)
''', '''        t.update({"turn_id": tid, "miner": miner.cfg.name})
        # A_match telemetry (2026-09-14, not scored): does the side's
        # action literally match the teacher's reference actions, and how
        # often do the refs agree with each other. Forfeit rows (no
        # parseable action) carry None.
        y_side = miner_rollouts[0][1] if (t.get("valid") and miner_rollouts) else None
        t["a_match"], t["ref_pair"] = amatch.turn_agreement(
            y_side, [r["y"] for r in ref], action_kind)
        rows.append(t)
''', 'amatch.turn_agreement(')
sub(du, '''    if score_mode in ("min_rg", "min_rga"):
        # Which-leg-binds telemetry (post-fork watch item): g_bind_frac
        # near 1.0 means grounding is the binding constraint for this side.
        out["mean_r_leg"] = s.mean_r_leg
        out["mean_g_leg"] = s.mean_g_leg
        out["g_bind_frac"] = s.g_bind_frac
    if score_mode == "min_rga":
        out["mean_a_leg"] = s.mean_a_leg
        out["a_bind_frac"] = s.a_bind_frac
    return out
''', '''    if score_mode in ("min_rg", "min_rga"):
        # Which-leg-binds telemetry (post-fork watch item): g_bind_frac
        # near 1.0 means grounding is the binding constraint for this side.
        out["mean_r_leg"] = s.mean_r_leg
        out["mean_g_leg"] = s.mean_g_leg
        out["g_bind_frac"] = s.g_bind_frac
    if score_mode == "min_rga":
        out["mean_a_leg"] = s.mean_a_leg
        out["a_bind_frac"] = s.a_bind_frac
    # A_match telemetry (2026-09-14): share of reference actions equal to
    # the side's action, and the same minus the refs' own agreement.
    out.update(amatch.summarize(rows))
    return out
''', 'out.update(amatch.summarize(rows))')
sub(du, '''            "mean_r_leg": s.mean_r_leg, "mean_g_leg": s.mean_g_leg,
            "g_bind_frac": s.g_bind_frac,
        }
        if score_mode == "min_rga":
            out[kind]["mean_a_leg"] = s.mean_a_leg
            out[kind]["a_bind_frac"] = s.a_bind_frac
    return out
''', '''            "mean_r_leg": s.mean_r_leg, "mean_g_leg": s.mean_g_leg,
            "g_bind_frac": s.g_bind_frac,
        }
        if score_mode == "min_rga":
            out[kind]["mean_a_leg"] = s.mean_a_leg
            out[kind]["a_bind_frac"] = s.a_bind_frac
        out[kind].update(amatch.summarize(grp))
    return out
''', 'out[kind].update(amatch.summarize(grp))')
sub(du, '''    for rec in turns:
        kind = rec.get("action_kind") or dialects.DEFAULT_KIND
        d = out.setdefault(kind, {"n_turns": 0, "zero_ref_turns": 0, "_refs": 0})
        n = len(refs_used.get(turn_id(rec)) or [])
        d["n_turns"] += 1
        d["zero_ref_turns"] += (n == 0)
        d["_refs"] += n
    for d in out.values():
        d["mean_refs"] = d.pop("_refs") / d["n_turns"] if d["n_turns"] else None
    return out
''', '''    for rec in turns:
        kind = rec.get("action_kind") or dialects.DEFAULT_KIND
        d = out.setdefault(kind, {"n_turns": 0, "zero_ref_turns": 0, "_refs": 0, "_pairs": []})
        refs = refs_used.get(turn_id(rec)) or []
        n = len(refs)
        d["n_turns"] += 1
        d["zero_ref_turns"] += (n == 0)
        d["_refs"] += n
        # Teacher self-agreement on this turn's reference actions (A_match
        # telemetry, 2026-09-14): None below two normalisable refs / `text`.
        _, pair = amatch.turn_agreement(None, [r["y"] for r in refs], kind)
        if pair is not None:
            d["_pairs"].append(pair)
    for d in out.values():
        d["mean_refs"] = d.pop("_refs") / d["n_turns"] if d["n_turns"] else None
        pairs = d.pop("_pairs")
        d["ref_pair_agreement"] = sum(pairs) / len(pairs) if pairs else None
        d["ref_pair_n"] = len(pairs)
    return out
''', 'd["ref_pair_agreement"]')
sub(du, '''    teacher_sum["by_dialect"] = _teacher_by_dialect(turns, refs_used)
''', '''    teacher_sum["by_dialect"] = _teacher_by_dialect(turns, refs_used)
    # Overall teacher self-agreement = turn-weighted mean over the dialects
    # that have a normal form (A_match telemetry, 2026-09-14).
    _pairs = [(d["ref_pair_agreement"], d["ref_pair_n"])
              for d in teacher_sum["by_dialect"].values() if d.get("ref_pair_agreement") is not None]
    _n = sum(n for _, n in _pairs)
    teacher_sum["ref_pair_agreement"] = (sum(p * n for p, n in _pairs) / _n) if _n else None
    teacher_sum["ref_pair_n"] = _n
''', 'teacher_sum["ref_pair_agreement"]')

dash = REPO / "affine/affine/dashboard.py"
sub(dash, '''                # min(R,G) v5 leg telemetry
                "mean_r_leg", "mean_g_leg", "g_bind_frac",
''', '''                # min(R,G) v5 leg telemetry
                "mean_r_leg", "mean_g_leg", "g_bind_frac",
                # A_match telemetry (2026-09-14, not scored)
                "a_match", "a_match_n", "a_match_centered",
''', '"a_match", "a_match_n", "a_match_centered"')

app = REPO / "affine/website/app.js"
sub(app, '''      ${card("king Reason", esc(fine(kgR)), "same slice, same teacher")}
''', '''      ${card("king Reason", esc(fine(kgR)), "same slice, same teacher")}
      ${(() => {
        // A_match telemetry (2026-09-14, not scored): share of the teacher's
        // k reference actions equal to the side's action after dialect
        // normalisation; `pair` = the refs' own agreement. Rendered only
        // on verdicts that carry it.
        const am = duel.challenger?.a_match, ak = duel.king?.a_match;
        if (am == null && ak == null) return "";
        const pct = (v) => (v == null ? "—" : `${Math.round(Number(v) * 100)}%`);
        const pair = duel.teacher?.ref_pair_agreement;
        const sub = `share of the teacher's reference actions equal to the side's action (dialect-normalised) · telemetry, not scored`
          + (pair != null ? ` · refs agree with each other ${pct(pair)}` : "");
        return card("A_match chall / king", `${esc(pct(am))} / ${esc(pct(ak))}`, esc(sub),
          am != null && ak != null ? passCls(Number(am) >= Number(ak)) : "")
          + (duel.challenger?.a_match_centered != null
            ? card("A_match − pair", `${esc(fine(duel.challenger.a_match_centered))} / ${esc(fine(duel.king?.a_match_centered))}`,
              "chall / king · per-turn A_match minus the refs' own agreement, averaged") : "");
      })()}
''', "A_match chall / king")
print("done")
