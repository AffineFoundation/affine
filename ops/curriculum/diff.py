#!/usr/bin/env python
"""One-page diff of a curriculum snapshot against the previous fold.

Eight fixed lines (plan §5): rule/mode · verdicts added and which king ·
θ old → new · group shares old → new with reasons · top-10 strata by
weight gain and loss (M~, S~, n) · multiplicity histogram · recurrence
above cap and memorised cells · projected SE change from the
counterfactual re-weight of the last 20 verdicts.

    python ops/curriculum/diff.py --new <snapshot dir> [--prev <snapshot dir>] --out diff.md
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


def _load(d: Path | None, name: str):
    if d is None or not (d / name).is_file():
        return None
    return json.loads((d / name).read_text())


def _weights(d: Path | None) -> dict[str, dict]:
    if d is None or not (d / "weights.parquet").is_file():
        return {}
    return {r["stratum"]: r for r in pq.read_table(d / "weights.parquet").to_pylist()}


def _pct(x) -> str:
    return "-" if x is None else f"{100 * float(x):.1f}%"


def _f(x, nd=4) -> str:
    return "-" if x is None else f"{float(x):.{nd}f}"


def build_diff(new: Path, prev: Path | None) -> str:
    rule_n, rule_p = _load(new, "rule.json"), _load(prev, "rule.json")
    grp_n, grp_p = _load(new, "groups.json"), _load(prev, "groups.json")
    rec_n = _load(new, "recurrence.json")
    cf_n = _load(new, "counterfactual.json")
    led_n, led_p = _load(new, "ledger.json"), _load(prev, "ledger.json")
    w_n, w_p = _weights(new), _weights(prev)
    win_n = (rule_n or {}).get("window") or {}
    win_p = (rule_p or {}).get("window") or {}
    lines = []
    # 1 rule / mode
    lines.append(f"1. **Rule / mode:** v{rule_n['rule_version']} `{rule_n['mode']}` — weights "
                 f"`{rule_n['weights_sha256'][:12]}` on corpus epoch {rule_n['corpus_epoch']} "
                 f"(manifest `{rule_n['manifest_sha256'][:12]}`); ledger `{rule_n['ledger_sha256'][:12]}`"
                 + (f"; previous weights `{rule_p['weights_sha256'][:12]}` (epoch {rule_p['corpus_epoch']})"
                    if rule_p else "; no previous snapshot") + ".")
    # 2 verdicts added, which king
    n_new = (win_n.get("n_verdicts") or 0) - (win_p.get("n_verdicts") or 0) if win_p else win_n.get("n_verdicts")
    kings = (led_n or {}).get("kings_in_window") or {}
    newest_king = (led_n or {}).get("newest_king") or {}
    lines.append(f"2. **Verdicts:** {win_n.get('n_verdicts')} in the window "
                 f"({win_n.get('first_challenge_id')} … {win_n.get('last_challenge_id')}), "
                 f"{n_new} new since the previous snapshot"
                 + (f"; last king reign {newest_king.get('reign')} `{str(newest_king.get('digest') or '')[:12]}`"
                    f" ({newest_king.get('n_verdicts')} verdicts)" if newest_king else "")
                 + (f"; king rows by reign: " + ", ".join(f"r{k} {v}" for k, v in sorted(kings.items())) if kings else "")
                 + f"; turn-id join {_pct(win_n.get('join_rate'))}.")
    # 3 theta
    lines.append(f"3. **θ (bottom-quartile king turn score):** {_f((rule_p or {}).get('theta'), 5)} → "
                 f"{_f(rule_n.get('theta'), 5)}.")
    # 4 group shares
    gs = []
    for g, r in sorted(grp_n["groups"].items(), key=lambda kv: -(kv[1]["share_after_clamp"] or 0)):
        old = (grp_p or {}).get("groups", {}).get(g, {}).get("share_after_clamp") if grp_p else None
        gs.append(f"{g} {_pct(old) if grp_p else _pct(r['share_current'])}→{_pct(r['share_after_clamp'])} "
                  f"[{r['reason']}; raw {_pct(r['share_raw'])}, base-strata raw {_pct(r.get('share_raw_base_strata'))}]")
    lines.append(f"4. **Group shares** ({'previous shadow' if grp_p else 'live'} → shadow; Σw over "
                 f"{grp_n.get('share_unit', 'slice_keys')}; applied = "
                 f"{'shadow' if grp_n['mode'] == 'apply' else 'live static mix'}): " + "; ".join(gs) + ".")
    # 5 top-10 gain / loss
    if w_p:
        deltas = sorted(((w_n[s]["w"] or 0) - (w_p[s]["w"] or 0), s) for s in w_n if s in w_p)
        gains = [(d, s) for d, s in reversed(deltas) if d > 0][:10]
        losses = [(d, s) for d, s in deltas if d < 0][:10]
        fmt = lambda d, s: (f"`{s}` {d:+.4f} (M~ {_f(w_n[s]['M_t'], 3)}, S~ {_f(w_n[s]['S_t'], 3)}, "  # noqa: E731
                            f"n {_f(w_n[s]['n_w'], 1)}, forfeit_share {_f(w_n[s].get('forfeit_share'), 2)})")
        lines.append("5. **Top-10 strata by weight gain:** " + ("; ".join(fmt(d, s) for d, s in gains) or "none")
                     + ". **Loss:** " + ("; ".join(fmt(d, s) for d, s in losses) or "none") + ".")
    else:
        top = sorted(w_n.values(), key=lambda r: (-(r["w"] or 0), r["stratum"]))[:10]
        lines.append("5. **Top-10 strata by weight** (first snapshot, no previous): " + "; ".join(
            f"`{r['stratum']}` w {_f(r['w'])} (M~ {_f(r['M_t'], 3)}, S~ {_f(r['S_t'], 3)}, n {_f(r['n_w'], 1)}, "
            f"forfeit_share {_f(r.get('forfeit_share'), 2)}, {r['group']})" for r in top) + ".")
    # 6 multiplicity histogram
    hist = Counter(int(r["m_shadow"]) for r in w_n.values())
    hist_g = {g: dict(sorted(Counter(int(r["m_shadow"]) for r in w_n.values() if r["group"] == g).items()))
              for g in sorted({r["group"] for r in w_n.values()})}
    lines.append(f"6. **Multiplicity (m_shadow) histogram:** " + ", ".join(f"m={k}: {v}" for k, v in sorted(hist.items()))
                 + " strata; per group " + "; ".join(f"{g} {h}" for g, h in hist_g.items()) + ".")
    # 7 recurrence
    caps = (rec_n or {}).get("caps") or {}
    ps = (rec_n or {}).get("projected_shadow") or {}
    over = [f"{g} {d['expected_draws_per_turn_per_duel']:.3f}" for g, d in (ps.get("groups") or {}).items()
            if d["expected_draws_per_turn_per_duel"] > caps.get("group_expected_draws_per_turn_per_duel", 0.18)]
    above = (rec_n or {}).get("ledger_strata_above_cap") or []
    lines.append(f"7. **Recurrence:** baseline {_f((rec_n or {}).get('baseline_expected_draws_per_turn_per_duel'), 3)} "
                 f"expected draws per turn per duel; projected under the shadow vector max per group "
                 + (", ".join(over) if over else "none above cap") +
                 f"; worst single turn {_f(ps.get('max_turn_draws_per_duel'), 3)} (`{ps.get('max_turn_stratum')}`); "
                 f"ledger strata with a turn drawn > 3× in the last 50 verdicts: {len(above)}; memorised cells: "
                 "guard not active before stage 4 (alert-only).")
    # 8 counterfactual
    if cf_n:
        lines.append(f"8. **Counterfactual (last {cf_n['n_verdicts']} verdicts under the shadow vector):** mean |z| "
                     f"{_f(cf_n['mean_abs_z_stored'], 2)} → {_f(cf_n['mean_abs_z_shadow'], 2)} "
                     f"({100 * (cf_n['mean_abs_z_shift'] or 0):+.1f}%), median SE {_f(cf_n['median_se_realized'], 6)} → "
                     f"{_f(cf_n['median_se_shadow'], 6)} ({100 * (cf_n['median_se_shift'] or 0):+.1f}%), sign flips among "
                     f"|z| ≥ 2: {len(cf_n['sign_flips_abs_z_ge_2'])} {cf_n['sign_flips_abs_z_ge_2'] or ''}; plan item "
                     f"(±10 %, 0 flips) {'PASS' if cf_n.get('pass') else 'FAIL'}; variant (≤ +25 %, 0 flips) "
                     f"{'PASS' if cf_n.get('pass_variant') else 'FAIL'}.")
    else:
        lines.append("8. **Counterfactual:** not computed.")
    # 9 decomposition (coordinator 2026-09-15 00:46 UTC): what drives each group's share
    dec = []
    for g, r in sorted(grp_n["groups"].items(), key=lambda kv: -(kv[1]["share_after_clamp"] or 0)):
        if (r.get("n_strata") or 0) == 0:
            continue
        dec.append(f"{g}: M~ {_f(r.get('mean_M_t_slice_keys'), 3)} · S~ {_f(r.get('mean_S_t_slice_keys'), 3)} · "
                   f"w {_f(r.get('mean_w_slice_keys'), 3)} → v1 {_pct(r['share_after_clamp'])} | M~-only "
                   f"{_pct(r.get('share_m_only_after_clamp'))} | v1.1 gate {_pct(r.get('share_v11_after_clamp'))} "
                   f"({r.get('v11_eligible_strata')}/{r.get('n_strata')} eligible) | v1.2 F~ {_f(r.get('mean_F_t_slice_keys'), 3)} + "
                   f"Dbar+~ {_f(r.get('mean_Dp_t_slice_keys'), 4)} → M12~ {_f(r.get('mean_M12_t_slice_keys'), 3)} → "
                   f"{_pct(r.get('share_v12_after_clamp'))} [{r.get('v12_reason')}] | floors-only {_pct(r.get('share_floors_only'))}")
    v11 = grp_n.get("v11") or {}
    v12 = grp_n.get("v12") or {}
    lines.append("9. **Decomposition per group** (means over slice keys; shares after floors + clamp; v1 counted, "
                 f"M~-only = same rule with S~ = 1, v1.1 = S~ as a gate ≥ {v11.get('s_gate', 0.5)} then weight = M~, "
                 f"v1.2 = king forfeit rate F~ + {_f(v12.get('dplus_scale'), 2)} × mean positive challenger gap Dbar+~ "
                 f"(scale = corpus forfeit {_f(v12.get('corpus_mean_forfeit'), 3)} / corpus Dbar+ {_f(v12.get('corpus_mean_Dbar_plus'), 4)}), "
                 "S~ gate — both informational, not counted; floors-only = live share with only the plan's floors): "
                 + "; ".join(dec) + ".")
    # 10 the king's miss rate per group, raw from the ledger
    km = []
    for g, r in sorted(grp_n["groups"].items(), key=lambda kv: -((kv[1].get("king_M") or 0))):
        if r.get("king_M") is None:
            continue
        km.append(f"{g}: M {_f(r['king_M'], 3)} (forfeit {_f(r.get('king_forfeit_rate'), 3)}, live-answered miss "
                  f"{_f(r.get('king_M_live_answered'), 3)}, n {r.get('king_n_live_answered')}) · S {_f(r.get('king_S'), 3)} · "
                  f"q25 live score {_f(r.get('king_q25_live_score'), 5)} vs θ {_f(grp_n.get('theta'), 5)}")
    lines.append("10. **King miss rate per group** (decayed means over king rows; M = forfeit OR live score < θ; "
                 "live-answered miss = share of answered live turns under the GLOBAL θ — 0.25 means the group sits on "
                 "the corpus distribution, higher means θ is easier to fall under there): " + "; ".join(km) + ".")
    # 11 criterion per rule
    cbr = _load(new, "criterion_by_rule.json") or {}
    if cbr:
        fl = lambda x: "PASS" if x is True else "FAIL" if x is False else "n/a"  # noqa: E731
        parts = []
        for name, r in cbr.items():
            parts.append(f"{name}{' (counted)' if r.get('counted') else ''}: counterfactual amended "
                         f"{fl(r['3_counterfactual_amended']['pass'])} ({100 * (r['3_counterfactual_amended']['mean_abs_z_shift'] or 0):+.1f} %, "
                         f"{len(r['3_counterfactual_amended']['sign_flips_abs_z_ge_2'])} flips) / original plan "
                         f"{fl(r['3a_counterfactual_original_plan']['pass'])} · "
                         f"stable {fl(r['4_stable_vs_previous'].get('pass'))} · recurrence {fl(r['5_recurrence']['pass'])} "
                         f"({_f(r['5_recurrence']['max_turn_draws_per_duel'], 3)}) · floors {fl(r['6_floors_and_cap']['pass'])}")
        lines.append("11. **Criterion per rule** (items 3–6; 1, 2 and 7 are rule-independent; item 3 as amended by the "
                     "coordinator 2026-09-15 01:05 UTC = 0 sign flips at |z| ≥ 2 AND mean |z| change within [−10 %, +50 %] — "
                     "concentrating signal is the intended effect; recurrence cap ≤ 0.18 draws/turn/duel and the floors are the "
                     "hard safety items; over-cap recurrence lowers multiplicity k before share): " + "; ".join(parts)
                     + f". Counted rule: {grp_n.get('counted_rule', 'v1')}"
                     + (f"; recurrence guard actions: {(grp_n.get('recurrence_guard') or {}).get('n_actions', 0)}" if grp_n.get("recurrence_guard") else "")
                     + ".")
    head = (f"# Curriculum diff — epoch {rule_n['corpus_epoch']} → next fold\n\n"
            f"Computed {rule_n.get('computed_at')} UTC. Files: `rule.json`, `weights.parquet`, `groups.json`, "
            f"`recurrence.json`, `deficit_by_source.json`, `counterfactual.json`, `criterion.json`.\n\n")
    return head + "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--new", required=True)
    ap.add_argument("--prev", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    text = build_diff(Path(args.new), Path(args.prev) if args.prev else None)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
