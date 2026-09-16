#!/usr/bin/env python
"""The curriculum job (pm2 `affine-curriculum`, daily before the 16:00 UTC
fold): ledger -> shadow weights -> counterfactual -> diff -> stage-3
criterion -> publish -> one private Discord line.

Every step is a plain function of published inputs; the fold reads the
result from affine/state/curriculum/latest.json (same bytes as
data.affine.io/curriculum/latest.json). `[curriculum].mode = off` still
builds and publishes (the ledger is an audit object) but the fold ignores
the vector.

    ops/curriculum/run.sh            # pm2 entry (env from ~/.affine-validator.env / .env)
    python ops/curriculum/run.py --no-publish --no-discord   # dry run
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
import tempfile
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import counterfactual  # noqa: E402
import ledger  # noqa: E402
import publish  # noqa: E402
import rule  # noqa: E402
import weights  # noqa: E402
from common import (  # noqa: E402
    CRITERION_HISTORY, DATA_BASE, EVALS_DIR, HISTORY_PATH, INDEX_CACHE, LATEST_PATH, REPO, SNAPSHOT_DIR,
    STATE_DIR, clean_float, load_curriculum_cfg, log, write_json,
)
from diff import build_diff  # noqa: E402

LEDGER_DIR = STATE_DIR / "ledger"
WORK_DIR = STATE_DIR / "work"


def top10_cards(snapshot: Path, rows_path: Path) -> None:
    """Hand-read cards for stage-3 item 7: the ten strata the rule upweights
    most, each with its cell and the king's drawn turns (score, forfeit) and
    a dashboard link to read the turn."""
    rule_doc = json.loads((snapshot / "rule.json").read_text())
    top = rule_doc["top10_by_weight"]
    strata = {t["stratum"] for t in top}
    t = pq.read_table(rows_path, columns=["base_stratum", "side", "turn_id", "challenge_id", "turn_score",
                                          "forfeit", "live", "miss", "is_king_row", "scored", "prefix_chars",
                                          "source", "harness", "action_kind"])
    by: dict[str, list[dict]] = {s: [] for s in strata}
    for r in t.to_pylist():
        if r["base_stratum"] in strata and r["side"] == "king":
            by[r["base_stratum"]].append(r)
    out = [f"# Top-10 upweighted strata — hand read (stage-3 item 7)\n",
           f"Weights `{rule_doc['weights_sha256'][:12]}`, epoch {rule_doc['corpus_epoch']}. For each stratum: is it a "
           "decision state (a point where the next action matters), not wreckage or a one-reply prompt? "
           "Mark ≥ 7/10 yes to pass. `forfeit_share` = share of the king's misses that were forfeits "
           "(\"cannot answer\": no parseable action / no `</think>`); the rest are live turns scored under θ "
           "(\"answers badly\").\n\n| # | stratum | group | M | forfeit_share | cannot answer | answers badly | n_obs | w |\n"
           "|---|---|---|---:|---:|---:|---:|---:|---:|\n"]
    for i, tp in enumerate(top, 1):
        kr = [r for r in by.get(tp["stratum"], []) if r["is_king_row"] and r["scored"]]
        n_miss = sum(1 for r in kr if r["miss"])
        n_forf = sum(1 for r in kr if r["forfeit"])
        fs = (n_forf / n_miss) if n_miss else None
        tp["_cannot"], tp["_badly"], tp["_fs"] = n_forf, n_miss - n_forf, fs
        out.append(f"| {i} | `{tp['stratum']}` | {tp['group']} | {tp['M'] if tp.get('M') is not None else '-'} | "
                   f"{'-' if fs is None else f'{fs:.2f}'} | {n_forf} | {n_miss - n_forf} | {len(kr)} | {tp['w']:.4f} |\n")
    for i, tp in enumerate(top, 1):
        rows = sorted(by.get(tp["stratum"], []), key=lambda r: (r["challenge_id"], r["turn_id"]))
        out.append(f"\n## {i}. `{tp['stratum']}` — {tp['group']} · cell `{tp['cell']}`\n")
        fs_txt = "-" if tp["_fs"] is None else f"{tp['_fs']:.2f}"
        out.append(f"w {tp['w']:.4f} · M~ {tp['M_t']:.3f} · S~ {tp['S_t']:.3f} · n_obs {tp['n_obs']} · "
                   f"turns in D {tp['n_turns']} · m_shadow {tp['m_shadow']} · forfeit_share {fs_txt} "
                   f"(cannot answer {tp['_cannot']} / answers badly {tp['_badly']})\n")
        if not rows:
            out.append("- no king draws in the window (weight comes from the cell / group prior)\n")
        for r in rows[-6:]:
            out.append(f"- `{r['challenge_id']}` `{r['turn_id']}` score {r['turn_score']} forfeit {r['forfeit']} "
                       f"live {r['live']} miss {r['miss']} prefix {r['prefix_chars']} chars · "
                       f"https://affine.io/api/v1/dataset/turn?turn_id={r['turn_id']}\n")
        out.append("- verdict: [ ] decision state  [ ] wreckage  [ ] one-reply prompt\n")
    (snapshot / "top10_cards.md").write_text("".join(out), encoding="utf-8")


RULE_SHARE_KEYS = {"v1": "share_v1_after_clamp", "v1.1": "share_v11_after_clamp", "v1.2": "share_v12_after_clamp",
                   "v2": "share_v2_after_clamp"}


def forgetting_check(rows_path: Path, snapshot: Path, group: str = "coding") -> dict:
    """Rule v2's no-forgetting loop, shown on the ledger for one group: the
    uniform floor keeps visiting "solved" strata; when the king's score on
    a stratum falls again its divergence D rises and the rule re-weights it.
    Split the window's verdicts in two halves; per stratum with king
    observations in both, compare mean turn score and a per-row divergence
    proxy (the v2 components each divided by its corpus mean)."""
    rule_doc = json.loads((snapshot / "rule.json").read_text())
    means = (rule_doc.get("v2") or {}).get("corpus_means") or {}
    wtab = {r["stratum"]: r for r in pq.read_table(snapshot / "weights.parquet",
                                                    columns=["stratum", "w_v2", "D_v2", "n_obs"]).to_pylist()}
    floor = (rule_doc.get("v2") or {}).get("uniform_floor_per_stratum") or 0.0
    t = pq.read_table(rows_path, columns=["challenge_id", "group", "base_stratum", "is_king_row", "scored",
                                          "side", "turn_score", "forfeit", "div_action", "div_score", "d",
                                          "gated_near_king"])
    rows = [r for r in t.to_pylist() if r["group"] == group and r["scored"]]
    cids = sorted({r["challenge_id"] for r in rows})
    if len(cids) < 4:
        return {"group": group, "n_verdicts": len(cids), "note": "too few verdicts"}
    mid = cids[len(cids) // 2]
    per: dict[str, dict] = {}
    for r in rows:
        if not (r["is_king_row"] and r["side"] == "king"):
            continue
        half = "early" if r["challenge_id"] < mid else "late"
        comps = []
        if means.get("forfeit"):
            comps.append((1.0 if r["forfeit"] else 0.0) / means["forfeit"])
        if r["div_action"] is not None and means.get("action"):
            comps.append(r["div_action"] / means["action"])
        if r["div_score"] is not None and means.get("score"):
            comps.append(r["div_score"] / means["score"])
        dprox = sum(comps) / len(comps) if comps else None
        slot = per.setdefault(r["base_stratum"], {"early": [], "late": []})
        slot[half].append((float(r["turn_score"]), dprox))
    both = {s: v for s, v in per.items() if v["early"] and v["late"]}
    fell, reweighted, examples = 0, 0, []
    for s, v in sorted(both.items()):
        se = sum(x for x, _ in v["early"]) / len(v["early"])
        sl = sum(x for x, _ in v["late"]) / len(v["late"])
        de = [d for _, d in v["early"] if d is not None]
        dl = [d for _, d in v["late"] if d is not None]
        if not de or not dl:
            continue
        de, dl = sum(de) / len(de), sum(dl) / len(dl)
        if sl < se - 0.002:
            fell += 1
            if dl > de:
                reweighted += 1
                w = wtab.get(s) or {}
                examples.append({"stratum": s, "score_early": clean_float(se), "score_late": clean_float(sl),
                                 "D_early": clean_float(de), "D_late": clean_float(dl),
                                 "w_v2": clean_float(w.get("w_v2")), "w_v2_over_floor": clean_float((w.get("w_v2") or 0) / floor) if floor else None,
                                 "n_obs": w.get("n_obs")})
    examples.sort(key=lambda e: -((e["D_late"] or 0) - (e["D_early"] or 0)))
    doc = {"group": group, "n_verdicts": len(cids), "split_at": mid, "n_strata_both_halves": len(both),
           "n_score_fell": fell, "n_fell_and_D_rose": reweighted,
           "share_fell_and_D_rose": clean_float(reweighted / fell) if fell else None,
           "uniform_floor_per_stratum": clean_float(floor), "examples_top": examples[:8],
           "loop": "floor keeps drawing solved strata -> a fall in the king's score raises forfeit / action / score "
                   "divergence -> D_s rises -> w_s rises above the floor next fold"}
    write_json(doc, snapshot / "forgetting_check.json")
    return doc


def per_rule_criterion(*, cfg: dict, groups: dict, prev_groups: dict | None, rows_path: Path,
                       strata_m: dict[str, dict], static: dict[str, float]) -> dict:
    """Items 3 / 4 / 5 / 6 for every published vector (v1 counted, v1.1 and
    v1.2 informational), so fold 3 can pick a rule with its criterion in hand."""
    out = {}
    for name, key in RULE_SHARE_KEYS.items():
        shares = {g: (r.get(key) or 0.0) for g, r in groups["groups"].items()}
        if sum(shares.values()) <= 0:
            continue
        # every rule is judged as it would be applied: after the recurrence
        # guard (k before share), on its own copy of the multiplicities
        sm = copy.deepcopy(strata_m)
        guard = rule.recurrence_guard(sm, shares, group_cap=float(cfg["recurrence_group_cap"]),
                                      turn_cap=float(cfg["recurrence_turn_cap"]))
        shares = guard["shares"]
        cf = counterfactual.run(rows_path, shares, int(cfg["counterfactual_verdicts"]))
        proj = rule.recurrence_projection(sm, shares)
        over_g = {g: d["expected_draws_per_turn_per_duel"] for g, d in proj["groups"].items()
                  if d["expected_draws_per_turn_per_duel"] > cfg["recurrence_group_cap"] + 1e-9}
        fc = rule.check_floors(shares, static, floor_frac=float(cfg["floor_frac_of_static"]),
                               floor_ct=float(cfg["floor_coding_terminal"]), cap=float(cfg["group_cap"]),
                               supply={g: (groups["groups"][g].get("n_strata") or 0) > 0 for g in shares},
                               guard_cut=set(guard["groups_cut"]))
        stab = None
        if prev_groups:
            deltas = {g: abs(shares.get(g, 0.0) - (prev_groups["groups"].get(g, {}).get(key) or 0.0))
                      for g in set(shares) | set(prev_groups["groups"])}
            worst = max(deltas.items(), key=lambda kv: kv[1]) if deltas else ("", 0.0)
            stab = {"pass": worst[1] < cfg["max_share_shift"], "max_abs_delta": clean_float(worst[1]), "group": worst[0]}
        out[name] = {
            "counted": name == str(cfg.get("counted_rule") or "v1"), "shares": {g: clean_float(v) for g, v in sorted(shares.items())},
            "3_counterfactual_amended": {"pass": bool(cf["pass_amended"]), "mean_abs_z_shift": cf["mean_abs_z_shift"],
                                         "sign_flips_abs_z_ge_2": cf["sign_flips_abs_z_ge_2"], "band": cf["amended_band"]},
            "3a_counterfactual_original_plan": {"pass": bool(cf["pass"])},
            "3b_counterfactual_variant": {"pass": bool(cf["pass_variant"])},
            "4_stable_vs_previous": stab if stab else {"pass": None},
            "5_recurrence": {"pass": not over_g and proj["max_turn_draws_per_duel"] <= cfg["recurrence_turn_cap"] + 1e-9,
                             "max_turn_draws_per_duel": clean_float(proj["max_turn_draws_per_duel"]),
                             "groups_over_cap": over_g},
            "6_floors_and_cap": {"pass": bool(fc["ok"]), "coding_plus_terminal": clean_float(fc["coding_plus_terminal"])},
            "recurrence_guard_actions": len(guard["actions"]),
        }
    return out


def criterion(*, cfg: dict, ledger_doc: dict, rebuild_ok: bool | None, groups: dict, rec: dict, cf: dict,
              prev_groups: dict | None, n_new_verdicts: int) -> dict:
    """Stage-3 pass criterion (plan §7.3), computed every fold so the apply
    decision is mechanical. Item 7 is a hand read and stays `manual`."""
    shares = {g: r["share_after_clamp"] for g, r in groups["groups"].items()}
    items = {}
    items["1_rebuild_sha_matches"] = {"pass": rebuild_ok, "detail": "ledger rebuilt from the same inputs gives the same sha"
                                      if rebuild_ok is not None else "not run this cycle"}
    jr = (ledger_doc.get("window") or {}).get("join_rate")
    items["2_turn_join_ge_95pct"] = {"pass": jr is not None and jr >= 0.95, "join_rate": jr}
    items["3_counterfactual"] = {"pass": bool(cf.get("pass_amended")), "mean_abs_z_shift": cf.get("mean_abs_z_shift"),
                                 "sign_flips_abs_z_ge_2": cf.get("sign_flips_abs_z_ge_2"),
                                 "rule": f"coordinator amendment 2026-09-15 01:05 UTC: 0 sign flips at |z| >= 2 AND mean |z| "
                                         f"change within {cf.get('amended_band')} (original plan §7.3: within ±"
                                         f"{cf.get('tolerance_abs_z_shift')}; kept as 3a_original_plan_informational)"}
    items["3a_original_plan_informational"] = {"pass": bool(cf.get("pass")), "counted": False,
                                               "rule": f"plan §7.3 as written: |shift| <= {cf.get('tolerance_abs_z_shift')}, 0 flips"}
    # printed next to item 3, not counted (operator 2026-09-15 00:39 UTC: adopt at fold 3 if fold 2 shows the same shape)
    items["3b_counterfactual_variant_informational"] = {
        "pass": bool(cf.get("pass_variant")), "counted": False,
        "mean_abs_z_shift": cf.get("mean_abs_z_shift"), "sign_flips_abs_z_ge_2": cf.get("sign_flips_abs_z_ge_2"),
        "rule": f"variant: mean |z| shift <= +{cf.get('variant_max_abs_z_shift')} and 0 flips at |z| >= 2"}
    if prev_groups:
        deltas = {g: abs(shares.get(g, 0.0) - (prev_groups["groups"].get(g, {}).get("share_after_clamp") or 0.0))
                  for g in set(shares) | set(prev_groups["groups"])}
        worst = max(deltas.items(), key=lambda kv: kv[1]) if deltas else ("", 0.0)
        items["4_shadow_vector_stable"] = {"pass": worst[1] < cfg["max_share_shift"], "max_abs_delta": clean_float(worst[1]),
                                           "group": worst[0], "vs_weights_sha256": prev_groups.get("weights_sha256")}
    else:
        items["4_shadow_vector_stable"] = {"pass": None, "detail": "first snapshot; needs a previous fold"}
    ps = rec["projected_shadow"]
    over_g = {g: d["expected_draws_per_turn_per_duel"] for g, d in ps["groups"].items()
              if d["expected_draws_per_turn_per_duel"] > cfg["recurrence_group_cap"] + 1e-9}
    items["5_recurrence_within_cap"] = {"pass": not over_g and ps["max_turn_draws_per_duel"] <= cfg["recurrence_turn_cap"] + 1e-9,
                                        "groups_over_cap": over_g, "max_turn_draws_per_duel": ps["max_turn_draws_per_duel"],
                                        "max_turn_stratum": ps["max_turn_stratum"]}
    fc = groups["floors_check"]
    items["6_floors_and_cap_hold"] = {"pass": bool(fc["ok"]), **{k: v for k, v in fc.items() if k != "ok"}}
    items["7_hand_read_top10"] = {"pass": None, "detail": "manual: read top10_cards.md; ≥ 7 of 10 decision states"}
    informational = {"7_hand_read_top10", "3a_original_plan_informational", "3b_counterfactual_variant_informational"}
    auto = [v["pass"] for k, v in items.items() if k not in informational]
    return {
        "counts_as_shadow_fold": n_new_verdicts >= cfg["min_new_verdicts"],
        "n_new_verdicts": n_new_verdicts, "min_new_verdicts": cfg["min_new_verdicts"],
        "automatic_items_pass": all(v is True for v in auto),
        "automatic_items_pending": [k for k, v in items.items() if v["pass"] is None and k not in informational],
        "informational_items": sorted(informational),
        "items": items,
        "decision_rule": "apply at the third fold iff every item passes on shadow folds 1 and 2 (plan §7.3); "
                         "one retry fold; a second failure keeps mode = shadow",
    }


def write_fold_vector(cfg: dict, latest: dict, groups: dict, snapshot: Path) -> Path:
    """The group vector in the shape ops/corpus_build.py `load_curriculum`
    reads (`[curriculum].weights_path`): {"groups": {g: {"share", "m", ...}},
    meta}. `share_applied` is present only in apply mode, so the fold's
    shadow announce shows the rule's vector while the static [mix] decides;
    `m` is the group's median m_shadow (the fold splits per group), the
    per-stratum values are in weights.parquet. `manifest_curriculum_block`
    is the exact block for the corpus manifest."""
    path = Path(cfg.get("weights_path") or "ops/curriculum/out/groups.json")
    path = path if path.is_absolute() else REPO / path
    out_groups = {}
    for g, r in groups["groups"].items():
        hist = {int(k): int(v) for k, v in (r.get("m_hist_shadow") or {}).items()}
        n = sum(hist.values())
        med, acc = 1, 0
        for k in sorted(hist):
            acc += hist[k]
            if acc * 2 >= n:
                med = k
                break
        row = {"share": r["share_after_clamp"], "share_shadow": r["share_after_clamp"],
               "share_v11_informational": r.get("share_v11_after_clamp"),
               "share_v12_informational": r.get("share_v12_after_clamp"),
               "share_v2_informational": r.get("share_v2_after_clamp"),
               "share_raw": r["share_raw"], "share_current": r["share_current"],
               "share_static": r["share_static"], "reason": r["reason"], "m": med if n else 1,
               "m_hist_shadow": r.get("m_hist_shadow"), "n_strata": r["n_strata"]}
        if latest["mode"] == "apply":
            row["share_applied"] = r["share_after_clamp"]
        out_groups[g] = row
    doc = {"epoch": latest["for_epoch"], "against_epoch": latest["against_epoch"],
           "generated_at": latest["computed_at"], "rule_version": latest["rule_version"],
           "mode": latest["mode"], "ledger_sha256": latest["ledger_sha256"],
           "weights_sha256": latest["weights_sha256"], "manifest_sha256": latest["manifest_sha256"],
           "counts_as_shadow_fold": latest["counts_as_shadow_fold"],
           "automatic_items_pass": latest["automatic_items_pass"],
           "manifest_curriculum_block": {
               "rule_version": latest["rule_version"], "mode": latest["mode"],
               "ledger_sha256": latest["ledger_sha256"], "weights_sha256": latest["weights_sha256"],
               "manifest_sha256": latest["manifest_sha256"]},
           "snapshot": str(snapshot), "groups": out_groups}
    write_json(doc, path)
    write_json(doc, snapshot / "fold_vector.json")
    return path


DECISION_ITEMS = ("1_rebuild_sha_matches", "2_turn_join_ge_95pct", "3_counterfactual", "4_shadow_vector_stable",
                  "5_recurrence_within_cap", "6_floors_and_cap_hold", "7_hand_read_top10")


def shadow_fold_history(counted_rule: str) -> list[dict]:
    """Previous runs that COUNT as shadow folds for this counted rule
    (>= min_new_verdicts new verdicts). Two kinds of operator records may
    follow in the same file: {"amends": <computed_at>, "items": {...},
    "reason": ...} overrides items of an earlier run (e.g. a FAIL that was a
    tooling artefact), and {"override_count": true, ...} declares a run a
    counting fold. Both are printed in the decision table."""
    if not CRITERION_HISTORY.is_file():
        return []
    rows: list[dict] = []
    amend: list[dict] = []
    for line in CRITERION_HISTORY.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        (amend if r.get("amends") else rows).append(r)
    for a in amend:
        for r in rows:
            if r.get("computed_at") == a["amends"]:
                r["items"].update(a.get("items") or {})
                r.setdefault("amendments", []).append(a.get("reason") or "")
                if "counted_rule" in a:
                    r["counted_rule"] = a["counted_rule"]
                if "counts_as_shadow_fold" in a:
                    r["counts_as_shadow_fold"] = a["counts_as_shadow_fold"]
    return [r for r in rows if r.get("counts_as_shadow_fold") and r.get("counted_rule", "v1") == counted_rule]


def decision_table(crit: dict) -> str:
    """The fold-3 apply decision, mechanical: every item PASS on shadow folds
    2 and 3 (counted rule v1.2, coordinator amendment 2026-09-15 01:05 UTC)
    -> APPLY; item 7 is the hand read; a FAIL -> one retry fold; a second
    FAIL -> mode stays shadow."""
    prev = shadow_fold_history(crit["counted_rule"])[-2:]
    cols = [(r["computed_at"][:16], r["items"]) for r in prev] + [("this run", {k: v["pass"] for k, v in crit["items"].items()})]
    fl = lambda x: "PASS" if x is True else "FAIL" if x is False else "n/a "  # noqa: E731
    head = f"  === fold-3 decision table (counted rule {crit['counted_rule']}; APPLY iff every row PASS on two counting shadow folds) ==="
    lines = [head, "  item".ljust(34) + "".join(c[0].ljust(18) for c in cols)]
    for item in DECISION_ITEMS:
        lines.append(f"  {item}".ljust(34) + "".join(fl(c[1].get(item)).ljust(18) for c in cols))
    counting = [c for c in cols[:-1]] + ([cols[-1]] if crit["counts_as_shadow_fold"] else [])
    two = counting[-2:]
    if len(two) < 2:
        verdict = f"HOLD -- {len(two)} counting shadow fold(s) so far, need 2"
    else:
        auto_ok = all(c[1].get(i) is True for c in two for i in DECISION_ITEMS if i != "7_hand_read_top10")
        hand = [c[1].get("7_hand_read_top10") for c in two]
        verdict = ("APPLY (pending item 7 hand read)" if auto_ok and not all(h is True for h in hand)
                   else "APPLY" if auto_ok else "RETRY/HOLD -- an automatic item failed on one of the two folds")
    lines.append(f"  verdict: {verdict}")
    for r in prev:
        for a in r.get("amendments") or []:
            lines.append(f"  amendment on {r['computed_at'][:16]}: {a}")
    return "\n".join(lines)


APPLY_NOTICE = """Dataset sampling update (data event, no scoring change, no weight_version_key change).

Since {date} the share of each part of D in every 1,300-turn duel slice follows a published rule instead of a hand-set table. Rule v2 ("the divergence rule"): for every stratum of D we measure, from the duel records we already publish under evals/, how far the sitting king is from the teacher on the turns it was duelled on -- action disagreement (1 - soft A_match, token-Jaccard of the normalised actions), the king's forfeit rate, the score deficit max(0, teacher own-action lift - king B), and the gap a near-king challenger opens on the same turns (Dbar+) -- each divided by its corpus mean and averaged. A stratum's draw weight is w = 0.20/N + 0.80 * D / sum D: 20 % of the mass is spread uniformly over ALL strata so every turn keeps a non-zero chance of being drawn (nothing can be forgotten -- a stratum the king solved and later regresses on rises again), 80 % follows the divergence. Coding + terminal stay >= 40 % of every slice; no group moves more than 5 points per fold; a stratum is drawn at most 3 times per duel and at most 0.18 expected draws per turn per duel per group.

Everything is auditable: https://data.affine.io/curriculum/{epoch}/ has the rule (rule.json), the per-stratum weights (weights.parquet), the group shares with the reason for every floor / clamp (groups.json), the recurrence stats, the counterfactual check and a one-page diff per fold; the ledger rebuilds byte-for-byte from evals/ + history (ops/curriculum/ledger.py --check <sha>). Verdicts stamp slice.curriculum_version. Details: https://affine.io/llms.txt -> "Adaptive curriculum".

Slices stay seeded by your reveal block hash and teacher references stay fresh per duel. The upweighted strata are public on purpose: they are the states where the king diverges most from the teacher, labelled by the teacher at duel time.
"""


def flip_mode_to_apply(apply_date: str, wsha: str) -> None:
    """Mechanical apply (coordinator 2026-09-16 17:04 UTC): set
    `[curriculum].mode = "apply"` in sources.toml as a ONE-LINE patch on git
    HEAD (other workers keep uncommitted edits in that file), update the
    llms.txt builder to the apply wording, rebuild llms.txt, commit."""
    toml_path = REPO / "rollouts" / "rollouts" / "sources.toml"
    live = toml_path.read_text()
    if re.search(r'^mode = "apply"', live, re.M):
        return
    live_new = re.sub(r'^mode = "shadow".*$', f'mode = "apply"   # curriculum apply fold {apply_date}, weights {wsha[:12]} (mechanical, criterion passed twice)',
                      live, count=1, flags=re.M)
    toml_path.write_text(live_new)
    head = subprocess.run(["git", "show", "HEAD:rollouts/rollouts/sources.toml"], cwd=REPO, capture_output=True,
                          text=True, check=True).stdout
    head_new = re.sub(r'^mode = "shadow".*$', f'mode = "apply"   # curriculum apply fold {apply_date}, weights {wsha[:12]} (mechanical, criterion passed twice)',
                      head, count=1, flags=re.M)
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td) / "a", Path(td) / "b"
        a.write_text(head), b.write_text(head_new)
        diff = subprocess.run(["git", "diff", "--no-index", "--", str(a), str(b)], cwd=REPO, capture_output=True, text=True).stdout
        # the --no-index header is `--- a/<tmp>/a`: keep the a/ b/ prefixes git strips
        diff = diff.replace(str(a), "/rollouts/rollouts/sources.toml").replace(str(b), "/rollouts/rollouts/sources.toml")
        subprocess.run(["git", "apply", "--cached", "--recount", "-"], cwd=REPO, input=diff, text=True, check=True)
    subprocess.run([sys.executable, str(REPO / "ops" / "curriculum" / "llms_edits.py"), "--mode", "apply",
                    "--apply-date", apply_date], cwd=REPO, check=True)
    subprocess.run([sys.executable, "scripts/build_llms_txt.py"], cwd=REPO / "affine", check=True)
    subprocess.run(["git", "add", "affine/scripts/build_llms_txt.py"], cwd=REPO, check=True)
    subprocess.run(["git", "commit", "-q", "-m",
                    f"curriculum: mode shadow -> apply ({apply_date}; weights {wsha[:12]}; criterion passed on two counting "
                    f"shadow folds -- mechanical apply per coordinator decision 2026-09-16 17:04 UTC); llms.txt apply wording"],
                   cwd=REPO, check=True)
    log(f"[run] APPLY: [curriculum].mode = apply committed on HEAD; llms.txt rebuilt")


def previous_snapshot() -> tuple[dict | None, Path | None]:
    if not LATEST_PATH.is_file():
        return None, None
    prev = json.loads(LATEST_PATH.read_text())
    d = SNAPSHOT_DIR / prev["weights_sha256"]
    return prev, (d if d.is_dir() else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-publish", action="store_true")
    ap.add_argument("--no-discord", action="store_true")
    ap.add_argument("--no-rebuild-check", action="store_true",
                    help="skip the second, independent ledger build (item 1 becomes 'not run')")
    ap.add_argument("--publish-prefix", default="", help="publish under this key prefix (e.g. staging/)")
    ap.add_argument("--history", default=str(HISTORY_PATH))
    ap.add_argument("--evals", default=str(EVALS_DIR))
    ap.add_argument("--data-base", default=DATA_BASE)
    ap.add_argument("--max-verdicts", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_curriculum_cfg()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    prev_latest, prev_dir = previous_snapshot()

    # 1. ledger
    largs = Namespace(history=args.history, evals=args.evals, index_cache=str(INDEX_CACHE),
                      data_base=args.data_base, since=cfg["first_challenge"], until="chal-99999",
                      max_verdicts=args.max_verdicts, out=str(LEDGER_DIR))
    ledger_doc = ledger.build(largs)
    lsha = ledger_doc["ledger_sha256"]
    log(f"[run] ledger {lsha} in {time.time() - t0:.0f}s")
    rebuild_ok: bool | None = None
    if not args.no_rebuild_check:
        t1 = time.time()
        chk = Namespace(**{**vars(largs), "out": str(WORK_DIR / "rebuild_check")})
        shutil.rmtree(chk.out, ignore_errors=True)
        rebuild_ok = ledger.build(chk)["ledger_sha256"] == lsha
        shutil.rmtree(chk.out, ignore_errors=True)
        log(f"[run] rebuild check {'OK' if rebuild_ok else 'MISMATCH'} in {time.time() - t1:.0f}s")
        if not rebuild_ok:
            raise SystemExit("ledger rebuild sha mismatch -- not publishing")

    # 2. weights (mode from [curriculum])
    wdir = WORK_DIR / "weights"
    shutil.rmtree(wdir, ignore_errors=True)
    wargs = Namespace(ledger_json=str(LEDGER_DIR / f"{lsha}.json"), manifest_sha=None, mode=None,
                      index_cache=str(INDEX_CACHE), data_base=args.data_base, out=str(wdir), probes=None)
    rule_doc = weights.compute(wargs)
    wsha = rule_doc["weights_sha256"]
    snapshot = SNAPSHOT_DIR / wsha
    if snapshot.exists():
        shutil.rmtree(snapshot)
    shutil.copytree(wdir, snapshot)
    shutil.copyfile(LEDGER_DIR / f"{lsha}.json", snapshot / "ledger.json")
    groups = json.loads((snapshot / "groups.json").read_text())
    rec = json.loads((snapshot / "recurrence.json").read_text())

    # 3. counterfactual + cards + diff + criterion
    shares = {g: r["share_after_clamp"] for g, r in groups["groups"].items()}
    cf = counterfactual.run(LEDGER_DIR / f"{lsha}.rows.parquet", shares, int(cfg["counterfactual_verdicts"]))
    write_json(cf, snapshot / "counterfactual.json")
    top10_cards(snapshot, LEDGER_DIR / f"{lsha}.rows.parquet")
    prev_groups = json.loads((prev_dir / "groups.json").read_text()) if prev_dir else None
    prev_n = ((json.loads((prev_dir / "ledger.json").read_text()).get("window") or {}).get("n_verdicts")
              if prev_dir else 0) or 0
    n_new = int(ledger_doc["window"]["n_verdicts"]) - int(prev_n)
    crit = criterion(cfg=cfg, ledger_doc=ledger_doc, rebuild_ok=rebuild_ok, groups=groups, rec=rec, cf=cf,
                     prev_groups=prev_groups, n_new_verdicts=n_new)
    wtab = pq.read_table(snapshot / "weights.parquet", columns=["stratum", "group", "n_turns", "m_shadow"]).to_pylist()
    strata_m = {r["stratum"]: {"group": r["group"], "n_turns": r["n_turns"], "m": r["m_shadow"]} for r in wtab}
    static_mix = {g: (r.get("share_static") or 0.0) for g, r in groups["groups"].items()}
    crit["by_rule"] = per_rule_criterion(cfg=cfg, groups=groups, prev_groups=prev_groups,
                                         rows_path=LEDGER_DIR / f"{lsha}.rows.parquet", strata_m=strata_m,
                                         static=static_mix)
    crit["counted_rule"] = str(cfg.get("counted_rule") or "v1")
    crit["amendment"] = ("coordinator 2026-09-15 01:05 UTC: counted rule v1.2 from fold 2; item 3 = 0 flips at |z| >= 2 AND "
                         "mean |z| change in [-10 %, +50 %]; recurrence cap (<= 0.18 draws/turn/duel) and floors are hard; "
                         "item 4 measured v1.2-vs-v1.2 across folds 2 and 3; recurrence over cap lowers k before share")
    crit["decision_table"] = decision_table(crit)
    write_json(crit["by_rule"], snapshot / "criterion_by_rule.json")
    forget = forgetting_check(LEDGER_DIR / f"{lsha}.rows.parquet", snapshot)
    crit["weights_sha256"] = wsha
    crit["ledger_sha256"] = lsha
    crit["computed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(crit, snapshot / "criterion.json")
    (snapshot / "diff.md").write_text(build_diff(snapshot, prev_dir), encoding="utf-8")
    with open(snapshot / "diff.md", "a", encoding="utf-8") as f:
        f.write("\n12. **Fold-3 decision table** (coordinator amendment 2026-09-15 01:05 UTC):\n\n```\n"
                + crit["decision_table"] + "\n```\n")
        if forget.get("n_strata_both_halves"):
            ex = "; ".join(f"`{e['stratum']}` score {e['score_early']:.4f}→{e['score_late']:.4f}, D {e['D_early']:.2f}→{e['D_late']:.2f}, "
                           f"w_v2 {e['w_v2_over_floor']:.1f}× floor" for e in forget["examples_top"][:4])
            f.write(f"\n13. **Forgetting feedback (rule v2, {forget['group']}):** of {forget['n_strata_both_halves']} strata the king "
                    f"was drawn on in both halves of the window (split at {forget['split_at']}), {forget['n_score_fell']} fell in score "
                    f"and {forget['n_fell_and_D_rose']} of those ({100 * (forget['share_fell_and_D_rose'] or 0):.0f} %) rose in divergence "
                    f"→ re-weighted above the uniform floor {forget['uniform_floor_per_stratum']:.2e}. Examples: {ex}. Loop: {forget['loop']}\n")

    for_epoch = int(rule_doc["corpus_epoch"]) + 1
    verdict_line = crit["decision_table"].splitlines()
    verdict = next((l for l in verdict_line if l.strip().startswith("verdict:")), "")
    if (cfg.get("auto_apply_on_pass") and rule_doc["mode"] == "shadow" and "APPLY" in verdict
            and not args.no_publish):
        apply_date = crit["computed_at"][:10]
        flip_mode_to_apply(apply_date, wsha)
        # recompute the same weights in apply mode (share_applied / m_applied), same sha inputs
        shutil.rmtree(wdir, ignore_errors=True)
        rule_doc = weights.compute(Namespace(**{**vars(wargs), "mode": "apply"}))
        wsha = rule_doc["weights_sha256"]
        old_snapshot, snapshot = snapshot, SNAPSHOT_DIR / wsha
        if snapshot.exists():
            shutil.rmtree(snapshot)
        shutil.copytree(wdir, snapshot)
        for name in ("ledger.json", "counterfactual.json", "top10_cards.md", "criterion_by_rule.json",
                     "forgetting_check.json"):
            if (old_snapshot / name).is_file():
                shutil.copyfile(old_snapshot / name, snapshot / name)
        groups = json.loads((snapshot / "groups.json").read_text())
        rec = json.loads((snapshot / "recurrence.json").read_text())
        shares = {g: r["share_after_clamp"] for g, r in groups["groups"].items()}
        crit["weights_sha256"] = wsha
        crit["applied"] = {"date": apply_date, "weights_sha256": wsha}
        write_json(crit, snapshot / "criterion.json")
        (snapshot / "diff.md").write_text(build_diff(snapshot, prev_dir), encoding="utf-8")
        (snapshot / "apply_notice.md").write_text(APPLY_NOTICE.format(date=apply_date, epoch=for_epoch), encoding="utf-8")
        log(f"[run] APPLY: weights recomputed in apply mode -> {wsha}")
    latest = {
        "rule_version": int(rule_doc["rule_version"]), "mode": rule_doc["mode"],
        "ledger_sha256": lsha, "weights_sha256": wsha, "manifest_sha256": rule_doc["manifest_sha256"],
        "against_epoch": int(rule_doc["corpus_epoch"]), "for_epoch": for_epoch,
        "theta": rule_doc["theta"], "knobs": rule_doc["knobs"],
        "computed_at": crit["computed_at"], "counts_as_shadow_fold": crit["counts_as_shadow_fold"],
        "automatic_items_pass": crit["automatic_items_pass"],
        "shares_after_clamp": {g: clean_float(v) for g, v in sorted(shares.items())},
        "paths": {"ledger": f"curriculum/ledger/{lsha}.json", "weights": f"curriculum/weights/{wsha}/",
                  "epoch": f"curriculum/{for_epoch}/"},
        "local_snapshot": str(snapshot),
    }
    latest_body = write_json(latest, snapshot / "latest.json")
    write_fold_vector(cfg, latest, groups, snapshot)

    # 4. print the criterion (the fold-3 decision is read off this block)
    print("\n=== stage-3 criterion ===")
    for k, v in crit["items"].items():
        print(f"  {k}: {'PASS' if v['pass'] is True else 'FAIL' if v['pass'] is False else 'n/a'}  "
              f"{ {kk: vv for kk, vv in v.items() if kk != 'pass'} }")
    print(f"  --- per rule (* = counted rule {crit['counted_rule']}; the others informational) ---")
    for name, r in crit["by_rule"].items():
        flag = lambda x: "PASS" if x is True else "FAIL" if x is False else "n/a"  # noqa: E731
        print(f"  {name:5s}{' *' if r['counted'] else '  '} cf-amended {flag(r['3_counterfactual_amended']['pass'])} ({100 * (r['3_counterfactual_amended']['mean_abs_z_shift'] or 0):+.1f}%, "
              f"flips {len(r['3_counterfactual_amended']['sign_flips_abs_z_ge_2'])}) · cf-plan-orig {flag(r['3a_counterfactual_original_plan']['pass'])} · "
              f"stable {flag(r['4_stable_vs_previous']['pass'])} · recurrence {flag(r['5_recurrence']['pass'])} "
              f"({r['5_recurrence']['max_turn_draws_per_duel']:.3f}) · floors {flag(r['6_floors_and_cap']['pass'])} · "
              + ", ".join(f"{g} {100 * v:.1f}" for g, v in sorted(r["shares"].items(), key=lambda kv: -kv[1])[:6]))
    print(decision_table(crit))
    print(f"  counts_as_shadow_fold={crit['counts_as_shadow_fold']} (n_new_verdicts={n_new}); "
          f"automatic_items_pass={crit['automatic_items_pass']} pending={crit['automatic_items_pending']}")
    print((snapshot / "diff.md").read_text())

    # 5. publish + pointers + discord
    if not args.no_publish:
        pub = publish.make_publisher(args.publish_prefix)
        publish.publish_snapshot(pub, snapshot_dir=snapshot, ledger_dir=LEDGER_DIR, ledger_sha=lsha,
                                 weights_sha=wsha, for_epoch=for_epoch, latest_body=latest_body)
        log(f"[run] published curriculum/{for_epoch}/ + curriculum/weights/{wsha[:12]}/ + ledger {lsha[:12]}")
    write_json(latest, LATEST_PATH)
    with open(CRITERION_HISTORY, "a", encoding="utf-8") as f:
        f.write(json.dumps({"computed_at": crit["computed_at"], "weights_sha256": wsha, "ledger_sha256": lsha,
                            "counted_rule": crit["counted_rule"],
                            "for_epoch": for_epoch, "counts_as_shadow_fold": crit["counts_as_shadow_fold"],
                            "automatic_items_pass": crit["automatic_items_pass"],
                            "items": {k: v["pass"] for k, v in crit["items"].items()}}, sort_keys=True) + "\n")
    moves = sorted(((g, shares[g] - (groups["groups"][g]["share_current"] or 0)) for g in shares),
                   key=lambda kv: -abs(kv[1]))[:3]
    line = (f"curriculum {rule_doc['mode']} rule {crit['counted_rule']} (v{rule_doc['rule_version']} family): ledger `{lsha[:12]}` "
            f"({ledger_doc['window']['n_verdicts']} verdicts, +{n_new}, θ {rule_doc['theta']:.4f}), weights "
            f"`{wsha[:12]}` for epoch {for_epoch}; top moves shadow vs live: "
            + ", ".join(f"{g} {100 * d:+.1f}pt" for g, d in moves)
            + f"; recurrence max/turn {rec['projected_shadow']['max_turn_draws_per_duel']:.3f}; criterion auto "
            f"{'PASS' if crit['automatic_items_pass'] else 'FAIL/pending ' + str(crit['automatic_items_pending'])}"
            f"{'' if crit['counts_as_shadow_fold'] else ' (does not count: < min_new_verdicts)'}; "
            f"{args.data_base}/curriculum/{for_epoch}/diff.md")
    print(line)
    if not args.no_discord:
        publish.discord_line(line)
    log(f"[run] done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
