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
import json
import shutil
import sys
import time
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

import counterfactual  # noqa: E402
import ledger  # noqa: E402
import publish  # noqa: E402
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
    items["3_counterfactual"] = {"pass": bool(cf.get("pass")), "mean_abs_z_shift": cf.get("mean_abs_z_shift"),
                                 "sign_flips_abs_z_ge_2": cf.get("sign_flips_abs_z_ge_2"),
                                 "rule": f"plan §7.3: |mean |z| shift| <= {cf.get('tolerance_abs_z_shift')} and 0 flips at |z| >= 2"}
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
              if d["expected_draws_per_turn_per_duel"] > cfg["recurrence_group_cap"]}
    items["5_recurrence_within_cap"] = {"pass": not over_g and ps["max_turn_draws_per_duel"] <= cfg["recurrence_turn_cap"],
                                        "groups_over_cap": over_g, "max_turn_draws_per_duel": ps["max_turn_draws_per_duel"],
                                        "max_turn_stratum": ps["max_turn_stratum"]}
    fc = groups["floors_check"]
    items["6_floors_and_cap_hold"] = {"pass": bool(fc["ok"]), **{k: v for k, v in fc.items() if k != "ok"}}
    items["7_hand_read_top10"] = {"pass": None, "detail": "manual: read top10_cards.md; ≥ 7 of 10 decision states"}
    informational = {"7_hand_read_top10", "3b_counterfactual_variant_informational"}
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
    crit["weights_sha256"] = wsha
    crit["ledger_sha256"] = lsha
    crit["computed_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    write_json(crit, snapshot / "criterion.json")
    (snapshot / "diff.md").write_text(build_diff(snapshot, prev_dir), encoding="utf-8")

    for_epoch = int(rule_doc["corpus_epoch"]) + 1
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
                            "for_epoch": for_epoch, "counts_as_shadow_fold": crit["counts_as_shadow_fold"],
                            "automatic_items_pass": crit["automatic_items_pass"],
                            "items": {k: v["pass"] for k, v in crit["items"].items()}}, sort_keys=True) + "\n")
    moves = sorted(((g, shares[g] - (groups["groups"][g]["share_current"] or 0)) for g in shares),
                   key=lambda kv: -abs(kv[1]))[:3]
    line = (f"curriculum {rule_doc['mode']} v{rule_doc['rule_version']}: ledger `{lsha[:12]}` "
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
