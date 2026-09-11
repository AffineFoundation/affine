"""Step 4 -- the side-table of judged pivotal turns + the proposed fold patch.

Writes `<out-dir>/king_pivots/<digest>.jsonl`: one row per (rollout, judged
pivotal turn) with the ids the fold needs (`rollout_id`, `traj_id`,
`turn_id = traj_id:turn_idx`, `turn_idx`, `node_id`), the judge's
category / rationale / confidence, the deterministic label of the same
turn, the judge model and the prompt hash. `admit` is true when the
judge's confidence is >= --min-confidence (0.7): that is the row the fold
would route into a `king_pivot` group in phase 2.

Also writes `<out-dir>/king_pivots/route_king_pivot.proposed.patch.txt`:
a PROPOSED diff for ops/corpus_build.py + rollouts/rollouts/sources.toml
(group `king_pivot`, strata `king_pivot:<sha256(task) % 1000>`, leak-rule
exemption like `king_loop_onset`). It is text, not applied: those two files
belong to the fold worker.

  python labels_out.py --out-dir <dir> [--min-confidence 0.7]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from aggregate import all_pivots, category_of, load_judgments
from krlib import read_jsonl, write_jsonl

DEFAULT_MIN_CONFIDENCE = 0.7
STRATA_BUCKETS = 1000


def det_label_at(rec: dict, turn: int) -> list[str]:
    d = rec["det"]
    out = []
    if turn in d["loop_onsets"]:
        out.append("loop_onset")
    elif turn in d["in_loop"]:
        out.append("in_loop")
    elif turn in d["escape"]:
        out.append("escape")
    else:
        out.append("normal")
    if turn in d["no_action"]:
        out.append("no_action")
    if turn in d["token_cap"]:
        out.append("token_cap")
    if turn in d["completion"]:
        out.append("completion")
    if turn == d["last_turn"]:
        out.append("last_turn")
    return out


def side_table(recs: list[dict], *, min_confidence: float, reign: str | None) -> list[dict]:
    rows: list[dict] = []
    for rec in recs:
        s2 = rec.get("stage2") or {}
        rp = s2.get("recoverable_from_pivot") if isinstance(s2.get("recoverable_from_pivot"), dict) else {}
        for rank, (p, src) in enumerate(all_pivots(rec), 1):
            turn = int(p["turn"])
            try:
                conf = float(p.get("confidence"))
            except (TypeError, ValueError):
                conf = 0.0
            node_ids = rec.get("node_ids") or []
            no_action = turn in rec["det"]["no_action"]
            rows.append({
                "king": rec["king"], "reign": reign, "rollout_id": rec["rollout_id"],
                "traj_id": rec["traj_id"], "turn_id": f"{rec['traj_id']}:{turn}",
                "turn_idx": turn, "node_id": node_ids[turn] if turn < len(node_ids) else None,
                "n_turns": rec["n_turns"], "sid": rec["sid"], "source": rec["source"],
                "env_group": rec["env_group"], "harness": rec["harness"],
                "policy_id": rec["policy_id"], "action_kind": rec["action_kind"],
                "failure_category": category_of(rec),
                "secondary_categories": s2.get("secondary_categories") or [],
                "pivot_rank": rank, "pivot_source": src,
                "pivot_pattern": s2.get("pivot_pattern"),
                "rationale": p.get("rationale"), "should_have": p.get("should_have"),
                "confidence": round(conf, 3),
                "recoverable": rp.get("estimate"), "recoverable_confidence": rp.get("confidence"),
                "det_labels": det_label_at(rec, turn),
                # The fold can only route a turn that has exactly one action;
                # a no-action pivot is reported but cannot enter D as is.
                "admit": conf >= min_confidence and not no_action and rank == 1,
                "judge_model": rec["judge_model"], "prompt_version": rec["prompt_version"],
                "prompt_hash": rec["prompt_hash"], "judged_at": rec["judged_at"],
            })
    return rows


PATCH_TEMPLATE = '''PROPOSED (not applied) -- phase 2 of the per-reign king review.
Target tree: ops/corpus_build.py + rollouts/rollouts/sources.toml AFTER the
king_loop_onset change (PR #4, branch cursor/king-loop-onset-fold-8929),
which already carries the leak-exempt split path this group reuses.

Side-table produced by ops/king-review/labels_out.py:
  {side_table_path}
  rows: {n_rows} pivots from {n_rollouts} judged rollouts of king-{digest};
  admissible (rank 1, confidence >= {min_conf}, turn has one action): {n_admit}

--- a/rollouts/rollouts/sources.toml
+++ b/rollouts/rollouts/sources.toml
@@ [mix] @@
 king_fail = 0.07
 king_loop_onset = 0.03
+# 2026-09-11: + king_pivot -- the turns an LLM judge marked as the decision
+# point of a failed king rollout (ops/king-review). Share taken out of
+# king_fail so the five real groups do not move.
+king_pivot = 0.02
 (and king_fail 0.07 -> 0.05 so the table still sums to 1.0)

+[king_pivot]
+# Judged pivotal turns of the king seat's failed rollouts. Rows come from
+# the side-table written per reign by ops/king-review (labels_out.py):
+# one JSON line per (rollout_id, turn_idx) with `admit`, `confidence`,
+# `failure_category`. Only rows with admit == true (judge confidence >=
+# min_confidence, the primary pivot, one parseable action) are routed.
+# Strata `king_pivot:NNNN` = sha256(instance_id) % strata_buckets, own
+# namespace like king_fail / king_loop_onset. The leak rule
+# (reference_leaked_into_prefix) is waived for these turns only, as for
+# king_loop_onset: a pivot that re-issues an earlier command is by
+# construction a "leak", and the duel never echoes the stored reference.
+strata_buckets = {buckets}
+policy_prefix = "king_"
+min_confidence = {min_conf}
+side_table_dir = "state/king_pivots"   # <digest>.jsonl per king, synced from the review
+exclude_categories = ["format_error"]  # no-action replies cannot enter D anyway

--- a/ops/corpus_build.py
+++ b/ops/corpus_build.py
@@ after load_king_loop_onset @@
+KING_PIVOT_GROUP = "king_pivot"
+
+
+def load_king_pivot() -> dict:
+    """[king_pivot] from sources.toml + the per-king side-tables under
+    side_table_dir: {{rollout_id: {{turn_idx: row}}}} for admitted rows."""
+    raw = tomllib.loads(SOURCES_TOML.read_text())
+    cfg = raw.get(KING_PIVOT_GROUP) or {{}}
+    if not cfg:
+        return {{}}
+    table: dict[str, dict[int, dict]] = {{}}
+    excluded = set(cfg.get("exclude_categories") or [])
+    min_conf = float(cfg.get("min_confidence", {min_conf}))
+    for path in sorted((REPO / cfg["side_table_dir"]).glob("*.jsonl")):
+        for row in iter_jsonl(path):
+            if not row.get("admit") or row.get("failure_category") in excluded:
+                continue
+            if float(row.get("confidence") or 0) < min_conf:
+                continue
+            table.setdefault(row["rollout_id"], {{}})[int(row["turn_idx"])] = row
+    return {{"group": KING_PIVOT_GROUP,
+            "strata_buckets": int(cfg.get("strata_buckets", 0) or 0),
+            "policy_prefix": str(cfg.get("policy_prefix") or "king_"),
+            "table": table}}
+
+
+def king_pivot_stratum(rec: dict, cfg: dict) -> str:
+    key = str(rec.get("instance_id") or rec.get("traj_id"))
+    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
+    return f"{{cfg['group']}}:{{h % cfg['strata_buckets']:04d}}"
+
@@ route_king_fail: pass pre-routed records through @@
-        if rec.get("fold_group") == KING_LOOP_GROUP:
+        if rec.get("fold_group") in (KING_LOOP_GROUP, KING_PIVOT_GROUP):
             out.append(rec)
             continue
@@ derive_chunk(..., king_loop=None, king_pivot=None, notes=None) @@
     for env in iter_jsonl_gz(path):
         convs = None
         onsets: dict[int, int] = {{}}
         in_loop: set[int] = set()
+        pivots: dict[int, dict] = {{}}
+        if king_pivot and king_pivot["strata_buckets"] > 0:
+            pivots = dict(king_pivot["table"].get(env.get("rollout_id", ""), {{}}))
+            # a turn that is both an onset and a pivot stays an onset
+            # (one group per turn; king_loop_onset was there first)
+            _count(notes, "king_pivot_rollouts", 1 if pivots else 0)
         ...
         rec = build_view_record(env, baker=baker, generated_at=env.get("stored_at"),
-                                convs=convs, leak_exempt=frozenset(onsets))
+                                convs=convs,
+                                leak_exempt=frozenset(onsets) | frozenset(pivots))
         ...
         turns = view_turns(rec)
         onset_turns = [t for t in turns if t["turn_idx"] in onsets]
+        pivot_turns = [t for t in turns if t["turn_idx"] in pivots
+                       and t["turn_idx"] not in onsets]
         rest = [t for t in turns if t["turn_idx"] not in onsets
-                and t["turn_idx"] not in in_loop]
+                and t["turn_idx"] not in in_loop and t["turn_idx"] not in pivots]
         ...
+        kept_pivots: list[dict] = []
+        if pivot_turns:
+            kept_pivots, d = validate_turns(pivot_turns, panel=panel,
+                                            allowed_kinds=allowed_kinds,
+                                            leak_check=False)
+            for k, v in d.items():
+                _count(drops, k, v)
         keep_idx: set[int] = set()
         keep_onset_idx: set[int] = set()
+        keep_pivot_idx: set[int] = set()
-        for t in [*kept, *kept_onsets]:
+        for t in [*kept, *kept_onsets, *kept_pivots]:
             ... (published / token-cap checks unchanged) ...
             if t["turn_idx"] in onsets:
                 keep_onset_idx.add(t["turn_idx"])
+            elif t["turn_idx"] in pivots:
+                keep_pivot_idx.add(t["turn_idx"])
+                _count(notes, "king_pivot_admitted")
             else:
                 keep_idx.add(t["turn_idx"])
         ...
+        if keep_pivot_idx:
+            pivot_rec = dict(rec)
+            pivot_rec["turns"] = [
+                {{**m, "pivot": {{"category": pivots[m["turn_idx"]]["failure_category"],
+                                "confidence": pivots[m["turn_idx"]]["confidence"],
+                                "judge": pivots[m["turn_idx"]]["judge_model"],
+                                "prompt_hash": pivots[m["turn_idx"]]["prompt_hash"]}}}}
+                for m in metas if m["turn_idx"] in keep_pivot_idx]
+            pivot_rec["fold_group"] = KING_PIVOT_GROUP
+            pivot_rec["stratum"] = king_pivot_stratum(pivot_rec, king_pivot)
+            out.append(pivot_rec)
@@ main() @@
+    king_pivot = load_king_pivot()
+    if king_pivot and mix.get(KING_PIVOT_GROUP, 0.0) <= 0:
+        fatal(f"[{{KING_PIVOT_GROUP}}] is configured but [mix] has no {{KING_PIVOT_GROUP}} share")
+    log(f"king pivots: {{'off' if not king_pivot else len(king_pivot['table'])}} rollouts with admitted pivots")
     ...
-        recs = derive_chunk(path, baker, panel, allowed, published, drops,
-                            king_loop=king_loop, notes=notes)
+        recs = derive_chunk(path, baker, panel, allowed, published, drops,
+                            king_loop=king_loop, king_pivot=king_pivot, notes=notes)
     ...
+    (after stamp_king_loop_onset) re-stamp `king_pivot:NNNN` strata on
+    fold_group == KING_PIVOT_GROUP records (same idempotent pattern) and log
+    the per-epoch count; `by_group` in the announce shows king_pivot.

Notes for the fold worker
- Already-published turns: a pivot turn that entered D earlier as a plain
  `king_fail` turn is skipped by the `published` set (turn_id unchanged), so
  routing pivots is forward-only unless `--rederive` re-splits it; the
  side-table `turn_id` field is exactly `traj_id:turn_idx` for that check.
- Deferred carryover records already carry `fold_group`; `route_king_fail`
  passes them through by the same test as king_loop_onset.
- The side-table is per king digest; a fold should read every table under
  side_table_dir so earlier kings' pivots stay routable (their prefixes are
  still valid states).
- Rows with `pivot_source == "revealed"` (the judge gave no blind pivot and
  named one only after seeing the grade) are admitted by the same rule; if
  that is too soft, filter on pivot_source == "blind" in load_king_pivot.
'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--sample", type=Path, default=None)
    ap.add_argument("--reign", default=None)
    ap.add_argument("--min-confidence", type=float, default=DEFAULT_MIN_CONFIDENCE)
    args = ap.parse_args()
    sample = read_jsonl(args.sample or (args.out_dir / "sample.jsonl"))
    recs = load_judgments(args.out_dir, sample)
    if not recs:
        raise SystemExit("no judgments for this sample yet")
    digest = sample[0]["king"].replace("king-", "")
    rows = side_table(recs, min_confidence=args.min_confidence, reign=args.reign)
    out_dir = args.out_dir / "king_pivots"
    table_path = out_dir / f"{digest}.jsonl"
    write_jsonl(table_path, rows)
    n_admit = sum(1 for r in rows if r["admit"])
    patch = PATCH_TEMPLATE.format(side_table_path=table_path, n_rows=len(rows),
                                  n_rollouts=len(recs), digest=digest,
                                  min_conf=args.min_confidence, n_admit=n_admit,
                                  buckets=STRATA_BUCKETS)
    (out_dir / "route_king_pivot.proposed.patch.txt").write_text(patch)
    print(f"side-table: {table_path} ({len(rows)} pivot rows, {n_admit} admissible at "
          f"confidence >= {args.min_confidence}); patch: "
          f"{out_dir / 'route_king_pivot.proposed.patch.txt'}")


if __name__ == "__main__":
    main()
