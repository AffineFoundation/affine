"""Join states + continuation results into the `admit` side-table and rates.

  python aggregate.py --states states.jsonl [states_extra.jsonl ...] \
      --out <out dir> --side-table recoverable/

Writes
  <side-table>/<digest12>.jsonl   one row per state:
      turn_id, node_id, rollout_id, turn_idx, state_kind, onset_rank, harness,
      source, teacher_solved, teacher_first_action_differs, admit,
      teacher_status, teacher_stop, teacher_turns, teacher_first_action,
      king_action, wiki_judge, wiki_contains, model, timestamp,
      continuations [per-continuation results], n_continuations, n_solved,
      n_differs, continuations_target, pending_reruns, rule,
      admit_first_continuation, admit_any_continuation
  <out>/summary.json + <out>/summary.md   recoverable rates by state kind,
      harness, env (source), depth bucket, onset rank; onset/pivot overlap;
      replay fidelity; wall time; cost.

Rule (3-continuation, 2026-09-13): a state is run `--continuations` times
(3, T = 0.8; run_states.py). `teacher_solved` = solved in a MAJORITY of the
OK continuations (2 of 3), `teacher_first_action_differs` = the first action
differs from the king's in a majority, `admit` = both. Rows with fewer OK
continuations than the target are judged on what ran (1 of 1 keeps the
pre-rule verdict) and carry `pending_reruns` = the missing count, which
candidates.py turns into re-run candidates. Why: one continuation at
T = 0.8 is a noisy label — the unhinted re-run of the "failed" states
recovered 35 % of them (hinted-teacher explorations §E4b).
`admit_first_continuation` keeps the old one-sample verdict for comparison.

Wiki states take `solved` from grade_wiki.py (the taskset publishes no
reward); every other source from the env's primary reward (>= 1.0).
--merge-into unions a state's continuations with the ones already in the
table (by trace id; legacy single-result rows are converted).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402

sys.path.insert(0, str(HERE))
from grade_wiki import undecided  # noqa: E402

FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
WS_RE = re.compile(r"\s+")
DEPTH_BUCKETS = ((0, 10), (10, 20), (20, 40), (40, 81))


def action_of(reply: dict | None, action_kind: str) -> str:
    if not reply:
        return ""
    if reply.get("tool_calls"):
        calls = []
        for tc in reply["tool_calls"]:
            fn = tc.get("function") or tc
            calls.append({"name": fn.get("name"), "arguments": fn.get("arguments")})
        return json.dumps(calls, sort_keys=True)
    text = FOREIGN_FENCE_RE.sub("```bash\n", reply.get("content") or "")
    try:
        return dialects.last_action(text, action_kind)
    except dialects.UnknownDialect:
        return ""


def norm(s: str) -> str:
    return WS_RE.sub(" ", s or "").strip()


def digest12(king_model: str) -> str:
    m = re.search(r"king-([0-9a-f]{12})", king_model or "")
    return m.group(1) if m else "unknown"


def depth_bucket(d: int) -> str:
    for lo, hi in DEPTH_BUCKETS:
        if lo <= d < hi:
            return f"{lo}-{hi - 1}"
    return f"{DEPTH_BUCKETS[-1][1]}+"


def rate_table(rows: list[dict], key) -> dict:
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        groups[str(key(r))].append(r)
    out = {}
    for k, rs in sorted(groups.items()):
        ran = [r for r in rs if r["teacher_status"] == "ok"]
        solved = sum(1 for r in ran if r["teacher_solved"])
        differs = sum(1 for r in ran if r["teacher_first_action_differs"])
        admit = sum(1 for r in ran if r["admit"])
        out[k] = {"states": len(rs), "ran": len(ran),
                  "errored": sum(1 for r in rs if r["teacher_status"] == "errored"),
                  "ungraded": sum(1 for r in rs if r["teacher_status"] == "ungraded"),
                  "not_run": sum(1 for r in rs if r["teacher_status"] == "not_run"),
                  "solved": solved,
                  "solved_rate": round(solved / len(ran), 3) if ran else None,
                  "first_action_differs": differs,
                  "admit": admit,
                  "admit_rate": round(admit / len(ran), 3) if ran else None}
    return out


def md_table(title: str, table: dict) -> str:
    lines = [f"### {title}", "",
             "| group | states | ran | errored | ungraded | not run | teacher solved | solved rate | first action differs | admit | admit rate |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for k, v in table.items():
        sr = "-" if v["solved_rate"] is None else f"{100 * v['solved_rate']:.0f}%"
        ar = "-" if v["admit_rate"] is None else f"{100 * v['admit_rate']:.0f}%"
        lines.append(f"| {k} | {v['states']} | {v['ran']} | {v['errored']} | {v['ungraded']} | "
                     f"{v['not_run']} | {v['solved']} | {sr} | "
                     f"{v['first_action_differs']} | {v['admit']} | {ar} |")
    return "\n".join(lines) + "\n"


def onset_ranks(states: list[dict]) -> dict[str, int]:
    """state_id -> 1-based rank of a loop_onset among its rollout's onsets
    (by turn_idx). Pivots get 0."""
    by_rollout: dict[str, list[dict]] = collections.defaultdict(list)
    for st in states:
        if st["state_kind"] == "loop_onset":
            by_rollout[st["rollout_id"]].append(st)
    ranks: dict[str, int] = {}
    for sts in by_rollout.values():
        for i, st in enumerate(sorted(sts, key=lambda s: s["turn_idx"]), start=1):
            ranks[st["state_id"]] = i
    return ranks


def p50(values: list) -> float | None:
    vals = sorted(v for v in values if v is not None)
    return vals[len(vals) // 2] if vals else None


def continuation_of(r: dict, st: dict) -> dict:
    """One stored result (run_states.py) -> the per-continuation record kept
    in the side-table row."""
    ok = r.get("status") == "ok"
    wiki_judge = wiki_contains = None
    if st["source"] == "affine_wiki":
        grade = r.get("wiki_grade")
        if grade is None or undecided(grade):
            ok = False
        solved = bool(grade and grade.get("solved"))
        wiki_judge = (grade or {}).get("judge")
        wiki_contains = (grade or {}).get("contains")
    else:
        score = r.get("reward_score")
        solved = ok and score is not None and float(score) >= 1.0
    first = action_of(r.get("first_reply"), st["action_kind"])
    first_reply = r.get("first_reply") or {}
    return {
        "trace_id": r.get("trace_id"),
        "k": int(r.get("continuation") or 0),
        "started_at": r.get("started_at"),
        "status": "ok" if ok else ("ungraded" if r.get("status") == "ok" else "errored"),
        "solved": solved if ok else None,
        "first_action_differs": (norm(first) != norm(st.get("king_action") or "")) if ok else None,
        "first_action": first,
        "first_action_kind": ("tool_call" if first_reply.get("tool_calls")
                              else st["action_kind"] if first else "text"),
        "first_reply_head": norm(first_reply.get("content") or "")[:300],
        "stop": r.get("stop_condition"),
        "turns": r.get("n_turns"),
        "wall_s": r.get("wall_s"),
        "cost_usd": r.get("cost_usd"),
        "error": (r.get("error") or "")[:300] if not ok else None,
        "wiki_judge": wiki_judge,
        "wiki_contains": wiki_contains,
    }


def legacy_continuations(row: dict) -> list[dict]:
    """A pre-rule side-table row (one result, scalar teacher_* fields) as a
    one-element continuation list; not_run rows have none."""
    if row.get("continuations") is not None:
        return list(row["continuations"])
    if row.get("teacher_status") in (None, "not_run"):
        return []
    return [{
        "trace_id": None, "k": 0, "started_at": row.get("timestamp"), "legacy": True,
        "status": row.get("teacher_status"),
        "solved": row.get("teacher_solved"),
        "first_action_differs": row.get("teacher_first_action_differs"),
        "first_action": row.get("teacher_first_action"),
        "first_action_kind": row.get("teacher_first_action_kind"),
        "first_reply_head": row.get("teacher_first_reply_head"),
        "stop": row.get("teacher_stop"), "turns": row.get("teacher_turns"),
        "wall_s": row.get("teacher_wall_s"), "cost_usd": None,
        "error": row.get("teacher_error"),
        "wiki_judge": row.get("wiki_judge"), "wiki_contains": row.get("wiki_contains"),
    }]


def fingerprint(c: dict) -> tuple:
    return (c.get("stop"), c.get("turns"), c.get("wall_s"), (c.get("first_action") or "")[:200])


def merge_continuations(old: list[dict], new: list[dict]) -> list[dict]:
    """Union by identity — the trace id, or for a legacy entry (converted
    from a pre-rule row, no trace id) a fingerprint of what the row stored;
    a legacy entry whose fingerprint matches a real one is the same
    continuation re-aggregated and is dropped. New wins on a clash;
    earliest first."""
    by_key: dict[tuple, dict] = {}
    for c in [*old, *new]:
        key = ("trace", c["trace_id"]) if c.get("trace_id") else ("fp", *fingerprint(c))
        by_key[key] = c
    real_fps = {fingerprint(c) for c in by_key.values() if c.get("trace_id")}
    conts = [c for c in by_key.values() if c.get("trace_id") or fingerprint(c) not in real_fps]
    return sorted(conts, key=lambda c: (c.get("started_at") or "", c.get("k") or 0))


def apply_verdict(row: dict, conts: list[dict], target: int) -> dict:
    """Majority rule over the OK continuations; scalar teacher_* fields come
    from a representative continuation (a solved one when there is one) so
    readers of the pre-rule layout (the fold copies teacher_turns /
    teacher_first_action_kind) keep working."""
    ok = [c for c in conts if c["status"] == "ok"]
    n_ok = len(ok)
    n_solved = sum(1 for c in ok if c["solved"])
    n_differs = sum(1 for c in ok if c["first_action_differs"])
    solved = (n_solved * 2 > n_ok) if n_ok else None
    differs = (n_differs * 2 > n_ok) if n_ok else None
    rep = next((c for c in ok if c["solved"]), None) or (ok[0] if ok else (conts[0] if conts else None))
    if n_ok:
        status = "ok"
    elif any(c["status"] == "errored" for c in conts):
        status = "errored"
    elif conts:
        status = "ungraded"
    else:
        status = "not_run"
    first = ok[0] if ok else None          # the pre-rule verdict: one valid sample
    row.update(
        continuations=conts, n_continuations=n_ok, n_solved=n_solved, n_differs=n_differs,
        continuations_target=target, pending_reruns=max(0, target - n_ok),
        rule=f"majority_of_{target}",
        teacher_status=status, teacher_solved=solved, teacher_first_action_differs=differs,
        admit=bool(solved and differs),
        admit_first_continuation=bool(first and first["solved"] and first["first_action_differs"]),
        admit_any_continuation=any(c["solved"] and c["first_action_differs"] for c in ok),
        teacher_stop=rep["stop"] if rep else None,
        teacher_turns=rep["turns"] if rep else None,
        teacher_wall_s=rep["wall_s"] if rep else None,
        teacher_first_action=rep["first_action"] if rep else None,
        teacher_first_action_kind=rep["first_action_kind"] if rep else None,
        teacher_first_reply_head=rep["first_reply_head"] if rep else None,
        teacher_error=(None if status == "ok"
                       else next((c["error"] for c in conts if c.get("error")), None)),
        wiki_judge=rep.get("wiki_judge") if rep else None,
        wiki_contains=rep.get("wiki_contains") if rep else None,
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, type=Path, nargs="+",
                    help="states.jsonl files (e.g. first onsets + pivots, later onsets)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--side-table", type=Path,
                    help="directory: writes <digest12>.jsonl with this run's rows")
    ap.add_argument("--merge-into", type=Path,
                    help="existing side-table file: this run's rows replace rows "
                         "with the same (rollout_id, turn_idx, state_kind), every "
                         "other row is kept; rewritten atomically")
    ap.add_argument("--model", default="engy2/qwen3.8-27b")
    ap.add_argument("--continuations", type=int, default=3,
                    help="OK continuations a state needs to be complete "
                         "(majority rule; pending_reruns = the missing count)")
    args = ap.parse_args()
    if not args.side_table and not args.merge_into:
        ap.error("one of --side-table / --merge-into is required")
    target = args.continuations

    states: list[dict] = []
    seen: set[str] = set()
    for path in args.states:
        for line in open(path, encoding="utf-8"):
            st = json.loads(line)
            if st["state_id"] not in seen:
                seen.add(st["state_id"])
                states.append(st)
    ranks = onset_ranks(states)
    results: dict[str, list[dict]] = collections.defaultdict(list)
    for p in (args.out / "results").glob("*.json"):
        r = json.loads(p.read_text())
        results[r["state_id"]].append(r)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    rows = []
    cost = 0.0
    tokens = collections.Counter()
    for st in states:
        row = {
            "turn_id": st["turn_id"], "node_id": st.get("node_id"),
            "rollout_id": st["rollout_id"], "turn_idx": st["turn_idx"],
            "state_kind": st["state_kind"], "onset_rank": ranks.get(st["state_id"], 0),
            "harness": st["harness"],
            "source": st["source"], "policy_id": st["policy_id"],
            "king_action": st.get("king_action") or "",
            "model": args.model, "timestamp": stamp,
        }
        conts = []
        for r in results.get(st["state_id"], []):
            cost += float(r.get("cost_usd") or 0.0)
            for key in ("prompt_tokens", "completion_tokens", "cached_tokens"):
                tokens[key] += int(r.get(key) or 0)
            conts.append(continuation_of(r, st))
        rows.append(apply_verdict(row, merge_continuations([], conts), target))

    digest = digest12(next((s.get("king_model") for s in states if s.get("king_model")), ""))
    run_rows = rows
    if args.merge_into:
        merged = {}
        if args.merge_into.is_file():
            for line in args.merge_into.read_text(encoding="utf-8").split("\n"):
                if line.strip():
                    row = json.loads(line)
                    merged[(row["rollout_id"], int(row["turn_idx"]), row["state_kind"])] = row
        n_before = len(merged)
        for row in run_rows:
            key = (row["rollout_id"], int(row["turn_idx"]), row["state_kind"])
            old = merged.get(key)
            if old is not None:
                # Union with what the table already holds for this state:
                # this run adds continuations, it does not restart the count.
                conts = merge_continuations(legacy_continuations(old), row["continuations"])
                row["onset_rank"] = row["onset_rank"] or old.get("onset_rank", 0)
                apply_verdict(row, conts, target)
            merged[key] = row
        for key, row in merged.items():
            if row.get("continuations") is None:
                # Pre-rule row: same verdict (1 of 1), now with the
                # continuation list and its pending_reruns marker.
                apply_verdict(row, legacy_continuations(row), target)
        rows = list(merged.values())
        args.merge_into.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.merge_into.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        os.chmod(tmp, 0o644)
        tmp.replace(args.merge_into)
        table_path = args.merge_into
        print(f"merged {len(run_rows)} row(s) into {table_path}: {n_before} -> {len(rows)} rows")
    else:
        args.side_table.mkdir(parents=True, exist_ok=True)
        table_path = args.side_table / f"{digest}.jsonl"
        with open(table_path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Rate, stop, turn and wall-time tables cover the whole (merged) table;
    # cost, tokens and replay fidelity come from this run's results only.
    ran = [r for r in rows if r["teacher_status"] != "not_run"]
    summary = {
        "king_digest12": digest, "generated_at": stamp, "model": args.model,
        "rule": f"majority_of_{target}",
        "n_states": len(rows), "n_ran": len(ran),
        "n_admit": sum(1 for r in rows if r["admit"]),
        "n_admit_first_continuation": sum(1 for r in rows if r.get("admit_first_continuation")),
        "n_admit_any_continuation": sum(1 for r in rows if r.get("admit_any_continuation")),
        "n_continuations": sum(r.get("n_continuations", 0) for r in rows),
        "n_states_complete": sum(1 for r in rows if r.get("pending_reruns") == 0),
        "n_pending_reruns": sum(r.get("pending_reruns", 0) for r in rows),
        "by_n_continuations": rate_table(rows, lambda r: r.get("n_continuations", 0)),
        "n_states_this_run": len(run_rows),
        "n_admit_this_run": sum(1 for r in run_rows if r["admit"]),
        "n_continuations_this_run": sum(len(v) for v in results.values()),
        "cost_usd": round(cost, 2),
        "tokens": dict(tokens),
        "by_state_kind": rate_table(rows, lambda r: r["state_kind"]),
        "by_harness": rate_table(rows, lambda r: r["harness"]),
        "by_source": rate_table(rows, lambda r: r["source"]),
        "by_depth": rate_table(rows, lambda r: depth_bucket(r["turn_idx"])),
        "by_kind_and_harness": rate_table(rows, lambda r: f"{r['state_kind']}/{r['harness']}"),
        "by_onset_rank": rate_table(
            [r for r in rows if r["state_kind"] == "loop_onset"],
            lambda r: "first onset" if r["onset_rank"] == 1 else "later onset"),
    }
    docker = [r for r in rows if r["harness"] != "null"]
    summary["by_depth_agent_harnesses"] = rate_table(docker, lambda r: depth_bucket(r["turn_idx"]))
    summary["by_onset_rank_agent_harnesses"] = rate_table(
        [r for r in docker if r["state_kind"] == "loop_onset"],
        lambda r: "first onset" if r["onset_rank"] == 1 else "later onset")
    summary["teacher_stop_by_harness"] = {
        h: dict(collections.Counter(r["teacher_stop"] for r in ran if r["harness"] == h))
        for h in sorted({r["harness"] for r in ran})}
    turns = collections.defaultdict(list)
    walls = collections.defaultdict(list)
    for r in ran:
        if r["teacher_turns"] is not None:
            turns[r["harness"]].append(r["teacher_turns"])
        if r.get("teacher_wall_s") is not None:
            walls[r["harness"]].append(r["teacher_wall_s"])
    summary["teacher_turns_p50_by_harness"] = {h: p50(v) for h, v in turns.items() if v}
    summary["wall_time_by_harness"] = {
        h: {"states": len(v), "p50_s": p50(v), "max_s": max(v),
            "total_h": round(sum(v) / 3600, 1)}
        for h, v in walls.items() if v}
    summary["wall_time_total_h"] = round(sum(sum(v) for v in walls.values()) / 3600, 1)
    # Replay fidelity (agent harnesses): replayed prefix commands whose
    # output matched the recorded observation byte for byte. Only the
    # harnesses whose replay records a comparison carry a rate: textbased
    # records output + return code, bash records output only (tool results
    # carry no rc), terminus replays keystrokes (no per-command comparison;
    # `timed_out` counts batches that hit the replay timeout).
    fid: dict[str, dict] = {}
    for st in states:
        for r in results.get(st["state_id"], []):
            rep = r.get("report") or {}
            if not rep.get("replay_n"):
                continue
            f = fid.setdefault(st["harness"], {"states": 0, "commands": 0, "output_match": None,
                                               "rc_match": None, "timed_out": None,
                                               "replay_seconds": []})
            f["states"] += 1
            f["commands"] += rep["replay_n"]
            if "replay_match" in rep:
                f["output_match"] = (f["output_match"] or 0) + rep["replay_match"]
            if "replay_rc_match" in rep:
                f["rc_match"] = (f["rc_match"] or 0) + rep["replay_rc_match"]
            timed_out = [x for x in rep.get("replay") or []
                         if isinstance(x, dict) and x.get("timed_out")]
            if any(isinstance(x, dict) and "timed_out" in x for x in rep.get("replay") or []):
                f["timed_out"] = (f["timed_out"] or 0) + len(timed_out)
            f["replay_seconds"].append(rep.get("replay_seconds", 0))
    for f in fid.values():
        secs = f.pop("replay_seconds")
        f["replay_seconds_p50"] = p50(secs)
        f["replay_seconds_max"] = max(secs) if secs else None
        for key in ("output_match", "rc_match"):
            f[key + "_rate"] = (round(f[key] / f["commands"], 3)
                                if f[key] is not None and f["commands"] else None)
    summary["replay_fidelity"] = fid
    # Overlap: a rollout may carry both an onset and a pivot state.
    onset_admit = {(r["rollout_id"]) for r in ran if r["state_kind"] == "loop_onset" and r["admit"]}
    pivot_admit = {(r["rollout_id"]) for r in ran if r["state_kind"] == "pivot" and r["admit"]}
    both = {r["rollout_id"] for r in ran if r["state_kind"] == "loop_onset"} & \
        {r["rollout_id"] for r in ran if r["state_kind"] == "pivot"}
    same_turn = collections.Counter((r["rollout_id"], r["turn_idx"]) for r in ran)
    summary["overlap"] = {
        "rollouts_with_both_state_kinds": len(both),
        "rollouts_admitted_by_both": len(onset_admit & pivot_admit),
        "rollouts_admitted_by_onset_only": len(onset_admit - pivot_admit),
        "rollouts_admitted_by_pivot_only": len(pivot_admit - onset_admit),
        "states_at_the_same_turn": sum(1 for v in same_turn.values() if v > 1),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    md = [f"# Teacher-recoverable filter — king {digest}", "",
          f"states {len(rows)}, ran {len(ran)}, admitted {summary['n_admit']} "
          f"(this run: {len(run_rows)} states, {summary['n_admit_this_run']} admitted), "
          f"teacher {args.model}, cost ${cost:.2f}, tokens {dict(tokens)}, "
          f"wall {summary['wall_time_total_h']} h", "",
          f"rule {summary['rule']}: {summary['n_continuations']} OK continuations over "
          f"{len(rows)} states, {summary['n_states_complete']} states complete, "
          f"{summary['n_pending_reruns']} continuations pending; admitted under the "
          f"one-sample rule (first continuation) {summary['n_admit_first_continuation']}, "
          f"by any continuation {summary['n_admit_any_continuation']}, "
          f"by majority {summary['n_admit']}", ""]
    for title, key in (("By state kind", "by_state_kind"), ("By harness", "by_harness"),
                       ("By number of OK continuations", "by_n_continuations"),
                       ("By env (source)", "by_source"), ("By depth (turn_idx)", "by_depth"),
                       ("By depth, agent harnesses only", "by_depth_agent_harnesses"),
                       ("By onset rank (loop onsets)", "by_onset_rank"),
                       ("By onset rank, agent harnesses only", "by_onset_rank_agent_harnesses"),
                       ("By state kind × harness", "by_kind_and_harness")):
        md.append(md_table(title, summary[key]))
    md.append("### Teacher stop condition by harness\n\n"
              + "\n".join(f"- {h}: {v}" for h, v in summary["teacher_stop_by_harness"].items())
              + "\n\n### Teacher turns used (p50) by harness\n\n"
              + "\n".join(f"- {h}: {v}" for h, v in summary["teacher_turns_p50_by_harness"].items())
              + "\n\n### Wall time by harness\n\n"
              + "\n".join(f"- {h}: {v}" for h, v in summary["wall_time_by_harness"].items())
              + "\n\n### Replay fidelity\n\n"
              + "\n".join(f"- {h}: {v}" for h, v in summary["replay_fidelity"].items()) + "\n")
    md.append("### Overlap\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary["overlap"].items()) + "\n")
    (args.out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"side-table -> {table_path}")


if __name__ == "__main__":
    main()
