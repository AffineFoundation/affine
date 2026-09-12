"""Join states + continuation results into the `admit` side-table and rates.

  python aggregate.py --states states.jsonl --out <out dir> --side-table recoverable/

Writes
  <side-table>/<digest12>.jsonl   one row per state:
      turn_id, node_id, rollout_id, turn_idx, state_kind, harness, source,
      teacher_solved, teacher_first_action_differs, admit, teacher_status,
      teacher_stop, teacher_turns, teacher_first_action, king_action,
      model, timestamp
  <out>/summary.json + <out>/summary.md   recoverable rates by state kind,
      harness, env (source), depth bucket; onset/pivot overlap; cost.

`admit` = teacher solved the task from the state AND the teacher's first
action differs from the king's action at that state. Wiki states take
`teacher_solved` from grade_wiki.py (the taskset publishes no reward);
every other source from the env's primary reward (>= 1.0).
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402

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
        out[k] = {"states": len(rs), "ran": len(ran), "errored": len(rs) - len(ran),
                  "solved": solved,
                  "solved_rate": round(solved / len(ran), 3) if ran else None,
                  "first_action_differs": differs,
                  "admit": admit,
                  "admit_rate": round(admit / len(ran), 3) if ran else None}
    return out


def md_table(title: str, table: dict) -> str:
    lines = [f"### {title}", "",
             "| group | states | ran | errored | teacher solved | solved rate | first action differs | admit | admit rate |",
             "|---|---|---|---|---|---|---|---|---|"]
    for k, v in table.items():
        sr = "-" if v["solved_rate"] is None else f"{100 * v['solved_rate']:.0f}%"
        ar = "-" if v["admit_rate"] is None else f"{100 * v['admit_rate']:.0f}%"
        lines.append(f"| {k} | {v['states']} | {v['ran']} | {v['errored']} | {v['solved']} | {sr} | "
                     f"{v['first_action_differs']} | {v['admit']} | {ar} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--side-table", required=True, type=Path)
    ap.add_argument("--model", default="engy2/qwen3.8-27b")
    args = ap.parse_args()

    states = [json.loads(l) for l in open(args.states, encoding="utf-8")]
    results: dict[str, dict] = {}
    for p in (args.out / "results").glob("*.json"):
        r = json.loads(p.read_text())
        results[r["state_id"]] = r
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    rows = []
    cost = 0.0
    for st in states:
        r = results.get(st["state_id"])
        row = {
            "turn_id": st["turn_id"], "node_id": st.get("node_id"),
            "rollout_id": st["rollout_id"], "turn_idx": st["turn_idx"],
            "state_kind": st["state_kind"], "harness": st["harness"],
            "source": st["source"], "policy_id": st["policy_id"],
            "king_action": st.get("king_action") or "",
            "model": args.model, "timestamp": stamp,
        }
        if r is None:
            row.update(teacher_status="not_run", teacher_solved=None,
                       teacher_first_action_differs=None, admit=False,
                       teacher_stop=None, teacher_turns=None, teacher_first_action=None)
            rows.append(row)
            continue
        cost += float(r.get("cost_usd") or 0.0)
        ok = r.get("status") == "ok"
        if st["source"] == "affine_wiki":
            grade = r.get("wiki_grade")
            if grade is None:
                ok = False
            solved = bool(grade and grade.get("solved"))
        else:
            score = r.get("reward_score")
            solved = ok and score is not None and float(score) >= 1.0
        first = action_of(r.get("first_reply"), st["action_kind"])
        differs = norm(first) != norm(row["king_action"]) if ok else None
        row.update(
            teacher_status="ok" if ok else ("ungraded" if r.get("status") == "ok" else "errored"),
            teacher_solved=solved if ok else None,
            teacher_first_action_differs=differs,
            admit=bool(ok and solved and differs),
            teacher_stop=r.get("stop_condition"),
            teacher_turns=r.get("n_turns"),
            teacher_first_action=first,
            teacher_first_action_kind=("tool_call" if (r.get("first_reply") or {}).get("tool_calls")
                                       else st["action_kind"] if first else "text"),
            teacher_first_reply_head=norm(((r.get("first_reply") or {}).get("content") or ""))[:300],
            teacher_error=(r.get("error") or "")[:300] if not ok else None,
        )
        rows.append(row)

    digest = digest12(next((s.get("king_model") for s in states if s.get("king_model")), ""))
    args.side_table.mkdir(parents=True, exist_ok=True)
    table_path = args.side_table / f"{digest}.jsonl"
    with open(table_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ran = [r for r in rows if r["teacher_status"] != "not_run"]
    summary = {
        "king_digest12": digest, "generated_at": stamp, "model": args.model,
        "n_states": len(rows), "n_ran": len(ran),
        "cost_usd": round(cost, 2),
        "by_state_kind": rate_table(ran, lambda r: r["state_kind"]),
        "by_harness": rate_table(ran, lambda r: r["harness"]),
        "by_source": rate_table(ran, lambda r: r["source"]),
        "by_depth": rate_table(ran, lambda r: depth_bucket(r["turn_idx"])),
        "by_kind_and_harness": rate_table(ran, lambda r: f"{r['state_kind']}/{r['harness']}"),
    }
    docker = [r for r in ran if r["harness"] != "null"]
    summary["by_depth_agent_harnesses"] = rate_table(docker, lambda r: depth_bucket(r["turn_idx"]))
    summary["teacher_stop_by_harness"] = {
        h: dict(collections.Counter(r["teacher_stop"] for r in ran if r["harness"] == h))
        for h in sorted({r["harness"] for r in ran})}
    turns = collections.defaultdict(list)
    for r in ran:
        if r["teacher_turns"] is not None:
            turns[r["harness"]].append(r["teacher_turns"])
    summary["teacher_turns_p50_by_harness"] = {
        h: sorted(v)[len(v) // 2] for h, v in turns.items() if v}
    # Replay fidelity (agent harnesses): replayed prefix commands whose
    # output matched the recorded observation byte for byte.
    fid: dict[str, dict] = {}
    for st in states:
        r = results.get(st["state_id"])
        rep = (r or {}).get("report") or {}
        if not rep.get("replay_n"):
            continue
        f = fid.setdefault(st["harness"], {"states": 0, "commands": 0, "output_match": 0,
                                           "rc_match": 0, "replay_seconds": []})
        f["states"] += 1
        f["commands"] += rep["replay_n"]
        f["output_match"] += rep.get("replay_match", 0)
        f["rc_match"] += rep.get("replay_rc_match", 0)
        f["replay_seconds"].append(rep.get("replay_seconds", 0))
    for f in fid.values():
        secs = sorted(f.pop("replay_seconds"))
        f["replay_seconds_p50"] = secs[len(secs) // 2] if secs else None
        f["output_match_rate"] = round(f["output_match"] / f["commands"], 3) if f["commands"] else None
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
          f"states {len(rows)}, ran {len(ran)}, teacher {args.model}, cost ${cost:.2f}", ""]
    for title, key in (("By state kind", "by_state_kind"), ("By harness", "by_harness"),
                       ("By env (source)", "by_source"), ("By depth (turn_idx)", "by_depth"),
                       ("By depth, agent harnesses only", "by_depth_agent_harnesses"),
                       ("By state kind × harness", "by_kind_and_harness")):
        md.append(md_table(title, summary[key]))
    md.append("### Teacher stop condition by harness\n\n"
              + "\n".join(f"- {h}: {v}" for h, v in summary["teacher_stop_by_harness"].items())
              + "\n\n### Teacher turns used (p50) by harness\n\n"
              + "\n".join(f"- {h}: {v}" for h, v in summary["teacher_turns_p50_by_harness"].items())
              + "\n\n### Replay fidelity\n\n"
              + "\n".join(f"- {h}: {v}" for h, v in summary["replay_fidelity"].items()) + "\n")
    md.append("### Overlap\n\n" + "\n".join(f"- {k}: {v}" for k, v in summary["overlap"].items()) + "\n")
    (args.out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    print(f"side-table -> {table_path}")


if __name__ == "__main__":
    main()
