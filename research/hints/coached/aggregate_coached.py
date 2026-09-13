#!/usr/bin/env python
"""Aggregate the coached-teacher recovery run.

Inputs: the run's states.jsonl (select_states.py), out/results/*.json +
out/traces/*.json (run_coached.py), out/coach/hints.jsonl (coach_proxy.py).

Per state (unit):
  coached   n / n_solved over the coached continuations (env grade)
  plain     n / n_solved over the plain continuations: fresh ones from this
            run, else the side-table's (`plain_control`, same teacher, same T)
  decisive  coached solved in >= 1 continuation AND plain solved in 0
            ("the hint changed the outcome" — Jacob's target set)
  majority  coached solved in a majority of its OK continuations
  hints     per kept (solved) coached continuation: one row per step with the
            note, its levels and gates, joined to the trace's sampled replies
            by step order (cont_turn) and the last-observation sha

Outputs (--out):
  side_table.jsonl        one row per state — the fold-admittable table for a
                          `king_coached` group (schema in docs/coached-recovery.md)
  coached_states.jsonl    the hint-decisive states with their hints (for the
                          hint-probe worker)
  hints_by_turn.jsonl     every injected note of every kept continuation
  summary.md / summary.json
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common as C  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").split("\n") if l.strip()]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k: int, n: int) -> str:
    return f"{k}/{n} ({100.0 * k / n:.0f} %)" if n else "0/0"


def solved(r: dict) -> bool:
    return r.get("status") == "ok" and r.get("outcome") == "solved"


def sampled_replies(trace: dict) -> list[dict]:
    out = []
    for i, nd in enumerate(trace.get("nodes") or []):
        m = nd.get("message") or {}
        if nd.get("sampled") and m.get("role") == "assistant":
            out.append({"node_idx": i, "node_id": nd.get("id"), "message": m})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", required=True, type=Path)
    ap.add_argument("--run-out", required=True, type=Path, help="run_coached.py --out")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    states = load_jsonl(args.states)
    units: dict[str, dict] = {}
    for s in states:
        units.setdefault(s.get("proxy_key") or s["state_id"], s)
    results = {}
    for p in (args.run_out / "results").glob("*.json"):
        r = json.loads(p.read_text())
        results.setdefault(r["state_id"], {}).setdefault(r["arm"], {})[int(r.get("continuation") or 0)] = r
    hints = load_jsonl(args.run_out / "coach" / "hints.jsonl")
    by_key: dict[str, list[dict]] = collections.defaultdict(list)
    for h in hints:
        by_key[h.get("arm_key") or f"{h['unit']}/c{h['k']}/{h['arm']}"].append(h)

    rows = []
    turn_rows = []
    gate_c: collections.Counter = collections.Counter()
    coach_cost = sum(float((h.get("coach") or {}).get("cost_usd") or 0) for h in hints)
    teacher_cost = 0.0
    n_steps = 0
    for unit, s in units.items():
        res = results.get(unit, {})
        co = res.get("coached", {})
        pl_fresh = res.get("plain", {})
        co_ok = [r for r in co.values() if r.get("status") == "ok"]
        co_err = [r for r in co.values() if r.get("status") != "ok"]
        pl_ok = [r for r in pl_fresh.values() if r.get("status") == "ok"]
        teacher_cost += sum(float(r.get("cost_usd") or 0) for r in list(co.values()) + list(pl_fresh.values()))
        plain: dict
        if pl_ok:
            plain = {"source": "fresh", "n": len(pl_ok), "n_solved": sum(solved(r) for r in pl_ok),
                     "trace_ids": [r.get("trace_id") for r in pl_ok]}
        elif s.get("plain_control"):
            plain = dict(s["plain_control"])
        else:
            plain = {"source": "none", "n": 0, "n_solved": 0, "trace_ids": []}
        co_solved = [r for r in co_ok if solved(r)]
        n_co = len(co_ok)
        st = "pending" if not co else ("errored" if not co_ok else "ok")
        row = {
            "run_id": args.run_id,
            "unit": unit,
            "state_id": s["state_id"], "turn_id": s.get("turn_id"), "node_id": s.get("node_id"),
            "rollout_id": s["rollout_id"], "turn_idx": s["turn_idx"], "state_kind": s["state_kind"],
            "onset_rank": (s.get("label") or {}).get("onset_rank") if isinstance(s.get("label"), dict) else None,
            "king_digest": s.get("king_digest"), "harness": s["harness"], "source": s["source"],
            "resume_kind": s["resume_kind"], "proxy": "same_task" if s["resume_kind"] == "same_task" else None,
            "policy_id": s.get("policy_id"), "select_tag": s.get("select_tag"), "tier": s.get("tier"),
            "pivot_category": s.get("pivot_category"), "depth": s.get("depth"), "max_turns": s.get("max_turns"),
            "coached_status": st,
            "coached_n": n_co, "coached_n_errored": len(co_err), "coached_n_solved": len(co_solved),
            "coached_solved_any": bool(co_solved),
            "coached_solved_majority": bool(n_co) and len(co_solved) * 2 > n_co,
            "coached_trace_ids": [r.get("trace_id") for r in co_ok],
            "coached_solved_trace_ids": [r.get("trace_id") for r in co_solved],
            "coached_stops": [r.get("stop_condition") for r in co_ok],
            "coached_turns": [r.get("n_turns") for r in co_ok],
            "plain_source": plain["source"], "plain_n": plain["n"], "plain_n_solved": plain["n_solved"],
            "plain_trace_ids": plain.get("trace_ids") or [],
            "plain_solved_any": plain["n_solved"] > 0,
            "hint_decisive": bool(co_solved) and plain["n"] >= 2 and plain["n_solved"] == 0,
            "hint_decisive_strict": (bool(n_co) and len(co_solved) * 2 > n_co and plain["n"] >= 3
                                     and plain["n_solved"] == 0),
            "plain_only": (not co_solved) and n_co >= 2 and plain["n_solved"] > 0,
            "admit": bool(co_solved) and plain["n"] >= 2 and plain["n_solved"] == 0,
            "hints": {},
        }
        # per-continuation hint rows for the SOLVED coached continuations
        for k, r in co.items():
            if not solved(r):
                continue
            key = f"{C.unit_stem(unit)}/c{k}/coached"
            steps = sorted(by_key.get(key, []), key=lambda h: (h.get("cont_turn", 0), h.get("ts", "")))
            trace_path = args.run_out / "traces" / f"{C.unit_stem(unit)}.coached.c{k}.json"
            replies = sampled_replies(json.loads(trace_path.read_text())) if trace_path.is_file() else []
            # steps that reached a completion path only (count_tokens etc. excluded)
            steps = [h for h in steps if h.get("path", "").endswith(("chat/completions", "messages"))]
            hint_list = []
            by_turn: dict[int, dict] = {}
            for h in steps:
                by_turn.setdefault(int(h.get("cont_turn") or 0), h)   # first request of a step wins
            for t, h in sorted(by_turn.items()):
                gate_c["steps"] += 1
                gate_c["injected"] += bool(h.get("injected"))
                if h.get("reason"):
                    gate_c[f"reason:{h['reason']}"] += 1
                if h.get("gate"):
                    gate_c["grounded"] += bool(h["gate"]["grounding"]["grounded"])
                    gate_c["leaks"] += bool(h["gate"]["leak"]["leaks_future"])
                    gate_c["rewrite"] += bool(h.get("rewrite"))
                rep = replies[t] if t < len(replies) else None
                item = {
                    "cont_turn": t, "node_idx": rep["node_idx"] if rep else None,
                    "node_id": rep["node_id"] if rep else None,
                    "injected": bool(h.get("injected")), "reason": h.get("reason"),
                    "note": h.get("note"), "levels": h.get("levels"),
                    "gate": h.get("gate"), "rewrite": h.get("rewrite"),
                    "last_msg_sha": h.get("last_msg_sha"), "reply_sha": h.get("reply_sha"),
                    "coach_cost_usd": (h.get("coach") or {}).get("cost_usd"),
                    "coach_model": (h.get("coach") or {}).get("model"),
                }
                hint_list.append(item)
                turn_rows.append({"unit": unit, "state_id": s["state_id"], "rollout_id": s["rollout_id"],
                                  "king_turn_idx": s["turn_idx"], "continuation": k,
                                  "trace_id": r.get("trace_id"), "hint_decisive": row["hint_decisive"],
                                  **item})
            row["hints"][r.get("trace_id") or f"c{k}"] = {"continuation": k, "n_steps": len(hint_list),
                                                             "n_injected": sum(i["injected"] for i in hint_list),
                                                             "steps": hint_list}
        n_steps += sum(1 for h in by_key.get(f"{C.unit_stem(unit)}/c0/coached", []))
        rows.append(row)

    with open(args.out / "side_table.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out / "coached_states.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            if r["hint_decisive"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.out / "hints_by_turn.jsonl", "w", encoding="utf-8") as f:
        for r in turn_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary -------------------------------------------------------------------
    done = [r for r in rows if r["coached_status"] == "ok"]
    paired = [r for r in done if r["plain_n"] >= 2]
    lines = [f"# coached-teacher recovery — {args.run_id or args.run_out}", ""]
    lines.append(f"states selected {len(rows)}; coached done {len(done)} "
                 f"(errored-only {sum(1 for r in rows if r['coached_status'] == 'errored')}, "
                 f"pending {sum(1 for r in rows if r['coached_status'] == 'pending')}); "
                 f"paired with a plain control {len(paired)} "
                 f"(fresh {sum(1 for r in paired if r['plain_source'] == 'fresh')}, "
                 f"side-table {sum(1 for r in paired if r['plain_source'] == 'side_table')})")
    lines.append("")
    lines.append("## Recovery (state solved in >= 1 of its continuations)")
    lines.append("")
    lines.append("| slice | states | coached any | coached majority | plain any | hint-decisive (coached ≥1, plain 0) | strict (coached majority, plain 0/3) | plain-only |")
    lines.append("|---|---|---|---|---|---|---|---|")

    def slice_row(name: str, rs: list[dict]) -> str:
        n = len(rs)
        ca = sum(r["coached_solved_any"] for r in rs)
        cm = sum(r["coached_solved_majority"] for r in rs)
        pa = sum(r["plain_solved_any"] for r in rs)
        hd = sum(r["hint_decisive"] for r in rs)
        hs = sum(r["hint_decisive_strict"] for r in rs)
        po = sum(r["plain_only"] for r in rs)
        lo, hi = wilson(ca, n)
        return (f"| {name} | {n} | {pct(ca, n)} [{lo:.2f}, {hi:.2f}] | {pct(cm, n)} | {pct(pa, n)} | "
                f"**{hd}** | {hs} | {po} |")

    lines.append(slice_row("all paired", paired))
    lines.append(slice_row("all done", done))
    for tag in sorted({r["select_tag"] for r in paired}):
        lines.append(slice_row(f"tag {tag}", [r for r in paired if r["select_tag"] == tag]))
    for h in sorted({r["harness"] for r in paired}):
        lines.append(slice_row(f"harness {h}", [r for r in paired if r["harness"] == h]))
    for src in sorted({r["source"] for r in paired}):
        lines.append(slice_row(f"source {src}", [r for r in paired if r["source"] == src]))
    for kd in sorted({r["state_kind"] for r in paired}):
        lines.append(slice_row(f"kind {kd}", [r for r in paired if r["state_kind"] == kd]))
    for lo_d, hi_d in ((0, 9), (10, 19), (20, 39), (40, 80)):
        rs = [r for r in paired if r["depth"] is not None and lo_d <= int(r["depth"]) <= hi_d]
        if rs:
            lines.append(slice_row(f"depth {lo_d}-{hi_d}", rs))
    cats = sorted({r["pivot_category"] for r in paired if r.get("pivot_category")})
    for c in cats:
        lines.append(slice_row(f"category {c}", [r for r in paired if r.get("pivot_category") == c]))
    lines.append("")
    # per-continuation rates
    co_n = sum(r["coached_n"] for r in done)
    co_s = sum(r["coached_n_solved"] for r in done)
    pl_n = sum(r["plain_n"] for r in paired)
    pl_s = sum(r["plain_n_solved"] for r in paired)
    lines.append(f"per-continuation solve rate: coached {pct(co_s, co_n)}; plain {pct(pl_s, pl_n)} "
                 f"(plain counts the paired states' controls)")
    fresh = [r for r in paired if r["plain_source"] == "fresh"]
    if fresh:
        lines.append(f"fresh-control subset ({len(fresh)} states, both arms this run): coached any "
                     f"{pct(sum(r['coached_solved_any'] for r in fresh), len(fresh))}, plain any "
                     f"{pct(sum(r['plain_solved_any'] for r in fresh), len(fresh))}")
    lines.append("")
    lines.append("## Hint gates (steps of SOLVED coached continuations)")
    lines.append("")
    n_st = gate_c["steps"]
    lines.append(f"steps {n_st}; injected {pct(gate_c['injected'], n_st)}; grounded (final) "
                 f"{pct(gate_c['grounded'], n_st)}; leaks future {pct(gate_c['leaks'], n_st)}; "
                 f"rewrite used {pct(gate_c['rewrite'], n_st)}")
    for k, v in sorted(gate_c.items()):
        if k.startswith("reason:"):
            lines.append(f"- not injected, {k[7:]}: {v}")
    # all-steps gate stats (every coached request, solved or not)
    all_steps = [h for h in hints if h.get("arm") == "coached" and h.get("path", "").endswith(("chat/completions", "messages"))]
    inj = sum(1 for h in all_steps if h.get("injected"))
    lines.append("")
    lines.append(f"all coached requests {len(all_steps)}: injected {pct(inj, len(all_steps))}; "
                 f"reasons: {dict(collections.Counter(h.get('reason') for h in all_steps if not h.get('injected')))}")
    gated = [h for h in all_steps if h.get("gate")]
    if gated:
        lines.append(f"of {len(gated)} gated notes: grounded first try "
                     f"{pct(sum(1 for h in gated if (h.get('gate_first') or h['gate'])['grounding']['grounded']), len(gated))}, "
                     f"grounded after rewrite {pct(sum(1 for h in gated if h['gate']['grounding']['grounded']), len(gated))}, "
                     f"leak {pct(sum(1 for h in gated if h['gate']['leak']['leaks_future']), len(gated))}; "
                     f"note chars p50 {sorted(h['gate']['n_chars'] for h in gated)[len(gated) // 2]}, "
                     f"sentences p50 {sorted(h['gate']['n_sentences'] for h in gated)[len(gated) // 2]}")
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"coach (OpenRouter, from usage.cost): ${coach_cost:.2f} over {len(hints)} proxied requests "
                 f"({sum(1 for h in hints if (h.get('coach') or {}).get('cost_usd'))} coach calls)")
    lines.append(f"teacher (Engy list price, traces' usage): ${teacher_cost:.2f}")
    (args.out / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump({"n_states": len(rows), "n_done": len(done), "n_paired": len(paired),
               "hint_decisive": sum(r["hint_decisive"] for r in rows),
               "hint_decisive_strict": sum(r["hint_decisive_strict"] for r in rows),
               "coached_any": sum(r["coached_solved_any"] for r in paired),
               "plain_any": sum(r["plain_solved_any"] for r in paired),
               "coach_cost_usd": round(coach_cost, 2), "teacher_cost_usd": round(teacher_cost, 2),
               "gates": dict(gate_c)}, open(args.out / "summary.json", "w"), indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
