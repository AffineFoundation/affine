#!/usr/bin/env python
"""E4b — recovery lift from hints at the states the UNHINTED teacher fails.

Inputs: the teacher-recoverable side-table (ops/recoverable, PR #13) and its
state files (`states_all/states/<rollout>_<turn>_<kind>.json`, wire-form
prefix + task + runtime + remaining budget), the trace mirror (hindsight),
and the king-review pivot table (judge rationale).

Phase 1  --phase turns   : write e4b_turns.jsonl — one row per unhinted-
         failure state in the turn-set format gen_hints.py consumes
         (prefix text for the grounding gate, compressed hindsight transcript,
         king action as the reference, pivot text where judged).
Phase 2  --phase states  : with RUN/hints.jsonl in hand, write the E4b state
         files: arms = nohint (control subset only), ds_fact, self_fact,
         ds_plan, pivot_action (where it exists); `--reps` continuations each
         (distinct state_ids, one file per arm). The hint is appended to the
         last user / tool message as a reviewer note, exactly as the probe
         does for teacher reference sampling.

  python e4b_build.py --phase turns  --out RUN/e4b
  python e4b_build.py --phase states --out RUN/e4b --hints RUN/e4b/hints.jsonl --control 40
"""
from __future__ import annotations

import argparse
import collections
import copy
import glob
import json
import os
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import turnset as TS  # noqa: E402
from clients import HINT_HEADER, TEACHER_REPO  # noqa: E402

sys.path.insert(0, str(HERE.parents[1] / "affine"))
from affine.corpus.trace import ToolParityError, trace_conversations  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402

RECOV = Path(os.environ.get("HINTS_RECOV", "/tmp/hints-data/recov"))
SIDE = RECOV / "side_table.jsonl"
STATES = RECOV / "states_all" / "states"
ARMS = {"ds_fact": ("deepseek", "fact"), "self_fact": ("self", "fact"),
        "ds_plan": ("deepseek", "plan"), "pivot_action": ("pivot", "action")}


def state_path(r: dict) -> Path:
    return STATES / f"{r['rollout_id']}_{r['turn_idx']}_{r['state_kind']}.json"


def failed_states() -> list[dict]:
    side = [json.loads(l) for l in open(SIDE)]
    out = []
    seen = set()
    for r in side:
        if r["teacher_status"] != "ok" or r["teacher_solved"] is not False:
            continue
        key = (r["rollout_id"], int(r["turn_idx"]))
        if key in seen or not state_path(r).exists():
            continue
        seen.add(key)
        out.append(r)
    return out


def message_text(m: dict) -> str:
    c = m.get("content") or ""
    if isinstance(c, list):
        c = "\n".join(p.get("text", "") for p in c if isinstance(p, dict))
    if m.get("tool_calls"):
        c += "\n" + "\n".join(json.dumps(tc.get("function") or tc) for tc in m["tool_calls"])
    return c


def phase_turns(out: Path) -> None:
    rows = failed_states()
    tidx = TS.load_trace_index()
    pivots = TS.load_pivots()
    baker = ToolBaker.from_pretrained(TEACHER_REPO)
    n = 0
    n_baked = 0
    with open(out / "e4b_turns.jsonl", "w") as f:
        for r in rows:
            st = json.loads(state_path(r).read_text())
            ri = tidx.get(r["rollout_id"])
            if ri is None:
                continue
            env = TS.load_trace(ri)
            trace = env["trace"]
            hs = TS.hindsight(trace, ri.get("outcome") or "failed")
            replies = TS.raw_reply_nodes(trace)
            t = int(r["turn_idx"])
            ref_thought = (replies[t].get("reasoning_content") or "").strip() if t < len(replies) else ""
            piv = pivots.get((r["rollout_id"], t))
            # The prefix the DUEL would score: tool schemas / calls / results
            # baked into plain content through the teacher's own template
            # (affine.corpus.trace.trace_conversations), same as the D view.
            try:
                convs = trace_conversations(trace, baker)
                baked = convs[t][:-1] if t < len(convs) else None
            except (ToolParityError, Exception):  # noqa: BLE001 — fall back to wire text
                baked = None
            n_baked += baked is not None
            row = {
                "turn_id": r["turn_id"], "group": f"recov_fail:{r['state_kind']}",
                "state_id": st["state_id"], "state_kind": r["state_kind"],
                "source": r["source"], "harness": r["harness"], "policy_id": r["policy_id"],
                "action_kind": st.get("action_kind") or "bash",
                "rollout_id": r["rollout_id"], "turn_idx": t, "n_prefix_chars": st.get("prefix_chars"),
                "outcome": "failed",
                "prefix": (baked if baked else
                           [{"role": m["role"], "content": message_text(m)} for m in st["messages"]]),
                "prefix_baked": baked is not None,
                "reference_turn": message_text(st.get("king_reply") or {}),
                "reference_action": st.get("king_action") or "",
                "reference_thought": ref_thought,
                "hindsight": hs,
                "pivot": ({k: piv.get(k) for k in ("rationale", "should_have", "failure_category",
                                                    "confidence", "admit")} if piv else None),
                "recoverable": {"teacher_solved": False, "teacher_first_action_kind": r.get("teacher_first_action_kind")},
                "stored": [],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} unhinted-failure states ({n_baked} with a baked prefix) -> {out / 'e4b_turns.jsonl'}",
          file=sys.stderr)


def hinted_messages(messages: list[dict], hint: str) -> list[dict]:
    msgs = copy.deepcopy(messages)
    for m in reversed(msgs):
        if m["role"] in ("user", "tool"):
            m["content"] = (m.get("content") or "") + HINT_HEADER + hint
            break
    return msgs


def phase_states(out: Path, hints_path: Path, control: int, reps: int, seed: int) -> None:
    hints = {}
    for line in open(hints_path):
        h = json.loads(line)
        if h.get("ok") and h.get("text"):
            hints[(h["turn_id"], h["generator"], h["level"])] = h
    turns = [json.loads(l) for l in open(out / "e4b_turns.jsonl")]
    rng = random.Random(seed)
    # control subset: stratified by harness, run-to-run noise of the unhinted teacher
    by_h = collections.defaultdict(list)
    for t in turns:
        by_h[t["harness"]].append(t["turn_id"])
    ctrl = set()
    while len(ctrl) < control and any(by_h.values()):
        for h in sorted(by_h):
            if by_h[h] and len(ctrl) < control:
                ctrl.add(by_h[h].pop(rng.randrange(len(by_h[h]))))
    (out / "states").mkdir(parents=True, exist_ok=True)
    rows = []
    counts = collections.Counter()
    for t in turns:
        base = json.loads((STATES / f"{t['rollout_id']}_{t['turn_idx']}_{t['state_kind']}.json").read_text())
        arms = []
        if t["turn_id"] in ctrl:
            arms.append(("nohint", None))
        for arm, (gen, level) in ARMS.items():
            h = hints.get((t["turn_id"], gen, level))
            if h is not None:
                arms.append((arm, h))
            else:
                counts[f"no_hint:{arm}"] += 1
        for arm, h in arms:
            st = dict(base)
            st["arm"] = arm
            st["hint"] = h["text"] if h else None
            st["hint_id"] = h.get("hint_id") if h else None
            st["hint_grounded"] = (h.get("grounding") or {}).get("grounded") if h else None
            st["hint_leaks_future"] = (h.get("leak") or {}).get("leaks_future") if h else None
            st["probe_turn_id"] = t["turn_id"]
            st["probe_group"] = t["group"]
            if h:
                st["messages"] = hinted_messages(base["messages"], h["text"])
            sid = f"{t['rollout_id']}:{t['turn_idx']}:{arm}"
            st["state_id"] = sid
            path = out / "states" / (sid.replace(":", "_") + ".json")
            path.write_text(json.dumps(st, ensure_ascii=False))
            for rep in range(reps):
                meta = {k: v for k, v in st.items()
                        if k not in ("messages", "king_reply", "task_system_prompt", "task_prompt", "label")}
                meta["state_id"] = f"{sid}:r{rep}"
                meta["rep"] = rep
                meta["path"] = str(path)
                rows.append(meta)
                counts[f"job:{arm}:{t['harness']}"] += 1
    rng.shuffle(rows)   # mix harnesses so every shard sees a similar load
    with open(out / "states.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items()):
        print(f"{v:6d}  {k}", file=sys.stderr)
    print(f"{len(rows)} continuation jobs over {len(turns)} states (control {len(ctrl)}) -> {out}", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["turns", "states"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--hints")
    ap.add_argument("--control", type=int, default=40)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.phase == "turns":
        phase_turns(out)
    else:
        phase_states(out, Path(args.hints), args.control, args.reps, args.seed)


if __name__ == "__main__":
    main()
