#!/usr/bin/env python
"""Turn rows + stored coach hints for the coached-recovery worker's states.

Input: the store's `internal/hints/coached-states.jsonl` (one row per state
the per-step coach verifiably rescued: `hint_decisive`), with the coach's
per-step notes under `hints[<trace_id>].steps[*]` (`levels.fact/plan/action`).
Output: a turn-set row per state (prefix baked through the teacher template,
hindsight transcript, recorded king reply) and a hints.jsonl with generator
`coached` = the coach's note at the FIRST step of the first solved coached
continuation (the note the teacher saw when it acted at this state), gated
like every other hint.

  python coached_turns.py --states coached-states.jsonl --out-turns T.jsonl --out-hints H.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "affine"))

import hints as H  # noqa: E402
import turnset as TS  # noqa: E402
from clients import TEACHER_REPO  # noqa: E402
from gen_hints import gate  # noqa: E402

from affine import dialects  # noqa: E402
from affine.corpus.trace import trace_conversations  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402


def first_note(state: dict) -> dict | None:
    hints = state.get("hints") or {}
    order = list(state.get("coached_solved_trace_ids") or []) + [t for t in hints if t not in (state.get("coached_solved_trace_ids") or [])]
    for tid in order:
        rec = hints.get(tid)
        if not rec:
            continue
        for step in rec.get("steps") or []:
            if step.get("cont_turn") == 0 and (step.get("levels") or step.get("note")):
                return {"trace_id": tid, "injected": step.get("injected"), "reason": step.get("reason"),
                        "levels": step.get("levels") or {"fact": step.get("note")}, "note": step.get("note")}
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--out-turns", required=True)
    ap.add_argument("--out-hints", required=True)
    args = ap.parse_args()
    states = [json.loads(l) for l in open(args.states)]
    tidx = TS.load_trace_index()
    baker = ToolBaker.from_pretrained(TEACHER_REPO)
    n = 0
    with open(args.out_turns, "w") as ft, open(args.out_hints, "w") as fh:
        for s in states:
            ri = tidx.get(s["rollout_id"])
            if ri is None:
                print(f"no trace for {s['rollout_id']}", file=sys.stderr)
                continue
            env = TS.load_trace(ri)
            trace = env["trace"]
            t = int(s["turn_idx"])
            try:
                convs = trace_conversations(trace, baker)
                prefix = convs[t][:-1]
                reply = convs[t][-1]["content"]
            except Exception as e:  # noqa: BLE001
                print(f"bake failed {s['turn_id']}: {e!r}", file=sys.stderr)
                continue
            hs = TS.hindsight(trace, ri.get("outcome") or "failed")
            replies = TS.raw_reply_nodes(trace)
            kind = (env.get("policy") or {}).get("action_kind") or "bash"
            row = {
                "turn_id": s["turn_id"], "group": "coached_decisive", "stratum": None,
                "source": s["source"], "harness": s["harness"], "policy_id": s.get("policy_id"),
                "action_kind": kind, "rollout_id": s["rollout_id"], "turn_idx": t,
                "king_digest": s.get("king_digest"), "state_kind": s.get("state_kind"),
                "n_prefix_chars": sum(len(m["content"]) for m in prefix), "outcome": "failed",
                "prefix": prefix, "reference_turn": reply,
                "reference_action": dialects.last_action(reply, kind) if kind in ("bash", "tool_call", "terminus_json", "text", "boxed") else "",
                "reference_thought": (replies[t].get("reasoning_content") or "").strip() if t < len(replies) else "",
                "hindsight": hs, "pivot": None, "recoverable": None, "stored": [],
                "coached": {k: s.get(k) for k in ("coached_n_solved", "plain_n_solved", "hint_decisive", "hint_decisive_strict", "pivot_category", "select_tag")},
            }
            ft.write(json.dumps(row, ensure_ascii=False) + "\n")
            note = first_note(s)
            if note:
                for level in ("fact", "plan", "action"):
                    text = (note["levels"] or {}).get(level)
                    if not text:
                        continue
                    text = " ".join(str(text).split())
                    hr = {"turn_id": s["turn_id"], "group": "coached_decisive", "action_kind": kind,
                          "generator": "coached", "level": level, "text": text, "ok": True,
                          "model": "deepseek/deepseek-v4-pro (per-step coach, coached-20260913a)",
                          "coach_trace_id": note["trace_id"], "coach_injected": note["injected"],
                          "coach_reason": note["reason"],
                          "hint_id": H.hint_id(s["turn_id"], "coached", level, text)}
                    hr.update(gate(row, text))
                    fh.write(json.dumps(hr, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} coached states -> {args.out_turns}, hints -> {args.out_hints}", file=sys.stderr)


if __name__ == "__main__":
    main()
