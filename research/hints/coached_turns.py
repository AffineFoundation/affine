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
from datagen.slicer import _normalize as normalize_fence  # noqa: E402
from evalsrv.chat import split_rollout  # noqa: E402


def normalize_messages(msgs: list[dict]) -> list[dict]:
    """The fold (datagen.slicer.slice_messages) rewrites mini-swe's foreign
    ```mswea_bash_command fence to ```bash in EVERY message before a turn
    enters D; the probe must see the same prefix the duel scores, or the
    teacher answers in the foreign fence and the bash parser forfeits it
    (52% of bash draws in the 2026-09-16 first pass)."""
    return [{**m, "content": normalize_fence(m["content"]) if isinstance(m.get("content"), str) else m.get("content")} for m in msgs]


def coached_first_actions(env_paths: list[str], states: dict[str, dict], baker) -> dict[str, list[dict]]:
    """state_id -> the first reply (z, y) of every coached continuation
    envelope for that state (arm-matched), decisive/solved flags attached."""
    import gzip
    out: dict[str, list[dict]] = {}
    for f in env_paths:
        for line in gzip.open(f, "rt"):
            e = json.loads(line)
            o = e["privileged"]["origin"]
            sid = o["state_id"]
            st = states.get(sid)
            if not st:
                continue
            arm = (e["privileged"].get("coach") or {}).get("arm") or "coached"
            if arm != (st.get("arm") or "coached"):
                continue
            kind = e["policy"].get("action_kind") or "bash"
            convs = trace_conversations(e["trace"], baker)
            if not convs:
                continue
            reply = normalize_fence(convs[0][-1]["content"])
            nodes = [nd for nd in e["trace"]["nodes"] if nd.get("sampled") and (nd.get("message") or {}).get("role") == "assistant"]
            rc = (nodes[0]["message"].get("reasoning_content") or "") if nodes else ""
            text = f"{rc}\n</think>\n{reply}" if rc and "</think>" not in reply else reply
            z, y = split_rollout(text, kind, require_think_close=False)
            ci = o.get("continuation")
            tids = st.get("coached_trace_ids") or []
            tid = tids[ci] if isinstance(ci, int) and ci < len(tids) else None
            out.setdefault(sid, []).append({
                "trace_id": tid or e["rollout_id"], "envelope_id": e["rollout_id"],
                "solved": tid in set(st.get("coached_solved_trace_ids") or []),
                "hinted_first": any(h.get("cont_turn") == 0 and h.get("injected") for h in e["privileged"].get("hints") or []),
                "z": z or rc, "y": y or ""})
    return out


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
    ap.add_argument("--envelopes", nargs="*", default=[],
                    help="coached envelope jsonl.gz files: attach the decisive continuation's first action as extra miner `coached_stored`")
    ap.add_argument("--stored", default="", help="json {turn_id: [stored duel records]} from the box's evals (scan_evals)")
    ap.add_argument("--raw-fence", action="store_true", help="keep the foreign mini-swe fence (NOT what the duel sees)")
    args = ap.parse_args()
    states = [json.loads(l) for l in open(args.states)]
    tidx = TS.load_trace_index()
    baker = ToolBaker.from_pretrained(TEACHER_REPO)
    stored = json.load(open(args.stored)) if args.stored else {}
    firsts = coached_first_actions(args.envelopes, {s["state_id"]: s for s in states}, baker) if args.envelopes else {}
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
                if not args.raw_fence:
                    prefix = normalize_messages(prefix)
                    reply = normalize_fence(reply)
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
                "hindsight": hs, "pivot": None, "recoverable": None, "stored": stored.get(s["turn_id"], []),
                "coached": {k: s.get(k) for k in ("coached_n_solved", "plain_n_solved", "hint_decisive", "hint_decisive_strict", "pivot_category", "select_tag",
                                                  "set", "arm", "label6", "also_arms", "run_id")},
            }
            recs = sorted(firsts.get(s["state_id"], []), key=lambda r: (not (r["solved"] and r["hinted_first"]), not r["solved"], not r["y"]))
            pick = next((r for r in recs if r["y"] and r["z"]), None)
            if pick:
                row["extra_miners"] = {"coached_stored": {k: pick[k] for k in ("z", "y", "trace_id", "solved", "hinted_first")}}
                row["coached_all"] = [{k: r[k] for k in ("trace_id", "solved", "hinted_first", "y")} for r in recs]
            ft.write(json.dumps(row, ensure_ascii=False) + "\n")
            note = first_note(s)
            if note:
                levels = dict(note["levels"] or {})
                # "note" = the reviewer note exactly as injected (fact+plan for
                # the DeepSeek coach, the reminder for the template coach).
                if note.get("note"):
                    levels["note"] = note["note"]
                coach_model = "template_coach.py" if s.get("arm") == "template" else "deepseek/deepseek-v4-pro (per-step coach)"
                for level in ("note", "fact", "plan", "action", "template"):
                    text = levels.get(level)
                    if not text:
                        continue
                    text = " ".join(str(text).split())
                    hr = {"turn_id": s["turn_id"], "group": "coached_decisive", "action_kind": kind,
                          "generator": "coached", "level": level, "text": text, "ok": True,
                          "model": f"{coach_model}, run {s.get('run_id')}",
                          "coach_trace_id": note["trace_id"], "coach_injected": note["injected"],
                          "coach_reason": note["reason"],
                          "hint_id": H.hint_id(s["turn_id"], "coached", level, text)}
                    hr.update(gate(row, text))
                    fh.write(json.dumps(hr, ensure_ascii=False) + "\n")
            n += 1
    print(f"{n} coached states -> {args.out_turns}, hints -> {args.out_hints}", file=sys.stderr)


if __name__ == "__main__":
    main()
