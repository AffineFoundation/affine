#!/usr/bin/env python
"""Build the hinted-teacher probe turn set (E1–E4 share it).

One JSONL row per turn with everything the experiments need:
  * the prefix x exactly as the duel materializes it (corpus view
    `duel_turns@v4`, epoch index + view chunks on data.affine.io);
  * the recorded reply at that turn (the king's reply for the king groups,
    the teacher's for `completion`), parsed into (thought, action);
  * hindsight: a compressed transcript of the WHOLE rollout (every reply's
    thought head, action, observation head) plus outcome, stop condition and
    the normalized future actions (for the leak check);
  * the pivot judge's rationale / should_have when the turn is a judged pivot;
  * stored duel rollouts (king / challenger z, y and the duel's teacher refs)
    for turns that appeared in a scored duel.

Groups: king_loop_onset (fold group of D), king_pivot (admitted pivots of the
king-review side-table that D holds as king_fail / king_pivot / onset turns),
completion (positive control: the teacher's own final reply of a solved
rollout).

  python turnset.py --out /tmp/hints-data/turns.jsonl --n-onset 200 \
      --n-pivot 120 --n-completion 100 --seed 20260912
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import json
import os
import random
import re
import sys
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402
from affine.corpus.materialize import materialize_turn  # noqa: E402
from affine.corpus.trace import message_text, sampled_paths  # noqa: E402

DATA = Path(os.environ.get("HINTS_DATA", "/tmp/hints-data"))
BASE = "https://data.affine.io"
INDEX = DATA / "turns_0027.parquet"
VIEWS = DATA / "views"
TRACES = DATA / "box" / "king_review" / "traces"
EVALS = DATA / "box" / "evals"
PIVOTS = DATA / "box" / "king_pivots" / "0ce59769300c.jsonl"
RECOVERABLE = DATA / "box" / "recoverable" / "0ce59769300c.jsonl"
MAX_PREFIX_CHARS = 200_000
WS = re.compile(r"\s+")

# Compressed-transcript budgets (chars). HHD-style: actions + observation
# heads, the whole rollout, the same text for every generator.
THOUGHT_HEAD = 240
ACTION_HEAD = 500
OBS_HEAD = 400
TRANSCRIPT_CAP = 60_000


def norm(s: str) -> str:
    return WS.sub(" ", s or "").strip()


def fetch(key: str) -> Path:
    dst = VIEWS / os.path.basename(key)
    if not dst.exists():
        VIEWS.mkdir(parents=True, exist_ok=True)
        req = urllib.request.Request(f"{BASE}/{key}", headers={"User-Agent": "affine-hints/1"})
        with urllib.request.urlopen(req, timeout=600) as r:
            blob = r.read()
        tmp = dst.with_suffix(".tmp")
        tmp.write_bytes(blob)
        tmp.replace(dst)
    return dst


def load_index() -> list[dict]:
    return pq.read_table(INDEX).to_pylist()


def group_of(stratum: str | None) -> str:
    s = stratum or ""
    return s.split(":", 1)[0] if ":" in s else "repo"


def load_view_records(rows: list[dict]) -> dict[tuple[str, int], dict]:
    """(chunk_key, traj_line) -> view record for every requested row."""
    want: dict[str, set[int]] = collections.defaultdict(set)
    for r in rows:
        want[r["chunk_key"]].add(int(r["traj_line"]))
    out: dict[tuple[str, int], dict] = {}
    for key, lines in want.items():
        path = fetch(key)
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i in lines:
                    out[(key, i)] = json.loads(line)
    return out


def load_trace_index() -> dict[str, dict]:
    idx = {}
    with open(TRACES / "rollout_index.jsonl") as f:
        for line in f:
            r = json.loads(line)
            idx[r["rollout_id"]] = r
    return idx


def load_trace(ri: dict) -> dict:
    with gzip.open(TRACES / "chunks" / ri["chunk"], "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == ri["line"]:
                return json.loads(line)
    raise KeyError(ri["rollout_id"])


def reply_parts(msg: dict) -> tuple[str, str]:
    """(thought, action-ish text) of one assistant message from the trace.
    Tool calls are rendered as name(arguments) so the transcript is short."""
    thought = (msg.get("reasoning_content") or "").strip()
    content = message_text(msg.get("content")).strip()
    if "<think>" in content and "</think>" in content:
        t2, _, content = content.partition("</think>")
        thought = (thought + "\n" + t2.replace("<think>", "")).strip()
    calls = msg.get("tool_calls") or []
    if calls:
        parts = []
        for c in calls:
            fn = (c.get("function") or {}) if isinstance(c, dict) else {}
            parts.append(f"{fn.get('name', '?')}({fn.get('arguments', '')})")
        action = "\n".join(parts)
        if content:
            action = content + "\n" + action
    else:
        action = content
    return thought, action


def hindsight(trace: dict, outcome: str) -> dict:
    """Compressed transcript + future-action list for the whole rollout."""
    paths = sampled_paths(trace)
    if not paths:
        return {"transcript": "", "n_replies": 0, "actions": []}
    task_text = ""
    for m in paths[0]:
        if m["role"] == "user":
            task_text = m["content"]
            break
    lines = [f"TASK (user prompt, head):\n{task_text[:2500]}", ""]
    actions: list[str] = []
    n = len(paths)
    # sampled_paths drops reasoning_content; the raw sampled nodes are in the
    # same order, so index i of both is the same reply.
    raw = raw_reply_nodes(trace)
    for i, path in enumerate(paths):
        reply = dict(path[-1])
        if i < len(raw) and raw[i].get("reasoning_content"):
            reply["reasoning_content"] = raw[i]["reasoning_content"]
        thought, action = reply_parts(reply)
        obs = ""
        if i + 1 < n:
            nxt = paths[i + 1]
            tail = nxt[len(path):]
            obs = "\n".join(m["content"] for m in tail if m["role"] in ("user", "tool"))
        actions.append(norm(action))
        lines.append(f"--- turn {i} ---")
        if thought:
            lines.append(f"thought: {norm(thought)[:THOUGHT_HEAD]}")
        lines.append(f"action: {action.strip()[:ACTION_HEAD]}")
        if obs:
            lines.append(f"observation: {norm(obs)[:OBS_HEAD]}")
    lines.append("")
    lines.append(f"OUTCOME: {outcome}; stop_condition={trace.get('stop_condition')}; "
                 f"n_replies={n}")
    text = "\n".join(lines)
    if len(text) > TRANSCRIPT_CAP:
        head = text[: TRANSCRIPT_CAP // 2]
        tail = text[-TRANSCRIPT_CAP // 2:]
        text = head + "\n[... middle of the transcript elided ...]\n" + tail
    return {"transcript": text, "n_replies": n, "actions": actions,
            "stop_condition": trace.get("stop_condition"),
            "rewards": trace.get("rewards")}


def reasoning_at(trace: dict, turn_idx: int) -> str:
    paths = sampled_paths(trace)
    if turn_idx < len(paths):
        return (paths[turn_idx][-1].get("reasoning_content") or "").strip()
    return ""


def raw_reply_nodes(trace: dict) -> list[dict]:
    """Sampled assistant node messages in order (with reasoning_content)."""
    out = []
    for nd in trace.get("nodes") or []:
        m = nd.get("message") or {}
        if nd.get("sampled") and m.get("role") == "assistant":
            out.append(m)
    return out


def load_pivots() -> dict[tuple[str, int], dict]:
    out = {}
    if not PIVOTS.exists():
        return out
    for line in open(PIVOTS):
        p = json.loads(line)
        key = (p["rollout_id"], int(p["turn_idx"]))
        if p.get("admit") or key not in out:
            out[key] = p
    return out


def load_recoverable() -> dict[tuple[str, int], dict]:
    out = {}
    if not RECOVERABLE.exists():
        return out
    for line in open(RECOVERABLE):
        p = json.loads(line)
        out[(p["rollout_id"], int(p["turn_idx"]))] = p
    return out


def load_stored(turn_ids: set[str]) -> dict[str, list[dict]]:
    """turn_id -> [{cid, king:{z,y}, chal:{z,y}, refs:[{z,y,...}]}] from the
    stored duel records (both sides valid)."""
    out: dict[str, list[dict]] = collections.defaultdict(list)
    for f in sorted(glob.glob(str(EVALS / "chal-*.json.gz"))):
        d = json.load(gzip.open(f, "rt", encoding="utf-8"))
        if not d.get("king_rows"):
            continue
        cid = os.path.basename(f)[:-8]
        kr = {r["turn_id"]: r for r in d["king_rows"]}
        cr = {r["turn_id"]: r for r in d.get("challenger_rows", [])}
        refs = d.get("teacher_refs") or {}
        v = d.get("verdict") or {}
        for tid in turn_ids & set(kr):
            k, c = kr[tid], cr.get(tid)
            rec = {"cid": cid, "verdict_z": v.get("z"),
                   "challenger": (v.get("challenger") or {}).get("repo") if isinstance(v.get("challenger"), dict) else v.get("challenger"),
                   "king": None, "chal": None, "refs": refs.get(tid)}
            if k.get("valid") and k.get("pairs"):
                p = k["pairs"][0]
                rec["king"] = {"z": p["z_a"], "y": p["y_a"], "m": p.get("lpC_za_x"),
                               "a": [q["lpC_yc_za"] - q["lpC_yc_e"] for q in k["pairs"]]}
            if c and c.get("valid") and c.get("pairs"):
                p = c["pairs"][0]
                rec["chal"] = {"z": p["z_a"], "y": p["y_a"], "m": p.get("lpC_za_x"),
                               "a": [q["lpC_yc_za"] - q["lpC_yc_e"] for q in c["pairs"]]}
            out[tid].append(rec)
    return out


def pick(rows: list[dict], n: int, rng: random.Random, prefer: set[str],
         by: str = "action_kind", quotas: dict[str, int] | None = None) -> list[dict]:
    """Stratified pick: `quotas` per action_kind (rest split evenly), turns
    with stored duel rows first inside each cell."""
    cells: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        cells[r[by]].append(r)
    for cell in cells.values():
        rng.shuffle(cell)
        cell.sort(key=lambda r: 0 if r["turn_id"] in prefer else 1)
    quotas = dict(quotas or {})
    picked: list[dict] = []
    for k, q in quotas.items():
        picked += cells.get(k, [])[:q]
    rest = [k for k in cells if k not in quotas]
    left = n - len(picked)
    if rest and left > 0:
        per = max(1, left // len(rest))
        for k in rest:
            picked += cells[k][:per]
    return picked[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DATA / "turns.jsonl"))
    ap.add_argument("--n-onset", type=int, default=200)
    ap.add_argument("--n-pivot", type=int, default=120)
    ap.add_argument("--n-completion", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20260912)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    index = load_index()
    by_tid = {r["turn_id"]: r for r in index}
    pivots = load_pivots()
    recov = load_recoverable()
    admitted = {k for k, p in pivots.items() if p.get("admit")}

    onset = [r for r in index if group_of(r["stratum"]) == "king_loop_onset"
             and r["n_prefix_chars"] <= MAX_PREFIX_CHARS
             and r["source"] not in ("affine_wiki", "affine_math", "affine_agent")]
    pivot_rows = [r for r in index
                  if (r["rollout_id"], int(r["turn_idx"])) in admitted
                  and group_of(r["stratum"]) in ("king_fail", "king_pivot")
                  and r["action_kind"] != "boxed"
                  and r["n_prefix_chars"] <= MAX_PREFIX_CHARS
                  and r["source"] not in ("affine_wiki", "affine_math", "affine_agent")]
    completion = [r for r in index if group_of(r["stratum"]) == "completion"
                  and r["n_prefix_chars"] <= MAX_PREFIX_CHARS]
    print(f"pool: onset {len(onset)} pivot {len(pivot_rows)} completion {len(completion)}",
          file=sys.stderr)

    all_tids = {r["turn_id"] for r in onset + pivot_rows + completion}
    stored = load_stored(all_tids)
    prefer = {tid for tid, recs in stored.items()
              if any(x["king"] and x["chal"] for x in recs)}
    print(f"turns with stored both-side rows: {len(prefer)}", file=sys.stderr)

    sel_onset = pick(onset, args.n_onset, rng, prefer,
                     quotas={"bash": 90, "terminus_json": 28})
    sel_pivot = pick(pivot_rows, args.n_pivot, rng, prefer,
                     quotas={"bash": 55, "terminus_json": 20})
    sel_comp = pick(completion, args.n_completion, rng, prefer,
                    quotas={"text": 55, "bash": 25, "terminus_json": 20})
    chosen = ([(r, "king_loop_onset") for r in sel_onset]
              + [(r, "king_pivot") for r in sel_pivot]
              + [(r, "completion") for r in sel_comp])
    seen = set()
    chosen = [(r, g) for r, g in chosen if not (r["turn_id"] in seen or seen.add(r["turn_id"]))]

    records = load_view_records([r for r, _ in chosen])
    tidx = load_trace_index()
    n_out = 0
    counts = collections.Counter()
    with open(args.out, "w", encoding="utf-8") as out:
        for r, grp in chosen:
            rec = records[(r["chunk_key"], int(r["traj_line"]))]
            meta = next(m for m in rec["turns"] if int(m["turn_idx"]) == int(r["turn_idx"]))
            turn = materialize_turn(rec, meta)
            kind = turn["action_kind"]
            ref_action = dialects.last_action(turn["reference_turn"], kind)
            ri = tidx.get(r["rollout_id"])
            hs: dict = {}
            ref_thought = ""
            if ri is not None:
                env = load_trace(ri)
                trace = env["trace"]
                hs = hindsight(trace, ri.get("outcome") or rec.get("outcome") or "")
                hs["task"] = {k: env.get("task", {}).get(k) for k in ("sid", "repo", "language", "uid")}
                replies = raw_reply_nodes(trace)
                if int(r["turn_idx"]) < len(replies):
                    ref_thought = (replies[int(r["turn_idx"])].get("reasoning_content") or "").strip()
            else:
                counts["no_trace"] += 1
            key = (r["rollout_id"], int(r["turn_idx"]))
            piv = pivots.get(key)
            row = {
                "turn_id": r["turn_id"],
                "group": grp,
                "stratum": r["stratum"],
                "fold_group": rec.get("fold_group"),
                "source": r["source"],
                "language": r["language"],
                "harness": (rec.get("policy") or {}).get("harness"),
                "policy_id": (rec.get("policy") or {}).get("id"),
                "model": rec.get("model"),
                "action_kind": kind,
                "rollout_id": r["rollout_id"],
                "traj_id": r["traj_id"],
                "turn_idx": int(r["turn_idx"]),
                "node_id": int(r["node_id"]),
                "n_prefix_chars": int(r["n_prefix_chars"]),
                "n_prefix_msgs": len(turn["prefix"]),
                "phase": r["phase"],
                "loop_onset_of": meta.get("loop_onset_of"),
                "outcome": rec.get("outcome"),
                "prefix": turn["prefix"],
                "reference_turn": turn["reference_turn"],
                "reference_action": ref_action,
                "reference_thought": ref_thought,
                "hindsight": hs,
                "pivot": ({k: piv.get(k) for k in ("rationale", "should_have", "failure_category",
                                                    "pivot_pattern", "confidence", "admit",
                                                    "recoverable")} if piv else None),
                "recoverable": ({k: recov[key].get(k) for k in ("teacher_solved", "admit",
                                                                 "teacher_first_action_kind",
                                                                 "teacher_status")}
                                if key in recov else None),
                "stored": stored.get(r["turn_id"], []),
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_out += 1
            counts[(grp, kind)] += 1
            counts[f"{grp}:stored"] += bool(row["stored"])
            counts[f"{grp}:pivot_text"] += bool(piv and piv.get("should_have"))
    print(f"wrote {n_out} turns -> {args.out}", file=sys.stderr)
    for k, v in sorted(counts.items(), key=str):
        print(f"  {k}: {v}", file=sys.stderr)


if __name__ == "__main__":
    main()
