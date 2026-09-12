"""Build teacher-continuation candidate states from the king's failure labels.

A *state* is one prefix of a failed king rollout — everything the model saw
before it had to reply — that a labeler flagged as the place the run went
wrong: a `loop_onset` (deterministic labeler, affine/corpus/labels.py) or a
`pivot` (LLM judge, ops/king-review). The teacher-recoverable filter asks,
for each state, whether the teacher can still solve the task from there.

Inputs
  --labels   king_turn_labels.jsonl.gz (per-turn rows; `loop == loop_onset`)
  --pivots   <digest>.jsonl from ops/king-review (rows with `admit == true`)
  --chunks   directory of trace chunks (traces/chunks/*.jsonl.gz from
             https://data.affine.io/traces/manifest.json)
Output
  --out/states.jsonl      one row per state (metadata only)
  --out/states/<id>.json  the full state: prefix messages (wire form), the
                          king's reply, task row, runtime, remaining budget

The prefix is `sampled_paths(trace)[turn_idx][:-1]` (affine.corpus.trace):
the exact root-to-node path of the sampled reply, so it is what the model was
sent for any harness. `turn_idx` counts sampled assistant replies, the same
index the labelers and the duel_turns view use.
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402
from affine.corpus.trace import sampled_paths  # noqa: E402

# Harnesses the resume plugin can continue (see plugin/recoverable_resume).
RESUMABLE = {
    "mini_swe_textbased": "textbased",
    "bash": "bash",
    "terminus_2": "terminus",
    "null": "null",
}
# ACP agents keep session state inside their own scaffold (todo lists, task
# trackers, subagents, compaction); a transcript is not enough to rebuild it.
NOT_RESUMABLE = {"claude_code", "pi", "kimi_code", "hermes_agent", "codex"}

MAX_TURNS_ORIGINAL = 80
MIN_REMAINING_TURNS = 20
FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
TRAILING_NUM_RE = re.compile(r"-(\d+)$")


def make_traj_id(instance_id: str, run_tag: str) -> str:
    """datagen/slicer.py::make_traj_id, kept inline so this script imports
    nothing from the pod-only datagen package."""
    m = TRAILING_NUM_RE.search(instance_id)
    num = m.group(1) if m else "0"
    stem = instance_id[: m.start()] if m else instance_id
    sha8 = hashlib.sha256(instance_id.encode()).hexdigest()[:8]
    return f"{stem}.{sha8}.pr_{num}__{run_tag}"


def run_tag(trace: dict) -> str:
    stamped = (trace.get("info") or {}).get("run_tag")
    if stamped:
        return stamped
    return hashlib.sha256(
        json.dumps(trace, sort_keys=True).encode()).hexdigest()[:8]


def wire_message(m: dict) -> dict:
    """Trace node message -> OpenAI wire dict (verifiers stores tool calls as
    {id, name, arguments})."""
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m["role"] == "assistant" and m.get("tool_calls"):
        out["tool_calls"] = [
            {"id": tc.get("id") or f"call_{i}", "type": "function",
             "function": {"name": tc.get("name") or (tc.get("function") or {}).get("name", ""),
                          "arguments": tc.get("arguments")
                          if tc.get("arguments") is not None
                          else (tc.get("function") or {}).get("arguments", "{}")}}
            for i, tc in enumerate(m["tool_calls"])]
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id") or ""
        if m.get("name"):
            out["name"] = m["name"]
    return out


def king_action(reply: dict, action_kind: str) -> str:
    """The king's action at the state, as the dialect parser sees it (mswea
    fences normalized to ```bash first, as the fold does)."""
    if reply.get("tool_calls"):
        return json.dumps([{"name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]}
                           for tc in reply["tool_calls"]], sort_keys=True)
    text = FOREIGN_FENCE_RE.sub("```bash\n", reply.get("content") or "")
    try:
        return dialects.last_action(text, action_kind)
    except dialects.UnknownDialect:
        return ""


def index_chunks(chunk_dir: Path, policy_prefix: str = "king_") -> dict:
    """rollout_id -> {chunk, line} for every envelope of a king policy."""
    idx: dict[str, dict] = {}
    for name in sorted(os.listdir(chunk_dir)):
        if not name.endswith(".jsonl.gz"):
            continue
        with gzip.open(chunk_dir / name, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                e = json.loads(line)
                if (e.get("policy") or {}).get("id", "").startswith(policy_prefix):
                    idx[e["rollout_id"]] = {"chunk": name, "line": i}
    return idx


class ChunkCache:
    def __init__(self, chunk_dir: Path, idx: dict):
        self.chunk_dir = chunk_dir
        self.idx = idx
        self._lines: dict[str, list[str]] = {}

    def load(self, rollout_id: str) -> dict | None:
        v = self.idx.get(rollout_id)
        if v is None:
            return None
        if v["chunk"] not in self._lines:
            with gzip.open(self.chunk_dir / v["chunk"], "rt", encoding="utf-8") as f:
                # split("\n"), not splitlines(): JSON strings may carry
                # U+2028 / U+001C-style separators that splitlines honours.
                self._lines[v["chunk"]] = f.read().split("\n")
            if len(self._lines) > 6:
                oldest = next(iter(self._lines))
                if oldest != v["chunk"]:
                    self._lines.pop(oldest)
        return json.loads(self._lines[v["chunk"]][v["line"]])


def load_onsets(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("loop") == "loop_onset" and r.get("outcome") == "failed"
                    and str(r.get("policy_id", "")).startswith("king_")):
                rows.append(r)
    return rows


def load_pivots(path: Path) -> list[dict]:
    return [r for r in (json.loads(l) for l in open(path, encoding="utf-8"))
            if r.get("admit")]


def build_state(envelope: dict, turn_idx: int, kind: str, label: dict) -> dict | None:
    trace = envelope["trace"]
    policy = envelope["policy"]
    harness = policy.get("harness", "")
    paths = sampled_paths(trace)
    if turn_idx >= len(paths):
        return None
    path = paths[turn_idx]
    if not path or path[-1]["role"] != "assistant":
        return None
    prefix = [wire_message(m) for m in path[:-1]]
    reply = wire_message(path[-1])
    agent_cfg = (trace.get("agent") or {}).get("config") or {}
    runtime = (trace.get("agent") or {}).get("runtime") or {}
    task_data = (trace.get("task") or {}).get("data") or {}
    action_kind = policy.get("action_kind") or "bash"
    sid = envelope["task"]["sid"]
    traj_id = make_traj_id(sid, run_tag(trace))
    remaining = max(MAX_TURNS_ORIGINAL - turn_idx, MIN_REMAINING_TURNS)
    state_id = f"{envelope['rollout_id']}:{turn_idx}:{kind}"
    return {
        "state_id": state_id,
        "state_kind": kind,
        "rollout_id": envelope["rollout_id"],
        "traj_id": traj_id,
        "turn_id": f"{traj_id}:{turn_idx}",
        "turn_idx": turn_idx,
        "node_id": label.get("node_id"),
        "depth": turn_idx,
        "n_replies": len(paths),
        "source": envelope["source"],
        "env_id": envelope["env_id"],
        "task": envelope["task"],
        "task_name": task_data.get("name") or envelope["task"]["uid"],
        "task_system_prompt": task_data.get("system_prompt"),
        "task_prompt": task_data.get("prompt") if isinstance(task_data.get("prompt"), str) else None,
        "task_answer": task_data.get("answer"),
        "policy_id": policy["id"],
        "harness": harness,
        "resume_kind": RESUMABLE.get(harness),
        "action_kind": action_kind,
        "king_model": policy.get("model"),
        "king_stop_condition": trace.get("stop_condition"),
        "runtime": runtime,
        "sampling": agent_cfg.get("sampling") or {"temperature": 0.8},
        "max_turns": remaining,
        "messages": prefix,
        "king_reply": reply,
        "king_action": king_action(reply, action_kind),
        "label": label,
        "prefix_chars": sum(len(m.get("content") or "") for m in prefix),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--labels", required=True, type=Path)
    ap.add_argument("--pivots", required=True, type=Path)
    ap.add_argument("--chunks", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--first-onset-only", action="store_true",
                    help="keep only the earliest loop onset of each rollout")
    ap.add_argument("--include-not-resumable", action="store_true")
    args = ap.parse_args()

    idx = index_chunks(args.chunks)
    cache = ChunkCache(args.chunks, idx)
    (args.out / "states").mkdir(parents=True, exist_ok=True)

    onsets = load_onsets(args.labels)
    if args.first_onset_only:
        first: dict[str, dict] = {}
        for r in onsets:
            cur = first.get(r["rollout_id"])
            if cur is None or r["turn_idx"] < cur["turn_idx"]:
                first[r["rollout_id"]] = r
        onsets = list(first.values())
    pivots = load_pivots(args.pivots)

    wanted: list[tuple[str, dict]] = [("loop_onset", r) for r in onsets]
    wanted += [("pivot", r) for r in pivots]
    counts: collections.Counter = collections.Counter()
    rows = []
    for kind, label in wanted:
        env = cache.load(label["rollout_id"])
        if env is None:
            counts[f"missing_trace:{label.get('policy_id')}"] += 1
            continue
        harness = env["policy"].get("harness", "")
        if harness not in RESUMABLE and not args.include_not_resumable:
            counts[f"not_resumable:{harness}"] += 1
            continue
        st = build_state(env, int(label["turn_idx"]), kind, label)
        if st is None:
            counts["bad_turn_idx"] += 1
            continue
        path = args.out / "states" / (st["state_id"].replace(":", "_") + ".json")
        path.write_text(json.dumps(st, ensure_ascii=False))
        meta = {k: v for k, v in st.items()
                if k not in ("messages", "king_reply", "task_system_prompt",
                             "task_prompt", "label")}
        meta["path"] = str(path)
        rows.append(meta)
        counts[f"state:{kind}:{harness}"] += 1
    with open(args.out / "states.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items()):
        print(f"{v:6d}  {k}")
    print(f"{len(rows)} states -> {args.out}")


if __name__ == "__main__":
    main()
