"""Stage 1 — candidate STATES for the frontier-arbiter outcome probe.

A state = a prefix of a TEACHER trajectory (everything the model saw before
reply `turn_idx`), on a harness the recoverable plugin can resume
(mini-swe textbased / verifiers bash / Terminus 2), from a rollout the
environment graded (solved or failed), depth 3–15. One state per rollout.

    python build_states.py --index /tmp/fa_outcome/teacher_index.jsonl \
        --chunks /tmp/fa_outcome/chunks --out /tmp/fa_outcome/states --n 260

Writes <out>/states.jsonl (metadata) and <out>/states/<id>.json (full state
in the layout ops/recoverable/states.py produces, plus `tools` for the bash
harness and `teacher_reply` = the reply the teacher actually gave there).
The prefix is `sampled_paths(trace)[turn_idx][:-1]` — byte-for-byte what
the model was sent (affine.corpus.trace, live tree).
"""

from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from common import REPO, render_tool_call  # noqa: E402

LIVE_TREE = Path(os.environ.get("AFFINE_LIVE_TREE", "/tmp/box/affine"))
sys.path.insert(0, str(LIVE_TREE if (LIVE_TREE / "affine").is_dir() else REPO / "affine"))
from affine import dialects  # noqa: E402
from affine.corpus.trace import sampled_paths  # noqa: E402

RESUMABLE = {"mini_swe_textbased": "textbased", "bash": "bash", "terminus_2": "terminus"}
ACTION_KIND = {"mini_swe_textbased": "bash", "bash": "tool_call", "terminus_2": "terminus_json"}
CODING = {"swesmith", "multiswe", "scaleswe", "swerebench_v2", "swelego"}
MAX_TURNS_ORIGINAL = 80
MIN_REMAINING_TURNS = 20
MIN_DEPTH, MAX_DEPTH = 3, 15
MAX_PREFIX_CHARS = 300_000
FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
MSWEA_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
TRAILING_NUM_RE = re.compile(r"-(\d+)$")

# Verifiers' bash harness tool schemas (harnesses/bash/program.py), used when
# the trace stored no `tools` (a few early rollouts).
BASH_TOOL = {"type": "function", "function": {
    "name": "bash", "description": "Run a bash command and return its combined stdout and stderr.",
    "parameters": {"type": "object", "properties": {
        "command": {"type": "string", "description": "The bash command to run."}},
        "required": ["command"]}}}
EDIT_TOOL = {"type": "function", "function": {
    "name": "edit",
    "description": "Replace a unique string in a file. old_str must appear exactly once in the file.",
    "parameters": {"type": "object", "properties": {
        "path": {"type": "string", "description": "File path (relative to cwd or absolute)."},
        "old_str": {"type": "string", "description": "Exact string to find (must appear exactly once)."},
        "new_str": {"type": "string", "description": "Replacement string."}},
        "required": ["path", "old_str", "new_str"]}}}


def make_traj_id(instance_id: str, run_tag: str) -> str:
    m = TRAILING_NUM_RE.search(instance_id)
    num = m.group(1) if m else "0"
    stem = instance_id[: m.start()] if m else instance_id
    sha8 = hashlib.sha256(instance_id.encode()).hexdigest()[:8]
    return f"{stem}.{sha8}.pr_{num}__{run_tag}"


def run_tag(trace: dict) -> str:
    stamped = (trace.get("info") or {}).get("run_tag")
    if stamped:
        return stamped
    return hashlib.sha256(json.dumps(trace, sort_keys=True).encode()).hexdigest()[:8]


def wire_message(m: dict) -> dict:
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


def reply_action(reply: dict, action_kind: str) -> str:
    """The action of one wire assistant message as the dialect parser sees it
    — the same rendering common.reply_to_rollout gives an Engy reply, so the
    stored teacher action compares 1:1 with fresh samples."""
    text = FOREIGN_FENCE_RE.sub("```bash\n", reply.get("content") or "")
    if reply.get("tool_calls"):
        text = text.rstrip() + "\n" + "\n".join(render_tool_call(c) for c in reply["tool_calls"])
    try:
        return dialects.last_action(text, action_kind)
    except dialects.UnknownDialect:
        return ""


def tool_schemas(trace: dict) -> list[dict]:
    tools = trace.get("tools") or []
    out = []
    for t in tools:
        if t.get("type") == "function" and t.get("function"):
            out.append(t)
        elif t.get("name"):
            out.append({"type": "function", "function": {
                "name": t["name"], "description": t.get("description") or "",
                "parameters": t.get("parameters") or t.get("input_schema") or {}}})
    return out or [BASH_TOOL, EDIT_TOOL]


def replayable(prefix: list[dict], harness: str) -> bool:
    """Every assistant message of the prefix must be something the replay can
    execute (textbased: exactly one mswea block; bash: tool calls with JSON
    object args or a plain reply; terminus: anything, parse errors are skipped)."""
    if harness == "mini_swe_textbased":
        return all(len(MSWEA_RE.findall(m.get("content") or "")) == 1
                   for m in prefix if m["role"] == "assistant")
    if harness == "bash":
        for m in prefix:
            for tc in (m.get("tool_calls") or []) if m["role"] == "assistant" else []:
                try:
                    if not isinstance(json.loads(tc["function"]["arguments"] or "{}"), dict):
                        return False
                except (TypeError, json.JSONDecodeError):
                    return False
    return True


def build_state(envelope: dict, turn_idx: int) -> dict | None:
    trace = envelope["trace"]
    policy = envelope["policy"]
    harness = policy.get("harness", "")
    paths = sampled_paths(trace)
    if turn_idx >= len(paths):
        return None
    path = paths[turn_idx]
    if not path or path[-1]["role"] != "assistant" or path[-2]["role"] not in ("user", "tool"):
        return None
    prefix = [wire_message(m) for m in path[:-1]]
    if not replayable(prefix, harness):
        return None
    reply = wire_message(path[-1])
    action_kind = ACTION_KIND[harness]
    teacher_action = reply_action(reply, action_kind)
    if not teacher_action:
        return None
    prefix_chars = sum(len(m.get("content") or "") for m in prefix)
    if prefix_chars > MAX_PREFIX_CHARS:
        return None
    agent_cfg = (trace.get("agent") or {}).get("config") or {}
    runtime = (trace.get("agent") or {}).get("runtime") or {}
    task_data = (trace.get("task") or {}).get("data") or {}
    sid = envelope["task"]["sid"]
    traj_id = make_traj_id(sid, run_tag(trace))
    state_id = f"{envelope['rollout_id']}:{turn_idx}:frontier"
    return {
        "state_id": state_id,
        "proxy_key": None,
        "state_kind": "frontier",
        "rollout_id": envelope["rollout_id"],
        "traj_id": traj_id,
        "turn_id": f"{traj_id}:{turn_idx}",
        "turn_idx": turn_idx,
        "depth": turn_idx,
        "n_replies": len(paths),
        "source": envelope["source"],
        "group": "coding" if envelope["source"] in CODING else "terminal",
        "env_id": envelope["env_id"],
        "task": envelope["task"],
        "task_name": task_data.get("name") or envelope["task"]["uid"],
        "task_system_prompt": task_data.get("system_prompt"),
        "task_prompt": task_data.get("prompt") if isinstance(task_data.get("prompt"), str) else None,
        "policy_id": policy["id"],
        "harness": harness,
        "resume_kind": RESUMABLE[harness],
        "action_kind": action_kind,
        "teacher_model": policy.get("model"),
        "orig_stop_condition": trace.get("stop_condition"),
        "orig_outcome": None,      # filled from the index row
        "runtime": runtime,
        "sampling": agent_cfg.get("sampling") or {"temperature": 0.8},
        "max_turns": max(MAX_TURNS_ORIGINAL - turn_idx, MIN_REMAINING_TURNS),
        "tools": tool_schemas(trace) if harness == "bash" else [],
        "messages": prefix,
        "teacher_reply": reply,
        # `king_*` names keep aggregate.py / the plugin happy: the plugin only
        # reads messages / resume_kind / sampling / max_turns / forced_reply.
        "king_reply": reply,
        "king_action": teacher_action,
        "teacher_action": teacher_action,
        "prefix_chars": prefix_chars,
        "stored_at": envelope.get("stored_at"),
    }


class ChunkCache:
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = chunk_dir
        self._lines: dict[str, list[str]] = {}

    def load(self, chunk: str, line: int) -> dict:
        if chunk not in self._lines:
            with gzip.open(self.chunk_dir / chunk, "rt", encoding="utf-8") as f:
                self._lines[chunk] = f.read().split("\n")
            while len(self._lines) > 4:
                oldest = next(iter(self._lines))
                if oldest == chunk:
                    break
                self._lines.pop(oldest)
        return json.loads(self._lines[chunk][line])


# Target candidate counts per (harness, group, outcome); the terminus supply
# is terminal_bench_2 only (~85 graded teacher rollouts in the window).
QUOTA = {
    ("mini_swe_textbased", "coding", "solved"): 35, ("mini_swe_textbased", "coding", "failed"): 35,
    ("mini_swe_textbased", "terminal", "solved"): 15, ("mini_swe_textbased", "terminal", "failed"): 15,
    ("bash", "coding", "solved"): 35, ("bash", "coding", "failed"): 35,
    ("bash", "terminal", "solved"): 15, ("bash", "terminal", "failed"): 15,
    ("terminus_2", "terminal", "solved"): 30, ("terminus_2", "terminal", "failed"): 30,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--index", type=Path, default=Path("/tmp/fa_outcome/teacher_index.jsonl"))
    ap.add_argument("--chunks", type=Path, default=Path("/tmp/fa_outcome/chunks"))
    ap.add_argument("--manifest", type=Path, default=Path("/tmp/fa_outcome/traces_manifest.json"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/fa_outcome/states"))
    ap.add_argument("--since", default="2026-09-05",
                    help="chunk created_at floor (task images of recent rollouts are pullable)")
    ap.add_argument("--multiswe-since", default="2026-08-25",
                    help="multiswe has no teacher rollouts after 09-05; its mswebench/ images persist")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every quota")
    args = ap.parse_args()
    rng = random.Random(args.seed)
    created = {ch["key"].split("/")[-1]: ch["created_at"] for ch in json.load(open(args.manifest))["chunks"]}
    rows = [json.loads(l) for l in open(args.index)]
    pool: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        floor = args.multiswe_since if r["source"] == "multiswe" else args.since
        if created.get(r["chunk"], "") < floor or r["outcome"] not in ("solved", "failed"):
            continue
        if r["n_replies"] < MIN_DEPTH + 2:
            continue
        group = "coding" if r["source"] in CODING else "terminal"
        pool[(r["harness"], group, r["outcome"])].append(r)
    for k in pool:
        rng.shuffle(pool[k])
    (args.out / "states").mkdir(parents=True, exist_ok=True)
    cache = ChunkCache(args.chunks)
    metas: list[dict] = []
    counts: collections.Counter = collections.Counter()
    seen_tasks: set[str] = set()
    for key, quota in QUOTA.items():
        want = int(round(quota * args.scale))
        got = 0
        # Spread over sources inside the bucket: round-robin by source.
        by_src: dict[str, list[dict]] = collections.defaultdict(list)
        for r in pool.get(key, []):
            by_src[r["source"]].append(r)
        order = sorted(by_src)
        while got < want and any(by_src.values()):
            for src in order:
                if got >= want or not by_src[src]:
                    continue
                r = by_src[src].pop()
                if r["sid"] in seen_tasks:
                    counts["dup_task"] += 1
                    continue
                env = cache.load(r["chunk"], r["line"])
                hi = min(MAX_DEPTH, r["n_replies"] - 2)
                turn_idx = rng.randint(MIN_DEPTH, hi)
                st = build_state(env, turn_idx)
                if st is None:
                    counts[f"unbuildable:{key[0]}"] += 1
                    continue
                st["orig_outcome"] = r["outcome"]
                st["orig_rewards"] = r["rewards"]
                seen_tasks.add(r["sid"])
                path = args.out / "states" / (st["state_id"].replace(":", "_") + ".json")
                path.write_text(json.dumps(st, ensure_ascii=False))
                meta = {k: v for k, v in st.items()
                        if k not in ("messages", "teacher_reply", "king_reply", "task_system_prompt",
                                     "task_prompt", "tools")}
                meta["path"] = str(path)
                metas.append(meta)
                got += 1
                counts[f"state:{key}"] += 1
    with open(args.out / "states.jsonl", "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    for k, v in sorted(counts.items()):
        print(f"{v:6d}  {k}")
    print(f"{len(metas)} states -> {args.out}")


if __name__ == "__main__":
    main()
