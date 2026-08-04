"""Build D_tau2 turns from Jarrodbarnes/tau2-sft-v4-dataset.

Two action contracts:
  --mode bash   (legacy probe): rewrite system to THOUGHT+```bash tool JSON
  --mode native (default now): keep original tau2 system; assistant =
                <thinking>z</thinking>\\n{tool_json}; action_kind=tool

Usage:
  source .venv/bin/activate
  python scripts/build_turns_tau2.py --n 200 --seed 7 --mode native \\
      --out data/turns_tau2_native.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

THINK_RE = re.compile(r"<thinking>(.*?)</thinking>", re.DOTALL | re.IGNORECASE)

SYSTEM_TEMPLATE = """You are a customer-service agent that resolves user issues by calling tools.
Your response must contain exactly ONE tool call as a fenced bash block (the JSON payload is the action).

Include a THOUGHT section before the tool call where you explain your reasoning.
Format your response as shown in <example>.

<example>
THOUGHT: Your reasoning and analysis here

```bash
{{"name": "tool_name", "arguments": {{"key": "value"}}}}
```
</example>

Failure to follow these rules will cause your response to be rejected.

Domain policy and available tools (from the original task):
{policy}
"""


def domain_of(task_id: str) -> str:
    tid = task_id.lower()
    for d in ("airline", "retail", "telecom"):
        if d in tid:
            return d
    return "other"


def _extract_tool_json(text: str) -> dict | None:
    """Pull the first JSON object with a tool `name` from text."""
    dec = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = dec.raw_decode(text[i:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "name" in obj:
            return obj
    return None


def parse_assistant(content: str) -> tuple[str, str] | None:
    """Return (thought, tool_json) or None if unparseable."""
    m = THINK_RE.search(content)
    thought = m.group(1).strip() if m else ""
    rest = content[m.end() :].strip() if m else content.strip()
    obj = _extract_tool_json(rest) or _extract_tool_json(content)
    if not obj:
        return None
    tool = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    return thought, tool


def to_assistant(thought: str, tool_json: str, mode: str) -> str:
    body = thought if thought else "Proceed with the next tool call."
    if mode == "native":
        return f"<thinking>{body}</thinking>\n{tool_json}"
    return f"THOUGHT: {body}\n\n```bash\n{tool_json}\n```"


def tool_user_message(name: str | None, content: str) -> str:
    label = name or "tool"
    return f"<tool_result name=\"{label}\">\n{content}\n</tool_result>"


def convert_trace(row: dict, mode: str = "native") -> list[dict]:
    task_id = row.get("task_id", "unknown")
    msgs = row.get("prompt") or []
    if not msgs or msgs[0].get("role") != "system":
        return []
    policy = msgs[0].get("content", "").strip()
    if mode == "native":
        system = {"role": "system", "content": policy}
        action_kind = "tool"
    else:
        system = {
            "role": "system",
            "content": SYSTEM_TEMPLATE.format(policy=policy[:6000]),
        }
        action_kind = "bash"

    turns: list[dict] = []
    history: list[dict] = [system]
    assistant_i = 0
    for msg in msgs[1:]:
        role = msg.get("role")
        content = msg.get("content") or ""
        if role == "user":
            history.append({"role": "user", "content": content})
            continue
        if role == "tool":
            history.append({
                "role": "user",
                "content": tool_user_message(msg.get("name"), content),
            })
            continue
        if role != "assistant":
            continue
        parsed = parse_assistant(content)
        if not parsed:
            break
        thought, tool = parsed
        asst = to_assistant(thought, tool, mode)
        prefix = [dict(m) for m in history]
        if not any(m["role"] == "user" for m in prefix):
            history.append({"role": "assistant", "content": asst})
            assistant_i += 1
            continue
        turns.append({
            "traj_id": f"{task_id}__a{assistant_i}",
            "turn_idx": assistant_i,
            "domain": domain_of(task_id),
            "action_kind": action_kind,
            "prefix": prefix,
            "reference_turn": asst,
            "n_prefix_chars": sum(len(m.get("content", "")) for m in prefix),
            "suffix_len": len(asst),
        })
        history.append({"role": "assistant", "content": asst})
        assistant_i += 1
    return turns


def stratified_sample(rows: list[dict], n: int, seed: int) -> list[dict]:
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r.get("domain", "other")].append(r)
    rng = random.Random(seed)
    for v in by.values():
        rng.shuffle(v)
    keys = sorted(by)
    out: list[dict] = []
    idx = {k: 0 for k in keys}
    while len(out) < n:
        progress = False
        for k in keys:
            i = idx[k]
            if i < len(by[k]):
                out.append(by[k][i])
                idx[k] = i + 1
                progress = True
                if len(out) >= n:
                    break
        if not progress:
            break
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--mode", choices=("native", "bash"), default="native")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-prefix-chars", type=int, default=24000,
                    help="drop turns whose prefix is too long for 32k ctx")
    args = ap.parse_args()
    if args.out is None:
        args.out = Path(
            "data/turns_tau2_native.jsonl" if args.mode == "native"
            else "data/turns_tau2.jsonl"
        )

    path = hf_hub_download(
        "Jarrodbarnes/tau2-sft-v4-dataset",
        "blended_traces_v4.jsonl",
        repo_type="dataset",
    )
    all_turns: list[dict] = []
    n_traces = 0
    for line in open(path):
        n_traces += 1
        all_turns.extend(convert_trace(json.loads(line), mode=args.mode))

    kept = [t for t in all_turns if t["n_prefix_chars"] <= args.max_prefix_chars]
    print(f"mode={args.mode} traces={n_traces} turns={len(all_turns)} "
          f"kept_len={len(kept)}")
    print("domain_all", Counter(t["domain"] for t in kept))

    picked = stratified_sample(kept, args.n, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for t in picked:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"wrote {len(picked)} → {args.out}")
    print("domain_picked", Counter(t["domain"] for t in picked))
    print("action_kind", Counter(t.get("action_kind") for t in picked))
    print("prefix_chars p50", sorted(t["n_prefix_chars"] for t in picked)[len(picked) // 2])
    print("ref sample:\n", picked[0]["reference_turn"][:300])


if __name__ == "__main__":
    main()
