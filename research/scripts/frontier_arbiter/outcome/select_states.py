"""Stage 3 — pick the KEPT states and write the per-arm state files for the
pod driver (ops/recoverable/run_states.py, patched copy in pod/).

    python select_states.py --samples research/results/frontier_arbiter/outcome/samples.jsonl \
        --out /tmp/fa_outcome/arms --n-contested 50 --n-agreed 25

Kept CONTESTED = surface rule contested AND judge says not the same decision
AND the frontier's greedy reply is harness-valid (a forced reply the harness
could have executed). Kept AGREED = surface rule agreed AND judge says same
decision. Balanced across harnesses as far as supply allows; inside a harness
balanced across the original trajectory outcome (solved / failed).

Arms (one state file each, same state_id suffix scheme <rollout>:<turn>:<arm>):
  T   the teacher continues from the state (plain recoverable run)
  F   glm-5.3 continues from the state (driver --model glm-5.3)
  F1  the frontier's GREEDY reply is forced as the next step (forced_reply),
      then the teacher continues
Writes <out>/{T,F,F1}/states.jsonl + states/*.json and <out>/kept.jsonl.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import read_jsonl, write_jsonl  # noqa: E402

META_DROP = ("messages", "teacher_reply", "king_reply", "task_system_prompt", "task_prompt",
             "tools", "forced_reply")


def forced_reply_of(row: dict) -> dict:
    fg = row["frontier_greedy"]
    tcs = []
    for i, tc in enumerate(fg.get("tool_calls") or []):
        fn = tc.get("function") or {}
        tcs.append({"id": tc.get("id") or f"call_forced_{i}", "type": "function",
                    "function": {"name": fn.get("name") or "", "arguments": fn.get("arguments") or "{}"}})
    content = fg.get("raw_content") or ""
    if row["harness"] == "mini_swe_textbased" and fg.get("fence_fixed"):
        # The forced reply must carry the fence mini-swe parses; the frontier
        # wrote ```bash, the action is unchanged.
        content = content.replace("```bash\n", "```mswea_bash_command\n", 1)
    return {"role": "assistant", "content": content, "tool_calls": tcs}


def eligible(row: dict) -> str | None:
    j = row.get("judge") or {}
    if row.get("class") == "contested" and j.get("same_decision") is False \
            and (row["frontier_greedy"] or {}).get("harness_valid", False) \
            and (row["frontier_greedy"] or {}).get("y"):
        return "contested"
    if row.get("class") == "agreed" and j.get("same_decision") is True:
        return "agreed"
    return None


def pick(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Balanced by harness, then by original outcome inside a harness."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["harness"], r["orig_outcome"])].append(r)
    for v in by.values():
        rng.shuffle(v)
    harnesses = sorted({h for h, _ in by})
    out: list[dict] = []
    while len(out) < n and any(by.values()):
        for h in harnesses:
            for oc in ("solved", "failed"):
                if len(out) >= n:
                    break
                if by[(h, oc)]:
                    out.append(by[(h, oc)].pop())
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--samples", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("/tmp/fa_outcome/arms"))
    ap.add_argument("--n-contested", type=int, default=50)
    ap.add_argument("--n-agreed", type=int, default=25)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--include", type=Path, default=None,
                    help="kept.jsonl of an earlier (partial) selection whose states are kept first")
    args = ap.parse_args()
    include = {r["state_id"] for r in read_jsonl(args.include)} if args.include else set()
    rng = random.Random(args.seed)
    rows = read_jsonl(args.samples)
    pools = collections.defaultdict(list)
    for r in rows:
        e = eligible(r)
        if e:
            pools[e].append(r)
    print("eligible:", {k: len(v) for k, v in pools.items()},
          "by harness:", {k: collections.Counter(r["harness"] for r in v) for k, v in pools.items()})
    kept = []
    for cls, n in (("contested", args.n_contested), ("agreed", args.n_agreed)):
        first = [r for r in pools[cls] if r["state_id"] in include]
        rest = [r for r in pools[cls] if r["state_id"] not in include]
        kept += [dict(r, kept_class=cls) for r in first[:n] + pick(rest, max(0, n - len(first)), rng)]
    for arm in ("T", "F", "F1"):
        (args.out / arm / "states").mkdir(parents=True, exist_ok=True)
    arm_rows = {"T": [], "F": [], "F1": []}
    for r in kept:
        st = json.loads(Path(r["path"]).read_text())
        for arm in ("T", "F", "F1"):
            if arm == "F" and r["kept_class"] != "contested":
                continue
            s = dict(st)
            s["state_id"] = st["state_id"].replace(":frontier", f":{arm}")
            s["arm"] = arm
            s["kept_class"] = r["kept_class"]
            if arm == "F1":
                s["forced_reply"] = forced_reply_of(r)
                s["forced_action"] = r["frontier_greedy"]["y"]
            p = args.out / arm / "states" / (s["state_id"].replace(":", "_") + ".json")
            p.write_text(json.dumps(s, ensure_ascii=False))
            meta = {k: v for k, v in s.items() if k not in META_DROP}
            meta["path"] = str(p)
            arm_rows[arm].append(meta)
    for arm, metas in arm_rows.items():
        write_jsonl(args.out / arm / "states.jsonl", metas)
        print(f"{arm}: {len(metas)} states")
    write_jsonl(args.out / "kept.jsonl", kept)
    c = collections.Counter((r["kept_class"], r["harness"], r["orig_outcome"]) for r in kept)
    for k, v in sorted(c.items()):
        print(f"{v:4d} {k}")


if __name__ == "__main__":
    main()
