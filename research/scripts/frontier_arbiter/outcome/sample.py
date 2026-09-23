"""Stage 2 — CLASSIFY every candidate state: teacher x3 @T0.8, frontier
(glm-5.3) greedy x1 + @T0.8 x1 on the same prefix through Engy, then a
semantic judge (glm-5.3-flash, T=0) on the frontier-greedy vs closest-teacher
action pair.

    python sample.py --states /tmp/fa_outcome/states/states.jsonl \
        --out research/results/frontier_arbiter/outcome/samples.jsonl

Terms
  action        the harness action of a reply in the turn's dialect (bash =
                the ```bash block, tool_call = the <tool_call> JSON, terminus
                = the JSON command batch) after evalsrv.chat.split_rollout.
  norm-exact    norm_action(a) == norm_action(b) (whitespace/quotes unified).
  jaccard       token-set Jaccard on the action bodies (common.jaccard).
  CONTESTED     frontier greedy action is norm-exact-different from all 3
                teacher samples AND max jaccard < 0.5.
  AGREED        not contested (norm-exact match with >=1 teacher sample, or
                max jaccard >= 0.5).
  judge         glm-5.3-flash: are the frontier action and the closest
                teacher action "the same decision"? + a taxonomy label for
                the frontier's action relative to the teacher's.
Idempotent: states already in --out are skipped.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common import (Engy, TEACHER_ENGY, agree, append_jsonl, exact, jaccard,  # noqa: E402
                    read_jsonl, reply_to_rollout, write_jsonl)

FRONTIER = "glm-5.3"
JUDGE = "glm-5.3-flash"
SAMPLE_MAX_TOKENS = 16384
FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
BASH_FENCE_RE = re.compile(r"```bash[ \t]*\n(.*?)\n```", re.DOTALL)
MSWEA_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
CONTESTED_JACCARD = 0.5

JUDGE_PROMPT = """You compare two candidate next actions of a coding/terminal agent at the same point of the same task.

TASK (truncated):
{task}

LAST OBSERVATION the agent saw (truncated):
{obs}

ACTION A (model 1):
{a}

ACTION B (model 2):
{b}

Question 1 — same_decision: do A and B make the SAME DECISION (same intent and same target: e.g. both inspect the same file, both apply the same fix, both run the same test, both declare the task finished), ignoring syntax, quoting, flags that do not change the outcome, and the choice of tool? Answer true/false.

Question 2 — a_type: what kind of step is ACTION A? One of:
  "explore"  = reads/searches/inspects (cat, grep, ls, read, running an existing test to see output) without changing files
  "act"      = changes state toward a fix (edit a file, write code, install, git operations, create files)
  "verify"   = runs tests / a check after a change to confirm it
  "finish"   = declares the task complete / submits / gives a final answer
  "other"

Question 3 — relation of A to B (pick ONE):
  "same"            = same decision (then same_decision is true)
  "explore_more"    = A keeps investigating where B acts, verifies or finishes
  "act_or_finish"   = A acts, verifies or finishes where B keeps exploring
  "different_fix"   = both act/verify but the change or the tested thing differs
  "different_target"= both explore but a different file / place / question
  "other"

Reply with ONE JSON object only: {{"same_decision": true|false, "a_type": "...", "b_type": "...", "relation": "...", "why": "<=25 words"}}"""


def clip(s: str, n: int, tail: bool = False) -> str:
    s = s or ""
    if len(s) <= n:
        return s
    return ("…" + s[-n:]) if tail else (s[:n] + "…")


def last_observation(messages: list[dict]) -> str:
    parts = []
    for m in reversed(messages):
        if m["role"] == "assistant":
            break
        parts.append(m.get("content") or "")
    return "\n".join(reversed(parts))


def action_of(reply: dict, harness: str, kind: str) -> dict:
    """(z, y) of one Engy reply in the harness's dialect + harness validity."""
    r = dict(reply)
    content = r.get("content") or ""
    harness_valid = True
    fence_fixed = False
    if harness == "mini_swe_textbased":
        n_mswea = len(MSWEA_RE.findall(content))
        harness_valid = n_mswea == 1
        if n_mswea == 0 and len(BASH_FENCE_RE.findall(content)) == 1:
            # The action is the same; only the fence name is off. mini-swe
            # would have raised a FormatError, which we record.
            content = BASH_FENCE_RE.sub(lambda m: "```mswea_bash_command\n" + m.group(1) + "\n```", content, count=1)
            fence_fixed = True
        r["content"] = FOREIGN_FENCE_RE.sub("```bash\n", content)
    elif harness == "bash":
        harness_valid = bool(r.get("tool_calls")) or not content.strip() == ""
    out = reply_to_rollout(r, kind)
    out.update({"harness_valid": harness_valid, "fence_fixed": fence_fixed,
                "raw_content": reply.get("content") or "",
                "tool_calls": reply.get("tool_calls") or [],
                "reasoning_chars": len(reply.get("reasoning") or "")})
    if harness == "bash" and not r.get("tool_calls"):
        # A plain reply under the bash harness ends the episode ("finish").
        out["y"] = ""
        out["finish_reply"] = True
    return out


async def classify_one(engy: Engy, meta: dict, sem: asyncio.Semaphore) -> dict:
    async with sem:
        st = json.loads(Path(meta["path"]).read_text())
        harness, kind = st["harness"], st["action_kind"]
        msgs = st["messages"]
        extra = {"tools": st["tools"]} if harness == "bash" and st.get("tools") else {}
        t0 = time.time()
        calls = [engy.chat(TEACHER_ENGY, msgs, temperature=0.8, max_tokens=SAMPLE_MAX_TOKENS, **extra)
                 for _ in range(3)]
        calls.append(engy.chat(FRONTIER, msgs, temperature=0.0, max_tokens=SAMPLE_MAX_TOKENS, **extra))
        calls.append(engy.chat(FRONTIER, msgs, temperature=0.8, max_tokens=SAMPLE_MAX_TOKENS, **extra))
        res = await asyncio.gather(*calls, return_exceptions=True)
        row = {k: meta[k] for k in ("state_id", "rollout_id", "turn_idx", "depth", "source", "group",
                                    "harness", "resume_kind", "action_kind", "orig_outcome",
                                    "n_replies", "prefix_chars", "path")}
        row["teacher_action_orig"] = st["teacher_action"]
        row["errors"] = [repr(r)[:300] for r in res if isinstance(r, Exception)]
        parsed = [None if isinstance(r, Exception) else action_of(r, harness, kind) for r in res]
        row["teacher"] = parsed[:3]
        row["frontier_greedy"] = parsed[3]
        row["frontier_t08"] = parsed[4]
        row["sample_seconds"] = round(time.time() - t0, 1)
        row["cost_usd"] = sum(float((r.get("cost_usd") or 0)) for r in res if not isinstance(r, Exception))
        t_actions = [p["y"] for p in parsed[:3] if p and p["y"]]
        fg = parsed[3]
        row["n_teacher_parsed"] = len(t_actions)
        if fg and fg["y"] and t_actions:
            ys = fg["y"]
            row["f_exact_any"] = exact(ys, t_actions, kind)
            row["f_jaccard_max"] = agree(ys, t_actions)
            row["f_jaccard_orig"] = jaccard(ys, st["teacher_action"]) if st["teacher_action"] else None
            row["f_exact_orig"] = exact(ys, [st["teacher_action"]], kind)
            # Teacher self-agreement: how often do the 3 teacher samples agree?
            row["t_pairwise_exact"] = sum(exact(a, [b], kind) for i, a in enumerate(t_actions)
                                          for b in t_actions[i + 1:])
            row["t_pairwise_n"] = len(t_actions) * (len(t_actions) - 1) // 2
            row["t_jaccard_mean"] = (sum(jaccard(a, b) for i, a in enumerate(t_actions)
                                         for b in t_actions[i + 1:]) / row["t_pairwise_n"]
                                     if row["t_pairwise_n"] else None)
            row["f_t08_vs_greedy_jaccard"] = jaccard(ys, parsed[4]["y"]) if parsed[4] and parsed[4]["y"] else None
            row["contested"] = (not row["f_exact_any"]) and row["f_jaccard_max"] < CONTESTED_JACCARD
            row["closest_teacher_idx"] = max(range(len(t_actions)), key=lambda i: jaccard(ys, t_actions[i]))
            row["closest_teacher_action"] = t_actions[row["closest_teacher_idx"]]
            row["class"] = "contested" if row["contested"] else "agreed"
        elif fg and not fg["y"] and fg.get("finish_reply"):
            row["class"] = "frontier_finish"
        else:
            row["class"] = "unparsed"
        # Semantic judge on the (frontier greedy, closest teacher) pair.
        if row["class"] in ("contested", "agreed"):
            row["judge"] = await judge_pair(engy, st, row)
            row["cost_usd"] += float(row["judge"].pop("_cost", 0) or 0)
        return row


async def judge_pair(engy: Engy, st: dict, row: dict) -> dict:
    """glm-5.3-flash verdict on (frontier greedy, closest teacher). The model
    thinks first; an empty `content` at the cap is retried once with a larger
    cap and the JSON is also looked for in the reasoning text."""
    fg = row["frontier_greedy"]
    prompt = JUDGE_PROMPT.format(
        task=clip(st.get("task_prompt") or "", 2500),
        obs=clip(last_observation(st["messages"]), 3000, tail=True),
        a=clip(fg["y"], 2500), b=clip(row["closest_teacher_action"], 2500))
    cost = 0.0
    last = None
    for cap in (4096, 12000):
        try:
            j = await engy.chat(JUDGE, [{"role": "user", "content": prompt}], temperature=0.0,
                                max_tokens=cap)
        except Exception as e:  # noqa: BLE001
            return {"error": repr(e)[:300], "_cost": cost}
        cost += float(j.get("cost_usd") or 0)
        for text in (j["content"], j.get("reasoning") or ""):
            m = re.search(r"\{[^{}]*\"same_decision\"[^{}]*\}", text, re.S)
            if m:
                try:
                    out = json.loads(m.group(0))
                    out["_cost"] = cost
                    return out
                except json.JSONDecodeError:
                    pass
        last = {"error": "no json", "raw": j["content"][:300], "finish": j.get("finish"),
                "reasoning_chars": len(j.get("reasoning") or "")}
    last["_cost"] = cost
    return last


async def rejudge(args: argparse.Namespace) -> None:
    rows = read_jsonl(args.out)
    engy = Engy(concurrency=args.concurrency, timeout=1200)
    todo = [r for r in rows if r.get("class") in ("contested", "agreed")
            and "same_decision" not in (r.get("judge") or {})]
    print(f"re-judging {len(todo)} rows")

    async def one(r):
        st = json.loads(Path(r["path"]).read_text())
        r["judge"] = await judge_pair(engy, st, r)
        r["cost_usd"] += float(r["judge"].pop("_cost", 0) or 0)
    await asyncio.gather(*[one(r) for r in todo])
    write_jsonl(args.out, rows)
    print("still failing:", sum(1 for r in todo if "same_decision" not in r["judge"]),
          f"judge cost ${engy.cost_usd:.3f}")


async def main_async(args: argparse.Namespace) -> None:
    if args.rejudge:
        await rejudge(args)
        return
    metas = read_jsonl(args.states)
    done = {r["state_id"] for r in read_jsonl(args.out)}
    todo = [m for m in metas if m["state_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(metas)} states, {len(done)} done, {len(todo)} to classify")
    engy = Engy(concurrency=args.concurrency, timeout=1200)
    sem = asyncio.Semaphore(args.parallel_states)
    n = 0
    for coro in asyncio.as_completed([classify_one(engy, m, sem) for m in todo]):
        row = await coro
        n += 1
        append_jsonl(args.out, row)
        print(f"[{n}/{len(todo)}] {row['harness']:18s} d={row['depth']:2d} {row['class']:16s} "
              f"jac={row.get('f_jaccard_max', float('nan')):.2f} judge={(row.get('judge') or {}).get('relation')} "
              f"${row['cost_usd']:.3f} (run ${engy.cost_usd:.2f})", flush=True)
        if engy.cost_usd > args.budget_usd:
            print("budget reached, stopping")
            break
    json.dump(engy.usage, open(args.out.with_suffix(".usage.json"), "w"), indent=1)
    print(json.dumps(engy.usage, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--states", type=Path, default=Path("/tmp/fa_outcome/states/states.jsonl"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--parallel-states", type=int, default=6)
    ap.add_argument("--budget-usd", type=float, default=40.0)
    ap.add_argument("--rejudge", action="store_true", help="only redo failed judge calls in --out")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
