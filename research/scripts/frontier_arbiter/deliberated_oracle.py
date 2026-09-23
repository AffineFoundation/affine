"""Deliberated frontier oracle probe (2026-09-21).

Question: does a DELIBERATED frontier — many long-thinking samples at the same
turn prefix, clustered into decision classes by self-consistency — agree with
itself and with the environment outcome far better than the single frontier
sample we tested (single-sample self-agreement 0.27 norm-exact / 0.69 Jaccard,
frontier-vs-teacher 0.25)? If yes, an offline per-state "right action" oracle
built from frontier deliberation could be the ranking target.

    python deliberated_oracle.py sample  [--states primary|all] [--variants v1,v2,v3,v4]
    python deliberated_oracle.py analyze
    (default: both)

Ground truth = states whose first actions carry an ENVIRONMENT OUTCOME:
  * research/results/frontier_arbiter/split_states/  (36 states; 7 SPLIT = teacher
    first actions that led to solved AND failed continuations; 11 CEILING =
    teacher 0/N with frontier proposals executed under arm F1)
  * research/results/frontier_arbiter/outcome/       (yesterday's paired T / F1
    continuations; used for extra labelled pairs and extra states)

Terms (one line each)
  state           a turn prefix x (messages the agent saw) at a fixed depth of a stored trajectory
  labelled action a first action at the state whose full continuation was graded by the env (solved / failed)
  class           an equivalence class of actions: tier 1 norm-exact, tier 2 token-Jaccard >= 0.5 (single linkage),
                  tier 3 glm-5.3-flash T=0 "same decision?" judge on the residual class-representative pairs
  modal class     the class holding most of a variant's parsed samples at a state; its share = CONFIDENCE
  single-sample   pairwise agreement between two independent draws (what one frontier sample vs another gives)
  correct/wrong   modal class contains a solved / only failed labelled action; MIXED = both; NOVEL = no labelled member
  accuracy        correct / (correct + wrong); mixed and novel reported separately
  v1              deliberated sampling: k=10 glm-5.3 T0.8 max_tokens 8192 on the exact prefix
  v2              critique-then-act: prefix + the teacher's N first actions and the king's action (A/B/C, no outcomes),
                  think long, critique each, output the best or a repaired action; k=3 (max_tokens 16384)
  v3              self-critique: one long sample, then a second call that critiques and revises it; k=3 chains
  v4              cross-frontier: k=5 deepseek-v4.1-flash long-thinking samples; joint = v1 + v4 pooled
Outputs: research/results/frontier_arbiter/deliberated/{samples.jsonl, judge_cache.jsonl, report.txt, report.json}
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import hashlib
import json
import math
import re
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (Engy, append_jsonl, jaccard, norm_action, read_jsonl,  # noqa: E402
                    render_tool_call, reply_to_rollout)

REPO = HERE.parents[2]
RES = REPO / "research" / "results" / "frontier_arbiter"
OUT = RES / "deliberated"
SAMPLES = OUT / "samples.jsonl"
JUDGE_CACHE = OUT / "judge_cache.jsonl"
STATE_DIR = Path("/tmp/fa_outcome/states/states")

FRONTIER = "glm-5.3"
FRONTIER_FALLBACK = "glm-5.3-flash"
JUDGE = "glm-5.3-flash"
CROSS = "deepseek-v4.1-flash"
CROSS_EXTRA = {"chat_template_kwargs": {"thinking": True}}
MAX_TOKENS = 8192            # v1 / v4 (spec)
CRITIQUE_MAX_TOKENS = 16384  # v2 / v3: the critique prompts make glm-5.3 think 5-15k tokens under the tool harness
LOOSE = 0.5
K = {"v1": 10, "v2": 3, "v3": 3, "v4": 5}
# List prices (USD per 1M tokens), used only for the projection table; the
# realised charge comes from x_engy.charged_micro.
PRICES = {"glm-5.3": (0.75, 2.84), "glm-5.3-flash": (0.10, 0.36),
          "deepseek-v4.1-flash": (0.03, 0.03), "kimi-k3": (1.5, 6.6)}

FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
BASH_FENCE_RE = re.compile(r"```bash[ \t]*\n(.*?)\n```", re.DOTALL)
MSWEA_RE = re.compile(r"```mswea_bash_command\s*\n(.*?)\n```", re.DOTALL)
# DeepSeek sometimes answers the text-based harness in its native DSML tool syntax,
# treating `mswea_bash_command` as a tool; the decision is recoverable.
DSML_RE = re.compile(r"<｜DSML｜\s*invoke name=\"mswea_bash_command\">\s*<｜DSML｜\s*parameter name=\"command\"[^>]*>(.*?)</｜DSML｜\s*parameter>", re.DOTALL)

JUDGE_PROMPT = """You compare two candidate next actions of a coding/terminal agent at the same point of the same task.

TASK (truncated):
{task}

LAST OBSERVATION the agent saw (truncated):
{obs}

ACTION A:
{a}

ACTION B:
{b}

Do A and B make the SAME DECISION — same intent and same target (e.g. both inspect the same file or the same question, both apply the same fix, both run the same test, both declare the task finished) — ignoring syntax, quoting, flags that do not change the outcome, and the choice of tool? Two different exploratory reads of DIFFERENT files/places are NOT the same decision; two different fixes are NOT the same decision.

Reply with ONE JSON object only: {{"same_decision": true|false, "why": "<=20 words"}}"""

CRITIQUE_INSTR = """Before you act, consider the following candidate next actions that other agents proposed at exactly this point (their outcomes are unknown to you):

{cands}

Think it through carefully (you have a generous but finite thinking budget — reach a decision and act): what does the task need next, what has already been established by the observations above, and what would each candidate accomplish or miss? Critique each candidate briefly in your thinking. Then reply EXACTLY as you would as the agent — same output format as your previous replies in this conversation (one action only) — with either the best candidate (possibly repaired) or a better action of your own. Do not mention the candidate letters in the final reply."""

SELF_CRITIQUE_INSTR = """Here is the reply you drafted for this turn (not yet executed):

<<<DRAFT
{draft}
DRAFT

Before it runs: think carefully and critique it (generous but finite thinking budget — reach a decision and act) — is this the right next step given the task and everything the observations above established? Would a different target, a fix instead of more reading, a verification, or finishing be better? Then give your FINAL reply for this turn, EXACTLY in the agent's output format used in this conversation (one action only) — either the draft unchanged or a revised action."""


# ------------------------------------------------------------------ helpers
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


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k: int, n: int) -> str:
    if not n:
        return "   —   "
    lo, hi = wilson(k, n)
    return f"{k}/{n} = {100 * k / n:5.1f}% [{100 * lo:3.0f}–{100 * hi:3.0f}]"


def fmt(x, nd=2) -> str:
    return "  —  " if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _balanced_end(text: str, pos: int) -> int:
    depth, in_str, esc = 0, False, False
    for i in range(pos, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return -1


def terminus_fallback(content: str) -> str:
    """Last balanced JSON object carrying `commands` (a reply with two objects
    forfeits under the dialect; for LABELLING we still want the action)."""
    best = ""
    pos = content.find("{")
    while pos != -1:
        end = _balanced_end(content, pos)
        if end == -1:
            break
        frag = content[pos:end]
        try:
            d = json.loads(frag)
            if isinstance(d, dict) and isinstance(d.get("commands"), list):
                best = frag
        except ValueError:
            pass
        pos = content.find("{", pos + 1)
    return best


def action_of(reply: dict, harness: str, kind: str) -> dict:
    """(z, y) of an Engy reply in the harness dialect + harness validity flags."""
    r = dict(reply)
    content = r.get("content") or ""
    harness_valid = True
    fence_fixed = False
    dsml_fixed = False
    if harness == "mini_swe_textbased":
        m = DSML_RE.search(content)
        if m and not MSWEA_RE.search(content) and not BASH_FENCE_RE.search(content):
            content = "```bash\n" + m.group(1).strip() + "\n```"
            harness_valid = False
            dsml_fixed = True
        n_mswea = len(MSWEA_RE.findall(content))
        harness_valid = n_mswea == 1
        if n_mswea == 0 and len(BASH_FENCE_RE.findall(content)) == 1:
            content = BASH_FENCE_RE.sub(lambda m: "```mswea_bash_command\n" + m.group(1) + "\n```", content, count=1)
            fence_fixed = True
        r["content"] = FOREIGN_FENCE_RE.sub("```bash\n", content)
    elif harness == "bash":
        harness_valid = bool(r.get("tool_calls"))
    out = reply_to_rollout(r, kind)
    out["harness_valid"] = harness_valid and not dsml_fixed
    out["fence_fixed"] = fence_fixed
    out["dsml_fixed"] = dsml_fixed
    out["raw_content"] = (reply.get("content") or "")[:3000]
    out["fallback_parse"] = False
    if kind == "terminus_json" and not out["y"]:
        fb = terminus_fallback(content)
        if fb:
            out["y"], out["parsed"], out["fallback_parse"] = fb, True, True
    if harness == "bash" and not r.get("tool_calls"):
        out["y"] = ""          # a plain reply under the bash harness ends the episode
        out["parsed"] = False
        out["finish_reply"] = True
    return out


def first_action(row: dict, harness: str, kind: str) -> str:
    fr = row.get("first_reply") or {}
    tcs = [{"function": {"name": tc.get("name") or (tc.get("function") or {}).get("name"),
                         "arguments": tc.get("arguments") if tc.get("arguments") is not None
                         else (tc.get("function") or {}).get("arguments")}}
           for tc in (fr.get("tool_calls") or [])]
    return action_of({"content": fr.get("content") or "", "reasoning": "", "tool_calls": tcs},
                     harness, kind)["y"] or ""


def render_draft(reply: dict) -> str:
    content = reply.get("content") or ""
    if reply.get("tool_calls"):
        content = content.rstrip() + "\n" + "\n".join(render_tool_call(c) for c in reply["tool_calls"])
    return content.strip()


# ------------------------------------------------------------------ states + labels
def load_states(which: str) -> list[dict]:
    J = json.load(open(RES / "split_states" / "report.json"))
    y2 = {r["state_id"]: r for r in J["y2"]}
    sel = {s["state_id"].rsplit(":", 1)[0]: s for s in read_jsonl(RES / "split_states" / "selected.jsonl")}
    kept = {k["state_id"]: k for k in read_jsonl(RES / "outcome" / "kept.jsonl")}
    props = {p["state_id"]: p for p in read_jsonl(RES / "split_states" / "proposals.jsonl")}

    split_c = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in read_jsonl(RES / "split_states" / "continuations.jsonl"):
        split_c[r["kept_state_id"]][r["arm"]].append(r)
    out_c = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in read_jsonl(RES / "outcome" / "continuations.jsonl"):
        out_c[r["kept_state_id"]][r["arm"]].append(r)

    def ok(r):
        return r.get("status") == "ok" and r.get("outcome") in ("solved", "failed")

    states: dict[str, dict] = {}

    def base(sid: str, cls: str) -> dict:
        rid, ti, _ = sid.split(":")
        path = STATE_DIR / f"{rid}_{ti}_frontier.json"
        stj = json.loads(path.read_text())
        return {"state_id": sid, "path": str(path), "class": cls, "harness": stj["harness"],
                "kind": stj["action_kind"], "depth": stj["depth"], "source": stj["source"],
                "origin": stj["orig_outcome"], "prefix_chars": stj["prefix_chars"],
                "king_action": stj.get("king_action") or "", "labelled": [], "teacher_unlabelled": []}

    for s in J["states"]:
        sid = s["state_id"]
        if s["N"] < 3:
            continue
        if s["split"]:
            cls = "split"
        elif s["ceiling"] and sid in y2 and any(p["solved"] is not None for p in y2[sid]["proposals"]):
            cls = "ceiling"
        else:
            continue
        states[sid] = base(sid, cls)
        states[sid]["teacher_unlabelled"] += list(s.get("api_first_actions") or [])
    if which == "all":
        for sid, arms in out_c.items():
            if sid in states:
                continue
            if any(ok(r) for r in arms.get("T", [])) and any(ok(r) for r in arms.get("F1", [])):
                states[sid] = base(sid, "outcome")

    for sid, stt in states.items():
        h, kind = stt["harness"], stt["kind"]
        for r in split_c.get(sid, {}).get("T", []):
            if ok(r):
                y = first_action(r, h, kind)
                if y:
                    stt["labelled"].append({"y": y, "outcome": r["outcome"], "who": "teacher", "src": "split:T"})
        if sid in props:
            for pr in props[sid]["proposals"]:
                runs = [r for r in split_c.get(sid, {}).get(pr["arm"], []) if ok(r)]
                if runs:
                    oc = "solved" if any(r["outcome"] == "solved" for r in runs) else "failed"
                    stt["labelled"].append({"y": pr["y"], "outcome": oc, "who": "frontier_proposal",
                                            "src": f"split:{pr['arm']}"})
        # Yesterday's rows: the split probe seeded its arm T (and F1 where it existed) with
        # them, so for split/ceiling states a (norm, who, outcome) already present is the
        # same continuation and is skipped; every split-probe row is kept as is.
        have = {(norm_action(l["y"], kind), l["who"], l["outcome"]) for l in stt["labelled"]}

        def add(lab):
            key = (norm_action(lab["y"], kind), lab["who"], lab["outcome"])
            if key in have:
                return
            have.add(key)
            stt["labelled"].append(lab)

        for r in out_c.get(sid, {}).get("T", []):
            if ok(r):
                y = first_action(r, h, kind)
                if y:
                    add({"y": y, "outcome": r["outcome"], "who": "teacher", "src": "outcome:T"})
        k = kept.get(sid)
        for r in out_c.get(sid, {}).get("F1", []):
            if ok(r) and k and (k.get("frontier_greedy") or {}).get("y"):
                add({"y": k["frontier_greedy"]["y"], "outcome": r["outcome"], "who": "frontier_proposal", "src": "outcome:F1"})
        for r in out_c.get(sid, {}).get("F", []):
            if ok(r):
                y = first_action(r, h, kind)
                if y:
                    add({"y": y, "outcome": r["outcome"], "who": "frontier_own", "src": "outcome:F"})
        if k:
            stt["teacher_unlabelled"] += [q["y"] for q in (k.get("teacher") or []) if q and q.get("y")]
        seen_u: set[str] = set()
        stt["teacher_unlabelled"] = [y for y in stt["teacher_unlabelled"]
                                     if not (norm_action(y, kind) in seen_u or seen_u.add(norm_action(y, kind)))]
        stt["teacher_actions"] = [l["y"] for l in stt["labelled"] if l["who"] == "teacher"]
        stt["n_solved"] = sum(1 for l in stt["labelled"] if l["outcome"] == "solved")
        stt["n_failed"] = sum(1 for l in stt["labelled"] if l["outcome"] == "failed")
    order = {"split": 0, "ceiling": 1, "outcome": 2}
    return sorted(states.values(), key=lambda s: (order[s["class"]], s["state_id"]))


# ------------------------------------------------------------------ sampling
class Sampler:
    def __init__(self, engy: Engy, budget: float):
        self.engy = engy
        self.budget = budget
        self.n = 0

    async def chat(self, model: str, messages: list[dict], **kw) -> dict:
        """glm-5.3 falls back to glm-5.3-flash after the client's retries (429s
        under load); the reply stamps the model that actually answered."""
        try:
            return await self.engy.chat(model, messages, **kw)
        except RuntimeError as e:
            if model != FRONTIER:
                raise
            print(f"  ! {model} failed ({str(e)[:80]}); falling back to {FRONTIER_FALLBACK}", flush=True)
            r = await self.engy.chat(FRONTIER_FALLBACK, messages, **kw)
            r["fallback"] = True
            return r

    def row(self, stt: dict, variant: str, idx: int, reply: dict, extra: dict | None = None) -> dict:
        a = action_of(reply, stt["harness"], stt["kind"])
        u = reply.get("usage") or {}
        self.n += 1
        return {"state_id": stt["state_id"], "variant": variant, "idx": idx, "model": reply.get("model"),
                "fallback": bool(reply.get("fallback")), "z": a["z"], "y": a["y"], "parsed": a["parsed"],
                "harness_valid": a["harness_valid"], "fence_fixed": a["fence_fixed"], "dsml_fixed": a["dsml_fixed"],
                "raw_content": a["raw_content"],
                "fallback_parse": a["fallback_parse"], "finish": reply.get("finish"),
                "reasoning_chars": len(reply.get("reasoning") or ""), "content_chars": len(reply.get("content") or ""),
                "prompt_tokens": int(u.get("prompt_tokens") or 0), "completion_tokens": int(u.get("completion_tokens") or 0),
                "cached_tokens": int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0),
                "reasoning_tokens": int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0),
                "cost_usd": float(reply.get("cost_usd") or 0), **(extra or {})}


def candidates_block(stt: dict) -> str:
    seen, cands = set(), []
    for y in stt["teacher_actions"] + ([stt["king_action"]] if stt["king_action"] else []):
        n = norm_action(y, stt["kind"])
        if n in seen or not y.strip():
            continue
        seen.add(n)
        cands.append(y.strip())
    return "\n\n".join(f"[{chr(65 + i)}]\n{clip(c, 2500)}" for i, c in enumerate(cands))


async def sample_state(smp: Sampler, stt: dict, variants: list[str], done: set) -> list[dict]:
    stj = json.loads(Path(stt["path"]).read_text())
    msgs = stj["messages"]
    tools = {"tools": stj["tools"]} if stt["harness"] == "bash" and stj.get("tools") else {}
    rows: list[dict] = []
    tasks = []

    async def v1(i):
        r = await smp.chat(FRONTIER, msgs, temperature=0.8, max_tokens=MAX_TOKENS, **tools)
        return smp.row(stt, "v1", i, r)

    async def v2(i):
        m2 = msgs + [{"role": "user", "content": CRITIQUE_INSTR.format(cands=candidates_block(stt))}]
        r = await smp.chat(FRONTIER, m2, temperature=0.8, max_tokens=CRITIQUE_MAX_TOKENS, **tools)
        return smp.row(stt, "v2", i, r)

    async def v3(i):
        r1 = await smp.chat(FRONTIER, msgs, temperature=0.8, max_tokens=MAX_TOKENS, **tools)
        d = smp.row(stt, "v3draft", i, r1)
        m2 = msgs + [{"role": "user", "content": SELF_CRITIQUE_INSTR.format(draft=clip(render_draft(r1), 6000))}]
        r2 = await smp.chat(FRONTIER, m2, temperature=0.8, max_tokens=CRITIQUE_MAX_TOKENS, **tools)
        f = smp.row(stt, "v3", i, r2, {"draft_y": d["y"], "draft_cost_usd": d["cost_usd"],
                                       "draft_prompt_tokens": d["prompt_tokens"],
                                       "draft_completion_tokens": d["completion_tokens"],
                                       "draft_cached_tokens": d["cached_tokens"]})
        return [d, f]

    async def v4(i):
        r = await smp.chat(CROSS, msgs, temperature=0.8, max_tokens=MAX_TOKENS, **tools, **CROSS_EXTRA)
        return smp.row(stt, "v4", i, r)

    fn = {"v1": v1, "v2": v2, "v3": v3, "v4": v4}
    for v in variants:
        for i in range(K[v]):
            if (stt["state_id"], v, i) in done:
                continue
            tasks.append(fn[v](i))
    res = await asyncio.gather(*tasks, return_exceptions=True)
    for r in res:
        if isinstance(r, Exception):
            print(f"  ! sample error {stt['state_id'][:12]}: {repr(r)[:160]}", flush=True)
        elif isinstance(r, list):
            rows.extend(r)
        else:
            rows.append(r)
    return rows


async def run_sample(args) -> None:
    states = load_states(args.states)
    have = read_jsonl(SAMPLES)
    done = {(r["state_id"], r["variant"], r["idx"]) for r in have if r["variant"] != "v3draft"}
    spent0 = sum(r["cost_usd"] for r in have)
    variants = args.variants.split(",")
    engy = Engy(concurrency=args.concurrency, timeout=1500, retries=8)
    smp = Sampler(engy, args.budget_usd)
    sem = asyncio.Semaphore(args.parallel_states)
    print(f"{len(states)} states ({collections.Counter(s['class'] for s in states)}), variants {variants}, "
          f"already banked ${spent0:.2f} in {len(have)} rows", flush=True)

    async def one(stt):
        async with sem:
            if engy.cost_usd + spent0 > args.budget_usd:
                print(f"  budget ${args.budget_usd} reached; skipping {stt['state_id'][:12]}", flush=True)
                return
            t0 = time.time()
            rows = await sample_state(smp, stt, variants, done)
            for r in rows:
                append_jsonl(SAMPLES, r)
            by = collections.Counter(r["variant"] for r in rows)
            print(f"[{stt['class']:8s}] {stt['state_id'][:12]} {stt['harness']:18s} d={stt['depth']:2d} "
                  f"rows {dict(by)} parsed {sum(r['parsed'] for r in rows)}/{len(rows)} "
                  f"${sum(r['cost_usd'] for r in rows):.3f} in {time.time() - t0:.0f}s  "
                  f"(running ${engy.cost_usd + spent0:.2f})", flush=True)

    await asyncio.gather(*[one(s) for s in states])
    print(json.dumps(engy.usage, indent=1))
    print(f"sampling spend this run ${engy.cost_usd:.2f}; banked total ${engy.cost_usd + spent0:.2f}")


# ------------------------------------------------------------------ clustering
class Judge:
    def __init__(self, engy: Engy):
        self.engy = engy
        self.cache: dict[str, dict] = {}
        for r in read_jsonl(JUDGE_CACHE):
            self.cache[r["key"]] = r
        self.n_calls = 0

    @staticmethod
    def key(sid: str, a: str, b: str, kind: str) -> str:
        na, nb = sorted([norm_action(a, kind), norm_action(b, kind)])
        return hashlib.sha256(f"{sid}\x00{na}\x00{nb}".encode()).hexdigest()[:24]

    async def same(self, stj: dict, sid: str, a: str, b: str, kind: str) -> bool | None:
        k = self.key(sid, a, b, kind)
        if k in self.cache and "same_decision" in self.cache[k]:
            return self.cache[k]["same_decision"]
        prompt = JUDGE_PROMPT.format(task=clip(stj.get("task_prompt") or "", 2500),
                                     obs=clip(last_observation(stj["messages"]), 3000, tail=True),
                                     a=clip(a, 2500), b=clip(b, 2500))
        cost, out = 0.0, None
        for cap in (4096, 12000):
            try:
                j = await self.engy.chat(JUDGE, [{"role": "user", "content": prompt}], temperature=0.0, max_tokens=cap)
            except Exception as e:  # noqa: BLE001
                out = {"error": repr(e)[:200]}
                break
            cost += float(j.get("cost_usd") or 0)
            self.n_calls += 1
            for text in (j["content"], j.get("reasoning") or ""):
                m = re.search(r"\{[^{}]*\"same_decision\"[^{}]*\}", text, re.S)
                if m:
                    try:
                        out = json.loads(m.group(0))
                        break
                    except json.JSONDecodeError:
                        pass
            if out and "same_decision" in out:
                break
        rec = {"key": k, "state_id": sid, "a": a[:400], "b": b[:400], "cost_usd": cost, **(out or {"error": "no json"})}
        self.cache[k] = rec
        append_jsonl(JUDGE_CACHE, rec)
        return rec.get("same_decision")


class Pool:
    """All actions at one state (samples of every variant, labelled actions,
    teacher API samples, king). Classes = union-find over the three tiers."""

    def __init__(self, stt: dict, items: list[dict]):
        self.stt = stt
        self.kind = stt["kind"]
        self.items = items          # {"y", "tag", ...}
        self.norm = [norm_action(it["y"], self.kind) for it in items]
        self.parent = list(range(len(items)))
        self.tier = [None] * len(items)

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, i, j):
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.parent[max(ri, rj)] = min(ri, rj)

    def tier1(self) -> None:
        n = len(self.items)
        for i in range(n):
            for j in range(i + 1, n):
                if self.norm[i] == self.norm[j]:
                    self.union(i, j)

    def tier2(self) -> None:
        n = len(self.items)
        for i in range(n):
            for j in range(i + 1, n):
                if jaccard(self.items[i]["y"], self.items[j]["y"]) >= LOOSE:
                    self.union(i, j)

    def snapshot(self) -> list[int]:
        return [self.find(i) for i in range(len(self.items))]

    def clusters(self) -> list[list[int]]:
        d = collections.defaultdict(list)
        for i in range(len(self.items)):
            d[self.find(i)].append(i)
        return sorted(d.values(), key=lambda c: (-len(c), c[0]))

    async def tier3(self, judge: Judge, stj: dict) -> int:
        """Sequential merge on class representatives: each class is judged
        against the representative of every merged group so far (largest
        first) and joins the first group the judge calls the same decision."""
        groups: list[list[int]] = []
        n_judged = 0
        for c in self.clusters():
            rep = c[0]
            merged = False
            for g in groups:
                same = await judge.same(stj, self.stt["state_id"], self.items[rep]["y"], self.items[g[0]]["y"], self.kind)
                n_judged += 1
                if same:
                    self.union(rep, g[0])
                    g.extend(c)
                    merged = True
                    break
            if not merged:
                groups.append(list(c))
        return n_judged

    def label_of(self, i: int) -> int:
        return self.find(i)


def modal(labels: list[int]) -> tuple[list[tuple[int, int]], int]:
    c = collections.Counter(labels).most_common()
    return c, len(labels)


def pairwise(ys: list[str], kind: str, labels: list[int] | None = None) -> dict:
    n = len(ys)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if not pairs:
        return {"n": n, "pairs": 0, "exact": None, "jac_ge": None, "jac_mean": None, "same_class": None}
    ex = sum(norm_action(ys[i], kind) == norm_action(ys[j], kind) for i, j in pairs)
    jg = sum(jaccard(ys[i], ys[j]) >= LOOSE for i, j in pairs)
    jm = st.mean(jaccard(ys[i], ys[j]) for i, j in pairs)
    sc = sum(labels[i] == labels[j] for i, j in pairs) if labels is not None else None
    return {"n": n, "pairs": len(pairs), "exact": ex / len(pairs), "jac_ge": jg / len(pairs), "jac_mean": jm,
            "same_class": (sc / len(pairs)) if sc is not None else None}


def outcome_of_class(pool: Pool, labels: list[int], lab: int) -> tuple[str, dict]:
    """correct / wrong / mixed / novel for one class id under the labelled actions."""
    members = [pool.items[i] for i in range(len(pool.items)) if labels[i] == lab]
    labelled = [m for m in members if m["tag"] == "labelled"]
    solved = [m for m in labelled if m["outcome"] == "solved"]
    failed = [m for m in labelled if m["outcome"] == "failed"]
    info = {"n_solved": len(solved), "n_failed": len(failed),
            "who_solved": sorted({m["who"] for m in solved}), "who_failed": sorted({m["who"] for m in failed}),
            "has_teacher": any(m["who"] == "teacher" for m in labelled) or any(m["tag"] == "teacher_api" for m in members),
            "has_king": any(m["tag"] == "king" for m in members)}
    if solved and failed:
        return "mixed", info
    if solved:
        return "correct", info
    if failed:
        return "wrong", info
    return "novel", info


async def run_analyze(args) -> None:
    states = load_states("all")
    rows = read_jsonl(SAMPLES)
    by_state = collections.defaultdict(list)
    for r in rows:
        by_state[r["state_id"]].append(r)
    states = [s for s in states if by_state.get(s["state_id"])]
    engy = Engy(concurrency=args.concurrency, timeout=600, retries=8)
    judge = Judge(engy)
    variants = ["v1", "v2", "v3", "v4", "joint"]
    per_state: list[dict] = []
    sem = asyncio.Semaphore(args.parallel_states)

    async def one(stt):
        async with sem:
            stj = json.loads(Path(stt["path"]).read_text())
            srows = by_state[stt["state_id"]]
            items: list[dict] = []
            for r in srows:
                if r["variant"] == "v3draft" or not r["parsed"] or not r["y"]:
                    continue
                items.append({"y": r["y"], "tag": "sample", "variant": r["variant"], "idx": r["idx"], "model": r["model"]})
            for lab in stt["labelled"]:
                items.append({"y": lab["y"], "tag": "labelled", **lab})
            for y in stt["teacher_unlabelled"]:
                items.append({"y": y, "tag": "teacher_api"})
            if stt["king_action"]:
                items.append({"y": stt["king_action"], "tag": "king"})
            pool = Pool(stt, items)
            pool.tier1()
            lab_by = {"strict": pool.snapshot()}
            pool.tier2()
            lab_by["loose"] = pool.snapshot()
            n12 = len(pool.clusters())
            nj = await pool.tier3(judge, stj)
            lab_by["judged"] = pool.snapshot()
            n3 = len(pool.clusters())
            out = {"state_id": stt["state_id"], "class": stt["class"], "harness": stt["harness"], "kind": stt["kind"],
                   "depth": stt["depth"], "source": stt["source"], "origin": stt["origin"],
                   "n_labelled": len(stt["labelled"]), "n_solved": stt["n_solved"], "n_failed": stt["n_failed"],
                   "pool": len(items), "classes_after_tier2": n12, "classes_after_tier3": n3, "judge_calls": nj,
                   "variants": {}, "by_tier": {}}
            for tier in ("strict", "loose", "judged"):
                labels = lab_by[tier]
                res_t: dict = {"variants": {}}
                t_idx = [i for i, it in enumerate(items) if it["tag"] == "labelled" and it["who"] == "teacher"]
                tp = pairwise([items[i]["y"] for i in t_idx], stt["kind"], [labels[i] for i in t_idx])
                tm, tn = modal([labels[i] for i in t_idx])
                res_t["teacher"] = {**tp, "n_classes": len(tm), "modal_share": (tm[0][1] / tn) if tn else None,
                                    "modal_outcome": outcome_of_class(pool, labels, tm[0][0])[0] if tn else None}
                t_all_idx = t_idx + [i for i, it in enumerate(items) if it["tag"] == "teacher_api"]
                res_t["teacher_plus_api"] = pairwise([items[i]["y"] for i in t_all_idx], stt["kind"], [labels[i] for i in t_all_idx])
                glm_modal_label = None
                for v in variants:
                    if v == "joint":
                        idx = [i for i, it in enumerate(items) if it["tag"] == "sample" and it["variant"] in ("v1", "v4")]
                    else:
                        idx = [i for i, it in enumerate(items) if it["tag"] == "sample" and it["variant"] == v]
                    n_total = sum(1 for r in srows if r["variant"] == v) if v != "joint" else \
                        sum(1 for r in srows if r["variant"] in ("v1", "v4"))
                    if not idx:
                        res_t["variants"][v] = {"n_parsed": 0, "n_total": n_total}
                        continue
                    lab_v = [labels[i] for i in idx]
                    mc, n = modal(lab_v)
                    res = {"n_parsed": n, "n_total": n_total, "n_classes": len(mc),
                           "modal_share": mc[0][1] / n, "second_share": (mc[1][1] / n) if len(mc) > 1 else 0.0,
                           "pairwise": pairwise([items[i]["y"] for i in idx], stt["kind"], lab_v),
                           "modal_y": items[next(i for i in idx if labels[i] == mc[0][0])]["y"][:300]}
                    oc, info = outcome_of_class(pool, labels, mc[0][0])
                    res["modal_outcome"], res["modal_info"] = oc, info
                    if len(mc) > 1:
                        oc2, info2 = outcome_of_class(pool, labels, mc[1][0])
                        res["second_outcome"], res["second_info"] = oc2, info2
                    else:
                        res["second_outcome"] = None
                    # any-class hit: does ANY class of this variant contain a solved action?
                    res["any_class_correct"] = any(outcome_of_class(pool, labels, l)[0] in ("correct", "mixed")
                                                   for l in set(lab_v))
                    res["vs_teacher_class"] = info["has_teacher"]
                    res["vs_king_class"] = info["has_king"]
                    res["modal_label"] = mc[0][0]
                    if tier == "judged":
                        # DIRECT (chain-free) check: a modal-class SAMPLE must itself be norm-exact or
                        # Jaccard>=0.5 to a labelled / teacher action (guards against single-linkage chaining
                        # through intermediate samples in a ~28-action pool).
                        mem = [items[i]["y"] for i in idx if labels[i] == mc[0][0]]

                        def direct(y2):
                            return any(norm_action(m_, stt["kind"]) == norm_action(y2, stt["kind"]) or jaccard(m_, y2) >= LOOSE
                                       for m_ in mem)
                        d_solved = any(direct(l["y"]) for l in stt["labelled"] if l["outcome"] == "solved")
                        d_failed = any(direct(l["y"]) for l in stt["labelled"] if l["outcome"] == "failed")
                        res["direct_outcome"] = ("mixed" if d_solved and d_failed else "correct" if d_solved
                                                 else "wrong" if d_failed else "novel")
                        res["direct_teacher"] = any(direct(y2) for y2 in stt["teacher_actions"] + stt["teacher_unlabelled"])
                    if v == "v1":
                        glm_modal_label = mc[0][0]
                    res["agrees_with_v1_modal"] = (glm_modal_label is not None and mc[0][0] == glm_modal_label) if v != "v1" else None
                    res_t["variants"][v] = res
                out["by_tier"][tier] = res_t
            out["variants"] = out["by_tier"]["judged"]["variants"]
            out["teacher"] = out["by_tier"]["judged"]["teacher"]
            out["teacher_plus_api"] = out["by_tier"]["judged"]["teacher_plus_api"]
            # costs per variant
            cost = {}
            for v in ("v1", "v2", "v3", "v4"):
                vr = [r for r in srows if r["variant"] == v]
                if not vr:
                    continue
                c = {"calls": len(vr), "cost_usd": sum(r["cost_usd"] for r in vr),
                     "prompt_tokens": sum(r["prompt_tokens"] for r in vr),
                     "completion_tokens": sum(r["completion_tokens"] for r in vr),
                     "cached_tokens": sum(r["cached_tokens"] for r in vr),
                     "reasoning_tokens": sum(r["reasoning_tokens"] for r in vr),
                     "fallback": sum(1 for r in vr if r["fallback"]),
                     "finish_length": sum(1 for r in vr if r["finish"] == "length"),
                     "models": dict(collections.Counter(r["model"] for r in vr))}
                if v == "v3":
                    c["calls"] += len(vr)
                    c["cost_usd"] += sum(r.get("draft_cost_usd", 0) for r in vr)
                    c["prompt_tokens"] += sum(r.get("draft_prompt_tokens", 0) for r in vr)
                    c["completion_tokens"] += sum(r.get("draft_completion_tokens", 0) for r in vr)
                    c["cached_tokens"] += sum(r.get("draft_cached_tokens", 0) for r in vr)
                cost[v] = c
            out["cost"] = cost
            out["judge_cost_usd"] = sum(judge.cache[k]["cost_usd"] for k in judge.cache
                                        if judge.cache[k]["state_id"] == stt["state_id"])
            per_state.append(out)
            v1 = out["variants"].get("v1", {})
            print(f"[{stt['class']:8s}] {stt['state_id'][:12]} {stt['harness']:18s} pool {len(items):2d} classes "
                  f"{n12:2d}->{n3:2d} judged {nj:3d}  v1 conf {fmt(v1.get('modal_share'))} {v1.get('modal_outcome', '-'):8s} "
                  f"teacher conf {fmt(out['teacher']['modal_share'])}  (judge $ {engy.cost_usd:.3f})", flush=True)

    await asyncio.gather(*[one(s) for s in states])
    per_state.sort(key=lambda s: ({"split": 0, "ceiling": 1, "outcome": 2}[s["class"]], s["state_id"]))
    write_report(per_state, rows, engy.cost_usd)


# ------------------------------------------------------------------ report
def mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.mean(xs) if xs else None


def write_report(per_state: list[dict], rows: list[dict], judge_spend: float) -> None:
    L: list[str] = []
    J: dict = {"per_state": per_state}
    variants = ["v1", "v2", "v3", "v4", "joint"]
    vname = {"v1": "v1 deliberated glm-5.3 k=10", "v2": "v2 critique-then-act k=3", "v3": "v3 self-critique k=3",
             "v4": "v4 deepseek-v4.1-flash k=5", "joint": "joint v1+v4 k=15"}
    sample_spend = sum(r["cost_usd"] for r in rows)
    judge_total = sum(r["cost_usd"] for r in read_jsonl(JUDGE_CACHE))
    n_states = len(per_state)
    ncls = collections.Counter(s["class"] for s in per_state)
    L.append("=" * 100)
    L.append("DELIBERATED FRONTIER ORACLE — self-agreement and outcome accuracy at ground-truth states (2026-09-21)")
    L.append("=" * 100)
    L.append(f"states {n_states}: {dict(ncls)}  (split = teacher first actions led to solved AND failed; ceiling = teacher 0/N "
             f"with frontier proposals executed; outcome = yesterday's paired T/F1 continuations)")
    L.append(f"by harness: {dict(collections.Counter(s['harness'] for s in per_state))}")
    L.append(f"labelled actions per state: mean {mean([s['n_labelled'] for s in per_state]):.1f} "
             f"(solved {sum(s['n_solved'] for s in per_state)}, failed {sum(s['n_failed'] for s in per_state)} in total)")
    L.append(f"classes per state pool (all samples + labelled + teacher API + king): after tier 1+2 mean "
             f"{mean([s['classes_after_tier2'] for s in per_state]):.1f} -> after tier 3 judge mean "
             f"{mean([s['classes_after_tier3'] for s in per_state]):.1f}; judge calls {sum(s['judge_calls'] for s in per_state)}")
    L.append("")
    L.append("Terms: class = equivalence class of actions (tier 1 norm-exact, tier 2 token-Jaccard>=0.5 single linkage, tier 3 "
             "glm-5.3-flash T=0 same-decision judge on class representatives); modal share = fraction of a variant's parsed "
             "samples in its largest class (= confidence); single-sample = pairwise agreement between two independent draws; "
             "correct/wrong = modal class contains a solved / only failed labelled action, mixed = both, novel = no labelled member; "
             "accuracy = correct/(correct+wrong).")

    # ---- 1. self-agreement
    L.append("")
    L.append("=" * 100)
    L.append("1. SELF-AGREEMENT — single-sample pairwise vs deliberated modal share")
    L.append("=" * 100)
    L.append(f"{'variant':32s} {'states':>6s} {'parsed/k':>9s} | single-sample pairwise: {'exact':>6s} {'jac>=.5':>8s} "
             f"{'jac mean':>8s} {'sameclass':>9s} | deliberated: {'classes':>7s} {'modal share':>11s} {'>=0.5':>6s} {'>=0.7':>6s} {'2nd share':>9s}")
    agg: dict = {}
    for v in variants:
        vs = [s["variants"][v] for s in per_state if s["variants"].get(v, {}).get("n_parsed")]
        if not vs:
            continue
        pw = [x["pairwise"] for x in vs if x["pairwise"]["pairs"]]
        a = {"states": len(vs), "parsed": sum(x["n_parsed"] for x in vs), "total": sum(x["n_total"] for x in vs),
             "exact": mean([p["exact"] for p in pw]), "jac_ge": mean([p["jac_ge"] for p in pw]),
             "jac_mean": mean([p["jac_mean"] for p in pw]), "same_class": mean([p["same_class"] for p in pw]),
             "classes": mean([x["n_classes"] for x in vs]), "modal": mean([x["modal_share"] for x in vs]),
             "ge05": sum(x["modal_share"] >= 0.5 for x in vs), "ge07": sum(x["modal_share"] >= 0.7 for x in vs),
             "second": mean([x["second_share"] for x in vs])}
        agg[v] = a
        L.append(f"{vname[v]:32s} {a['states']:6d} {a['parsed']:4d}/{a['total']:<4d} | {'':24s}{fmt(a['exact']):>6s} {fmt(a['jac_ge']):>8s} "
                 f"{fmt(a['jac_mean']):>8s} {fmt(a['same_class']):>9s} | {'':13s}{fmt(a['classes'], 1):>7s} {fmt(a['modal']):>11s} "
                 f"{pct(a['ge05'], a['states']).split('=')[0].strip():>6s} {pct(a['ge07'], a['states']).split('=')[0].strip():>6s} {fmt(a['second']):>9s}")
    ts = [s["teacher"] for s in per_state if s["teacher"]["pairs"]]
    ta = {"states": len(ts), "n": sum(t["n"] for t in ts), "exact": mean([t["exact"] for t in ts]),
          "jac_ge": mean([t["jac_ge"] for t in ts]), "jac_mean": mean([t["jac_mean"] for t in ts]),
          "same_class": mean([t["same_class"] for t in ts]), "classes": mean([t["n_classes"] for t in ts]),
          "modal": mean([t["modal_share"] for t in ts]), "ge05": sum(t["modal_share"] >= 0.5 for t in ts),
          "ge07": sum(t["modal_share"] >= 0.7 for t in ts)}
    agg["teacher"] = ta
    L.append(f"{'teacher (its N labelled first actions)':32s} {ta['states']:6d} {ta['n']:4d}/{ta['n']:<4d} | {'':24s}{fmt(ta['exact']):>6s} {fmt(ta['jac_ge']):>8s} "
             f"{fmt(ta['jac_mean']):>8s} {fmt(ta['same_class']):>9s} | {'':13s}{fmt(ta['classes'], 1):>7s} {fmt(ta['modal']):>11s} "
             f"{pct(ta['ge05'], ta['states']).split('=')[0].strip():>6s} {pct(ta['ge07'], ta['states']).split('=')[0].strip():>6s} {'':>9s}")
    tpa = [s["teacher_plus_api"] for s in per_state if s["teacher_plus_api"]["pairs"]]
    L.append(f"{'teacher + 3 API samples (unlabelled)':32s} {len(tpa):6d} {sum(t['n'] for t in tpa):4d}/{sum(t['n'] for t in tpa):<4d} | {'':24s}"
             f"{fmt(mean([t['exact'] for t in tpa])):>6s} {fmt(mean([t['jac_ge'] for t in tpa])):>8s} "
             f"{fmt(mean([t['jac_mean'] for t in tpa])):>8s} {fmt(mean([t['same_class'] for t in tpa])):>9s} |")
    L.append("  NOTE the teacher's N is 3–5 per state (modal share of 3 draws is coarse: 1/3, 2/3, 1); the frontier's k=10 modal share "
             "has finer support. 'sameclass' is the pairwise rate of landing in the same 3-tier class = the deliberated notion of agreement.")
    L.append("  Prior single-frontier-sample numbers for reference: self-agreement 0.27 exact / 0.69 Jaccard; frontier-vs-teacher 0.25.")
    # by class / harness for v1
    L.append("")
    L.append("-- v1 modal share by state class and harness --")
    for keyf, name in ((lambda s: s["class"], "class"), (lambda s: s["harness"], "harness")):
        groups = collections.defaultdict(list)
        for s in per_state:
            if s["variants"].get("v1", {}).get("n_parsed"):
                groups[keyf(s)].append(s["variants"]["v1"])
        for g, vs in sorted(groups.items()):
            L.append(f"  {name:8s} {g:20s} n={len(vs):2d} modal share {fmt(mean([x['modal_share'] for x in vs]))} "
                     f">=0.5 {sum(x['modal_share'] >= 0.5 for x in vs)}  classes {fmt(mean([x['n_classes'] for x in vs]), 1)}  "
                     f"pairwise sameclass {fmt(mean([x['pairwise']['same_class'] for x in vs]))}")

    # ---- 2. outcome accuracy
    L.append("")
    L.append("=" * 100)
    L.append("2. OUTCOME ACCURACY — does the modal class match a SOLVED first action?")
    L.append("=" * 100)
    L.append("split states: labelled = teacher first actions (solved / failed continuations); ceiling states: labelled = teacher's "
             "failed first actions + frontier proposals executed under F1 (solved / failed); outcome states: yesterday's T and F1 "
             "(+ F = frontier ran the rest itself) first actions.")
    L.append(f"{'variant':32s} {'set':10s} {'n':>3s} {'correct':>7s} {'wrong':>5s} {'mixed':>5s} {'novel':>5s} | {'accuracy':>22s} | "
             f"{'conf>=.5: n':>11s} {'acc':>22s} | {'2nd-modal correct':>17s} | {'any class correct':>17s}")
    acc_tab: dict = {}
    for v in variants:
        for setname in ("split", "ceiling", "primary18", "outcome", "pooled"):
            ss = [s for s in per_state if (setname == "pooled" or s["class"] == setname
                                           or (setname == "primary18" and s["class"] in ("split", "ceiling")))
                  and s["variants"].get(v, {}).get("n_parsed")]
            if not ss:
                continue
            oc = collections.Counter(s["variants"][v]["modal_outcome"] for s in ss)
            conf = [s for s in ss if s["variants"][v]["modal_share"] >= 0.5]
            occ = collections.Counter(s["variants"][v]["modal_outcome"] for s in conf)
            sec = collections.Counter(s["variants"][v].get("second_outcome") for s in ss)
            anyc = sum(1 for s in ss if s["variants"][v]["any_class_correct"])
            c, w = oc["correct"], oc["wrong"]
            cc, cw = occ["correct"], occ["wrong"]
            acc_tab[(v, setname)] = {"n": len(ss), "correct": c, "wrong": w, "mixed": oc["mixed"], "novel": oc["novel"],
                                     "accuracy": (c / (c + w)) if c + w else None, "n_conf": len(conf), "conf_correct": cc,
                                     "conf_wrong": cw, "conf_accuracy": (cc / (cc + cw)) if cc + cw else None,
                                     "second_correct": sec["correct"], "second_wrong": sec["wrong"], "any_class_correct": anyc}
            L.append(f"{vname[v] if setname == 'split' or v == 'joint' and setname == 'split' else '':32s} {setname:10s} {len(ss):3d} {c:7d} {w:5d} {oc['mixed']:5d} {oc['novel']:5d} | "
                     f"{pct(c, c + w):>22s} | {len(conf):11d} {pct(cc, cc + cw):>22s} | {pct(sec['correct'], sec['correct'] + sec['wrong']):>17s} | {pct(anyc, len(ss)):>17s}")
    L.append("  lenient accuracy (modal class contains ANY solved action; mixed counted as correct) and granularity sensitivity:")
    L.append(f"  {'variant':32s} {'tier':16s} {'n':>3s} {'modal share':>11s} {'>=0.5':>6s} | {'corr':>4s} {'wrong':>5s} {'mixed':>5s} {'novel':>5s} | "
             f"{'accuracy':>22s} | {'lenient acc':>22s} | {'novel rate':>22s}")
    tier_tab: dict = {}
    for v in variants:
        for tier in ("strict", "loose", "judged", "judged/primary18"):
            t = tier.split("/")[0]
            ss = [s for s in per_state if s["by_tier"][t]["variants"].get(v, {}).get("n_parsed")
                  and (tier == t or s["class"] in ("split", "ceiling"))]
            if not ss:
                continue
            xs = [s["by_tier"][t]["variants"][v] for s in ss]
            oc = collections.Counter(x["modal_outcome"] for x in xs)
            c, w, m, nv = oc["correct"], oc["wrong"], oc["mixed"], oc["novel"]
            tier_tab[f"{v}/{tier}"] = {"n": len(ss), "modal_share": mean([x["modal_share"] for x in xs]),
                                       "ge05": sum(x["modal_share"] >= 0.5 for x in xs), "correct": c, "wrong": w,
                                       "mixed": m, "novel": nv, "accuracy": (c / (c + w)) if c + w else None,
                                       "lenient": ((c + m) / (c + m + w)) if c + m + w else None}
            L.append(f"  {vname[v] if tier == 'strict' else '':32s} {tier:16s} {len(ss):3d} {fmt(mean([x['modal_share'] for x in xs])):>11s} "
                     f"{sum(x['modal_share'] >= 0.5 for x in xs):6d} | {c:4d} {w:5d} {m:5d} {nv:5d} | {pct(c, c + w):>22s} | "
                     f"{pct(c + m, c + m + w):>22s} | {pct(nv, len(ss)):>22s}")
    for v in variants:
        for setname in ("all", "primary18"):
            ss = [s for s in per_state if s["variants"].get(v, {}).get("n_parsed")
                  and (setname == "all" or s["class"] in ("split", "ceiling"))]
            oc = collections.Counter(s["variants"][v]["direct_outcome"] for s in ss)
            c, w, m, nv = oc["correct"], oc["wrong"], oc["mixed"], oc["novel"]
            tier_tab[f"{v}/direct/{setname}"] = {"n": len(ss), "correct": c, "wrong": w, "mixed": m, "novel": nv,
                                                 "teacher_overlap": sum(1 for s in ss if s["variants"][v]["direct_teacher"])}
            L.append(f"  {vname[v] if setname == 'all' else '':32s} {'direct/' + setname:16s} {len(ss):3d} {'':>11s} {'':>6s} | {c:4d} {w:5d} {m:5d} {nv:5d} | "
                     f"{pct(c, c + w):>22s} | {pct(c + m, c + m + w):>22s} | {pct(nv, len(ss)):>22s}   teacher overlap {tier_tab[f'{v}/direct/{setname}']['teacher_overlap']}/{len(ss)}")
    L.append("  'direct' = the judged-tier modal class, but a match to a labelled/teacher action must be carried by one of the modal class's "
             "own samples (norm-exact or Jaccard>=0.5) — no chaining through the pool.")
    for tier in ("strict", "loose", "judged"):
        ts_ = [s["by_tier"][tier]["teacher"] for s in per_state if s["by_tier"][tier]["teacher"]["n"]]
        oc = collections.Counter(t["modal_outcome"] for t in ts_)
        c, w, m, nv = oc["correct"], oc["wrong"], oc["mixed"], oc["novel"]
        L.append(f"  {'teacher modal first action' if tier == 'strict' else '':32s} {tier:16s} {len(ts_):3d} {fmt(mean([t['modal_share'] for t in ts_])):>11s} "
                 f"{sum(t['modal_share'] >= 0.5 for t in ts_):6d} | {c:4d} {w:5d} {m:5d} {nv:5d} | {pct(c, c + w):>22s} | {pct(c + m, c + m + w):>22s} |")
    L.append("  'mixed' at loose/judged granularity = the class holds BOTH a solved and a failed labelled action: at that granularity the "
             "first-action decision did not determine the outcome (the split probe already found solved/failed diverge at the state "
             "under norm-exact in 6/7 splits but only 3/7 under Jaccard>=0.5).")
    J["by_tier"] = tier_tab
    # baselines: teacher modal class, king, random labelled
    L.append("")
    L.append("-- baselines on the same states --")
    for setname in ("split", "ceiling", "primary18", "outcome", "pooled"):
        ss = [s for s in per_state if setname == "pooled" or s["class"] == setname
              or (setname == "primary18" and s["class"] in ("split", "ceiling"))]
        if not ss:
            continue
        toc = collections.Counter(s["teacher"]["modal_outcome"] for s in ss if s["teacher"]["n"])
        # a random labelled action's solve rate = base rate a blind pick would get
        base = [s["n_solved"] / (s["n_solved"] + s["n_failed"]) for s in ss if s["n_solved"] + s["n_failed"]]
        L.append(f"  {setname:10s} teacher MODAL first action: correct {toc['correct']} wrong {toc['wrong']} mixed {toc['mixed']} "
                 f"-> accuracy {pct(toc['correct'], toc['correct'] + toc['wrong'])}; blind pick of a labelled action solves "
                 f"{fmt(mean(base))} on average")
    J["accuracy"] = {f"{v}/{s}": d for (v, s), d in acc_tab.items()}
    J["self_agreement"] = agg

    # ---- 3. variants vs v1 and frontier vs teacher
    L.append("")
    L.append("=" * 100)
    L.append("3. VARIANT AGREEMENT WITH v1 MODAL, FRONTIER-vs-TEACHER OVERLAP, KING")
    L.append("=" * 100)
    L.append(f"{'variant':32s} {'n':>3s} {'modal = v1 modal':>18s} | {'modal class contains a TEACHER action':>38s} "
             f"{'..of which solved':>17s} {'..failed':>8s} {'..mixed':>7s} | {'contains KING action':>20s}")
    ovl: dict = {}
    for v in variants:
        ss = [s for s in per_state if s["variants"].get(v, {}).get("n_parsed")]
        if not ss:
            continue
        agree_v1 = [s for s in ss if s["variants"][v].get("agrees_with_v1_modal") is not None]
        na = sum(1 for s in agree_v1 if s["variants"][v]["agrees_with_v1_modal"])
        t_in = [s for s in ss if s["variants"][v]["vs_teacher_class"]]
        t_oc = collections.Counter(s["variants"][v]["modal_outcome"] for s in t_in)
        k_in = sum(1 for s in ss if s["variants"][v]["vs_king_class"])
        ovl[v] = {"n": len(ss), "agree_v1": na, "agree_v1_n": len(agree_v1), "teacher_overlap": len(t_in),
                  "teacher_overlap_solved": t_oc["correct"], "teacher_overlap_failed": t_oc["wrong"],
                  "teacher_overlap_mixed": t_oc["mixed"], "king_overlap": k_in}
        L.append(f"{vname[v]:32s} {len(ss):3d} {(pct(na, len(agree_v1)) if agree_v1 else '  (is v1)'):>18s} | {pct(len(t_in), len(ss)):>38s} "
                 f"{t_oc['correct']:17d} {t_oc['wrong']:8d} {t_oc['mixed']:7d} | {pct(k_in, len(ss)):>20s}")
    L.append("  (teacher action = any of the teacher's labelled first actions or its 3 unlabelled API samples at the state; "
             "'solved' = that class holds a solved labelled action)")
    L.append("  frontier-vs-teacher overlap by granularity (modal class contains a teacher action):")
    for v in variants:
        parts = []
        for tier in ("strict", "loose", "judged"):
            ss = [s for s in per_state if s["by_tier"][tier]["variants"].get(v, {}).get("n_parsed")]
            k = sum(1 for s in ss if s["by_tier"][tier]["variants"][v]["vs_teacher_class"])
            k18 = sum(1 for s in ss if s["class"] in ("split", "ceiling") and s["by_tier"][tier]["variants"][v]["vs_teacher_class"])
            n18 = sum(1 for s in ss if s["class"] in ("split", "ceiling"))
            parts.append(f"{tier} {k}/{len(ss)} ({k18}/{n18} primary18)")
            ovl.setdefault(v, {})[f"teacher_overlap_{tier}"] = [k, len(ss)]
        L.append(f"    {vname[v]:32s} " + " | ".join(parts))
    J["overlap"] = ovl

    # ---- 4. cost
    L.append("")
    L.append("=" * 100)
    L.append("4. COST — as run (Engy realised charge) and projections")
    L.append("=" * 100)
    L.append(f"{'variant':32s} {'states':>6s} {'calls/state':>11s} {'prompt tok/state':>16s} {'cached share':>12s} {'compl tok/state':>15s} "
             f"{'reasoning share':>15s} {'$/state realised':>16s} {'$/state @list':>13s} {'@flash list':>11s} {'fallback':>8s} {'len-cap':>7s}")
    cost_tab: dict = {}
    for v in ("v1", "v2", "v3", "v4"):
        cs = [s["cost"][v] for s in per_state if s.get("cost", {}).get(v)]
        if not cs:
            continue
        n = len(cs)
        pt = sum(c["prompt_tokens"] for c in cs) / n
        ct = sum(c["completion_tokens"] for c in cs) / n
        cached = sum(c["cached_tokens"] for c in cs) / max(sum(c["prompt_tokens"] for c in cs), 1)
        rs = sum(c["reasoning_tokens"] for c in cs) / max(sum(c["completion_tokens"] for c in cs), 1)
        usd = sum(c["cost_usd"] for c in cs) / n
        model = FRONTIER if v != "v4" else CROSS
        pi, po = PRICES[model]
        fi, fo = PRICES["glm-5.3-flash"]
        at_list = (pt * pi + ct * po) / 1e6
        at_flash = (pt * fi + ct * fo) / 1e6
        calls = sum(c["calls"] for c in cs) / n
        cost_tab[v] = {"states": n, "calls_per_state": calls, "prompt_tokens_per_state": pt, "completion_tokens_per_state": ct,
                       "cached_share": cached, "reasoning_share": rs, "usd_per_state": usd, "usd_per_state_list": at_list,
                       "usd_per_state_flash_list": at_flash, "fallback_calls": sum(c["fallback"] for c in cs),
                       "finish_length": sum(c["finish_length"] for c in cs), "models": dict(sum((collections.Counter(c["models"]) for c in cs), collections.Counter()))}
        L.append(f"{vname[v]:32s} {n:6d} {calls:11.1f} {pt:16,.0f} {cached:12.2f} {ct:15,.0f} {rs:15.2f} {usd:16.4f} {at_list:13.4f} {at_flash:11.4f} "
                 f"{cost_tab[v]['fallback_calls']:8d} {cost_tab[v]['finish_length']:7d}")
    L.append("  '$/state @list' = tokens x list price with no cache discount; 'realised' = Engy's charge (cache discount + actual model mix); "
             "'@flash list' = same tokens priced at glm-5.3-flash ($0.10/M in, $0.36/M out).")
    L.append("")
    L.append("-- projections (realised $/state; 1,000-turn duel = 2 sides share the oracle so ONE oracle per turn; D = 635k turns) --")
    L.append(f"{'variant':32s} {'duel all turns':>14s} {'duel 40% decision':>17s} {'D all':>12s} {'D 40%':>12s} | {'@flash: duel all':>16s} {'D all':>12s} {'D 40%':>12s}")
    proj: dict = {}
    for v, c in cost_tab.items():
        usd, fl = c["usd_per_state"], c["usd_per_state_flash_list"]
        proj[v] = {"duel_all": usd * 1000, "duel_40": usd * 400, "D_all": usd * 635_000, "D_40": usd * 254_000,
                   "flash_duel_all": fl * 1000, "flash_D_all": fl * 635_000, "flash_D_40": fl * 254_000}
        p = proj[v]
        L.append(f"{vname[v]:32s} {p['duel_all']:14,.0f} {p['duel_40']:17,.0f} {p['D_all']:12,.0f} {p['D_40']:12,.0f} | "
                 f"{p['flash_duel_all']:16,.0f} {p['flash_D_all']:12,.0f} {p['flash_D_40']:12,.0f}")
    tj = sum(s["judge_calls"] for s in per_state) / max(n_states, 1)
    L.append(f"  + clustering judge (glm-5.3-flash): {tj:.1f} calls/state here (pool of ~{mean([s['pool'] for s in per_state]):.0f} actions incl. labels), "
             f"${judge_total / max(n_states, 1):.4f}/state; a production oracle clusters only its own k samples.")
    L.append(f"SPENT: sampling ${sample_spend:.2f} + judge ${judge_total:.2f} = ${sample_spend + judge_total:.2f} (this analyze run's judge share ${judge_spend:.2f})")
    J["cost"] = {"table": cost_tab, "projection": proj, "sampling_usd": sample_spend, "judge_usd": judge_total,
                 "total_usd": sample_spend + judge_total}

    # ---- 5. per state
    L.append("")
    L.append("=" * 100)
    L.append("5. PER STATE (conf = modal share; outcome of modal class; 2nd = second-modal outcome)")
    L.append("=" * 100)
    for s in per_state:
        v = s["variants"]
        t = s["teacher"]
        parts = []
        for k in variants:
            x = v.get(k)
            if x and x.get("n_parsed"):
                parts.append(f"{k} {x['modal_share']:.2f}/{x['n_classes']}c {x['modal_outcome'][:4]}"
                             + (f"|2nd {x['second_outcome'][:4]}" if x.get("second_outcome") else ""))
            else:
                parts.append(f"{k} —")
        L.append(f"  {s['class']:8s} {s['state_id'][:12]} {s['harness']:18s} d={s['depth']:2d} labels s{s['n_solved']}/f{s['n_failed']} "
                 f"teacher conf {fmt(t['modal_share'])} {str(t['modal_outcome'])[:4]} | " + "  ".join(parts))
        if v.get("v1", {}).get("modal_y"):
            L.append(f"      v1 modal: {v['v1']['modal_y'][:150]!r}")

    # ---- 6. verdict (numbers pulled from the tables above)
    a1, at = agg.get("v1", {}), agg.get("teacher", {})
    p18 = acc_tab.get(("v1", "primary18"), {})
    pout = acc_tab.get(("v1", "outcome"), {})
    L.append("")
    L.append("=" * 100)
    L.append("6. VERDICT")
    L.append("=" * 100)
    L.append(f"1. Deliberation buys CONFIDENCE, not agreement between draws: v1 pairwise same-class {fmt(a1.get('same_class'))} "
             f"(norm-exact {fmt(a1.get('exact'))}, Jaccard>=.5 {fmt(a1.get('jac_ge'))}) vs the teacher's own {fmt(at.get('same_class'))} "
             f"(exact {fmt(at.get('exact'))}); the k=10 modal class holds {fmt(a1.get('modal'))} of samples, >=0.5 at {a1.get('ge05')}/{a1.get('states')} states.")
    L.append(f"2. Confidence is not accuracy: on the 18 ground-truth split+ceiling states the v1 modal class is correct {p18.get('correct')} / "
             f"wrong {p18.get('wrong')} / mixed {p18.get('mixed')} / novel {p18.get('novel')} (accuracy {fmt(p18.get('accuracy'))}); "
             f"confident (>=0.5) states are no better ({p18.get('conf_correct')}/{p18.get('n_conf')} correct).")
    L.append(f"3. The deliberated frontier converges on the TEACHER's decision class ({ovl.get('v1', {}).get('teacher_overlap')}/30 states; "
             f"direct, chain-free {tier_tab.get('v1/direct/all', {}).get('teacher_overlap')}/30) — at {acc_tab.get(('v1', 'ceiling'), {}).get('wrong')}/11 ceiling states "
             f"that class is one the teacher already failed with, at the other {acc_tab.get(('v1', 'ceiling'), {}).get('mixed')} it is 'mixed' (the judge "
             f"equates the solved F1 proposal with the teacher's failed action); no ceiling state has a modal class that is cleanly a solved action.")
    L.append(f"4. The easy 'outcome' states ({pout.get('n')} states, mostly all-solved) inflate pooled accuracy to {fmt(acc_tab.get(('v1', 'pooled'), {}).get('accuracy'))}; "
             f"critique (v2) and self-critique (v3) raise modal share to {fmt(agg.get('v2', {}).get('modal'))}/{fmt(agg.get('v3', {}).get('modal'))} and "
             f"add novel actions (v2 direct novel {tier_tab.get('v2/direct/all', {}).get('novel')}/30) without lifting accuracy on the hard 18.")
    L.append("5. A per-state frontier oracle from deliberation is NOT viable as the ranking target on this evidence: it would rank miners by "
             "similarity to a confident consensus that is the teacher's own failure mode at exactly the states where the teacher fails; "
             "only environment-verified continuations separate solved from failed first actions, and even those separate only at norm-exact granularity.")
    J["verdict_numbers"] = {"v1": a1, "teacher": at, "primary18": p18, "outcome": pout}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(L) + "\n")
    json.dump(J, open(OUT / "report.json", "w"), indent=1)
    print("\n".join(L))
    print(f"\nwrote {OUT / 'report.txt'} and report.json")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("stage", nargs="?", default="all", choices=["sample", "analyze", "all"])
    ap.add_argument("--states", default="all", choices=["primary", "all"])
    ap.add_argument("--variants", default="v1,v2,v3,v4")
    ap.add_argument("--budget-usd", type=float, default=40.0)
    ap.add_argument("--concurrency", type=int, default=10)
    ap.add_argument("--parallel-states", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage in ("sample", "all"):
        asyncio.run(run_sample(args))
    if args.stage in ("analyze", "all"):
        asyncio.run(run_analyze(args))


if __name__ == "__main__":
    main()
