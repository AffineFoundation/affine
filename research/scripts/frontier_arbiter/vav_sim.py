"""Verified-action-value (VAV) rule simulation — store doc §6 candidate (a)+(d),
2026-09-21.

The rule: at a decision state x, D stores a TABLE of verified action classes
c with (solved, total) counted over the continuations that started with an
action in c (teacher continuations and forced frontier proposals alike).
V(c) = solved / total.  Teacher baseline B(x) = mean over the teacher's OWN
sampled first actions of V(class).  Ranking term for a miner action y =
V(class(y)) − B(x); an action matching no stored class scores 0 (prior).
Class assignment cascade: norm_action exact → token-Jaccard >= 0.5 → a
glm-5.3-flash (T0) EQUIVALENCE judge ("same decision as this stored action?",
blind pairwise against every stored class; verdicts stored).

Data (all local): the split-states probe of 2026-09-21
(research/results/frontier_arbiter/split_states/{continuations,proposals}.jsonl)
+ the state prefixes under /tmp/fa_split/arms_split/T/states/*.json.
States simulated: the 7 SPLIT states (teacher 0 < s < N, N >= 3) and the 11
CEILING states (teacher 0/N, N >= 3, with >= 1 graded frontier F1 arm).

Miners scored per state (leave-one-out wherever the action came from the table):
  teacher held-out   each teacher continuation's first action vs the table WITHOUT itself, mean over N
  frontier LOO       each graded frontier proposal vs the table WITHOUT itself (ceiling states)
  king               3 fresh reign-20 samples at the prefix (T0.8, 2048+768 tokens, raw /completions), mean
  generic            `ls -la` in the state's dialect
  repeat             the previous assistant action of the prefix re-issued
  filler             the king's action under a content-free thought = the king's V (only a thought gate can catch it)

    python vav_sim.py table            # build + print the tables (no API)
    python vav_sim.py king             # sample the king (free)
    python vav_sim.py judge            # equivalence + preference judges (glm-5.3-flash, T0)
    python vav_sim.py report           # score everything, write report.txt / report.json
    python vav_sim.py all

Terms (one line each):
  state x            a prefix of a teacher trajectory at which continuations were re-run (rollout_id:turn_idx).
  split / ceiling    teacher continuations 0 < s < N / s = 0 with N >= 3 (report.txt of the split-states probe).
  table              {class -> (solved, total)} over graded continuations whose FIRST action is in the class.
  class              norm_action-exact group of first actions (strict); miners join a class by the cascade.
  V(c) / B(x)        solved/total of class c / mean V over the teacher's own N first actions (with themselves).
  term               V(class(y)) − B(x) on the table the miner is scored against; unmatched → 0.
  credited           the miner action joined some stored class (exact, Jaccard or judge).
  LOO                the scored action's own continuation is removed from the table (and from B) first.
  equivalence judge  glm-5.3-flash T0: "do A and B make the same decision at this point?" YES/NO, stored.
  preference judge   glm-5.3-flash T0: "which of A/B is the better next step?" asked in both orders;
                     correct only if both orders agree and pick the env-solved action; flip = orders disagree.
  sign test          two-sided exact binomial on the sign of the per-state (teacher − king) term, ties dropped.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import hashlib
import json
import math
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "outcome"))
import common  # noqa: E402
from common import (  # noqa: E402
    REPO, Engy, append_jsonl, jaccard, norm_action, read_jsonl, reply_to_rollout, write_jsonl,
)
from split_yield import first_action  # noqa: E402

SPLIT_DIR = REPO / "research" / "results" / "frontier_arbiter" / "split_states"
RESULTS = REPO / "research" / "results" / "frontier_arbiter" / "vav"
STATES_GLOB = "/tmp/fa_split/arms_split/T/states/*.json"
KING_SAMPLES = RESULTS / "king_samples.jsonl"
EQUIV = RESULTS / "equiv_judge.jsonl"
PREF = RESULTS / "pref_judge.jsonl"
COST = RESULTS / "cost.jsonl"
BUDGET_USD = 15.0

JUDGE_MODEL = "glm-5.3-flash"
JUDGE_MAX_TOKENS = 6000
NOTHINK = {"reasoning_effort": "none"}      # Engy passes it through: glm-5.3-flash answers with 0 reasoning tokens (verified 2026-09-21)      # glm-5.3-flash thinks before the one-word answer; 1500 hit the cap on 2 % (equiv) / 38 % (pref) of calls
JACCARD_MIN = 0.5
K_KING = 3
KING_TEMPERATURE = 0.8
KING_MAX_TOKENS = 2048 + 768
MIN_N = 3
TAIL_MSGS = 3
TAIL_CAP = 10_000
TASK_CAP = 2_500
ACTION_CAP = 3_000
DECISION_SHARE = 0.40
DUEL_TURNS = 1000

# reign-20 king box (ops/king-datagen/state/state.json, pod king-dg-50e3c081ab1c-0e6d) — same as hint_design.py
KING_BASE = "http://167.150.153.180:3000/v1"
KING_MODEL = "king-50e3c081ab1c"
KING_KEY = "e85415c9ddde54e95bf3aa0388e1f9fac275f460a203d747"
KING_REIGN = 20
KING_DIGEST = "50e3c081ab1c45bae37f3b1677728f06168ed38c6a3a3ac1b7711b702b8cbd12"
KING_TOK_DIR = Path("/tmp/king_tok")
KING_TOK_FILES = ("tokenizer_config.json", "tokenizer.json", "chat_template.jinja", "config.json", "generation_config.json")

MSWEA_FENCE = "```mswea_bash_command\n"
GENERIC_BASH = "```bash\nls -la\n```"
GENERIC_TOOL = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls -la"}}\n</tool_call>'
GENERIC_TERMINUS = json.dumps({
    "analysis": "Checking the current state of the working directory before continuing.",
    "plan": "List the files in the current directory.",
    "commands": [{"keystrokes": "ls -la\n", "duration": 1.0}]}, indent=2)
XML_FUNC_RE = re.compile(r"<function=([^>\s]+)>(.*?)</function>", re.S)
XML_PARAM_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)


# ------------------------------------------------------------------ cost ledger
def log_cost(stage: str, engy: Engy, note: str = "", run: str | None = None) -> None:
    append_jsonl(COST, {"at": time.time(), "stage": stage, "run": run or str(id(engy)), "cost_usd": engy.cost_usd,
                        "usage": engy.usage, "note": note})
    tot = total_cost()
    print(f"  [$] {stage}: this run ${engy.cost_usd:.3f} | total so far ${tot:.3f} {note}", flush=True)
    if tot > BUDGET_USD:
        raise SystemExit(f"budget exceeded: ${tot:.2f} > ${BUDGET_USD}")


def total_cost() -> float:
    """Each (stage, run) logs a running total; sum the per-run maxima."""
    runs: dict[tuple, float] = collections.defaultdict(float)
    for r in read_jsonl(COST):
        runs[(r["stage"], r.get("run"))] = max(runs[(r["stage"], r.get("run"))], r["cost_usd"])
    return sum(runs.values())


# ------------------------------------------------------------------ actions
def canon_tool_call(y: str) -> str:
    """One <tool_call> block in the JSON form norm_action understands. The
    king (Qwen3.6 template) emits the XML form <function=..><parameter=..>;
    teacher/frontier actions were rendered from structured tool_calls as JSON."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", y, re.S)
    body = m.group(1) if m else y.strip()
    try:
        d = json.loads(body)
        if isinstance(d, dict) and "name" in d:
            return y
    except ValueError:
        pass
    fm = XML_FUNC_RE.search(body)
    if not fm:
        return y
    args = {k: v for k, v in XML_PARAM_RE.findall(fm.group(2))}
    return "<tool_call>\n" + json.dumps({"name": fm.group(1), "arguments": args}, ensure_ascii=False) + "\n</tool_call>"


def canon(y: str, kind: str) -> str:
    return canon_tool_call(y) if kind == "tool_call" else y


def action_body(y: str, kind: str) -> str:
    """The decision-bearing text of an action, without the dialect envelope:
    bash → the command; tool_call → tool name + argument values; terminus →
    the keystrokes. Token-Jaccard on the canonical JSON envelope is inflated
    by boilerplate tokens ('arguments', 'command', 'keystrokes', 'duration'…):
    `ls -la` reached >= .5 against unrelated tool_call / terminus actions."""
    if kind == "tool_call":
        m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", canon_tool_call(y), re.S)
        try:
            d = json.loads(m.group(1) if m else y)
            args = d.get("arguments") or {}
            vals = [str(v) for v in (args.values() if isinstance(args, dict) else [args])]
            return " ".join([str(d.get("name") or "")] + vals)
        except (ValueError, AttributeError):
            return y
    if kind == "terminus_json":
        try:
            d = json.loads(y)
            ks = [str(c.get("keystrokes", "")) for c in d.get("commands") or [] if isinstance(c, dict)]
            return " ".join(ks) + (" TASK_COMPLETE" if d.get("task_complete") else "")
        except (ValueError, AttributeError):
            return y
    return common.strip_fence(y)


def body_jaccard(a: str, b: str, kind: str) -> float:
    return jaccard(action_body(a, kind), action_body(b, kind))


def parse_king_raw(raw: str, kind: str) -> dict:
    """(z, y) of a raw king completion (started inside <think>) in the dialect."""
    closed = common.THINK_CLOSE in raw
    if not closed:
        return {"z": raw, "y": "", "parsed": False, "think_closed": False, "fence": None}
    reasoning, _, content = raw.partition(common.THINK_CLOSE)
    fence = None
    if kind == "bash":
        fence = "mswea" if MSWEA_FENCE in content else ("bash" if "```bash" in content else None)
        content = content.replace(MSWEA_FENCE, "```bash\n")
    r = reply_to_rollout({"content": content, "reasoning": reasoning, "tool_calls": []}, kind)
    y = canon(r["y"], kind) if r["y"] else ""
    return {"z": r["z"], "y": y, "parsed": bool(y), "think_closed": True, "fence": fence,
            "reasoning_chars": len(reasoning), "content_chars": len(content)}


def repeat_last(messages: list[dict], kind: str) -> str | None:
    last = [m for m in messages if m["role"] == "assistant"]
    if not last:
        return None
    m = last[-1]
    if kind == "tool_call" and m.get("tool_calls"):
        tc = m["tool_calls"][0]
        fn = tc.get("function") or tc
        return common.render_tool_call({"function": {"name": fn.get("name"), "arguments": fn.get("arguments")}})
    content = (m.get("content") or "").replace(MSWEA_FENCE, "```bash\n")
    _, y = common.dialects.split_action(content, kind)
    return canon(y, kind) if y else None


def generic_action(kind: str) -> str:
    return {"bash": GENERIC_BASH, "tool_call": GENERIC_TOOL, "terminus_json": GENERIC_TERMINUS}[kind]


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


# ------------------------------------------------------------------ states + tables
def load_states() -> dict[str, dict]:
    out = {}
    for f in glob.glob(STATES_GLOB):
        d = json.load(open(f))
        out[d["state_id"].rsplit(":", 1)[0]] = d
    return out


def base_id(state_id: str) -> str:
    return state_id.rsplit(":", 1)[0]


def build_tables() -> list[dict]:
    """One record per simulated state: rows (src, arm, idx, y, norm, outcome), kind, depth, harness, tag."""
    states = load_states()
    cont = read_jsonl(SPLIT_DIR / "continuations.jsonl")
    props = {base_id(p["state_id"]): p for p in read_jsonl(SPLIT_DIR / "proposals.jsonl")}
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cont:
        if c["outcome"] in ("solved", "failed"):
            by[base_id(c["kept_state_id"])].append(c)
    out = []
    for sid, rows in sorted(by.items()):
        stt = states.get(sid)
        if not stt:
            continue
        kind = stt["action_kind"]
        T = sorted([r for r in rows if r["arm"] == "T"], key=lambda r: r["continuation"])
        F = sorted([r for r in rows if r["arm"] != "T"], key=lambda r: r["arm"])
        s, n = sum(r["outcome"] == "solved" for r in T), len(T)
        if n < MIN_N:
            continue
        if 0 < s < n:
            tag = "split"
        elif s == 0 and F:
            tag = "ceiling"
        else:
            continue
        table = []
        for i, r in enumerate(T):
            y = canon(first_action(r, kind), kind)
            table.append({"src": "T", "arm": "T", "idx": i, "cont": r["continuation"], "y": y,
                          "norm": norm_action(y, kind), "outcome": r["outcome"], "ok": bool(y)})
        pp = props.get(sid)
        for j, r in enumerate(F):
            prop = next((q for q in (pp or {}).get("proposals", []) if q["arm"] == r["arm"]), None)
            if prop is None:
                continue
            y = canon(prop["y"], kind)
            table.append({"src": "F", "arm": r["arm"], "idx": j, "y": y, "norm": norm_action(y, kind),
                          "outcome": r["outcome"], "ok": bool(y), "judge": prop.get("judge"), "source": prop.get("source")})
        table = [t for t in table if t["ok"]]
        out.append({"sid": sid, "tag": tag, "kind": kind, "harness": stt["harness"], "depth": stt["depth"],
                    "source": stt["source"], "s": s, "n": n, "table": table,
                    "n_F": sum(1 for t in table if t["src"] == "F"),
                    "s_F": sum(1 for t in table if t["src"] == "F" and t["outcome"] == "solved")})
    return out


def classes_of(table: list[dict]) -> dict[str, dict]:
    """norm -> {y (representative), solved, total, members}."""
    cl: dict[str, dict] = {}
    for t in table:
        c = cl.setdefault(t["norm"], {"y": t["y"], "solved": 0, "total": 0, "members": []})
        c["total"] += 1
        c["solved"] += t["outcome"] == "solved"
        c["members"].append((t["src"], t["idx"]))
    for c in cl.values():
        c["V"] = c["solved"] / c["total"]
    return cl


def baseline(table: list[dict], cl: dict[str, dict]) -> float | None:
    vs = [cl[t["norm"]]["V"] for t in table if t["src"] == "T"]
    return st.mean(vs) if vs else None


# ------------------------------------------------------------------ class matching (the cascade)
class Verdicts:
    """Stored equivalence verdicts keyed by (sid, h(a), h(b)); a = stored action, b = candidate."""

    def __init__(self):
        self.d: dict[tuple, dict] = {}
        for r in read_jsonl(EQUIV):
            if r.get("same") is not None and r.get("finish") == "stop":     # a capped reply's word was scraped from reasoning: re-ask
                self.d[(r["sid"], r["ha"], r["hb"])] = r

    def get(self, sid: str, a: str, b: str) -> bool | None:
        r = self.d.get((sid, h(a), h(b)))
        return None if r is None else r["same"]


def match(y: str, kind: str, table: list[dict], sid: str, verdicts: Verdicts | None) -> dict:
    """Class assignment by the cascade. Judge-credited classes are pooled (a
    candidate equivalent to several stored actions inherits their union)."""
    cl = classes_of(table)
    B = baseline(table, cl)
    ny = norm_action(y, kind)
    res = {"how": "none", "V": None, "B": B, "term": 0.0, "classes": [], "judge_needed": 0, "judge_missing": 0,
           "best_jaccard": 0.0, "best_jaccard_envelope": max((jaccard(ny, k) for k in cl), default=0.0)}
    if ny in cl:
        c = cl[ny]
        res.update({"how": "exact", "V": c["V"], "classes": [ny], "term": c["V"] - B})
        return res
    scored = sorted(((body_jaccard(y, c["y"], kind), k) for k, c in cl.items()), reverse=True)
    res["best_jaccard"] = scored[0][0] if scored else 0.0
    if scored and scored[0][0] >= JACCARD_MIN:
        c = cl[scored[0][1]]
        res.update({"how": "jaccard", "V": c["V"], "classes": [scored[0][1]], "term": c["V"] - B})
        return res
    res["judge_needed"] = len(cl)
    if verdicts is None:
        return res
    yes, missing = [], 0
    for k, c in cl.items():
        v = verdicts.get(sid, c["y"], y)
        if v is None:
            missing += 1
        elif v:
            yes.append(k)
    res["judge_missing"] = missing
    if yes:
        s_ = sum(cl[k]["solved"] for k in yes)
        n_ = sum(cl[k]["total"] for k in yes)
        res.update({"how": "judge", "V": s_ / n_, "classes": yes, "term": s_ / n_ - B})
    return res


def judge_all_classes(y: str, kind: str, table: list[dict], sid: str, verdicts: Verdicts) -> dict:
    """Judge verdict for y against every class regardless of surface match (agreement analysis)."""
    cl = classes_of(table)
    ny = norm_action(y, kind)
    out = {"n_classes": len(cl), "yes": [], "no": [], "missing": 0, "surface": []}
    for k, c in cl.items():
        surf = "exact" if k == ny else ("jaccard" if body_jaccard(y, c["y"], kind) >= JACCARD_MIN else "none")
        v = verdicts.get(sid, c["y"], y)
        if v is None:
            out["missing"] += 1
        elif v:
            out["yes"].append(k)
        else:
            out["no"].append(k)
        out["surface"].append((k, surf, v))
    return out


# ------------------------------------------------------------------ context rendering for the judges
def msg_text(m: dict) -> str:
    c = m.get("content")
    if isinstance(c, list):
        c = "\n".join(x.get("text", "") for x in c if isinstance(x, dict))
    c = (c or "").strip()
    if m.get("tool_calls"):
        c = (c + "\n\n" if c else "") + "\n".join(common.render_tool_call(
            {"function": {"name": (tc.get("function") or tc).get("name"), "arguments": (tc.get("function") or tc).get("arguments")}})
            for tc in m["tool_calls"])
    return c


def context_text(stt: dict) -> str:
    msgs = stt["messages"]
    task = next((msg_text(m) for m in msgs if m["role"] == "user"), "")
    if len(task) > TASK_CAP:
        task = task[:TASK_CAP] + " …[truncated]"
    tail = "\n\n".join(f"[{m['role'].upper()}]\n{msg_text(m)}" for m in msgs[-TAIL_MSGS:])
    if len(tail) > TAIL_CAP:
        tail = "…[earlier part elided]\n" + tail[-TAIL_CAP:]
    return (f"Task (first user message):\n{task}\n\n--- Most recent transcript (last {TAIL_MSGS} messages of "
            f"{len(msgs)}; dialect {stt['action_kind']}, harness {stt['harness']}) ---\n{tail}")


EQUIV_SYSTEM = (
    "You compare two candidate NEXT ACTIONS proposed by an agent at the same point of a coding / terminal task. "
    "Decide whether they make the SAME DECISION: the same operation on the same target with the same intent. "
    "Cosmetic differences (flags, ordering, quoting, output formatting, line ranges that overlap, wording of a "
    "commit message) do NOT matter. A different target (other file / function / directory), a different "
    "operation (read vs edit vs run tests vs finish), or a different intent DOES matter. "
    "Answer with exactly one word: YES or NO."
)
PREF_SYSTEM = (
    "You are an expert software engineer reviewing an agent that is solving a coding / terminal task. "
    "You see the task, the most recent transcript, and two candidate NEXT ACTIONS. Decide which candidate is the "
    "better next step towards actually solving the task (correctness and progress, not style). "
    "Answer with exactly one letter: A or B."
)


def one_word(txt: str, words: tuple[str, ...]) -> str | None:
    t = (txt or "").strip().upper()
    for w in words:
        if t.startswith(w):
            return w
    found = re.findall(r"\b(" + "|".join(words) + r")\b", t)
    return found[-1] if found else None


# ------------------------------------------------------------------ king client (raw /completions, as hint_design.King)
def king_tokenizer():
    from transformers import AutoTokenizer  # heavy import kept lazy for CLI help
    KING_TOK_DIR.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=300) as c:
        for f in KING_TOK_FILES:
            p = KING_TOK_DIR / f
            if not p.exists():
                r = c.get(f"https://models.affine.io/models/sha256/{KING_DIGEST}/{f}")
                r.raise_for_status()
                p.write_bytes(r.content)
    return AutoTokenizer.from_pretrained(str(KING_TOK_DIR))


def wire_to_template(messages: list[dict]) -> list[dict]:
    """Engy wire messages -> what the Qwen chat template expects (tool-call
    arguments as dicts; the template iterates `arguments|items`)."""
    out = []
    for m in messages:
        m2 = {k: v for k, v in m.items() if k in ("role", "content", "tool_calls", "tool_call_id", "name")}
        if m2.get("content") is None:
            m2["content"] = ""
        if m2.get("tool_calls"):
            tcs = []
            for tc in m2["tool_calls"]:
                fn = dict(tc.get("function") or {})
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except ValueError:
                        fn["arguments"] = {"_raw": fn["arguments"]}
                tcs.append({"id": tc.get("id"), "type": "function", "function": fn})
            m2["tool_calls"] = tcs
        out.append(m2)
    return out


class King:
    def __init__(self, concurrency: int = 6, timeout: float = 1500.0, retries: int = 3):
        self.cli = httpx.AsyncClient(base_url=KING_BASE, timeout=timeout, headers={"Authorization": f"Bearer {KING_KEY}"})
        self.sem = asyncio.Semaphore(concurrency)
        self.retries = retries
        self.tok = king_tokenizer()

    def prompt(self, messages: list[dict], tools: list[dict] | None) -> str:
        kw = {"tools": tools} if tools else {}
        p = self.tok.apply_chat_template(wire_to_template(messages), tokenize=False, add_generation_prompt=True, **kw)
        if not p.rstrip().endswith(common.THINK_OPEN):
            p = p + common.THINK_OPEN + "\n"
        return p

    async def complete(self, prompt: str) -> dict:
        payload = {"model": KING_MODEL, "prompt": prompt, "max_tokens": KING_MAX_TOKENS, "temperature": KING_TEMPERATURE}
        last = None
        for attempt in range(self.retries):
            async with self.sem:
                try:
                    r = await self.cli.post("/completions", json=payload)
                    if r.status_code == 200:
                        d = r.json()
                        ch = d["choices"][0]
                        return {"raw": ch.get("text") or "", "finish": ch.get("finish_reason"), "usage": d.get("usage")}
                    last = f"HTTP {r.status_code}: {r.text[:200]}"
                    if r.status_code in (400, 404, 413, 422):
                        raise RuntimeError(f"king {last}")
                except httpx.HTTPError as e:
                    last = repr(e)[:200]
            await asyncio.sleep(5 * (attempt + 1))
        raise RuntimeError(f"king failed after {self.retries} tries: {last}")


async def king_reachable() -> bool:
    try:
        async with httpx.AsyncClient(timeout=20, headers={"Authorization": f"Bearer {KING_KEY}"}) as c:
            r = await c.get(KING_BASE + "/models")
            return r.status_code == 200 and KING_MODEL in r.text
    except httpx.HTTPError:
        return False


async def sample_king(tables: list[dict], concurrency: int) -> None:
    if not await king_reachable():
        raise SystemExit(f"king endpoint {KING_BASE} unreachable")
    states = load_states()
    king = King(concurrency=concurrency)
    done = {(r["sid"], r["i"]) for r in read_jsonl(KING_SAMPLES) if "error" not in r}
    jobs = [(t, i) for t in tables for i in range(K_KING) if (t["sid"], i) not in done]
    print(f"king: {len(jobs)} samples to draw ({len(tables)} states x {K_KING})", flush=True)
    t0 = time.time()
    n_done = 0

    async def one(t: dict, i: int) -> None:
        nonlocal n_done
        stt = states[t["sid"]]
        prompt = king.prompt(stt["messages"], stt.get("tools"))
        try:
            r = await king.complete(prompt)
        except Exception as ex:  # noqa: BLE001
            append_jsonl(KING_SAMPLES, {"sid": t["sid"], "i": i, "error": repr(ex)[:300]})
            n_done += 1
            return
        p = parse_king_raw(r["raw"], t["kind"])
        append_jsonl(KING_SAMPLES, {"sid": t["sid"], "i": i, "kind": t["kind"], "finish": r["finish"], "usage": r["usage"],
                                    "prompt_chars": len(prompt), "raw": r["raw"], **p})
        n_done += 1
        print(f"  king {n_done}/{len(jobs)} {t['sid'][:8]} i={i} parsed={p['parsed']} closed={p['think_closed']} "
              f"finish={r['finish']} {time.time() - t0:.0f}s", flush=True)

    await asyncio.gather(*[one(*j) for j in jobs])


# ------------------------------------------------------------------ miner candidate sets
def king_actions(sid: str) -> list[dict]:
    rows = sorted([r for r in read_jsonl(KING_SAMPLES) if r["sid"] == sid and "error" not in r], key=lambda r: r["i"])
    return rows


def candidates(t: dict, states: dict) -> list[dict]:
    """(miner, label, y, exclude) — exclude = (src, idx) row to drop from the table (LOO)."""
    stt = states[t["sid"]]
    kind = t["kind"]
    out = []
    for r in t["table"]:
        if r["src"] == "T":
            out.append({"miner": "teacher_heldout", "label": f"T{r['idx']}", "y": r["y"], "exclude": ("T", r["idx"]), "outcome": r["outcome"]})
        else:
            out.append({"miner": "frontier_loo", "label": r["arm"], "y": r["y"], "exclude": ("F", r["idx"]), "outcome": r["outcome"]})
    for r in king_actions(t["sid"]):
        out.append({"miner": "king", "label": f"K{r['i']}", "y": r["y"] if r["parsed"] else "", "exclude": None,
                    "forfeit": not r["parsed"], "think_closed": r["think_closed"], "finish": r["finish"]})
    out.append({"miner": "generic", "label": "gen", "y": generic_action(kind), "exclude": None})
    rl = repeat_last(stt["messages"], kind)
    if rl:
        out.append({"miner": "repeat", "label": "rep", "y": rl, "exclude": None})
    return out


def reduced(table: list[dict], exclude: tuple | None) -> list[dict]:
    if exclude is None:
        return table
    return [r for r in table if (r["src"], r["idx"]) != exclude]


# ------------------------------------------------------------------ judges
async def run_equivalence(tables: list[dict], engy: Engy, run: str) -> None:
    """Every miner candidate vs every stored class of its (reduced) table —
    surface-matched pairs included, for the agreement analysis."""
    states = load_states()
    verdicts = Verdicts()
    capped = {(r["sid"], r["ha"], r["hb"]) for r in read_jsonl(EQUIV) if r.get("finish") == "length"}
    jobs = []
    for t in tables:
        ctx = context_text(states[t["sid"]])
        for c in candidates(t, states):
            if not c["y"]:
                continue
            for k, cl in classes_of(reduced(t["table"], c["exclude"])).items():
                if verdicts.get(t["sid"], cl["y"], c["y"]) is None and cl["y"] != c["y"]:
                    jobs.append((t["sid"], ctx, cl["y"], c["y"], c["miner"], c["label"]))
    jobs = list({(j[0], h(j[2]), h(j[3])): j for j in jobs}.values())
    print(f"equivalence judge: {len(jobs)} calls ({sum((j[0], h(j[2]), h(j[3])) in capped for j in jobs)} no-think retries)", flush=True)

    async def one(sid, ctx, a, b, miner, label):
        user = (f"{ctx}\n\n--- Candidate action A ---\n{a[:ACTION_CAP]}\n\n--- Candidate action B ---\n{b[:ACTION_CAP]}\n\n"
                "Do A and B make the SAME DECISION at this point? One word: YES or NO.")
        nothink = (sid, h(a), h(b)) in capped       # T0 thinking looped past the cap once: it will again; answer without thinking
        try:
            r = await engy.chat(JUDGE_MODEL, [{"role": "system", "content": EQUIV_SYSTEM}, {"role": "user", "content": user}],
                                temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, **(NOTHINK if nothink else {}))
            w = one_word(r["content"], ("YES", "NO"))
            append_jsonl(EQUIV, {"sid": sid, "ha": h(a), "hb": h(b), "a": a[:400], "b": b[:400], "miner": miner, "label": label,
                                 "same": None if w is None else (w == "YES"), "finish": r["finish"], "cost_usd": r["cost_usd"],
                                 "usage": r["usage"], "nothink": nothink})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(EQUIV, {"sid": sid, "ha": h(a), "hb": h(b), "miner": miner, "label": label, "same": None, "error": repr(ex)[:200]})

    for s in range(0, len(jobs), 32):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + 32]])
        log_cost("equiv", engy, f"{min(s + 32, len(jobs))}/{len(jobs)}", run)


def preference_pairs(tables: list[dict]) -> list[dict]:
    """(better = env-solved action, worse = env-failed action) pairs with known outcomes."""
    pairs = []
    for t in tables:
        cl = classes_of(t["table"])
        T = [r for r in t["table"] if r["src"] == "T"]
        F = [r for r in t["table"] if r["src"] == "F"]
        if t["tag"] == "split":
            for a in T:
                for b in T:
                    if a["outcome"] == "solved" and b["outcome"] == "failed" and a["norm"] != b["norm"]:
                        pairs.append({"sid": t["sid"], "kind": "split_T_vs_T", "good": a["y"], "bad": b["y"],
                                      "good_V": cl[a["norm"]]["V"], "bad_V": cl[b["norm"]]["V"]})
        else:
            for a in F:
                if a["outcome"] != "solved":
                    continue
                for b in T:
                    if a["norm"] != b["norm"]:
                        pairs.append({"sid": t["sid"], "kind": "ceiling_Fsolved_vs_Tfailed", "good": a["y"], "bad": b["y"],
                                      "good_V": cl[a["norm"]]["V"], "bad_V": cl[b["norm"]]["V"]})
                for b in F:
                    if b["outcome"] == "failed" and a["norm"] != b["norm"]:
                        pairs.append({"sid": t["sid"], "kind": "ceiling_Fsolved_vs_Ffailed", "good": a["y"], "bad": b["y"],
                                      "good_V": cl[a["norm"]]["V"], "bad_V": cl[b["norm"]]["V"]})
    # dedupe identical (good, bad) texts
    return list({(p["sid"], h(p["good"]), h(p["bad"])): p for p in pairs}.values())


async def run_preference(tables: list[dict], engy: Engy, run: str) -> None:
    states = load_states()
    prev = read_jsonl(PREF)
    done = {(r["sid"], r["hg"], r["hb"], r["order"]) for r in prev if r.get("pick") and r.get("finish") == "stop"}
    capped = {(r["sid"], r["hg"], r["hb"], r["order"]) for r in prev if r.get("finish") == "length"}
    jobs = []
    for p in preference_pairs(tables):
        for order in ("good_first", "bad_first"):
            if (p["sid"], h(p["good"]), h(p["bad"]), order) not in done:
                jobs.append((p, order))
    print(f"preference judge: {len(jobs)} calls ({len(preference_pairs(tables))} pairs x 2 orders; "
          f"{sum((p['sid'], h(p['good']), h(p['bad']), o) in capped for p, o in jobs)} no-think retries)", flush=True)

    async def one(p: dict, order: str):
        ctx = context_text(states[p["sid"]])
        A, B = (p["good"], p["bad"]) if order == "good_first" else (p["bad"], p["good"])
        user = (f"{ctx}\n\n--- Candidate action A ---\n{A[:ACTION_CAP]}\n\n--- Candidate action B ---\n{B[:ACTION_CAP]}\n\n"
                "Which is the better next step here? One letter: A or B.")
        nothink = (p["sid"], h(p["good"]), h(p["bad"]), order) in capped
        try:
            r = await engy.chat(JUDGE_MODEL, [{"role": "system", "content": PREF_SYSTEM}, {"role": "user", "content": user}],
                                temperature=0.0, max_tokens=JUDGE_MAX_TOKENS, **(NOTHINK if nothink else {}))
            w = one_word(r["content"], ("A", "B"))
            picked_good = None if w is None else ((w == "A") == (order == "good_first"))
            append_jsonl(PREF, {"sid": p["sid"], "kind": p["kind"], "hg": h(p["good"]), "hb": h(p["bad"]), "order": order,
                                "pick": w, "picked_good": picked_good, "good_V": p["good_V"], "bad_V": p["bad_V"],
                                "finish": r["finish"], "cost_usd": r["cost_usd"], "usage": r["usage"], "nothink": nothink})
        except Exception as ex:  # noqa: BLE001
            append_jsonl(PREF, {"sid": p["sid"], "kind": p["kind"], "hg": h(p["good"]), "hb": h(p["bad"]), "order": order,
                                "pick": None, "error": repr(ex)[:200]})

    for s in range(0, len(jobs), 32):
        await asyncio.gather(*[one(*j) for j in jobs[s:s + 32]])
        log_cost("pref", engy, f"{min(s + 32, len(jobs))}/{len(jobs)}", run)


# ------------------------------------------------------------------ report
def _mean(v):
    v = [x for x in v if x is not None and math.isfinite(x)]
    return st.mean(v) if v else None


def _f(x, w=6, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    return f"{x:{w}.{p}f}"


def sign_test(diffs: list[float]) -> dict:
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return {"pos": pos, "neg": neg, "ties": len(diffs), "p": None}
    k = min(pos, neg)
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
    return {"pos": pos, "neg": neg, "ties": len(diffs) - n, "p": p}


def paired(diffs: list[float]) -> dict:
    d = [x for x in diffs if x is not None]
    if len(d) < 3:
        return {"n": len(d), "mean": _mean(d), "se": None, "z": None, **sign_test(d)}
    se = st.stdev(d) / math.sqrt(len(d))
    return {"n": len(d), "mean": st.mean(d), "se": se, "z": st.mean(d) / se if se > 0 else None, **sign_test(d)}


MINERS = ("teacher_heldout", "king", "frontier_loo", "generic", "repeat", "filler")


def score_all(tables: list[dict], use_judge: bool) -> list[dict]:
    states = load_states()
    verdicts = Verdicts() if use_judge else None
    out = []
    for t in tables:
        cl = classes_of(t["table"])
        B = baseline(t["table"], cl)
        rec = {"sid": t["sid"], "tag": t["tag"], "kind": t["kind"], "harness": t["harness"], "depth": t["depth"], "source": t["source"],
               "s": t["s"], "n": t["n"], "s_F": t["s_F"], "n_F": t["n_F"], "B": B, "n_classes": len(cl),
               "classes": [{"y": c["y"][:160], "solved": c["solved"], "total": c["total"], "V": c["V"], "members": c["members"]} for c in cl.values()],
               "cands": []}
        for c in candidates(t, states):
            tab = reduced(t["table"], c["exclude"])
            if not c["y"]:
                m = {"how": "forfeit", "V": None, "B": baseline(tab, classes_of(tab)), "term": 0.0, "classes": [], "judge_needed": 0,
                     "judge_missing": 0, "best_jaccard": 0.0}
            else:
                m = match(c["y"], t["kind"], tab, t["sid"], verdicts)
            agreement = judge_all_classes(c["y"], t["kind"], tab, t["sid"], verdicts) if (verdicts and c["y"]) else None
            rec["cands"].append({**{k: v for k, v in c.items() if k != "exclude"}, "exclude": c["exclude"], **m,
                                 "y_short": c["y"][:200], "agreement": agreement})
        per = {}
        for miner in MINERS:
            src = "king" if miner == "filler" else miner
            cs = [c for c in rec["cands"] if c["miner"] == src]
            if not cs:
                per[miner] = None
                continue
            per[miner] = {"n": len(cs), "term": st.mean(c["term"] for c in cs),
                          "credited": st.mean(1.0 if c["how"] in ("exact", "jaccard", "judge") else 0.0 for c in cs),
                          "how": dict(collections.Counter(c["how"] for c in cs)),
                          "forfeit": st.mean(1.0 if c.get("forfeit") else 0.0 for c in cs) if src == "king" else None}
        rec["per_miner"] = per
        out.append(rec)
    return out


def aggregate(rows: list[dict], label: str) -> dict:
    out: dict = {"label": label, "n_states": len(rows)}
    for miner in MINERS:
        rs = [r for r in rows if r["per_miner"].get(miner)]
        if not rs:
            out[miner] = None
            continue
        terms = [r["per_miner"][miner]["term"] for r in rs]
        out[miner] = {"n_states": len(rs), "n_actions": sum(r["per_miner"][miner]["n"] for r in rs),
                      "term": st.mean(terms), "se": st.stdev(terms) / math.sqrt(len(terms)) if len(terms) > 1 else None,
                      "credited_actions": _mean([r["per_miner"][miner]["credited"] for r in rs]),
                      "credited_states": st.mean(1.0 if r["per_miner"][miner]["credited"] > 0 else 0.0 for r in rs),
                      "positive_states": st.mean(1.0 if r["per_miner"][miner]["term"] > 0 else 0.0 for r in rs),
                      "how": dict(sum((collections.Counter(r["per_miner"][miner]["how"]) for r in rs), collections.Counter())),
                      "forfeit": _mean([r["per_miner"][miner]["forfeit"] for r in rs]) if miner in ("king", "filler") else None}
    # paired contrasts
    out["paired"] = {}
    for a, b in (("teacher_heldout", "king"), ("teacher_heldout", "generic"), ("teacher_heldout", "repeat"), ("king", "generic"),
                 ("king", "repeat"), ("frontier_loo", "king"), ("frontier_loo", "teacher_heldout")):
        d = [r["per_miner"][a]["term"] - r["per_miner"][b]["term"] for r in rows if r["per_miner"].get(a) and r["per_miner"].get(b)]
        out["paired"][f"{a}-{b}"] = paired(d)
    return out


def judge_stats(rows: list[dict]) -> dict:
    """Equivalence-judge agreement with surface matching + how many king actions are credited only via the judge."""
    agree = {"exact": [0, 0], "jaccard": [0, 0], "none": [0, 0]}     # [yes, total]
    king_only_judge = king_total = king_surface = king_judge_yes_any = 0
    per_miner_how = collections.defaultdict(collections.Counter)
    unmatched_surface = collections.defaultdict(list)
    envelope_only = collections.Counter()
    for r in rows:
        for c in r["cands"]:
            per_miner_how[c["miner"]][c["how"]] += 1
            if c["y"]:
                unmatched_surface[c["miner"]].append(1.0 if c["how"] in ("judge", "none") else 0.0)
                if c["how"] not in ("exact", "jaccard") and c.get("best_jaccard_envelope", 0) >= JACCARD_MIN:
                    envelope_only[c["miner"]] += 1
            ag = c.get("agreement")
            if ag:
                for _, surf, v in ag["surface"]:
                    if v is not None:
                        agree[surf][1] += 1
                        agree[surf][0] += v
            if c["miner"] == "king" and c["y"]:
                king_total += 1
                if c["how"] in ("exact", "jaccard"):
                    king_surface += 1
                if c["how"] == "judge":
                    king_only_judge += 1
                if ag and ag["yes"]:
                    king_judge_yes_any += 1
    return {"agreement_yes_rate": {k: {"yes": v[0], "n": v[1], "rate": (v[0] / v[1]) if v[1] else None} for k, v in agree.items()},
            "king": {"parsed_actions": king_total, "surface_credited": king_surface, "judge_only_credited": king_only_judge,
                     "judge_yes_any": king_judge_yes_any},
            "how_by_miner": {k: dict(v) for k, v in per_miner_how.items()},
            "surface_unmatched_rate": {k: _mean(v) for k, v in unmatched_surface.items()},
            "envelope_jaccard_only_matches": dict(envelope_only)}


def preference_stats() -> dict:
    recs = [r for r in read_jsonl(PREF) if r.get("pick") and r.get("finish") == "stop"]
    by = collections.defaultdict(dict)
    for r in recs:
        by[(r["sid"], r["hg"], r["hb"], r["kind"])][r["order"]] = r["picked_good"]      # last clean verdict wins
    out = {}
    for kind in sorted({k[3] for k in by} | {"ALL"}):
        ps = [v for k, v in by.items() if kind == "ALL" or k[3] == kind]
        both = [v for v in ps if len(v) == 2]
        consistent = [v for v in both if v["good_first"] == v["bad_first"]]
        correct = [v for v in consistent if v["good_first"]]
        wrong = [v for v in consistent if not v["good_first"]]
        first_pos = sum(1 for v in both if v["good_first"] and not v["bad_first"])      # picks A both times
        second_pos = sum(1 for v in both if (not v["good_first"]) and v["bad_first"])   # picks B both times
        out[kind] = {"pairs": len(both), "consistent": len(consistent), "correct": len(correct), "wrong": len(wrong),
                     "flip": len(both) - len(consistent), "accuracy_strict": len(correct) / len(both) if both else None,
                     "accuracy_given_consistent": len(correct) / len(consistent) if consistent else None,
                     "position_bias_rate": (len(both) - len(consistent)) / len(both) if both else None,
                     "prefers_A_when_flipping": first_pos, "prefers_B_when_flipping": second_pos,
                     "single_call_accuracy": _mean([1.0 if v.get(o) else 0.0 for v in ps for o in ("good_first", "bad_first") if o in v])}
    return out


def cost_stats(rows: list[dict]) -> dict:
    eq = [r for r in read_jsonl(EQUIV) if "error" not in r and r.get("cost_usd") is not None]
    per_call = _mean([r["cost_usd"] for r in eq])
    ptok = _mean([(r.get("usage") or {}).get("prompt_tokens") for r in eq if r.get("usage")])
    ctok = _mean([(r.get("usage") or {}).get("completion_tokens") for r in eq if r.get("usage")])
    # judge calls actually needed per king action (cascade): 0 if surface-matched, else one per stored class
    need = [c["judge_needed"] for r in rows for c in r["cands"] if c["miner"] == "king" and c["y"]]
    unmatched = [1.0 if c["how"] in ("judge", "none") else 0.0 for r in rows for c in r["cands"] if c["miner"] == "king" and c["y"]]
    calls_per_turn = _mean(need) or 0.0
    n_classes = _mean([r["n_classes"] for r in rows])
    per_side = DUEL_TURNS * DECISION_SHARE * calls_per_turn * (per_call or 0.0)
    return {"judge_usd_per_call": per_call, "judge_prompt_tokens": ptok, "judge_completion_tokens": ctok, "n_judge_calls_logged": len(eq),
            "king_surface_unmatched_rate": _mean(unmatched), "classes_per_state": n_classes,
            "judge_calls_per_decision_turn_per_side": calls_per_turn,
            "decision_turns_per_duel": DUEL_TURNS * DECISION_SHARE,
            "usd_per_duel_one_side": per_side, "usd_per_duel_both_sides": 2 * per_side}


def print_report(tables: list[dict], rows: list[dict], rows_nojudge: list[dict], rep: dict) -> str:
    L: list[str] = []
    P = L.append
    P("Verified-action-value (VAV) rule simulation — §6 candidate (a)+(d) — 2026-09-21")
    P(f"states {len(rows)} (split {sum(r['tag'] == 'split' for r in rows)}, ceiling {sum(r['tag'] == 'ceiling' for r in rows)}); "
      f"king {KING_MODEL} (reign {KING_REIGN}) at {KING_BASE}; judge {JUDGE_MODEL} T0; $ spent (Engy) {rep['cost_usd_total']:.3f}")
    P("")
    P("Terms: state = rollout_id:turn_idx prefix; table = {class -> (solved,total)} over graded continuations whose FIRST action is in the class")
    P("       (T = teacher continuations @T0.8, F1/F1b/F1c = forced frontier proposals, teacher finishes); class = norm_action-exact group;")
    P("       V = solved/total; B = mean V over the teacher's own N first actions (with themselves); term = V(class(y)) − B, unmatched → 0;")
    P("       LOO = the scored action's own continuation removed from the table (and B) first; credited = joined a class (exact / Jaccard>=.5 / judge YES);")
    P("       teacher_heldout = mean over the N teacher first actions (LOO); frontier_loo = graded frontier proposals (LOO; ceiling states);")
    P("       king = mean over 3 fresh reign-20 samples (forfeit = no </think> or no action, term 0); generic = `ls -la` in the dialect;")
    P("       repeat = previous assistant action re-issued; filler = king's action with a content-free thought (same V; only a thought gate differs);")
    P("       sign = exact two-sided binomial on per-state sign of the difference (ties dropped); z = paired mean/SE over states.")
    P("")
    P("== 1. TABLES per state ==")
    P(f"{'tag':<7} {'state':<8} {'harness':<19} {'d':>2} {'T s/N':>5} {'F s/n':>5} {'cls':>3} {'B':>6}  classes (s/n : action)")
    for r in rows:
        P(f"{r['tag']:<7} {r['sid'][:8]:<8} {r['harness']:<19} {r['depth']:>2} {r['s']}/{r['n']:<3} {r['s_F']}/{r['n_F']:<3} {r['n_classes']:>3} {_f(r['B'],6,3)}")
        for c in sorted(r["classes"], key=lambda c: (-c["V"], -c["total"])):
            mem = ",".join(f"{s}{i}" for s, i in c["members"])
            P(f"{'':<50} {c['solved']}/{c['total']} [{mem:<12}] {c['y'].replace(chr(10), ' ')[:110]}")
    P("")
    P("== 2. MINERS — per state term (V − B) and how the action was credited ==")
    P(f"{'tag':<7} {'state':<8} {'harness':<19} {'B':>6} | {'teacherLOO':>10} {'king':>7} {'frontLOO':>8} {'generic':>7} {'repeat':>7} | king how / forfeits")
    for r in rows:
        pm = r["per_miner"]

        def cell(m, w=7):
            return _f(pm[m]["term"], w, 3) if pm.get(m) else " " * (w - 3) + "n/a"
        kh = pm["king"]["how"] if pm.get("king") else {}
        P(f"{r['tag']:<7} {r['sid'][:8]:<8} {r['harness']:<19} {_f(r['B'],6,3)} | {cell('teacher_heldout',10)} {cell('king')} {cell('frontier_loo',8)} "
          f"{cell('generic')} {cell('repeat')} | {kh} forfeit {_f(pm['king']['forfeit'],4,2) if pm.get('king') else 'n/a'}")
    P("")
    for key in ("pooled", "split", "ceiling"):
        a = rep["agg"][key]
        P(f"-- {key.upper()} (n states {a['n_states']}) -- mean term per miner, credited share, positive-state share")
        P(f"{'miner':<16} {'states':>6} {'acts':>5} {'term':>8} {'se':>7} {'credited/act':>12} {'credited/state':>14} {'term>0':>7}  how (actions) | forfeit")
        for m in MINERS:
            g = a.get(m)
            if not g:
                continue
            P(f"{m:<16} {g['n_states']:>6} {g['n_actions']:>5} {_f(g['term'],8,4)} {_f(g['se'],7,4)} {_f(g['credited_actions'],12,2)} {_f(g['credited_states'],14,2)} "
              f"{_f(g['positive_states'],7,2)}  {g['how']} | {_f(g['forfeit'],4,2) if g['forfeit'] is not None else ''}")
        P("   paired contrasts (a − b): mean, z, sign test pos/neg/ties, p")
        for k, p in a["paired"].items():
            P(f"     {k:<32} mean {_f(p['mean'],8,4)} z {_f(p['z'],6,2)} sign +{p['pos']}/−{p['neg']}/={p['ties']} p {_f(p['p'],6,3)}")
        P("")
    P("   filler note: the ranking term never reads the thought, so a king action under a content-free thought scores exactly the king row;")
    P("   an 80+-char filler passes the live length floor (min_thought_chars 80) — only a thought GATE (G band / B license) can separate filler from king.")
    P("")
    a0 = rep["agg_nojudge"]["pooled"]
    P("-- POOLED WITHOUT the equivalence judge (surface cascade only: exact → Jaccard>=.5) --")
    P(f"{'miner':<16} {'term':>8} {'credited/act':>12} {'credited/state':>14}  how")
    for m in MINERS:
        g = a0.get(m)
        if g:
            P(f"{m:<16} {_f(g['term'],8,4)} {_f(g['credited_actions'],12,2)} {_f(g['credited_states'],14,2)}  {g['how']}")
    p = a0["paired"]["teacher_heldout-king"]
    P(f"   teacher_heldout − king without judge: mean {_f(p['mean'],8,4)} z {_f(p['z'],6,2)} sign +{p['pos']}/−{p['neg']}/={p['ties']} p {_f(p['p'],6,3)}")
    P("")
    P("== 3. JUDGES ==")
    js = rep["judge"]
    P("-- equivalence judge (glm-5.3-flash T0, 'same decision?') vs surface matching: YES rate by surface relation of the pair --")
    for k, v in js["agreement_yes_rate"].items():
        P(f"   surface {k:<8} judge YES {v['yes']}/{v['n']} = {_f(v['rate'],5,2)}")
    kg = js["king"]
    P(f"   king actions parsed {kg['parsed_actions']}: surface-credited {kg['surface_credited']}, credited ONLY via judge {kg['judge_only_credited']}, "
      f"judge YES to some class {kg['judge_yes_any']}")
    P(f"   how by miner: {js['how_by_miner']}")
    P(f"   surface-unmatched rate by miner: { {k: round(v, 3) for k, v in js['surface_unmatched_rate'].items() if v is not None} }")
    P(f"   actions Jaccard-matched ONLY on the JSON envelope (would be credited by envelope-Jaccard, not by body-Jaccard): {js['envelope_jaccard_only_matches']}")
    P("-- preference judge (glm-5.3-flash T0, 'which is the better next step?', both orders) vs env outcome --")
    P(f"{'pair kind':<30} {'pairs':>5} {'consist':>7} {'correct':>7} {'wrong':>5} {'flip':>4} {'acc(strict)':>11} {'acc|consist':>11} {'pos-bias':>8} {'1-call acc':>10}  flips→A / flips→B")
    for k, v in rep["preference"].items():
        P(f"{k:<30} {v['pairs']:>5} {v['consistent']:>7} {v['correct']:>7} {v['wrong']:>5} {v['flip']:>4} {_f(v['accuracy_strict'],11,2)} {_f(v['accuracy_given_consistent'],11,2)} "
          f"{_f(v['position_bias_rate'],8,2)} {_f(v['single_call_accuracy'],10,2)}  {v['prefers_A_when_flipping']} / {v['prefers_B_when_flipping']}")
    P("   (strict = both orders agree AND pick the env-solved action; flip = the two orders disagree = position-dependent verdict; chance = 0.25 strict / 0.5 per call)")
    P("")
    P("== 4. COST ==")
    c = rep["cost"]
    P(f"judge ${_f(c['judge_usd_per_call'],7,5)}/call (prompt {_f(c['judge_prompt_tokens'],6,0)} tok, completion {_f(c['judge_completion_tokens'],5,0)} tok, {c['n_judge_calls_logged']} calls logged); "
      f"king actions surface-unmatched {_f(c['king_surface_unmatched_rate'],4,2)}; classes/state {_f(c['classes_per_state'],4,2)} → "
      f"{_f(c['judge_calls_per_decision_turn_per_side'],4,2)} judge calls per decision turn per side;")
    P(f"per {DUEL_TURNS}-turn duel at {DECISION_SHARE:.0%} decision share = {c['decision_turns_per_duel']:.0f} decision turns: "
      f"${c['usd_per_duel_one_side']:.3f} one side (challenger only), ${c['usd_per_duel_both_sides']:.3f} both sides (king re-sampled per duel)")
    P(f"$ spent this probe (Engy, judges only; king sampling free): {rep['cost_usd_total']:.3f}")
    txt = "\n".join(L) + "\n"
    return txt


def cmd_table(args):
    tables = build_tables()
    for t in tables:
        cl = classes_of(t["table"])
        print(f"{t['tag']:<7} {t['sid'][:8]} {t['harness']:<19} d={t['depth']:>2} T {t['s']}/{t['n']} F {t['s_F']}/{t['n_F']} classes {len(cl)} B {_f(baseline(t['table'], cl))}")
        for c in cl.values():
            print(f"     {c['solved']}/{c['total']} {[f'{s}{i}' for s, i in c['members']]} {c['y'][:100]!r}")
    write_jsonl(RESULTS / "tables.jsonl", tables)
    print(f"{len(tables)} states -> {RESULTS / 'tables.jsonl'}")


def cmd_king(args):
    asyncio.run(sample_king(build_tables(), args.concurrency))


def cmd_judge(args):
    tables = build_tables()
    # one Engy client per asyncio.run: the client's semaphore is bound to the loop that created it
    asyncio.run(run_equivalence(tables, Engy(concurrency=16), f"{time.time():.0f}"))
    asyncio.run(run_preference(tables, Engy(concurrency=16), f"{time.time():.0f}"))


def cmd_report(args):
    tables = build_tables()
    rows = score_all(tables, use_judge=True)
    rows_nj = score_all(tables, use_judge=False)
    rep = {"n_states": len(rows), "king": {"base": KING_BASE, "model": KING_MODEL, "reign": KING_REIGN},
           "judge_model": JUDGE_MODEL, "cost_usd_total": total_cost(),
           "agg": {"pooled": aggregate(rows, "pooled"), "split": aggregate([r for r in rows if r["tag"] == "split"], "split"),
                   "ceiling": aggregate([r for r in rows if r["tag"] == "ceiling"], "ceiling")},
           "agg_nojudge": {"pooled": aggregate(rows_nj, "pooled")},
           "judge": judge_stats(rows), "preference": preference_stats(), "cost": cost_stats(rows), "states": rows}
    txt = print_report(tables, rows, rows_nj, rep)
    (RESULTS / "report.txt").write_text(txt)
    (RESULTS / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print(txt)
    print(f"-> {RESULTS / 'report.txt'} / report.json")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("table").set_defaults(fn=cmd_table)
    k = sub.add_parser("king"); k.add_argument("--concurrency", type=int, default=6); k.set_defaults(fn=cmd_king)
    sub.add_parser("judge").set_defaults(fn=cmd_judge)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = sub.add_parser("all"); a.add_argument("--concurrency", type=int, default=6)

    def _all(args):
        cmd_table(args); cmd_king(args); cmd_judge(args); cmd_report(args)
    a.set_defaults(fn=_all)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    args.fn(args)


if __name__ == "__main__":
    main()
