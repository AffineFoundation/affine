"""FT-match killing test (2026-09-22, Jacob's frame of 09:01 UTC).

Frame under test: frontier-authored TARGET actions are published in D on
purpose ("leaking is knowing"); freshness comes from generators and from
publishing after the challenger's reveal; the teacher is only a hygiene gate.

Rule ("FT-match"): at each turn D stores a target class set from frontier
deliberation — modal class weight 1.0, every other frontier class weight
0.5 x its share, everything else 0. T(y_miner) = weight of the class the
miner's action falls in, matched by the cascade norm-exact -> body-Jaccard
>= 0.5 -> stored glm-5.3-flash "same decision?" verdict. Gates (think-close,
length floor, B licence, no-repeat) are noted, not scored.

Stages (all resumable; outputs under research/results/frontier_arbiter/ft_match/):
  select   200 Set-A duel turns (50 per dialect: bash / tool_call / terminus_json / text,
           10 per stored record) with prefixes re-materialised from the pinned public
           corpus + 250 panel turns (25 per king 11-20 from each king's crowning duel,
           chosen among the turns that already carry a genesis sample; the previous
           king sits on the king side of the same turns, so every panel model gets
           >= 25 turns and targets are shared).                    -> turns.jsonl.gz
  sample   glm-5.3 as the agent on the prefix: k=5 (Set A) / k=3 (panel), T=0.8,
           max_tokens 8192 (long reasoning allowed), glm-5.3-flash fallback on a
           failed call (stamped); genesis (qwen3.6-35b-a3b) x1 at the duel cap on
           100 Set-A turns (25 per dialect).            -> samples.jsonl, genesis.jsonl
  targets  cluster the frontier samples into decision classes (cascade; judge only
           residual class-representative pairs; verdicts cached) -> targets.jsonl
  score    T for every candidate (teacher refs x3, king, challenger, genesis, filler =
           king's action, `ls -la`, repeat-last; panel sides) by the same cascade
           against the target classes                             -> scores.jsonl
  truth    target accuracy where an environment grade exists (30 deliberated states
           with k=10 glm-5.3 samples + labels; harvest N=8 tables)   -> in report
  ceiling  frontier-vs-teacher margin per bench axis (kingboard matrix teacher row vs
           published GLM-5.3 / DeepSeek-V4 numbers) + the traces' glm_* vs teacher_*
           solve rates on identical tasks (/tmp/vr/rollouts.jsonl)   -> in report
  report   report.txt + report.json

  cd /workspace && source .venv/bin/activate
  python research/scripts/frontier_arbiter/ft_match_test.py all [--budget 55]

Terms (one line each)
  turn / prefix    the messages the agent saw before one reply in a stored trajectory (x)
  dialect          the action grammar of the turn: bash fence / <tool_call> / terminus JSON / text (whole visible reply)
  frontier sample  one glm-5.3 reply to the prefix, acting as the agent (long reasoning), parsed to its action y
  class            an equivalence class of actions = "the same decision" (norm-exact, body-Jaccard >= 0.5, or judge YES)
  target set       the frontier's classes at a turn with their sample shares; modal = largest share
  T (FT-match)     weight of the class the candidate's action joins: modal 1.0, minority 0.5 x share, unmatched 0
  credited         the candidate's action joined some frontier class (T > 0)
  agreement        the teacher's ref action falls in the frontier's MODAL class
  excess           mean over a model's turns of (T_model - mean T of the 3 teacher refs on the same turn)
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import hashlib
import json
import math
import random
import re
import statistics as st
import sys
import time
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import deliberated_oracle as DO  # noqa: E402
import panel as P  # noqa: E402
import vav_sim as VS  # noqa: E402

RES = C.REPO / "research" / "results" / "frontier_arbiter"
OUT = RES / "ft_match"
TURNS = OUT / "turns.jsonl.gz"
SAMPLES = OUT / "samples.jsonl"
GENESIS = OUT / "genesis.jsonl"
VERDICTS = OUT / "verdicts.jsonl"
TARGETS = OUT / "targets.jsonl"
SCORES = OUT / "scores.jsonl"
COST = OUT / "cost.jsonl"
SETA = RES / "iprm" / "setA_turns.jsonl"
PANEL_TURNS = RES / "panel" / "turns.jsonl.gz"
PANEL_META = RES / "panel" / "meta.json"
PANEL_GENESIS = RES / "panel" / "frontier" / "qwen3.6-35b-a3b.jsonl"
HARVEST = RES / "harvest_n8" / "tables.jsonl"
ROLLOUTS_INDEX = Path("/tmp/vr/rollouts.jsonl")

FRONTIER = "glm-5.3"
FALLBACK = "glm-5.3-flash"
JUDGE = "glm-5.3-flash"
DIALECTS = ("bash", "tool_call", "terminus_json", "text")
PER_DIALECT = 50
K_A = 5
K_P = 3
PER_KING = 25
GENESIS_TURNS = 100
SAMPLE_TEMP = 0.8
MAX_TOKENS = 8192
JUDGE_MAX_TOKENS = 3000
JACCARD_MIN = VS.JACCARD_MIN
MINORITY_W = 0.5
FOLD_TURNS_PER_DAY = 6_500
D_TURNS = 635_000
MODEL_LABEL = {d: f"king{r}" for r, (d, _, _) in P.KINGS.items()}
MODEL_LABEL.update({"genesis": "genesis", "teacher": "teacher"})
FOREIGN_FENCE_RE = re.compile(r"```mswea_bash_command[ \t]*\n")
TOOLS_BLOCK_RE = re.compile(r"<tools>\n(.*?)\n</tools>", re.S)
# Engy's tool parser mangles GLM's XML tool calls when no `tools` are passed
# ("<tool_call>Bash>\ncommand</arg_key>…</arg_value>…</tool_call>",
#  "<tool_call>Write<arg_key>file_path=…\n<parameter=content>…</parameter>\n</function>");
# tool_call turns therefore pass the prefix's own tool schemas so the reply comes
# back structured, and the text form is recovered best-effort as a fallback.
MANGLED_TC_RE = re.compile(r"<tool_call>\s*(?:<function=)?([A-Za-z_][\w.-]*)>?(.*?)(?:</tool_call>|</function>|\Z)", re.S)
ARG_PATTERNS = (
    re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S),
    re.compile(r"<arg_key>([\w.-]+)</arg_key>\s*<arg_value>(.*?)</arg_value>", re.S),
    re.compile(r"(?:^|\n|<arg_key>)([\w.-]+)(?:</arg_key>|>|=)(.*?)</arg_value>", re.S),
    re.compile(r"<arg_key>([\w.-]+)=(.*?)(?=\n<parameter=|\n<arg_key>|</|\Z)", re.S),
)


def tools_of(turn: dict) -> list[dict]:
    """Tool schemas baked into the prefix's system message (teacher template)."""
    if turn["kind"] != "tool_call" or not turn["prefix"] or turn["prefix"][0]["role"] != "system":
        return []
    m = TOOLS_BLOCK_RE.search(turn["prefix"][0]["content"] or "")
    out = []
    for line in (m.group(1).split("\n") if m else []):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if isinstance(d, dict) and d.get("type") == "function":
            out.append(d)
    return out


def clean_tool_calls(calls: list[dict]) -> list[dict]:
    out = []
    for c in calls:
        fn = dict(c.get("function") or {})
        name = str(fn.get("name") or "")
        name = re.sub(r"^<?function=", "", name).strip("<> \n")
        fn["name"] = name
        out.append({**c, "function": fn})
    return out


def is_canonical_tool_call(y: str) -> bool:
    """True when the <tool_call> body is JSON with a name or the <function=…> XML form."""
    m = re.search(r"<tool_call>\s*(.*?)\s*</tool_call>", VS.canon_tool_call(y), re.S)
    if not m:
        return False
    try:
        d = json.loads(m.group(1))
        return isinstance(d, dict) and bool(d.get("name"))
    except ValueError:
        return False


def recover_mangled(content: str) -> str:
    """Best-effort: rebuild one JSON <tool_call> from Engy-mangled XML text."""
    if "<tool_call>" not in content or re.search(r"<tool_call>\s*\{", content):
        return content
    m = MANGLED_TC_RE.search(content)
    if not m:
        return content
    name, body = m.group(1), m.group(2)
    args: dict[str, str] = {}
    for pat in ARG_PATTERNS:
        for k, v in pat.findall(body):
            if k not in args and k != name:
                args[k] = v.strip()
    if not args and not body.strip():
        return content
    head = content[: m.start()]
    return head + "\n" + C.render_tool_call({"function": {"name": name, "arguments": args or {"_raw": body.strip()[:2000]}}})

# ------------------------------------------------------------------ public bench numbers for the ceiling table
# Filled from web sources on 2026-09-22 (see `ceiling` stage); None = not found.
PUBLIC_BENCH: dict[str, dict[str, tuple[float | None, str]]] = {}
PUBLIC_BENCH_PATH = OUT / "public_bench.json"


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def log_cost(stage: str, engy: C.Engy, note: str = "") -> float:
    C.append_jsonl(COST, {"at": time.time(), "stage": stage, "run": id(engy), "cost_usd": engy.cost_usd,
                          "usage": engy.usage, "note": note})
    tot = total_cost()
    print(f"  [$] {stage}: run ${engy.cost_usd:.3f} | total ${tot:.2f} {note}", flush=True)
    return tot


def total_cost() -> float:
    runs: dict[tuple, float] = collections.defaultdict(float)
    for r in C.read_jsonl(COST):
        runs[(r["stage"], r["run"])] = max(runs[(r["stage"], r["run"])], r["cost_usd"])
    return sum(runs.values())


# ------------------------------------------------------------------ actions
def parse_reply(reply: dict, kind: str) -> dict:
    """(z, y) of a frontier / genesis reply in the turn's dialect; the
    mini-swe fence and a two-object terminus reply are recovered for
    LABELLING (a miner would forfeit them under the live parser)."""
    r = dict(reply)
    content = r.get("content") or ""
    fence_fixed = False
    recovered = False
    if kind == "bash" and FOREIGN_FENCE_RE.search(content):
        content = FOREIGN_FENCE_RE.sub("```bash\n", content)
        fence_fixed = True
    if kind == "tool_call":
        r["tool_calls"] = clean_tool_calls(r.get("tool_calls") or [])
        if not r["tool_calls"]:
            fixed, _ = C.repair_tool_call(content)
            y0 = C.dialects.split_action(fixed, kind)[1]
            if not y0 or not is_canonical_tool_call(y0):
                rec = recover_mangled(content)
                recovered = rec != content
                content = rec
    r["content"] = content
    out = C.reply_to_rollout(r, kind)
    out["fence_fixed"] = fence_fixed
    out["recovered"] = recovered
    out["fallback_parse"] = False
    if kind == "terminus_json" and not out["y"]:
        fb = DO.terminus_fallback(content)
        if fb:
            out["y"], out["parsed"], out["fallback_parse"] = fb, True, True
    if out["y"]:
        out["y"] = VS.canon(out["y"], kind)
    return out


def norm(y: str, kind: str) -> str:
    return C.norm_action(VS.canon(y, kind), kind)


def bj(a: str, b: str, kind: str) -> float:
    return VS.body_jaccard(a, b, kind)


# ------------------------------------------------------------------ judge (glm-5.3-flash T0, "same decision?", cached)
class Judge:
    def __init__(self, engy: C.Engy | None):
        self.engy = engy
        self.cache: dict[str, dict] = {}
        for r in C.read_jsonl(VERDICTS):
            if r.get("same") is not None:
                self.cache[r["key"]] = r
        self.calls = 0
        self.missing = 0

    @staticmethod
    def key(turn_id: str, a: str, b: str, kind: str) -> str:
        na, nb = sorted([norm(a, kind), norm(b, kind)])
        return hashlib.sha256(f"{turn_id}\x00{na}\x00{nb}".encode()).hexdigest()[:24]

    async def same(self, turn: dict, a: str, b: str) -> bool | None:
        kind = turn["kind"]
        k = self.key(turn["turn_id"], a, b, kind)
        if k in self.cache:
            return self.cache[k]["same"]
        if self.engy is None:
            self.missing += 1
            return None
        ctx = VS.context_text({"messages": turn["prefix"], "action_kind": kind, "harness": turn.get("source")})
        user = (f"{ctx}\n\n--- Candidate action A ---\n{a[:VS.ACTION_CAP]}\n\n--- Candidate action B ---\n{b[:VS.ACTION_CAP]}\n\n"
                "Do A and B make the SAME DECISION at this point? One word: YES or NO.")
        msgs = [{"role": "system", "content": VS.EQUIV_SYSTEM}, {"role": "user", "content": user}]
        same, finish, cost, nothink = None, None, 0.0, False
        for attempt in range(2):
            try:
                r = await self.engy.chat(JUDGE, msgs, temperature=0.0, max_tokens=JUDGE_MAX_TOKENS,
                                         **(VS.NOTHINK if nothink else {}))
            except Exception as e:  # noqa: BLE001 — a judge failure leaves the pair unmatched
                C.append_jsonl(VERDICTS, {"key": k, "turn_id": turn["turn_id"], "a": a[:300], "b": b[:300],
                                          "same": None, "error": repr(e)[:200]})
                return None
            self.calls += 1
            cost += r["cost_usd"]
            finish = r["finish"]
            w = VS.one_word(r["content"], ("YES", "NO"))
            if w is not None and finish == "stop":
                same = w == "YES"
                break
            nothink = True        # thinking hit the cap: answer without thinking
        rec = {"key": k, "turn_id": turn["turn_id"], "a": a[:300], "b": b[:300], "same": same, "finish": finish,
               "cost_usd": cost, "nothink": nothink}
        if same is not None:
            self.cache[k] = rec
        C.append_jsonl(VERDICTS, rec)
        return same


# ------------------------------------------------------------------ classes + matching (the cascade)
class UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, i: int) -> int:
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            self.p[max(ri, rj)] = min(ri, rj)


async def cluster(turn: dict, ys: list[str], judge: Judge) -> tuple[list[list[int]], int, int]:
    """Classes over ys (indices) by the cascade; returns (classes largest-first, n_judged, n_missing)."""
    kind = turn["kind"]
    n = len(ys)
    uf = UF(n)
    norms = [norm(y, kind) for y in ys]
    for i in range(n):
        for j in range(i + 1, n):
            if norms[i] == norms[j] or bj(ys[i], ys[j], kind) >= JACCARD_MIN:
                uf.union(i, j)
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    pre = sorted(groups.values(), key=lambda c: (-len(c), c[0]))
    merged: list[list[int]] = []
    n_judged = n_missing = 0
    for c in pre:
        joined = False
        for g in merged:
            same = await judge.same(turn, ys[g[0]], ys[c[0]])
            n_judged += 1
            if same is None:
                n_missing += 1
            elif same:
                g.extend(c)
                joined = True
                break
        if not joined:
            merged.append(list(c))
    merged.sort(key=lambda c: (-len(c), c[0]))
    return merged, n_judged, n_missing


def weights_of(classes: list[list[int]], n_parsed: int) -> list[dict]:
    top = len(classes[0]) if classes else 0
    out = []
    for ci, c in enumerate(classes):
        share = len(c) / n_parsed
        modal = len(c) == top
        out.append({"idx": ci, "members": c, "n": len(c), "share": share, "modal": modal,
                    "weight": 1.0 if modal else MINORITY_W * share})
    return out


async def match(turn: dict, y: str, tgt: dict, judge: Judge) -> dict:
    """Class assignment of candidate y by the cascade against the turn's target set."""
    kind = turn["kind"]
    if not y:
        return {"how": "forfeit", "class": None, "T": 0.0, "best_jaccard": 0.0}
    ny = norm(y, kind)
    ys = tgt["ys"]
    best = (0.0, None)
    for cl in tgt["classes"]:
        for i in cl["members"]:
            if norm(ys[i], kind) == ny:
                return {"how": "exact", "class": cl["idx"], "T": cl["weight"], "best_jaccard": 1.0}
            j = bj(y, ys[i], kind)
            if j > best[0]:
                best = (j, cl["idx"])
    if best[0] >= JACCARD_MIN:
        cl = tgt["classes"][best[1]]
        return {"how": "jaccard", "class": cl["idx"], "T": cl["weight"], "best_jaccard": best[0]}
    missing = 0
    for cl in tgt["classes"]:
        same = await judge.same(turn, ys[cl["members"][0]], y)
        if same is None:
            missing += 1
        elif same:
            return {"how": "judge", "class": cl["idx"], "T": cl["weight"], "best_jaccard": best[0]}
    return {"how": "none" if not missing else "none_missing", "class": None, "T": 0.0, "best_jaccard": best[0],
            "judge_missing": missing}


# ------------------------------------------------------------------ select
def load_turns() -> list[dict]:
    return C.read_jsonl(TURNS)


def cmd_select(args) -> int:
    if TURNS.exists() and not args.force:
        print(f"{TURNS} exists ({len(load_turns())} turns); --force to rebuild")
        return 0
    rng = random.Random(0)
    rows = C.read_jsonl(SETA)
    by: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by[(r["dialect"], r["record"])].append(r)
    picked: list[dict] = []
    per_rec = PER_DIALECT // len({r["record"] for r in rows})
    for (d, rec), rs in sorted(by.items()):
        rs = sorted(rs, key=lambda r: r["turn_id"])
        rng.shuffle(rs)
        picked += rs[:per_rec]
    # genesis subsample: 25 per dialect
    gen: set[str] = set()
    for d in DIALECTS:
        ds = sorted([r["turn_id"] for r in picked if r["dialect"] == d])
        rng.shuffle(ds)
        gen |= set(ds[: GENESIS_TURNS // len(DIALECTS)])
    out: list[dict] = []
    for rec in sorted({r["record"] for r in picked}):
        tids = [r["turn_id"] for r in picked if r["record"] == rec]
        print(f"materializing {len(tids)} turns of {rec}", flush=True)
        mats = C.materialize(rec, tids)
        for r in picked:
            if r["record"] != rec:
                continue
            m = mats.get(r["turn_id"])
            if not m:
                print(f"  ! {r['turn_id']} not materialized", flush=True)
                continue
            cands = [{"role": c["role"], "model": c.get("model"), "y": VS.canon(c["y"], r["dialect"])}
                     for c in r["candidates"]]
            out.append({"set": "A", "turn_id": r["turn_id"], "record": r["record"], "kind": r["dialect"],
                        "source": r["source"], "group": r["group"], "depth": r["depth"],
                        "n_prefix_chars": r["n_prefix_chars"], "k": K_A, "genesis": r["turn_id"] in gen,
                        "king_model": r["king_model"], "challenger_model": r["challenger_model"],
                        "prefix": m["prefix"], "candidates": cands})
    # panel
    pturns = {t["turn_id"]: t for t in C.read_jsonl(PANEL_TURNS)}
    meta = json.loads(PANEL_META.read_text())
    gsamp = {r["turn_id"]: r for r in C.read_jsonl(PANEL_GENESIS) if r.get("parsed")}
    gset = set(meta["genesis_turns"])
    for reign, (digest, crowning, _) in sorted(P.KINGS.items()):
        cands = [t for t in pturns.values() if t["record"] == crowning and t["turn_id"] in gset]
        cands.sort(key=lambda t: (t["turn_id"] not in gsamp, t["turn_id"]))
        with_g = [t for t in cands if t["turn_id"] in gsamp]
        rng.shuffle(with_g)
        rest = [t for t in cands if t["turn_id"] not in gsamp]
        rng.shuffle(rest)
        sel = (with_g + rest)[:PER_KING]
        for t in sel:
            cc = [{"role": f"ref_{i}", "model": "teacher", "y": VS.canon(r["y"], t["kind"])} for i, r in enumerate(t["refs"])]
            for side in ("king", "challenger"):
                s = t["sides"].get(side)
                if s and s.get("y"):
                    cc.append({"role": side, "model": s["model"], "y": VS.canon(s["y"], t["kind"])})
            g = gsamp.get(t["turn_id"])
            if g:
                cc.append({"role": "genesis", "model": "genesis", "y": VS.canon(g["y"], t["kind"])})
            out.append({"set": "P", "turn_id": t["turn_id"], "record": t["record"], "reign": reign, "kind": t["kind"],
                        "source": t["source"], "group": t["group"], "depth": t["depth"],
                        "n_prefix_chars": t["n_prefix_chars"], "k": K_P, "genesis": bool(g),
                        "king_model": t["sides"]["king"]["model"] if t["sides"].get("king") else None,
                        "challenger_model": digest, "prefix": t["prefix"], "candidates": cc})
    OUT.mkdir(parents=True, exist_ok=True)
    with gzip.open(TURNS, "wt") as f:
        for t in out:
            f.write(json.dumps(t) + "\n")
    a = [t for t in out if t["set"] == "A"]
    p = [t for t in out if t["set"] == "P"]
    print(f"selected {len(a)} Set-A turns {dict(collections.Counter(t['kind'] for t in a))} "
          f"(genesis on {sum(t['genesis'] for t in a)}) + {len(p)} panel turns "
          f"{dict(collections.Counter(t['kind'] for t in p))} (genesis on {sum(t['genesis'] for t in p)})")
    return 0


# ------------------------------------------------------------------ sample
class Sampler:
    def __init__(self, engy: C.Engy):
        self.engy = engy

    async def chat(self, model: str, messages: list[dict], **kw) -> dict:
        try:
            return await self.engy.chat(model, messages, **kw)
        except RuntimeError as e:
            if model != FRONTIER:
                raise
            print(f"  ! {model} failed ({str(e)[:90]}); falling back to {FALLBACK}", flush=True)
            r = await self.engy.chat(FALLBACK, messages, **kw)
            r["fallback"] = True
            return r


def sample_row(turn: dict, idx: int, reply: dict, model_req: str) -> dict:
    a = parse_reply(reply, turn["kind"])
    u = reply.get("usage") or {}
    return {"set": turn["set"], "turn_id": turn["turn_id"], "idx": idx, "kind": turn["kind"],
            "model_requested": model_req, "model": reply.get("model"), "fallback": bool(reply.get("fallback")),
            "y": a["y"], "z_chars": len(a["z"] or ""), "parsed": a["parsed"], "fence_fixed": a["fence_fixed"],
            "fallback_parse": a["fallback_parse"], "repaired": a.get("repaired"), "recovered": a.get("recovered"),
            "finish": reply.get("finish"), "n_tool_calls": len(reply.get("tool_calls") or []),
            "reasoning_chars": len(reply.get("reasoning") or ""), "content_chars": len(reply.get("content") or ""),
            "prompt_tokens": int(u.get("prompt_tokens") or 0), "completion_tokens": int(u.get("completion_tokens") or 0),
            "cached_tokens": int(((u.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0),
            "reasoning_tokens": int(((u.get("completion_tokens_details") or {}).get("reasoning_tokens")) or 0),
            "cost_usd": float(reply.get("cost_usd") or 0), "raw_content": (reply.get("content") or "")[:2000]}


async def cmd_sample(args) -> int:
    turns = load_turns()
    if args.set:
        turns = [t for t in turns if t["set"] in args.set]
    have = collections.defaultdict(set)
    for r in C.read_jsonl(SAMPLES):
        have[r["turn_id"]].add(r["idx"])
    engy = C.Engy(concurrency=args.concurrency, timeout=1500, retries=6)
    smp = Sampler(engy)
    sem = asyncio.Semaphore(args.parallel_turns)
    t0 = time.time()
    done = {"n": 0}
    todo = [t for t in turns if len(have[t["turn_id"]]) < t["k"]]
    print(f"sampling {len(todo)} turns ({sum(t['k'] - len(have[t['turn_id']]) for t in todo)} calls); "
          f"banked ${total_cost():.2f}", flush=True)
    stop = {"v": False}

    async def one(turn: dict) -> None:
        async with sem:
            if stop["v"]:
                return
            need = [i for i in range(turn["k"]) if i not in have[turn["turn_id"]]]
            if not need:
                return
            rows = []
            # first sample alone so the prefix is cached for the rest
            first, rest = need[0], need[1:]
            tk = {"tools": tools_of(turn)} if tools_of(turn) else {}
            try:
                r = await smp.chat(FRONTIER, turn["prefix"], temperature=SAMPLE_TEMP, max_tokens=MAX_TOKENS, **tk)
                rows.append(sample_row(turn, first, r, FRONTIER))
            except Exception as e:  # noqa: BLE001
                print(f"  ! {turn['turn_id'][:30]} i={first}: {repr(e)[:120]}", flush=True)
            if rest:
                res = await asyncio.gather(*[smp.chat(FRONTIER, turn["prefix"], temperature=SAMPLE_TEMP, max_tokens=MAX_TOKENS, **tk)
                                             for _ in rest], return_exceptions=True)
                for i, r in zip(rest, res):
                    if isinstance(r, Exception):
                        print(f"  ! {turn['turn_id'][:30]} i={i}: {repr(r)[:120]}", flush=True)
                    else:
                        rows.append(sample_row(turn, i, r, FRONTIER))
            for row in rows:
                C.append_jsonl(SAMPLES, row)
            done["n"] += 1
            tot = total_cost() + engy.cost_usd
            print(f"  [{turn['set']}] {done['n']}/{len(todo)} {turn['kind']:13s} {turn['turn_id'][:28]} parsed "
                  f"{sum(r['parsed'] for r in rows)}/{len(rows)} fb {sum(r['fallback'] for r in rows)} "
                  f"${sum(r['cost_usd'] for r in rows):.3f} {time.time() - t0:.0f}s (total ${tot:.2f})", flush=True)
            if tot > args.budget:
                stop["v"] = True
                print(f"BUDGET ${args.budget} reached; stopping", flush=True)

    await asyncio.gather(*[one(t) for t in todo])
    log_cost("sample", engy, f"{done['n']} turns")
    # genesis on the Set-A subsample (duel cap, T=0.8)
    if not args.set or "A" in args.set:
        ghave = {r["turn_id"] for r in C.read_jsonl(GENESIS)}
        gtodo = [t for t in turns if t["set"] == "A" and t["genesis"] and t["turn_id"] not in ghave]
        print(f"genesis: {len(gtodo)} calls", flush=True)
        engy2 = C.Engy(concurrency=args.concurrency, timeout=900, retries=6)

        async def gone(turn: dict) -> None:
            try:
                tk = {"tools": tools_of(turn)} if tools_of(turn) else {}
                r = await engy2.chat(C.GENESIS_ENGY, turn["prefix"], temperature=SAMPLE_TEMP, max_tokens=C.DUEL_MAX_TOKENS, **tk)
                C.append_jsonl(GENESIS, sample_row(turn, 0, r, C.GENESIS_ENGY))
            except Exception as e:  # noqa: BLE001
                C.append_jsonl(GENESIS, {"turn_id": turn["turn_id"], "idx": 0, "kind": turn["kind"], "y": "",
                                         "parsed": False, "error": repr(e)[:200], "cost_usd": 0.0})
        await asyncio.gather(*[gone(t) for t in gtodo])
        log_cost("genesis", engy2, f"{len(gtodo)} calls")
    return 0


# ------------------------------------------------------------------ targets
def load_samples() -> dict[str, list[dict]]:
    by = collections.defaultdict(list)
    for r in C.read_jsonl(SAMPLES):
        by[r["turn_id"]].append(r)
    for v in by.values():
        v.sort(key=lambda r: r["idx"])
    return by


async def cmd_targets(args) -> int:
    turns = load_turns()
    samples = load_samples()
    engy = C.Engy(concurrency=args.concurrency, timeout=600, retries=6)
    judge = Judge(engy)
    have = {r["turn_id"] for r in C.read_jsonl(TARGETS)}
    if args.set:
        turns = [t for t in turns if t["set"] in args.set]
    # cluster only turns whose sampling is complete (all k replies stored, or an error-free partial when --force)
    todo = [t for t in turns if t["turn_id"] not in have and samples.get(t["turn_id"])
            and (len(samples[t["turn_id"]]) >= t["k"] or args.force)]
    print(f"targets: {len(todo)} turns to cluster (verdict cache {len(judge.cache)})", flush=True)
    sem = asyncio.Semaphore(args.parallel_turns)
    n = {"v": 0}

    async def one(turn: dict) -> None:
        async with sem:
            rows = samples[turn["turn_id"]]
            parsed = [r for r in rows if r["parsed"] and r["y"]]
            ys = [r["y"] for r in parsed]
            classes, nj, nm = await cluster(turn, ys, judge)
            rec = {"set": turn["set"], "turn_id": turn["turn_id"], "kind": turn["kind"], "n_total": len(rows),
                   "n_parsed": len(ys), "n_cap": sum(r["finish"] == "length" for r in rows),
                   "n_fallback_model": sum(r["fallback"] for r in rows),
                   "n_fallback_parse": sum(r["fallback_parse"] for r in rows),
                   "cost_usd": sum(r["cost_usd"] for r in rows), "judge_calls": nj, "judge_missing": nm,
                   "ys": ys, "sample_idx": [r["idx"] for r in parsed],
                   "classes": weights_of(classes, len(ys)) if ys else []}
            C.append_jsonl(TARGETS, rec)
            n["v"] += 1
            if n["v"] % 20 == 0:
                print(f"  {n['v']}/{len(todo)} clustered; judge calls {judge.calls} ${engy.cost_usd:.3f}", flush=True)

    await asyncio.gather(*[one(t) for t in todo])
    log_cost("targets_judge", engy, f"{judge.calls} judge calls")
    return 0


def load_targets() -> dict[str, dict]:
    return {r["turn_id"]: r for r in C.read_jsonl(TARGETS)}


# ------------------------------------------------------------------ score
def candidates_of(turn: dict, genesis: dict[str, dict]) -> list[dict]:
    cands = list(turn["candidates"])
    if turn["set"] == "A":
        g = genesis.get(turn["turn_id"])
        if g is not None:
            cands.append({"role": "genesis", "model": "genesis", "y": g.get("y") or "", "parsed": g.get("parsed")})
        king = next((c for c in cands if c["role"] == "king"), None)
        if king:
            cands.append({"role": "filler", "model": king["model"], "y": king["y"]})
    return cands


async def cmd_score(args) -> int:
    turns = load_turns()
    targets = load_targets()
    genesis = {r["turn_id"]: r for r in C.read_jsonl(GENESIS)}
    engy = C.Engy(concurrency=args.concurrency, timeout=600, retries=6)
    judge = Judge(engy)
    # keyed by (turn, role, model): 4 panel turns are also Set-A turns, with different models on the same roles
    have = {(r["turn_id"], r["role"], r.get("model")) for r in C.read_jsonl(SCORES)}
    sem = asyncio.Semaphore(args.parallel_turns)
    n = {"v": 0}
    if args.set:
        turns = [t for t in turns if t["set"] in args.set]
    todo = [t for t in turns if t["turn_id"] in targets]
    print(f"score: {len(todo)} turns", flush=True)

    async def one(turn: dict) -> None:
        async with sem:
            tgt = targets[turn["turn_id"]]
            for c in candidates_of(turn, genesis):
                if (turn["turn_id"], c["role"], c.get("model")) in have:
                    continue
                m = await match(turn, c["y"], tgt, judge) if tgt["classes"] else \
                    {"how": "no_targets", "class": None, "T": 0.0, "best_jaccard": 0.0}
                C.append_jsonl(SCORES, {"set": turn["set"], "turn_id": turn["turn_id"], "kind": turn["kind"],
                                        "group": turn["group"], "role": c["role"], "model": c.get("model"),
                                        "y": (c["y"] or "")[:400], **m})
            n["v"] += 1
            if n["v"] % 25 == 0:
                print(f"  {n['v']}/{len(todo)} scored; judge calls {judge.calls} ${engy.cost_usd:.3f}", flush=True)

    await asyncio.gather(*[one(t) for t in todo])
    log_cost("score_judge", engy, f"{judge.calls} judge calls")
    return 0


# ------------------------------------------------------------------ truth (task 4)
async def cmd_truth(args) -> dict:
    """Target accuracy at the env-graded states. Uses the deliberated k=10 glm-5.3
    samples (v1) + labelled first actions (split_states / outcome / harvest N=8
    tables) — no new continuations."""
    states = DO.load_states("all")
    rows = C.read_jsonl(DO.SAMPLES)
    by_state = collections.defaultdict(list)
    for r in rows:
        if r["variant"] == "v1" and r["parsed"] and r["y"]:
            by_state[r["state_id"]].append(r)
    harvest = {t["sid"]: t for t in C.read_jsonl(HARVEST)}
    engy = C.Engy(concurrency=args.concurrency, timeout=600, retries=6)
    judge = DO.Judge(engy)
    per: list[dict] = []
    sem = asyncio.Semaphore(args.parallel_turns)

    async def one(stt: dict) -> None:
        async with sem:
            sid = stt["state_id"]
            base = sid.rsplit(":", 1)[0]
            kind = stt["kind"]
            labelled = list(stt["labelled"])
            have = {(C.norm_action(l["y"], kind), l["who"], l["outcome"]) for l in labelled}
            n_added = 0
            hv = harvest.get(base)
            if hv:
                for r in hv["table"]:
                    who = "frontier_proposal" if r["src"] == "F" else "teacher"
                    key = (C.norm_action(r["y"], kind), who, r["outcome"])
                    if key in have or not r["y"]:
                        continue
                    have.add(key)
                    labelled.append({"y": r["y"], "outcome": r["outcome"], "who": who, "src": f"harvest:{r['arm']}"})
                    n_added += 1
            fs = by_state.get(sid, [])
            items = [{"y": r["y"], "tag": "sample"} for r in fs] + [{"y": l["y"], "tag": "labelled", **l} for l in labelled]
            pool = DO.Pool(stt, items)
            pool.tier1()
            pool.tier2()
            stj = json.loads(Path(stt["path"]).read_text())
            nj = await pool.tier3(judge, stj)
            labels = pool.snapshot()
            f_idx = [i for i, it in enumerate(items) if it["tag"] == "sample"]
            t_idx = [i for i, it in enumerate(items) if it["tag"] == "labelled" and it["who"] == "teacher"]
            lab_idx = [i for i, it in enumerate(items) if it["tag"] == "labelled"]

            def class_outcome(lab: int) -> str:
                mem = [items[i] for i in range(len(items)) if labels[i] == lab and items[i]["tag"] == "labelled"]
                s = any(m["outcome"] == "solved" for m in mem)
                f = any(m["outcome"] == "failed" for m in mem)
                return "mixed" if s and f else "correct" if s else "wrong" if f else "novel"

            def class_rate(lab: int) -> float | None:
                mem = [items[i] for i in range(len(items)) if labels[i] == lab and items[i]["tag"] == "labelled"]
                return (sum(m["outcome"] == "solved" for m in mem) / len(mem)) if mem else None

            rec = {"state_id": sid, "class": stt["class"], "harness": stt["harness"], "kind": kind, "depth": stt["depth"],
                   "n_frontier": len(f_idx), "n_labelled": len(lab_idx), "n_teacher_labelled": len(t_idx),
                   "n_harvest_added": n_added, "judge_calls": nj,
                   "any_solved_label": any(items[i]["outcome"] == "solved" for i in lab_idx)}
            if f_idx:
                fm = collections.Counter(labels[i] for i in f_idx).most_common()
                modal = fm[0][0]
                rec["frontier_modal_share"] = fm[0][1] / len(f_idx)
                rec["frontier_n_classes"] = len(fm)
                rec["frontier_modal_outcome"] = class_outcome(modal)
                rec["frontier_modal_rate"] = class_rate(modal)
                rec["frontier_any_class_solved"] = any(class_outcome(l) in ("correct", "mixed") for l, _ in fm)
                # a random frontier sample's class solve rate (expected T-weighted truth)
                rates = [class_rate(labels[i]) for i in f_idx]
                rates = [x for x in rates if x is not None]
                rec["frontier_random_rate"] = st.mean(rates) if rates else None
            if t_idx:
                rec["teacher_random_rate"] = sum(items[i]["outcome"] == "solved" for i in t_idx) / len(t_idx)
                tm = collections.Counter(labels[i] for i in t_idx).most_common()
                rec["teacher_modal_share"] = tm[0][1] / len(t_idx)
                rec["teacher_modal_outcome"] = class_outcome(tm[0][0])
                rec["teacher_modal_rate"] = class_rate(tm[0][0])
                rec["teacher_any_class_solved"] = any(class_outcome(l) in ("correct", "mixed") for l, _ in tm)
            per.append(rec)

    await asyncio.gather(*[one(s) for s in states if by_state.get(s["state_id"])])
    log_cost("truth_judge", engy, f"{judge.n_calls} judge calls")
    per.sort(key=lambda r: r["state_id"])
    # harvest-only view (38 states, no deliberation): single F1 greedy proposal vs T
    hv_rows = C.read_jsonl(HARVEST)
    hv = {"n_states": len(hv_rows),
          "T_solved": sum(t["s"] for t in hv_rows), "T_n": sum(t["n"] for t in hv_rows),
          "F_solved": sum(t["s_F"] for t in hv_rows), "F_n": sum(t["n_F"] for t in hv_rows),
          "X_solved": sum(t.get("s_X", 0) for t in hv_rows), "X_n": sum(t.get("n_X", 0) for t in hv_rows),
          "states_with_any_solve": sum(1 for t in hv_rows if t["s"] > 0 or t["s_F"] > 0 or t.get("s_X", 0) > 0),
          "states_F_solved_where_T_zero": sum(1 for t in hv_rows if t["s"] == 0 and t["s_F"] > 0),
          "states_T_zero": sum(1 for t in hv_rows if t["s"] == 0)}
    return {"per_state": per, "harvest_only": hv}


# ------------------------------------------------------------------ ceiling (task 5)
BENCH_MAP = {   # report name -> matrix key
    "SWE-bench Verified": "swebench-verified", "Terminal-Bench 2": "terminal-bench-2",
    "tau2-airline": "tau2-airline", "tau2-retail": "tau2-retail", "tau2-telecom": "tau2-telecom",
    "BFCL v3": "bfcl-v3", "LiveCodeBench": "livecodebench", "GPQA Diamond": "gpqa-diamond",
    "AIME 2025": "aime25", "IFBench": "ifbench",
}


def matrix_rows() -> dict[str, dict[str, float]]:
    cache = OUT / "matrix.json"
    if cache.exists():
        m = json.loads(cache.read_text())
    else:
        m = httpx.get(P.MATRIX_URL, timeout=60).json()
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(m))
    out = {}
    for row in m["rows"]:
        key = row["kind"] if row["kind"] in ("teacher", "genesis") else (row.get("digest12") or row["label"])
        cells = {k[6:]: v.get("score") for k, v in (row.get("cells") or {}).items()
                 if k.startswith("bench:") and isinstance(v, dict) and v.get("score") is not None}
        out[key] = {"label": row["label"], "cells": cells}
    out["_generated_at"] = m.get("generated_at")
    return out


def traces_join() -> dict:
    """glm_* vs teacher_* solve rates on identical (source, sid, harness) tasks from the
    local rollout index (verified_refs stage; traces manifest 2026-09-20)."""
    if not ROLLOUTS_INDEX.exists():
        return {"missing": str(ROLLOUTS_INDEX)}
    by: dict[tuple, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    fam_counts = collections.Counter()
    glm_models = collections.Counter()
    unpaired: dict[tuple, dict[str, list[int]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    with open(ROLLOUTS_INDEX) as f:
        for line in f:
            r = json.loads(line)
            pid = r.get("policy_id") or ""
            fam = "glm" if pid.startswith("glm_") else "teacher" if pid.startswith("teacher_") else None
            if fam is None or r.get("outcome") not in ("solved", "failed"):
                continue
            if fam == "glm":
                glm_models[r.get("model")] += 1
                if "glm-5" not in (r.get("model") or ""):
                    continue          # the glm_* seat also ran GLM-4.5-Air in the GLM-teacher era; frontier = GLM-5.x only
            fam_counts[fam] += 1
            by[(r["source"], r["sid"], r["harness"])][fam].append(int(r["outcome"] == "solved"))
            unpaired[(r["source"], r["harness"])][fam].append(int(r["outcome"] == "solved"))
    res: dict = {"families": dict(fam_counts), "glm_models_seen": dict(glm_models), "by_group": {}, "by_source": {}, "by_harness": {},
                 "unpaired_by_source": {}}
    # the datagen pool was shared before the seat scheduler (a task ran once, by whichever policy drew it),
    # so coding sources have no paired tasks; report the unpaired per-source rates as the weaker proxy
    for (src, harn), v in sorted(unpaired.items()):
        if len(v["glm"]) >= 30 and len(v["teacher"]) >= 30:
            g, t = st.mean(v["glm"]), st.mean(v["teacher"])
            res["unpaired_by_source"][f"{src}/{harn}"] = {
                "glm_p1": g, "glm_n": len(v["glm"]), "teacher_p1": t, "teacher_n": len(v["teacher"]), "diff": g - t,
                "se": math.sqrt(g * (1 - g) / len(v["glm"]) + t * (1 - t) / len(v["teacher"])), "group": P.group_of(src)}

    def agg(keys: list[tuple]) -> dict:
        g = [st.mean(by[k]["glm"]) for k in keys]
        t = [st.mean(by[k]["teacher"]) for k in keys]
        d = [a - b for a, b in zip(g, t)]
        return {"tasks": len(keys), "glm_p1": st.mean(g) if g else None, "teacher_p1": st.mean(t) if t else None,
                "diff": st.mean(d) if d else None, "se": (st.stdev(d) / math.sqrt(len(d))) if len(d) > 2 else None,
                "glm_rollouts": sum(len(by[k]["glm"]) for k in keys),
                "teacher_rollouts": sum(len(by[k]["teacher"]) for k in keys), **VS.sign_test(d)}
    paired = [k for k, v in by.items() if v["glm"] and v["teacher"]]
    res["all"] = agg(paired)
    groups = collections.defaultdict(list)
    sources = collections.defaultdict(list)
    harn = collections.defaultdict(list)
    for k in paired:
        groups[P.group_of(k[0])].append(k)
        sources[k[0]].append(k)
        harn[k[2]].append(k)
    res["by_group"] = {g: agg(ks) for g, ks in sorted(groups.items())}
    res["by_source"] = {s: agg(ks) for s, ks in sorted(sources.items()) if len(ks) >= 20}
    res["by_harness"] = {s: agg(ks) for s, ks in sorted(harn.items())}
    return res


def cmd_ceiling(args) -> dict:
    mx = matrix_rows()
    teacher = mx["teacher"]["cells"]
    pub = json.loads(PUBLIC_BENCH_PATH.read_text()) if PUBLIC_BENCH_PATH.exists() else {}
    table = []
    for name, key in BENCH_MAP.items():
        row = {"bench": name, "matrix_key": key, "teacher": teacher.get(key)}
        for model in ("glm-5.3", "deepseek-v4"):
            e = (pub.get(model) or {}).get(name) or {}
            row[model] = e.get("score")
            row[model + "_note"] = e.get("note")
            row[model + "_source"] = e.get("source")
            row[model + "_margin"] = (e["score"] - teacher[key]) if (e.get("score") is not None and teacher.get(key) is not None) else None
        table.append(row)
    return {"matrix_generated_at": mx["_generated_at"], "teacher_row": teacher, "public": pub, "table": table,
            "traces_join": traces_join(), "public_sources_note": pub.get("_note")}


# ------------------------------------------------------------------ report
def _m(v):
    v = [x for x in v if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return st.mean(v) if v else None


def _f(x, w=7, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    return f"{x:{w}.{p}f}"


def _pct(x, w=6):
    return " " * (w - 3) + "n/a" if x is None else f"{100 * x:{w}.1f}"


def cmd_report(args, truth: dict | None, ceiling: dict | None) -> int:
    turns = {t["turn_id"]: t for t in load_turns()}
    targets = load_targets()
    scores = C.read_jsonl(SCORES)
    samples = load_samples()
    L: list[str] = []
    J: dict = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "cost_usd_total": total_cost()}
    Pn = L.append
    Pn("FT-match killing test — frontier-authored target classes as the ranked quantity (2026-09-22)")
    Pn(f"generated {J['generated_at']}; Engy spend this probe ${J['cost_usd_total']:.2f}; frontier {FRONTIER} T={SAMPLE_TEMP} max_tokens {MAX_TOKENS}; "
       f"judge {JUDGE} T0; k={K_A} (Set A) / {K_P} (panel); minority weight {MINORITY_W} x share")
    Pn("Terms: turn = prefix x of a stored duel turn; class = same-decision equivalence class (norm-exact -> body-Jaccard>=0.5 -> judge YES); "
       "target set = frontier classes with sample shares; T = weight of the class the candidate's action joins (modal 1.0, minority 0.5*share, none 0); "
       "credited = T>0; agreement = the ref's action is in the frontier's modal class; excess = T_model - mean T of the teacher refs on the same turn; "
       "cap = reply hit max_tokens; parse = a dialect action was recovered.")
    Pn("")
    # ---------------- 1. targets
    Pn("== 1. TARGETS on Set-A duel turns (glm-5.3 deliberation as the agent on the prefix) ==")
    Pn(f"{'dialect':13s} {'turns':>5} {'samples':>7} {'parsed%':>7} {'cap%':>6} {'fb_model%':>9} {'fb_parse%':>9} {'cls/turn':>8} {'modal':>6} {'1-class%':>8} {'judge/turn':>10} {'$/turn':>7} {'$/sample':>8} {'tok_in':>7} {'tok_out':>7} {'cached%':>7} {'think%':>6} {'think_tok':>9}")
    tg: dict = {}
    for kind in DIALECTS + ("all",):
        ts = [targets[t] for t in targets if targets[t]["set"] == "A" and (kind == "all" or targets[t]["kind"] == kind)]
        if not ts:
            continue
        srows = [r for t in ts for r in samples.get(t["turn_id"], [])]
        n_s = len(srows)
        row = {"turns": len(ts), "samples": n_s,
               "parsed": _m([r["parsed"] for r in srows]), "cap": _m([r["finish"] == "length" for r in srows]),
               "fb_model": _m([r["fallback"] for r in srows]), "fb_parse": _m([r["fallback_parse"] for r in srows]),
               "classes_per_turn": _m([len(t["classes"]) for t in ts if t["n_parsed"]]),
               "modal_share": _m([t["classes"][0]["share"] for t in ts if t["classes"]]),
               "one_class": _m([len(t["classes"]) == 1 for t in ts if t["classes"]]),
               "judge_per_turn": _m([t["judge_calls"] for t in ts]),
               "usd_per_turn": _m([t["cost_usd"] for t in ts]), "usd_per_sample": (sum(r["cost_usd"] for r in srows) / n_s) if n_s else None,
               "tok_in": _m([r["prompt_tokens"] for r in srows]), "tok_out": _m([r["completion_tokens"] for r in srows]),
               "cached": (sum(r["cached_tokens"] for r in srows) / max(1, sum(r["prompt_tokens"] for r in srows))),
               "think_share": _m([r["reasoning_tokens"] > 0 for r in srows]),
               "think_tokens_when_thinking": _m([r["reasoning_tokens"] for r in srows if r["reasoning_tokens"] > 0]),
               "no_parsed_turns": sum(1 for t in ts if not t["n_parsed"])}
        tg[kind] = row
        Pn(f"{kind:13s} {row['turns']:5d} {n_s:7d} {_pct(row['parsed'], 7)} {_pct(row['cap'], 6)} {_pct(row['fb_model'], 9)} {_pct(row['fb_parse'], 9)} "
           f"{_f(row['classes_per_turn'], 8, 2)} {_f(row['modal_share'], 6, 2)} {_pct(row['one_class'], 8)} {_f(row['judge_per_turn'], 10, 1)} "
           f"{_f(row['usd_per_turn'], 7, 4)} {_f(row['usd_per_sample'], 8, 4)} {_f(row['tok_in'], 7, 0)} {_f(row['tok_out'], 7, 0)} {_pct(row['cached'], 7)} "
           f"{_pct(row['think_share'], 6)} {_f(row['think_tokens_when_thinking'], 9, 0)}")
    Pn("think% = share of glm-5.3 samples that used reasoning tokens (thinking is dynamic on GLM-5.3; 'long reasoning allowed' = max_tokens 8192, not forced); "
       "think_tok = mean reasoning tokens on those. Same regime as the deliberated-oracle v1 samples (median 0).")
    Pn("unparsed frontier samples are the frontier's own choice not to act in the dialect (final answer / prose without a call), not parser misses; "
       f"Set-A turns with zero parsed samples = {sum(v['no_parsed_turns'] for k, v in tg.items() if k != 'all')} (excluded from the score tables)")
    J["targets"] = tg
    # judge-cache spend on the targets/score stages
    vrows = C.read_jsonl(VERDICTS)
    J["judge"] = {"verdicts": len(vrows), "yes": sum(1 for r in vrows if r.get("same") is True),
                  "no": sum(1 for r in vrows if r.get("same") is False), "none": sum(1 for r in vrows if r.get("same") is None),
                  "cost_usd": sum(r.get("cost_usd") or 0 for r in vrows)}
    Pn(f"judge verdicts stored {J['judge']['verdicts']} (YES {J['judge']['yes']} / NO {J['judge']['no']} / unanswered {J['judge']['none']}), ${J['judge']['cost_usd']:.3f}")
    Pn("")
    # ---------------- 2. FT-match scores
    Pn("== 2. FT-MATCH SCORES on Set A (T per candidate; mean over turns with >=1 parsed frontier sample) ==")
    ROLES = ("teacher", "king", "challenger", "genesis", "filler", "ls", "repeat")
    role_of = {"attack_ls": "ls", "attack_repeat": "repeat"}
    by_turn: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for s in scores:
        if s["set"] != "A":
            continue
        role = "teacher" if s["role"].startswith("ref_") else role_of.get(s["role"], s["role"])
        by_turn[s["turn_id"]][role].append(s)
    valid = [tid for tid in by_turn if targets.get(tid, {}).get("classes")]
    J["n_scored_turns"] = len(valid)
    sc: dict = {}
    Pn(f"{'candidate':11s} " + " ".join(f"{k:>16s}" for k in DIALECTS + ("all",)) + "   (mean T | credited% | n)")
    for role in ROLES:
        row = {}
        cells = []
        for kind in DIALECTS + ("all",):
            Ts, cr = [], []
            for tid in valid:
                if kind != "all" and turns[tid]["kind"] != kind:
                    continue
                rs = by_turn[tid].get(role)
                if not rs:
                    continue
                Ts.append(st.mean(r["T"] for r in rs))
                cr.append(any(r["T"] > 0 for r in rs))
            row[kind] = {"mean_T": _m(Ts), "credited": _m(cr), "n": len(Ts)}
            cells.append(f"{_f(_m(Ts), 6, 3)}|{_pct(_m(cr), 5)}|{len(Ts):3d}")
        sc[role] = row
        Pn(f"{role:11s} " + " ".join(f"{c:>16s}" for c in cells))
    Pn("filler = the king's action with a filler thought: T reads only the action, so filler == king; filler is a GATE item (length floor / B licence), not a T item.")
    Pn("repeat = the previous assistant action re-issued: the no-repeat GATE would zero it; its raw T is shown to size the leak if the gate were absent.")
    J["scores"] = sc
    # how-matched breakdown
    how = collections.defaultdict(collections.Counter)
    for s in scores:
        if s["set"] == "A" and s["turn_id"] in valid:
            role = "teacher" if s["role"].startswith("ref_") else role_of.get(s["role"], s["role"])
            how[role][s["how"]] += 1
    Pn("match path per candidate (exact / jaccard / judge / none / forfeit): " +
       "; ".join(f"{r}: " + ", ".join(f"{k} {v}" for k, v in sorted(how[r].items())) for r in ROLES if how.get(r)))
    J["match_paths"] = {r: dict(c) for r, c in how.items()}
    # paired contrasts
    Pn("")
    Pn("-- paired contrasts (per turn; sign test exact binomial, ties dropped; z = mean/SE) --")
    con: dict = {}
    for a, b in (("teacher", "king"), ("king", "ls"), ("king", "repeat"), ("teacher", "challenger"), ("challenger", "king"),
                 ("teacher", "genesis"), ("king", "genesis"), ("teacher", "ls")):
        con[f"{a}-{b}"] = {}
        line = f"{a + '-' + b:20s}"
        for kind in DIALECTS + ("all",):
            d = []
            for tid in valid:
                if kind != "all" and turns[tid]["kind"] != kind:
                    continue
                ra, rb = by_turn[tid].get(a), by_turn[tid].get(b)
                if ra and rb:
                    d.append(st.mean(r["T"] for r in ra) - st.mean(r["T"] for r in rb))
            p = VS.paired(d)
            con[f"{a}-{b}"][kind] = p
            line += f" | {kind[:9]:9s} {_f(p['mean'], 6, 3)} z{_f(p['z'], 5, 1)} +{p['pos']:3d}/-{p['neg']:3d} p{_f(p['p'], 6, 3)}"
        Pn(line)
    J["contrasts"] = con
    # teacher agreement with the frontier modal class
    Pn("")
    Pn("-- frontier-teacher decision agreement on real duel turns: share of teacher refs whose action is in the frontier's MODAL class; "
       "turn-level = >=1 of 3 refs / all 3 refs / mean of the 3 refs' T --")
    ag: dict = {}
    for kind in DIALECTS + ("all",):
        ref_modal, any_modal, all_modal, any_cred = [], [], [], []
        for tid in valid:
            if kind != "all" and turns[tid]["kind"] != kind:
                continue
            rs = by_turn[tid].get("teacher") or []
            if not rs:
                continue
            tg_ = targets[tid]
            modal_idx = {c["idx"] for c in tg_["classes"] if c["modal"]}
            m = [r["class"] in modal_idx for r in rs]
            ref_modal += m
            any_modal.append(any(m))
            all_modal.append(all(m))
            any_cred.append(any(r["T"] > 0 for r in rs))
        side_modal = {}
        for role in ("king", "challenger", "genesis"):
            v = []
            for tid in valid:
                if kind != "all" and turns[tid]["kind"] != kind:
                    continue
                rs = by_turn[tid].get(role) or []
                modal_idx = {c["idx"] for c in targets[tid]["classes"] if c["modal"]}
                v += [r["class"] in modal_idx for r in rs]
            side_modal[role] = _m(v)
        ag[kind] = {"ref_in_modal": _m(ref_modal), "turn_any_ref_modal": _m(any_modal), "turn_all_refs_modal": _m(all_modal),
                    "turn_any_ref_credited": _m(any_cred), "n_turns": len(any_modal), "side_in_modal": side_modal}
        Pn(f"  {kind:13s} refs in modal {_pct(_m(ref_modal))}%  turns >=1 ref modal {_pct(_m(any_modal))}%  all 3 modal {_pct(_m(all_modal))}%  "
           f">=1 ref credited {_pct(_m(any_cred))}%  n={len(any_modal)} | in modal: king {_pct(side_modal['king'])}% challenger {_pct(side_modal['challenger'])}% "
           f"genesis {_pct(side_modal['genesis'])}%")
    J["agreement"] = ag
    Pn("")
    # ---------------- 3. panel
    Pn("== 3. PANEL (kings 11-20 + genesis + teacher on panel turns; k=3 targets per turn, shared across the models on a turn) ==")
    cards, cinfo = P.bench_axes()
    keys = [d for _, (d, _, _) in sorted(P.KINGS.items())] + [P.GENESIS_KEY, P.TEACHER_KEY]
    ax = P.axis_means(cards, keys)
    p_ids = {t for t, v in turns.items() if v["set"] == "P"}
    pv = [tid for tid in targets if tid in p_ids and targets[tid]["classes"]]
    p_scores = collections.defaultdict(list)
    for s in scores:
        if s["set"] == "P":
            p_scores[s["turn_id"]].append(s)
    pscore: dict[str, dict[str, list]] = collections.defaultdict(lambda: {"T": [], "excess": [], "credited": []})
    for tid in pv:
        rs = p_scores.get(tid, [])
        refs = [s["T"] for s in rs if s["role"].startswith("ref_")]
        if not refs:
            continue
        tmean = st.mean(refs)
        pscore["teacher"]["T"].append(tmean)
        pscore["teacher"]["excess"].append(0.0)
        pscore["teacher"]["credited"].append(any(t > 0 for t in refs))
        for s in rs:
            if s["role"] in ("king", "challenger", "genesis"):
                key = "genesis" if s["role"] == "genesis" else s["model"]
                if key not in keys:
                    continue
                pscore[key]["T"].append(s["T"])
                pscore[key]["excess"].append(s["T"] - tmean)
                pscore[key]["credited"].append(s["T"] > 0)
    Pn(f"{'model':9s} {'n':>4} {'meanT':>7} {'excess':>8} {'se':>7} {'cred%':>6}  bench(centered): {'total':>6} {'agentic':>7} {'noTau2':>6} {'chat':>6}")
    prow = []
    for k in keys:
        v = pscore.get(k)
        if not v or not v["T"]:
            continue
        mT, _ = P.mean_se(v["T"])
        mE, seE = P.mean_se(v["excess"])
        b = ax[k]
        prow.append({"model": MODEL_LABEL.get(k, k), "key": k, "n": len(v["T"]), "mean_T": mT, "excess": mE, "excess_se": seE,
                     "credited": _m(v["credited"]), **{a: b.get(a) for a in P.AXES}})
        Pn(f"{MODEL_LABEL.get(k, k):9s} {len(v['T']):4d} {_f(mT, 7, 3)} {_f(mE, 8, 3)} {_f(seE, 7, 3)} {_pct(_m(v['credited']), 6)}  "
           f"                 {_f(b.get('total'), 6, 1)} {_f(b.get('agentic'), 7, 1)} {_f(b.get('agentic_no_tau2'), 6, 1)} {_f(b.get('chat'), 6, 1)}")
    Pn(f"-- Spearman(metric, bench axis) over the {len(prow)} models; perm p two-sided (5000 shuffles); teacher included (excess 0 by definition) --")
    sp: dict = {}
    for metric in ("mean_T", "excess"):
        sp[metric] = {}
        for axis in P.AXES:
            pts = [(r[metric], r[axis]) for r in prow if r.get(axis) is not None and r.get(metric) is not None]
            if len(pts) < 4:
                continue
            x, y = [p[0] for p in pts], [p[1] for p in pts]
            rho = P.spearman(x, y)
            pval = P.permutation_p(x, y, rho, 5000)
            pts_k = [(r[metric], r[axis]) for r in prow if r.get(axis) is not None and r["key"] not in ("teacher", "genesis")]
            rho_k = P.spearman([p[0] for p in pts_k], [p[1] for p in pts_k]) if len(pts_k) >= 4 else None
            sp[metric][axis] = {"rho": rho, "p": pval, "n": len(pts), "rho_kings_only": rho_k, "n_kings_only": len(pts_k)}
            Pn(f"  {metric:7s} vs {axis:16s} rho {_f(rho, 6, 3)} p {_f(pval, 6, 3)} n={len(pts)}   kings-only rho {_f(rho_k, 6, 3)} n={len(pts_k)}")
    Pn("Under this frame a null here can coexist with a right mechanism: no king was trained on frontier targets, so nothing in the panel was optimised for T; "
       "the panel only tells whether incidental frontier-agreement tracks the benches.")
    Pn("Caveat: each king is scored on its own duel turns (its crowning record as challenger + the next crowning as king), so mean_T mixes model and turn "
       "difficulty; excess (T minus the teacher refs' mean T on the same turn) is the turn-controlled reading.")
    J["panel"] = {"rows": prow, "spearman": sp, "axes": ax["_axes"], "matrix_generated_at": cinfo.get("generated_at")}
    # panel per-dialect target stats
    pt = [targets[t] for t in pv]
    J["panel_targets"] = {"turns": len(pt), "classes_per_turn": _m([len(t["classes"]) for t in pt]),
                          "modal_share": _m([t["classes"][0]["share"] for t in pt]),
                          "usd_per_turn": _m([t["cost_usd"] for t in pt]),
                          "by_kind": dict(collections.Counter(t["kind"] for t in pt))}
    Pn(f"panel targets: {len(pt)} turns {J['panel_targets']['by_kind']}, classes/turn {_f(J['panel_targets']['classes_per_turn'], 5, 2)}, "
       f"modal share {_f(J['panel_targets']['modal_share'], 5, 2)}, ${_f(J['panel_targets']['usd_per_turn'], 6, 4)}/turn")
    Pn("")
    # ---------------- 4. truth
    Pn("== 4. TARGET ACCURACY where an environment grade exists (deliberated k=10 glm-5.3 samples at 30 graded states; labels = env-graded first actions from split_states / outcome / harvest N=8 tables) ==")
    if truth:
        per = truth["per_state"]
        J["truth"] = truth

        def acc(rows, key):
            oc = collections.Counter(r.get(key) for r in rows if r.get(key))
            nn = oc["correct"] + oc["mixed"] + oc["wrong"]
            return {"correct": oc["correct"], "mixed": oc["mixed"], "wrong": oc["wrong"], "novel": oc["novel"],
                    "p_solved_class": ((oc["correct"] + oc["mixed"]) / nn) if nn else None,
                    "p_strict_correct": (oc["correct"] / nn) if nn else None, "n_labelled_modal": nn}
        for cls in ("all", "split", "ceiling", "outcome"):
            rows = per if cls == "all" else [r for r in per if r["class"] == cls]
            if not rows:
                continue
            fa, ta = acc(rows, "frontier_modal_outcome"), acc(rows, "teacher_modal_outcome")
            fr = _m([r.get("frontier_modal_rate") for r in rows])
            tr = _m([r.get("teacher_random_rate") for r in rows])
            tmr = _m([r.get("teacher_modal_rate") for r in rows])
            frr = _m([r.get("frontier_random_rate") for r in rows])
            ceil = _m([r.get("frontier_any_class_solved") for r in rows])
            tceil = _m([r.get("teacher_any_class_solved") for r in rows])
            anys = _m([r["any_solved_label"] for r in rows])
            Pn(f"  [{cls:8s} n={len(rows):2d}] frontier MODAL class: solved-class {_pct(fa['p_solved_class'])}% (strict {_pct(fa['p_strict_correct'])}%; "
               f"correct/mixed/wrong/novel {fa['correct']}/{fa['mixed']}/{fa['wrong']}/{fa['novel']}) mean class solve rate {_f(fr, 5, 2)} | "
               f"random frontier sample class rate {_f(frr, 5, 2)}")
            Pn(f"  {'':16s} teacher: random action solves {_pct(tr)}% | MODAL class solved-class {_pct(ta['p_solved_class'])}% (strict {_pct(ta['p_strict_correct'])}%; "
               f"{ta['correct']}/{ta['mixed']}/{ta['wrong']}/{ta['novel']}) mean class rate {_f(tmr, 5, 2)}")
            Pn(f"  {'':16s} execution-verified ceiling: SOME frontier class verified-solved {_pct(ceil)}% of states | SOME teacher class {_pct(tceil)}% | any solved label at all {_pct(anys)}%")
            J.setdefault("truth_summary", {})[cls] = {"n": len(rows), "frontier_modal": fa, "teacher_modal": ta, "frontier_modal_rate": fr,
                                                      "frontier_random_rate": frr, "teacher_random_rate": tr, "teacher_modal_rate": tmr,
                                                      "frontier_any_class_solved": ceil, "teacher_any_class_solved": tceil, "any_solved_label": anys}
        hv = truth["harvest_only"]
        Pn(f"  harvest N=8 tables ({hv['n_states']} failed-teacher states, no deliberation): teacher first actions solve {hv['T_solved']}/{hv['T_n']} = {_pct(hv['T_solved'] / max(1, hv['T_n']))}%; "
           f"single glm-5.3 proposal (F1 greedy + F1b T0.8) forced then teacher finishes {hv['F_solved']}/{hv['F_n']} = {_pct(hv['F_solved'] / max(1, hv['F_n']))}%; "
           f"extra teacher draws (TX) {hv['X_solved']}/{hv['X_n']}; states with T=0/N: {hv['states_T_zero']}, of which a frontier proposal solved {hv['states_F_solved_where_T_zero']}")
        Pn("  MISSING for a full verify-where-possible read: the frontier's modal class is verified only where a labelled action happens to fall in it "
           "(novel classes carry no label); the ceiling row counts states where ANY frontier class touches a verified solve, not the frontier's own continuation; "
           "no new continuations were run.")
    Pn("")
    # ---------------- 5. ceiling
    Pn("== 5. CEILING — frontier vs teacher on our bench axes (kingboard matrix teacher row vs published numbers; margins in points) ==")
    if ceiling:
        J["ceiling"] = ceiling
        Pn(f"matrix teacher row generated {ceiling['matrix_generated_at']}")
        Pn(f"{'bench':20s} {'teacher':>8} {'GLM-5.3':>8} {'margin':>7} {'DeepSeek-V4':>12} {'margin':>7}  notes / sources")
        for r in ceiling["table"]:
            Pn(f"{r['bench']:20s} {_f(r['teacher'], 8, 2)} {_f(r['glm-5.3'], 8, 2)} {_f(r['glm-5.3_margin'], 7, 1)} {_f(r['deepseek-v4'], 12, 2)} {_f(r['deepseek-v4_margin'], 7, 1)}  "
               f"{(r.get('glm-5.3_note') or '')}{(' | ' + r['deepseek-v4_note']) if r.get('deepseek-v4_note') else ''}")
        srcs = {}
        for r in ceiling["table"]:
            for m in ("glm-5.3", "deepseek-v4"):
                if r.get(m + "_source"):
                    srcs.setdefault(r[m + "_source"], set()).add(f"{m}:{r['bench']}")
        for s, names in srcs.items():
            Pn(f"  source: {s}  [{', '.join(sorted(names))}]")
        if ceiling.get("public_sources_note"):
            Pn(f"  note: {ceiling['public_sources_note']}")
        tj = ceiling["traces_join"]
        if "all" in tj:
            a = tj["all"]
            Pn(f"internal proxy — public traces, glm_* vs teacher_* policies on IDENTICAL (source, task, harness): tasks={a['tasks']} "
               f"glm p@1 {_pct(a['glm_p1'])}% teacher p@1 {_pct(a['teacher_p1'])}% diff {_f(a['diff'], 6, 3)} (se {_f(a['se'], 6, 3)}) sign +{a['pos']}/-{a['neg']} p {_f(a['p'], 6, 3)} "
               f"[{a['glm_rollouts']} glm / {a['teacher_rollouts']} teacher rollouts]")
            for g, v in tj["by_group"].items():
                Pn(f"   group {g:10s} tasks={v['tasks']:5d} glm {_pct(v['glm_p1'])}% teacher {_pct(v['teacher_p1'])}% diff {_f(v['diff'], 6, 3)} sign +{v['pos']}/-{v['neg']} p {_f(v['p'], 6, 3)}")
            for g, v in tj["by_harness"].items():
                Pn(f"   harness {g:16s} tasks={v['tasks']:5d} glm {_pct(v['glm_p1'])}% teacher {_pct(v['teacher_p1'])}% diff {_f(v['diff'], 6, 3)}")
            for g, v in tj["by_source"].items():
                Pn(f"   source {g:20s} tasks={v['tasks']:5d} glm {_pct(v['glm_p1'])}% teacher {_pct(v['teacher_p1'])}% diff {_f(v['diff'], 6, 3)} sign +{v['pos']}/-{v['neg']} p {_f(v['p'], 6, 3)}")
            Pn(f"  glm_* models in the traces: {tj.get('glm_models_seen')} (GLM-5.x rows only are used above; the coding pools were shared "
               f"before the seat scheduler so a task ran once — no paired coding tasks exist). UNPAIRED per-source p@1 (weaker proxy; task mix may differ):")
            for g, v in tj.get("unpaired_by_source", {}).items():
                Pn(f"   {g:36s} [{v['group']:8s}] glm {_pct(v['glm_p1'])}% (n={v['glm_n']:5d}) teacher {_pct(v['teacher_p1'])}% (n={v['teacher_n']:5d}) "
                   f"diff {_f(v['diff'], 6, 3)} +-{_f(v['se'], 5, 3)}")
        else:
            Pn(f"traces join unavailable: {tj}")
    Pn("")
    # ---------------- 6. cost
    Pn("== 6. COST ==")
    a_t = [targets[t] for t in targets if targets[t]["set"] == "A"]
    usd_turn = _m([t["cost_usd"] for t in a_t])
    by_stage: dict[str, float] = collections.defaultdict(float)
    runs: dict[tuple, float] = {}
    for r in C.read_jsonl(COST):
        runs[(r["stage"], r["run"])] = max(runs.get((r["stage"], r["run"]), 0), r["cost_usd"])
    for (stg, _), v in runs.items():
        by_stage[stg] += v
    # clustering judge = per new D turn (production); scoring judge = per candidate action matched (per duel side-turn)
    j_cluster = by_stage.get("targets_judge", 0.0) / max(1, len(targets))
    n_scored = sum(1 for s in scores)
    j_score = by_stage.get("score_judge", 0.0) / max(1, n_scored)
    per_turn = (usd_turn or 0) + j_cluster
    cost = {"usd_per_turn_k5_sampling": usd_turn, "usd_per_turn_cluster_judge": j_cluster, "usd_per_turn_total": per_turn,
            "usd_per_candidate_score_judge": j_score, "usd_per_duel_scoring_1300x2": j_score * 1300 * 2,
            "fold_per_day": per_turn * FOLD_TURNS_PER_DAY, "all_D": per_turn * D_TURNS,
            "spent_total": J["cost_usd_total"], "by_stage": dict(by_stage)}
    J["cost"] = cost
    Pn(f"per D turn (production): k={K_A} glm-5.3 samples ${_f(usd_turn, 7, 4)} + clustering judge ${j_cluster:.4f} = ${per_turn:.4f}  "
       f"(k={K_A} at list price; cached prefix share {_pct(tg.get('all', {}).get('cached'))}%)")
    Pn(f"per candidate action scored (duel time): judge ${j_score:.4f} -> a 1300-turn duel, both sides: ${cost['usd_per_duel_scoring_1300x2']:.2f}")
    Pn(f"projection: fold of {FOLD_TURNS_PER_DAY:,} new turns/day = ${cost['fold_per_day']:,.0f}/day; all of D ({D_TURNS:,} turns) = ${cost['all_D']:,.0f} "
       f"(sampling only: ${(usd_turn or 0) * FOLD_TURNS_PER_DAY:,.0f}/day, ${(usd_turn or 0) * D_TURNS:,.0f}); k=3 would be ~0.6x")
    Pn(f"spent this probe: ${J['cost_usd_total']:.2f} by stage {json.dumps({k: round(v, 3) for k, v in by_stage.items()})}")
    # ---------------- 7. verdict (numbers pulled from the tables above)
    Pn("")
    Pn("== 7. VERDICT ==")
    ca = con.get("teacher-king", {}).get("all", {})
    cl = con.get("king-ls", {}).get("all", {})
    cr = con.get("king-repeat", {}).get("all", {})
    cc = con.get("challenger-king", {}).get("all", {})
    cg = con.get("king-genesis", {}).get("all", {})
    Ts_all = [st.mean(r["T"] for r in by_turn[tid]["king"]) for tid in valid if by_turn[tid].get("king")]
    sd_T = st.stdev(Ts_all) if len(Ts_all) > 2 else float("nan")
    bar = 2 * sd_T * math.sqrt(2) / math.sqrt(1300)
    ts_ = J.get("truth_summary", {}).get("all", {})
    tc_ = J.get("truth_summary", {}).get("ceiling", {})
    spm = sp.get("mean_T", {}).get("agentic", {})
    spe = sp.get("excess", {}).get("agentic", {})
    V = [
        f"1. Separation today: FT-match separates acting from not-acting (king-ls +{_f(cl.get('mean'), 5, 3).strip()} z {_f(cl.get('z'), 4, 1).strip()}; "
        f"king-repeat +{_f(cr.get('mean'), 5, 3).strip()} z {_f(cr.get('z'), 4, 1).strip()}) but barely separates real policies: teacher-king "
        f"{_f(ca.get('mean'), 6, 3).strip()} (z {_f(ca.get('z'), 4, 1).strip()}, sign p {_f(ca.get('p'), 5, 2).strip()}), challenger-king {_f(cc.get('mean'), 6, 3).strip()} "
        f"(p {_f(cc.get('p'), 5, 2).strip()}), king-genesis {_f(cg.get('mean'), 6, 3).strip()} (p {_f(cg.get('p'), 5, 2).strip()}); per-turn SD of T ~{sd_T:.2f} -> the paired 2-sigma bar "
        f"at n=1300 is ~{bar:.3f}, so the teacher-king gap would clear it and the live king-challenger gaps would not.",
        f"2. What the targets are: {ag['all']['ref_in_modal'] * 100:.0f}% of teacher ref actions already sit in the frontier's modal class "
        f"({ag['tool_call']['ref_in_modal'] * 100:.0f}% tool_call .. {ag['text']['ref_in_modal'] * 100:.0f}% text); the modal class holds {tg['all']['modal_share']:.2f} of frontier samples "
        f"({tg['all']['classes_per_turn']:.2f} classes/turn). On live D turns a frontier target set is ~70% 'what the teacher does anyway' plus a ~30% frontier-only decision.",
        f"3. Is the target right where truth exists (30 graded states): frontier modal class is a solved class {_pct(ts_.get('frontier_modal', {}).get('p_solved_class')).strip()}% vs teacher modal "
        f"{_pct(ts_.get('teacher_modal', {}).get('p_solved_class')).strip()}%; a random frontier sample's class solves {_f(ts_.get('frontier_random_rate'), 4, 2).strip()} vs a random teacher action "
        f"{_f(ts_.get('teacher_random_rate'), 4, 2).strip()}; execution-verified ceiling identical ({_pct(ts_.get('frontier_any_class_solved')).strip()}% both). On the 11 teacher-fails states the frontier modal touches a solve on "
        f"{tc_.get('frontier_modal', {}).get('correct', 0) + tc_.get('frontier_modal', {}).get('mixed', 0)} vs {tc_.get('teacher_modal', {}).get('correct', 0) + tc_.get('teacher_modal', {}).get('mixed', 0)} states; "
        f"harvest: 1/32 forced frontier proposals rescued a failed state. The frontier's target is about as right as the teacher's, not more.",
        f"4. Panel: mean_T tracks the agentic axis (rho {_f(spm.get('rho'), 5, 2).strip()}, p {_f(spm.get('p'), 5, 3).strip()}, n=12) but the turn-controlled excess does not reach significance "
        f"(rho {_f(spe.get('rho'), 5, 2).strip()}, p {_f(spe.get('p'), 5, 3).strip()}); kings 11-15 sit ~-0.16 under the teacher refs, 16-20 ~-0.04 — consistent with, not proof of, the mechanism; "
        f"no king was trained on these targets, so a null here would not falsify the frame.",
        "5. Ceiling: published frontier margins over the teacher are large on agentic axes (TB2 +17..+38, tau2-telecom +23 DS-V4, LCB +17..+30, SWE-V +2..+17 with harness caveats, GPQA +9..+10, IFBench +13; "
        "tau2 airline/retail, BFCL, AIME unreported) — but the internal proxy inverts: GLM-5.2 as datagen policy is BELOW the teacher on our own coding/terminal sources "
        "(swesmith -13.6, multiswe -11.9, swerebench -9.6, TB2 -8.9 pts; math +0.9). The affordable target generator (GLM-5.x via Engy) is not above the teacher on D's tasks.",
        f"6. A miner trained to reproduce these targets: on the ~70% agreeing turns it learns the teacher (what min(R,G) already rewards); on the ~30% divergent turns it learns GLM-5.3's decision, "
        f"whose measured truth rate equals the teacher's. Plausible gains: format/decision compliance on text and terminus (modal 0.85-0.94, but the teacher is already credited 86-98% there) and "
        f"whatever GLM-5.3's SWE/terminal edge is on tasks where it IS stronger — not demonstrable from D's own traces today. Cost ${per_turn:.3f}/turn (${cost['fold_per_day']:,.0f}/day fold, "
        f"${cost['all_D']:,.0f} for all D) is affordable; the binding constraint is the generator's margin, not the price.",
    ]
    for line in V:
        Pn(line)
    J["verdict"] = V
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(L) + "\n")
    (OUT / "report.json").write_text(json.dumps(J, indent=1, default=str))
    print("\n".join(L))
    print(f"\nwrote {OUT / 'report.txt'} and report.json")
    return 0


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("select", "sample", "targets", "score", "truth", "ceiling", "report", "all"))
    ap.add_argument("--budget", type=float, default=55.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--parallel-turns", type=int, default=12)
    ap.add_argument("--set", default=None, help="A, P or AP (sample stage)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.stage == "select":
        return cmd_select(args)
    if args.stage == "sample":
        return asyncio.run(cmd_sample(args))
    if args.stage == "targets":
        return asyncio.run(cmd_targets(args))
    if args.stage == "score":
        return asyncio.run(cmd_score(args))
    truth = ceiling = None
    if args.stage in ("truth", "report", "all"):
        truth = asyncio.run(cmd_truth(args))
        (OUT / "truth.json").write_text(json.dumps(truth, indent=1))
    if args.stage in ("ceiling", "report", "all"):
        ceiling = cmd_ceiling(args)
    if args.stage == "all":
        cmd_select(args)
        asyncio.run(cmd_sample(args))
        asyncio.run(cmd_targets(args))
        asyncio.run(cmd_score(args))
    if args.stage in ("report", "all"):
        return cmd_report(args, truth, ceiling)
    print(json.dumps({"truth": truth, "ceiling": ceiling}, indent=1, default=str)[:20000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
