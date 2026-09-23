"""G1 sizing — outcome duel at contested turns (2026-09-21).

G1: on turns where the king's and the challenger's ACTIONS differ as a
decision, force each action in a rebuilt copy of the environment, let a fixed
continuation policy (the frozen teacher through Engy) finish the episode,
grade with the environment's own test, and score the turn by the paired solve
difference; turns where the actions agree keep the dense teacher meter.

    python outcome_duel_sizing.py compare   # 6 stored wvk-22 duels -> turns.jsonl.gz
    python outcome_duel_sizing.py judge     # 150-turn glm-5.3-flash "same decision?" calibration
    python outcome_duel_sizing.py report    # report.txt + report.json
    python outcome_duel_sizing.py all

Terms (one line each)
  contested        king action != challenger action on the same turn, by one of the
                   rules below (norm-exact differs; token-Jaccard < 0.5; < 0.3).
  norm-exact       common.norm_action equality (whitespace / quotes unified, dialect body).
  jaccard          token-set Jaccard of the two action bodies (common.jaccard).
  decision-contested  the judge (glm-5.3-flash, T=0) says the two actions are NOT the
                   same decision; the calibrated rate per Jaccard bin converts a
                   surface rule into an expected number of decision-contested turns.
  resumable        the turn's rollout harness can be rebuilt + resumed by ops/recoverable:
                   mini_swe_textbased, verifiers bash, terminus_2 (NOT pi / claude_code /
                   kimi_code / hermes_agent / null).
  graded           the environment scores the finished episode: coding (swesmith, multiswe,
                   scaleswe, swerebench_v2, swelego, r2e_gym), terminal (terminal_lego,
                   terminal_bench_2), math (single-turn, gradable directly), tau2.
  eligible         resumable AND graded AND the action has an environment effect (not a
                   `text` final report, whose content never changes the grade).
  continuation     one teacher-policy run from the forced state to episode end (+ grade).
  k                continuations per forced action (each contested turn costs 2k).
  q = E[p(1-p)]    within-state Bernoulli noise of one continuation; the only power knob
                   besides n and k.
  SE_o(n,k)        sqrt(2 q / (k n)): SE of the mean paired solve difference over n turns.
  min edge         2 * SE_o: the smallest true per-contested-turn solve-rate edge (in
                   probability points) that clears a 2-sigma bar.
  sd unit          the wvk-22 dense meter's per-turn unit (turn = min(z_R, typ_c, z_A));
                   the crown bar is margin > max(2*SE, 0.2 sd) at n = 1000.
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import gzip
import heapq
import json
import math
import random
import re
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (DATA, REPO, Engy, append_jsonl, exact, jaccard, load_verdict,  # noqa: E402
                    corpus_for, norm_action, not_forfeit, read_jsonl)

OUT = REPO / "research" / "results" / "frontier_arbiter" / "outcome_duel"
CONT = REPO / "research" / "results" / "frontier_arbiter" / "outcome" / "continuations.jsonl"
SENS = sorted((REPO / "research" / "results" / "frontier_arbiter" / "outcome").glob("sensitivity*.jsonl"))
TURNS = OUT / "turns.jsonl.gz"
JUDGE_OUT = OUT / "judge.jsonl"

# Six recent wvk-22 duels with both sides valid (index.jsonl on the box):
# 00614 = reign-20 crown (+0.46 sd, z 6.1); 00613 (+0.20, z 2.55, below the
# 0.2-sd floor? no: 0.197 < 0.2); 00615 (-0.37); 00621 (-0.04); 00623 (+0.01);
# 00625 (-0.02).
DUELS = ["chal-00614", "chal-00613", "chal-00615", "chal-00621", "chal-00623", "chal-00625"]

RESUMABLE = {"mini_swe_textbased", "bash", "terminus_2"}
CODING = {"swesmith", "multiswe", "scaleswe", "swerebench_v2", "swelego", "r2e_gym"}
TERMINAL = {"terminal_lego", "terminal_bench_2"}
MATH = {"affine_math", "affine_i3math", "affine_numina"}
TAU2 = {"affine_tau2", "affine_tau2_synth"}
GRADED = CODING | TERMINAL | MATH | TAU2
JUDGE = "glm-5.3-flash"
CONTAINERS_PER_POD = 4
POD_USD_PER_H = 1.0          # ASSUMPTION: CPU-only docker pod (the policy runs on Engy)
ENGY_TEACHER_USD_PER_CONT = None  # filled from continuations.jsonl

JUDGE_PROMPT = """You compare two candidate next actions of a coding/terminal agent at the same point of the same task.

TASK (truncated):
{task}

LAST OBSERVATION the agent saw (truncated):
{obs}

ACTION A:
{a}

ACTION B:
{b}

same_decision: do A and B make the SAME DECISION (same intent and same target: both inspect the same file / place, both apply the same fix, both run the same test, both declare the task finished), ignoring syntax, quoting, flags that do not change the outcome, and the choice of tool? If they differ, would the episode's FINAL RESULT plausibly differ because of this choice (outcome_relevant)?

Reply with ONE JSON object only: {{"same_decision": true|false, "outcome_relevant": true|false, "why": "<=25 words"}}"""


# ------------------------------------------------------------------ helpers
def group_of(source: str) -> str:
    if source in CODING:
        return "coding"
    if source in TERMINAL:
        return "terminal"
    if source in MATH:
        return "math"
    if source in TAU2:
        return "tau2"
    return "other"


def node_path(nodes: list[dict], node_id: int) -> list[dict]:
    path = []
    i = nodes[node_id].get("parent")
    while i is not None:
        path.append(nodes[i])
        i = nodes[i].get("parent")
    return path[::-1]


def is_eligible(r: dict) -> bool:
    """Continuation-eligible (resumable harness, graded source, env-effective
    action) OR directly gradable (single-turn math: compare the boxed answers
    to the gold, no continuation)."""
    return bool(r.get("eligible")) or (r.get("group") == "math" and r.get("kind") == "boxed")


def needs_continuation(r: dict) -> bool:
    return bool(r.get("eligible"))


MUTATING_RE = re.compile(
    r"(?:^|[\s;&|(])(?:sed\s+-i|tee\b|mv\b|cp\b|rm\b|mkdir\b|touch\b|chmod\b|git\s+(?:apply|checkout|stash|commit|reset|revert)|"
    r"patch\b|pip\s+install|npm\s+(?:install|i)\b|cargo\s+(?:add|build)|make\b|pytest\b|python\s+-m\s+pytest|go\s+test|"
    r"npm\s+test|jest\b|cargo\s+test|mvn\b|gradle\b|tox\b|unittest\b|COMPLETE_TASK|MINI_SWE_AGENT_FINAL_OUTPUT|submit\b)"
    r"|>>?\s*[^&|\s]|<<\s*['\"]?EOF|\"name\":\s*\"(?:edit|write|str_replace_editor|apply_patch)\"|\"task_complete\":\s*true",
    re.M)


def env_effective(y: str, kind: str) -> bool:
    """Rule (no judge): the action plausibly changes files / installs / runs a
    test / finishes, as opposed to read-only exploration (cat, grep, ls, sed -n)."""
    body = norm_action(y, kind) if kind != "text" else ""
    if kind == "boxed":
        return True
    if kind == "tool_call":
        return bool(re.search(r"'name': '(?:edit|write|str_replace_editor|apply_patch)'", body)) or bool(MUTATING_RE.search(body))
    return bool(MUTATING_RE.search(body))


def pct(a: int, b: int) -> str:
    return f"{a}/{b} = {100.0 * a / b:5.1f}%" if b else f"{a}/0"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def depth_bin(d: int) -> str:
    for lo, hi in ((0, 2), (3, 5), (6, 9), (10, 15), (16, 25), (26, 40), (41, 10 ** 6)):
        if lo <= d <= hi:
            return f"depth {lo:>2}-{hi}" if hi < 10 ** 6 else f"depth {lo:>2}+"
    return "?"


# ------------------------------------------------------------------ stage 1: compare
def compare(duels: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    done = {(r["duel"], r["turn_id"]) for r in read_jsonl(TURNS)}
    corpora: dict[str, object] = {}
    for rec in duels:
        d = load_verdict(rec)
        v = d["verdict"]
        sl = v["slice"]
        key = sl["manifest_sha256"]
        if key not in corpora:
            corpora[key] = corpus_for(key, sl.get("corpus_base_url"))
        corpus = corpora[key]
        rows = {r["turn_id"]: r for r in corpus.load_index_rows()}
        kr = {r["turn_id"]: r for r in d["king_rows"]}
        cr = {r["turn_id"]: r for r in d["challenger_rows"]}
        refs = d.get("teacher_refs") or {}
        todo = [t for t in d["turn_ids"] if (rec, t) not in done and t in rows]
        todo.sort(key=lambda t: (rows[t]["chunk_key"], int(rows[t]["traj_line"])))
        print(f"{rec}: margin {v['margin']:+.4f} z {v['z']:+.2f} se {v['se']:.4f} "
              f"n_paired {v['n_paired_turns']} -> {len(todo)} turns to compare", flush=True)
        n_out = 0
        for t in todo:
            row = rows[t]
            traj = corpus._traj_at(row["chunk_key"], int(row["traj_line"]))
            meta = next(m for m in traj["turns"] if int(m["turn_idx"]) == int(row["turn_idx"]))
            path = node_path(traj["nodes"], int(meta["node_id"]))
            depth = sum(1 for m in path if m["role"] == "assistant")
            kind = row.get("action_kind") or meta.get("action_kind") or "bash"
            harness = (traj.get("policy") or {}).get("harness") or ""
            source = traj.get("source") or row.get("source") or ""
            k_row, c_row = kr.get(t), cr.get(t)
            both = not_forfeit(k_row) and not_forfeit(c_row)
            out = {"duel": rec, "turn_id": t, "kind": kind, "harness": harness,
                   "source": source, "group": group_of(source), "depth": depth,
                   "n_prefix_chars": int(row.get("n_prefix_chars") or 0),
                   "orig_outcome": traj.get("outcome"), "policy_id": (traj.get("policy") or {}).get("id"),
                   "king_valid": not_forfeit(k_row), "chal_valid": not_forfeit(c_row),
                   "resumable": harness in RESUMABLE, "graded": source in GRADED,
                   "eligible": harness in RESUMABLE and source in GRADED and kind != "text"}
            ref_ys = [r["y"] for r in (refs.get(t) or []) if r.get("y")]
            if len(ref_ys) >= 2:
                pairs = [(a, b) for i, a in enumerate(ref_ys) for b in ref_ys[i + 1:]]
                out["ref_pair_exact"] = st.mean(float(norm_action(a, kind) == norm_action(b, kind)) for a, b in pairs)
                out["ref_pair_jaccard"] = st.mean(jaccard(a, b) for a, b in pairs)
            if both:
                yk = k_row["pairs"][0]["y_a"]
                yc = c_row["pairs"][0]["y_a"]
                out.update({"y_k": yk, "y_c": yc, "exact": exact(yk, [yc], kind),
                            "jaccard": jaccard(yk, yc),
                            "k_vs_ref_exact": exact(yk, ref_ys, kind) if ref_ys else None,
                            "c_vs_ref_exact": exact(yc, ref_ys, kind) if ref_ys else None,
                            "k_vs_ref_j": max((jaccard(yk, r) for r in ref_ys), default=None),
                            "c_vs_ref_j": max((jaccard(yc, r) for r in ref_ys), default=None)})
                if out["eligible"]:
                    task = next((m["content"] for m in path if m["role"] == "user"), "")
                    obs = next((m["content"] for m in reversed(path) if m["role"] in ("user", "tool")), "")
                    out["task_head"] = task[:1500]
                    out["obs_tail"] = obs[-1500:]
            with gzip.open(TURNS, "at") as f:
                f.write(json.dumps(out) + "\n")
            n_out += 1
        print(f"  wrote {n_out}", flush=True)


# ------------------------------------------------------------------ stage 2: judge
def jbin(r: dict) -> str:
    if r.get("exact"):
        return "exact"
    j = r["jaccard"]
    if j >= 0.5:
        return "J>=0.5"
    if j >= 0.3:
        return "0.3<=J<0.5"
    return "J<0.3"


JUDGE_QUOTA = {"exact": 10, "J>=0.5": 40, "0.3<=J<0.5": 40, "J<0.3": 60}


async def judge_async(n_total: int, budget_usd: float) -> None:
    turns = [r for r in read_jsonl(TURNS) if r.get("eligible") and "jaccard" in r and r.get("task_head")]
    done = {(r["duel"], r["turn_id"]) for r in read_jsonl(JUDGE_OUT)}
    rng = random.Random(20260921)
    by = collections.defaultdict(list)
    for r in turns:
        by[jbin(r)].append(r)
    picked = []
    scale = n_total / sum(JUDGE_QUOTA.values())
    for b, q in JUDGE_QUOTA.items():
        pool = [r for r in by[b] if (r["duel"], r["turn_id"]) not in done]
        rng.shuffle(pool)
        picked += pool[: int(round(q * scale))]
    print(f"judge: {len(picked)} turns, pools", {b: len(v) for b, v in by.items()}, flush=True)
    engy = Engy(concurrency=8)

    async def one(r: dict) -> None:
        if engy.cost_usd > budget_usd:
            return
        # A/B order randomised so the judge cannot learn "A = king".
        flip = rng.random() < 0.5
        a, b = (r["y_c"], r["y_k"]) if flip else (r["y_k"], r["y_c"])
        rep: dict = {}
        prompt = JUDGE_PROMPT.format(task=r.get("task_head", ""), obs=r.get("obs_tail", ""),
                                     a=a[:3000], b=b[:3000])
        try:
            rep = await engy.chat(JUDGE, [{"role": "user", "content": prompt}], temperature=0.0,
                                  max_tokens=2500)
            txt = rep["content"] or rep["reasoning"]   # thinking model: JSON in content, budget permitting
            m = txt[txt.find("{"): txt.rfind("}") + 1]
            js = json.loads(m)
        except Exception as e:  # noqa: BLE001
            js = {"same_decision": None, "outcome_relevant": None, "why": f"judge_error: {e!r}"[:200]}
        append_jsonl(JUDGE_OUT, {"duel": r["duel"], "turn_id": r["turn_id"], "bin": jbin(r),
                                 "kind": r["kind"], "harness": r["harness"], "group": r["group"],
                                 "jaccard": r["jaccard"], "exact": r["exact"], "flipped": flip,
                                 **{k: js.get(k) for k in ("same_decision", "outcome_relevant", "why")},
                                 "cost_usd": rep.get("cost_usd")})

    await asyncio.gather(*(one(r) for r in picked))
    print(f"judge cost ${engy.cost_usd:.3f}", flush=True)


# ------------------------------------------------------------------ stage 3: report
def rate_table(turns: list[dict], key, title: str, lines: list[str], js: dict) -> None:
    groups = collections.defaultdict(list)
    for r in turns:
        groups[key(r)].append(r)
    lines.append(f"-- {title} --")
    lines.append(f"  {'':34} {'n':>5} {'exact-diff':>12} {'J<0.5':>12} {'J<0.3':>12}  {'ref-pair exact':>14} {'ref-pair J':>10}")
    for g, rs in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        n = len(rs)
        e = sum(1 for r in rs if not r["exact"])
        j5 = sum(1 for r in rs if r["jaccard"] < 0.5)
        j3 = sum(1 for r in rs if r["jaccard"] < 0.3)
        rpe = [r["ref_pair_exact"] for r in rs if "ref_pair_exact" in r]
        rpj = [r["ref_pair_jaccard"] for r in rs if "ref_pair_jaccard" in r]
        lines.append(f"  {str(g):34} {n:5d} {100 * e / n:11.1f}% {100 * j5 / n:11.1f}% {100 * j3 / n:11.1f}%  "
                     f"{100 * (1 - st.mean(rpe)) if rpe else float('nan'):13.1f}% {st.mean(rpj) if rpj else float('nan'):10.2f}")
        js[str(g)] = {"n": n, "exact_diff": e, "j_lt_05": j5, "j_lt_03": j3,
                      "ref_pair_diff": (1 - st.mean(rpe)) if rpe else None,
                      "ref_pair_jaccard": st.mean(rpj) if rpj else None}


def makespan_pods(walls: list[float], n_jobs: int, t_max_s: float, containers: int,
                  sims: int = 200, quantile: float = 0.9, rng=None) -> int:
    """Smallest pod count whose greedy list-scheduled makespan of n_jobs
    (walls bootstrapped from the empirical distribution, longest-first) is
    <= t_max_s in `quantile` of simulations."""
    rng = rng or random.Random(1)
    if n_jobs == 0:
        return 0
    draws = [sorted((rng.choice(walls) for _ in range(n_jobs)), reverse=True) for _ in range(sims)]
    lo = max(1, math.ceil(sum(walls) / len(walls) * n_jobs / (containers * t_max_s)))
    for pods in range(lo, 10_000):
        slots = pods * containers
        ok = 0
        for d in draws:
            h = [0.0] * slots
            for w in d:
                t = heapq.heappop(h)
                heapq.heappush(h, t + w)
            if max(h) <= t_max_s:
                ok += 1
        if ok >= quantile * sims:
            return pods
    return -1


def report() -> None:
    Phi = st.NormalDist().cdf
    turns_all = read_jsonl(TURNS)
    judge = read_jsonl(JUDGE_OUT)
    conts = read_jsonl(CONT)
    lines: list[str] = []
    js: dict = {"duels": DUELS, "definitions": __doc__.split("Terms (one line each)")[1].strip()}
    P = lines.append

    both = [r for r in turns_all if "jaccard" in r]
    n_slice = len(turns_all)
    n_duels = len({r["duel"] for r in turns_all})
    P("=" * 100)
    P(f"1. CONTESTED RATE king vs challenger — {n_duels} wvk-22 duels, {n_slice} slice turns, "
      f"{len(both)} with both sides valid ({100 * len(both) / n_slice:.1f}%)")
    P("=" * 100)
    per_duel = {}
    for rec in DUELS:
        v = load_verdict(rec)["verdict"]
        rs = [r for r in both if r["duel"] == rec]
        el = [r for r in rs if is_eligible(r)]
        per_duel[rec] = {"margin_sd": v["margin"], "z": v["z"], "se": v["se"], "n_paired": v["n_paired_turns"],
                         "sd_turn_diff": v["se"] * math.sqrt(v["n_paired_turns"]),
                         "n_both": len(rs), "exact_diff": sum(1 for r in rs if not r["exact"]),
                         "j_lt_05": sum(1 for r in rs if r["jaccard"] < 0.5),
                         "j_lt_03": sum(1 for r in rs if r["jaccard"] < 0.3),
                         "eligible": len(el), "eligible_j_lt_05": sum(1 for r in el if r["jaccard"] < 0.5),
                         "eligible_j_lt_03": sum(1 for r in el if r["jaccard"] < 0.3),
                         "eligible_exact_diff": sum(1 for r in el if not r["exact"])}
        d = per_duel[rec]
        P(f"  {rec}: margin {d['margin_sd']:+.3f} sd  z {d['z']:+.2f}  SE {d['se']:.4f}  per-turn sd of paired diff "
          f"{d['sd_turn_diff']:.2f} sd | both-valid {d['n_both']}: exact-diff {d['exact_diff']}  J<0.5 {d['j_lt_05']}  "
          f"J<0.3 {d['j_lt_03']} | eligible {d['eligible']}: exact-diff {d['eligible_exact_diff']}  "
          f"J<0.5 {d['eligible_j_lt_05']}  J<0.3 {d['eligible_j_lt_03']}")
    js["per_duel"] = per_duel
    js["rates"] = {}
    rate_table(both, lambda r: "ALL both-valid", "all", lines, js["rates"].setdefault("all", {}))
    rate_table(both, lambda r: r["kind"], "by dialect (action_kind)", lines, js["rates"].setdefault("kind", {}))
    rate_table(both, lambda r: f"{r['group']}/{r['source']}", "by group/source", lines, js["rates"].setdefault("source", {}))
    rate_table(both, lambda r: r["harness"] or "(null)", "by harness", lines, js["rates"].setdefault("harness", {}))
    rate_table(both, lambda r: depth_bin(r["depth"]), "by prefix depth (assistant turns in prefix)", lines,
               js["rates"].setdefault("depth", {}))
    elig = [r for r in both if is_eligible(r)]
    n_cont_elig = sum(1 for r in elig if needs_continuation(r))
    P("")
    P(f"-- ELIGIBLE = (resumable harness AND graded source AND non-text action) OR single-turn math: {len(elig)}/{len(both)} "
      f"both-valid turns ({100 * len(elig) / len(both):.1f}%), i.e. {1000 * len(elig) / n_slice:.0f} per 1000-turn slice; "
      f"{n_cont_elig} need a continuation, {len(elig) - n_cont_elig} are math (graded directly) --")
    rate_table(elig, lambda r: "ELIGIBLE", "eligible", lines, js["rates"].setdefault("eligible", {}))
    rate_table(elig, lambda r: f"{r['harness']}/{r['kind']}", "eligible by harness/dialect", lines,
               js["rates"].setdefault("eligible_harness", {}))
    rate_table(elig, lambda r: f"{r['group']}/{r['source']}", "eligible by source", lines,
               js["rates"].setdefault("eligible_source", {}))
    rate_table(elig, lambda r: depth_bin(r["depth"]), "eligible by depth", lines, js["rates"].setdefault("eligible_depth", {}))
    rate_table(elig, lambda r: r.get("orig_outcome"), "eligible by original rollout outcome", lines,
               js["rates"].setdefault("eligible_outcome", {}))
    # why turns are not eligible
    why = collections.Counter()
    for r in both:
        if is_eligible(r):
            continue
        if r["kind"] == "text":
            why["text action (no env effect)"] += 1
        elif not r["resumable"] and not r["graded"]:
            why[f"non-resumable harness + ungraded source"] += 1
        elif not r["resumable"]:
            why[f"non-resumable harness ({r['harness'] or 'null'})"] += 1
        else:
            why[f"ungraded source ({r['source']})"] += 1
    P("-- why both-valid turns are NOT eligible --")
    for k, v in why.most_common():
        P(f"  {k:60} {v:5d}  ({100 * v / len(both):.1f}%)")
    js["not_eligible_why"] = dict(why)
    graded_nonres = [r for r in both if r["graded"] and not r["resumable"] and r["kind"] != "text"]
    P(f"  graded but on a non-resumable harness (pi / claude_code / kimi / hermes): {len(graded_nonres)} "
      f"({100 * len(graded_nonres) / len(both):.1f}% of both-valid) — the fallback pool if those harnesses became resumable")
    math_turns = [r for r in both if r["group"] == "math"]
    P(f"  math (single-turn, directly gradable, no continuation): {len(math_turns)} both-valid turns, "
      f"{sum(1 for r in math_turns if not r['exact'])} with a different boxed answer")
    js["graded_nonresumable"] = len(graded_nonres)
    js["math_turns"] = {"n": len(math_turns), "diff": sum(1 for r in math_turns if not r["exact"])}

    # judge calibration
    P("")
    P(f"-- JUDGE calibration ({JUDGE}, T=0): P(NOT same decision | surface bin), eligible turns only; n={len(judge)} --")
    jb = collections.defaultdict(list)
    for r in judge:
        if r.get("same_decision") is not None:
            jb[r["bin"]].append(r)
    p_diff = {}
    for b in ("exact", "J>=0.5", "0.3<=J<0.5", "J<0.3"):
        rs = jb.get(b, [])
        nd = sum(1 for r in rs if r["same_decision"] is False)
        nr = sum(1 for r in rs if r.get("outcome_relevant") is True)
        lo, hi = wilson(nd, len(rs))
        p_diff[b] = nd / len(rs) if rs else None
        P(f"  {b:12} n={len(rs):3d}  different decision {pct(nd, len(rs))} [{100 * lo:.0f}-{100 * hi:.0f}]   "
          f"judge says outcome-relevant {pct(nr, len(rs))}")
    js["judge"] = {b: {"n": len(jb.get(b, [])), "p_diff": p_diff[b],
                       "p_outcome_relevant": (sum(1 for r in jb.get(b, []) if r.get("outcome_relevant")) / len(jb[b])) if jb.get(b) else None}
                   for b in p_diff}
    errs = sum(1 for r in judge if r.get("same_decision") is None)
    if errs:
        P(f"  judge errors: {errs}")

    # expected decision-contested per 1000-turn slice
    def expected_contested(rs: list[dict]) -> float:
        tot = 0.0
        for r in rs:
            p = p_diff.get(jbin(r))
            if r["kind"] == "boxed":
                p = 0.0 if r["exact"] else 1.0     # a different boxed answer is a different decision
            tot += p if p is not None else (0.0 if r["exact"] else 1.0)
        return tot

    P("")
    P("-- PER 1000-TURN SLICE: turns entering the outcome branch under each rule (eligible turns only) --")
    per1000 = {}
    for name, fn in (("norm-exact differs", lambda r: not r["exact"]),
                     ("Jaccard < 0.5", lambda r: r["jaccard"] < 0.5),
                     ("Jaccard < 0.3", lambda r: r["jaccard"] < 0.3)):
        c = sum(1 for r in elig if fn(r))
        per1000[name] = 1000 * c / n_slice
        P(f"  {name:22} {c:5d} / {n_slice} slice turns  -> {per1000[name]:6.1f} per 1000")
    ec = expected_contested(elig)
    per1000["decision-contested (judge-calibrated)"] = 1000 * ec / n_slice
    P(f"  {'decision-contested':22} {ec:5.0f} / {n_slice} slice turns  -> {per1000['decision-contested (judge-calibrated)']:6.1f} per 1000 "
      f"(sum over eligible turns of P(different | bin))")
    p_rel = {b: js["judge"][b]["p_outcome_relevant"] for b in js["judge"]}
    er = 0.0
    for r in elig:
        if r["kind"] == "boxed":
            er += 0.0 if r["exact"] else 1.0
        else:
            pr = p_rel.get(jbin(r))
            er += pr if pr is not None else 0.0
    per1000["outcome-relevant (judge-calibrated)"] = 1000 * er / n_slice
    P(f"  {'outcome-relevant':22} {er:5.0f} / {n_slice} slice turns  -> {per1000['outcome-relevant (judge-calibrated)']:6.1f} per 1000 "
      f"(judge: 'would the FINAL RESULT plausibly differ because of this choice?'; math answer differences count 1)")
    # rule-based filter: contested AND at least one side's action is state-changing
    mut = [r for r in elig if r["jaccard"] < 0.5 and (env_effective(r["y_k"], r["kind"]) or env_effective(r["y_c"], r["kind"]))]
    per1000["J<0.5 AND a state-changing action (rule)"] = 1000 * len(mut) / n_slice
    P(f"  {'J<0.5 & mutating':22} {len(mut):5d} / {n_slice} slice turns  -> {per1000['J<0.5 AND a state-changing action (rule)']:6.1f} per 1000 "
      f"(rule, no judge: either action edits / installs / tests / finishes rather than reads)")
    jm = [(r, next((q for q in judge if q['duel'] == r['duel'] and q['turn_id'] == r['turn_id']), None)) for r in elig]
    jm = [(r, q) for r, q in jm if q and q.get("outcome_relevant") is not None and not r["exact"]]
    tp = sum(1 for r, q in jm if q["outcome_relevant"] and (env_effective(r["y_k"], r["kind"]) or env_effective(r["y_c"], r["kind"])))
    fp = sum(1 for r, q in jm if not q["outcome_relevant"] and (env_effective(r["y_k"], r["kind"]) or env_effective(r["y_c"], r["kind"])))
    fn = sum(1 for r, q in jm if q["outcome_relevant"] and not (env_effective(r["y_k"], r["kind"]) or env_effective(r["y_c"], r["kind"])))
    P(f"    rule vs judge outcome-relevant on the {len(jm)} judged non-exact turns: recall {pct(tp, tp + fn)}, precision {pct(tp, tp + fp)}")
    js["mutating_rule"] = {"n": len(mut), "per_1000": per1000["J<0.5 AND a state-changing action (rule)"], "tp": tp, "fp": fp, "fn": fn}
    P(f"  => of the {per1000['Jaccard < 0.5']:.0f} J<0.5 turns per slice only ~{per1000['outcome-relevant (judge-calibrated)']:.0f} "
      f"({100 * per1000['outcome-relevant (judge-calibrated)'] / per1000['Jaccard < 0.5']:.0f}%) can carry any outcome signal; the rest "
      f"differ in exploration order (which file to read first) and dilute the paired estimate with pure Bernoulli noise.")
    js["per_1000"] = per1000
    # noise floor: teacher-vs-teacher
    rpe = [r["ref_pair_exact"] for r in elig if "ref_pair_exact" in r]
    rpj = [r["ref_pair_jaccard"] for r in elig if "ref_pair_jaccard" in r]
    P(f"  noise floor — teacher's own 3 refs on the same eligible turns: pairwise norm-exact differ "
      f"{100 * (1 - st.mean(rpe)):.1f}%, mean pairwise Jaccard {st.mean(rpj):.2f}: two teacher-level miners would be "
      f"'contested' on roughly as many turns as king vs challenger (exact-diff {100 * js['rates']['eligible']['ELIGIBLE']['exact_diff'] / len(elig):.1f}%).")

    # ------------------------------------------------------------ 2. cost/time
    P("")
    P("=" * 100)
    P(f"2. COST / TIME per continuation — {len(conts)} sibling continuations (teacher via Engy, one per state)")
    P("=" * 100)
    walls = [c["wall_s"] for c in conts]
    costs = [c["cost_usd"] for c in conts]
    nts = [c["n_turns"] for c in conts]

    def stats(xs):
        xs = sorted(xs)
        return {"n": len(xs), "mean": st.mean(xs), "median": st.median(xs),
                "p90": xs[min(len(xs) - 1, int(0.9 * len(xs)))], "max": xs[-1]}

    P(f"  {'':28} {'n':>3} {'wall mean':>10} {'median':>8} {'p90':>8} {'max':>8} | {'$ mean':>8} {'$ max':>8} | {'turns mean':>10} {'max':>5}")
    cost_js = {}
    for name, sel in (("ALL", lambda c: True),
                      *[(f"harness {h}", (lambda c, h=h: c["harness"] == h)) for h in sorted({c["harness"] for c in conts})],
                      *[(f"outcome {o}", (lambda c, o=o: c["outcome"] == o)) for o in sorted({c["outcome"] for c in conts})],
                      *[(f"stop {s}", (lambda c, s=s: c["stop_condition"] == s)) for s in sorted({c["stop_condition"] for c in conts})]):
        cs = [c for c in conts if sel(c)]
        if not cs:
            continue
        w, m, t = stats([c["wall_s"] for c in cs]), stats([c["cost_usd"] for c in cs]), stats([c["n_turns"] for c in cs])
        cost_js[name] = {"wall_s": w, "cost_usd": m, "n_turns": t}
        P(f"  {name:28} {w['n']:3d} {w['mean']:9.0f}s {w['median']:7.0f}s {w['p90']:7.0f}s {w['max']:7.0f}s | "
          f"{m['mean']:8.4f} {m['max']:8.4f} | {t['mean']:10.1f} {t['max']:5d}")
    js["continuations"] = cost_js
    mean_wall, mean_cost = st.mean(walls), st.mean(costs)
    P(f"  wall includes the prefix replay in a fresh container; the sibling's states were depth 3-15 and the "
      f"per-rollout budget 3600 s (max seen {max(walls):.0f}s). Terminus states end in 3-4 turns (cheap); mini-swe "
      f"runs to the 75-turn cap on {sum(1 for c in conts if c['stop_condition'] == 'max_turns')}/{len(conts)}.")
    P("")
    P(f"-- pods needed per duel (containers/pod = {CONTAINERS_PER_POD}; greedy longest-first list scheduling, walls "
      f"bootstrapped from the {len(conts)} empirical walls, 90th-percentile makespan <= T_max) --")
    P(f"  contested turns per slice: J<0.5 rule = {per1000['Jaccard < 0.5']:.0f}, decision-calibrated = "
      f"{per1000['decision-contested (judge-calibrated)']:.0f}; continuations = contested x 2 actions x k")
    P(f"  Engy teacher cost per continuation ${mean_cost:.4f} (mean); pod cost ASSUMED ${POD_USD_PER_H:.2f}/h per CPU pod")
    P(f"  {'contested':>9} {'k':>2} {'n_cont':>7} {'cont-hours':>10} {'Engy $':>7} | {'pods@60m':>8} {'pods@90m':>8} {'pods@120m':>9} | {'pod $@90m':>9}")
    pods_js = []
    rng = random.Random(7)
    for label, nc in (("J<0.5", per1000["Jaccard < 0.5"]), ("decision", per1000["decision-contested (judge-calibrated)"]),
                      ("mutating", per1000["J<0.5 AND a state-changing action (rule)"]),
                      ("relevant", per1000["outcome-relevant (judge-calibrated)"]),
                      ("exact-diff", per1000["norm-exact differs"])):
        for k in (1, 2, 3):
            n_cont = int(round(nc * 2 * k))
            hours = n_cont * mean_wall / 3600
            engy = n_cont * mean_cost
            pods = {tm: makespan_pods(walls, n_cont, tm * 60, CONTAINERS_PER_POD, rng=rng) for tm in (60, 90, 120)}
            pod_cost = pods[90] * POD_USD_PER_H * 1.5
            pods_js.append({"rule": label, "contested": nc, "k": k, "n_cont": n_cont, "container_hours": hours,
                            "engy_usd": engy, "pods": pods, "pod_usd_90m": pod_cost})
            P(f"  {label + ' ' + str(round(nc)):>9} {k:2d} {n_cont:7d} {hours:10.1f} {engy:7.2f} | {pods[60]:8d} {pods[90]:8d} {pods[120]:9d} | {pod_cost:9.2f}")
    js["pods"] = pods_js
    P(f"  today's dense duel: ~27 min scoring + swap, ~29 min per verdict on the eval box; T_max here is ADDITIONAL "
      f"wall unless the outcome branch overlaps the dense pass (it cannot: it needs the contested set).")

    # ------------------------------------------------------------ 3. power
    P("")
    P("=" * 100)
    P("3. STATISTICAL POWER of the paired outcome estimate")
    P("=" * 100)
    P("  Model: contested turn t has state solve-probability p_t for a teacher continuation; s_K, s_C ~ Bernoulli(p_t)")
    P("  under H0 (the two actions are equally good). d_t = mean_k(s_C) - mean_k(s_K); Var(d_t | p_t) = 2 p_t(1-p_t)/k.")
    P("  SE_o(n,k) = sqrt(2 q / (k n)), q = E_t[p_t(1-p_t)] = within-state Bernoulli noise. Sibling data (24 T-arm runs):")
    P("  solve 15/24 = 0.625 overall, 3/11 = 0.27 from originally-failed states, 12/13 = 0.92 from originally-solved.")
    solved_frac = st.mean(c["outcome"] == "solved" for c in conts)
    q_common = solved_frac * (1 - solved_frac)
    q_bimodal = 0.5 * 0.27 * 0.73 + 0.5 * 0.92 * 0.08
    q_det = 0.5 * 0.1 * 0.9 + 0.5 * 0.9 * 0.1
    scen = {"S1 no state info, common p=0.6 (worst)": q_common,
            "S2 bimodal by orig outcome 0.27/0.92 (data-implied upper bound)": q_bimodal,
            "S3 near-deterministic states p in {0.1,0.9}": q_det}
    P(f"  q scenarios: S1 {q_common:.3f} = p(1-p) with no per-state information; S2 {q_bimodal:.3f} = the split by original")
    P(f"  outcome (Var(s)=0.24 = q + Var(p) with Var(p) >= 0.105 from the two-group split => q <= 0.135); S3 {q_det:.3f}.")
    if SENS:
        P(f"  D1 sensitivity files present: {[p.name for p in SENS]} (not modelled here)")
    else:
        P("  No D1 action-sensitivity file (research/results/frontier_arbiter/outcome/sensitivity*.jsonl) — q is assumed, not measured;")
        P("  the sibling's F1 arm (same state, frontier action) never ran, so P(solve | action) vs P(solve | state) is unmeasured.")
    P("")
    P(f"  {'scenario':>8} {'n':>5} {'k':>2} {'SE_o':>7} {'min edge (2SE)':>14} {'solves needed':>13} | {'pods@90m':>8} {'Engy $':>7}")
    power = []
    n_opts = sorted({int(round(per1000["Jaccard < 0.5"])), int(round(per1000["decision-contested (judge-calibrated)"])), 50, 100, 200, 400})
    for sname, q in scen.items():
        tag = sname.split()[0]
        for n in n_opts:
            for k in (1, 2, 3):
                se = math.sqrt(2 * q / (k * n))
                edge = 2 * se
                n_cont = 2 * n * k
                pods = makespan_pods(walls, n_cont, 90 * 60, CONTAINERS_PER_POD, rng=rng)
                power.append({"scenario": sname, "q": q, "n": n, "k": k, "se": se, "min_edge": edge,
                              "extra_solves": edge * n, "pods_90m": pods, "engy_usd": n_cont * mean_cost})
                P(f"  {tag:>8} {n:5d} {k:2d} {se:7.4f} {edge:14.3f} {edge * n:13.1f} | {pods:8d} {n_cont * mean_cost:7.2f}")
    js["power"] = power
    P("  'min edge' = per-contested-turn solve-probability advantage the challenger's actions must have (under the")
    P("  teacher continuation) to clear 2 sigma; 'solves needed' = the same edge as extra solved episodes per slice.")
    f_rel = per1000["outcome-relevant (judge-calibrated)"] / per1000["Jaccard < 0.5"]
    P(f"  Dilution: only ~{100 * f_rel:.0f}% of J<0.5 contested turns are judged outcome-relevant, so a challenger whose")
    P(f"  relevant actions solve +X more often shows a per-contested-turn edge of only ~{f_rel:.2f} X; the S2 k=1 min edge")
    P(f"  of {2 * math.sqrt(2 * q_bimodal / (1 * round(per1000['Jaccard < 0.5']))):.3f} then means +{2 * math.sqrt(2 * q_bimodal / (1 * round(per1000['Jaccard < 0.5']))) / f_rel:.2f} solve probability on the relevant turns "
      f"(the teacher's own solve rate from these states is ~0.6).")
    P("  Caveat on walls: the sibling's states were depth 3-15; eligible contested turns run to depth 41+ (median "
      f"{int(st.median(r['depth'] for r in elig if r['jaccard'] < 0.5))}), and the wall includes the prefix replay, so the pod counts above are optimistic for deep turns.")
    js["dilution"] = {"f_outcome_relevant": f_rel}

    P("")
    P("-- CONFIRMATION-STAGE power: outcome branch only after a dense pass, on the winning slice's contested turns --")
    P("  pass rule: mean paired solve difference over n_c turns >= threshold. P(pass | true edge e) = Phi((e - thr) / SE_o). S2 q.")
    conf = []
    n_c = int(round(per1000["Jaccard < 0.5"]))
    P(f"  {'n_c':>4} {'k':>2} {'SE_o':>6} | {'thr':>7} | " + " ".join(f"e={e:+.2f}" for e in (-0.05, 0.0, 0.03, 0.05, 0.10)))
    for k in (1, 2, 3):
        se = math.sqrt(2 * q_bimodal / (k * n_c))
        for thr_name, thr in (("0", 0.0), ("-1 SE", -se), ("+1 SE", se), ("+2 SE", 2 * se)):
            row = {"n_c": n_c, "k": k, "se": se, "thr": thr_name,
                   "p_pass": {e: Phi((e - thr) / se) for e in (-0.05, 0.0, 0.03, 0.05, 0.10)}}
            conf.append(row)
            P(f"  {n_c:4d} {k:2d} {se:6.3f} | {thr_name:>7} | " + " ".join(f"{100 * v:6.0f}%" for v in row["p_pass"].values()))
    js["confirmation"] = conf
    P("  e = the challenger's true per-contested-turn solve edge under the teacher continuation (e = 0: a dense-meter")
    P("  king-clone or a 'looks like the teacher' improver with no action edge; e < 0: dense pass with worse actions).")

    # dense meter today
    P("")
    P("-- today's crown bar (wvk 22, sd meter): margin > max(2*SE, 0.2 sd) at n = 1000 --")
    ses = [per_duel[r]["se"] for r in DUELS]
    sdt = [per_duel[r]["sd_turn_diff"] for r in DUELS]
    se_d = st.mean(ses)
    P(f"  measured over the 6 duels: SE {min(ses):.4f}-{max(ses):.4f} sd (mean {se_d:.4f}); per-turn sd of the paired dense "
      f"difference {min(sdt):.2f}-{max(sdt):.2f} sd (mean {st.mean(sdt):.2f}).")
    bar = max(2 * se_d, 0.2)
    fp_today = 1 - Phi(bar / se_d)
    P(f"  the 0.2-sd floor = {0.2 / se_d:.2f} SE > 2 SE, so the floor binds: a null challenger crowns with P = "
      f"{100 * fp_today:.2f}% per duel (vs 2.28% at a bare 2 sigma).")
    js["dense_today"] = {"se_mean": se_d, "sd_turn_mean": st.mean(sdt), "bar_sd": bar, "fp_per_duel": fp_today}

    P("")
    P("-- combined score: common unit proposal + false-positive rate --")
    P("  Proposal: standardise each branch by its NULL per-turn sd. Agree turn: u_t = (dense_c - dense_k) / sd_dense")
    P(f"  (sd_dense = {st.mean(sdt):.2f} sd measured above); contested turn: v_t = d_t / sqrt(2 q / k). The slice score is the")
    P("  mean of u and v over all n turns; under H0 every turn has unit variance, so SE = 1/sqrt(n) whatever the")
    P("  contested share, and margin > max(2 SE, floor) keeps the z-test's false-positive rate EXACTLY as today")
    P("  (2.28% at bare 2 sigma; the 0.2-sd floor translates to 0.2/sd_dense = "
      f"{0.2 / st.mean(sdt):.3f} standardised units = {0.2 / st.mean(sdt) * math.sqrt(1000):.2f} SE at n=1000 -> "
      f"{100 * (1 - Phi(max(2.0, 0.2 / st.mean(sdt) * math.sqrt(1000)))):.2f}%).")
    P("  What changes is not the FP rate but WHICH null is tested: on contested turns the dense meter's null is")
    P("  'looks like the teacher'; the outcome null is 'the teacher finishes equally well from either action'. A miner")
    P("  that is the teacher on agree turns and better-than-teacher on contested turns gains z ~ sqrt(n_c) * edge / sqrt(2q/k).")
    comb = []
    for nc in (int(round(per1000["Jaccard < 0.5"])), int(round(per1000["decision-contested (judge-calibrated)"]))):
        for k in (1, 3):
            for edge in (0.05, 0.10, 0.20):
                z = math.sqrt(nc) * edge / math.sqrt(2 * q_bimodal / k) / math.sqrt(1000 / 1000)
                # contribution of the contested turns to the combined slice mean at n=1000, in SE units:
                z_comb = (nc * edge / math.sqrt(2 * q_bimodal / k)) / math.sqrt(1000)
                comb.append({"n_c": nc, "k": k, "edge": edge, "z_outcome_alone": z, "z_combined_n1000": z_comb})
                P(f"    n_c={nc:3d} k={k} true edge {edge:.2f}: outcome branch alone z = {z:5.2f}; its contribution to the "
                  f"combined slice z (n=1000, S2) = {z_comb:5.2f}  (bar: 2.0, or {0.2 / st.mean(sdt) * math.sqrt(1000):.2f} with the floor)")
    js["combined"] = comb
    P("  Alternative weightings (outcome in dense units via a fixed exchange rate, or outcome-overrides-dense) change")
    P("  the FP rate only through mis-estimated variance; with the empirical SE they too stay at the nominal rate.")
    P("  As an AND gate (dense pass AND outcome > 0 on the winning slice's contested turns): FP = "
      f"{100 * fp_today * 0.5:.2f}%; dense pass AND outcome at 2 sigma: {100 * fp_today * 0.0228:.3f}%.")

    # ------------------------------------------------------------ 4. mechanism notes
    P("")
    P("=" * 100)
    P("4. MECHANISM DESIGN NOTES")
    P("=" * 100)
    notes = [
        "Copy-the-king everywhere: zero contested turns -> outcome branch empty -> the score is the dense meter alone, "
        "where an epsilon-copy cannot clear max(2 SE, 0.2 sd) (RT-4). So agreement is safe but cannot win; a challenger "
        "can only win by DISAGREEING somewhere and being right there. Good: the branch rewards differing-and-better.",
        "Cheap-win actions: the challenger picks actions that make the TEACHER continuation succeed (e.g. dump the "
        "whole failing test and the diff at once; write the fix the teacher would recognise) rather than actions it "
        "would take itself. That IS the target quantity (the continuation policy is fixed = a distillation meter on "
        "actions, dual of R), but a miner can also learn the teacher's blind spots: emit a leading comment / plan text "
        "inside the action that steers the teacher (prompt injection through the observation). Mitigation: score only "
        "the executed command's effect, strip non-executable text from the forced action (dialect body), cap action bytes.",
        "Grader gaming: actions that edit / delete tests, print expected outputs, or `git checkout` the gold patch when "
        "it is reachable in the image. The env grade is the same one datagen trusts, so this is the datagen threat "
        "model too; the king suffers it equally only if it takes the same action. Mitigation: outcome branch only on "
        "sources whose graders run hidden tests in a fresh container (swesmith / swerebench / multiswe / terminal-bench "
        "do), and forbid actions touching the test tree (harness-level block list already exists in the recoverable plugin).",
        "Continuation-policy weaknesses: the teacher loops at depth (max_turns on "
        f"{sum(1 for c in conts if c['stop_condition'] == 'max_turns')}/{len(conts)} sibling runs); an action that leaves "
        "the environment in a state the teacher handles badly (long outputs, deleted files) drags the KING's arm down "
        "if the king took it, so both sides face the same policy — symmetric noise, not a bias. But the paired test is "
        "one-sided on the challenger's benefit: pick contested turns adversarially? The miner does not choose which turns "
        "are contested (the king's action is unknown to it at inference), only its own action.",
        "Selection: a miner can choose to disagree ONLY on states it judges easy (p_t near 1 for any sensible action) and "
        "copy the king elsewhere, making d_t ~ 0 with tiny variance and inflating nothing — that's harmless. Disagreeing "
        "on hopeless states (p_t ~ 0) is also harmless. The dangerous case is a miner that knows the king's action "
        "distribution (public verdict artefacts) and disagrees exactly where the king is wrong — which is legitimate skill.",
        "Replayability: the verdict must store, per contested turn, both forced actions, the container image + task id, "
        "the continuation traces (messages + tool outputs) and the grader's reward JSON, plus the continuation model / "
        "temperature / seed. At ~0.15 MB per trace, 2k continuations = ~300 MB per verdict (today ~3.7 MB): ship to R2 "
        "next to the record. Non-determinism: T=0.8 continuations are not bit-replayable; publish k traces and the "
        "grader output so a third party can re-grade the stored end state, and re-run for a fresh sample.",
        "Non-resumable harnesses (pi / Claude Code / Kimi / Hermes, tool_call) and ungraded sources hold "
        f"{100 * (1 - len(elig) / len(both)):.0f}% of both-valid turns: they keep the dense meter (the branch is a "
        "per-turn override, never a requirement). Making the outcome branch matter therefore also means steering the "
        "curriculum toward resumable+graded turns or porting the recoverable plugin to the ACP harnesses.",
        "Composition with the confirmation precedent (wvk 19, removed wvk 21 as 'too difficult'): the branch fits "
        "better as a CROWN CONFIRMATION than a per-turn score — run it only after a dense pass, on the contested "
        "eligible turns of the winning slice (~"
        f"{per1000['Jaccard < 0.5']:.0f} turns, 2k continuations, {[p for p in pods_js if p['rule'] == 'J<0.5' and p['k'] == 1][0]['pods'][90]} pods "
        "at k=1 for 90 min), require the paired solve difference >= 0 (or > -2 SE_o) to crown. Cost is paid only on "
        "the ~3% of verdicts that pass; the honest improver waits ~90 min more (the wvk-19 objection was +40 min).",
    ]
    for i, n in enumerate(notes, 1):
        P(f"  {i}. {n}")
    js["mechanism_notes"] = notes

    P("")
    P("=" * 100)
    P("5. VERDICT")
    P("=" * 100)
    nc5 = int(round(per1000["Jaccard < 0.5"]))
    p9 = [p for p in pods_js if p["rule"] == "J<0.5" and p["k"] == 1][0]
    verdict = [
        f"Per-turn score: NOT powerful enough. ~{nc5} of 1000 turns are eligible+contested (J<0.5), only ~{per1000['outcome-relevant (judge-calibrated)']:.0f} "
        f"outcome-relevant; the branch's own 2-sigma edge is {2 * math.sqrt(2 * q_bimodal / nc5):.3f} solve-probability per contested turn (k=1, S2) = "
        f"+{2 * math.sqrt(2 * q_bimodal / nc5) / f_rel:.2f} on the relevant turns — larger than any plausible king-vs-challenger action gap; "
        "in the combined standardised score it moves the slice z by < 1 unless the edge is >= 0.2.",
        f"Affordable per duel only at k=1: {p9['n_cont']} continuations, {p9['container_hours']:.0f} container-hours, ~${p9['engy_usd']:.0f} Engy + "
        f"{p9['pods'][90]} CPU pods for +90 min (or {p9['pods'][60]} for +60) — tripling today's ~29 min verdict wall; k=3 needs "
        f"{[p for p in pods_js if p['rule'] == 'J<0.5' and p['k'] == 3][0]['pods'][90]} pods. The eligible pool is 37% of the slice "
        "(text finals 33%, non-resumable ACP harnesses 16%, ungraded curriculum envs 8% are out).",
        "Confirmation stage: affordable AND meaningful. Run only after a dense pass (~3% of verdicts), on the winning slice's ~131 contested "
        f"turns with k=1-2 ({p9['pods'][90]}-{[p for p in pods_js if p['rule'] == 'J<0.5' and p['k'] == 2][0]['pods'][90]} pods, +90 min, ~$8-15 Engy); "
        "require mean paired solve difference >= 0: halves noise crowns (0.07% -> 0.03%), passes an honest improver with a +0.05 "
        "action edge 86-94% of the time, and — its real value — rejects a dense-meter mimic whose actions are WORSE (e = -0.05 passes 14%/6%).",
        "It tests a different null than the dense meter ('the teacher finishes equally well from either action' vs 'looks like the teacher'), "
        "so as a gate it is orthogonal evidence, not more of the same; as a score it is too noisy and too diluted by exploration-order differences.",
        f"Prerequisites before any flip: measure q (the D1 F1 arm), extend the recoverable plugin to pi/claude_code (adds {100 * len(graded_nonres) / len(both):.0f}% of turns), "
        "a rule filter for state-changing actions (77/1000, recall 80% of judged-relevant), store both continuation traces + grader JSON in the verdict (~300 MB).",
    ]
    for i, v in enumerate(verdict, 1):
        P(f"  {i}. {v}")
    js["verdict"] = verdict

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    (OUT / "report.json").write_text(json.dumps(js, indent=1, default=str))
    print("\n".join(lines))
    print(f"\nwrote {OUT / 'report.txt'} and report.json")


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["compare", "judge", "report", "all"])
    ap.add_argument("--duels", nargs="*", default=DUELS)
    ap.add_argument("--judge-n", type=int, default=150)
    ap.add_argument("--judge-budget", type=float, default=3.0)
    a = ap.parse_args()
    if a.stage in ("compare", "all"):
        compare(a.duels)
    if a.stage in ("judge", "all"):
        asyncio.run(judge_async(a.judge_n, a.judge_budget))
    if a.stage in ("report", "all"):
        report()


if __name__ == "__main__":
    main()
