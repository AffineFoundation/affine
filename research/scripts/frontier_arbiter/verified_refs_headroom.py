"""Verified-references headroom — how much would "teacher conditioned on
success" (design P3/A1) or "king conditioned on success" (A2) move the KOTH
turn meter, and on how many D turns does it bite? Data only (public traces on
data.affine.io + the public corpus index); no API spend.

    python verified_refs_headroom.py index  [--chunks /tmp/fa/chunks --out /tmp/vr]
    python verified_refs_headroom.py report [--work /tmp/vr]

`index` reads every trace chunk once and writes one row per rollout to
<work>/rollouts.jsonl: envelope metadata, outcome (affine.corpus.view rule),
and the main-root reply sequence (normalised action hash per reply, latent /
visible / action byte counts, finish reason). `report` answers Q1–Q5 of the
2026-09-21 brief and writes research/results/frontier_arbiter/verified_refs/
report.{txt,json}.

Terms (one line each):
  pass@1        mean over tasks of (solved / graded rollouts).
  pass@k        unbiased 1 − C(n−c, k)/C(n, k) averaged over tasks with n ≥ k.
  mixed task    a task with ≥ 1 solved AND ≥ 1 failed graded rollout by the
                same sampler (teacher, or one king) — the only place a
                success filter changes the reference set.
  first-divergence depth  for a (solved, failed) pair on one task: the first
                main-root turn whose normalised action differs.
  pre-divergence share    turns before that depth / all turns of the pair —
                turns where both branches took the same action, so a success
                filter leaves the action refs unchanged.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures as cf
import datetime
import gzip
import hashlib
import json
import math
import os
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REPO, norm_action, render_tool_call  # noqa: E402
from affine import dialects  # noqa: E402  (live tree via common's sys.path)

RESULTS = REPO / "research" / "results" / "frontier_arbiter" / "verified_refs"
CONTINUATIONS = REPO / "research" / "results" / "frontier_arbiter" / "outcome" / "continuations.jsonl"

TURN_CAP_STOP = "max_turns"
TURN_CAP_ARTIFACT = "rollout stopped: max_turns"
CLEAN_STOP_CONDITIONS = frozenset({"agent_completed", "max_turns"})
PRIMARY_REWARD_KEYS = ("solved", "correct", "passed_fraction")
THINK_CLOSE = "</think>"
KS = (1, 2, 3, 4, 8)

# reign -> revision digest (affine/state/state.json on the validator box,
# 2026-09-21; kings serve as `king-<digest12>`).
REIGNS = {
    20: "50e3c081ab1c", 19: "f82deda2ffbd", 18: "73dd5bbcf1f7", 17: "e416aab8599d",
    16: "6ea86b6e1868", 15: "5b1679a0db66", 14: "0f4029fd59ed", 13: "6d0ee567e33e",
    12: "6c877fd2242d", 11: "0ce59769300c", 10: "93b1f299dfdb", 9: "4f7dea97ac79",
    8: "0a04e3d1c57d", 7: "4cd7c090d0bc", 6: "492e3ef293c2",
}
# Two crowns were REVOKED under the window rule (history.jsonl: chal-00454
# crowned 2026-09-12 as reign 12, revoked 21:48 UTC; chal-00461 crowned
# 2026-09-13 as reign 13, revoked 09:59 UTC) but sat in the datagen king seat
# meanwhile; they are absent from state.json's `previous` chain.
REVOKED = {"d76150805915": "12r", "ed3111d3b0b9": "13r"}
DIGEST_TO_REIGN = {v: str(k) for k, v in REIGNS.items()} | REVOKED
WINDOW = [str(k) for k in range(11, 21)] + list(REVOKED.values())
WINDOW_ORDER = ["11", "12r", "12", "13r", "13", "14", "15", "16", "17", "18", "19", "20"]


# ------------------------------------------------------------------ outcome (affine.corpus.view rule, standalone)
def _is_turn_cap_artifact(err: dict, trace: dict) -> bool:
    return trace.get("stop_condition") == TURN_CAP_STOP and \
        TURN_CAP_ARTIFACT in str(err.get("message") or "")


def rollout_outcome(trace: dict) -> str:
    if any(not _is_turn_cap_artifact(e, trace) for e in (trace.get("errors") or [])):
        return "errored"
    if trace.get("stop_condition") not in CLEAN_STOP_CONDITIONS:
        return "errored"
    rewards = trace.get("rewards") or {}
    score = next(((rewards.get(k) or {}).get("score")
                  for k in PRIMARY_REWARD_KEYS if rewards.get(k)), None)
    if isinstance(score, bool) or not isinstance(score, (int, float, str)):
        return "failed" if trace.get("stop_condition") == TURN_CAP_STOP else "unscored"
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "unscored"
    return "solved" if value >= 1.0 else "failed"


SOURCE_GROUP = {
    "swesmith": "coding", "multiswe": "coding", "scaleswe": "coding", "swerebench_v2": "coding",
    "swelego": "coding", "r2e_gym": "coding", "affine_i3code": "coding", "nl2repobench": "nl2repo",
    "affine_nl2lib": "nl2repo", "terminal_lego": "terminal", "terminal_bench_2": "terminal",
    "affine_tmax": "terminal", "affine_math": "math", "affine_i3math": "math", "affine_numina": "math",
    "affine_agent": "tool_use", "affine_wiki": "tool_use", "affine_when2call": "tool_use",
    "affine_tau2": "tool_use", "affine_tau2_synth": "tool_use", "affine_kb_synth": "tool_use",
}


# ------------------------------------------------------------------ index
def _text(content) -> str:
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text")
    return content or ""


def _h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def _as_openai_call(call: dict) -> dict:
    """verifiers stores tool calls flat ({id, name, arguments}); common's
    renderer expects the OpenAI shape ({function: {name, arguments}})."""
    if "function" in call:
        return call
    return {"function": {"name": call.get("name"), "arguments": call.get("arguments")}}


def reply_features(msg: dict, kind: str) -> dict:
    """Normalised action hash + byte shape of one sampled assistant reply."""
    # mini-swe-agent v2 fences its command ```mswea_bash_command; the corpus
    # stores it as ```bash (evalsrv.swerunner.ACTION_REGEX accepts both).
    content = _text(msg.get("content")).replace("```mswea_bash_command", "```bash")
    latent = msg.get("reasoning_content") or ""
    if not latent and THINK_CLOSE in content:
        latent, _, content = content.partition(THINK_CLOSE)
        latent = latent.replace("<think>", "")
    tool_calls = msg.get("tool_calls") or []
    action, visible, used = "", content, kind
    if tool_calls:
        action = "\n".join(render_tool_call(_as_openai_call(c)) for c in tool_calls)
        used = "tool_call"
    else:
        before, y = dialects.split_action(content, kind)
        if y:
            action, visible = y, before
    return {
        "ah": _h(norm_action(action, used)) if action else None,
        "alen": len(action), "lat": len(latent), "vis": len(visible.strip()),
    }


def index_envelope(e: dict) -> dict:
    tr = e["trace"]
    pol = e.get("policy") or {}
    task = e.get("task") or {}
    kind = pol.get("action_kind") or dialects.DEFAULT_KIND
    nodes = tr.get("nodes") or []
    linear = bool(nodes) and "parent" not in nodes[-1]
    finish = {}
    for c in tr.get("calls") or []:
        finish[c.get("node")] = c.get("finish_reason")
    replies, main_root, n_roots = [], None, set()
    for i, nd in enumerate(nodes):
        m = nd.get("message") or {}
        if not nd.get("sampled") or m.get("role") != "assistant":
            continue
        j = i
        if not linear:
            while nodes[j].get("parent") is not None:
                j = nodes[j]["parent"]
        else:
            j = 0
        n_roots.add(j)
        if main_root is None:
            main_root = j
        if j != main_root:
            continue
        f = reply_features(m, kind)
        f["t"] = len(replies)
        f["fin"] = finish.get(i)
        replies.append(f)
    model = pol.get("model") or ""
    digest = model.split("king-")[-1][:12] if "king-" in model else None
    return {
        "chunk": None, "line": None, "rollout_id": e.get("rollout_id"),
        "source": e.get("source"), "policy_id": pol.get("id"), "harness": pol.get("harness"),
        "action_kind": kind, "model": model, "temperature": pol.get("temperature"),
        "king_digest": digest, "reign": DIGEST_TO_REIGN.get(digest or ""),
        "outcome": rollout_outcome(tr), "stop": tr.get("stop_condition"),
        "sid": task.get("sid"), "uid": task.get("uid"), "teacher_solved": task.get("teacher_solved"),
        "stored_at": e.get("stored_at"), "n_roots": len(n_roots),
        "n_replies": len(replies), "replies": replies,
    }


def index_chunk(path: Path) -> list[dict]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                r = index_envelope(json.loads(line))
            except Exception as ex:  # noqa: BLE001 — one bad envelope must not kill the chunk
                r = {"chunk": path.name, "line": i, "error": repr(ex)[:200]}
            r["chunk"], r["line"] = path.name, i
            rows.append(r)
    return rows


def cmd_index(args: argparse.Namespace) -> None:
    paths = sorted(Path(args.chunks).glob("*.jsonl.gz"))
    out = Path(args.out) / "rollouts.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as f, cf.ProcessPoolExecutor(args.workers) as ex:
        for k, rows in enumerate(ex.map(index_chunk, paths, chunksize=4), 1):
            for r in rows:
                f.write(json.dumps(r) + "\n")
            n += len(rows)
            if k % 250 == 0:
                print(f"  {k}/{len(paths)} chunks, {n} rollouts", flush=True)
    print(f"{n} rollouts -> {out}")


# ------------------------------------------------------------------ pass@k
def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def task_key(r: dict, same_policy: bool) -> tuple:
    return (r["source"], r["sid"], r["policy_id"]) if same_policy else (r["source"], r["sid"])


def task_table(rows: list[dict], same_policy: bool = False) -> dict:
    """Graded outcome counts per task. Task = (source, sid) — the brief's
    grouping, which pools the teacher's harnesses on one task — or, with
    same_policy, (source, sid, policy_id) = one sampler on one task, the
    only grouping in which a "mixed" task is a genuine sampler split."""
    by = collections.defaultdict(lambda: [0, 0])   # key -> [n_graded, n_solved]
    for r in rows:
        if r["outcome"] not in ("solved", "failed"):
            continue
        t = by[task_key(r, same_policy)]
        t[0] += 1
        t[1] += r["outcome"] == "solved"
    return by


def passk_summary(tasks: dict) -> dict:
    multi = {k: v for k, v in tasks.items() if v[0] >= 2}
    out = {"tasks_graded": len(tasks), "tasks_ge2": len(multi),
           "rollouts_ge2": sum(v[0] for v in multi.values())}
    if not multi:
        return out
    ns = sorted(v[0] for v in multi.values())
    out["n_per_task"] = {"p50": st.median(ns), "mean": round(st.mean(ns), 2), "max": ns[-1],
                         "hist": dict(sorted(collections.Counter(ns).items()))}
    out["pass1_all"] = round(st.mean(v[1] / v[0] for v in tasks.values()), 4)
    for k in KS:
        el = [v for v in multi.values() if v[0] >= k]
        if el:
            out[f"pass@{k}"] = round(st.mean(pass_at_k(n, c, k) for n, c in el), 4)
            out[f"pass@{k}_tasks"] = len(el)
    out["pass@n"] = round(st.mean(1.0 if c > 0 else 0.0 for _, c in multi.values()), 4)
    kinds = collections.Counter(
        "mixed" if 0 < c < n else ("all_solved" if c == n else "all_failed")
        for n, c in multi.values())
    out["mixed_share"] = round(kinds["mixed"] / len(multi), 4)
    out["all_solved_share"] = round(kinds["all_solved"] / len(multi), 4)
    out["all_failed_share"] = round(kinds["all_failed"] / len(multi), 4)
    return out


# ------------------------------------------------------------------ divergence
def first_divergence(a: list[dict], b: list[dict]) -> int:
    """First main-root turn whose normalised action differs (None = no
    action in the dialect; None vs None is equal). Beyond the shorter reply
    list the branches trivially differ."""
    for t in range(min(len(a), len(b))):
        if a[t]["ah"] != b[t]["ah"]:
            return t
    return min(len(a), len(b))


def divergence_stats(rows: list[dict], label: str, same_policy: bool = True) -> dict:
    by = collections.defaultdict(lambda: {"solved": [], "failed": []})
    for r in rows:
        if r["outcome"] in ("solved", "failed") and r["n_replies"] > 0:
            by[task_key(r, same_policy)][r["outcome"]].append(r)
    depths, shares, per_h, per_s, pre_turns, all_turns, n_pairs = [], [], \
        collections.defaultdict(list), collections.defaultdict(list), 0, 0, 0
    div0_same_first_thought = 0
    for key, g in by.items():
        src = key[0]
        if not g["solved"] or not g["failed"]:
            continue
        for s in g["solved"]:
            for f in g["failed"]:
                d = first_divergence(s["replies"], f["replies"])
                n_tot = len(s["replies"]) + len(f["replies"])
                depths.append(d)
                shares.append(2 * d / n_tot)
                pre_turns += 2 * d
                all_turns += n_tot
                per_h[s["harness"]].append(d)
                per_s[src].append(d)
                n_pairs += 1
    if not depths:
        return {"label": label, "n_pairs": 0}

    def summ(ds: list[int]) -> dict:
        c = collections.Counter(ds)
        return {"n": len(ds), "p50": st.median(ds), "mean": round(st.mean(ds), 2),
                "d0": round(c[0] / len(ds), 3), "d_le1": round((c[0] + c[1]) / len(ds), 3),
                "d_le2": round((c[0] + c[1] + c[2]) / len(ds), 3),
                "d_ge5": round(sum(v for k, v in c.items() if k >= 5) / len(ds), 3)}
    hist = collections.Counter(min(d, 10) for d in depths)
    return {
        "label": label, "n_pairs": n_pairs, "n_mixed_tasks": sum(
            1 for g in by.values() if g["solved"] and g["failed"]),
        "depth": summ(depths),
        "depth_hist_capped10": {str(k): v for k, v in sorted(hist.items())},
        "pre_divergence_share_pairmean": round(st.mean(shares), 4),
        "pre_divergence_share_turnweighted": round(pre_turns / all_turns, 4),
        "by_harness": {h: summ(v) for h, v in sorted(per_h.items())},
        "by_source": {s_: summ(v) for s_, v in sorted(per_s.items())},
    }


# ------------------------------------------------------------------ band proxy (Q5)
def band_proxy(rows: list[dict]) -> dict:
    """Thought shape of solved vs failed teacher replies at matched depth
    (per harness): latent chars, visible chars, action chars, share of
    replies with no latent / no visible text, finish=length share."""
    buckets = ("t0", "t1-3", "t4-9", "t10+")

    def bucket(t: int) -> str:
        return "t0" if t == 0 else "t1-3" if t <= 3 else "t4-9" if t <= 9 else "t10+"
    acc: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["outcome"] not in ("solved", "failed"):
            continue
        for rep in r["replies"]:
            acc[(r["harness"], bucket(rep["t"]), r["outcome"])]["lat"].append(rep["lat"])
            acc[(r["harness"], bucket(rep["t"]), r["outcome"])]["vis"].append(rep["vis"])
            acc[(r["harness"], bucket(rep["t"]), r["outcome"])]["alen"].append(rep["alen"])
            acc[(r["harness"], bucket(rep["t"]), r["outcome"])]["nolat"].append(rep["lat"] == 0)
            acc[(r["harness"], bucket(rep["t"]), r["outcome"])]["novis"].append(rep["vis"] == 0)
            acc[(r["harness"], bucket(rep["t"]), r["outcome"])]["length"].append(rep["fin"] == "length")
    out: dict = {}
    for (h, b, o), d in acc.items():
        n = len(d["lat"])
        if n < 30:
            continue
        out.setdefault(h, {}).setdefault(b, {})[o] = {
            "n": n, "lat_p50": st.median(d["lat"]), "lat_mean": round(st.mean(d["lat"])),
            "vis_p50": st.median(d["vis"]), "alen_p50": st.median(d["alen"]),
            "no_latent": round(st.mean(d["nolat"]), 3), "no_visible": round(st.mean(d["novis"]), 3),
            "finish_length": round(st.mean(d["length"]), 3),
        }
    # log-ratio of median latent length solved/failed where both exist
    for h, bs in out.items():
        for b, d in bs.items():
            if "solved" in d and "failed" in d and d["failed"]["lat_p50"] > 0:
                d["lat_p50_ratio_solved_over_failed"] = round(
                    d["solved"]["lat_p50"] / d["failed"]["lat_p50"], 3)
    pooled: dict = {}
    agg: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r["outcome"] not in ("solved", "failed"):
            continue
        for rep in r["replies"]:
            if rep["t"] == 0:
                continue          # t0 is the task prompt reply; depth >= 1 is the agentic regime
            a = agg[(r["harness"], r["outcome"])]
            a["lat"].append(rep["lat"])
            a["novis"].append(rep["vis"] == 0)
            a["length"].append(rep["fin"] == "length")
    for h in sorted({k[0] for k in agg}):
        s_, f_ = agg.get((h, "solved")), agg.get((h, "failed"))
        if not s_ or not f_ or len(s_["lat"]) < 100 or len(f_["lat"]) < 100:
            continue
        pooled[h] = {"n_solved": len(s_["lat"]), "n_failed": len(f_["lat"]),
                     "lat_p50_solved": st.median(s_["lat"]), "lat_p50_failed": st.median(f_["lat"]),
                     "lat_p90_solved": sorted(s_["lat"])[int(0.9 * (len(s_["lat"]) - 1))],
                     "lat_p90_failed": sorted(f_["lat"])[int(0.9 * (len(f_["lat"]) - 1))],
                     "no_visible_solved": round(st.mean(s_["novis"]), 3), "no_visible_failed": round(st.mean(f_["novis"]), 3),
                     "finish_length_solved": round(st.mean(s_["length"]), 4), "finish_length_failed": round(st.mean(f_["length"]), 4)}
    return {"buckets": buckets, "by_harness": out, "pooled_depth_ge1": pooled}


# ------------------------------------------------------------------ report
def load_rows(work: Path) -> list[dict]:
    rows = []
    with open(work / "rollouts.jsonl") as f:
        for line in f:
            r = json.loads(line)
            if "error" not in r:
                r["reign"] = DIGEST_TO_REIGN.get(r.get("king_digest") or "")
                rows.append(r)
    return rows


def teacher_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if (r["policy_id"] or "").startswith("teacher_")]


def king_rows(rows: list[dict]) -> list[dict]:
    return [r for r in rows if (r["policy_id"] or "").startswith("king_")]


def q1_teacher(trows: list[dict]) -> dict:
    out = {"overall": passk_summary(task_table(trows)),
           "overall_same_policy": passk_summary(task_table(trows, True)),
           "by_source": {}, "by_source_same_policy": {}, "by_harness_same_policy": {},
           "by_source_harness": {}, "coverage": {}}
    for v in sorted({r["source"] for r in trows if r["source"]}):
        sub = [r for r in trows if r["source"] == v]
        graded = [r for r in sub if r["outcome"] in ("solved", "failed")]
        out["coverage"][v] = {"graded_rollouts": len(graded), "tasks": len({r["sid"] for r in graded}),
                              "tasks_ge2_any_harness": passk_summary(task_table(sub))["tasks_ge2"],
                              "tasks_ge2_same_policy": passk_summary(task_table(sub, True))["tasks_ge2"]}
        s = passk_summary(task_table(sub))
        if s["tasks_ge2"] >= 5:
            out["by_source"][v] = s
        s = passk_summary(task_table(sub, True))
        if s["tasks_ge2"] >= 5:
            out["by_source_same_policy"][v] = s
    for v in sorted({r["harness"] for r in trows if r["harness"]}):
        sub = [r for r in trows if r["harness"] == v]
        s = passk_summary(task_table(sub, True))
        if s["tasks_ge2"] >= 5:
            out["by_harness_same_policy"][v] = s
    combos = sorted({(r["source"], r["harness"]) for r in trows})
    for src, h in combos:
        sub = [r for r in trows if r["source"] == src and r["harness"] == h]
        s = passk_summary(task_table(sub, True))
        if s["tasks_ge2"] >= 5:
            out["by_source_harness"][f"{src}|{h}"] = s
    out["outcome_counts"] = dict(collections.Counter(r["outcome"] for r in trows))
    out["temperature_values"] = dict(collections.Counter(str(r["temperature"]) for r in trows))
    return out


def in_window(r: dict) -> bool:
    return r.get("reign") in WINDOW


def q2_kings(rows: list[dict]) -> dict:
    trows = teacher_rows(rows)
    t_tasks = task_table(trows)
    krows = [r for r in king_rows(rows) if in_window(r)]
    out = {"n_king_rollouts": len(krows),
           "king_policies": dict(collections.Counter(r["policy_id"] for r in krows)),
           "by_reign": {}}
    greedy = [r for r in krows if (r["policy_id"] or "").endswith("_greedy")]
    sampled = [r for r in krows if not (r["policy_id"] or "").endswith("_greedy")]
    for label, sub in (("sampled", sampled), ("greedy", greedy), ("all", krows)):
        # one king + one policy on one task = the sampler; kings are told apart by policy? no —
        # by digest, so tag the sid with the digest before grouping.
        tagged = [dict(r, sid=f"{r['sid']}@{r['king_digest']}") for r in sub]
        out[f"pooled_{label}"] = passk_summary(task_table(tagged, True))
    for reign in WINDOW_ORDER:
        kr = [r for r in sampled if r["reign"] == reign]
        if not kr:
            continue
        k_tasks = task_table(kr, True)
        s = passk_summary(k_tasks)
        s["by_source_ge2"] = dict(collections.Counter(k[0] for k, v in k_tasks.items() if v[0] >= 2))
        s["n_rollouts"] = len(kr)
        s["harnesses"] = dict(collections.Counter(r["harness"] for r in kr))
        s["digest"] = kr[0]["king_digest"]
        s["vs_teacher"] = king_vs_teacher(kr, trows)
        out["by_reign"][reign] = s
    out["vs_teacher_pooled_sampled"] = king_vs_teacher(sampled, trows)
    out["vs_teacher_pooled_greedy"] = king_vs_teacher(greedy, trows)
    return out


def king_vs_teacher(kr: list[dict], trows: list[dict]) -> dict:
    """Same task, same harness: the king's graded rollouts vs the teacher's.
    Per task: king solved-any / teacher solved-any -> 2x2; king pass@1 and
    pass@n vs teacher pass@1 and pass@n on exactly those tasks. 'king|success'
    (A2) reaches king pass@n on tasks where the king has >= 1 solve; the
    union column is what pooling both samplers' successes would reach."""
    kt = collections.defaultdict(lambda: [0, 0])
    tt = collections.defaultdict(lambda: [0, 0])
    for r in kr:
        if r["outcome"] in ("solved", "failed"):
            t = kt[(r["source"], r["sid"], r["harness"])]
            t[0] += 1
            t[1] += r["outcome"] == "solved"
    for r in trows:
        if r["outcome"] in ("solved", "failed"):
            t = tt[(r["source"], r["sid"], r["harness"])]
            t[0] += 1
            t[1] += r["outcome"] == "solved"
    shared = [k for k in kt if k in tt]
    if not shared:
        return {"n_tasks": 0}
    cells = collections.Counter()
    for k in shared:
        cells[("K+" if kt[k][1] else "K-") + ("T+" if tt[k][1] else "T-")] += 1
    n = len(shared)
    by_group = collections.defaultdict(lambda: collections.Counter())
    for k in shared:
        g = SOURCE_GROUP.get(k[0], "general")
        by_group[g]["n"] += 1
        by_group[g]["K+"] += bool(kt[k][1])
        by_group[g]["T+"] += bool(tt[k][1])
        by_group[g]["K+T-"] += bool(kt[k][1]) and not tt[k][1]
        by_group[g]["K-T+"] += (not kt[k][1]) and bool(tt[k][1])
    return {
        "n_tasks": n, "king_rollouts": sum(kt[k][0] for k in shared),
        "teacher_rollouts": sum(tt[k][0] for k in shared),
        "king_pass1": round(st.mean(kt[k][1] / kt[k][0] for k in shared), 4),
        "king_passn": round(st.mean(1.0 if kt[k][1] else 0.0 for k in shared), 4),
        "teacher_pass1": round(st.mean(tt[k][1] / tt[k][0] for k in shared), 4),
        "teacher_passn": round(st.mean(1.0 if tt[k][1] else 0.0 for k in shared), 4),
        "union_passn": round(st.mean(1.0 if (kt[k][1] or tt[k][1]) else 0.0 for k in shared), 4),
        "cells": {k: cells[k] for k in ("K+T+", "K+T-", "K-T+", "K-T-")},
        "king_only_share": round(cells["K+T-"] / n, 4), "teacher_only_share": round(cells["K-T+"] / n, 4),
        "by_group": {g: {"n": c["n"], "king_passn": round(c["K+"] / c["n"], 3), "teacher_passn": round(c["T+"] / c["n"], 3),
                         "king_only": round(c["K+T-"] / c["n"], 3), "teacher_only": round(c["K-T+"] / c["n"], 3)}
                     for g, c in sorted(by_group.items())},
    }


KEPT_STATES = CONTINUATIONS.parent / "kept.jsonl"


def at_depth_split(cont: list[dict]) -> dict:
    """The sibling resumed teacher states at depth 3-15 (frontier-contested
    pivots, mini_swe/bash/terminus) and ran ONE fresh teacher continuation to
    the grader. Original outcome vs continuation outcome = a per-STATE
    resample: the share of states whose outcome flips with one resample, and
    the implied N=4 mixed probability 1 - p^4 - (1-p)^4 at p = the
    continuation solve rate of that class of state."""
    if not KEPT_STATES.exists() or not cont:
        return {}
    kept = {}
    with open(KEPT_STATES) as f:
        for line in f:
            k = json.loads(line)
            kept[k["state_id"]] = k
            kept[k["rollout_id"] + ":" + str(k["depth"])] = k
    cells = collections.Counter()
    depths = []
    for c in cont:
        rid, depth = c["state_id"].split(":")[0], c["state_id"].split(":")[1]
        k = kept.get(c.get("kept_state_id") or "") or kept.get(f"{rid}:{depth}")
        if not k or c["outcome"] not in ("solved", "failed"):
            continue
        cells[(k["orig_outcome"], c["outcome"])] += 1
        depths.append(int(k["depth"]))
    n = sum(cells.values())
    if not n:
        return {}
    out = {"n_states": n, "depth_range": [min(depths), max(depths)],
           "cells": {f"orig_{a}->cont_{b}": v for (a, b), v in sorted(cells.items())},
           "flip_share": round((cells[("solved", "failed")] + cells[("failed", "solved")]) / n, 3)}
    for orig in ("solved", "failed"):
        tot = cells[(orig, "solved")] + cells[(orig, "failed")]
        if tot:
            p_ = cells[(orig, "solved")] / tot
            out[f"orig_{orig}"] = {"n": tot, "cont_solve_rate": round(p_, 3),
                                   "p_mixed_N4": round(1 - p_ ** 4 - (1 - p_) ** 4, 3)}
    return out


def q4_cost(rows: list[dict], q1: dict) -> dict:
    cont = []
    if CONTINUATIONS.exists():
        with open(CONTINUATIONS) as f:
            cont = [json.loads(line) for line in f if line.strip()]
    wall = [c["wall_s"] for c in cont if isinstance(c.get("wall_s"), (int, float))]
    cost = [c["cost_usd"] for c in cont if isinstance(c.get("cost_usd"), (int, float))]
    out = {"continuations_n": len(cont), "at_depth_split": at_depth_split(cont)}
    if wall:
        out["wall_s"] = {"p50": round(st.median(wall), 1), "mean": round(st.mean(wall), 1),
                         "p90": round(sorted(wall)[int(0.9 * (len(wall) - 1))], 1)}
    if cost:
        out["cost_usd"] = {"p50": round(st.median(cost), 4), "mean": round(st.mean(cost), 4),
                           "sum": round(sum(cost), 2)}
    by_kind = collections.defaultdict(list)
    for c in cont:
        by_kind[c.get("resume_kind") or c.get("harness") or "?"].append(c)
    out["by_kind"] = {k: {"n": len(v),
                          "wall_p50": round(st.median([x["wall_s"] for x in v if x.get("wall_s") is not None]), 1)
                          if any(x.get("wall_s") is not None for x in v) else None,
                          "cost_mean": round(st.mean([x["cost_usd"] for x in v if x.get("cost_usd") is not None]), 4)
                          if any(x.get("cost_usd") is not None for x in v) else None}
                      for k, v in by_kind.items()}
    if wall and cost:
        n_samples = 4
        mean_wall_h = st.mean(wall) / 3600
        mean_cost = st.mean(cost)
        for pivot_share, label in ((0.10, "pivots_10pct"), (q1["overall"].get("mixed_share", 0.0), "mixed_task_share")):
            per_1000 = 1000 * n_samples
            out[label] = {
                "pivot_share": round(pivot_share, 3),
                "continuations_per_1000_new_turns": round(per_1000 * pivot_share),
                "container_hours_per_1000_new_turns": round(per_1000 * pivot_share * mean_wall_h, 1),
                "usd_per_1000_new_turns": round(per_1000 * pivot_share * mean_cost, 2),
                "container_hours_per_1000_pivot_turns": round(per_1000 * mean_wall_h, 1),
                "usd_per_1000_pivot_turns": round(per_1000 * mean_cost, 2),
            }
    return out


def q4_d_share(rows: list[dict], work: Path) -> dict:
    """Share of D turns (corpus index, epoch 58) whose source rollout solved,
    per source and per policy family, by joining the index's rollout_id to the
    trace-derived outcome."""
    import pyarrow.parquet as pq  # noqa: PLC0415 — optional heavy dep, report-only
    pq_path = work / "turns_0058.parquet"
    if not pq_path.exists():
        return {"error": "index parquet missing"}
    t = pq.read_table(pq_path, columns=["rollout_id", "source", "stratum", "action_kind"])
    rid = t.column("rollout_id").to_pylist()
    src = t.column("source").to_pylist()
    strat = t.column("stratum").to_pylist()
    by_rid = {r["rollout_id"]: r for r in rows}
    per_src = collections.defaultdict(collections.Counter)
    per_fam = collections.defaultdict(collections.Counter)
    per_group = collections.defaultdict(collections.Counter)
    for a, s_, g in zip(rid, src, strat):
        r = by_rid.get(a)
        o = r["outcome"] if r else "not_in_traces"
        fam = (r["policy_id"] or "?").split("_")[0] if r else "not_in_traces"
        per_src[s_][o] += 1
        per_fam[fam][o] += 1
        per_group[(g or "").split(":")[0]][o] += 1

    def fmt(c: collections.Counter) -> dict:
        n = sum(c.values())
        return {"n_turns": n, **{k: round(v / n, 3) for k, v in sorted(c.items())}}
    return {"n_turns": len(rid),
            "by_source": {k: fmt(v) for k, v in sorted(per_src.items())},
            "by_family": {k: fmt(v) for k, v in per_fam.items()},
            "by_stratum_group": {k: fmt(v) for k, v in sorted(per_group.items(), key=lambda kv: -sum(kv[1].values()))[:25]}}


def fmt_passk_row(name: str, s: dict) -> str:
    if not s or s.get("tasks_ge2", 0) == 0:
        return f"{name:<34} (no tasks with >=2 graded rollouts)"
    p = {k: s.get(f"pass@{k}") for k in KS}
    npt = s.get("n_per_task", {})
    return (f"{name:<34} {s['tasks_ge2']:>5} {s['rollouts_ge2']:>6} {npt.get('p50', 0):>4} {npt.get('max', 0):>4} "
            f"{s.get('pass1_all', 0):>6.3f} "
            + " ".join(f"{(p[k] if p[k] is not None else float('nan')):>6.3f}" for k in KS)
            + f" {s.get('pass@n', 0):>6.3f} {s.get('mixed_share', 0):>6.3f} {s.get('all_solved_share', 0):>6.3f} {s.get('all_failed_share', 0):>6.3f}")


PASSK_HEADER = (f"{'group':<34} {'tasks':>5} {'rolls':>6} {'n50':>4} {'nmax':>4} {'p1all':>6} "
                + " ".join(f"{'p@'+str(k):>6}" for k in KS) + f" {'p@n':>6} {'mixed':>6} {'allS':>6} {'allF':>6}")


def cmd_report(args: argparse.Namespace) -> None:
    work = Path(args.work)
    rows = load_rows(work)
    trows = teacher_rows(rows)
    print(f"{len(rows)} rollouts, {len(trows)} teacher, {len(king_rows(rows))} king")
    q1 = q1_teacher(trows)
    q2 = q2_kings(rows)
    q3 = {"teacher": divergence_stats(trows, "teacher (solved,failed) pairs, same task, same policy"),
          "teacher_any_harness": divergence_stats(trows, "teacher pairs, same task, ANY harness (cross-harness pairs diverge at 0 by construction)", False)}
    kr = [r for r in king_rows(rows) if in_window(r)
          and not (r["policy_id"] or "").endswith("_greedy")]
    # kings: pairs within one king only
    q3["kings_11_20"] = {}
    per_king = collections.defaultdict(list)
    for r in kr:
        per_king[r["reign"]].append(r)
    pooled_k = []
    for reign, sub in sorted(per_king.items()):
        # tag sid with reign so pairs never cross kings
        tagged = [dict(r, sid=f"{r['sid']}@{reign}") for r in sub]
        pooled_k.extend(tagged)
    q3["kings_11_20"] = divergence_stats(pooled_k, "king (solved,failed) pairs, same task, same king, same policy")
    q4 = {"continuation_cost": q4_cost(rows, q1), "d_turns_by_outcome": q4_d_share(rows, work)}
    q5 = band_proxy(trows)
    rep = {"generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
           "n_rollouts": len(rows), "n_teacher": len(trows), "n_king": len(king_rows(rows)),
           "q1_teacher": q1, "q2_kings": q2, "q3_divergence": q3, "q4_cost": q4, "q5_band_proxy": q5}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "report.json").write_text(json.dumps(rep, indent=1, default=str))

    lines = []
    P = lines.append
    P("Verified-references headroom (P3/A1 teacher-conditioned-on-success, A2 king-conditioned)")
    P(f"generated {rep['generated_at']}; traces manifest 2026-09-20T13:57Z (5,352 chunks, {len(rows):,} rollouts;"
      f" {len(trows):,} teacher_*, {rep['n_king']:,} king_*)")
    P("")
    P("Terms: p1all = pass@1 over ALL graded tasks incl. n=1; pass@1 = mean over tasks (n>=2) of solved/graded; pass@k = unbiased 1-C(n-c,k)/C(n,k) over tasks with n>=k;")
    P("       p@n = share of tasks with >=1 solve among their n rollouts; mixed = tasks with both a solved and a failed rollout")
    P("       (the only tasks where a success filter changes the reference set); allS/allF = all rollouts solved/failed.")
    P("       Tasks = (source, sid) with >=2 graded (solved|failed) rollouts; errored/unscored rollouts excluded.")
    P("")
    P("== Q1 TEACHER (all teacher_* policies, T=0.8 sampled; no greedy teacher policy exists) ==")
    P(PASSK_HEADER)
    P(fmt_passk_row("ALL (source,sid) any harness", q1["overall"]))
    P(fmt_passk_row("ALL (source,sid,policy) same", q1["overall_same_policy"]))
    P("-- by source, task=(source,sid) pooled over the teacher's harnesses (the brief's grouping)")
    for k, v in sorted(q1["by_source"].items(), key=lambda kv: -kv[1]["tasks_ge2"]):
        P(fmt_passk_row(k, v))
    P("-- by source, task=(source,sid,policy) = same sampler (genuine split)")
    for k, v in sorted(q1["by_source_same_policy"].items(), key=lambda kv: -kv[1]["tasks_ge2"]):
        P(fmt_passk_row(k, v))
    P("-- by harness (same policy)")
    for k, v in sorted(q1["by_harness_same_policy"].items(), key=lambda kv: -kv[1]["tasks_ge2"]):
        P(fmt_passk_row(k, v))
    P("-- by source|harness (same policy, >=5 tasks)")
    for k, v in sorted(q1["by_source_harness"].items(), key=lambda kv: -kv[1]["tasks_ge2"]):
        P(fmt_passk_row(k, v))
    P("-- coverage: how often the teacher REPLAYS a task at all (graded rollouts / distinct tasks / tasks with >=2)")
    for k, v in sorted(q1["coverage"].items(), key=lambda kv: -kv[1]["graded_rollouts"]):
        P(f"    {k:<20} graded={v['graded_rollouts']:>6} tasks={v['tasks']:>6} ge2_any={v['tasks_ge2_any_harness']:>5} ge2_same_policy={v['tasks_ge2_same_policy']:>5}")
    P(f"teacher outcome counts: {q1['outcome_counts']}")
    P(f"n rollouts/task histogram (ALL any-harness, n>=2): {q1['overall'].get('n_per_task', {}).get('hist')}")
    P("")
    P("== Q2 KINGS reigns 11-20 (+ revoked crowns 12r/13r that sat in the datagen seat; king_* policies; sampled T=0.8 unless *_greedy T=0) ==")
    P(f"king rollouts in window: {q2['n_king_rollouts']}; policies: {q2['king_policies']}")
    P("-- pass@k where one king replayed a task under the same policy (task=(source,sid,king,policy))")
    P(PASSK_HEADER)
    for lab in ("pooled_sampled", "pooled_greedy", "pooled_all"):
        P(fmt_passk_row(lab, q2[lab]))
    for reign, s in q2["by_reign"].items():
        P(fmt_passk_row(f"reign {reign} ({s['digest']})", s))
        P(f"    n_rollouts={s['n_rollouts']} ge2 tasks by source={s['by_source_ge2']}")
    P("-- king vs TEACHER on the SAME task under the SAME harness (n=1 each mostly): K+/T+ = solved at least once")
    P("   A2 'king|success' reaches king p@n; the plain teacher is teacher p@1; union = pooling both samplers' successes")
    def vs_line(name, v):
        if not v.get("n_tasks"):
            P(f"    {name:<26} (no shared tasks)"); return
        P(f"    {name:<26} tasks={v['n_tasks']:>5} king p@1={v['king_pass1']:.3f} p@n={v['king_passn']:.3f} | teacher p@1={v['teacher_pass1']:.3f} p@n={v['teacher_passn']:.3f}"
          f" | union p@n={v['union_passn']:.3f} | cells K+T+ {v['cells']['K+T+']} K+T- {v['cells']['K+T-']} K-T+ {v['cells']['K-T+']} K-T- {v['cells']['K-T-']}"
          f" | king-only {v['king_only_share']:.3f} teacher-only {v['teacher_only_share']:.3f}")
        P("      by group: " + "; ".join(f"{g}: n={c['n']} K={c['king_passn']:.2f} T={c['teacher_passn']:.2f} K-only={c['king_only']:.2f} T-only={c['teacher_only']:.2f}"
                                          for g, c in v["by_group"].items()))
    vs_line("pooled sampled kings", q2["vs_teacher_pooled_sampled"])
    vs_line("pooled greedy kings", q2["vs_teacher_pooled_greedy"])
    for reign, s in q2["by_reign"].items():
        vs_line(f"reign {reign} (sampled)", s["vs_teacher"])
    P("")
    P("== Q3 FIRST-DIVERGENCE DEPTH (mixed tasks; every (solved, failed) pair; main-root actions, norm_action by dialect) ==")
    for key in ("teacher", "teacher_any_harness", "kings_11_20"):
        d = q3[key]
        P(f"-- {d['label']}: pairs={d.get('n_pairs')} mixed_tasks={d.get('n_mixed_tasks')}")
        if d.get("n_pairs"):
            P(f"   depth p50={d['depth']['p50']} mean={d['depth']['mean']} | d=0: {d['depth']['d0']:.1%}  d<=1: {d['depth']['d_le1']:.1%}"
              f"  d<=2: {d['depth']['d_le2']:.1%}  d>=5: {d['depth']['d_ge5']:.1%}")
            P(f"   hist (capped at 10): {d['depth_hist_capped10']}")
            P(f"   pre-divergence share of turns: pair-mean {d['pre_divergence_share_pairmean']:.1%}, turn-weighted {d['pre_divergence_share_turnweighted']:.1%}")
            P("   by harness: " + "; ".join(f"{h}: n={v['n']} p50={v['p50']} d0={v['d0']:.0%} d<=2={v['d_le2']:.0%}"
                                             for h, v in d["by_harness"].items()))
            P("   by source: " + "; ".join(f"{h}: n={v['n']} p50={v['p50']} d0={v['d0']:.0%}"
                                            for h, v in d["by_source"].items()))
    P("")
    P("== Q4 COST of verified refs at fold time ==")
    cc = q4["continuation_cost"]
    P(f"sibling continuations (Engy teacher, resume + run to end): n={cc.get('continuations_n')} wall_s={cc.get('wall_s')} cost_usd={cc.get('cost_usd')}")
    P(f"  by kind: {cc.get('by_kind')}")
    P(f"  at-depth state resample (orig outcome vs one fresh teacher continuation from the same state, contested pivots depth 3-15): {cc.get('at_depth_split')}")
    for lab in ("pivots_10pct", "mixed_task_share"):
        if lab in cc:
            P(f"  {lab}: {cc[lab]}")
    ds = q4["d_turns_by_outcome"]
    P(f"D (epoch 58 index, {ds.get('n_turns')} turns) joined to trace outcomes by rollout_id:")
    P(f"  by policy family: {ds.get('by_family')}")
    P("  by stratum group (fold groups):")
    for k, v in ds.get("by_stratum_group", {}).items():
        P(f"    {k:<18} n={v['n_turns']:>6} solved={v.get('solved', 0):.3f} failed={v.get('failed', 0):.3f} errored={v.get('errored', 0):.3f} not_in_traces={v.get('not_in_traces', 0):.3f}")
    P("  by source (share of D turns from solved / failed / other rollouts):")
    for k, v in ds.get("by_source", {}).items():
        P(f"    {k:<22} n={v['n_turns']:>6} solved={v.get('solved', 0):.3f} failed={v.get('failed', 0):.3f} "
          f"unscored={v.get('unscored', 0):.3f} errored={v.get('errored', 0):.3f} not_in_traces={v.get('not_in_traces', 0):.3f}")
    P("")
    P("== Q5 BAND PROXY: thought shape of solved vs failed TEACHER replies at matched depth, per harness ==")
    P("-- pooled over depth >= 1 (agentic regime): latent chars p50/p90, share of replies with no visible prose, share truncated (finish=length)")
    for h, d in q5["pooled_depth_ge1"].items():
        P(f"   {h:<20} S n={d['n_solved']:>6} lat p50/p90={d['lat_p50_solved']:>5.0f}/{d['lat_p90_solved']:>6.0f} noVis={d['no_visible_solved']:.2f} trunc={d['finish_length_solved']:.3f}"
          f" | F n={d['n_failed']:>6} lat p50/p90={d['lat_p50_failed']:>5.0f}/{d['lat_p90_failed']:>6.0f} noVis={d['no_visible_failed']:.2f} trunc={d['finish_length_failed']:.3f}")
    P("-- by depth bucket")
    P("   (lat = latent/reasoning chars p50; vis = visible chars p50; alen = action chars p50; noLat/noVis = share of replies with none)")
    for h, bs in sorted(q5["by_harness"].items()):
        for b, d in bs.items():
            if "solved" in d and "failed" in d:
                s_, f_ = d["solved"], d["failed"]
                P(f"   {h:<20} {b:<6} solved n={s_['n']:>6} lat={s_['lat_p50']:>6.0f} vis={s_['vis_p50']:>5.0f} alen={s_['alen_p50']:>5.0f} noLat={s_['no_latent']:.2f} noVis={s_['no_visible']:.2f} len={s_['finish_length']:.2f}"
                  f" | failed n={f_['n']:>6} lat={f_['lat_p50']:>6.0f} vis={f_['vis_p50']:>5.0f} alen={f_['alen_p50']:>5.0f} noLat={f_['no_latent']:.2f} noVis={f_['no_visible']:.2f} len={f_['finish_length']:.2f}"
                  f" | lat ratio S/F={d.get('lat_p50_ratio_solved_over_failed')}")
    P("")
    P("== VERDICT (numbers above) ==")
    o, osp = q1["overall"], q1["overall_same_policy"]
    vs = q2["vs_teacher_pooled_sampled"]
    d3 = q3["teacher"]
    ad = cc.get("at_depth_split", {})
    P(f"1. Teacher task-level headroom is thin: same-sampler pass@1 {osp.get('pass@1')} -> pass@2 {osp.get('pass@2')} -> pass@n {osp.get('pass@n')};"
      f" only {osp.get('mixed_share', 0):.1%} of replayed tasks are mixed (any-harness pooling: {o.get('mixed_share', 0):.1%}); the 42-50 % terminal_bench_2 figure is the outlier (89 tasks x 7-11 harness-pooled runs).")
    P(f"2. The teacher runs every coding/terminal task ONCE (swesmith/multiswe/scaleswe/swerebench/swelego/terminal_lego: 0 tasks with >=2 rollouts), so pass@n there is unmeasurable from traces; the only depth evidence is the sibling's 24 resampled pivot states: {ad.get('flip_share', 0):.0%} flip with one resample, implied N=4 split {ad.get('orig_solved', {}).get('p_mixed_N4')} at solved-trajectory states / {ad.get('orig_failed', {}).get('p_mixed_N4')} at failed-trajectory states.")
    P(f"3. Kings never replay their own tasks except math under reign 11 (pass@1 {q2['by_reign'].get('11', {}).get('pass@1')} -> pass@2 {q2['by_reign'].get('11', {}).get('pass@2')}, mixed {q2['by_reign'].get('11', {}).get('mixed_share')}); on the SAME task+harness the king solves {vs.get('king_pass1')} vs teacher {vs.get('teacher_pass1')},"
      f" and only {vs.get('king_only_share', 0):.1%} of tasks are king-solved-teacher-failed: 'king|success' cannot exceed the plain teacher, it can only close part of a {vs.get('teacher_pass1', 0) - vs.get('king_pass1', 0):.2f} gap.")
    P(f"4. Where a same-sampler split exists it is at the ROOT: first-divergence depth 0 on {d3['depth']['d0']:.0%} of pairs, <=1 on {d3['depth']['d_le1']:.0%}; pre-divergence turns are {d3['pre_divergence_share_turnweighted']:.1%} of pair turns -> a success filter re-labels the sampled state itself, never a shared prefix, so every scored turn needs its own N continuations.")
    P(f"5. Cost: {cc.get('wall_s', {}).get('mean')} s / ${cc.get('cost_usd', {}).get('mean')} Engy per continuation -> N=4 at 10 % pivots = {cc.get('pivots_10pct', {}).get('container_hours_per_1000_new_turns')} container-h + ${cc.get('pivots_10pct', {}).get('usd_per_1000_new_turns')} per 1,000 new D turns ({cc.get('pivots_10pct', {}).get('container_hours_per_1000_pivot_turns')} h per 1,000 pivot turns);"
      f" the zero-continuation variant (refs from solved trajectories only) keeps {ds['by_family'].get('teacher', {}).get('solved', 0):.0%} of teacher D turns (coding {ds['by_stratum_group'].get('coding', {}).get('solved', 0):.0%}, terminal {ds['by_stratum_group'].get('terminal', {}).get('solved', 0):.0%}) but drops every failed-trajectory prefix, which is exactly the king_fail material D was built to carry.")
    (RESULTS / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n-> {RESULTS / 'report.txt'} / report.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("index")
    a.add_argument("--chunks", default="/tmp/fa/chunks")
    a.add_argument("--out", default="/tmp/vr")
    a.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a.set_defaults(fn=cmd_index)
    b = sub.add_parser("report")
    b.add_argument("--work", default="/tmp/vr")
    b.set_defaults(fn=cmd_report)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
