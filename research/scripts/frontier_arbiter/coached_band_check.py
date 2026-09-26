"""N5 — coached references (He, Daumé & Eisner 2012 "hope actions"): at states where a
verified-BETTER action exists that the blind teacher did not take, is that action still
PLAUSIBLE under the blind teacher, and is the thought that produced it inside the
typicality band?

    python coached_band_check.py build     # states, actions, thoughts from the stored probes (no API)
    python coached_band_check.py echo      # teacher echoes (reuses advisor_ppi/probe/echoes.jsonl)
    python coached_band_check.py report    # results/frontier_arbiter/coached/report.{txt,json}
  Intermediate files (states.jsonl, echoes.jsonl, cost.jsonl) live in $FA_COACHED_SCRATCH
  (default /tmp/fa_coached); only the two reports are written under results/.

Terms (one line each):
  state             a graded prefix x of the outcome / split-states probes (rollout, turn_idx); prefix =
                    the advisor_ppi materialisation (/tmp/fa_ppi/prefix, trace mirror + ToolBaker).
  teacher-failed    first action of an arm-T teacher continuation from x that the env graded failed.
  teacher-solved    first action of an arm-T teacher continuation from x that the env graded solved.
  blind ref         one of the 3 stored blind teacher samples at x (outcome/samples.jsonl), unlabelled.
  frontier proposal the forced glm-5.3 first action of an arm-F1/F1b/F1c continuation (trace node just
                    before the first sampled reply); solved/failed = that continuation's env grade
                    (teacher finished the trajectory).
  stuck state       a state with >= 1 teacher-failed action.
  verified-better   at a stuck state, an action whose continuation solved: teacher-solved (tier T) or
                    frontier proposal (tier F1); arm F (frontier continues itself) is tier F, reported
                    separately.
  lp0               lpC(y | x, ∅): teacher logprob of the action bytes with no thought (per byte + summed).
  lpB               mean_j lpC(y | x, z_blind_j): the same under each stored blind teacher thought.
  teacher range     [min, max] of lp0 over the teacher's own actions at x (failed first actions, or
                    failed + blind refs); in-range = inside it; >= floor = at or above its min.
  z-pos             (lp0(v) − mean_T) / sd_T over the teacher's own actions at x (needs >= 2).
  m_c               content-masked mean token logprob of a thought under x (live content_stats, θ = 1).
  band              μ_c = mean m_c of the blind refs at x; σ_c pooled per dialect (sqrt mean within-state
                    variance of the blind m_c) and the live verdict σ_Mc; in-band = |z| <= 2 & >= 10 content
                    tokens (live typ_c >= 0).
  coached pick      at a stuck state with >= 1 verified-better action, the one with the highest lp0
                    (per byte; summed as a check); Δ = lp0(coached) − max lp0(teacher-failed).
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import glob
import hashlib
import json
import math
import os
import statistics as st
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "outcome"))
from common import (  # noqa: E402
    REPO, Engy, TeacherEcho, append_jsonl, norm_action, read_jsonl, write_jsonl,
)
from advisor_ppi_checks import echo_form, load_prefix, prefix_path  # noqa: E402
from privileged_refs import TURNS  # noqa: E402
from solved_band import reply_zy  # noqa: E402
from split_yield import first_action  # noqa: E402
from vav_sim import canon  # noqa: E402
from evalsrv.sdmeter import content_stats  # noqa: E402

FA = REPO / "research" / "results" / "frontier_arbiter"
OUT = FA / "coached"                       # report.txt + report.json only
SCRATCH = Path(os.environ.get("FA_COACHED_SCRATCH", "/tmp/fa_coached"))   # states / echoes / cost cache
STATES = SCRATCH / "states.jsonl"
ECHOES = SCRATCH / "echoes.jsonl"
COST = SCRATCH / "cost.jsonl"
PPI_ECHOES = FA / "advisor_ppi" / "probe" / "echoes.jsonl"
TRACE_DIRS = (Path("/tmp/fa_split/collect/traces"), Path("/tmp/fa_outcome/collect/traces"))
ACTION_KIND = {"mini_swe_textbased": "bash", "bash": "tool_call", "terminus_2": "terminus_json"}
K = 3
THETA = 1.0
WIDTH = 2.0
MIN_CONTENT = 10
BUDGET_USD = 6.0
KILL_SHARE = 0.70


def h(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:12]


def base_id(state_id: str) -> str:
    return state_id.rsplit(":", 1)[0]


def akey(sid: str, z: str, y: str) -> str:
    """advisor_ppi's echo key — lets lpC(y|x, z) rows be reused verbatim."""
    return f"{sid}|{h(z)}|{h(y)}"


def total_cost() -> float:
    groups: dict[str, list[float]] = collections.defaultdict(list)
    for r in read_jsonl(COST):
        groups[r["run"]].append(r["cost_usd"])
    return sum(max(v) for v in groups.values())


def log_cost(engy: Engy, run: str, note: str = "") -> None:
    append_jsonl(COST, {"at": time.time(), "run": run, "cost_usd": engy.cost_usd, "usage": engy.usage, "note": note})
    tot = total_cost()
    print(f"  [$] this run ${engy.cost_usd:.3f} | total ${tot:.2f} {note}", flush=True)
    if tot > BUDGET_USD:
        raise SystemExit(f"budget exceeded: ${tot:.2f} > ${BUDGET_USD}")


# ------------------------------------------------------------------ build
def trace_files() -> dict[str, Path]:
    out = {}
    for d in TRACE_DIRS:
        for f in glob.glob(str(d / "*.json")):
            try:
                out[json.load(open(f))["id"]] = Path(f)
            except (ValueError, KeyError):
                continue
    return out


def first_sampled(nodes: list[dict]) -> int | None:
    return next((i for i, n in enumerate(nodes) if n.get("sampled") and n["message"]["role"] == "assistant"), None)


def node_zy(m: dict, harness: str, kind: str) -> dict:
    return reply_zy(m.get("reasoning_content") or "", m.get("content") or "", m.get("tool_calls"), harness, kind)


def forced_proposal(nodes: list[dict], fi: int, harness: str, kind: str) -> dict | None:
    """The unsampled assistant node injected before the teacher's first sampled reply."""
    for i in range(fi - 1, -1, -1):
        n = nodes[i]
        if n["message"]["role"] == "assistant" and not n.get("sampled"):
            zy = node_zy(n["message"], harness, kind)
            zy["node"] = i
            return zy
        if n.get("sampled"):
            break
    return None


def rebuild_blind(b: dict, harness: str, kind: str) -> dict:
    """The stored blind refs' `z` (outcome/samples.jsonl, 2026-09-20 pipeline) is reasoning + "\\n" +
    the THOUGHT-stripped visible text with no </think>; echoing it as-is renders the visible prose
    as latent. Rebuild the live (as_generated) thought from the stored reasoning_chars + raw_content."""
    z_old = (b.get("z") or "").strip()
    raw = b.get("raw_content")
    rc = int(b.get("reasoning_chars") or 0)
    if raw is None:
        return {"z": z_old, "y": canon(b["y"], kind), "rebuilt": False, "y_parity": None}
    zy = reply_zy((b.get("z") or "")[:rc], raw, b.get("tool_calls") or None, harness, kind)
    y_new = canon(zy["y"], kind) if zy["parsed"] else None
    parity = bool(y_new) and norm_action(y_new, kind) == norm_action(canon(b["y"], kind), kind)
    return {"z": zy["z"].strip() if zy["parsed"] else z_old, "y": canon(b["y"], kind), "rebuilt": zy["parsed"],
            "y_parity": parity, "z_old": z_old}


def match_frontier_thought(kept: dict | None, y: str, kind: str) -> tuple[str | None, str | None]:
    """The GLM thought (latent + visible) that produced proposal y, from the kept-state samples."""
    if not kept:
        return None, None
    ny = norm_action(canon(y, kind), kind)
    for tag in ("frontier_greedy", "frontier_t08"):
        fr = kept.get(tag) or {}
        if fr.get("y") and norm_action(canon(fr["y"], kind), kind) == ny:
            return (fr.get("z") or "").strip(), tag
    return None, None


def build() -> list[dict]:
    conts = read_jsonl(FA / "split_states" / "continuations.jsonl") + read_jsonl(FA / "outcome" / "continuations.jsonl")
    ok = [c for c in conts if c.get("status") == "ok" and c.get("outcome") in ("solved", "failed")]
    seen_tr: set[str] = set()
    by: dict[str, list[dict]] = collections.defaultdict(list)
    for c in ok:
        tid = c.get("trace_id") or f"{c['state_id']}#{c['continuation']}"
        if tid in seen_tr:      # the 2026-09-20 arm-T rows were seeded into the split run too
            continue
        seen_tr.add(tid)
        by[base_id(c["kept_state_id"])].append(c)
    smp = {base_id(s["state_id"]): s for s in read_jsonl(FA / "outcome" / "samples.jsonl")}
    kept = {base_id(k["state_id"]): k for k in read_jsonl(FA / "outcome" / "kept.jsonl")}
    props = {base_id(p["state_id"]): p for p in read_jsonl(FA / "split_states" / "proposals.jsonl")}
    traces = trace_files()
    out = []
    skipped = collections.Counter()
    for sid, cs in sorted(by.items()):
        s = smp.get(sid)
        if not s or not prefix_path(sid).exists():
            skipped["no_samples_or_prefix"] += 1
            continue
        harness = s["harness"]
        kind = ACTION_KIND[harness]
        judge_by_y = {}
        for pp in (props.get(sid) or {}).get("proposals", []):
            judge_by_y[norm_action(canon(pp["y"], kind), kind)] = pp.get("judge") or {}
        actions = []
        for c in sorted(cs, key=lambda c: (c["arm"], c["continuation"])):
            tr = traces.get(c.get("trace_id"))
            nodes = json.load(open(tr))["nodes"] if tr else None
            fi = first_sampled(nodes) if nodes else None
            if c["arm"] == "T":
                y = canon(first_action(c, kind), kind)
                if not y:
                    skipped["T_unparsed"] += 1
                    continue
                z = None
                if nodes and fi is not None:
                    zy = node_zy(nodes[fi]["message"], harness, kind)
                    z = zy["z"].strip() if zy["parsed"] else None
                    if zy["parsed"] and norm_action(zy["y"], kind) != norm_action(y, kind):
                        skipped["T_trace_action_mismatch"] += 1
                actions.append({"src": "T", "arm": "T", "cont": c["continuation"], "y": y, "outcome": c["outcome"],
                                "z": z, "z_src": "teacher_trace" if z else None, "trace": bool(nodes)})
            elif c["arm"].startswith("F1"):
                if not nodes or fi is None:
                    skipped["F1_no_trace"] += 1
                    continue
                fp = forced_proposal(nodes, fi, harness, kind)
                if not fp or not fp["parsed"]:
                    skipped["F1_no_forced_node"] += 1
                    continue
                y = canon(fp["y"], kind)
                z_full, tag = match_frontier_thought(kept.get(sid), y, kind)
                actions.append({"src": "F1", "arm": c["arm"], "cont": c["continuation"], "y": y, "outcome": c["outcome"],
                                "z": z_full, "z_src": tag, "z_visible": fp["z"].strip() or None,
                                "judge": judge_by_y.get(norm_action(y, kind)), "trace": True})
            elif c["arm"] == "F":
                if not nodes or fi is None:
                    skipped["F_no_trace"] += 1
                    continue
                zy = node_zy(nodes[fi]["message"], harness, kind)
                if not zy["parsed"]:
                    skipped["F_unparsed"] += 1
                    continue
                actions.append({"src": "F", "arm": "F", "cont": c["continuation"], "y": canon(zy["y"], kind),
                                "outcome": c["outcome"], "z": zy["z"].strip() or None, "z_src": "frontier_trace",
                                "trace": True})
        blind = []
        for i, b in enumerate([b for b in s["teacher"] if b.get("parsed") and b.get("y")][:K]):
            rb = rebuild_blind(b, harness, kind)
            skipped["blind_rebuilt" if rb["rebuilt"] else "blind_not_rebuilt"] += 1
            if rb["rebuilt"] and not rb["y_parity"]:
                skipped["blind_y_parity_fail"] += 1
            blind.append({"src": "B", "i": i, **rb})
        tf = [a for a in actions if a["src"] == "T" and a["outcome"] == "failed"]
        vb = [a for a in actions if a["src"] in ("T", "F1") and a["outcome"] == "solved"]
        out.append({"sid": sid, "harness": harness, "kind": kind, "source": s["source"], "group": s["group"],
                    "orig_outcome": s["orig_outcome"], "depth": int(s["depth"]),
                    "stuck": bool(tf), "n_T_failed": len(tf),
                    "n_T_solved": sum(1 for a in actions if a["src"] == "T" and a["outcome"] == "solved"),
                    "n_F1_solved": sum(1 for a in actions if a["src"] == "F1" and a["outcome"] == "solved"),
                    "n_F1_failed": sum(1 for a in actions if a["src"] == "F1" and a["outcome"] == "failed"),
                    "n_F_solved": sum(1 for a in actions if a["src"] == "F" and a["outcome"] == "solved"),
                    "n_verified_better": len(vb) if tf else 0,
                    "actions": actions, "blind": blind})
    write_jsonl(STATES, out)
    n_stuck = sum(1 for r in out if r["stuck"])
    n_sv = sum(1 for r in out if r["stuck"] and r["n_verified_better"])
    print(f"states {len(out)} | stuck {n_stuck} | stuck with >= 1 verified-better (T∪F1) {n_sv} | "
          f"verified-better actions {sum(r['n_verified_better'] for r in out)} | "
          f"F1 proposals {sum(r['n_F1_solved'] + r['n_F1_failed'] for r in out)} "
          f"(solved {sum(r['n_F1_solved'] for r in out)}) | skipped {dict(skipped)}")
    return out


def load_states() -> list[dict]:
    return read_jsonl(STATES) if STATES.exists() else build()


# ------------------------------------------------------------------ echo
def thought_keys(sid: str, tag: str) -> tuple[str, str]:
    return f"{sid}|th|{tag}", f"{sid}|un|{tag}"


def echo_jobs(states: list[dict]) -> list[tuple]:
    """(key, kind, sid, z, y) — kind = action | thought | uncond. Action echoes use advisor_ppi keys."""
    jobs: dict[str, tuple] = {}
    for stt in states:
        sid = stt["sid"]
        blind_z = [b["z"] for b in stt["blind"] if b["z"]]
        ys = [a["y"] for a in stt["actions"]] + [b["y"] for b in stt["blind"]]
        for y in dict.fromkeys(ys):
            jobs.setdefault(akey(sid, "", y), (akey(sid, "", y), "action", sid, "", y))
            for z in blind_z:
                jobs.setdefault(akey(sid, z, y), (akey(sid, z, y), "action", sid, z, y))
        for i, b in enumerate(stt["blind"]):
            if b["z"]:
                th, un = thought_keys(sid, f"B{i}v2")
                jobs.setdefault(th, (th, "thought", sid, b["z"], None))
                jobs.setdefault(un, (un, "uncond", sid, b["z"], None))
        for a in stt["actions"]:
            tag = f"{a['src']}{a['cont']}" if a["src"] != "F1" else f"{a['arm']}{a['cont']}"
            if a.get("z"):
                th, un = thought_keys(sid, tag)
                jobs.setdefault(th, (th, "thought", sid, a["z"], None))
                jobs.setdefault(un, (un, "uncond", sid, a["z"], None))
            if a.get("z_visible"):
                th, un = thought_keys(sid, tag + "vis")
                jobs.setdefault(th, (th, "thought", sid, a["z_visible"], None))
                jobs.setdefault(un, (un, "uncond", sid, a["z_visible"], None))
    return list(jobs.values())


def stored_echoes() -> dict[str, dict]:
    out = {r["key"]: r for r in read_jsonl(PPI_ECHOES) if "error" not in r}
    out.update({r["key"]: r for r in read_jsonl(ECHOES) if "error" not in r})
    return out


async def echo_all(states: list[dict], engy: Engy, run: str, limit: int | None) -> None:
    te = TeacherEcho(engy)
    done = stored_echoes()
    jobs = [j for j in echo_jobs(states) if j[0] not in done]
    n_reused = len(echo_jobs(states)) - len(jobs)
    if limit:
        jobs = jobs[:limit]
    print(f"echo: {len(jobs)} to run, {n_reused} reused from stored echoes")
    kinds = {s["sid"]: s["kind"] for s in states}
    prefixes: dict[str, list[dict]] = {}
    queue: asyncio.Queue = asyncio.Queue()
    for j in jobs:
        queue.put_nowait(j)
    n_done = 0

    async def worker() -> None:
        nonlocal n_done
        while True:
            try:
                key, kind, sid, z, y = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            if sid not in prefixes:
                prefixes[sid] = load_prefix(sid)
            try:
                if kind == "action":
                    r = await te.lp_action(prefixes[sid], z, echo_form(y, kinds[sid]))
                    append_jsonl(ECHOES, {"key": key, "sid": sid, "why": "action", **r})
                elif kind == "thought":
                    r = await te.lp_thought(prefixes[sid], z, tokens=True)
                    append_jsonl(ECHOES, {"key": key, "sid": sid, "why": "thought", **r})
                else:
                    r = await te.lp_thought_uncond(z, tokens=True)
                    append_jsonl(ECHOES, {"key": key, "sid": sid, "why": "uncond", **r})
            except Exception as ex:  # noqa: BLE001
                append_jsonl(ECHOES, {"key": key, "sid": sid, "why": kind, "error": repr(ex)[:300]})
            n_done += 1
            if n_done % 50 == 0 or n_done == len(jobs):
                log_cost(engy, run, f"{n_done}/{len(jobs)}")

    await asyncio.gather(*[worker() for _ in range(12)])


def cmd_echo(args: argparse.Namespace) -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    engy = Engy(concurrency=12, retries=5)
    asyncio.run(echo_all(load_states(), engy, run=f"echo-{int(time.time())}", limit=args.limit))


# ------------------------------------------------------------------ report helpers
def _mean(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return st.mean(v) if v else None


def _p50(v):
    v = [x for x in v if x is not None and isinstance(x, (int, float)) and math.isfinite(x)]
    return st.median(v) if v else None


def _rate(f):
    f = [x for x in f if x is not None]
    return (sum(1.0 for x in f if x) / len(f)) if f else None


def _f(x, w=6, p=3):
    if x is None:
        return " " * (w - 3) + "n/a"
    return f"{x:{w}.{p}f}"


def _pct(x, w=5):
    return " " * (w - 3) + "n/a" if x is None else f"{100 * x:{w}.0f}%"


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    z = 1.96
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - hw), min(1.0, c + hw))


def mc_of(ex: dict | None, eu: dict | None) -> tuple[float | None, int | None]:
    if not ex or not eu:
        return None, None
    cs = content_stats([tuple(x) for x in ex["tokens"]], [tuple(x) for x in eu["tokens"]], THETA)
    return cs["mc"], cs["n_content"]


def live_sigma() -> dict[str, float]:
    out = {}
    for t in read_jsonl(TURNS):
        if t.get("sigma_mc"):
            out.setdefault(t["dialect"], t["sigma_mc"])
    return out


# ------------------------------------------------------------------ report
def annotate(states: list[dict], E: dict[str, dict]) -> tuple[list[dict], dict]:
    """Per state: lp0 / lpB per action and blind ref; m_c per thought; band + z per thought."""
    for stt in states:
        sid = stt["sid"]
        blind_z = [b["z"] for b in stt["blind"] if b["z"]]
        for a in stt["actions"] + stt["blind"]:
            e0 = E.get(akey(sid, "", a["y"]))
            a["lp0_pb"] = e0["lp_per_byte"] if e0 else None
            a["lp0_sum"] = e0["sum_lp"] if e0 else None
            a["n_bytes"] = e0["n_bytes"] if e0 else None
            eb = [E.get(akey(sid, z, a["y"])) for z in blind_z]
            a["lpB_pb"] = _mean([e["lp_per_byte"] for e in eb if e])
            a["lpB_sum"] = _mean([e["sum_lp"] for e in eb if e])
        for i, b in enumerate(stt["blind"]):
            th, un = thought_keys(sid, f"B{i}v2")
            b["mc"], b["nc"] = mc_of(E.get(th), E.get(un))
        for a in stt["actions"]:
            tag = f"{a['src']}{a['cont']}" if a["src"] != "F1" else f"{a['arm']}{a['cont']}"
            th, un = thought_keys(sid, tag)
            a["mc"], a["nc"] = mc_of(E.get(th), E.get(un))
            if a.get("z_visible"):
                th, un = thought_keys(sid, tag + "vis")
                a["mc_vis"], a["nc_vis"] = mc_of(E.get(th), E.get(un))
        mcb = [b["mc"] for b in stt["blind"] if b.get("mc") is not None]
        stt["mu_c"] = st.mean(mcb) if len(mcb) >= 2 else None
        stt["var_c"] = st.variance(mcb) if len(mcb) >= 2 else None
        stt["n_blind_mc"] = len(mcb)
    sig_pooled = {}
    for kd in ("bash", "tool_call", "terminus_json"):
        vs = [s["var_c"] for s in states if s["kind"] == kd and s["n_blind_mc"] == K]
        sig_pooled[kd] = math.sqrt(st.mean(vs)) if vs else None
    sig_live = live_sigma()
    sigmas = {"pooled": sig_pooled, "live": sig_live}
    for stt in states:
        for a in stt["actions"] + stt["blind"]:
            for tag, sig in (("pooled", sig_pooled.get(stt["kind"])), ("live", sig_live.get(stt["kind"]))):
                for suf in ("", "_vis"):
                    mc = a.get("mc" + suf)
                    if suf and "mc_vis" not in a:
                        continue
                    z = ((mc - stt["mu_c"]) / sig) if (mc is not None and stt["mu_c"] is not None and sig) else None
                    a[f"z_{tag}{suf}"] = z
                    a[f"inband_{tag}{suf}"] = (z is not None and abs(z) <= WIDTH and (a.get("nc" + suf) or 0) >= MIN_CONTENT)
        # blind LOO (the band's own hit rate): each blind ref vs the other two
        mcb = [b.get("mc") for b in stt["blind"]]
        if all(m is not None for m in mcb) and len(mcb) == K and sig_live.get(stt["kind"]):
            for j, b in enumerate(stt["blind"]):
                others = [mcb[i] for i in range(K) if i != j]
                zz = (mcb[j] - st.mean(others)) / sig_live[stt["kind"]]
                b["loo_inband_live"] = abs(zz) <= WIDTH and (b.get("nc") or 0) >= MIN_CONTENT
    return states, sigmas


def plaus_rows(states: list[dict], ref: str, tiers: tuple[str, ...], metric: str) -> list[dict]:
    """One row per verified-better action at a stuck state: position vs the teacher's own actions.
    ref = 'failed' (teacher-failed first actions) | 'failed+blind' (+ the stored blind refs)."""
    rows = []
    for stt in states:
        if not stt["stuck"]:
            continue
        tf = [a for a in stt["actions"] if a["src"] == "T" and a["outcome"] == "failed" and a.get(metric) is not None]
        refs = list(tf)
        if ref == "failed+blind":
            refs += [b for b in stt["blind"] if b.get(metric) is not None]
        # dedup by normalised action: the same command sampled twice is one teacher action
        seen = {}
        for r in refs:
            seen.setdefault(norm_action(r["y"], stt["kind"]), r)
        refs = list(seen.values())
        vals = [r[metric] for r in refs]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        mu = st.mean(vals)
        sd = st.stdev(vals) if len(vals) >= 2 else None
        best_failed = max(a[metric] for a in tf) if tf else None
        for a in stt["actions"]:
            if a["src"] not in tiers or a["outcome"] != "solved" or a.get(metric) is None:
                continue
            v = a[metric]
            rows.append({"sid": stt["sid"], "harness": stt["harness"], "kind": stt["kind"], "src": a["src"], "arm": a["arm"],
                         "cont": a["cont"], "v": v, "lo": lo, "hi": hi, "n_ref": len(vals),
                         "in_range": lo <= v <= hi, "ge_floor": v >= lo, "gt_max": v > hi,
                         "z": ((v - mu) / sd) if sd else None, "delta_best_failed": (v - best_failed) if best_failed is not None else None,
                         "same_as_teacher": any(norm_action(a["y"], stt["kind"]) == k for k in seen)})
    return rows


def agg_plaus(rows: list[dict]) -> dict:
    n = len(rows)
    ir = sum(1 for r in rows if r["in_range"])
    gf = sum(1 for r in rows if r["ge_floor"])
    r2 = [r for r in rows if r["n_ref"] >= 2]
    return {"n": n, "n_states": len({r["sid"] for r in rows}),
            "in_range": ir / n if n else None, "in_range_ci": wilson(ir, n) if n else None,
            "n_ref2": len(r2), "in_range_ref2": _rate([r["in_range"] for r in r2]),
            "ge_floor": gf / n if n else None, "ge_floor_ci": wilson(gf, n) if n else None,
            "gt_max": _rate([r["gt_max"] for r in rows]),
            "z_mean": _mean([r["z"] for r in rows]), "z_p50": _p50([r["z"] for r in rows]),
            "n_z": sum(1 for r in rows if r["z"] is not None),
            "delta_best_failed_mean": _mean([r["delta_best_failed"] for r in rows]),
            "delta_best_failed_p50": _p50([r["delta_best_failed"] for r in rows]),
            "same_as_teacher": _rate([r["same_as_teacher"] for r in rows])}


def paired_solved_failed(states: list[dict], metric: str) -> dict:
    """Split states: mean lp0 of teacher-solved minus teacher-failed first actions, paired per state."""
    diffs, by_h = [], collections.defaultdict(list)
    for stt in states:
        ts = [a[metric] for a in stt["actions"] if a["src"] == "T" and a["outcome"] == "solved" and a.get(metric) is not None]
        tf = [a[metric] for a in stt["actions"] if a["src"] == "T" and a["outcome"] == "failed" and a.get(metric) is not None]
        if ts and tf:
            d = st.mean(ts) - st.mean(tf)
            diffs.append(d)
            by_h[stt["harness"]].append(d)
    def summ(v):
        if not v:
            return {"n": 0}
        se = (st.stdev(v) / math.sqrt(len(v))) if len(v) >= 2 else None
        return {"n": len(v), "mean": st.mean(v), "p50": st.median(v), "se": se,
                "share_pos": sum(1 for x in v if x > 0) / len(v)}
    return {"ALL": summ(diffs), **{k: summ(v) for k, v in by_h.items()}}


def thought_groups(states: list[dict]) -> dict[str, list[dict]]:
    g: dict[str, list[dict]] = collections.defaultdict(list)
    for stt in states:
        for a in stt["actions"]:
            if a.get("mc") is None:
                continue
            row = {"sid": stt["sid"], "harness": stt["harness"], "kind": stt["kind"], "stuck": stt["stuck"], **a}
            if a["src"] == "T":
                g[f"teacher first thought, {a['outcome']}"].append(row)
                if stt["stuck"] and a["outcome"] == "solved":
                    g["teacher first thought, solved @stuck (split states)"].append(row)
            elif a["src"] == "F1":
                g[f"frontier proposal thought (GLM latent+visible), {a['outcome']}"].append(row)
                if stt["stuck"]:
                    g[f"frontier proposal thought (GLM latent+visible), {a['outcome']} @stuck"].append(row)
            elif a["src"] == "F":
                g[f"frontier own-continuation first thought, {a['outcome']}"].append(row)
        for a in stt["actions"]:
            if a["src"] == "F1" and a.get("mc_vis") is not None:
                row = {"sid": stt["sid"], "harness": stt["harness"], "kind": stt["kind"], "stuck": stt["stuck"],
                       **{k: v for k, v in a.items()},
                       "mc": a["mc_vis"], "nc": a["nc_vis"], "z_pooled": a.get("z_pooled_vis"), "z_live": a.get("z_live_vis"),
                       "inband_pooled": a.get("inband_pooled_vis"), "inband_live": a.get("inband_live_vis")}
                g[f"frontier proposal visible text only, {a['outcome']}"].append(row)
        for b in stt["blind"]:
            if b.get("loo_inband_live") is not None:
                g["blind ref LOO (band's own hit rate)"].append({"sid": stt["sid"], "harness": stt["harness"], "kind": stt["kind"],
                                                                 "inband_live": b["loo_inband_live"], "inband_pooled": None,
                                                                 "z_live": None, "z_pooled": None, "mc": b["mc"], "nc": b["nc"]})
    return g


def agg_thought(rows: list[dict]) -> dict:
    n = len(rows)
    il = [r.get("inband_live") for r in rows]
    return {"n": n, "n_states": len({r["sid"] for r in rows}),
            "inband_live": _rate(il), "inband_live_ci": wilson(sum(1 for x in il if x), sum(1 for x in il if x is not None)),
            "inband_pooled": _rate([r.get("inband_pooled") for r in rows]),
            "z_live_mean": _mean([r.get("z_live") for r in rows]), "z_live_p50": _p50([r.get("z_live") for r in rows]),
            "abs_z_live_p50": _p50([abs(r["z_live"]) for r in rows if r.get("z_live") is not None]),
            "below": _rate([(r["z_live"] < -WIDTH) for r in rows if r.get("z_live") is not None]),
            "above": _rate([(r["z_live"] > WIDTH) for r in rows if r.get("z_live") is not None]),
            "lt_min_content": _rate([(r.get("nc") or 0) < MIN_CONTENT for r in rows]),
            "mc_mean": _mean([r.get("mc") for r in rows]),
            "len_p50": _p50([len(r["z"]) for r in rows if r.get("z")])}


def coached_picks(states: list[dict], tiers: tuple[str, ...], metric: str = "lp0_pb") -> list[dict]:
    picks = []
    for stt in states:
        if not stt["stuck"]:
            continue
        vb = [a for a in stt["actions"] if a["src"] in tiers and a["outcome"] == "solved" and a.get(metric) is not None]
        if not vb:
            continue
        tf = [a for a in stt["actions"] if a["src"] == "T" and a["outcome"] == "failed" and a.get(metric) is not None]
        blind = [b for b in stt["blind"] if b.get(metric) is not None]
        best = max(vb, key=lambda a: a[metric])
        best_sum = max(vb, key=lambda a: a["lp0_sum"])
        best_failed = max(tf, key=lambda a: a[metric]) if tf else None
        floor_failed = min(a[metric] for a in tf) if tf else None
        floor_all = min([a[metric] for a in tf] + [b[metric] for b in blind]) if (tf or blind) else None
        ts_inband = [a for a in stt["actions"] if a["src"] == "T" and a["outcome"] == "solved" and a.get("inband_live")]
        vb_t = [a for a in vb if a["src"] == "T"]
        best_t = max(vb_t, key=lambda a: a[metric]) if vb_t else None
        picks.append({"sid": stt["sid"], "harness": stt["harness"], "kind": stt["kind"], "depth": stt["depth"],
                      "n_T_failed": len(tf), "n_verified": len(vb),
                      "coached_src": best["src"], "coached_arm": best["arm"], "coached_cont": best["cont"],
                      "coached_lp0_pb": best[metric], "coached_lp0_sum": best["lp0_sum"], "coached_bytes": best["n_bytes"],
                      "coached_same_by_sum": (best is best_sum),
                      "best_failed_lp0_pb": best_failed[metric] if best_failed else None,
                      "delta_pb": (best[metric] - best_failed[metric]) if best_failed else None,
                      "delta_sum": (best["lp0_sum"] - best_failed["lp0_sum"]) if best_failed else None,
                      "ge_floor_failed": (best[metric] >= floor_failed) if floor_failed is not None else None,
                      "ge_floor_all": (best[metric] >= floor_all) if floor_all is not None else None,
                      "coached_inband_live": best.get("inband_live"), "coached_z_live": best.get("z_live"),
                      "coached_mc_known": best.get("mc") is not None,
                      "band_unchanged": best["src"] == "T",
                      "n_T_solved_inband": len(ts_inband),
                      "n_T_solved": sum(1 for a in stt["actions"] if a["src"] == "T" and a["outcome"] == "solved"),
                      "teacher_only_pick": bool(best_t),
                      "teacher_only_ge_floor_failed": (best_t[metric] >= floor_failed) if (best_t and floor_failed is not None) else None,
                      "teacher_only_inband_live": best_t.get("inband_live") if best_t else None,
                      "teacher_only_delta_pb": (best_t[metric] - best_failed[metric]) if (best_t and best_failed) else None,
                      "coached_y": best["y"][:160]})
    return picks


def frontier_verified_rows(states: list[dict]) -> list[dict]:
    out = []
    for stt in states:
        tf = [a for a in stt["actions"] if a["src"] == "T" and a["outcome"] == "failed" and a.get("lp0_pb") is not None]
        blind = [b for b in stt["blind"] if b.get("lp0_pb") is not None]
        for a in stt["actions"]:
            if a["src"] != "F1" or a["outcome"] != "solved":
                continue
            def rng(rs, m):
                vals = [r[m] for r in rs]
                return (min(vals), max(vals)) if vals else (None, None)
            out.append({"sid": stt["sid"], "harness": stt["harness"], "kind": stt["kind"], "stuck": stt["stuck"],
                        "depth": stt["depth"], "arm": a["arm"], "z_src": a.get("z_src"),
                        "judge": (a.get("judge") or {}).get("relation"),
                        "lp0_pb": a["lp0_pb"], "lp0_sum": a["lp0_sum"], "n_bytes": a["n_bytes"], "lpB_pb": a["lpB_pb"],
                        "failed_range_pb": rng(tf, "lp0_pb"), "failed_range_sum": rng(tf, "lp0_sum"),
                        "all_range_pb": rng(tf + blind, "lp0_pb"),
                        "failed_lpB_range": rng([t for t in tf if t.get("lpB_pb") is not None], "lpB_pb"),
                        "n_failed": len(tf), "n_blind": len(blind),
                        "mc": a.get("mc"), "nc": a.get("nc"), "z_live": a.get("z_live"), "inband_live": a.get("inband_live"),
                        "mc_vis": a.get("mc_vis"), "z_live_vis": a.get("z_live_vis"), "inband_live_vis": a.get("inband_live_vis"),
                        "mu_c": stt["mu_c"], "blind_mc": [b.get("mc") for b in stt["blind"]],
                        "y": a["y"][:200]})
    return out


def cmd_report(args: argparse.Namespace) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    states = load_states()
    E = stored_echoes()
    n_err = sum(1 for r in read_jsonl(ECHOES) if "error" in r)
    states, sigmas = annotate(states, E)
    usd = total_cost()
    lines: list[str] = []
    P = lines.append
    tiers_main = ("T", "F1")
    tiers_all = ("T", "F1", "F")

    stuck = [s for s in states if s["stuck"]]
    stuck_vb = [s for s in stuck if s["n_verified_better"]]
    stuck_vb_f = [s for s in stuck if s["n_verified_better"] or s["n_F_solved"]]
    n_reused = sum(1 for k in E if k.split("|")[1] not in ("th", "un") and k in {r["key"] for r in read_jsonl(PPI_ECHOES) if "error" not in r})

    P("N5 — COACHED REFERENCES: is the verified-better action plausible under the blind teacher, and is its thought in band?")
    P(f"states {len(states)} graded (outcome + split-states probes; prefix = advisor_ppi materialisation) | stuck (>= 1 teacher-failed first action) {len(stuck)} | "
      f"stuck with >= 1 verified-better action (tiers T∪F1) {len(stuck_vb)} (+ tier F: {len(stuck_vb_f)}) | "
      f"verified-better actions {sum(s['n_verified_better'] for s in states)} "
      f"(teacher-solved {sum(s['n_T_solved'] for s in stuck)}, frontier F1-solved {sum(s['n_F1_solved'] for s in stuck)})")
    P(f"echoes: {len(E)} usable ({n_err} errors), advisor_ppi action echoes reused where the (state, thought, action) key matched; $ spent here {usd:.2f}")
    P(f"σ_c pooled on these states (sqrt mean within-state var of the 3 blind m_c): {{{', '.join(f'{k}: {_f(v,5,3).strip()}' for k, v in sigmas['pooled'].items())}}}; "
      f"live verdict σ_Mc: {{{', '.join(f'{k}: {v:.3f}' for k, v in sigmas['live'].items())}}}")
    P("Terms: lp0 = lpC(y|x,∅) (blind teacher, no thought; per byte / summed); lpB = mean over the 3 stored blind thoughts of lpC(y|x,z_blind); teacher range = [min,max] lp0 over the teacher's OWN actions at x "
      "(F = failed first actions only; F+B = failed + the 3 blind refs; duplicates by normalised action collapsed); in-range = inside it; >= floor = at/above its min; z = (v − mean)/sd of those teacher values (needs >= 2);")
    P("       Δbest = lp0(verified) − max lp0(teacher-failed); m_c/band as in the live sd-meter (θ = 1 nat, width 2, >= 10 content tokens); in-band uses the live σ_Mc unless marked pooled.")

    # ---------------- 1. plausibility
    P("")
    P("=" * 120)
    P("1. ACTION PLAUSIBILITY — verified-better actions vs the blind teacher's own actions at the same stuck state")
    P("=" * 120)
    plaus = {}
    for metric, mname in (("lp0_pb", "lp0 per byte"), ("lp0_sum", "lp0 summed"), ("lpB_pb", "lpB per byte (under blind thoughts)")):
        P(f"-- metric {mname} --")
        P(f"  {'group':44s} {'ref':4s} {'n':>3s} {'st':>3s} {'in-range':>9s} {'inR|n>=2':>9s} {'>=floor':>8s} {'>max':>5s} {'z mean':>7s} {'z p50':>7s} {'Δbest mean':>10s} {'Δbest p50':>9s} {'=teacher':>8s}")
        for ref in ("failed", "failed+blind"):
            rows_all = plaus_rows(states, ref, tiers_main, metric)
            groups = [("ALL verified-better (T∪F1)", rows_all)]
            for hz in ("mini_swe_textbased", "bash", "terminus_2"):
                groups.append((f"  {hz}", [r for r in rows_all if r["harness"] == hz]))
            groups.append(("  tier T (teacher-solved first actions)", [r for r in rows_all if r["src"] == "T"]))
            groups.append(("  tier F1 (frontier proposals, teacher finished)", [r for r in rows_all if r["src"] == "F1"]))
            rows_f = plaus_rows(states, ref, ("F",), metric)
            groups.append(("  tier F (frontier own continuation solved)", rows_f))
            for name, rs in groups:
                a = agg_plaus(rs)
                plaus[f"{metric}|{ref}|{name.strip()}"] = a
                tag = "F" if ref == "failed" else "F+B"
                P(f"  {name:44s} {tag:4s} {a['n']:>3d} {a['n_states']:>3d} {_pct(a['in_range'],8):>9s} {_pct(a['in_range_ref2'],4) + f'({a['n_ref2']})':>9s} {_pct(a['ge_floor'],7):>8s} {_pct(a['gt_max'],4):>5s} "
                  f"{_f(a['z_mean'],7,2)} {_f(a['z_p50'],7,2)} {_f(a['delta_best_failed_mean'],10,4)} {_f(a['delta_best_failed_p50'],9,4)} {_pct(a['same_as_teacher'],7):>8s}")
    P("-- paired, split states: mean lp0(teacher-solved) − mean lp0(teacher-failed) at the same state --")
    paired = {m: paired_solved_failed(states, m) for m in ("lp0_pb", "lp0_sum", "lpB_pb")}
    for m, d in paired.items():
        P(f"  {m:8s} " + "  ".join(f"{k}: n={v['n']} mean={_f(v.get('mean'),7,4).strip()} p50={_f(v.get('p50'),7,4).strip()} se={_f(v.get('se'),6,4).strip()} solved>failed {_pct(v.get('share_pos'),3).strip()}"
                                    for k, v in d.items() if v["n"]))
    # 3 frontier-verified individually
    P("-- the frontier-verified actions individually (F1-family proposals whose teacher-finished continuation solved) --")
    fv = frontier_verified_rows(states)
    P(f"  {'state':13s} {'harness':18s} {'stuck':5s} {'arm':3s} {'rel':16s} {'lp0/B':>7s} {'F range/B':>17s} {'F+B range/B':>17s} {'lp0 sum':>8s} {'F range sum':>19s} {'lpB/B':>7s} {'F lpB range':>17s} {'m_c':>6s} {'μ_c':>6s} {'z_live':>6s} {'band':>4s} {'vis z':>6s}")
    for r in fv:
        def rr(t, p=3, w=7):
            return f"[{_f(t[0],w,p).strip()},{_f(t[1],w,p).strip()}]" if t[0] is not None else "n/a"
        P(f"  {r['sid'][:8]+':'+r['sid'].split(':')[1]:13s} {r['harness']:18s} {str(r['stuck']):5s} {r['arm']:3s} {str(r['judge'] or '')[:16]:16s} {_f(r['lp0_pb'],7,3)} {rr(r['failed_range_pb']):>17s} {rr(r['all_range_pb']):>17s} "
          f"{_f(r['lp0_sum'],8,1)} {rr(r['failed_range_sum'],1,8):>19s} {_f(r['lpB_pb'],7,3)} {rr(r['failed_lpB_range']):>17s} {_f(r['mc'],6,2)} {_f(r['mu_c'],6,2)} {_f(r['z_live'],6,2)} "
          f"{('IN' if r['inband_live'] else ('out' if r['z_live'] is not None else 'n/a')):>4s} {_f(r['z_live_vis'],6,2)}")
        P(f"      y: {r['y'][:150]!r}")

    # ---------------- 2. thought typicality
    P("")
    P("=" * 120)
    P("2. THOUGHT TYPICALITY — content-masked m_c vs the state's blind teacher thoughts (band = μ_c ± 2σ_c)")
    P("=" * 120)
    G = thought_groups(states)
    P(f"  {'group':70s} {'n':>3s} {'st':>3s} {'inband':>7s} {'95% CI':>11s} {'inbP':>6s} {'z mean':>7s} {'z p50':>7s} {'|z|p50':>7s} {'below':>6s} {'above':>6s} {'<10ct':>6s} {'m_c':>7s} {'len50':>6s}")
    thought = {}
    order = ["teacher first thought, solved", "teacher first thought, failed",
             "teacher first thought, solved @stuck (split states)",
             "frontier proposal thought (GLM latent+visible), solved", "frontier proposal thought (GLM latent+visible), failed",
             "frontier proposal thought (GLM latent+visible), solved @stuck", "frontier proposal thought (GLM latent+visible), failed @stuck",
             "frontier proposal visible text only, solved", "frontier proposal visible text only, failed",
             "frontier own-continuation first thought, solved", "frontier own-continuation first thought, failed",
             "blind ref LOO (band's own hit rate)"]
    for name in order:
        rs = G.get(name, [])
        a = agg_thought(rs)
        thought[name] = a
        ci = f"[{100*a['inband_live_ci'][0]:3.0f}–{100*a['inband_live_ci'][1]:3.0f}]" if a["n"] else "n/a"
        P(f"  {name:70s} {a['n']:>3d} {a['n_states']:>3d} {_pct(a['inband_live'],6):>7s} {ci:>11s} {_pct(a['inband_pooled'],5):>6s} {_f(a['z_live_mean'],7,2)} {_f(a['z_live_p50'],7,2)} {_f(a['abs_z_live_p50'],7,2)} "
          f"{_pct(a['below'],5):>6s} {_pct(a['above'],5):>6s} {_pct(a['lt_min_content'],5):>6s} {_f(a['mc_mean'],7,3)} {_f(a['len_p50'],6,0)}")
    P("  by harness (in-band live share, n):")
    for name in ("teacher first thought, solved", "teacher first thought, failed", "frontier proposal thought (GLM latent+visible), solved",
                 "frontier proposal thought (GLM latent+visible), failed", "frontier proposal visible text only, solved", "frontier proposal visible text only, failed"):
        cells = []
        for hz in ("mini_swe_textbased", "bash", "terminus_2"):
            rs = [r for r in G.get(name, []) if r["harness"] == hz]
            a = agg_thought(rs)
            thought[f"{name} | {hz}"] = a
            cells.append(f"{hz[:10]} {_pct(a['inband_live'],4).strip()} ({a['n']})")
        P(f"    {name:70s} " + "  ".join(cells))

    # ---------------- 3. coached pick
    P("")
    P("=" * 120)
    P("3. THE COACHED PICK — per stuck state with >= 1 verified-better action: the verified action the blind teacher finds most plausible (max lp0 per byte)")
    P("=" * 120)
    picks = coached_picks(states, tiers_main)
    picks_f = coached_picks(states, tiers_all)
    def pick_summary(ps: list[dict]) -> dict:
        n = len(ps)
        return {"n_states": n,
                "coached_is_teacher": sum(1 for p in ps if p["coached_src"] == "T"),
                "coached_is_frontier": sum(1 for p in ps if p["coached_src"] != "T"),
                "ge_floor_failed": sum(1 for p in ps if p["ge_floor_failed"]),
                "ge_floor_all": sum(1 for p in ps if p["ge_floor_all"]),
                "below_floor_failed": sum(1 for p in ps if p["ge_floor_failed"] is False),
                "below_floor_failed_share": (sum(1 for p in ps if p["ge_floor_failed"] is False) / n) if n else None,
                "below_floor_all_share": (sum(1 for p in ps if p["ge_floor_all"] is False) / n) if n else None,
                "delta_pb_mean": _mean([p["delta_pb"] for p in ps]), "delta_pb_p50": _p50([p["delta_pb"] for p in ps]),
                "delta_pos": sum(1 for p in ps if p["delta_pb"] is not None and p["delta_pb"] > 0),
                "delta_sum_mean": _mean([p["delta_sum"] for p in ps]),
                "coached_inband_live": sum(1 for p in ps if p["coached_inband_live"]),
                "coached_mc_known": sum(1 for p in ps if p["coached_mc_known"]),
                "band_unchanged": sum(1 for p in ps if p["band_unchanged"]),
                "band_unchanged_and_inband": sum(1 for p in ps if p["band_unchanged"] and p["coached_inband_live"]),
                "same_pick_by_sum": sum(1 for p in ps if p["coached_same_by_sum"]),
                "teacher_only_available": sum(1 for p in ps if p["teacher_only_pick"]),
                "teacher_only_ge_floor": sum(1 for p in ps if p["teacher_only_ge_floor_failed"]),
                "teacher_only_inband": sum(1 for p in ps if p["teacher_only_inband_live"]),
                "teacher_only_ge_floor_and_inband": sum(1 for p in ps if p["teacher_only_ge_floor_failed"] and p["teacher_only_inband_live"]),
                "by_harness": {hz: {"n": sum(1 for p in ps if p["harness"] == hz),
                                    "teacher": sum(1 for p in ps if p["harness"] == hz and p["coached_src"] == "T"),
                                    "ge_floor_failed": sum(1 for p in ps if p["harness"] == hz and p["ge_floor_failed"]),
                                    "inband": sum(1 for p in ps if p["harness"] == hz and p["coached_inband_live"])}
                               for hz in ("mini_swe_textbased", "bash", "terminus_2")}}
    S = pick_summary(picks)
    SF = pick_summary(picks_f)
    P(f"  stuck states with a coached pick: {S['n_states']} (tiers T∪F1); with tier F added: {SF['n_states']}")
    P(f"  coached pick is a TEACHER action (band unchanged: reference thoughts stay the teacher's own): {S['coached_is_teacher']}/{S['n_states']}; a FRONTIER action (no in-band thought by construction → action-only entry into the A/R legs): {S['coached_is_frontier']}/{S['n_states']}")
    P(f"  coached pick >= teacher floor (min lp0/B over failed first actions): {S['ge_floor_failed']}/{S['n_states']}  |  >= floor over failed + blind refs: {S['ge_floor_all']}/{S['n_states']}  |  BELOW the failed floor: {S['below_floor_failed']}/{S['n_states']} = {_pct(S['below_floor_failed_share'],3).strip()} (kill line: > 70%)")
    P(f"  Δbest (lp0/B coached − best teacher-failed): mean {_f(S['delta_pb_mean'],7,4).strip()} p50 {_f(S['delta_pb_p50'],7,4).strip()}, positive at {S['delta_pos']}/{S['n_states']}; summed Δ mean {_f(S['delta_sum_mean'],7,2).strip()} nats; per-byte and summed argmax agree at {S['same_pick_by_sum']}/{S['n_states']}")
    P(f"  coached pick's thought in band (live σ): {S['coached_inband_live']}/{S['coached_mc_known']} with a thought echo; teacher-pick AND in band: {S['band_unchanged_and_inband']}/{S['n_states']}")
    P(f"  teacher-ONLY coached set (restrict the pick to teacher-solved actions, so every reference thought is the teacher's own): available at {S['teacher_only_available']}/{S['n_states']}; its best action >= failed floor {S['teacher_only_ge_floor']}/{S['teacher_only_available']}, thought in band {S['teacher_only_inband']}/{S['teacher_only_available']}, both {S['teacher_only_ge_floor_and_inband']}/{S['teacher_only_available']}")
    P("  by harness: " + "; ".join(f"{hz} n={v['n']} teacher-pick {v['teacher']} >=floor {v['ge_floor_failed']} in-band {v['inband']}" for hz, v in S["by_harness"].items()))
    P(f"  {'state':13s} {'harness':18s} {'d':>2s} {'nTf':>3s} {'nV':>2s} {'pick':5s} {'lp0/B':>7s} {'bestF/B':>7s} {'Δ/B':>7s} {'Δsum':>7s} {'>=flr':>5s} {'>=flrB':>6s} {'z_c':>6s} {'band':>4s} {'nTs':>3s} {'nTs_in':>6s} y")
    for p in picks_f:
        band = "IN" if p["coached_inband_live"] else ("out" if p["coached_z_live"] is not None else "n/a")
        P(f"  {p['sid'][:8]+':'+p['sid'].split(':')[1]:13s} {p['harness']:18s} {p['depth']:>2d} {p['n_T_failed']:>3d} {p['n_verified']:>2d} {p['coached_arm']:5s} {_f(p['coached_lp0_pb'],7,3)} {_f(p['best_failed_lp0_pb'],7,3)} "
          f"{_f(p['delta_pb'],7,3)} {_f(p['delta_sum'],7,1)} {str(p['ge_floor_failed']):>5s} {str(p['ge_floor_all']):>6s} {_f(p['coached_z_live'],6,2)} {band:>4s} {p['n_T_solved']:>3d} {p['n_T_solved_inband']:>6d} {p['coached_y'][:60]!r}")

    # ---------------- verdict
    P("")
    P("=" * 120)
    P("VERDICT")
    P("=" * 120)
    main_pb = plaus["lp0_pb|failed|ALL verified-better (T∪F1)"]
    main_pb_b = plaus["lp0_pb|failed+blind|ALL verified-better (T∪F1)"]
    f1_pb = plaus["lp0_pb|failed|tier F1 (frontier proposals, teacher finished)"]
    t_pb = plaus["lp0_pb|failed|tier T (teacher-solved first actions)"]
    tsol = thought["teacher first thought, solved"]
    tfail = thought["teacher first thought, failed"]
    fpro = thought["frontier proposal thought (GLM latent+visible), solved"]
    fpro_all = agg_thought(G.get("frontier proposal thought (GLM latent+visible), solved", []) + G.get("frontier proposal thought (GLM latent+visible), failed", []))
    killed = S["below_floor_failed_share"] is not None and S["below_floor_failed_share"] > KILL_SHARE
    verdict = [
        f"1. Plausibility: verified-better actions sit at/above the blind teacher's own lp0/B floor at {_pct(main_pb['ge_floor'],3).strip()} vs the failed first actions and {_pct(main_pb_b['ge_floor'],3).strip()} vs failed+blind refs (n={main_pb['n']} actions / {main_pb['n_states']} stuck states); "
        f"strictly inside the range at {_pct(main_pb['in_range'],3).strip()} vs failed ({_pct(main_pb['in_range_ref2'],3).strip()} where the teacher has >= 2 distinct failed actions, n={main_pb['n_ref2']}) and {_pct(main_pb_b['in_range'],3).strip()} vs failed+blind — {_pct(main_pb['gt_max'],3).strip()} sit ABOVE every failed action; z vs failed+blind mean {_f(main_pb_b['z_mean'],5,2).strip()} (teacher-solved tier {_pct(t_pb['ge_floor'],3).strip()} >= failed floor, n={t_pb['n']}; frontier F1 tier {_pct(f1_pb['ge_floor'],3).strip()}, n={f1_pb['n']}).",
        f"2. Thoughts: teacher-solved first thoughts in band {_pct(tsol['inband_live'],3).strip()} (n={tsol['n']}) vs teacher-failed {_pct(tfail['inband_live'],3).strip()} (n={tfail['n']}); blind LOO {_pct(thought['blind ref LOO (band\'s own hit rate)']['inband_live'],3).strip()} — no solved/failed gap, X2's 0.84/0.85 confirmed; "
        f"frontier proposal thoughts (GLM latent+visible) in band {_pct(fpro_all['inband_live'],3).strip()} (n={fpro_all['n']}; solved ones {_pct(fpro['inband_live'],3).strip()}, n={fpro['n']}) — {'out of band as expected' if (fpro_all['inband_live'] or 0) < 0.5 else 'NOT the expected out-of-band'}.",
        f"3. Coached pick: {S['n_states']} stuck states have one; it is a teacher action at {S['coached_is_teacher']}/{S['n_states']} (band unchanged) and a frontier action at {S['coached_is_frontier']}/{S['n_states']} (action-only); a teacher-only coached set exists at {S['teacher_only_available']}/{S['n_states']} and is >= floor AND in band at {S['teacher_only_ge_floor_and_inband']}/{S['teacher_only_available']}; "
        f"Δ vs the best teacher-failed action is positive at {S['delta_pos']}/{S['n_states']} (p50 {_f(S['delta_pb_p50'],6,4).strip()}/B); the pick lies BELOW the teacher's failed-action floor at {S['below_floor_failed']}/{S['n_states']} = {_pct(S['below_floor_failed_share'],3).strip()}.",
        f"4. Kill line (dead if the best verified action is below the teacher floor at > 70% of stuck states): {'DEAD' if killed else 'NOT DEAD'} — {_pct(S['below_floor_failed_share'],3).strip()} below; "
        f"a coached reference can be built at these {S['n_states']} states without leaving the teacher's band {'in most cases' if not killed else 'only rarely'}; "
        f"caveats: n is small ({S['n_states']} stuck states with a verified action, {len(stuck)} stuck in total → {len(stuck) - S['n_states']} stuck states have NO verified-better action at all), the 'better' label is one graded continuation per action, and frontier-derived picks bring no in-band thought.",
    ]
    lines += verdict
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    rep = {"n_states": len(states), "n_stuck": len(stuck), "n_stuck_with_verified_better": len(stuck_vb),
           "n_stuck_with_verified_better_incl_F": len(stuck_vb_f),
           "n_verified_better_actions": sum(s["n_verified_better"] for s in states),
           "n_echoes_usable": len(E), "n_echo_errors": n_err, "usd": usd, "sigmas": sigmas,
           "plausibility": plaus, "paired_solved_minus_failed": paired, "frontier_verified": fv,
           "thought_typicality": thought, "coached_picks": picks_f, "coached_summary": {"T_F1": S, "T_F1_F": SF},
           "kill_line": {"threshold_below_floor_share": KILL_SHARE, "below_floor_share": S["below_floor_failed_share"], "dead": killed},
           "verdict": verdict,
           "terms": {"lp0": "lpC(y|x,∅): blind teacher logprob of the action bytes (per byte / summed)",
                     "lpB": "mean_j lpC(y|x,z_blind_j) over the 3 stored blind teacher thoughts",
                     "teacher range": "[min,max] lp0 over the teacher's own actions at x: failed first actions (F) or failed + blind refs (F+B)",
                     "in-range / >= floor": "verified action inside the range / at or above its min",
                     "z": "(lp0(v) − mean)/sd over the teacher's own actions (>= 2 needed)",
                     "m_c": "content-masked mean token logprob of a thought under x (tokens with |lp(x) − lp(∅)| > 1 nat)",
                     "band": "μ_c ± 2σ_c, μ_c = mean m_c of the blind refs at x, σ_c = live verdict σ_Mc(dialect) (or pooled here)",
                     "coached pick": "verified-better action with the highest lp0 per byte at a stuck state",
                     "Δbest": "lp0(coached) − max lp0(teacher-failed) at the same state",
                     "stuck state": ">= 1 teacher-failed first action at x",
                     "verified-better": "teacher-solved first action (tier T) or F1-family frontier proposal whose teacher-finished continuation solved (tier F1); arm F = frontier's own continuation (tier F, secondary)"}}
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=str))
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("build")
    e = sub.add_parser("echo")
    e.add_argument("--limit", type=int, default=None)
    sub.add_parser("report")
    args = ap.parse_args()
    if args.cmd == "build":
        build()
    elif args.cmd == "echo":
        cmd_echo(args)
    else:
        cmd_report(args)


if __name__ == "__main__":
    main()
