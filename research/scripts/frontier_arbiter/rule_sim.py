"""Frontier-arbiter rule simulation on stored duel data (2026-09-20).

Data: research/results/frontier_rule_probe/ — 4 stored wvk-11 duels
(chal-00286..289), per turn the teacher refs + king/challenger pairs, the
frontier's (engy glm-5.2) 3 sampled + 1 greedy rollouts, and the teacher
echoes of every cross combination (see frontier_rule_probe.py::cmd_echo).

    python research/scripts/frontier_arbiter/rule_sim.py echo  --n 200   # new attack echoes (Engy teacher, canonical)
    python research/scripts/frontier_arbiter/rule_sim.py run             # all tests -> results/frontier_arbiter/rule/

Tests (per dialect and pooled), rules from arbiter.py:
  1 HEADROOM  leave-one-out ordering of frontier_own / teacher_own / king / challenger
  2 NOISE     challenger − king margin, SE, z, bootstrap SE ratio vs live, rho, disagree share of variance
  3 ATTACKS   filler / generic / parrot (stored), frontier-parrot(greedy) / copy-frontier / generic-command (new)
  4 RANGE     gate-threshold sweeps
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import statistics as st
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from frontier_arbiter import arbiter as ab  # noqa: E402
from frontier_arbiter.common import Engy, TeacherEcho, append_jsonl, read_jsonl  # noqa: E402
from affine.score import leakage  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "research/results/frontier_rule_probe"
OUT = REPO / "research/results/frontier_arbiter/rule"
ECHOES = OUT / "echoes.jsonl"
KINDS = ("bash", "boxed", "tool_call")
FAM_HEADROOM = ("frontier_own", "teacher_own", "king", "challenger")
LS_CMD = "```bash\nls -la\n```"
PATH_RE = re.compile(r"(?<![\w/])((?:/|\./)?(?:[\w.-]+/)*[\w-]+\.(?:py|js|ts|tsx|go|java|rs|rb|md|txt|json|toml|yaml|yml|cfg|ini|sh|c|h|cpp|cc|html|css|xml|csv))(?![\w/])")


# ------------------------------------------------------------------ data model
@dataclass
class Fam:
    """One thought/action family on one turn, at one ref subset."""
    a: list[float]                  # per-teacher-ref Reason a_i
    f: list[float]                  # per-frontier-ref Reason f_j (sampled F)
    m: float                        # lpC(z|x)
    y: str                          # the family's action
    z: str = ""
    b: list[float] | None = None    # frontier-thought action lifts b_j
    B: float | None = None          # lpC(y|z) − lpC(y|∅)
    leak: bool = False


@dataclass
class View:
    """One turn at one ref subset (full = k refs / 3 F samples; loo = drop ref 0 and F sample 0)."""
    tid: str
    record: str
    kind: str
    yC: list[str]
    t: list[float]
    lpC_e: list[float]
    yF: list[str]
    lpF_e: list[float]
    yG: str
    lpG_e: float
    fams: dict[str, Fam] = field(default_factory=dict)

    def fset(self, which: str) -> tuple[list[str], list[float]]:
        if which == "greedy":
            return [self.yG], [self.lpG_e]
        return self.yF, self.lpF_e


@dataclass
class Turn:
    tid: str
    record: str
    kind: str
    full: View
    loo: View
    prefix: list[dict]
    zC: list[str]                   # teacher reference thoughts
    zF: list[str]                   # frontier sampled thoughts
    echo: View | None = None        # Engy-echoed subset (honest king re-echoed + new attacks)


def _reason(p: dict) -> float:
    return p["lpC_yc_za"] - p["lpC_yc_e"]


def first_file_cmd(prefix: list[dict]) -> str:
    for msg in prefix:
        if msg.get("role") != "user":
            continue
        c = msg.get("content") if isinstance(msg.get("content"), str) else json.dumps(msg.get("content"))
        m = PATH_RE.search(c or "")
        if m:
            return f"```bash\ncat {m.group(1)}\n```"
    return "```bash\ncat README.md\n```"


def load_turns(probe: Path = PROBE, echoes: Path = ECHOES) -> tuple[list[Turn], dict]:
    meta = json.loads((probe / "meta.json").read_text())
    new_echo = {e["turn_id"]: e for e in read_jsonl(echoes) if "lp" in e}
    turns: list[Turn] = []
    dropped = {"lt3_frontier": 0, "no_greedy": 0, "no_echo": 0}
    for rec in sorted(meta):
        tmap = {t["turn_id"]: t for t in read_jsonl(probe / "turns" / f"{rec}.jsonl")}
        fmap = {f["turn_id"]: f for f in read_jsonl(probe / "frontier" / f"{rec}.jsonl")}
        emap: dict[str, dict] = {}
        for e in read_jsonl(probe / "echoes" / f"{rec}.jsonl"):
            prev = emap.get(e["turn_id"])
            if prev is None or ("lp" in e and "lp" not in prev):
                emap[e["turn_id"]] = e
        for tid, t in tmap.items():
            e = emap.get(tid)
            if not e or "lp" not in e:
                dropped["no_echo"] += 1
                continue
            if e["n_frontier"] < 3:
                dropped["lt3_frontier"] += 1
                continue
            if not e["has_greedy"]:
                dropped["no_greedy"] += 1
                continue
            turns.append(build_turn(t, fmap[tid], e, new_echo.get(tid)))
    return turns, {"meta": meta, "dropped": dropped}


def build_turn(t: dict, f: dict, e: dict, ne: dict | None) -> Turn:
    lp = e["lp"]
    kind = t["kind"]
    refs = t["refs"]
    k = len(refs)
    F = [s for s in f["samples"] if s["parsed"]][:3]
    yF, zF = [s["y"] for s in F], [s["z"] for s in F]
    yG = f["greedy"]["y"]
    yC = [r["y"] for r in refs]
    tC = [r["lp_thought"] for r in refs]
    eC = [r["lp_empty"] for r in refs]
    eF = [lp[f"yF_e.{j}"] for j in range(3)]
    eG = lp["yG_e"]
    kp, cp = t["king"], t["challenger"]
    for pairs in (kp, cp):
        assert all(abs(p["lpC_yc_e"] - refs[i]["lp_empty"]) < 1e-9 for i, p in enumerate(pairs)), "pair/ref order"

    def side(pairs: list[dict], tag: str) -> Fam:
        p0 = pairs[0]
        return Fam(a=[_reason(p) for p in pairs],
                   f=[lp[f"{tag}.f.{j}"] - eF[j] for j in range(3)],
                   m=p0["lpC_za_x"], y=p0["y_a"], z=p0["z_a"],
                   b=[lp[f"AF.{tag}.{j}"] - p0["lpC_ya_e"] for j in range(3)],
                   B=p0["lpC_ya_za"] - p0["lpC_ya_e"], leak=leakage(p0["z_a"], p0["y_a"]))

    king, chal = side(kp, "king"), side(cp, "chal")
    fams_full: dict[str, Fam] = {"king": king, "challenger": chal}
    for vn in ("filler", "generic", "parrot"):
        fams_full[vn] = Fam(a=[lp[f"var.{vn}.a.{i}"] - eC[i] for i in range(k)],
                            f=[lp[f"var.{vn}.f.{j}"] - eF[j] for j in range(3)],
                            m=lp[f"var.{vn}.m"], y=king.y, b=king.b)
    # action-only attacks on the king (thought unchanged): action-space closeness + stored B where available
    fams_full["copyF"] = Fam(a=king.a, f=king.f, m=king.m, y=yG, z=king.z,
                             B=lp["king.fG"] - eG, leak=leakage(king.z, yG))
    if kind == "bash":
        fams_full["lscmd"] = Fam(a=king.a, f=king.f, m=king.m, y=LS_CMD, z=king.z)
        fams_full["catcmd"] = Fam(a=king.a, f=king.f, m=king.m, y=first_file_cmd(t["prefix"]), z=king.z)
    full = View(t["turn_id"], t["record"], kind, yC, tC, eC, yF, eF, yG, eG, fams_full)

    def loo_of(fm: Fam) -> Fam:
        return Fam(a=fm.a[1:], f=fm.f[1:], m=fm.m, y=fm.y, z=fm.z,
                   b=fm.b[1:] if fm.b else None, B=fm.B, leak=fm.leak)

    fams_loo = {"king": loo_of(king), "challenger": loo_of(chal),
                "teacher_own": Fam(a=[lp[f"town.a.{i}"] - eC[i] for i in range(1, k)],
                                   f=[lp[f"tref.0.f.{j}"] - eF[j] for j in range(1, 3)],
                                   m=tC[0], y=yC[0], z=refs[0]["z"]),
                "frontier_own": Fam(a=[lp[f"fown.a.{i}"] - eC[i] for i in range(1, k)],
                                    f=[lp[f"fown.f.{j}"] - eF[j] for j in range(1, 3)],
                                    m=lp["mF.0"], y=yF[0], z=zF[0])}
    loo = View(t["turn_id"], t["record"], kind, yC[1:], tC[1:], eC[1:], yF[1:], eF[1:], yG, eG, fams_loo)
    echo = build_echo_view(t, full, ne) if ne else None
    return Turn(t["turn_id"], t["record"], kind, full, loo, t["prefix"], [r["z"] for r in refs], zF, echo)


def build_echo_view(t: dict, full: View, ne: dict) -> View:
    """Same-source (Engy) view: honest king re-echoed next to the new attacks."""
    lp = ne["lp"]
    k = len(t["refs"])
    king = full.fams["king"]
    eC = [lp[f"e.C.{i}"] for i in range(k)]
    eF = [lp[f"e.F.{j}"] for j in range(3)]
    tC = [lp[f"t.{i}"] for i in range(k)]
    fams = {"king": Fam(a=[lp[f"k.a.{i}"] - eC[i] for i in range(k)],
                        f=[lp[f"k.f.{j}"] - eF[j] for j in range(3)],
                        m=lp["k.m"], y=king.y, z=king.z,
                        b=[lp[f"k.af.{j}"] - lp["k.e"] for j in range(3)],
                        B=lp["k.B"] - lp["k.e"], leak=king.leak)}
    fams["fparrotG"] = Fam(a=[lp[f"fp.a.{i}"] - eC[i] for i in range(k)],
                           f=[lp[f"fp.f.{j}"] - eF[j] for j in range(3)],
                           m=lp["fp.m"], y=king.y, z=ne["z_fp"], b=fams["king"].b,
                           B=lp["fp.B"] - lp["k.e"], leak=leakage(ne["z_fp"], king.y))
    fams["copyF"] = Fam(a=fams["king"].a, f=fams["king"].f, m=fams["king"].m, y=full.yG, z=king.z,
                        b=[lp[f"cf.af.{j}"] - lp["cf.e"] for j in range(3)],
                        B=lp["cf.B"] - lp["cf.e"], leak=leakage(king.z, full.yG))
    for tag, name in (("ls", "lscmd"), ("cat", "catcmd")):
        if f"{tag}.e" in lp:
            fams[name] = Fam(a=fams["king"].a, f=fams["king"].f, m=fams["king"].m,
                             y=ne[f"y_{tag}"], z=king.z,
                             b=[lp[f"{tag}.af.{j}"] - lp[f"{tag}.e"] for j in range(3)],
                             B=lp[f"{tag}.B"] - lp[f"{tag}.e"], leak=leakage(king.z, ne[f"y_{tag}"]))
    return View(full.tid, full.record, full.kind, full.yC, tC, eC, full.yF, eF, full.yG, lp["cf.e"], fams)


# ------------------------------------------------------------------ new attack echoes (Engy teacher, canonical rendering)
def pick_echo_turns(turns: list[Turn], n: int, seed: int = 11) -> list[Turn]:
    rng = random.Random(seed)
    by: dict[str, list[Turn]] = {}
    for t in turns:
        by.setdefault(t.kind, []).append(t)
    total = len(turns)
    picked: list[Turn] = []
    for kind, pool in sorted(by.items()):
        take = max(12, round(n * len(pool) / total))
        pool = sorted(pool, key=lambda t: t.tid)
        rng.shuffle(pool)
        picked += pool[:take]
    return picked


async def cmd_echo(args) -> int:
    turns, _ = load_turns()
    done = {e["turn_id"] for e in read_jsonl(ECHOES) if "lp" in e}
    todo = [t for t in pick_echo_turns(turns, args.n) if t.tid not in done]
    print(f"{len(todo)} turns to echo ({len(done)} done)")
    engy = Engy(concurrency=args.concurrency)
    te = TeacherEcho(engy)
    lock = asyncio.Lock()
    t0 = time.time()
    n_done = 0
    sem = asyncio.Semaphore(max(2, args.concurrency // 8))

    async def lp_y(x, z, y):
        return (await te.lp_action(x, z, y, rendering="canonical"))["lp_per_byte"]

    async def lp_z(x, z):
        return (await te.lp_thought(x, z, rendering="canonical"))["lp_per_byte"]

    async def one(t: Turn) -> None:
        nonlocal n_done
        v = t.full
        x = t.prefix
        king = v.fams["king"]
        zk, yk = king.z, king.y
        fr = t.zF
        z_fp = zk + "\n\n" + v.yG
        jobs: dict[str, asyncio.Future] = {}

        def add(name, coro):
            jobs[name] = asyncio.ensure_future(coro)

        async with sem:
            try:
                add("k.m", lp_z(x, zk))
                add("k.B", lp_y(x, zk, yk))
                add("k.e", lp_y(x, "", yk))
                for i, yc in enumerate(v.yC):
                    add(f"k.a.{i}", lp_y(x, zk, yc))
                    add(f"e.C.{i}", lp_y(x, "", yc))
                    add(f"fp.a.{i}", lp_y(x, z_fp, yc))
                for i, zc in enumerate(t.zC):
                    add(f"t.{i}", lp_z(x, zc))
                for j, yf in enumerate(v.yF):
                    add(f"k.f.{j}", lp_y(x, zk, yf))
                    add(f"e.F.{j}", lp_y(x, "", yf))
                    add(f"fp.f.{j}", lp_y(x, z_fp, yf))
                    add(f"k.af.{j}", lp_y(x, fr[j], yk))
                    add(f"cf.af.{j}", lp_y(x, fr[j], v.yG))
                add("fp.m", lp_z(x, z_fp))
                add("fp.B", lp_y(x, z_fp, yk))
                add("cf.e", lp_y(x, "", v.yG))
                add("cf.B", lp_y(x, zk, v.yG))
                extra = {}
                if t.kind == "bash":
                    extra = {"y_ls": LS_CMD, "y_cat": v.fams["catcmd"].y}
                    for tag, y in (("ls", LS_CMD), ("cat", extra["y_cat"])):
                        add(f"{tag}.e", lp_y(x, "", y))
                        add(f"{tag}.B", lp_y(x, zk, y))
                        for j in range(3):
                            add(f"{tag}.af.{j}", lp_y(x, fr[j], y))
                vals = dict(zip(jobs.keys(), await asyncio.gather(*jobs.values())))
                rec = {"turn_id": t.tid, "record": t.record, "kind": t.kind, "lp": vals,
                       "z_fp": z_fp, **extra}
            except Exception as ex:  # noqa: BLE001 — one bad turn must not kill the run
                for j in jobs.values():
                    j.cancel()
                rec = {"turn_id": t.tid, "record": t.record, "kind": t.kind, "error": repr(ex)[:300]}
        async with lock:
            append_jsonl(ECHOES, rec)
            n_done += 1
            if n_done % 10 == 0:
                print(f"  {n_done}/{len(todo)} turns, {te.n_calls} echoes, ${engy.cost_usd:.3f} ({time.time() - t0:.0f}s)", flush=True)

    await asyncio.gather(*[one(t) for t in todo])
    cost_path = OUT / "echoes_cost.json"
    prev = json.loads(cost_path.read_text()) if cost_path.exists() else {"cost_usd": 0.0, "echoes": 0, "runs": []}
    prev["cost_usd"] += engy.cost_usd
    prev["echoes"] += te.n_calls
    prev["runs"].append({"turns": n_done, "echoes": te.n_calls, "cost_usd": engy.cost_usd,
                         "usage": engy.usage, "seconds": time.time() - t0})
    cost_path.write_text(json.dumps(prev, indent=1))
    print(f"done: {n_done} turns, {te.n_calls} echoes, ${engy.cost_usd:.3f}")
    return 0


# ------------------------------------------------------------------ rules
@dataclass(frozen=True)
class Rule:
    name: str
    gate: tuple[str, float, str] | None      # (method, theta|nats, fset) ; None = no arbiter (baseline)
    alt: str                                  # act:<sim>:<scale> | RF | V | AF | minRFG | minVG | live | wB:<mode>:<param>
    units: str = "raw"                        # raw | sd
    band_c: float = ab.BAND_C_LIVE


@dataclass
class Ctx:
    sd_live: dict[str, float]                 # per dialect: sd of live (band 4) king+challenger turn scores
    sigma_t: dict[str, float]                 # per dialect: pooled within-turn sd of t_i
    std: ab.Standardizer                      # teacher_own LOO anchors per (dialect, leg)
    legs_cache: dict = field(default_factory=dict)


_GATE_CACHE: dict[tuple, bool] = {}


@lru_cache(maxsize=None)
def _act(y: str, yF: tuple[str, ...], yC: tuple[str, ...], kind: str, sim: str) -> float:
    return ab.action_closeness(y, list(yF), list(yC), kind, sim)


def legs(v: View, fam_name: str, rule: Rule, ctx: Ctx) -> dict[str, float]:
    """Every leg a rule may need on one (view, family); cached per band_c."""
    key = (id(v), fam_name, rule.band_c)
    hit = ctx.legs_cache.get(key)
    if hit is not None:
        return hit
    fam = v.fams[fam_name]
    R = ab.reason_leg(fam.a)
    G = ab.grounding_leg(fam.m, v.t, rule.band_c)
    out = {"R": R, "G": G, "live": min(R, G), "RF": ab.frontier_reason(fam.f),
           "V": ab.frontier_contrast(fam.f, fam.a),
           "AF": ab.frontier_action_leg(fam.b) if fam.b else float("nan"),
           "typ": ab.typicality(fam.m, v.t, ctx.sigma_t.get(v.kind, float("nan")))}
    ctx.legs_cache[key] = out
    return out


def gate_of(v: View, rule: Rule) -> bool:
    if rule.gate is None:
        return False
    key = (id(v), rule.gate)
    hit = _GATE_CACHE.get(key)
    if hit is not None:
        return hit
    method, theta, fset = rule.gate
    yF, lpF_e = v.fset(fset)
    out = ab.gate_disagree(yF, v.yC, v.kind, method, theta, lpF_e, v.lpC_e,
                           surprise_nats=theta if method == "surprise" else ab.SURPRISE_NATS)
    _GATE_CACHE[key] = out
    return out


def score(v: View, fam_name: str, rule: Rule, ctx: Ctx) -> float:
    fam = v.fams[fam_name]
    L = legs(v, fam_name, rule, ctx)
    kind = v.kind
    if rule.alt.startswith("wB:"):
        _, mode, param, fset = rule.alt.split(":")
        yF, _ = v.fset(fset)
        sims = [ab.mean_sim(yc, yF, kind, "jaccard") for yc in v.yC]
        w = ab.ref_weights(sims, mode, temperature=float(param), theta=float(param))
        return ab.weighted_min_rg(fam.a, fam.m, v.t, w, band_c=rule.band_c)
    if rule.units == "sd":
        live = min(ctx.std.z(kind, "R", L["R"]), L["typ"])
    else:
        live = L["live"]
    if rule.gate is None or not gate_of(v, rule):
        return live
    alt = alt_value(v, fam, rule, ctx, L)
    return ab.arbiter_turn(True, live, alt)


def needs_b(rule: Rule) -> bool:
    return rule.alt in ("AF", "minAFG")


def alt_value(v: View, fam: Fam, rule: Rule, ctx: Ctx, L: dict[str, float]) -> float:
    kind = v.kind
    a = rule.alt
    sd = rule.units == "sd"
    g = L["typ"] if sd else L["G"]

    def act(spec: str) -> float:
        _, sim, scale = spec.split(":")
        yF, _ = v.fset(rule.gate[2])
        s = _act(fam.y, tuple(yF), tuple(v.yC), kind, sim)
        return ctx.std.z(kind, f"act:{sim}:{rule.gate[2]}", s) if sd else s * float(scale) * ctx.sd_live[kind]

    def leg(name: str) -> float:
        return ctx.std.z(kind, name, L[name]) if sd else L[name]

    if a.startswith("act:"):
        return act(a)
    if a == "minAG":
        return min(act("act:jaccard:1"), g)
    if a in ("RF", "V", "AF"):
        return leg(a)
    if a == "minRFG":
        return min(leg("RF"), g)
    if a == "minVG":
        return min(leg("V"), g)
    if a == "minAFG":
        return min(leg("AF"), g)
    raise ValueError(a)


def weights_of(v: View, rule: Rule) -> list[float]:
    _, mode, param, fset = rule.alt.split(":")
    yF, _ = v.fset(fset)
    sims = [ab.mean_sim(yc, yF, v.kind, "jaccard") for yc in v.yC]
    return ab.ref_weights(sims, mode, temperature=float(param), theta=float(param))


GATES_MAIN = [("exact", 0.0, "greedy"), ("exact", 0.0, "sampled"),
              ("jaccard", 0.5, "greedy"), ("jaccard", 0.5, "sampled"),
              ("jaccard", 0.3, "sampled"), ("surprise", 0.02, "greedy"), ("surprise", 0.02, "sampled")]
ALTS_MAIN = ["act:jaccard:0.5", "act:jaccard:1", "act:jaccard:2", "act:exact:1", "minAG",
             "RF", "V", "AF", "minRFG", "minVG", "minAFG"]
GATES_SHORT = [("exact", 0.0, "sampled"), ("jaccard", 0.5, "greedy"), ("surprise", 0.02, "sampled")]
ALTS_SD = ["act:jaccard:1", "minAG", "V", "minRFG", "minVG"]


def gate_name(g: tuple[str, float, str]) -> str:
    m, th, fs = g
    core = m if m == "exact" else f"{'jac' if m == 'jaccard' else 'sur'}{th:g}"
    return f"{core}/{fs[0].upper()}"


def build_rules() -> list[Rule]:
    rules = [Rule("live_c2", None, "live", "raw", 2.0), Rule("live_c4", None, "live", "raw", 4.0),
             Rule("live_sd", None, "live", "sd", 4.0)]
    for g in GATES_MAIN:
        for alt in ALTS_MAIN:
            rules.append(Rule(f"{gate_name(g)}|{alt}", g, alt))
    for g in GATES_SHORT:
        for alt in ALTS_SD:
            rules.append(Rule(f"{gate_name(g)}|{alt}|sd", g, alt, "sd"))
    for fset in ("greedy", "sampled"):
        rules.append(Rule(f"wB_soft0.1/{fset[0].upper()}", None, f"wB:softmax:0.1:{fset}"))
        rules.append(Rule(f"wB_soft0.3/{fset[0].upper()}", None, f"wB:softmax:0.3:{fset}"))
        rules.append(Rule(f"wB_hard0.5/{fset[0].upper()}", None, f"wB:hard:0.5:{fset}"))
    return rules


# ------------------------------------------------------------------ stats helpers
def rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    x, y = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def paired(d: list[float]) -> dict:
    x = np.asarray([v for v in d if math.isfinite(v)], float)
    n = len(x)
    if n < 3:
        return {"n": n, "mean": float("nan"), "se": float("nan"), "z": float("nan")}
    m, se = float(x.mean()), float(x.std(ddof=1) / math.sqrt(n))
    return {"n": n, "mean": m, "se": se, "z": m / se if se > 0 else float("nan"),
            "win": float((x > 0).mean())}


def boot_se(d: list[float], rng: np.random.Generator, n_boot: int = 400) -> float:
    x = np.asarray([v for v in d if math.isfinite(v)], float)
    if len(x) < 3:
        return float("nan")
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    return float(x[idx].mean(axis=1).std(ddof=1))


def var_share(d: list[float], flags: list[bool]) -> float:
    x = np.asarray(d, float)
    fl = np.asarray(flags, bool)
    ok = np.isfinite(x)
    x, fl = x[ok], fl[ok]
    if len(x) < 3:
        return float("nan")
    dev = (x - x.mean()) ** 2
    return float(dev[fl].sum() / dev.sum()) if dev.sum() > 0 else float("nan")


def fmt(x: float, w: int = 8, p: int = 4, sign: bool = True) -> str:
    if x is None or not math.isfinite(x):
        return f"{'nan':>{w}}"
    return f"{x:{'+' if sign else ''}{w}.{p}f}"


# ------------------------------------------------------------------ context
def fit_ctx(turns: list[Turn]) -> Ctx:
    sd_live: dict[str, float] = {}
    sigma_t: dict[str, float] = {}
    std = ab.Standardizer()
    for kind in KINDS:
        sub = [t for t in turns if t.kind == kind]
        if not sub:
            continue
        vals = []
        for t in sub:
            for fam in ("king", "challenger"):
                fm = t.full.fams[fam]
                vals.append(ab.live_min_rg(fm.a, fm.m, t.full.t))
        sd_live[kind] = st.stdev(vals)
        sigma_t[kind] = ab.pooled_within_sd([t.full.t for t in sub])
        # teacher_own LOO anchors
        legs_T: dict[str, list[float]] = {}
        for t in sub:
            v, fm = t.loo, t.loo.fams["teacher_own"]
            legs_T.setdefault("R", []).append(ab.reason_leg(fm.a))
            legs_T.setdefault("RF", []).append(ab.frontier_reason(fm.f))
            legs_T.setdefault("V", []).append(ab.frontier_contrast(fm.f, fm.a))
            for fset in ("greedy", "sampled"):
                yF, _ = v.fset(fset)
                for sim in ab.SIM_NAMES:
                    legs_T.setdefault(f"act:{sim}:{fset}", []).append(
                        ab.action_closeness(fm.y, yF, v.yC, kind, sim))
        # AF has no teacher_own values (no lpC(y_C^0|z_F^j) echoes): anchor on king+challenger pooled, flagged
        legs_T["AF"] = [ab.frontier_action_leg(t.full.fams[f].b) for t in sub for f in ("king", "challenger")]
        for leg, vals in legs_T.items():
            std.fit(kind, leg, vals)
    return Ctx(sd_live, sigma_t, std)


# ------------------------------------------------------------------ tests
def subsets(turns: list[Turn]) -> list[tuple[str, list[Turn]]]:
    out = [("ALL", turns)]
    for kind in KINDS:
        sub = [t for t in turns if t.kind == kind]
        if sub:
            out.append((kind, sub))
    return out


def test_gates(turns: list[Turn]) -> dict:
    res: dict = {}
    for name, sub in subsets(turns):
        row = {}
        for g in GATES_MAIN + [("jaccard", 0.3, "greedy")]:
            r = Rule("g", g, "V")
            row[gate_name(g)] = {"full": st.mean(gate_of(t.full, r) for t in sub),
                                 "loo": st.mean(gate_of(t.loo, r) for t in sub)}
        # agreement landscape for context
        row["_ctx"] = {
            "yG_exact_any_yF": st.mean(ab.best_sim(t.full.yG, t.full.yF, t.kind, "exact") for t in sub),
            "yG_jac_yF": st.mean(ab.mean_sim(t.full.yG, t.full.yF, t.kind) for t in sub),
            "yF_self_jac": st.mean(ab.mean_sim(t.full.yF[0], t.full.yF[1:], t.kind) for t in sub),
            "yC_self_jac": st.mean(ab.mean_sim(t.full.yC[0], t.full.yC[1:], t.kind) for t in sub),
            "yC_vs_yF_jac": st.mean(st.mean(ab.mean_sim(yc, t.full.yF, t.kind) for yc in t.full.yC) for t in sub),
            "yF_all_same": st.mean(len({ab.norm_action(y, t.kind) for y in t.full.yF}) == 1 for t in sub),
        }
        res[name] = row
    return res


def test_headroom(turns: list[Turn], rules: list[Rule], ctx: Ctx) -> dict:
    res: dict = {}
    for name, sub in subsets(turns):
        rows = {}
        nan = float("nan")
        for r in rules:
            vals = {f: [score(t.loo, f, r, ctx) for t in sub] for f in FAM_HEADROOM}
            if needs_b(r):
                # no lpC(y_C^0|z_F^j) / lpC(y_F^0|z_F^j) echoes exist: the own families cannot be scored
                vals["frontier_own"] = [nan] * len(sub)
                vals["teacher_own"] = [nan] * len(sub)
            means = {f: float(np.nanmean(v)) if any(math.isfinite(x) for x in v) else nan for f, v in vals.items()}
            dFT = [a - b for a, b in zip(vals["frontier_own"], vals["teacher_own"])]
            dTK = [a - b for a, b in zip(vals["teacher_own"], vals["king"])]
            dFK = [a - b for a, b in zip(vals["frontier_own"], vals["king"])]
            fires = st.mean(gate_of(t.loo, r) for t in sub) if r.gate else 0.0
            rows[r.name] = {"means": means, "F-T": paired(dFT), "T-king": paired(dTK), "F-king": paired(dFK),
                            "disagree": fires}
            if r.alt.startswith("wB:"):
                rows[r.name]["collapse"] = st.mean(ab.weights_collapsed(weights_of(t.loo, r)) for t in sub)
        res[name] = rows
    return res


def test_noise(turns: list[Turn], rules: list[Rule], ctx: Ctx) -> dict:
    """Baseline for the SE ratio is the same-units live rule (live_c4 for raw, live_sd for sd);
    rho and z retention are unit-free and always relative to live_c4."""
    res: dict = {}
    rng = np.random.default_rng(3)
    by_name = {r.name: r for r in rules}
    for name, sub in subsets(turns):
        rows = {}
        d0 = [score(t.full, "challenger", by_name["live_c4"], ctx) - score(t.full, "king", by_name["live_c4"], ctx) for t in sub]
        z0 = paired(d0)["z"]
        base_bs = {u: boot_se([score(t.full, "challenger", by_name[b], ctx) - score(t.full, "king", by_name[b], ctx) for t in sub], rng)
                   for u, b in (("raw", "live_c4"), ("sd", "live_sd"))}
        for r in rules:
            d = [score(t.full, "challenger", r, ctx) - score(t.full, "king", r, ctx) for t in sub]
            flags = [gate_of(t.full, r) for t in sub]
            p = paired(d)
            bs = boot_se(d, rng)
            b0 = base_bs[r.units]
            p.update({"boot_se": bs, "se_ratio": bs / b0 if b0 else float("nan"), "rho": spearman(d0, d),
                      "z_keep": abs(p["z"]) / abs(z0) if z0 else float("nan"),
                      "disagree": st.mean(flags) if flags else 0.0, "var_share_disagree": var_share(d, flags)})
            if r.alt.startswith("wB:"):
                p["collapse"] = st.mean(ab.weights_collapsed(weights_of(t.full, r)) for t in sub)
            rows[r.name] = p
        res[name] = rows
    return res


ATTACKS_STORED = ("filler", "generic", "parrot", "copyF", "lscmd", "catcmd")
ATTACKS_ECHO = ("fparrotG", "copyF", "lscmd", "catcmd")


def attack_rows(views: list[View], attacks: tuple[str, ...], rules: list[Rule], ctx: Ctx) -> dict:
    rows = {}
    for r in rules:
        row = {}
        for atk in attacks:
            vs = [v for v in views if atk in v.fams]
            if needs_b(r):
                vs = [v for v in vs if v.fams[atk].b is not None and v.fams["king"].b is not None]
            if len(vs) < 10:
                continue
            k = np.array([score(v, "king", r, ctx) for v in vs])
            a = np.array([score(v, atk, r, ctx) for v in vs])
            ok = np.isfinite(k) & np.isfinite(a)
            k, a = k[ok], a[ok]
            if len(k) < 10:
                continue
            row[atk] = {"n": int(len(k)), "king": float(k.mean()), "attack": float(a.mean()),
                        "le": float((a <= k + 1e-12).mean()), "lt": float((a < k - 1e-12).mean()),
                        "tie": float((np.abs(a - k) <= 1e-12).mean()),
                        "harmless": bool((a <= k + 1e-12).mean() >= 0.85)}
        rows[r.name] = row
    return rows


def test_attacks(turns: list[Turn], rules: list[Rule], ctx: Ctx) -> dict:
    res: dict = {"stored": {}, "echo": {}, "licence": {}, "copy_payoff": {}}
    for name, sub in subsets(turns):
        res["stored"][name] = attack_rows([t.full for t in sub], ATTACKS_STORED, rules, ctx)
        ev = [t.echo for t in sub if t.echo]
        if len(ev) >= 10:
            res["echo"][name] = attack_rows(ev, ATTACKS_ECHO, rules, ctx)
        # B licence + leakage on the action swaps (stored king.fG for copyF; echo subset for the rest)
        lic = {}
        kB = [t.full.fams["king"].B for t in sub]
        lic["king"] = {"B_mean": st.mean(kB), "B_pass": st.mean(b >= 0.02 and not t.full.fams["king"].leak for b, t in zip(kB, sub))}
        cB = [t.full.fams["copyF"].B for t in sub]
        lic["copyF"] = {"B_mean": st.mean(cB), "B_pass": st.mean(b >= 0.02 and not t.full.fams["copyF"].leak for b, t in zip(cB, sub)),
                        "leak": st.mean(t.full.fams["copyF"].leak for t in sub)}
        for atk in ("fparrotG", "lscmd", "catcmd", "king"):
            vs = [v for v in ev if atk in v.fams and v.fams[atk].B is not None]
            if len(vs) >= 10:
                lic[f"{atk}@echo"] = {"n": len(vs), "B_mean": st.mean(v.fams[atk].B for v in vs),
                                      "B_pass": st.mean(v.fams[atk].B >= 0.02 and not v.fams[atk].leak for v in vs),
                                      "leak": st.mean(v.fams[atk].leak for v in vs)}
        res["licence"][name] = lic
        # copy-frontier memorisation payoff: greedy vs the sampled F it would be scored against
        gate_r = Rule("g", ("exact", 0.0, "sampled"), "V")
        dis = [t for t in sub if gate_of(t.full, gate_r)]
        pay = {}
        for lab, part in (("all", sub), ("disagree(exact/S)", dis)):
            if len(part) < 10:
                continue
            pay[lab] = {"n": len(part),
                        "yG_exact_any_yF": st.mean(ab.best_sim(t.full.yG, t.full.yF, t.kind, "exact") for t in part),
                        "yG_best_jac_ge0.5": st.mean(ab.best_sim(t.full.yG, t.full.yF, t.kind) >= 0.5 for t in part),
                        "sA_jac": {f: st.mean(ab.action_closeness(t.full.fams[f].y, t.full.yF, t.full.yC, t.kind) for t in part)
                                   for f in ("copyF", "king", "challenger")},
                        "sA_exact": {f: st.mean(ab.action_closeness(t.full.fams[f].y, t.full.yF, t.full.yC, t.kind, "exact") for t in part)
                                     for f in ("copyF", "king", "challenger")}}
        res["copy_payoff"][name] = pay
    return res


def test_range(turns: list[Turn], ctx: Ctx) -> dict:
    res: dict = {}
    rng = np.random.default_rng(5)
    base = Rule("live_c4", None, "live")
    for name, sub in subsets(turns):
        d0 = [score(t.full, "challenger", base, ctx) - score(t.full, "king", base, ctx) for t in sub]
        b0 = boot_se(d0, rng)
        rows = []
        sweeps = [("jaccard", th, fs) for th in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9) for fs in ("greedy", "sampled")]
        sweeps += [("surprise", nats, fs) for nats in (0.005, 0.01, 0.02, 0.05, 0.1) for fs in ("greedy", "sampled")]
        for g in sweeps:
            for alt in ("act:jaccard:1", "V", "minRFG"):
                r = Rule(f"{gate_name(g)}|{alt}", g, alt)
                d = [score(t.full, "challenger", r, ctx) - score(t.full, "king", r, ctx) for t in sub]
                flags = [gate_of(t.full, r) for t in sub]
                vF = [score(t.loo, "frontier_own", r, ctx) for t in sub]
                vT = [score(t.loo, "teacher_own", r, ctx) for t in sub]
                vK = [score(t.loo, "king", r, ctx) for t in sub]
                bs = boot_se(d, rng)
                rows.append({"gate": gate_name(g), "alt": alt, "rate": st.mean(flags),
                             "var_share": var_share(d, flags), "se_ratio": bs / b0 if b0 else float("nan"),
                             "rho": spearman(d0, d), "z_margin": paired(d)["z"],
                             "z_FT": paired([a - b for a, b in zip(vF, vT)])["z"],
                             "z_FK": paired([a - b for a, b in zip(vF, vK)])["z"],
                             "z_TK": paired([a - b for a, b in zip(vT, vK)])["z"]})
        res[name] = rows
    return res


# ------------------------------------------------------------------ report
def write_report(turns: list[Turn], info: dict, rules: list[Rule], ctx: Ctx, R: dict) -> str:
    L: list[str] = []
    P = L.append
    n_echo = sum(1 for t in turns if t.echo)
    cost = json.loads((OUT / "echoes_cost.json").read_text()) if (OUT / "echoes_cost.json").exists() else {}
    # stored lp entries are the ground truth for echo count; runs interrupted before writing echoes_cost.json
    # (16 turns of the first launch) are priced at the recorded per-echo rate
    n_lp = sum(len(json.loads(l)["lp"]) for l in ECHOES.open()) if ECHOES.exists() else 0
    rec_n, rec_usd = cost.get("echoes", 0), cost.get("cost_usd", 0.0)
    usd = rec_usd + max(0, n_lp - rec_n) * (rec_usd / rec_n if rec_n else 0.0)
    P(f"Frontier-arbiter rule simulation — {len(turns)} stored turns from {len(info['meta'])} duels "
      f"(dropped {info['dropped']}); frontier = engy glm-5.2 (3 samples @T0.8 + greedy); teacher echoes only, tau={ab.TAU}")
    P(f"new-attack echo subset: {n_echo} turns, {n_lp} Engy teacher echoes (qwen3.8-27b, canonical rendering), ${usd:.2f} "
      f"({rec_n} echoes / ${rec_usd:.2f} recorded by completed runs; the rest priced at that rate) "
      f"(honest king re-echoed on the same turns so attack comparisons are same-source)")
    P("")
    P("DEFINITIONS")
    P("  gate <method>/<F>: F disagrees when its action differs from ALL k teacher refs — exact (norm_action), jacθ (best token-Jaccard < θ),")
    P("    surθ (lpC(y_F|∅) < min_i lpC(y_C^i|∅) − θ per byte); F = G (one greedy sample) or S (3 samples @T0.8, strict majority must disagree).")
    P("  agree branch = live min(R,G), band_c 4.0 (live_c2 = the stored era's 2.0); disagree branch = the alt:")
    P("    act:<sim>:<s> = [mean_j sim(y_A,y_F^j) − mean_ij sim(y_C^i,y_F^j)] × s × sd(live turn scores of the dialect); sim jaccard or exact (0/1)")
    P("    RF = centered tempered LME_j[lpC(y_F^j|z_A) − lpC(y_F^j|∅)]; V = mean_j f_j − mean_i a_i; AF = LME_j[lpC(y_A|z_F^j) − lpC(y_A|∅)]")
    P("    minRFG = min(RF, G); minVG = min(V, G). Echo-space alts always use the 3 sampled F refs; act uses the gate's F set.")
    P("  |sd = every leg standardised per dialect by (mu, sd) of the TEACHER's own leave-one-out values (R, RF, V, act; AF anchored on king+challenger,")
    P("    no teacher_own echo exists); thought leg = 2 − |m − mean(t_i)|/sigma_t with sigma_t = pooled within-turn sd of t_i per dialect.")
    P("    No cross echoes lpC(y_C^i|z_C^j), i≠j, exist in this data, so the anchor is the teacher's ref-0 thought on refs 1..2 (one LOO value per turn).")
    P("  wB_<mode>/<F> = variant B: teacher refs weighted by mean Jaccard to the F set (softmax at T or hard 1[sim≥0.5]); R = weighted centered LME, G from weighted mu/sd.")
    P("  HEADROOM (leave-one-out: ref 0 and F sample 0 dropped everywhere, so every family is scored on k−1=2 teacher refs and 2 F refs):")
    P("    mean turn score per family; z(F−T) = paired mean(F_own − T_own)/SE; z(T−K) likewise; dis% = disagree-branch rate on the LOO gate.")
    P("  NOISE (full refs, live pairing): margin = mean(chal − king); SE = sd/√n; z = margin/SE; bSE/live = bootstrap SE ÷ that of the same-units")
    P("    live rule (live_c4 raw, live_sd sd); |z|/|z0| = |z| ÷ |z of live_c4| (how much of the live separation of these losing challengers survives);")
    P("    rho = Spearman of per-turn (chal − king) vs live_c4; dis% = gate rate; varD = share of Σ(d−d̄)² carried by disagree turns.")
    P("    minAG = min(act:jaccard:1, G), minAFG = min(AF, G): G (or typ) still bounds the disagree branch.")
    P("  ATTACKS (king's thought or action replaced): ≤king% = turns where attack ≤ honest king (ties count); harmless if ≥ 85%. '=' = exact-tie share.")
    P("    filler/generic/parrot(sampled y_F^0 pasted): stored swarm echoes, all turns. fparrotG (greedy y_F pasted into the thought), copyF (action := greedy y_F),")
    P("    lscmd/catcmd (action := `ls -la` / `cat <first file in prefix>`, bash only): Engy echoes on the subset; copyF/ls/cat on ALL turns for action-space alts.")
    P("")
    # ---- gates
    P("== GATE RATES (fraction of turns where F disagrees with all teacher refs) ==")
    gnames = [gate_name(g) for g in GATES_MAIN] + [gate_name(("jaccard", 0.3, "greedy"))]
    P(f"{'set':10} {'n':>5} " + " ".join(f"{g:>10}" for g in gnames) + "   | yG=any yF  yG~yF  F self  C self  C~F  F all same")
    for name, sub in subsets(turns):
        g = R["gates"][name]
        c = g["_ctx"]
        P(f"{name:10} {len(sub):5d} " + " ".join(f"{g[gn]['full']:10.1%}" for gn in gnames)
          + f"   | {c['yG_exact_any_yF']:8.0%} {c['yG_jac_yF']:6.2f} {c['yF_self_jac']:7.2f} {c['yC_self_jac']:7.2f} {c['yC_vs_yF_jac']:5.2f} {c['yF_all_same']:8.0%}")
    P("  (yG=any yF: greedy action exactly equals one of the 3 samples; ~: mean token-Jaccard; self = sample 0 vs samples 1..2)")
    P("")
    # ---- headroom
    P("== TEST 1 HEADROOM (leave-one-out; the arbiter's point is F_own > T_own) ==")
    for name, _ in subsets(turns):
        rows = R["headroom"][name]
        P(f"-- {name} --")
        P(f"{'rule':28} {'F_own':>8} {'T_own':>8} {'king':>8} {'chal':>8} {'z(F-T)':>7} {'F>T%':>5} {'z(T-K)':>7} {'z(F-K)':>7} {'dis%':>5}")
        for r in rules:
            x = rows[r.name]
            m = x["means"]
            P(f"{r.name:28} {fmt(m['frontier_own'])} {fmt(m['teacher_own'])} {fmt(m['king'])} {fmt(m['challenger'])} "
              f"{fmt(x['F-T']['z'], 7, 2)} {x['F-T'].get('win', float('nan')):5.0%} {fmt(x['T-king']['z'], 7, 2)} {fmt(x['F-king']['z'], 7, 2)} {x['disagree']:5.0%}"
              + (f"  collapse {x['collapse']:.0%}" if "collapse" in x else ""))
        P("")
    # ---- noise
    P("== TEST 2 NOISE (challenger − king, full refs; all four stored challengers lost live) ==")
    for name, _ in subsets(turns):
        rows = R["noise"][name]
        P(f"-- {name} --")
        P(f"{'rule':28} {'margin':>9} {'SE':>8} {'z':>7} {'bSE/live':>8} {'|z|/|z0|':>8} {'rho':>6} {'dis%':>5} {'varD':>5}")
        for r in rules:
            x = rows[r.name]
            P(f"{r.name:28} {fmt(x['mean'], 9, 5)} {fmt(x['se'], 8, 5, False)} {fmt(x['z'], 7, 2)} {fmt(x['se_ratio'], 8, 2, False)} "
              f"{fmt(x['z_keep'], 8, 2, False)} {fmt(x['rho'], 6, 3)} {x['disagree']:5.0%} {fmt(x['var_share_disagree'], 5, 2, False)}"
              + (f"  collapse {x['collapse']:.0%}" if "collapse" in x else ""))
        P("")
    # ---- attacks
    P("== TEST 3 ATTACKS (≤king% = share of turns the attack scores ≤ the honest king; harmless ≥ 85%) ==")
    for name, _ in subsets(turns):
        P(f"-- {name}: stored variants (all turns; copyF/lscmd/catcmd action-space only here) --")
        rows = R["attacks"]["stored"][name]
        P(f"{'rule':28} " + " ".join(f"{a:>17}" for a in ATTACKS_STORED))
        P(f"{'':28} " + " ".join(f"{'mean  ≤king%  =%':>17}" for _ in ATTACKS_STORED))
        for r in rules:
            cells = []
            for a in ATTACKS_STORED:
                x = rows[r.name].get(a)
                cells.append(f"{fmt(x['attack'], 7, 4)} {x['le']:5.0%} {x['tie']:3.0%}" if x else f"{'-':>17}")
            P(f"{r.name:28} " + " ".join(cells))
        if name in R["attacks"]["echo"]:
            rows = R["attacks"]["echo"][name]
            n_e = next((x["n"] for rr in rows.values() for x in rr.values()), 0)
            P(f"-- {name}: Engy-echo subset (n≈{n_e}; honest king re-echoed; AF now real for copyF/lscmd/catcmd) --")
            P(f"{'rule':28} " + " ".join(f"{a:>17}" for a in ATTACKS_ECHO))
            for r in rules:
                cells = []
                for a in ATTACKS_ECHO:
                    x = rows[r.name].get(a)
                    cells.append(f"{fmt(x['attack'], 7, 4)} {x['le']:5.0%} {x['tie']:3.0%}" if x else f"{'-':>17}")
                P(f"{r.name:28} " + " ".join(cells))
        lic = R["attacks"]["licence"][name]
        P("B licence (B = lpC(y|z) − lpC(y|∅) ≥ 0.02 and no z⊃y leakage): " + "; ".join(
            f"{k}: B {v['B_mean']:+.4f} pass {v['B_pass']:.0%}" + (f" leak {v['leak']:.0%}" if "leak" in v else "") for k, v in lic.items()))
        pay = R["attacks"]["copy_payoff"][name]
        for lab, x in pay.items():
            P(f"copy-frontier payoff [{lab}, n={x['n']}]: greedy exactly equals a sampled y_F on {x['yG_exact_any_yF']:.0%}, best Jaccard ≥0.5 on {x['yG_best_jac_ge0.5']:.0%}; "
              f"s_A(jac) copyF {x['sA_jac']['copyF']:+.3f} vs king {x['sA_jac']['king']:+.3f} chal {x['sA_jac']['challenger']:+.3f}; "
              f"s_A(exact) copyF {x['sA_exact']['copyF']:+.3f} king {x['sA_exact']['king']:+.3f}")
        P("")
    # ---- range
    P("== TEST 4 RANGE (gate-threshold sweep; full-ref noise + LOO headroom) ==")
    for name, _ in subsets(turns):
        P(f"-- {name} --")
        P(f"{'gate':10} {'alt':14} {'rate':>6} {'varD':>6} {'bSE/live':>8} {'rho':>6} {'z_margin':>8} {'z(F-T)':>7} {'z(F-K)':>7} {'z(T-K)':>7}")
        for x in R["range"][name]:
            P(f"{x['gate']:10} {x['alt']:14} {x['rate']:6.1%} {fmt(x['var_share'], 6, 2, False)} {fmt(x['se_ratio'], 8, 2, False)} "
              f"{fmt(x['rho'], 6, 3)} {fmt(x['z_margin'], 8, 2)} {fmt(x['z_FT'], 7, 2)} {fmt(x['z_FK'], 7, 2)} {fmt(x['z_TK'], 7, 2)}")
        P("")
    # ---- anchors
    P("== sd-unit anchors (teacher_own LOO per dialect) ==")
    for kind in KINDS:
        parts = [f"{leg} mu {ctx.std.mu[(kind, leg)]:+.4f} sd {ctx.std.sigma[(kind, leg)]:.4f}"
                 for leg in ("R", "RF", "V", "act:jaccard:sampled", "act:jaccard:greedy", "AF") if (kind, leg) in ctx.std.mu]
        P(f"{kind:10} sd_live {ctx.sd_live.get(kind, float('nan')):.4f} sigma_t {ctx.sigma_t.get(kind, float('nan')):.4f} | " + "; ".join(parts))
    P("")
    P("== DECISION per rule (pooled): headroom = F_own beats BOTH the teacher's own thought and the king, z(F−T) ≥ +2 AND z(F−K) ≥ +2 "
      "(AF rules: n/a); attacks = every stored + echo attack harmless (≤king on ≥85%; a rule blind to the action ties action swaps, "
      "flagged 'blind'); noise is reported, not gated: |z|/|z0| = retained separation of the live losers, rho vs live_c4 ==")
    P("  'opened' = ≤king% < 85% AND more than 5 pp below live_c4's own ≤king% on that attack (live_c4 itself sits near 58% on the sampled parrot).")
    dec = {}
    A_h, A_n, A_s, A_e = R["headroom"]["ALL"], R["noise"]["ALL"], R["attacks"]["stored"]["ALL"], R["attacks"]["echo"].get("ALL", {})
    live_le = {k: a["le"] for k, a in (A_s["live_c4"] | A_e.get("live_c4", {})).items()}
    for r in rules:
        h = A_h[r.name]
        ok_h = (h["F-T"]["z"] >= 2.0) and (h["F-king"]["z"] >= 2.0)
        nz = A_n[r.name]
        merged = A_s[r.name] | A_e.get(r.name, {})
        bad = [k for k, a in merged.items() if not a["harmless"]]
        opened = [k for k in bad if merged[k]["le"] < live_le.get(k, 1.0) - 0.05]
        blind = [k for k, a in merged.items() if a["tie"] >= 0.99]
        ok_a = bool(merged) and not opened
        dec[r.name] = {"headroom": ok_h, "attacks": ok_a, "below85": bad, "opened": opened, "blind_to": blind,
                       "z_keep": nz["z_keep"], "rho": nz["rho"], "se_ratio": nz["se_ratio"],
                       "pass": ok_h and ok_a}
        P(f"{r.name:28} headroom {str(ok_h):5} attacks {str(ok_a):5} |z|/|z0| {fmt(nz['z_keep'], 4, 2, False)} rho {fmt(nz['rho'], 5, 2)} "
          f"-> {'PASS' if dec[r.name]['pass'] else 'fail'}"
          + (f"  opened: {', '.join(opened)}" if opened else "")
          + (f"  <85%: {', '.join(k for k in bad if k not in opened)}" if [k for k in bad if k not in opened] else "")
          + (f"  blind: {', '.join(blind)}" if blind else ""))
    R["decision"] = dec
    return "\n".join(L) + "\n"


def cmd_run(args) -> int:
    turns, info = load_turns()
    if args.limit:
        turns = turns[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    rules = build_rules()
    ctx = fit_ctx(turns)
    t0 = time.time()
    R: dict = {"n": len(turns), "dropped": info["dropped"], "rules": [r.__dict__ for r in rules]}
    R["gates"] = test_gates(turns)
    print(f"gates {time.time() - t0:.0f}s", flush=True)
    R["headroom"] = test_headroom(turns, rules, ctx)
    print(f"headroom {time.time() - t0:.0f}s", flush=True)
    R["noise"] = test_noise(turns, rules, ctx)
    print(f"noise {time.time() - t0:.0f}s", flush=True)
    R["attacks"] = test_attacks(turns, rules, ctx)
    print(f"attacks {time.time() - t0:.0f}s", flush=True)
    R["range"] = test_range(turns, ctx)
    print(f"range {time.time() - t0:.0f}s", flush=True)
    text = write_report(turns, info, rules, ctx, R)
    (OUT / "report.txt").write_text(text)
    (OUT / "report.json").write_text(json.dumps(R, indent=1, default=str))
    print(text)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("echo")
    s.add_argument("--n", type=int, default=200)
    s.add_argument("--concurrency", type=int, default=24)
    s = sub.add_parser("run")
    s.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "echo":
        return asyncio.run(cmd_echo(args))
    return cmd_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
