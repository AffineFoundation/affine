#!/usr/bin/env python
"""Per-reign attribution: did the new king move on the held-out benchmarks,
and what changed in D between the two kings' submissions?

  python attribution.py --run <run_id>                # against the previous reign's card
  python attribution.py --run <run_id> --against <run_id>
  python attribution.py --latest                      # newest complete card
  ... [--out DIR] [--discord] [--dry-run]

Inputs (all read-only):
  affine/state/benchsuite/*.json      scorecards written by ops/benchsuite/publish.py
  <run dir>/king/<cell>/summary.json  per-rollout rows (local run dir, else R2)
  affine/state/history.jsonl          crowned events -> submission (reveal) block
  affine/state/state.json             intake rows -> block/time anchors
  ops/corpus_build/state.json         fold history (epoch, published at)
  ops/corpus_build/work/pack_NNNN/turns_NNNN_merged.parquet   D at each epoch
  rollouts/rollouts/sources.toml      source -> group, [mix] targets
  affine/state/king_review/reign-<digest12>/report.json      failure taxonomy

Outputs: <out>/<run_id>.md (the report), <out>/<run_id>.json (the numbers),
<out>/axis.jsonl (one line per report), <out>/latest.json. Optionally one
line to the private Arbos Discord channel.

Terms. Cell = one (benchmark, temperature). Paired delta = mean over the
tasks both runs graded of (new king score - previous king score), in
percentage points; its noise band is 1.96 x the paired standard error, so
"outside noise" means the 95 % interval of the paired delta excludes 0.
Submission = the reveal block of the crowning submission; D "before a king"
is the last corpus epoch published before that block's time.
"""

from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import math
import os
import sys
import time
import tomllib
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CARDS_DIR = Path(os.environ.get("BENCHSUITE_CARDS", str(REPO / "affine" / "state" / "benchsuite")))
HISTORY = REPO / "affine" / "state" / "history.jsonl"
VALIDATOR_STATE = REPO / "affine" / "state" / "state.json"
FOLD_STATE = REPO / "ops" / "corpus_build" / "state.json"
FOLD_WORK = REPO / "ops" / "corpus_build" / "work"
SOURCES_TOML = REPO / "rollouts" / "rollouts" / "sources.toml"
KING_REVIEW = REPO / "affine" / "state" / "king_review"
OUT_DIR = REPO / "affine" / "state" / "improvement_loop"
RUNS_DIR = Path(os.environ.get("BENCHSUITE_RUNS", str(Path.home() / "benchsuite" / "runs")))
PUBLIC_BASE = "https://data.affine.io/research/benchsuite"
DISCORD_CHANNEL = "1510910974498967613"   # private Arbos ops channel only (operator directive 2026-09-12)
DISCORD_TOKEN_ENV = "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR"
VALIDATOR_ENV = Path.home() / ".affine-validator.env"
REPO_ENV = REPO / ".env"
SECONDS_PER_BLOCK = 12.0
KING_GROUP_PREFIX = "king_"
BENCH_TO_GROUPS = {   # which D groups a benchmark's capability axis leans on (for the axis table)
    "math": ["math"], "coding": ["coding"], "coding-agent": ["coding", "king_fail", "king_pivot", "king_recoverable"],
    "tool-use": ["tool_use", "completion", "king_done"], "instruction-following": ["general"],
    "knowledge": ["general"], "long-context": ["general"], "formal-math": [],
}


def log(msg: str) -> None:
    print(f"[attribution] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}", flush=True)


def env_file_value(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for path in (VALIDATOR_ENV, REPO_ENV):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix("export ").strip()
            if line.startswith(f"{name}="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def parse_ts(s: str) -> float:
    s = s.replace("Z", "+00:00")
    d = dt.datetime.fromisoformat(s)
    if d.tzinfo is None:
        d = d.replace(tzinfo=dt.timezone.utc)
    return d.timestamp()


def iso(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ---------------------------------------------------------------- scorecards

def load_cards(cards_dir: Path = CARDS_DIR) -> list[dict]:
    cards = []
    for p in sorted(cards_dir.glob("*.json")):
        try:
            c = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(c, dict) and c.get("rows") and (c.get("king") or {}).get("digest"):
            cards.append(c)
    cards.sort(key=lambda c: c.get("created_at") or "")
    return cards


def card_by_run(cards: list[dict], run_id: str) -> dict | None:
    return next((c for c in cards if c["run_id"] == run_id), None)


def pick_previous(cards: list[dict], cur: dict) -> dict | None:
    """A merged view of the highest reign below the current one: every card
    of that reign (full pass, Lium parity run, single-env add-ons) contributes
    the cells it has; the newest card wins a cell. Rows carry `_run_id` so the
    per-task summaries are fetched from the right run. Hardware moves numbers
    a little (benchsuite.md 9.1), far less than a missing cell would."""
    reign = (cur.get("king") or {}).get("reign")
    digest = cur["king"]["digest"]
    older = [c for c in cards if c["king"]["digest"] != digest
             and (c["king"].get("reign") or 0) < (reign or 10 ** 9)
             and (c.get("created_at") or "") < (cur.get("created_at") or "~")]
    if not older:
        return None
    best_reign = max(c["king"].get("reign") or 0 for c in older)
    cands = sorted([c for c in older if (c["king"].get("reign") or 0) == best_reign],
                   key=lambda c: c.get("created_at") or "")
    rows: dict[str, dict] = {}
    for c in cands:   # oldest first, so the newest card overwrites
        for r in c["rows"]:
            if r.get("king") and r["king"].get("score") is not None:
                rows[cell_key(r)] = {**r, "_run_id": c["run_id"]}
    base = cands[-1]
    return {**base, "run_id": " + ".join(c["run_id"] for c in cands) if len(cands) > 1 else base["run_id"],
            "rows": list(rows.values()), "merged_from": [c["run_id"] for c in cands]}


# ------------------------------------------------------------ per-task rows

def cell_key(row: dict) -> str:
    t = row["temperature"]
    return f"{row['env']}__t{int(t) if float(t).is_integer() else t}"


def load_summary(run_id: str, cell: str, cache_dir: Path, model: str = "king") -> dict | None:
    local = RUNS_DIR / run_id / model / cell / "summary.json"
    if local.exists():
        try:
            return json.loads(local.read_text())
        except ValueError:
            pass
    cache = cache_dir / run_id / f"{model}__{cell}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except ValueError:
            pass
    url = f"{PUBLIC_BASE}/{run_id}/{model}/{cell}/summary.json"
    try:
        r = requests.get(url, timeout=60)
        if r.status_code != 200:
            return None
        s = r.json()
    except (requests.RequestException, ValueError):
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(s))
    return s


FINISH_FAILS = ("timeout", "context_overflow")   # scored 0 by the suite; depend on the pod's contention too


def task_scores(summary: dict | None, finished_only: bool = False) -> dict[str, float]:
    """task_key -> mean score over that task's graded rollouts. Infra errors
    (no score) are left out: they are infrastructure, not the model. With
    finished_only, rollouts that hit the time budget or the context window
    are left out too (benchsuite.md 9.4: compare finished-only across runs)."""
    if not summary:
        return {}
    acc: dict[str, list[float]] = collections.defaultdict(list)
    for r in summary.get("rollouts") or []:
        if r.get("score") is None or r.get("error_class") == "infra":
            continue
        if finished_only and r.get("error_class") in FINISH_FAILS:
            continue
        acc[r["task_key"]].append(float(r["score"]))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def paired(prev: dict[str, float], cur: dict[str, float]) -> dict | None:
    common = sorted(set(prev) & set(cur))
    n = len(common)
    if n < 2:
        return None
    d = [cur[k] - prev[k] for k in common]
    mean = sum(d) / n
    var = sum((x - mean) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return {"n": n, "prev": sum(prev[k] for k in common) / n, "cur": sum(cur[k] for k in common) / n,
            "delta": mean, "se": se, "band": 1.96 * se, "z": (mean / se) if se else 0.0,
            "wins": sum(1 for x in d if x > 0), "losses": sum(1 for x in d if x < 0)}


def unpaired(prev_side: dict, cur_side: dict) -> dict | None:
    """Fallback when per-task rows are missing: difference of two independent
    scores with the Wilson half-widths combined in quadrature (conservative)."""
    if not prev_side or not cur_side or prev_side.get("score") is None or cur_side.get("score") is None:
        return None
    hp = (prev_side["ci95"][1] - prev_side["ci95"][0]) / 2
    hc = (cur_side["ci95"][1] - cur_side["ci95"][0]) / 2
    band = math.sqrt(hp * hp + hc * hc)
    delta = cur_side["score"] - prev_side["score"]
    return {"n": min(prev_side.get("n") or 0, cur_side.get("n") or 0), "prev": prev_side["score"],
            "cur": cur_side["score"], "delta": delta, "se": band / 1.96, "band": band,
            "z": (delta / (band / 1.96)) if band else 0.0, "wins": None, "losses": None}


def compare_cards(cur: dict, prev: dict, cache_dir: Path) -> list[dict]:
    prev_rows = {cell_key(r): r for r in prev["rows"]}
    out = []
    for r in sorted(cur["rows"], key=lambda r: (r.get("group") or "", r["env"], r["temperature"])):
        ck = cell_key(r)
        pr = prev_rows.get(ck)
        if not pr or not r.get("king") or not pr.get("king"):
            continue
        cur_s = load_summary(cur["run_id"], ck, cache_dir)
        prev_s = load_summary(pr.get("_run_id") or prev["run_id"], ck, cache_dir)
        stat = paired(task_scores(prev_s), task_scores(cur_s))
        method = "paired"
        finished = None
        if not stat:
            stat = unpaired(pr["king"], r["king"])
            method = "unpaired"
        elif (pr["king"].get("n_timeout") or r["king"].get("n_timeout")
              or pr["king"].get("n_context_overflow") or r["king"].get("n_context_overflow")):
            finished = paired(task_scores(prev_s, True), task_scores(cur_s, True))
        if not stat:
            continue
        verdict = "up" if stat["delta"] > stat["band"] else "down" if stat["delta"] < -stat["band"] else "flat"
        k, pk = r["king"], pr["king"]
        out.append({"cell": ck, "env": r["env"], "group": r.get("group"), "temperature": r["temperature"],
                    "method": method, "verdict": verdict, **stat, "finished_only": finished,
                    "teacher": (r.get("teacher") or {}).get("score"),
                    "cap_prev": pk.get("finish_length_frac"), "cap_cur": k.get("finish_length_frac"),
                    "timeout_prev": pk.get("n_timeout"), "timeout_cur": k.get("n_timeout"),
                    "n_prev": pk.get("n"), "n_cur": k.get("n"),
                    "partial": ck in (cur.get("unfinished_cells") or [])})
    return out


def axis_answer(cells: list[dict]) -> tuple[str, str]:
    ups = [c for c in cells if c["verdict"] == "up"]
    downs = [c for c in cells if c["verdict"] == "down"]
    n = len(cells)
    if not n:
        return "UNKNOWN", "no comparable cells"
    if ups and not downs:
        return "YES", f"{len(ups)}/{n} cells up outside noise, none down"
    if downs and not ups:
        return "NO", f"{len(downs)}/{n} cells down outside noise, none up"
    if ups and downs:
        return "MIXED", f"{len(ups)} up / {len(downs)} down outside noise of {n}"
    return "NO", f"all {n} cells inside noise (flat)"


# --------------------------------------------------------------- kings and D

def crowned_events() -> list[dict]:
    ev = []
    if not HISTORY.exists():
        return ev
    for line in HISTORY.read_text().splitlines():
        try:
            v = json.loads(line)
        except ValueError:
            continue
        if v.get("event") == "crowned":
            ev.append(v)
    return ev


def block_anchors() -> list[tuple[int, float]]:
    """(block, unix time) pairs from the validator's intake log, plus the
    crowned events themselves (crown_block ~ crown time)."""
    anchors = []
    try:
        st = json.loads(VALIDATOR_STATE.read_text())
        for it in st.get("intake") or []:
            if it.get("block") and it.get("at"):
                anchors.append((int(it["block"]), parse_ts(it["at"])))
    except (OSError, ValueError):
        pass
    for v in crowned_events():
        cb = v.get("crown_block")
        if cb and v.get("at"):
            anchors.append((int(cb), parse_ts(v["at"])))
    anchors.sort()
    return anchors


def block_time(block: int, anchors: list[tuple[int, float]]) -> float | None:
    if not anchors:
        return None
    b0, t0 = min(anchors, key=lambda a: abs(a[0] - block))
    return t0 + (block - b0) * SECONDS_PER_BLOCK


def king_submission(digest: str, anchors) -> dict:
    for v in crowned_events():
        if str(v.get("revision", "")).startswith(digest[:12]):
            block = int(v["block"]) if v.get("block") else None
            crowned_at = parse_ts(v["at"])
            sub = block_time(block, anchors) if block else None
            return {"digest": digest, "reign": int(v.get("reign_number") or 0), "block": block,
                    "submitted_at": sub, "crowned_at": crowned_at, "challenge_id": v.get("challenge_id"),
                    "hotkey": v.get("hotkey"), "margin": (v.get("verdict") or {}).get("margin"),
                    "z": (v.get("verdict") or {}).get("z"), "via": (v.get("verdict") or {}).get("via")}
    return {"digest": digest, "reign": None, "block": None, "submitted_at": None, "crowned_at": None}


def fold_history() -> list[dict]:
    try:
        return json.load(open(FOLD_STATE))["history"]
    except (OSError, ValueError, KeyError):
        return []


def epoch_before(ts: float | None, hist: list[dict]) -> dict | None:
    if ts is None:
        return hist[-1] if hist else None
    before = [h for h in hist if parse_ts(h["at"]) <= ts]
    return before[-1] if before else None


def composition(epoch: int, cache_dir: Path) -> dict | None:
    """Turns / strata per group, kinds, sources of D at one published epoch,
    from the fold's merged index parquet (cached: an epoch never changes)."""
    cache = cache_dir / f"composition_{epoch:04d}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    p = FOLD_WORK / f"pack_{epoch:04d}" / f"turns_{epoch:04d}_merged.parquet"
    if not p.exists():
        return None
    import pyarrow.parquet as pq   # noqa: PLC0415  (heavy import, only on a cache miss)
    src = tomllib.load(open(SOURCES_TOML, "rb"))
    src2grp = {k: v.get("group", "coding") for k, v in src.get("source", {}).items()}
    groups = set(src["mix"]) | {"king_fail", "king_loop_onset", "king_pivot", "king_recoverable",
                                "king_done", "completion", "general", "math", "tool_use", "nl2repo",
                                "coding", "terminal"}
    t = pq.read_table(p, columns=["stratum", "source", "action_kind", "rollout_id"]).to_pydict()
    gt = collections.Counter(); gs = collections.defaultdict(set); st = collections.Counter()
    kt = collections.Counter(); gk = collections.defaultdict(collections.Counter)
    for stratum, source, kind, rid in zip(t["stratum"], t["source"], t["action_kind"], t["rollout_id"]):
        pre = str(stratum).split(":")[0]
        g = pre if pre in groups else src2grp.get(source or "", "coding")
        gt[g] += 1; gs[g].add(stratum); st[source or "?"] += 1; kt[kind or "?"] += 1; gk[g][kind or "?"] += 1
    n = len(t["stratum"]); ns = sum(len(v) for v in gs.values())
    out = {"epoch": epoch, "n_turns": n, "n_strata": ns, "group_turns": dict(gt),
           "group_strata": {g: len(v) for g, v in gs.items()}, "source_turns": dict(st),
           "kind_turns": dict(kt), "group_kinds": {g: dict(c) for g, c in gk.items()}}
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def mix_targets() -> dict[str, float]:
    try:
        return {k: float(v) for k, v in tomllib.load(open(SOURCES_TOML, "rb"))["mix"].items()}
    except (OSError, KeyError, ValueError):
        return {}


def d_changes(prev_sub: float | None, cur_sub: float | None, cache_dir: Path) -> dict:
    hist = fold_history()
    e0 = epoch_before(prev_sub, hist)
    e1 = epoch_before(cur_sub, hist)
    between = [h for h in hist if e0 and e1 and e0["epoch"] < h["epoch"] <= e1["epoch"]]
    c0 = composition(e0["epoch"], cache_dir) if e0 else None
    c1 = composition(e1["epoch"], cache_dir) if e1 else None
    return {"epoch_prev": e0, "epoch_cur": e1, "epochs_between": between, "comp_prev": c0, "comp_cur": c1,
            "mix": mix_targets()}


def king_review_summary(digest: str) -> dict | None:
    p = KING_REVIEW / f"reign-{digest[:12]}" / "report.json"
    if not p.exists():
        return None
    try:
        r = json.loads(p.read_text())
    except ValueError:
        return None
    cats = r.get("categories") or {}
    n = sum(cats.values()) or 1
    return {"n_judged": r.get("n_judged"), "n_multi_turn": r.get("n_multi_turn"),
            "categories": {k: v for k, v in sorted(cats.items(), key=lambda kv: -kv[1])},
            "category_share": {k: round(v / n, 3) for k, v in cats.items()},
            "top_patterns": (r.get("top_patterns") or [])[:5], "cost_usd": r.get("cost_usd")}


# ------------------------------------------------------------------- report

def pct(x, d=1):
    return "–" if x is None else f"{100 * x:.{d}f}"


def fmt(x, spec: str) -> str:
    return "?" if x is None else format(float(x), spec)


def share_table(c0: dict | None, c1: dict | None, mix: dict) -> list[str]:
    if not c0 or not c1:
        return ["_(fold index parquet not available for one of the epochs)_"]
    groups = sorted(set(c0["group_strata"]) | set(c1["group_strata"]),
                    key=lambda g: -(c1["group_strata"].get(g, 0)))
    lines = ["| group | strata share before (%) | after (%) | Δ pt | turns before → after | `[mix]` target now |",
             "|---|---:|---:|---:|---|---:|"]
    for g in groups:
        a = c0["group_strata"].get(g, 0) / c0["n_strata"]; b = c1["group_strata"].get(g, 0) / c1["n_strata"]
        lines.append(f"| {g} | {100 * a:.1f} | {100 * b:.1f} | {100 * (b - a):+.1f} | "
                     f"{c0['group_turns'].get(g, 0):,} → {c1['group_turns'].get(g, 0):,} | "
                     f"{pct(mix.get(g), 0) if g in mix else '–'} |")
    kinds = sorted(set(c0["kind_turns"]) | set(c1["kind_turns"]), key=lambda k: -c1["kind_turns"].get(k, 0))
    lines += ["", "| action kind (share of turns) | before (%) | after (%) |", "|---|---:|---:|"]
    for k in kinds:
        lines.append(f"| {k} | {100 * c0['kind_turns'].get(k, 0) / c0['n_turns']:.1f} | "
                     f"{100 * c1['kind_turns'].get(k, 0) / c1['n_turns']:.1f} |")
    new_src = [(s, c1["source_turns"].get(s, 0) - c0["source_turns"].get(s, 0)) for s in c1["source_turns"]
               if c1["source_turns"].get(s, 0) - c0["source_turns"].get(s, 0) > 0]
    new_src.sort(key=lambda x: -x[1])
    if new_src:
        lines += ["", "Sources that grew (turns added): " + ", ".join(
            f"`{s}` +{d:,}" + (" (new)" if s not in c0["source_turns"] else "") for s, d in new_src[:16])]
    return lines


def render(cur: dict, prev: dict, cells: list[dict], subs: dict, dch: dict, reviews: dict,
           answer: tuple[str, str]) -> tuple[str, str]:
    k, pk = cur["king"], prev["king"]
    cs, ps = subs["cur"], subs["prev"]
    e0, e1 = dch["epoch_prev"], dch["epoch_cur"]
    hdr = f"## Reign {k.get('reign')} (`king-{k['digest'][:12]}`) vs reign {pk.get('reign')} (`king-{pk['digest'][:12]}`)"
    L = [hdr, "",
         f"- Runs: `{cur['run_id']}` ({(cur.get('where') or {}).get('provider', '?')}, status {cur.get('status')}) "
         f"vs `{prev['run_id']}` ({(prev.get('where') or {}).get('provider', '?')}). Teacher cells from the reference run; "
         f"the comparison below is king vs king on the same tasks.",
         f"- Submissions: reign {pk.get('reign')} block {ps.get('block')} ≈ {iso(ps['submitted_at']) if ps.get('submitted_at') else '?'}; "
         f"reign {k.get('reign')} block {cs.get('block')} ≈ {iso(cs['submitted_at']) if cs.get('submitted_at') else '?'} "
         f"(crowned {iso(cs['crowned_at']) if cs.get('crowned_at') else '?'}, duel margin {fmt(cs.get('margin'), '+.5f')}, "
         f"z {fmt(cs.get('z'), '.2f')}{', via ' + cs['via'] if cs.get('via') else ''}).",
         f"- D before each submission: epoch {e0['epoch'] if e0 else '?'} ({e0['at'][:16] if e0 else '?'}) → "
         f"epoch {e1['epoch'] if e1 else '?'} ({e1['at'][:16] if e1 else '?'}); "
         f"{len(dch['epochs_between'])} folds in between, +{sum(h['n_turns'] for h in dch['epochs_between']):,} turns. "
         f"A checkpoint is trained on D published hours to days before its upload, so the last one or two epochs "
         f"before a submission were most likely not in its training data yet.",
         "", f"**Axis answer: {answer[0]} — {answer[1]}.**", "",
         "### Benchmark deltas (king vs previous king, same tasks, paired)", "",
         "| cell | group | n | prev % | new % | Δ pt | ±band | verdict | W/L | cap % prev→new | t/o prev→new | teacher % |",
         "|---|---|---:|---:|---:|---:|---:|---|---|---|---|---:|"]
    for c in cells:
        wl = f"{c['wins']}/{c['losses']}" if c["wins"] is not None else "–"
        flag = " (partial)" if c.get("partial") else ("" if c["method"] == "paired" else " (unpaired)")
        fo = c.get("finished_only")
        tmo = f"{c['timeout_prev']}→{c['timeout_cur']}"
        if fo:
            tmo += f" (finished-only Δ {100 * fo['delta']:+.1f} ±{100 * fo['band']:.1f}, n={fo['n']})"
        L.append(f"| {c['cell']} | {c.get('group') or ''} | {c['n']} | {pct(c['prev'])} | {pct(c['cur'])} | "
                 f"{100 * c['delta']:+.1f} | {100 * c['band']:.1f} | {c['verdict']}{flag} | {wl} | "
                 f"{pct(c['cap_prev'], 0)}→{pct(c['cap_cur'], 0)} | {tmo} | {pct(c['teacher'])} |")
    L += ["", "Band = 1.96 × paired SE over the tasks both runs graded; a cell is 'up'/'down' only when |Δ| exceeds it. "
          "Time-budget and context-window failures score 0 as in the suite; they also depend on pod contention, so cells "
          "with many of them carry a finished-only Δ (both runs finished the task) as the model-only read.", "",
          f"### What changed in D between the two submissions (epoch {e0['epoch'] if e0 else '?'} → {e1['epoch'] if e1 else '?'})", ""]
    if dch["epochs_between"]:
        L.append("Folds: " + "; ".join(f"epoch {h['epoch']} {h['at'][5:16]} (+{h['n_turns']:,})" for h in dch["epochs_between"]))
        L.append("")
    L += share_table(dch["comp_prev"], dch["comp_cur"], dch["mix"])
    L += ["", "### King review (LLM-judged failure taxonomy of failed king rollouts)", ""]
    for label, dg in (("previous", pk["digest"]), ("new", k["digest"])):
        rv = reviews.get(dg)
        if not rv:
            L.append(f"- {label} king `{dg[:12]}`: no review report yet.")
            continue
        cats = ", ".join(f"{c} {v} ({pct(rv['category_share'][c], 0)} %)" for c, v in list(rv["categories"].items())[:6])
        L.append(f"- {label} king `{dg[:12]}`: {rv['n_judged']} judged ({rv['n_multi_turn']} multi-turn): {cats}.")
    L += ["", "### Does the movement line up with the D change?", ""]
    L += lineup_notes(cells, dch)
    body = "\n".join(L) + "\n"
    one = one_liner(cur, prev, cells, answer, dch)
    return body, one


def lineup_notes(cells: list[dict], dch: dict) -> list[str]:
    """Mechanical cross-check per moved cell: did the D groups that cell leans
    on grow or shrink between the submissions? A human still has to judge
    causation; this only states co-movement or its absence."""
    c0, c1 = dch["comp_prev"], dch["comp_cur"]
    notes = []
    moved = [c for c in cells if c["verdict"] != "flat"]
    if not moved:
        notes.append("- No cell moved outside its noise band, so there is nothing to attribute: the D change between "
                     "these submissions did not show up on the held-out sets (or the miner did not train on it yet).")
    for c in moved:
        groups = BENCH_TO_GROUPS.get(c.get("group") or "", [])
        parts = []
        if c0 and c1:
            for g in groups:
                a = c0["group_strata"].get(g, 0) / c0["n_strata"]; b = c1["group_strata"].get(g, 0) / c1["n_strata"]
                parts.append(f"{g} {100 * a:.1f}→{100 * b:.1f} %")
        direction = "up" if c["delta"] > 0 else "down"
        notes.append(f"- `{c['cell']}` {direction} {100 * c['delta']:+.1f} pt: related D strata shares "
                     f"{'; '.join(parts) if parts else 'n/a'}. "
                     + ("Co-movement is not causation; check the failure class behind the cell." if parts else
                        "No D group maps to this benchmark (no lever here)."))
    return notes


def one_liner(cur: dict, prev: dict, cells: list[dict], answer: tuple[str, str], dch: dict) -> str:
    k, pk = cur["king"], prev["king"]
    moved = sorted([c for c in cells if c["verdict"] != "flat"], key=lambda c: -abs(c["delta"]))
    mv = ", ".join(f"{c['env']}@T{c['temperature']} {100 * c['delta']:+.1f}" for c in moved[:5]) or "no cell outside noise"
    e0, e1 = dch["epoch_prev"], dch["epoch_cur"]
    status = "" if (cur.get("status") or "complete") == "complete" else f" ({cur.get('status')} card)"
    return (f"[improvement-loop] axis: {answer[0]} — reign {k.get('reign')} vs {pk.get('reign')}: {answer[1]}; "
            f"{mv}; D epoch {e0['epoch'] if e0 else '?'}→{e1['epoch'] if e1 else '?'} between submissions; "
            f"run `{cur['run_id']}`{status}")


def post_discord(text: str, dry_run: bool) -> bool:
    if dry_run:
        log(f"DRY-RUN discord: {text}")
        return True
    token = env_file_value(DISCORD_TOKEN_ENV)
    if not token:
        log("no discord token; not posting")
        return False
    try:
        r = requests.post(f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL}/messages",
                          headers={"Authorization": f"Bot {token}"}, json={"content": text[:1900]}, timeout=20)
        if r.status_code >= 300:
            log(f"discord HTTP {r.status_code}: {r.text[:200]}")
            return False
        return True
    except requests.RequestException as e:
        log(f"discord post failed: {e!r}")
        return False


# --------------------------------------------------------------------- main

def attribute(run_id: str, against: str | None, out_dir: Path, discord: bool, dry_run: bool) -> dict:
    cards = load_cards()
    cur = card_by_run(cards, run_id)
    if not cur:
        raise SystemExit(f"no scorecard for run {run_id} under {CARDS_DIR}")
    prev = card_by_run(cards, against) if against else pick_previous(cards, cur)
    if not prev:
        raise SystemExit(f"no previous reign's scorecard to compare {run_id} against")
    cache_dir = out_dir / "cache"
    cells = compare_cards(cur, prev, cache_dir)
    anchors = block_anchors()
    subs = {"cur": king_submission(cur["king"]["digest"], anchors),
            "prev": king_submission(prev["king"]["digest"], anchors)}
    dch = d_changes(subs["prev"].get("submitted_at"), subs["cur"].get("submitted_at"), cache_dir)
    reviews = {d: king_review_summary(d) for d in (cur["king"]["digest"], prev["king"]["digest"])}
    answer = axis_answer(cells)
    body, one = render(cur, prev, cells, subs, dch, reviews, answer)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{run_id}.md").write_text(body)
    result = {"run_id": run_id, "against": prev["run_id"], "generated_at": iso(time.time()),
              "king": cur["king"], "previous_king": prev["king"], "answer": answer[0], "why": answer[1],
              "one_liner": one, "cells": cells, "submissions": subs,
              "d": {kk: v for kk, v in dch.items() if kk not in ("comp_prev", "comp_cur")},
              "d_groups_prev": (dch["comp_prev"] or {}).get("group_strata"),
              "d_groups_cur": (dch["comp_cur"] or {}).get("group_strata"),
              "reviews": reviews, "card_status": cur.get("status")}
    (out_dir / f"{run_id}.json").write_text(json.dumps(result, indent=1, default=str))
    (out_dir / "latest.json").write_text(json.dumps(result, indent=1, default=str))
    with open(out_dir / "axis.jsonl", "a") as f:
        f.write(json.dumps({"at": result["generated_at"], "run_id": run_id, "against": prev["run_id"],
                            "reign": cur["king"].get("reign"), "answer": answer[0], "why": answer[1],
                            "moved": [(c["cell"], round(100 * c["delta"], 1)) for c in cells if c["verdict"] != "flat"]}) + "\n")
    log(one)
    if discord:
        post_discord(one, dry_run)
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="scorecard run_id to attribute")
    ap.add_argument("--latest", action="store_true", help="newest complete scorecard")
    ap.add_argument("--against", help="previous run_id (default: newest card of the previous reign)")
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--discord", action="store_true", help="post the one-line axis readout (private channel)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run_id = a.run
    if a.latest:
        cards = [c for c in load_cards() if c.get("status") == "complete"] or load_cards()
        if not cards:
            raise SystemExit("no scorecards")
        run_id = cards[-1]["run_id"]
    if not run_id:
        ap.error("--run or --latest")
    r = attribute(run_id, a.against, Path(a.out), a.discord, a.dry_run)
    print(json.dumps({k: r[k] for k in ("run_id", "against", "answer", "why")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
