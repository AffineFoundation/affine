"""Post-T0 green watch for the wvk-25 cutover (GLM-5.3-Flash + 262k + scoring bundle).

Runs 6-hourly (pm2 cron on the box) and once right after the first wvk-25
verdict. Prints a <= 8-line report and writes ops/v19/green_watch.jsonl.

  --public   only what is reachable off the box: affine.io contract / history /
             llms.txt, Hippius verdict artifacts (control per leg, forfeits,
             sequential stamps, duel minutes, rollback trigger)
  --box      adds the on-box JSON the datagen / fold workers write:
             ops/teacher-swarm/state/watch.json  (replicas, echo latency, OOMs, KV)
             rollouts/state/watch.json           (GLM seat yield vs Qwen baseline, upstream_fetch drops)
             ops/state/fold_watch.jsonl          (clean folds, floors, unknown-source gate)
             ops/state/pipeline_health_pages.jsonl (one line per page; 48 h quiet = criterion)

Exit criteria (Jacob 2026-09-26 09:10 UTC), all six must hold for "GREEN":
  1. >= 10 post-fork verdicts with control_matched positive on every leg, no rollback trigger
  2. forfeits < 1 % per side (median over post-fork verdicts)
  3. sequential-stop agreement with the full slice >= 9/10 (first 10: 5 shadow full slices + 5 replays)
  4. 3 consecutive clean folds with the re-baked D
  5. GLM teacher-seat yield >= 80 % of the Qwen baseline, per source
  6. 48 h without a pipeline-health page
Rollback trigger = control_matched sign flip on typicality or A (fork worker's rule).
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import statistics as st
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
SITE = "https://affine.io"
EVALS = "https://s3.hippius.com/affine-sn120/evals"
OUT = REPO / "ops" / "v19" / "green_watch.jsonl"
CACHE = Path("/tmp/green_watch/evals")
UA = {"User-Agent": "affine-green-watch/1.0"}
WVK = 25
TEACHER = "zai-org/GLM-5.3-Flash"
WINDOW = 262144
# pre-fork GLM shadow (internal/teacher-swap/shadow_glm53_report_glm-5.3-flash_v2.txt, k-matched floor-dropped)
SHADOW_CONTROL = {"all": 0.80, "R": 0.04, "Gc": 1.68, "A": 0.50}


def get_json(url: str, timeout: float = 30):
    r = httpx.get(url, headers=UA, timeout=timeout, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def parse(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except Exception:
            return None
    return x


def history(pages: int = 4) -> list[dict]:
    items, cur = [], None
    for _ in range(pages):
        d = get_json(f"{SITE}/api/v1/history" + (f"?cursor={cur}" if cur else ""))
        items += d["items"]
        cur = d.get("next_cursor")
        if not cur:
            break
    return items


def artifact(chal: str) -> dict | None:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{chal}.json.gz"
    if not p.exists():
        r = httpx.get(f"{EVALS}/{chal}.json.gz", headers=UA, timeout=300, follow_redirects=True)
        if r.status_code != 200 or not r.content[:2] == b"\x1f\x8b":
            return None
        p.write_bytes(r.content)
    return json.load(gzip.open(p))


def post_fork_verdicts(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        if it.get("event") != "verdict":
            continue
        dp = parse(it.get("duel_params")) or {}
        if int(dp.get("weight_version_key") or 0) < WVK and not str(parse(it.get("teacher")) or "").count(TEACHER):
            # duel_params may not carry wvk; fall back to the teacher stamp on the artifact below
            pass
        out.append(it)
    return out


def check_verdicts() -> dict:
    items = history()
    verdicts = [i for i in items if i.get("event") == "verdict"]
    post = []
    for it in verdicts[:40]:
        art = artifact(it["challenge_id"])
        if not art:
            continue
        v = art.get("verdict") or {}
        dp = v.get("duel_params") or {}
        teacher = (v.get("teacher") or {}).get("repo") or (art.get("request") or {}).get("teacher_repo") or ""
        wvk = int(dp.get("weight_version_key") or 0)
        if wvk < WVK and TEACHER not in str(teacher):
            continue
        sd = ((v.get("shadow") or {}).get("sd_meter") or {})
        loo = (sd.get("by_anchor") or {}).get("loo") or {}
        ctrl = loo.get("control_kmatched") or {}
        seq = v.get("sequential") or dp.get("sequential") or {}
        n_k = (v.get("king") or {}).get("n_turns") or 1
        n_c = (v.get("challenger") or {}).get("n_turns") or 1
        post.append({
            "chal": it["challenge_id"], "at": it.get("at"), "z": v.get("z"), "margin": v.get("margin"),
            "wins": v.get("challenger_wins"), "wvk": wvk, "teacher": teacher,
            "control": {leg: (ctrl.get(leg) or {}).get("margin") for leg in ("all", "R", "Gc", "A")},
            "forfeit_rate": {"king": ((v.get("king") or {}).get("n_forfeits") or 0) / n_k,
                             "challenger": ((v.get("challenger") or {}).get("n_forfeits") or 0) / n_c},
            "empty_gate": {"king": (v.get("king") or {}).get("empty_gate_applied"),
                           "challenger": (v.get("challenger") or {}).get("empty_gate_applied")},
            "seq": {"stop_reason": seq.get("stop_reason") or v.get("stop_reason"),
                    "n_turns_scored": seq.get("n_turns_scored") or v.get("n_turns_scored"),
                    "n_looks": seq.get("n_looks") or v.get("n_looks"),
                    "full_slice_crown": ((v.get("shadow") or {}).get("full_slice") or {}).get("crown")},
            "duel_min": (v.get("duel_seconds") or 0) / 60.0,
            "max_model_len": dp.get("max_model_len"),
        })
    n = len(post)
    all_pos = [p for p in post if all((p["control"].get(l) or 0) > 0 for l in ("R", "Gc", "A"))]
    rollback = [p for p in post if (p["control"].get("Gc") or 0) < 0 or (p["control"].get("A") or 0) < 0]
    forf = [max(p["forfeit_rate"].values()) for p in post]
    agree = [(p["wins"] == p["seq"]["full_slice_crown"]) for p in post if p["seq"]["full_slice_crown"] is not None]
    ttd = [p["seq"]["n_turns_scored"] for p in post if p["seq"]["n_turns_scored"]]
    ctrl_mean = {leg: st.mean([p["control"][leg] for p in post if p["control"].get(leg) is not None])
                 for leg in ("all", "R", "Gc", "A") if any(p["control"].get(leg) is not None for p in post)}
    return {
        "n_post_fork": n, "n_control_all_positive": len(all_pos), "rollback_triggers": [p["chal"] for p in rollback],
        "control_mean": ctrl_mean, "control_vs_shadow": {k: (ctrl_mean.get(k), SHADOW_CONTROL[k]) for k in SHADOW_CONTROL},
        "forfeit_median": (st.median(forf) if forf else None), "forfeit_max": (max(forf) if forf else None),
        "seq_agreement": f"{sum(agree)}/{len(agree)}" if agree else None, "seq_agree_ok": (len(agree) >= 10 and sum(agree) >= 9),
        "turns_to_decision_p50": (st.median(ttd) if ttd else None),
        "duel_min_p50": (st.median([p["duel_min"] for p in post]) if post else None),
        "empty_gate_hits": sum(1 for p in post for s in p["empty_gate"].values() if s),
        "latest": post[:3],
    }


def check_site() -> dict:
    c = get_json(f"{SITE}/api/v1/contract")
    llms = httpx.get(f"{SITE}/llms.txt", headers=UA, timeout=30).text
    html = httpx.get(f"{SITE}/", headers=UA, timeout=30).text
    return {
        "contract_wvk": c["subnet"]["weight_version_key"], "contract_teacher": c["teacher"]["repo"],
        "contract_ok": c["subnet"]["weight_version_key"] == WVK and c["teacher"]["repo"] == TEACHER,
        "llms_fork_history": f"Fork history: wvk {WVK}" in llms,
        "llms_upcoming_removed": "Upcoming fork: wvk 25" not in llms and "Upcoming fork wvk 25" not in llms,
        "banner_removed": "wvk 25" not in (html.split('id="fork-notice"')[1][:600] if 'id="fork-notice"' in html else ""),
    }


def read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def check_box() -> dict:
    swarm = read_json(REPO / "ops/teacher-swarm/state/watch.json")
    dg = read_json(REPO / "rollouts/state/watch.json")
    folds, pages = [], []
    fp = REPO / "ops/state/fold_watch.jsonl"
    if fp.exists():
        folds = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
    pp = REPO / "ops/state/pipeline_health_pages.jsonl"
    if pp.exists():
        pages = [json.loads(l) for l in pp.read_text().splitlines() if l.strip()]
    now = time.time()
    last_page = max((datetime.fromisoformat(p["at"].replace("Z", "+00:00")).timestamp() for p in pages if p.get("at")), default=None)
    clean_streak = 0
    for f in reversed(folds):
        if f.get("clean"):
            clean_streak += 1
        else:
            break
    yield_ok, yield_by = None, {}
    if dg:
        for src, r in dg.items():
            base = r.get("qwen_baseline_rollouts_24h") or 0
            got = r.get("glm_seat_rollouts_24h") or 0
            yield_by[src] = (got / base) if base else None
        vals = [v for v in yield_by.values() if v is not None]
        yield_ok = bool(vals) and min(vals) >= 0.8
    return {
        "swarm": swarm, "swarm_ok": bool(swarm) and swarm.get("replicas_healthy", 0) >= swarm.get("replicas_target", 1)
                                    and swarm.get("oom_restarts_24h", 1) == 0 and swarm.get("max_model_len") == WINDOW,
        "seat_yield_by_source": yield_by, "seat_yield_ok": yield_ok,
        "upstream_fetch_drops_24h": (sum((r.get("upstream_fetch_drops_24h") or 0) for r in dg.values()) if dg else None),
        "clean_fold_streak": clean_streak, "folds_ok": clean_streak >= 3,
        "hours_since_last_page": ((now - last_page) / 3600 if last_page else None),
        "pages_ok": (last_page is None and bool(pages) is False) or (last_page is not None and now - last_page >= 48 * 3600),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--public", action="store_true")
    ap.add_argument("--box", action="store_true")
    args = ap.parse_args()
    rep = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    try:
        rep["site"] = check_site()
    except Exception as e:
        rep["site"] = {"error": f"{type(e).__name__}: {e}"}
    try:
        rep["verdicts"] = check_verdicts()
    except Exception as e:
        rep["verdicts"] = {"error": f"{type(e).__name__}: {e}"}
    if args.box or not args.public:
        rep["box"] = check_box()
    v, s, b = rep.get("verdicts", {}), rep.get("site", {}), rep.get("box", {})
    crit = {
        "1 control+ (>=10, no rollback)": (v.get("n_control_all_positive", 0) >= 10 and not v.get("rollback_triggers")),
        "2 forfeits<1%": (v.get("forfeit_median") is not None and v["forfeit_median"] < 0.01),
        "3 seq agree>=9/10": bool(v.get("seq_agree_ok")),
        "4 three clean folds": bool(b.get("folds_ok")),
        "5 GLM seat yield>=80%": bool(b.get("seat_yield_ok")),
        "6 48h no health page": bool(b.get("pages_ok")),
    }
    rep["criteria"] = crit
    rep["green"] = all(crit.values()) and bool(s.get("contract_ok"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a") as f:
        f.write(json.dumps(rep, default=str) + "\n")
    cm = v.get("control_mean") or {}
    lines = [
        f"[{rep['at']}] {'GREEN' if rep['green'] else 'NOT GREEN'} — open: {', '.join(k for k, ok in crit.items() if not ok) or 'none'}",
        f"contract wvk {s.get('contract_wvk')} teacher {str(s.get('contract_teacher'))[-22:]} | llms fork-history {s.get('llms_fork_history')} upcoming-removed {s.get('llms_upcoming_removed')} banner-removed {s.get('banner_removed')}",
        f"verdicts post-fork {v.get('n_post_fork')} | control all-leg+ {v.get('n_control_all_positive')} | mean all {cm.get('all')} R {cm.get('R')} typ {cm.get('Gc')} A {cm.get('A')} (shadow 0.80/0.04/1.68/0.50) | rollback triggers {v.get('rollback_triggers')}",
        f"forfeits median {v.get('forfeit_median')} max {v.get('forfeit_max')} | empty-gate hits {v.get('empty_gate_hits')} | seq agreement {v.get('seq_agreement')} turns-to-decision p50 {v.get('turns_to_decision_p50')} | duel min p50 {v.get('duel_min_p50')}",
    ]
    if b:
        sw = b.get("swarm") or {}
        lines += [
            f"swarm replicas {sw.get('replicas_healthy')}/{sw.get('replicas_target')} echo p50/p95 {sw.get('echo_p50_s')}/{sw.get('echo_p95_s')} s OOM/restarts 24h {sw.get('oom_restarts_24h')} KV max {sw.get('kv_used_frac_max')} window {sw.get('max_model_len')}",
            f"datagen GLM seat yield/baseline by source {b.get('seat_yield_by_source')} | upstream_fetch drops 24h {b.get('upstream_fetch_drops_24h')}",
            f"fold clean streak {b.get('clean_fold_streak')} | hours since last health page {b.get('hours_since_last_page')}",
        ]
    print("\n".join(lines[:8]))


if __name__ == "__main__":
    main()
