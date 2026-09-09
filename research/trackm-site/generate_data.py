#!/usr/bin/env python3
"""Build the per-track JSON datasets for the multi-track dashboard.

Track M: parses LOG_PATH (`ISO8601Z [source] key=value ...` lines) when it
exists; falls back to a mock dataset otherwise -> data.json.
Track S: parses S_LOG_PATH (self-play control)      -> dataS.json.
Track T: parses T_LOG_PATH (pure-GAN run; the log may not exist yet —
missing file means an honest "starting up" empty state) -> dataT.json.

A refresh loop (see refresh_data.sh) reruns this every 2 minutes.

Usage: python3 generate_data.py [-o data.json]
"""

import json
import math
import random
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

LOG_PATH = Path("/home/const/subnet120/research/logs/trackM_status.log")
S_LOG_PATH = Path("/home/const/subnet120/research/logs/trackS_status.log")
T_LOG_PATH = Path("/home/const/subnet120/research/logs/trackT_status.log")
HERE = Path(__file__).resolve().parent

TEACHER = "Qwen3.8-27B"
MINER_BASE = "Qwen3.6-35B-A3B"
SWE_BASELINE = 13.33  # untrained miner base
SWE_TEACHER = 31.33   # teacher reference


# ---------------------------------------------------------------- real log

# `2026-08-22T14:04:25Z [eval] batch=3 reign=0 dver=v0 king_fool=0.3629 ...`
LINE_RE = re.compile(r"^(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+\[(\w+)\]\s+(.*)$")
# Tolerates odd keys like `rewards<0.01=0.03`.
KV_RE = re.compile(r"([\w<>.]+)=(\S+)")
JUDGE_GATE_RE = re.compile(r"JUDGE (\S+) gate:")
PUBLISH_RE = re.compile(r"PUBLISH (\S+):")
DVER_RE = re.compile(r"dver=(\w+)")
DVER_RENAME_RE = re.compile(r"dver=(\w+)\s*->\s*(\w+)")


def _val(s):
    """'-' -> None; numeric strings -> int/float; else keep the string."""
    if s in ("-", "—", "none", "None"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _kv(rest):
    return {k: _val(v) for k, v in KV_RE.findall(rest)}


def parse_status_log(path: Path):
    """Fold the run log into the dataset the site renders.
    Returns None when the log is missing or has no event lines yet."""
    if not path.exists():
        return None

    crown_rule = None
    batches = []        # [eval] batch=N ...
    rounds = []         # [miner] round=N ...
    ratchets = []       # [eval] RATCHET ...
    judges = []         # gate / zero-shot judge measurements
    publishes = []      # (ts, version)
    crowns = []         # future CROWN lines (format TBD — kept raw)
    swes = []           # future SWE lines
    driver = {}         # last "eval driver up" kv (reign, dver, king, eps, N)
    quarantined_dvers = set()
    q_stop_ts = None
    q_done_ts = None
    state = None        # (ts, evals_state)
    last_ts = None

    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            # Untimestamped [trackM] header/meta lines.
            if "crown rule:" in line:
                crown_rule = line.split("crown rule:", 1)[1].strip()
            continue
        ts, src, rest = m.groups()
        last_ts = ts

        if src == "eval":
            if rest.startswith("batch="):
                b = _kv(rest)
                b["t"] = ts
                batches.append(b)
                state = (ts, "running")
            elif rest.startswith("RATCHET"):
                r = _kv(rest)
                r["t"] = ts
                ratchets.append(r)
            elif rest.startswith("JUDGE zero-shot"):
                j = _kv(rest)
                judges.append({"version": "base (0-shot)", "at": ts,
                               "held_acc": j.get("held_acc"),
                               "zero_shot": True, "dver": None})
            elif JUDGE_GATE_RE.match(rest):
                ver = JUDGE_GATE_RE.match(rest).group(1)
                j = _kv(rest)
                judges.append({"version": ver, "at": ts,
                               "held_acc": j.get("held_acc"),
                               "train_loss": j.get("train_loss"),
                               "matched": j.get("matched"),
                               "zero_shot": False, "dver": ver})
            elif PUBLISH_RE.match(rest):
                publishes.append((ts, PUBLISH_RE.match(rest).group(1)))
                state = (ts, "running")
            elif "training judge" in rest:
                state = (ts, "paused — training judge")
            elif rest.startswith("QUARANTINE done"):
                q_done_ts = ts
                for a, b2 in DVER_RENAME_RE.findall(rest):
                    quarantined_dvers.update((a, b2))
            elif rest.startswith("QUARANTINE"):
                q_stop_ts = ts
                quarantined_dvers.update(DVER_RE.findall(rest))
                state = (ts, "paused — quarantine")
            elif "eval driver up" in rest:
                driver = _kv(rest)
            elif "CROWN" in rest:
                c = _kv(rest)
                c["t"] = ts
                crowns.append(c)
            elif "SWE" in rest.upper() and "score" in _kv(rest):
                s = _kv(rest)
                s["t"] = ts
                swes.append(s)
        elif src == "miner":
            if rest.startswith("round="):
                r = _kv(rest)
                r["t"] = ts
                rounds.append(r)
            elif rest.startswith("QUARANTINE"):
                quarantined_dvers.update(DVER_RE.findall(rest))

    if not (batches or rounds or ratchets or judges or publishes):
        return None

    # The base zero-shot reference gets re-measured before each retrain;
    # keep only the latest so the judge chart shows one "base" point.
    last_zero = next((j for j in reversed(judges) if j["zero_shot"]), None)
    judges = [j for j in judges if not j["zero_shot"]]
    if last_zero:
        judges.insert(0, last_zero)

    def quarantined(dver):
        return dver in quarantined_dvers

    # Quarantine window for the UI: publish of the bad judge -> QUARANTINE done.
    quarantine = None
    if quarantined_dvers and q_stop_ts:
        bad_pub = next((t for t, v in publishes if v in quarantined_dvers), None)
        quarantine = {
            "start": bad_pub or (batches[0]["t"] if batches else q_stop_ts),
            "end": q_done_ts or q_stop_ts,
            "dvers": sorted(quarantined_dvers),
            "note": "judge v0 was a serving no-op (LoRA key mismatch) — "
                    "scores in this window were base-model noise",
        }

    cur_pub_ts, cur_dver = publishes[-1] if publishes else (None, driver.get("dver"))
    cur_judge = next((j for j in reversed(judges)
                      if j["dver"] == cur_dver and not j["zero_shot"]), None)

    live_batches = [b for b in batches
                    if not quarantined(b.get("dver")) and b.get("king_fool") is not None]
    last_live = live_batches[-1] if live_batches else None
    last_ch = next((b for b in reversed(live_batches)
                    if b.get("ch_fool") is not None), None)

    reign = driver.get("reign", batches[-1].get("reign") if batches else 0) or 0
    king = driver.get("king") or (ratchets[-1].get("king") if ratchets else None) \
        or f"Qwen/{MINER_BASE}"

    live_rounds = [r for r in rounds if not quarantined(r.get("dver"))]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": "live",
        "log_end": last_ts,
        "run": "Track M",
        "teacher": TEACHER,
        "miner_base": MINER_BASE,
        "margin_pp": driver.get("eps", 0.03) or 0.03,
        "min_turns": driver.get("N", 400) or 400,
        "crown_rule": crown_rule,
        "swe_baseline": SWE_BASELINE,
        "swe_teacher": SWE_TEACHER,
        "quarantine": quarantine,
        "current": {
            "reign": reign,
            "king": king,
            "judge_version": cur_dver,
            "judge_acc": cur_judge["held_acc"] if cur_judge else None,
            "king_fool_rate": last_live.get("king_fool") if last_live else None,
            "challenger": None,  # no submissions duelling yet; ch_fool drives display
            "challenger_fool_rate": last_ch.get("ch_fool") if last_ch else None,
            "reign_started_at": cur_pub_ts,
            "paired_turns": (last_live or {}).get("duel_n", 0) or 0,
            "evals_state": state[1] if state else "starting",
            "miner_round": (live_rounds or rounds or [{}])[-1].get("round"),
            "miner_fool_local": (live_rounds[-1].get("fool_local")
                                 if live_rounds else None),
        },
        "reign_series": [{
            "t": b["t"], "batch": b.get("batch"),
            "king": b.get("king_fool"), "challenger": b.get("ch_fool"),
            "quarantined": quarantined(b.get("dver")),
        } for b in batches],
        "miner_series": [{
            "t": r["t"], "round": r.get("round"),
            "fool_local": r.get("fool_local"), "best_fool": r.get("best_fool"),
            "quarantined": quarantined(r.get("dver")),
        } for r in rounds],
        "ratchet": [{
            "t": r["t"], "reign": r.get("reign"), "dver": r.get("dver"),
            "value": r.get("king_fool_fresh_judge"), "n": r.get("n"),
            "quarantined": quarantined(r.get("dver")),
        } for r in ratchets],
        "judges": [{**j, "quarantined": quarantined(j["dver"])} for j in judges],
        "crowns": crowns,
        "swe": swes,
    }


# ------------------------------------------------------------- track S / T

WALL_RE = re.compile(r"wall=(\d+(?:\.\d+)?)s")
EFF_RE = re.compile(r"\(eff=([-\d.eE]+)\)")
PTRIPLE_RE = re.compile(r"p\([^)]*\)=([\d.]+)/([\d.]+)/([\d.]+)")
SCORE_FRAC_RE = re.compile(r"score=(\d+)/(\d+)")
RESOLVED_RE = re.compile(r"resolved=(\d+)/(\d+)")

SWE_TEACHER_FRAC = 0.3133  # teacher full-panel SWE score, as a fraction


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _num(v):
    return v if isinstance(v, (int, float)) else None


def _wall(rest):
    m = WALL_RE.search(rest)
    return float(m.group(1)) if m else None


def _push_event(events, ts, src, msg):
    """Run-log entry; consecutive identical messages collapse into one row."""
    if events and events[-1]["src"] == src and events[-1]["msg"] == msg:
        events[-1]["count"] += 1
        events[-1]["t_last"] = ts
        return
    low = msg.lower()
    if "fatal" in low or "failed" in low:
        sev = "bad"
    elif "stalled" in low or "watchdog" in low or "rejected" in low:
        sev = "warn"
    else:
        sev = "info"
    events.append({"t": ts, "t_last": ts, "src": src, "msg": msg,
                   "count": 1, "sev": sev})


def parse_trackS(path: Path):
    """Self-play control (the recursive check). Never raises: unknown lines
    land in the run log, a missing file yields an empty-state dataset."""
    data = {
        "generated_at": _now_iso(), "track": "S", "source": "missing",
        "teacher": TEACHER, "header": [],
        "log_start": None, "log_end": None,
        "rounds": [], "d_updates": [], "events": [],
        "verdict": {"label": "PENDING", "mean_fool5": None, "n": 0,
                    "reason": "no rounds logged yet"},
        "current": {},
    }
    if not path.exists():
        return data

    rounds, dups, events = [], [], []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            if line.startswith("[trackS]"):
                data["header"].append(line[len("[trackS]"):].strip())
            continue
        ts, src, rest = m.groups()
        data["log_start"] = data["log_start"] or ts
        data["log_end"] = ts
        try:
            if src == "loop" and rest.startswith("round=") and " fool=" in rest:
                kv = _kv(rest)
                em = EFF_RE.search(rest)
                g = kv.get("g")
                rounds.append({
                    "t": ts, "round": kv.get("round"),
                    "fool": _num(kv.get("fool")),
                    "fool_best": _num(kv.get("fool_best")),
                    "fool_win": _num(kv.get("fool_win")),
                    "valid": _num(kv.get("valid")),
                    "t_valid": _num(kv.get("t_valid")),
                    "agree_exact": _num(kv.get("agree_exact")),
                    "agree_tok": _num(kv.get("agree_tok")),
                    "tt_exact": _num(kv.get("tt_exact")),
                    "tt_tok": _num(kv.get("tt_tok")),
                    "g_self_tok": _num(kv.get("g_self_tok")),
                    "typ_lp": _num(kv.get("typ_lp")),
                    "think_len": kv.get("think_len"),
                    "winners": kv.get("winners"),
                    "g": g if isinstance(g, str) else None,
                    "eff": float(em.group(1)) if em else None,
                    "dver": kv.get("dver"), "errs": kv.get("errs"),
                    "wall_s": _wall(rest),
                })
            elif src == "loop" and rest.startswith("D UPDATE"):
                kv = _kv(rest)
                pm = PTRIPLE_RE.search(rest)
                dups.append({
                    "t": ts, "dver": kv.get("dver"),
                    "held_live": _num(kv.get("held_live")),
                    "held_ctrl": _num(kv.get("held_ctrl")),
                    "held_tt": _num(kv.get("held_tt")),
                    "p_live": float(pm.group(1)) if pm else None,
                    "p_ctrl": float(pm.group(2)) if pm else None,
                    "p_tt": float(pm.group(3)) if pm else None,
                    "n": kv.get("n"), "train_acc": _num(kv.get("train_acc")),
                    "pairs": kv.get("pairs"),
                    "probe_delta": _num(kv.get("probe_delta")),
                    "wall_s": _wall(rest),
                })
            else:
                _push_event(events, ts, src, rest)
        except Exception:
            _push_event(events, ts, src, rest)

    data["rounds"] = rounds
    data["d_updates"] = dups
    data["events"] = events[-120:]
    data["source"] = "live" if (rounds or dups) else "starting"

    # Verdict from recent data (readout guide in the log header).
    fools = [r["fool"] for r in rounds if r["fool"] is not None]
    recent = fools[-5:]
    mean5 = round(sum(recent) / len(recent), 4) if recent else None
    lastd = dups[-2:]
    leak = bool(lastd) and all(
        max(u["held_ctrl"] or 0.0, u["held_tt"] or 0.0) > 0.58 for u in lastd)
    if leak:
        v = ("LEAK", "held_ctrl / held_tt sustained above 0.58 — "
                     "pipeline artifact, not a real signal")
    elif mean5 is None:
        v = ("PENDING", "no scored rounds yet")
    elif 0.45 <= mean5 <= 0.55:
        v = ("STABLE", f"mean fool over last {len(recent)} rounds = {mean5} "
                       f"— inside the 0.45–0.55 equilibrium band")
    else:
        v = ("DRIFT", f"mean fool over last {len(recent)} rounds = {mean5} "
                      f"— outside the 0.45–0.55 equilibrium band")
    data["verdict"] = {"label": v[0], "reason": v[1],
                       "mean_fool5": mean5, "n": len(recent)}

    r = rounds[-1] if rounds else {}
    u = dups[-1] if dups else {}
    walls = [x["wall_s"] for x in rounds[-5:] if x.get("wall_s")]
    data["current"] = {
        "round": r.get("round"), "fool": r.get("fool"),
        "typ_lp": r.get("typ_lp"), "think_len": r.get("think_len"),
        "g": r.get("g"), "eff": r.get("eff"),
        "dver": r.get("dver") or u.get("dver"),
        "held_live": u.get("held_live"), "held_ctrl": u.get("held_ctrl"),
        "held_tt": u.get("held_tt"), "d_updated_at": u.get("t"),
        "avg_wall_s": round(sum(walls) / len(walls)) if walls else None,
    }
    return data


def parse_trackT(path: Path):
    """Pure-GAN run. The log may not exist yet — that is the honest
    'experiment starting up' state, not an error."""
    data = {
        "generated_at": _now_iso(), "track": "T", "source": "missing",
        "teacher": TEACHER, "header": [],
        "log_start": None, "log_end": None,
        "swe_teacher": SWE_TEACHER_FRAC, "swe_student_baseline": None,
        "rounds": [], "judges": [], "bench_proxy": [], "bench_full": [],
        "events": [], "banner": {},
    }
    if not path.exists():
        return data

    rounds, judges, proxies, fulls, events = [], [], [], [], []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            if line.startswith("[trackT]"):
                data["header"].append(line[len("[trackT]"):].strip())
            continue
        ts, src, rest = m.groups()
        data["log_start"] = data["log_start"] or ts
        data["log_end"] = ts
        try:
            if src == "loop" and rest.startswith("round=") and "fool=" in rest:
                kv = _kv(rest)
                rounds.append({
                    "t": ts, "round": kv.get("round"),
                    "fool": _num(kv.get("fool")),
                    "valid": _num(kv.get("valid")),
                    "dver": kv.get("dver"), "wall_s": _wall(rest),
                })
            elif src == "judge" and "UPDATE" in rest:
                kv = _kv(rest)
                judges.append({
                    "t": ts, "dver": kv.get("dver"),
                    "held_acc": _num(kv.get("held_acc")),
                    "train_acc": _num(kv.get("train_acc")),
                    "pairs": kv.get("pairs"), "wall_s": _wall(rest),
                })
                _push_event(events, ts, src, rest)
            elif src == "bench" and rest.startswith("PROXY"):
                kv = _kv(rest)
                sm = SCORE_FRAC_RE.search(rest)
                proxies.append({
                    "t": ts, "round": kv.get("round"),
                    "num": int(sm.group(1)) if sm else None,
                    "den": int(sm.group(2)) if sm else None,
                    "frac": (int(sm.group(1)) / int(sm.group(2)))
                            if sm and int(sm.group(2)) else None,
                    "gpu_hours": _num(kv.get("gpu_hours")),
                })
                _push_event(events, ts, src, rest)
            elif src == "bench" and rest.startswith("FULL"):
                kv = _kv(rest)
                rm = RESOLVED_RE.search(rest)
                ckpt = str(kv.get("ckpt") or "")
                is_base = bool(re.search(r"base|raw|student|untrained",
                                         ckpt, re.I))
                entry = {
                    "t": ts, "ckpt": ckpt or None,
                    "score": _num(kv.get("score")),
                    "resolved": int(rm.group(1)) if rm else None,
                    "panel": int(rm.group(2)) if rm else None,
                    "p_vs_baseline": _num(kv.get("p_vs_baseline")),
                    "gpu_hours": _num(kv.get("gpu_hours")),
                    "baseline": is_base,
                }
                if is_base and entry["score"] is not None:
                    data["swe_student_baseline"] = entry["score"]
                fulls.append(entry)
                _push_event(events, ts, src, rest)
            else:
                _push_event(events, ts, src, rest)
        except Exception:
            _push_event(events, ts, src, rest)

    data["rounds"] = rounds
    data["judges"] = judges
    data["bench_proxy"] = proxies
    data["bench_full"] = fulls
    data["events"] = events[-120:]
    data["source"] = ("live" if (rounds or judges or proxies or fulls)
                      else "starting")

    gpu = [x["gpu_hours"] for x in proxies + fulls if x.get("gpu_hours")]
    r = rounds[-1] if rounds else {}
    data["banner"] = {
        "rounds": r.get("round") if r else None,
        "started_at": data["log_start"],
        "judge_version": (judges[-1]["dver"] if judges else r.get("dver")),
        "gpu_hours": max(gpu) if gpu else None,
        "last_fool": r.get("fool"),
        "benches": {"proxy": len(proxies), "full": len(fulls)},
    }
    return data


# ---------------------------------------------------------------- mock run

def mock_dataset(seed: int = 120):
    """Fallback when the log is missing — same shape as the live dataset."""
    rng = random.Random(seed)
    now = datetime(2026, 8, 22, 11, 40, tzinfo=timezone.utc)
    t0 = now - timedelta(days=13, hours=7)

    ratchet_vals = [0.081, 0.117, 0.146, 0.183, 0.221]
    judge_acc = [0.934, 0.921, 0.914, 0.902, 0.893]
    swe = [13.6, 13.9, 14.4, 14.8, 15.2]

    crowns, ratchet, judges, swes = [], [], [], []
    t = t0
    for i in range(5):
        r = i + 1
        king = f"trackm-r{r}-{rng.randbytes(3).hex()}"
        iso = t.isoformat()
        crowns.append({"t": iso, "reign": r, "king": king,
                       "margin": round(rng.uniform(0.031, 0.052), 3),
                       "dver": f"D-{r}",
                       "retrain_hours": round(rng.uniform(4.2, 7.8), 1),
                       "judge_acc": judge_acc[i], "n": rng.randint(410, 780)})
        ratchet.append({"t": iso, "reign": r, "dver": f"D-{r}",
                        "value": ratchet_vals[i], "n": 400, "quarantined": False})
        judges.append({"version": f"D-{r}", "at": iso, "held_acc": judge_acc[i],
                       "zero_shot": False, "dver": f"D-{r}", "quarantined": False})
        swes.append({"t": iso, "reign": r, "score": swe[i]})
        t += timedelta(days=rng.uniform(2.0, 3.5))

    start = crowns[-1]["t"]
    series = []
    ts = datetime.fromisoformat(start)
    for turn in range(0, 1241, 20):
        p = turn / 2400
        k = ratchet_vals[-1] + 0.028 * (1 - math.exp(-3.0 * p)) + rng.gauss(0, 0.006)
        c = 0.164 + 0.085 * (1 - math.exp(-2.2 * p)) + rng.gauss(0, 0.008)
        ts += timedelta(minutes=2.4)
        series.append({"t": ts.isoformat(), "batch": turn // 20,
                       "king": round(max(0.0, k), 4),
                       "challenger": round(max(0.0, c), 4), "quarantined": False})

    return {
        "generated_at": now.isoformat(),
        "source": "mock",
        "log_end": now.isoformat(),
        "run": "Track M",
        "teacher": TEACHER,
        "miner_base": MINER_BASE,
        "margin_pp": 0.03,
        "min_turns": 400,
        "crown_rule": None,
        "swe_baseline": SWE_BASELINE,
        "swe_teacher": SWE_TEACHER,
        "quarantine": None,
        "current": {
            "reign": 5,
            "king": crowns[-1]["king"],
            "judge_version": "D-5",
            "judge_acc": judge_acc[-1],
            "king_fool_rate": series[-1]["king"],
            "challenger": f"trackm-ch-{rng.randbytes(3).hex()}",
            "challenger_fool_rate": series[-1]["challenger"],
            "reign_started_at": start,
            "paired_turns": 1240,
            "evals_state": "running",
            "miner_round": None,
            "miner_fool_local": None,
        },
        "reign_series": series,
        "miner_series": [],
        "ratchet": ratchet,
        "judges": judges,
        "crowns": crowns,
        "swe": swes,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(HERE / "data.json"))
    args = ap.parse_args()

    out = Path(args.out)
    data = parse_status_log(LOG_PATH)
    if data is None:
        data = mock_dataset()
    out.write_text(json.dumps(data, indent=1))

    site = out.parent
    data_s = parse_trackS(S_LOG_PATH)
    (site / "dataS.json").write_text(json.dumps(data_s, indent=1))
    data_t = parse_trackT(T_LOG_PATH)
    (site / "dataT.json").write_text(json.dumps(data_t, indent=1))
    print(f"wrote {out} (M={data['source']} S={data_s['source']} "
          f"T={data_t['source']})")


if __name__ == "__main__":
    main()
