#!/usr/bin/env python
"""Synthetic drills: one per failure class of 2026-09-17/19, each proves the
page fires. Runs entirely in a temp dir — no production file is read or
written, nothing is posted, nothing is restarted or released.

  python ops/health/drills.py            # all drills, table, exit 1 if any fails
  python ops/health/drills.py -k c       # only drills whose id contains "c"

  a  fold died on a KeyError and published nothing     -> fold_wrap page + fold_failed
  b  no fold for 23 h after the fork                    -> fold_overdue + corpus_epoch_age
  c  curriculum fitted in per-byte units, live sd units -> contract_mismatch + auto FREEZE,
                                                          preflight blocks a `block` consumer,
                                                          auto-unfreeze once re-declared
  d  king seat empty 25 h; kingctl running stale code   -> king_seat + code_restart
  e  bench queue silently budget-held                   -> bench_queue_held
  f  env boxes idle 40 h, driver dead                   -> reaper release_expired ($ saved) + pods
  g  fold ran on a stale sources.toml                   -> sources_toml_drift (+ recorded on the fold)
  x  in_flight stale vs history                         -> inflight_stale
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "ops" / "pods"))
sys.path.insert(0, str(REPO / "ops" / "fold"))
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))
import common  # noqa: E402
import contract_compat  # noqa: E402
import health  # noqa: E402

NOW = time.time()
H = 3600.0
CODE_WATCH_RULES = "[rules]\nsettle_min = 10\nmin_uptime_min = 10\nmin_restart_gap_min = 30\nmax_restarts_per_tick = 2\n"


class Capture:
    """Replaces common.discord_post; remembers every line."""

    def __init__(self):
        self.lines: list[str] = []

    def __call__(self, text, **kw):
        self.lines.append(f"{kw.get('prefix', '')} {text}".strip())
        return True


def write(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(obj, (dict, list)):
        path.write_text(json.dumps(obj, indent=1, default=str))
    else:
        path.write_text(str(obj))
    return path


def pm2_fixture(procs: dict[str, dict]) -> list[dict]:
    out = []
    for i, (name, spec) in enumerate(procs.items()):
        out.append({"name": name, "pid": 1000 + i, "pm2_env": {
            "status": spec.get("status", "online"), "restart_time": 0,
            "pm_uptime": int(spec.get("started_at", NOW - 2 * H) * 1000),
            "cron_restart": spec.get("cron"), "pm_exec_path": spec.get("script", "/x"),
            "args": spec.get("args", []), "pm_cwd": "/x"}})
    return out


class Sandbox:
    """A temp tree mirroring the paths health.toml points at, plus a Monitor
    wired to it with every external probe replaced by a fixture."""

    def __init__(self, root: Path):
        self.root = root
        self.state_dir = root / "affine/state"
        self.state_dir.mkdir(parents=True)
        self.procs = pm2_fixture({
            "affine-corpus-refresh": {"status": "stopped", "cron": "0 */6 * * *",
                                      "script": str(root / "ops/fold/run_fold.sh")},
            "affine-king-datagen": {}, "affine-coverage-bench": {}, "affine-pod-reaper": {},
            "affine-validator": {}, "affine-pipeline-health": {},
        })
        self.manifest = {"corpus_epoch": 51, "published_at": common.iso(NOW - 1 * H), "n_turns": 1, "n_strata": 1}
        self.router = {"reachable": True, "ok": True, "healthy": 16, "backends": 16}
        self.dirty = False
        # healthy defaults for every file
        write(self.state_dir / "state.json", {
            "king": {"revision": "f82deda2ffbd" + "0" * 52, "reign_number": 19},
            "queue": [], "in_flight": None, "last_weights_at": common.iso(NOW - 600)})
        (self.state_dir / "history.jsonl").write_text(
            json.dumps({"event": "verdict", "challenge_id": "chal-00610", "at": common.iso(NOW - 1800),
                        "verdict": {"duel_params": {"score_mode": "sd_min_rga"}}}) + "\n")
        write(self.state_dir / "fold/last_run.json", {
            "status": "done", "started_ts": NOW - 2 * H, "started_at": common.iso(NOW - 2 * H),
            "ended_at": common.iso(NOW - 2 * H + 600), "exit_code": 0, "classification": "ok",
            "attempts": 1, "duration_s": 600, "git_head": "abc1234", "sources_toml_sha256": "s1"})
        write(self.state_dir / "curriculum/latest.json", {
            "mode": "apply", "computed_at": common.iso(NOW - 3 * H), "for_epoch": 51,
            "input_units": "unit_free", "fitted_score_modes": ["min_rg", "sd_min_rga"]})
        write(root / "ops/king-datagen/state/state.json", {
            "published": {"pod": "king-dg-f82deda2ffbd-ef71", "at": NOW - H},
            "pods": {"king-dg-f82deda2ffbd-ef71": {"king": {"ident": "f82deda2ffbd"}, "last_ok": NOW - 60,
                                                   "ready_at": NOW - H}}})
        write(root / "ops/coverage/state/bench_queue_meta.json", {"active": True, "pods_seen": []})
        write(root / "ops/coverage/state/budget.json", {"cap_usd": 1400.0, "actuals": {"fill": 400.0}})
        write(self.state_dir / "pods/reaper.json", {"at": common.iso(NOW - 300), "ours": [], "n_ours": 0,
                                                   "ours_usd_h": 0, "foreign": {"count": 0, "usd_h": 0},
                                                   "problems": [], "actions": []})
        write(root / "rollouts/rollouts/sources.toml", "[mix]\n")
        write(root / "affine/affine.toml", '[duel]\nweight_version_key = 22\nscore_mode = "sd_min_rga"\n')
        write(root / "ops/health/code_watch.toml", CODE_WATCH_RULES)
        self.consumers_path = root / "ops/health/consumers.toml"
        self.consumers_path.write_text((HERE / "consumers.toml").read_text()
                                       .replace('toml = "affine/affine.toml"', f'toml = "{root}/affine/affine.toml"')
                                       .replace('history = "affine/state/history.jsonl"', f'history = "{self.state_dir}/history.jsonl"')
                                       .replace('declaration_file = "affine/state/curriculum/latest.json"',
                                                f'declaration_file = "{self.state_dir}/curriculum/latest.json"')
                                       .replace('freeze_file = "affine/state/curriculum/FROZEN.json"',
                                                f'freeze_file = "{self.state_dir}/curriculum/FROZEN.json"'))
        toml = (HERE / "health.toml").read_text()
        # the integrity hooks point at real repo files; not part of the drills
        toml = re.sub(r"\[\[integrity\]\].*?(?=\n\[discord\])", "", toml, flags=re.S)
        for key, rel in (("state_dir", "affine/state"), ("kingctl_state", "ops/king-datagen/state/state.json"),
                         ("fold_last_run", "affine/state/fold/last_run.json"),
                         ("curriculum_latest", "affine/state/curriculum/latest.json"),
                         ("curriculum_frozen", "affine/state/curriculum/FROZEN.json"),
                         ("sources_toml", "rollouts/rollouts/sources.toml"),
                         ("bench_queue_meta", "ops/coverage/state/bench_queue_meta.json"),
                         ("bench_budget", "ops/coverage/state/budget.json"),
                         ("reaper_report", "affine/state/pods/reaper.json"),
                         ("corpus_manifest_local", "affine/state/corpus_cache/current_manifest.json"),
                         ("out_json", "affine/state/pipeline_health.json"),
                         ("own_state", "ops/health/state/health_state.json"),
                         ("code_watch", "ops/health/code_watch.toml"), ("consumers", "ops/health/consumers.toml")):
            toml = re.sub(rf'^{key} = ".*"$', f'{key} = "{root / rel}"', toml, flags=re.M)
        self.health_toml = write(root / "ops/health/health.toml", toml)

    def monitor(self, *, since: dict[str, float] | None = None) -> tuple[health.Monitor, Capture]:
        cfg = health.Config(self.health_toml)
        # contract_compat resolves consumer paths against REPO; the sandbox
        # wrote absolute paths, so REPO / abs == abs
        mon = health.Monitor(cfg, dry_run=True)
        if since:
            mon.own["since"].update(since)
        cap = Capture()
        health.common.discord_post = cap
        health.common.pm2_jlist = lambda: self.procs
        health.fetch_manifest = lambda cfg: (self.manifest, "fixture")
        health.router_health = lambda cfg: dict(self.router)
        health.git_dirty = lambda path: self.dirty
        return mon, cap


def keys(out: dict) -> set[str]:
    return {k for k, v in out["checks"].items() if v["level"] == "page"}


# -- drills -----------------------------------------------------------------------

def drill_a(sb: Sandbox) -> tuple[bool, str]:
    """fold_wrap: fake corpus_build raises KeyError -> exit 1, traceback recorded, page; monitor pages fold_failed."""
    import fold_wrap
    fake = write(sb.root / "fake_corpus_build.py",
                 "import sys\nprint('fold starting')\nraw = {}\nprint(raw['raw'])\n")
    cap = Capture()
    fold_wrap.common.discord_post = cap
    fold_wrap.CORPUS_BUILD = fake
    fold_wrap.FOLD_DIR = sb.state_dir / "fold"
    fold_wrap.LAST_RUN = fold_wrap.FOLD_DIR / "last_run.json"
    fold_wrap.HISTORY = fold_wrap.FOLD_DIR / "history.jsonl"
    fold_wrap.RUNS_DIR = fold_wrap.FOLD_DIR / "runs"
    fold_wrap.SOURCES_TOML = sb.root / "rollouts/rollouts/sources.toml"
    fold_wrap.common.pm2_jlist = lambda: sb.procs
    sys.argv = ["fold_wrap.py", "--retry-wait-s", "0", "--"]
    rc = fold_wrap.main()
    rec = json.loads(fold_wrap.LAST_RUN.read_text())
    ok1 = rc != 0 and rec["exit_code"] == rc and rec["classification"] == "error" \
        and "KeyError: 'raw'" in (rec["traceback_tail"] or "") and rec["attempts"] == 1 \
        and any("FOLD FAILED" in l and "KeyError" in l for l in cap.lines)
    mon, cap2 = sb.monitor()
    out = mon.tick()
    fired = [k for k in keys(out) if k.startswith("fold_failed")]
    ok2 = bool(fired) and any("fold_failed" in l for l in cap2.lines)
    return ok1 and ok2, (f"fold_wrap exit {rc} cls={rec['classification']} attempts={rec['attempts']} "
                         f"page={'yes' if ok1 else 'NO'}; monitor {fired} page={'yes' if ok2 else 'NO'}")


def drill_b(sb: Sandbox) -> tuple[bool, str]:
    """No fold started for 23 h on a 6-h cron; manifest 23 h old."""
    write(sb.state_dir / "fold/last_run.json", {
        "status": "done", "started_ts": NOW - 23 * H, "started_at": common.iso(NOW - 23 * H),
        "exit_code": 0, "classification": "ok", "attempts": 1, "ended_at": common.iso(NOW - 23 * H + 600)})
    sb.manifest = {"corpus_epoch": 50, "published_at": common.iso(NOW - 23 * H)}
    mon, cap = sb.monitor()
    out = mon.tick()
    k = keys(out)
    ok = {"fold_overdue", "corpus_epoch_age"} <= k and any("fold_overdue" in l for l in cap.lines) \
        and any("corpus_epoch_age" in l for l in cap.lines)
    return ok, f"pages {sorted(k & {'fold_overdue', 'corpus_epoch_age'})}"


def drill_c(sb: Sandbox) -> tuple[bool, str]:
    """Curriculum vector without a units stamp (static per_byte) vs live sd_min_rga."""
    write(sb.state_dir / "curriculum/latest.json", {"mode": "apply", "computed_at": common.iso(NOW - 3 * H),
                                                     "for_epoch": 51, "theta": 7.1e-06})
    frozen = sb.state_dir / "curriculum/FROZEN.json"
    if frozen.exists():
        frozen.unlink()
    mon, cap = sb.monitor()
    mon.dry_run = False  # let enforce() write the sandbox FROZEN.json (discord is captured anyway)
    health.Monitor.restart = lambda self, name: None
    out = mon.tick()
    k = keys(out)
    ok1 = "contract_mismatch" in k and frozen.exists() and "units mismatch" in json.loads(frozen.read_text())["reason"] \
        and any("FROZE curriculum" in l for l in cap.lines)
    # preflight against a hypothetical wvk 23 mode with a `block` consumer that does not know it
    cfg = tomllib.loads(sb.consumers_path.read_text())
    cfg["units"]["sd_v2"] = "teacher_sd_v2"
    res = contract_compat.evaluate(cfg, target_mode="sd_v2", target_wvk=23)
    ok2 = res["blocked"] and any(c["name"] == "payout" for c in res["consumers"])  # payout declares any -> not blocked...
    # payout declares "any": it is compatible; make the block real by narrowing it
    for c in cfg["consumer"]:
        if c["name"] == "payout":
            c["expects_units"] = ["teacher_sd"]
    res = contract_compat.evaluate(cfg, target_mode="sd_v2", target_wvk=23)
    ok2 = res["blocked"] and any(c["name"] == "payout" and c["status"] == "mismatch" for c in res["consumers"])
    # re-declare unit-free -> auto-unfreeze
    write(sb.state_dir / "curriculum/latest.json", {"mode": "apply", "computed_at": common.iso(NOW - 600),
                                                     "for_epoch": 52, "input_units": "unit_free",
                                                     "fitted_score_modes": ["min_rg", "sd_min_rga"]})
    out2 = mon.tick()
    ok3 = not frozen.exists() and "contract_mismatch" not in keys(out2) and any("UNFROZE" in l for l in cap.lines)
    return ok1 and ok2 and ok3, (f"mismatch+freeze={'yes' if ok1 else 'NO'} preflight-block(payout narrowed)="
                                 f"{'yes' if ok2 else 'NO'} auto-unfreeze={'yes' if ok3 else 'NO'}")


def drill_d(sb: Sandbox) -> tuple[bool, str]:
    """King seat empty 25 h (published box serves the old king); kingctl started before its code changed."""
    write(sb.root / "ops/king-datagen/state/state.json", {
        "published": {"pod": "king-dg-73dd5bbcf1f7-aaaa", "at": NOW - 30 * H},
        "pods": {"king-dg-73dd5bbcf1f7-aaaa": {"king": {"ident": "73dd5bbcf1f7"}, "last_ok": NOW - 60}},
        "last_rent_failure": "no stock under price caps for any type (0 executors listed)"})
    # stale code: kingctl.py "changed" 30 min ago, process started 2 h ago
    for p in sb.procs:
        if p["name"] == "affine-king-datagen":
            p["pm2_env"]["pm_uptime"] = int((NOW - 2 * H) * 1000)
    stale_file = write(sb.root / "ops/king-datagen/kingctl.py", "# fixed\n")
    os.utime(stale_file, (NOW - 1800, NOW - 1800))
    (sb.root / "ops/health/code_watch.toml").write_text(
        CODE_WATCH_RULES + f'[[process]]\nname = "affine-king-datagen"\naction = "restart"\nfiles = ["{stale_file}"]\n')
    restarted = []
    health.Monitor.restart = lambda self, name: restarted.append(name)
    mon, cap = sb.monitor(since={"king_seat_empty": NOW - 25 * H})
    out = mon.tick()
    k = keys(out)
    ok = "king_seat" in k and any("EMPTY" in l and "king_seat" in l for l in cap.lines) \
        and restarted == ["affine-king-datagen"] and any(x.startswith("code_restart:affine-king-datagen") for x in k)
    return ok, f"king_seat page={'yes' if 'king_seat' in k else 'NO'} restarted={restarted}"


def drill_e(sb: Sandbox) -> tuple[bool, str]:
    """Bench queue held: a restart reset the cap to $400 while $539 was already spent."""
    write(sb.root / "ops/coverage/state/budget.json", {"cap_usd": 400.0, "actuals": {"fill": 499.37, "pre_fill": 39.89}})
    mon, cap = sb.monitor(since={"bench_held": NOW - 61 * 60})
    out = mon.tick()
    k = keys(out)
    ok = "bench_queue_held" in k and any("HELD" in l for l in cap.lines)
    return ok, f"bench_queue_held page={'yes' if ok else 'NO'}: {out['checks'].get('bench_queue_held', {}).get('detail', '')[:90]}"


def drill_f(sb: Sandbox) -> tuple[bool, str]:
    """Reaper: bench box 40 h old, owner (pass log) dead, expected 14 h -> release + $ saved; unregistered pod paged."""
    import lium_api
    import reaper
    import registry
    cfg = tomllib.loads((REPO / "ops/pods/pods.toml").read_text())
    st = sb.root / "ops/pods/state"
    cfg["paths"] = {"registry": str(st / "registry.json"), "ledger": str(st / "ledger.jsonl"),
                    "own_state": str(st / "reaper_state.json"), "report": str(sb.state_dir / "pods/reaper.json")}
    cfg["adopt"] = []
    cfg["reaper"]["unregistered_page_min"] = 0
    (sb.root / "ops/pods").mkdir(parents=True, exist_ok=True)
    registry.load_config = lambda: cfg
    reaper.HERE = sb.root / "ops/pods"
    registry.HERE = sb.root / "ops/pods"
    registry.register("bench-king-0f4029fd59ed-f2c1", purpose="bench", owner="passlog:ops/benchsuite/state",
                      expected_hours=14, price_usd_h=3.0, source="explicit")
    listing = [
        {"pod_name": "bench-king-0f4029fd59ed-f2c1", "status": "RUNNING", "price": 3.0,
         "created_at": common.iso(NOW - 40 * H), "removal_scheduled_at": common.iso(NOW + 8 * H),
         "executor": {"machine_name": "NVIDIA H200"}},
        {"pod_name": "affine-probe-nobody", "status": "RUNNING", "price": 2.0, "created_at": common.iso(NOW - 3 * H),
         "executor": {"machine_name": "NVIDIA H200"}},
        {"pod_name": "mining-teacher-3", "status": "RUNNING", "price": 64.0, "created_at": common.iso(NOW - 90 * H),
         "executor": {"machine_name": "NVIDIA B300"}},
    ]
    lium_api.pods = lambda sess: listing
    lium_api.session = lambda: None
    removed = []
    lium_api.remove = lambda name, prefix: removed.append((name, prefix)) or True
    reaper.lium_api = lium_api
    reaper._proc_cmdlines = lambda: []
    reaper.common.pm2_jlist = lambda: sb.procs
    cap = Capture()
    reaper.common.discord_post = cap
    r = reaper.Reaper(cfg, dry_run=False)
    r.enforce = True
    rep = r.tick()
    act = [a for a in rep["actions"] if a["pod"] == "bench-king-0f4029fd59ed-f2c1"]
    ok1 = bool(act) and act[0]["released"] and abs(act[0]["saved_usd_est"] - 24.0) < 0.5 \
        and ("bench-king-0f4029fd59ed-f2c1", "bench-king-") in removed \
        and any("released `bench-king-0f4029fd59ed-f2c1`" in l and "saved" in l for l in cap.lines)
    ok2 = any("unregistered pod `affine-probe-nobody`" in l for l in cap.lines) \
        and rep["foreign"]["count"] == 1 and "mining-teacher-3" not in [n for n, _ in removed]
    # monitor surfaces it
    rep_over = {"at": common.iso(NOW), "n_ours": 1, "ours_usd_h": 3.0, "foreign": {"count": 1, "usd_h": 64},
                "ours": [{"name": "bench-king-x", "age_h": 40.2, "price_usd_h": 3.0, "owner_alive": False}],
                "problems": ["over lifetime bench-king-x (40 h > 14 h)"], "actions": []}
    write(sb.state_dir / "pods/reaper.json", rep_over)
    mon, cap2 = sb.monitor()
    out = mon.tick()
    ok3 = "pods" in keys(out) and any("without a live driver" in l for l in cap2.lines)
    return ok1 and ok2 and ok3, (f"release_expired={'yes' if ok1 else 'NO'} saved=${act[0]['saved_usd_est'] if act else '?'} "
                                 f"unregistered-page+foreign-untouched={'yes' if ok2 else 'NO'} monitor-page={'yes' if ok3 else 'NO'}")


def drill_g(sb: Sandbox) -> tuple[bool, str]:
    """sources.toml differs from HEAD for 31 min -> page; a fold started meanwhile records the dirty flag."""
    sb.dirty = True
    mon, cap = sb.monitor(since={"sources_dirty": NOW - 31 * 60})
    out = mon.tick()
    k = keys(out)
    ok1 = "sources_toml_drift" in k and any("sources_toml_drift" in l for l in cap.lines)
    import fold_wrap
    fold_wrap.sources_dirty = lambda: True
    fold_wrap.CORPUS_BUILD = write(sb.root / "ok_fold.py", "print('ok')\n")
    fold_wrap.FOLD_DIR = sb.state_dir / "fold"
    fold_wrap.LAST_RUN = fold_wrap.FOLD_DIR / "last_run.json"
    fold_wrap.HISTORY = fold_wrap.FOLD_DIR / "history.jsonl"
    fold_wrap.RUNS_DIR = fold_wrap.FOLD_DIR / "runs"
    fold_wrap.common.discord_post = Capture()
    sys.argv = ["fold_wrap.py", "--"]
    rc = fold_wrap.main()
    rec = json.loads(fold_wrap.LAST_RUN.read_text())
    ok2 = rc == 0 and rec["sources_toml_dirty"] is True and rec["classification"] == "ok"
    sb.dirty = False
    return ok1 and ok2, f"drift page={'yes' if ok1 else 'NO'}; fold recorded sources_toml_dirty={rec.get('sources_toml_dirty')}"


def drill_x(sb: Sandbox) -> tuple[bool, str]:
    """state.json in_flight already has a verdict in history."""
    st = json.loads((sb.state_dir / "state.json").read_text())
    st["in_flight"] = {"challenge_id": "chal-00610"}
    write(sb.state_dir / "state.json", st)
    mon, cap = sb.monitor()
    out = mon.tick()
    ok = "inflight_stale" in keys(out) and any("inflight_stale" in l for l in cap.lines)
    st["in_flight"] = None
    write(sb.state_dir / "state.json", st)
    return ok, f"inflight_stale page={'yes' if ok else 'NO'}"


DRILLS = [("a", "fold died on KeyError, published nothing", drill_a),
          ("b", "no fold for 23 h after the fork", drill_b),
          ("c", "curriculum units vs live score_mode (+freeze, preflight block, unfreeze)", drill_c),
          ("d", "king seat empty 25 h; kingctl on stale code", drill_d),
          ("e", "bench queue budget-held after a pm2 restart", drill_e),
          ("f", "env boxes idle 40 h, driver dead ($ saved) + unregistered pod", drill_f),
          ("g", "fold on a stale sources.toml", drill_g),
          ("x", "in_flight stale vs history", drill_x)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-k", default="", help="only drills whose id contains this")
    args = ap.parse_args()
    failures = 0
    print(f"{'id':3} {'result':6} drill")
    for did, title, fn in DRILLS:
        if args.k and args.k not in did:
            continue
        with tempfile.TemporaryDirectory(prefix="affine-drill-") as td:
            sb = Sandbox(Path(td))
            try:
                ok, detail = fn(sb)
            except Exception as e:  # a drill that crashes is a failed drill
                ok, detail = False, f"CRASH {e!r}"
        failures += 0 if ok else 1
        print(f"{did:3} {'PASS' if ok else 'FAIL':6} ({title}) — {detail}")
    print("ALL DRILLS PASSED" if not failures else f"{failures} DRILL(S) FAILED")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
