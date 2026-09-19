#!/usr/bin/env python
"""affine-pipeline-health — is the data pipeline moving, and does every
consumer still understand the numbers it reads?

Every tick (pm2 `affine-pipeline-health`, 5 min) this reads, without writing
anything on the production side:

  corpus manifest (data.affine.io)   last epoch + its age              (a)(b)
  affine/state/fold/last_run.json    fold exit / traceback / cadence   (a)(b)(g)
  pm2 jlist                          fold cron, process liveness, start times
  affine/state/curriculum/latest.json  vector age, mode, declared units (c)
  ops/health/contract_compat.py      units vs live score_mode, freeze   (c)
  ops/king-datagen/state/state.json  king seat published for the king   (d)
  127.0.0.1:9100/health              teacher swarm healthy replicas
  affine/state/{state,history}.json  verdict cadence, in_flight vs history
  ops/coverage/state/*.json          bench queue held / budget          (e)
  affine/state/pods/reaper.json      pods over lifetime, reaper alive   (f)
  git diff rollouts/sources.toml     drift vs HEAD                      (g)
  code_watch.toml                    processes running stale code       (d)

and writes affine/state/pipeline_health.json (served by the kingboard at
/api/pipeline_health.json). A condition that holds is an alert; each alert
key posts ONE Discord line per dedupe window, plus one hourly summary.

    python health.py --once --dry-run      # one pass, print, post nothing
    python health.py --interval 300        # daemon
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import common  # noqa: E402
import contract_compat  # noqa: E402

TAG = "pipeline-health"


def log(msg: str) -> None:
    common.log(TAG, msg)


class Config:
    def __init__(self, path: Path):
        d = tomllib.loads(path.read_text())
        self.raw = d
        w, t, p, s, dc = d["watch"], d["thresholds"], d["paths"], d["sources"], d["discord"]
        self.interval_s = int(w.get("interval_s", 300))
        self.dedupe_s = float(w.get("alert_dedupe_h", 6)) * 3600
        self.undeclared_dedupe_s = float(w.get("undeclared_dedupe_h", 24)) * 3600
        self.summary_s = float(w.get("summary_every_min", 60)) * 60
        self.recovery_lines = bool(w.get("recovery_lines", True))
        self.t = t
        self.paths = {k: (HERE / v).resolve() for k, v in p.items()}
        self.sources = s
        self.discord_enabled = bool(dc.get("enabled", True))
        self.discord_channel = str(dc.get("channel_id", ""))
        self.discord_token_env = str(dc.get("token_env", common.DISCORD_TOKEN_ENV))
        self.discord_prefix = str(dc.get("prefix", f"[{TAG}]"))
        self.code_watch = tomllib.loads(self.paths["code_watch"].read_text())
        self.consumers = tomllib.loads(self.paths["consumers"].read_text())


class Check:
    """One named observation. level: ok | info | warn | page."""

    def __init__(self, key: str, level: str, detail: str, **values):
        self.key, self.level, self.detail, self.values = key, level, detail, values

    def as_dict(self) -> dict:
        return {"level": self.level, "detail": self.detail, **self.values}


# -- collectors -----------------------------------------------------------------------

def fetch_manifest(cfg: Config) -> tuple[dict | None, str]:
    url = cfg.sources["corpus_manifest_url"]
    try:
        r = requests.get(url, timeout=20, headers={"User-Agent": "affine-pipeline-health/1"})
        if r.status_code == 200:
            return r.json(), "public"
    except (requests.RequestException, ValueError):
        pass
    local = common.read_json(cfg.paths["corpus_manifest_local"])
    return (local, "local-cache") if local else (None, "unreachable")


def router_health(cfg: Config) -> dict:
    try:
        r = requests.get(cfg.sources["router_health_url"], timeout=10)
        d = r.json()
        return {"reachable": True, "ok": bool(d.get("ok")), "healthy": int(d.get("healthy") or 0),
                "backends": int(d.get("backends") or 0)}
    except (requests.RequestException, ValueError, TypeError):
        return {"reachable": False, "ok": False, "healthy": 0, "backends": 0}


CHAT_POD_SCRIPT = r'''
for d in __DIRS__; do [ -d "$d" ] && { echo "DF $(df -BG --output=avail,size "$d" | tail -1 | tr -s " ") $d"; break; }; done
echo "PS_BEGIN"; ps -eo pid=,ppid=,args= | grep -E "EngineCore|vllm serve|evalsrv.chatsrv" | grep -v grep; echo "PS_END"
echo "SNAP_BEGIN"; ls -d /root/hf/hub/models--*/snapshots/* 2>/dev/null; echo "SNAP_END"
'''


def chat_pod_probe(ssh_str: str, dirs: list[str], timeout_s: int) -> dict:
    """One ssh round trip to the chat pod: free GB on the model volume, the
    vLLM / EngineCore process table, the king snapshots on disk. Returns
    {"ok": False, "error": ...} when ssh fails."""
    parts = ssh_str.split()
    if not parts or "-p" not in parts:
        return {"ok": False, "error": f"bad ssh string {ssh_str!r}"}
    userhost, port = parts[0], parts[parts.index("-p") + 1]
    script = CHAT_POD_SCRIPT.replace("__DIRS__", " ".join(dirs))
    cmd = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
           "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR",
           "-o", f"ConnectTimeout={min(15, timeout_s)}", "-p", port, userhost, "bash -s"]
    try:
        r = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout_s)
    except (subprocess.SubprocessError, OSError) as e:
        return {"ok": False, "error": repr(e)[:120]}
    if r.returncode != 0 and "DF " not in r.stdout:
        return {"ok": False, "error": (r.stderr.strip() or f"exit {r.returncode}")[:160]}
    out: dict = {"ok": True, "free_gb": None, "size_gb": None, "df_path": None,
                 "procs": [], "snapshots": []}
    section = None
    for line in r.stdout.splitlines():
        if line.startswith("DF "):
            f = line.split()
            try:
                out["free_gb"], out["size_gb"], out["df_path"] = int(f[1].rstrip("G")), int(f[2].rstrip("G")), f[3]
            except (IndexError, ValueError):
                pass
        elif line in ("PS_BEGIN", "SNAP_BEGIN"):
            section = line
        elif line in ("PS_END", "SNAP_END"):
            section = None
        elif section == "PS_BEGIN":
            f = line.split(None, 2)
            if len(f) == 3:
                out["procs"].append({"pid": int(f[0]), "ppid": int(f[1]), "args": f[2][:120]})
        elif section == "SNAP_BEGIN" and line.strip():
            out["snapshots"].append(line.strip())
    # an EngineCore whose parent is not a live `vllm serve` is an orphan
    serve_pids = {p["pid"] for p in out["procs"] if "vllm serve" in p["args"]}
    out["orphans"] = [p for p in out["procs"] if "EngineCore" in p["args"] and p["ppid"] not in serve_pids]
    out["n_engines"] = sum(1 for p in out["procs"] if "EngineCore" in p["args"])
    return out


def git_dirty(path: Path) -> bool | None:
    try:
        r = subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--",
                            str(path.relative_to(REPO))], timeout=20)
    except (subprocess.SubprocessError, OSError, ValueError):
        return None
    return r.returncode != 0


# -- the monitor ----------------------------------------------------------------------

class Monitor:
    def __init__(self, cfg: Config, *, dry_run: bool):
        self.cfg = cfg
        self.dry_run = dry_run
        self.own = common.read_json(cfg.paths["own_state"], default={}) or {}
        for k, v in (("alerts_sent", {}), ("active", {}), ("since", {}), ("restarted", {}),
                     ("last_summary", 0.0)):
            self.own.setdefault(k, v)

    # ---- state helpers
    def since(self, key: str, holds: bool, now: float) -> float | None:
        """Track how long a condition has held. Returns seconds or None."""
        if holds:
            self.own["since"].setdefault(key, now)
            return now - float(self.own["since"][key])
        self.own["since"].pop(key, None)
        return None

    def post(self, text: str) -> None:
        if not self.cfg.discord_enabled:
            log(f"(discord disabled) {text}")
            return
        common.discord_post(text, channel=self.cfg.discord_channel,
                            token_env=self.cfg.discord_token_env, dry_run=self.dry_run,
                            prefix=self.cfg.discord_prefix)

    # ---- checks
    def run_checks(self, now: float) -> list[Check]:
        cfg, t = self.cfg, self.cfg.t
        checks: list[Check] = []
        procs = common.pm2_jlist()
        state = common.read_json(cfg.paths["state_dir"] / "state.json")
        history = common.jsonl_tail(cfg.paths["state_dir"] / "history.jsonl", 3_000_000)

        # (a)(b) corpus epoch age
        manifest, src = fetch_manifest(cfg)
        if manifest is None:
            checks.append(Check("corpus_manifest", "page", "corpus manifest unreachable (public + local cache)"))
        else:
            pub = common.parse_iso(manifest.get("published_at") or manifest.get("created_at"))
            age = (now - pub) if pub else None
            epoch = manifest.get("corpus_epoch")
            lvl = "page" if age is None or age > t["corpus_epoch_max_age_h"] * 3600 else "ok"
            checks.append(Check("corpus_epoch_age", lvl,
                                f"corpus epoch {epoch} published {common.fmt_age(age)} ago ({src})"
                                + (f" > {t['corpus_epoch_max_age_h']:.0f} h" if lvl == "page" else ""),
                                epoch=epoch, age_h=round(age / 3600, 2) if age else None,
                                n_turns=manifest.get("n_turns"), n_strata=manifest.get("n_strata")))

        # (a)(b) fold: exit status, cadence, pm2 cron
        fold_p = common.pm2_process(procs, cfg.sources["fold_pm2_name"])
        cron = (fold_p or {}).get("cron")
        interval = common.cron_interval_seconds(cron)
        if procs is None:
            checks.append(Check("pm2", "page", "pm2 jlist failed — process checks unavailable"))
        elif fold_p is None:
            checks.append(Check("fold_pm2", "page", f"pm2 process {cfg.sources['fold_pm2_name']} is missing"))
        elif not cron:
            checks.append(Check("fold_pm2", "page", f"{cfg.sources['fold_pm2_name']} has no cron schedule"))
        else:
            cmdline = " ".join(str(x) for x in ((fold_p.get("args") or []) + [fold_p.get("script")]))
            wrapped = "run_fold" in cmdline or "fold_wrap" in cmdline
            checks.append(Check("fold_pm2", "ok" if wrapped else "warn",
                                f"fold cron `{cron}` (every {common.fmt_age(interval)})"
                                + ("" if wrapped else " — NOT running through ops/fold/run_fold.sh (no exit record / page)"),
                                cron=cron, interval_h=round(interval / 3600, 2) if interval else None,
                                wrapped=wrapped))
        last = common.read_json(cfg.paths["fold_last_run"])
        if last is None:
            checks.append(Check("fold_last_run", "warn", "no affine/state/fold/last_run.json yet (fold wrapper never ran)"))
            # until the wrapper has run once, the manifest's publish time is
            # the only clock for "is the fold overdue"
            pub = common.parse_iso((manifest or {}).get("published_at") or (manifest or {}).get("created_at"))
            if interval and pub:
                overdue_by = now - pub - interval - t["fold_overdue_slack_h"] * 3600
                if overdue_by > 0:
                    checks.append(Check("fold_overdue", "page",
                                        f"no fold published for {common.fmt_age(now - pub)} "
                                        f"(cron every {common.fmt_age(interval)} + {t['fold_overdue_slack_h']:.0f} h slack; "
                                        f"no wrapper record yet)", last_published=(manifest or {}).get("published_at")))
        else:
            started = float(last.get("started_ts") or 0) or common.parse_iso(last.get("started_at")) or 0.0
            if last.get("status") == "running":
                run_for = now - started
                lvl = "page" if run_for > t["fold_running_max_h"] * 3600 else "ok"
                checks.append(Check("fold_running", lvl, f"fold running for {common.fmt_age(run_for)}"
                                    + (f" > {t['fold_running_max_h']:.0f} h" if lvl == "page" else ""),
                                    pid=last.get("pid"), started_at=last.get("started_at")))
            else:
                rc = last.get("exit_code")
                cls = last.get("classification")
                if rc not in (0, None) and cls != "skipped_lock":
                    tb = (last.get("traceback_tail") or "").splitlines()
                    checks.append(Check(f"fold_failed:{last.get('started_at')}", "page",
                                        f"last fold FAILED exit {rc} ({cls}, {last.get('attempts')} attempt(s), "
                                        f"started {last.get('started_at')}) — tail: "
                                        + " | ".join(l.strip() for l in tb[-4:])[:600],
                                        exit_code=rc, classification=cls, log=last.get("log")))
                else:
                    checks.append(Check("fold_last_run", "ok",
                                        f"last fold {cls or 'ok'} exit {rc} at {last.get('ended_at')} "
                                        f"({last.get('duration_s')}s, HEAD {last.get('git_head')})",
                                        exit_code=rc, ended_at=last.get("ended_at"),
                                        sources_toml_sha256=last.get("sources_toml_sha256")))
            if interval and started:
                overdue_by = now - started - interval - t["fold_overdue_slack_h"] * 3600
                if overdue_by > 0:
                    checks.append(Check("fold_overdue", "page",
                                        f"no fold started for {common.fmt_age(now - started)} "
                                        f"(cron every {common.fmt_age(interval)} + {t['fold_overdue_slack_h']:.0f} h slack)",
                                        last_started=last.get("started_at")))

        # (c) curriculum vector age / mode / units
        cur = common.read_json(cfg.paths["curriculum_latest"])
        frozen = common.read_json(cfg.paths["curriculum_frozen"])
        if cur is None:
            checks.append(Check("curriculum", "warn", "curriculum latest.json missing"))
        else:
            cat = common.parse_iso(cur.get("computed_at"))
            age = (now - cat) if cat else None
            lvl = "page" if age is None or age > t["curriculum_max_age_h"] * 3600 else "ok"
            checks.append(Check("curriculum_age", lvl,
                                f"curriculum vector mode={cur.get('mode')} computed {common.fmt_age(age)} ago "
                                f"(for epoch {cur.get('for_epoch')}, units={cur.get('input_units') or 'undeclared'}"
                                f"{', FROZEN' if frozen else ''})"
                                + (f" > {t['curriculum_max_age_h']:.0f} h" if lvl == "page" else ""),
                                mode=cur.get("mode"), age_h=round(age / 3600, 2) if age else None,
                                input_units=cur.get("input_units"),
                                fitted_score_modes=cur.get("fitted_score_modes"),
                                frozen=bool(frozen), frozen_reason=(frozen or {}).get("reason")))

        # (c) contract compatibility (+ auto freeze / unfreeze)
        try:
            res = contract_compat.evaluate(cfg.consumers)
            events = contract_compat.enforce(cfg.consumers, res, dry_run=self.dry_run)
            for e in events:
                checks.append(Check(f"contract_event:{e.split(':')[0]}:{int(now)}", "page", e))
            mism = [c for c in res["consumers"] if c["status"] == "mismatch"]
            undecl = [c for c in res["consumers"] if c["status"] == "undeclared"]
            live = res["live"]
            base = (f"score_mode {live['toml_score_mode']} ({live['toml_units']}, wvk {live['wvk']}); "
                    f"stamped {live['stamped_score_mode']}")
            if mism:
                checks.append(Check("contract_mismatch", "page",
                                    f"{base}; MISMATCH: " + "; ".join(
                                        f"{c['name']} declares {c['declared_units'] or c['declared_score_modes']}"
                                        f"{' (frozen)' if c['frozen'] else ''}" for c in mism),
                                    consumers=res["consumers"], problems=res["problems"]))
            else:
                checks.append(Check("contract_mismatch", "ok", f"{base}; all consumers compatible",
                                    consumers=res["consumers"]))
            if undecl:
                checks.append(Check("contract_undeclared", "warn",
                                    "consumers without a live units stamp: " + ", ".join(c["name"] for c in undecl)
                                    + " (static declaration used)"))
            for p in res["problems"]:
                if "stamps" in p:
                    checks.append(Check("contract_stamp_lag", "warn", p))
        except (OSError, ValueError, KeyError) as e:
            checks.append(Check("contract_compat", "page", f"contract_compat failed: {e!r}"))

        # (d) king seat
        ks = common.read_json(cfg.paths["kingctl_state"])
        king = (state or {}).get("king") or {}
        ident = str(king.get("revision") or "")[:12]
        kingctl_online = common.pm2_online(procs, cfg.sources["kingctl_pm2_name"])
        if kingctl_online is False:
            checks.append(Check("kingctl_down", "page", f"pm2 {cfg.sources['kingctl_pm2_name']} is not online"))
        serving = False
        pub = (ks or {}).get("published") or {}
        pub_pod = pub.get("pod")
        pod_mem = ((ks or {}).get("pods") or {}).get(pub_pod) if pub_pod else None
        if pod_mem:
            pod_ident = ((pod_mem.get("king") or {}).get("ident")) or ""
            last_ok = float(pod_mem.get("last_ok") or 0)
            serving = pod_ident == ident and (now - last_ok) < t["king_box_last_ok_max_min"] * 60
        empty_for = self.since("king_seat_empty", not serving, now)
        if serving:
            n_boxes = sum(1 for m in ((ks or {}).get("pods") or {}).values()
                          if ((m.get("king") or {}).get("ident")) == ident and m.get("ready_at"))
            checks.append(Check("king_seat", "ok",
                                f"king seat serving reign {king.get('reign_number')} ({ident}) on {pub_pod} "
                                f"({n_boxes} box(es))", ident=ident, pod=pub_pod, boxes=n_boxes))
        else:
            lvl = "page" if (empty_for or 0) > t["king_seat_empty_max_min"] * 60 else "warn"
            why = ("kingctl state unreadable" if ks is None else
                   "nothing published" if not pub_pod else
                   f"published {pub_pod} serves {((pod_mem or {}).get('king') or {}).get('ident')} not {ident}"
                   if pod_mem and ((pod_mem.get("king") or {}).get("ident")) != ident else
                   f"published {pub_pod} canary stale" if pod_mem else f"published {pub_pod} unknown to kingctl")
            checks.append(Check("king_seat", lvl,
                                f"king seat EMPTY for {common.fmt_age(empty_for)} (reign {king.get('reign_number')} "
                                f"{ident}): {why}; last_rent_failure={((ks or {}).get('last_rent_failure') or '')[:120]}",
                                ident=ident, empty_min=round((empty_for or 0) / 60)))

        # chat.affine.io: endpoint, served digest vs king, disk, orphan engines
        checks.extend(self.chat_checks(state, king, now))

        # teacher swarm replicas
        rh = router_health(cfg)
        lvl = "page" if (not rh["reachable"] or rh["healthy"] < t["teacher_replicas_min"]) else "ok"
        checks.append(Check("teacher_replicas", lvl,
                            f"teacher swarm {rh['healthy']}/{rh['backends']} healthy replicas"
                            + ("" if rh["reachable"] else " (router unreachable)")
                            + (f" < {t['teacher_replicas_min']}" if lvl == "page" else ""), **rh))

        # verdict cadence + in_flight vs history
        if state is None:
            checks.append(Check("state_json", "page", "affine/state/state.json unreadable"))
        else:
            queue = state.get("queue") or []
            inflight = state.get("in_flight") or {}
            cid = inflight.get("challenge_id") if isinstance(inflight, dict) else inflight
            terminal = [r for r in history if r.get("event") in ("verdict", "crowned", "failed")]
            last_v = terminal[-1] if terminal else None
            last_at = common.parse_iso((last_v or {}).get("at"))
            age = (now - last_at) if last_at else None
            busy = bool(queue) or bool(cid)
            lvl = "page" if busy and (age is None or age > t["verdict_max_age_h"] * 3600) else "ok"
            checks.append(Check("verdict_cadence", lvl,
                                f"last verdict {(last_v or {}).get('challenge_id')} {common.fmt_age(age)} ago; "
                                f"queue {len(queue)}, in_flight {cid or '-'}"
                                + (f" — no verdict for > {t['verdict_max_age_h']} h while work is queued" if lvl == "page" else ""),
                                last_challenge=(last_v or {}).get("challenge_id"),
                                age_h=round(age / 3600, 2) if age else None, queue=len(queue), in_flight=cid))
            if cid and any(r.get("challenge_id") == cid for r in terminal):
                checks.append(Check("inflight_stale", "page",
                                    f"state.json in_flight = {cid} but its verdict is already in history.jsonl "
                                    f"(State.load would requeue it — clear it before a validator restart)", cid=cid))
            wts = common.parse_iso(state.get("last_weights_at"))
            wage = (now - wts) if wts else None
            if wage is None or wage > t["weights_max_age_h"] * 3600:
                checks.append(Check("weights_stale", "page", f"last set_weights {common.fmt_age(wage)} ago"))

        # (e) bench queue held / budget
        meta = common.read_json(cfg.paths["bench_queue_meta"]) or {}
        budget = common.read_json(cfg.paths["bench_budget"]) or {}
        cap = float(budget.get("cap_usd") or 0)
        actuals = sum(float(v) for v in (budget.get("actuals") or {}).values())
        bench_online = common.pm2_online(procs, cfg.sources["bench_pm2_name"])
        held_reason = None
        if cap and actuals >= cap:
            held_reason = f"budget reached ${actuals:.0f} >= cap ${cap:.0f}"
        if meta.get("held") or meta.get("hold_reason"):
            held_reason = str(meta.get("hold_reason") or meta.get("held"))
        if meta.get("active") is False:
            held_reason = held_reason or "queue inactive"
        held_for = self.since("bench_held", held_reason is not None, now)
        if bench_online is False:
            checks.append(Check("bench_queue_down", "page", f"pm2 {cfg.sources['bench_pm2_name']} is not online"))
        if held_reason:
            lvl = "page" if (held_for or 0) > t["bench_held_max_min"] * 60 else "warn"
            checks.append(Check("bench_queue_held", lvl,
                                f"bench queue HELD for {common.fmt_age(held_for)}: {held_reason} "
                                f"(spent ${actuals:.0f} of cap ${cap:.0f}; buckets {budget.get('actuals')})",
                                cap_usd=cap, spent_usd=round(actuals, 2), reason=held_reason))
        else:
            checks.append(Check("bench_queue", "ok",
                                f"bench queue active; spent ${actuals:.0f} of cap ${cap:.0f} "
                                f"({len(meta.get('pods_seen') or [])} pods seen)",
                                cap_usd=cap, spent_usd=round(actuals, 2)))

        # (f) pods: the reaper's report
        rep = common.read_json(cfg.paths["reaper_report"])
        reaper_online = common.pm2_online(procs, cfg.sources["reaper_pm2_name"])
        if reaper_online is False:
            checks.append(Check("reaper_down", "page", f"pm2 {cfg.sources['reaper_pm2_name']} is not online"))
        if rep is None:
            checks.append(Check("pods", "warn", "no reaper report yet (affine/state/pods/reaper.json)"))
        else:
            rage = now - (common.parse_iso(rep.get("at")) or 0)
            if rage > t["pods_report_max_age_min"] * 60:
                checks.append(Check("reaper_stale", "page", f"reaper report is {common.fmt_age(rage)} old"))
            over = [x for x in rep.get("ours", []) if (x.get("age_h") or 0) > t["pod_max_age_no_driver_h"]
                    and x.get("owner_alive") is False]
            problems = rep.get("problems") or []
            lvl = "page" if (over or problems) else "ok"
            names = ", ".join(f"{x['name']} ({x['age_h']} h, ${x['price_usd_h']}/h)" for x in over)
            checks.append(Check("pods", lvl,
                                f"{rep.get('n_ours')} pods of ours ${rep.get('ours_usd_h')}/h, "
                                f"{(rep.get('foreign') or {}).get('count')} foreign ${(rep.get('foreign') or {}).get('usd_h')}/h"
                                + (f"; > {t['pod_max_age_no_driver_h']} h without a live driver: {names}" if over else "")
                                + (f"; problems: {'; '.join(problems)}" if problems else ""),
                                n_ours=rep.get("n_ours"), ours_usd_h=rep.get("ours_usd_h"),
                                foreign=rep.get("foreign"), over=[x["name"] for x in over], problems=problems,
                                actions=rep.get("actions")))

        # (g) sources.toml drift vs HEAD, and vs the last fold
        dirty = git_dirty(cfg.paths["sources_toml"])
        dirty_for = self.since("sources_dirty", bool(dirty), now)
        cur_sha = common.sha256_file(cfg.paths["sources_toml"])
        fold_sha = (last or {}).get("sources_toml_sha256")
        if dirty:
            lvl = "page" if (dirty_for or 0) > t["sources_dirty_max_min"] * 60 else "warn"
            checks.append(Check("sources_toml_drift", lvl,
                                f"rollouts/sources.toml differs from HEAD for {common.fmt_age(dirty_for)} "
                                f"(uncommitted edit — the next fold would run on it)", dirty_min=round((dirty_for or 0) / 60)))
        elif dirty is None:
            checks.append(Check("sources_toml_drift", "warn", "git diff for sources.toml failed"))
        else:
            note = ""
            if fold_sha and cur_sha and fold_sha != cur_sha:
                note = " (changed since the last fold; next fold picks it up)"
            checks.append(Check("sources_toml_drift", "ok", "sources.toml matches HEAD" + note,
                                sha256=cur_sha, last_fold_sha256=fold_sha))

        # guard hooks living in other files must still be there
        missing = []
        for it in cfg.raw.get("integrity", []):
            try:
                text = (REPO / it["file"]).read_text()
            except OSError:
                missing.append(f"{it['file']} unreadable")
                continue
            if it["must_contain"] not in text:
                note = ""
                if it.get("repair") and not self.dry_run:
                    try:
                        r = subprocess.run([common.python_bin(), str(REPO / it["repair"])], capture_output=True,
                                           text=True, timeout=60)
                        note = f" — repair `{it['repair']}`: {(r.stdout or r.stderr).strip()[:80]}"
                    except (subprocess.SubprocessError, OSError) as e:
                        note = f" — repair failed: {e!r}"
                missing.append(f"{it['file']} lost `{it['must_contain']}` ({it['why']}){note}")
        checks.append(Check("guard_integrity", "page" if missing else "ok",
                            "guard hook missing: " + "; ".join(missing) if missing
                            else f"{len(cfg.raw.get('integrity', []))} guard hooks present", missing=missing))

        # (d) processes running stale code
        checks.extend(self.code_watch(procs, now))
        return checks

    # ---- chat box
    def chat_checks(self, state: dict | None, king: dict, now: float) -> list[Check]:
        cfg, t, s = self.cfg, self.cfg.t, self.cfg.sources
        out: list[Check] = []
        king_rev = str(king.get("revision") or "")
        crowned = common.parse_iso(king.get("crowned_at"))
        since_crown = (now - crowned) if crowned else None
        in_grace = since_crown is not None and since_crown < t["chat_crown_grace_min"] * 60

        # 1. /health + /v1/models through the validator's forward
        health: dict | None = None
        models_ok = False
        try:
            r = requests.get(s["chat_health_url"], timeout=10)
            health = r.json() if r.status_code == 200 else {"state": f"http {r.status_code}"}
        except (requests.RequestException, ValueError):
            health = None
        token = common.env_file_value(s.get("chat_token_env", "AFFINE_EVAL_TOKEN"))
        try:
            r = requests.get(s["chat_models_url"], timeout=10,
                             headers={"Authorization": f"Bearer {token}"} if token else {})
            models_ok = r.status_code == 200 and any(
                m.get("id") == "affine-king" for m in (r.json().get("data") or []))
        except (requests.RequestException, ValueError, AttributeError):
            models_ok = False
        st = (health or {}).get("state") or "unreachable"
        served = str(((health or {}).get("king") or {}).get("revision") or "")
        serving = st == "serving" and models_ok and served == king_rev
        not_serving_for = self.since("chat_not_serving", not serving, now)
        if serving:
            out.append(Check("chat_endpoint", "ok",
                             f"chat.affine.io serving reign {king.get('reign_number')} ({king_rev[:12]}), /v1/models ok",
                             state=st, served=served[:12]))
        else:
            why = ("/health unreachable" if health is None else
                   f"state={st}" if st != "serving" else
                   "/v1/models does not list affine-king" if not models_ok else
                   f"serves {served[:12] or '?'} not the king {king_rev[:12]}")
            over = (not_serving_for or 0) > t["chat_loading_max_min"] * 60
            lvl = "page" if over and not (in_grace and served != king_rev and st == "serving") else "warn"
            out.append(Check("chat_endpoint", lvl,
                             f"chat.affine.io NOT serving the king for {common.fmt_age(not_serving_for)}: {why}"
                             + (f" (crown {common.fmt_age(since_crown)} ago, inside the {t['chat_crown_grace_min']}-min grace)" if in_grace else "")
                             + (f"; error={str((health or {}).get('error') or '')[:120]}" if (health or {}).get("error") else ""),
                             state=st, served=served[:12], king=king_rev[:12],
                             not_serving_min=round((not_serving_for or 0) / 60)))

        # 2. on the pod: disk, orphan EngineCore, stale snapshots (one ssh)
        ssh_str = ((state or {}).get(s.get("chat_pod_state_key", "chat_machine")) or {}).get("ssh") or ""
        if not ssh_str:
            out.append(Check("chat_pod", "warn", "no chat pod in state.json (chat_machine.ssh empty)"))
            return out
        probe = chat_pod_probe(ssh_str, list(s.get("chat_pod_model_dirs", ["/root"])), int(s.get("chat_ssh_timeout_s", 25)))
        if not probe.get("ok"):
            fail_for = self.since("chat_pod_ssh_fail", True, now)
            out.append(Check("chat_pod_ssh", "page" if (fail_for or 0) > 1800 else "warn",
                             f"chat pod {ssh_str} ssh failed for {common.fmt_age(fail_for)}: {probe.get('error')}"))
            return out
        self.since("chat_pod_ssh_fail", False, now)
        free = probe.get("free_gb")
        lvl = "page" if free is None or free < t["chat_disk_min_gb"] else "ok"
        out.append(Check("chat_disk", lvl,
                         f"chat pod {probe.get('df_path')} free {free} GB of {probe.get('size_gb')} GB"
                         + (f" < {t['chat_disk_min_gb']} GB" if lvl == "page" else ""),
                         free_gb=free, size_gb=probe.get("size_gb"), path=probe.get("df_path")))
        orphans = probe.get("orphans") or []
        out.append(Check("chat_orphan_engines", "page" if orphans else "ok",
                         (f"{len(orphans)} orphan VLLM::EngineCore on the chat pod (parent not a live vllm serve): "
                          + ", ".join(f"pid {p['pid']} ppid {p['ppid']}" for p in orphans)
                          + " — they hold GPU memory; kill them") if orphans
                         else f"{probe.get('n_engines')} EngineCore, all owned by vllm serve",
                         orphans=orphans, n_engines=probe.get("n_engines")))
        snaps = probe.get("snapshots") or []
        stale = [p for p in snaps if king_rev and king_rev not in p]
        out.append(Check("chat_snapshots", "warn" if len(stale) >= t["chat_snapshots_max"] else "ok",
                         f"{len(snaps)} king snapshot(s) on disk, {len(stale)} not the current king"
                         + (" — prune before the disk fills" if len(stale) >= t["chat_snapshots_max"] else ""),
                         snapshots=[p.rsplit('/', 1)[-1][:12] for p in snaps], stale=len(stale)))
        return out

    # ---- process hygiene
    def code_watch(self, procs: list[dict] | None, now: float) -> list[Check]:
        cw = self.cfg.code_watch
        rules = cw.get("rules", {})
        settle = float(rules.get("settle_min", 10)) * 60
        min_up = float(rules.get("min_uptime_min", 10)) * 60
        gap = float(rules.get("min_restart_gap_min", 30)) * 60
        budget = int(rules.get("max_restarts_per_tick", 2))
        out: list[Check] = []
        stale_rows = []
        self_restart = False
        for spec in cw.get("process", []):
            name = spec["name"]
            p = common.pm2_process(procs, name)
            if not p or p["status"] != "online" or not p.get("started_at"):
                continue
            newest, newest_file = 0.0, None
            for f in spec.get("files", []):
                try:
                    mt = (REPO / f).stat().st_mtime
                except OSError:
                    continue
                if mt > newest:
                    newest, newest_file = mt, f
            if newest <= p["started_at"]:
                continue
            if now - newest < settle:
                continue  # edit still settling
            stale_for = now - newest
            row = {"process": name, "file": newest_file, "code_age_min": round(stale_for / 60),
                   "uptime_min": round((now - p["started_at"]) / 60), "action": spec.get("action", "page")}
            if spec.get("action") == "restart" and now - p["started_at"] > min_up \
                    and now - float(self.own["restarted"].get(name, 0)) > gap and budget > 0:
                budget -= 1
                row["restarted"] = True
                if name == "affine-pipeline-health":
                    self_restart = True
                else:
                    self.restart(name)
                self.own["restarted"][name] = now
                out.append(Check(f"code_restart:{name}:{int(newest)}", "page",
                                 f"restarted pm2 {name}: it ran code older than {newest_file} "
                                 f"(changed {common.fmt_age(stale_for)} ago)"))
            else:
                stale_rows.append(row)
        if stale_rows:
            out.append(Check("stale_code", "page" if any(r["action"] == "page" for r in stale_rows) else "warn",
                             "processes running stale code: " + "; ".join(
                                 f"{r['process']} (behind {r['file']} by {common.fmt_age(r['code_age_min'] * 60)}, "
                                 f"action {r['action']})" for r in stale_rows), rows=stale_rows))
        self._self_restart = self_restart
        return out

    def restart(self, name: str) -> None:
        if self.dry_run:
            log(f"(dry-run) would pm2 restart {name}")
            return
        try:
            subprocess.run(["pm2", "restart", name], check=False, timeout=120,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            log(f"pm2 restart {name}")
        except (subprocess.SubprocessError, OSError) as e:
            log(f"pm2 restart {name} failed: {e!r}")

    # ---- tick
    def tick(self) -> dict:
        now = common.now()
        self._self_restart = False
        checks = self.run_checks(now)
        by_key = {c.key: c.as_dict() for c in checks}
        pages = [c for c in checks if c.level == "page"]
        warns = [c for c in checks if c.level == "warn"]

        # dedupe + post
        sent = self.own["alerts_sent"]
        new_lines = []
        for c in pages + warns:
            window = self.cfg.undeclared_dedupe_s if c.key.startswith("contract_undeclared") else self.cfg.dedupe_s
            if c.level == "warn" and not c.key.startswith(("contract_", "stale_code", "sources_toml", "fold_pm2", "chat_")):
                continue  # warnings are in the JSON / summary only
            dk = f"{c.level}:{c.key}"   # a warn that escalates to a page posts again
            last = sent.get(dk)
            if last is not None and now - float(last) < window:
                continue
            sent[dk] = now
            new_lines.append(f"{'PAGE' if c.level == 'page' else 'warn'} {c.key}: {c.detail}")
        # recovered
        active_prev = set(self.own["active"])
        active_now = {c.key for c in pages if not c.key.startswith(("contract_event", "code_restart", "fold_failed"))}
        recovered = sorted(k for k in active_prev - active_now if not k.startswith(("contract_event", "code_restart", "fold_failed")))
        self.own["active"] = {k: now for k in active_now}
        for line in new_lines:
            self.post(line)
        if recovered and self.cfg.recovery_lines:
            self.post("recovered: " + ", ".join(recovered))
        # hourly summary
        if now - float(self.own.get("last_summary", 0)) >= self.cfg.summary_s:
            self.own["last_summary"] = now
            self.post(self.summary_line(checks, pages))
        for k in [k for k, v in sent.items() if now - float(v) > 3 * 86400]:
            sent.pop(k, None)

        out = {"at": common.now_iso(), "ok": not pages, "n_page": len(pages), "n_warn": len(warns),
               "alerts_active": [f"{c.key}: {c.detail}" for c in pages],
               "warnings": [f"{c.key}: {c.detail}" for c in warns],
               "checks": by_key, "summary": self.summary_line(checks, pages, short=True)}
        common.atomic_write_json(self.cfg.paths["out_json"], out)
        common.atomic_write_json(self.cfg.paths["own_state"], self.own)
        log(f"{'OK' if not pages else 'ALERT ' + str(len(pages))} warn={len(warns)} posted={len(new_lines)} — {out['summary']}")
        if self._self_restart:
            self.restart("affine-pipeline-health")
        return out

    def summary_line(self, checks: list[Check], pages: list[Check], *, short: bool = False) -> str:
        v = {c.key: c.values for c in checks}
        ce = v.get("corpus_epoch_age", {})
        cu = v.get("curriculum_age", {})
        ks = v.get("king_seat", {})
        tr = v.get("teacher_replicas", {})
        vc = v.get("verdict_cadence", {})
        bq = v.get("bench_queue") or v.get("bench_queue_held") or {}
        pd = v.get("pods", {})
        ch = v.get("chat_endpoint", {})
        parts = [
            f"epoch {ce.get('epoch')} {ce.get('age_h', '?')}h",
            f"curriculum {cu.get('mode')} {cu.get('age_h', '?')}h{' FROZEN' if cu.get('frozen') else ''}",
            f"king seat {'ok' if 'pod' in ks else 'EMPTY ' + str(ks.get('empty_min', '?')) + 'm'}",
            f"swarm {tr.get('healthy', '?')}/{tr.get('backends', '?')}",
            f"verdict {vc.get('age_h', '?')}h q{vc.get('queue', '?')}",
            f"bench ${bq.get('spent_usd', '?')}/{bq.get('cap_usd', '?')}",
            f"pods {pd.get('n_ours', '?')} ${pd.get('ours_usd_h', '?')}/h",
            f"chat {'ok' if ch.get('state') == 'serving' and 'not_serving_min' not in ch else 'NOT serving ' + str(ch.get('state'))}",
        ]
        head = "all clear" if not pages else f"{len(pages)} ALERT(S): " + "; ".join(c.key for c in pages)
        line = f"{head} — " + ", ".join(parts)
        return line if short else f"hourly: {line}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "health.toml"))
    ap.add_argument("--interval", type=int, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print alerts, post nothing, restart nothing, freeze nothing")
    args = ap.parse_args()
    cfg = Config(Path(args.config))
    interval = args.interval or cfg.interval_s
    mon = Monitor(cfg, dry_run=args.dry_run)
    log(f"start interval={interval}s dry_run={args.dry_run}")
    while True:
        try:
            out = mon.tick()
            if args.once:
                print(json.dumps(out, indent=1, default=str))
                return 0
        except Exception as e:  # a monitor must never die on one bad tick
            log(f"tick failed: {e!r}")
            if args.once:
                raise
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
