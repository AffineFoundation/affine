#!/usr/bin/env python
"""affine-evalwatch — is the duel pipeline actually moving?

Every tick (pm2 `affine-evalwatch`, --interval seconds) this reads, without
writing anything on the production side:

  * affine/state/state.json      queue, in_flight, last_weights_at, machines
  * affine/state/history.jsonl   last verdict time
  * affine/logs/validator.err.log  `processing` / `requeued` lines (dispatch
                                 age, repeated infra faults)
  * 127.0.0.1:9000/health        eval pod (through the validator's tunnel)
  * pm2 jlist                    validator status + restart count
  * Lium /pods                   the eval / bench / chat pods exist + RUNNING

and writes one observation file, affine/state/evalwatch.json. Conditions
that hold become alerts; each condition key posts ONE Discord line per
`alert_dedupe_h` hours. `--dry-run` logs the lines instead of posting.

Conditions (thresholds in evalwatch.toml):
  inflight_age      in-flight duel older than 60 min since its dispatch
  queue_depth       queue depth > 8
  eval_health       /health unreachable or ok=false
  eval_busy         /health busy=true continuously for > 90 min
  pod_missing       a machine's Lium pod is gone or not RUNNING
  validator_restart pm2 restart count of affine-validator changed
  validator_down    affine-validator not online in pm2
  weights_stale     last set_weights older than 2 h
  repeat_fault      the same infra fault requeued the same challenge >= 3x
                    within 2 h (the 2026-09-11 king_launch_failed loop)

Usage:
    python evalwatch.py --once --dry-run     # one pass, print alerts
    python evalwatch.py --interval 300       # daemon (posting enabled)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import tomllib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))
import lium_api  # noqa: E402

VALIDATOR_ENV = Path.home() / ".affine-validator.env"
REPO_ENV = REPO / ".env"

PROCESSING_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ affine\.validator INFO processing (chal-\d+)")
REQUEUE_RE = re.compile(
    r"^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d),\d+ affine\.state WARNING requeued (chal-\d+) at front "
    r"\(retry \d+, counted=\w+\) due to (.*)$")
PM2_PREFIX_RE = re.compile(r"^\d{4}-\d\d-\d\dT[\d:]+: ")


def log(msg: str) -> None:
    print(f"[evalwatch] {datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}",
          flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(str(s))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def parse_log_ts(s: str) -> float:
    return datetime.strptime(s, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=timezone.utc).timestamp()


def env_file_value(name: str) -> str:
    """One value from the operator env snapshots (validator env first, then
    the repo .env). Missing file / key -> ""."""
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


def fault_key(reason: str) -> str:
    """Collapse a requeue reason to its fault class so repeats compare equal:
    `eval server infra fault [king_launch_failed]: king r2://...` ->
    `king_launch_failed`; `duel stream broke: ...` -> `duel stream broke`."""
    m = re.search(r"\[([a-z_]+)\]", reason)
    if m:
        return m.group(1)
    head = reason.split(":", 1)[0].strip()
    return re.sub(r"\s+", " ", head)[:60] or "unknown"


class Config:
    def __init__(self, path: Path):
        d = tomllib.loads(path.read_text())
        w, p, e, dc = d["watch"], d["paths"], d["eval"], d["discord"]
        self.interval_s = int(w.get("interval_s", 300))
        self.inflight_max_s = float(w.get("inflight_max_min", 60)) * 60
        self.queue_depth_max = int(w.get("queue_depth_max", 8))
        self.busy_max_s = float(w.get("busy_unchanged_max_min", 90)) * 60
        self.weights_max_s = float(w.get("weights_max_min", 120)) * 60
        self.repeat_min_count = int(w.get("repeat_fault_min_count", 3))
        self.repeat_window_s = float(w.get("repeat_fault_window_min", 120)) * 60
        self.dedupe_s = float(w.get("alert_dedupe_h", 6)) * 3600
        self.log_tail_bytes = int(w.get("log_tail_bytes", 4_000_000))
        self.state_dir = (HERE / p["state_dir"]).resolve()
        self.validator_log = (HERE / p["validator_log"]).resolve()
        self.out_json = (HERE / p["out_json"]).resolve()
        self.own_state = (HERE / p["own_state"]).resolve()
        self.health_url = str(e["health_url"])
        self.token_env = str(e.get("token_env", "AFFINE_EVAL_TOKEN"))
        self.pm2_validator = str(e.get("pm2_validator", "affine-validator"))
        self.machines = list(e.get("machines", []))
        self.discord_enabled = bool(dc.get("enabled", True))
        self.discord_channel = str(dc.get("channel_id", ""))
        self.discord_token_env = str(dc.get("token_env", "DISCORD_BOT_TOKEN_ARBOS_BITTENSOR"))
        self.discord_prefix = str(dc.get("prefix", "[evalwatch]"))


# -- collectors -------------------------------------------------------------------

def read_state(cfg: Config) -> dict | None:
    try:
        return json.loads((cfg.state_dir / "state.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        log(f"state.json unreadable: {e!r}")
        return None


def last_verdict(cfg: Config) -> dict:
    """Newest terminal history row (verdict / crowned / failed)."""
    path = cfg.state_dir / "history.jsonl"
    last: dict = {}
    try:
        with path.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 2_000_000))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return last
    for line in tail.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("event") in ("verdict", "crowned", "failed") and r.get("at"):
            last = {"challenge_id": r.get("challenge_id"), "event": r.get("event"),
                    "at": r.get("at")}
    return last


def scan_validator_log(cfg: Config) -> tuple[dict[str, float], list[tuple[float, str, str]]]:
    """(last `processing` time per challenge, requeue events (ts, cid, fault))
    from the log tail. pm2 prefixes lines with `<iso>: `; strip it."""
    processing: dict[str, float] = {}
    requeues: list[tuple[float, str, str]] = []
    try:
        with cfg.validator_log.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - cfg.log_tail_bytes))
            tail = fh.read().decode("utf-8", errors="replace")
    except OSError as e:
        log(f"validator log unreadable: {e!r}")
        return processing, requeues
    for line in tail.splitlines():
        line = PM2_PREFIX_RE.sub("", line)
        m = PROCESSING_RE.match(line)
        if m:
            processing[m.group(2)] = parse_log_ts(m.group(1))
            continue
        m = REQUEUE_RE.match(line)
        if m:
            requeues.append((parse_log_ts(m.group(1)), m.group(2), fault_key(m.group(3))))
    return processing, requeues


def eval_health(cfg: Config) -> dict:
    token = env_file_value(cfg.token_env)
    headers = {"X-Affine-Token": token} if token else {}
    try:
        r = requests.get(cfg.health_url, headers=headers, timeout=15)
        if r.status_code != 200:
            return {"reachable": True, "ok": False, "http": r.status_code}
        d = r.json()
        eng = d.get("engine") or {}
        return {
            "reachable": True, "ok": bool(d.get("ok")), "busy": bool(d.get("busy")),
            "busy_kind": d.get("busy_kind"), "role": d.get("role"),
            "versions": d.get("versions"),
            "corpus_epoch": (d.get("corpus") or {}).get("corpus_epoch"),
            "engine_ready": {k: bool((v or {}).get("ready")) for k, v in eng.items()},
        }
    except (requests.RequestException, ValueError) as e:
        return {"reachable": False, "ok": False, "error": type(e).__name__}


def pm2_validator(cfg: Config) -> dict:
    try:
        raw = subprocess.check_output(["pm2", "jlist"], text=True,
                                      stderr=subprocess.DEVNULL, timeout=30)
        for p in json.loads(raw):
            if p.get("name") == cfg.pm2_validator:
                env = p.get("pm2_env") or {}
                return {"found": True, "status": env.get("status"),
                        "restarts": int(env.get("restart_time") or 0),
                        "pm_uptime": env.get("pm_uptime")}
        return {"found": False}
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as e:
        return {"found": None, "error": repr(e)[:200]}


def lium_pods(cfg: Config, state: dict | None) -> dict:
    """{machine_key: {id, status}} for the configured machines; status None
    when the Lium listing failed (never treated as 'gone')."""
    out: dict = {}
    if not state:
        return out
    try:
        pods = lium_api.pods(lium_api.session())
    except (SystemExit, requests.RequestException) as e:
        log(f"lium listing failed: {e!r}")
        pods = None
    by_name = ({lium_api.pod_name(p): str(p.get("status", "")).upper()
                for p in pods} if pods is not None else None)
    for key in cfg.machines:
        m = state.get(key) or {}
        pid = m.get("id")
        if not pid or m.get("provider") not in (None, "lium"):
            out[key] = {"id": pid, "status": "n/a"}
            continue
        if by_name is None:
            out[key] = {"id": pid, "status": None}
        else:
            out[key] = {"id": pid, "status": by_name.get(pid, "MISSING")}
    return out


# -- evaluation -----------------------------------------------------------------------

class Watch:
    def __init__(self, cfg: Config, dry_run: bool):
        self.cfg = cfg
        self.dry_run = dry_run
        self.own = self._load_own()

    def _load_own(self) -> dict:
        try:
            return json.loads(self.cfg.own_state.read_text())
        except (OSError, json.JSONDecodeError):
            return {"alerts_sent": {}, "busy_since": None, "restarts": None}

    def _save_own(self) -> None:
        self.cfg.own_state.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cfg.own_state.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(self.own, indent=1))
        tmp.replace(self.cfg.own_state)

    def tick(self) -> dict:
        cfg = self.cfg
        now = time.time()
        state = read_state(cfg)
        processing, requeues = scan_validator_log(cfg)
        health = eval_health(cfg)
        pm2 = pm2_validator(cfg)
        pods = lium_pods(cfg, state)
        verdict = last_verdict(cfg)

        alerts: list[tuple[str, str]] = []  # (dedupe key, text)
        obs: dict = {"at": now_iso(), "health": health, "pm2": pm2, "pods": pods,
                     "last_verdict": verdict}

        if state is None:
            alerts.append(("state_unreadable", "state.json unreadable"))
        else:
            queue = state.get("queue") or []
            inflight = state.get("in_flight") or None
            obs["queue_depth"] = len(queue)
            obs["queue_head"] = [e.get("challenge_id") for e in queue[:5]]
            obs["in_flight"] = inflight.get("challenge_id") if inflight else None
            if len(queue) > cfg.queue_depth_max:
                alerts.append(("queue_depth",
                               f"queue depth {len(queue)} > {cfg.queue_depth_max} "
                               f"(head {obs['queue_head'][:3]})"))
            if inflight:
                cid = inflight["challenge_id"]
                started = processing.get(cid) or parse_iso(inflight.get("queued_at"))
                age = (now - started) if started else None
                obs["in_flight_age_min"] = round(age / 60, 1) if age else None
                if age is not None and age > cfg.inflight_max_s:
                    alerts.append((f"inflight_age:{cid}",
                                   f"{cid} in flight for {age/60:.0f} min "
                                   f"(> {cfg.inflight_max_s/60:.0f})"))
            wts = parse_iso(state.get("last_weights_at"))
            obs["weights_age_min"] = round((now - wts) / 60, 1) if wts else None
            if wts is None or now - wts > cfg.weights_max_s:
                age_txt = f"{(now - wts)/3600:.1f} h" if wts else "never"
                alerts.append(("weights_stale", f"last set_weights {age_txt} ago"))

        # repeated infra fault on one challenge inside the window
        window = [(ts, cid, f) for ts, cid, f in requeues if now - ts <= cfg.repeat_window_s]
        counts = Counter((cid, f) for _, cid, f in window)
        last_at = {}
        for ts, cid, f in window:
            last_at[(cid, f)] = max(ts, last_at.get((cid, f), 0.0))
        obs["requeues_in_window"] = {f"{cid}:{f}": n for (cid, f), n in counts.items()}
        for (cid, fault), n in counts.items():
            if n >= cfg.repeat_min_count:
                last_txt = datetime.fromtimestamp(last_at[(cid, fault)], timezone.utc).strftime("%H:%M")
                alerts.append((f"repeat_fault:{cid}:{fault}",
                               f"{cid} requeued {n}x with `{fault}` in the last "
                               f"{cfg.repeat_window_s/3600:.0f} h (last {last_txt} UTC) — "
                               f"the eval pod is not producing verdicts"))

        # eval pod health / busy stuck
        if not health.get("reachable") or not health.get("ok"):
            what = "unreachable" if not health.get("reachable") else "ok=false"
            why = health.get("error") or health.get("http")
            alerts.append(("eval_health",
                           f"eval /health {what}" + (f" ({why})" if why else "")))
            self.own["busy_since"] = None
        elif health.get("busy"):
            if not self.own.get("busy_since"):
                self.own["busy_since"] = now
            busy_for = now - float(self.own["busy_since"])
            obs["busy_for_min"] = round(busy_for / 60, 1)
            if busy_for > cfg.busy_max_s:
                alerts.append(("eval_busy",
                               f"eval pod busy={health.get('busy_kind')} for "
                               f"{busy_for/60:.0f} min without going idle"))
        else:
            self.own["busy_since"] = None

        # Lium pods
        for key, p in pods.items():
            st = p.get("status")
            if st is None or st == "n/a":
                continue
            if st != "RUNNING":
                alerts.append((f"pod_missing:{p.get('id')}",
                               f"Lium pod {p.get('id')} ({key}) is {st}"))

        # validator process
        if pm2.get("found") is False or (pm2.get("found") and pm2.get("status") != "online"):
            alerts.append(("validator_down",
                           f"pm2 {cfg.pm2_validator} is {pm2.get('status') or 'missing'}"))
        if pm2.get("found") and pm2.get("restarts") is not None:
            prev = self.own.get("restarts")
            if prev is not None and pm2["restarts"] != prev:
                alerts.append((f"validator_restart:{pm2['restarts']}",
                               f"{cfg.pm2_validator} restart count {prev} -> {pm2['restarts']}"))
            self.own["restarts"] = pm2["restarts"]

        obs["alerts_active"] = [t for _, t in alerts]
        new = self._dedupe(alerts, now)
        obs["alerts_posted"] = [t for _, t in new]
        if new:
            self.notify("; ".join(t for _, t in new))
        else:
            log(f"ok queue={obs.get('queue_depth')} in_flight={obs.get('in_flight')} "
                f"age={obs.get('in_flight_age_min')}m busy={health.get('busy')} "
                f"weights_age={obs.get('weights_age_min')}m active={len(alerts)}")
        self._write_out(obs)
        self._save_own()
        return obs

    def _dedupe(self, alerts: list[tuple[str, str]], now: float) -> list[tuple[str, str]]:
        sent: dict = self.own.setdefault("alerts_sent", {})
        out = []
        for key, text in alerts:
            last = sent.get(key)
            if last is not None and now - float(last) < self.cfg.dedupe_s:
                continue
            sent[key] = now
            out.append((key, text))
        # forget keys older than two dedupe windows
        for key in [k for k, v in sent.items() if now - float(v) > 2 * self.cfg.dedupe_s]:
            sent.pop(key, None)
        return out

    def _write_out(self, obs: dict) -> None:
        try:
            self.cfg.out_json.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.cfg.out_json.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(obs, indent=1, default=str))
            tmp.replace(self.cfg.out_json)
        except OSError as e:
            log(f"could not write {self.cfg.out_json}: {e!r}")

    def notify(self, text: str) -> None:
        line = f"{self.cfg.discord_prefix} {text}"
        if self.dry_run or not self.cfg.discord_enabled or not self.cfg.discord_channel:
            log(f"ALERT (not posted): {line}")
            return
        token = env_file_value(self.cfg.discord_token_env)
        if not token:
            log(f"ALERT (no discord token): {line}")
            return
        log(f"ALERT: {line}")
        try:
            r = requests.post(
                f"https://discord.com/api/v10/channels/{self.cfg.discord_channel}/messages",
                headers={"Authorization": f"Bot {token}"},
                json={"content": line[:1900]}, timeout=20)
            if r.status_code >= 300:
                log(f"discord HTTP {r.status_code}")
        except requests.RequestException as e:
            log(f"discord post failed: {e!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Affine eval-pipeline health monitor")
    ap.add_argument("--config", default=str(HERE / "evalwatch.toml"))
    ap.add_argument("--interval", type=int, default=None,
                    help="seconds between ticks (default: toml watch.interval_s)")
    ap.add_argument("--once", action="store_true", help="one tick, then exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="log alert lines instead of posting to Discord")
    args = ap.parse_args()
    cfg = Config(Path(args.config))
    interval = args.interval or cfg.interval_s
    watch = Watch(cfg, dry_run=args.dry_run)
    log(f"start interval={interval}s dry_run={args.dry_run} state_dir={cfg.state_dir}")
    while True:
        try:
            watch.tick()
        except Exception as e:  # a monitor must never die on one bad tick
            log(f"tick failed: {e!r}")
        if args.once:
            return
        time.sleep(interval)


if __name__ == "__main__":
    main()
