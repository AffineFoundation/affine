#!/usr/bin/env python
"""affine-pod-reaper — every Lium pod we own has an owner and a lifetime.

Each tick (pm2 `affine-pod-reaper`, 15 min):
  1. list every pod in the Lium account; split OURS (ops/pods/pods.toml
     [scope].prefixes) from FOREIGN (listed in the report, never touched);
  2. adopt pods that controllers already track (validator state.json
     machines, kingctl, swarm manager, kingpod) into the registry so a rent
     path that forgot to register still has an owner;
  3. for every pod of ours decide:
       registered, lifetime passed, owner dead   -> RELEASE, page with $ saved
       registered, lifetime passed, owner alive  -> warn
       registered, owner dead inside lifetime    -> warn after owner_dead_warn_h
       registered as released, still listed      -> RELEASE again (zombie), warn
       unregistered                              -> page after unregistered_page_min,
                                                    RELEASE after unregistered_release_min
  4. append every action to state/ledger.jsonl, write
     affine/state/pods/reaper.json for the health monitor / kingboard.

`--once --dry-run` prints decisions and posts nothing; `[reaper].enforce =
false` or a file ops/pods/state/DRY_RUN keeps the decisions but never calls
`lium rm`.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
import sys
import time
import tomllib
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "health"))
sys.path.insert(0, str(REPO / "ops" / "teacher-swarm"))
import common  # noqa: E402
import lium_api  # noqa: E402
import registry  # noqa: E402

TAG = "pod-reaper"


def log(msg: str) -> None:
    common.log(TAG, msg)


# -- owner liveness -----------------------------------------------------------------

def _proc_cmdlines() -> list[str]:
    try:
        out = subprocess.run(["ps", "-eo", "args="], capture_output=True, text=True,
                             timeout=20).stdout
    except (subprocess.SubprocessError, OSError):
        return []
    return out.splitlines()


def passlog_alive(pod: str, rel_dir: str, cmdlines: list[str]) -> bool:
    """A benchsuite pass owns the pod while it runs. Alive when any of:
    * a live process names the pod on its command line;
    * a `pass-<run>.pid` in the state dir holds a LIVE pid and that run's
      log (`<run>.log`) names the pod (2026-09-24 21:25: a fast pass is
      silent for hours while a long cell runs — Albedo's chat1 was released
      mid-BFCL, reign 12's fenced @4h250 at 499/500 — so the pid, not the
      log's mtime, is the liveness signal);
    * (legacy) a pass log naming the pod written in the last 3 h without a
      .exit sibling."""
    if any(pod in c for c in cmdlines):
        return True
    d = (REPO / rel_dir).resolve()
    for pf in d.glob("pass-*.pid"):
        try:
            pid = int(pf.read_text().strip() or 0)
        except (OSError, ValueError):
            continue
        if pid <= 0 or not common.pid_alive(pid):
            continue
        run_id = pf.stem[len("pass-"):]
        lp = d / f"{run_id}.log"
        try:
            if lp.exists() and pod in lp.read_text(errors="replace"):
                return True
        except OSError:
            continue
    cutoff = common.now() - 3 * 3600
    for lp in d.glob("*.log"):
        try:
            if lp.stat().st_mtime < cutoff or lp.with_suffix(".exit").exists():
                continue
            if pod in lp.read_text(errors="replace"):
                return True
        except OSError:
            continue
    return False


def owner_alive(pod: str, owner: str, procs: list[dict] | None,
                cmdlines: list[str]) -> bool | None:
    """True / False, None = cannot tell (pm2 unreadable) -> treated as alive."""
    kind, _, arg = str(owner or "unknown").partition(":")
    if kind == "manual":
        return True
    if kind == "pm2":
        return common.pm2_online(procs, arg)
    if kind == "pid":
        try:
            return common.pid_alive(int(arg))
        except ValueError:
            return False
    if kind == "proc":
        return any(arg and arg in c for c in cmdlines)
    if kind == "passlog":
        return passlog_alive(pod, arg or "ops/benchsuite/state", cmdlines)
    return False


# -- adoption -------------------------------------------------------------------------

def adopt(cfg: dict, listed: set[str]) -> int:
    """Register (source adopted:<name>) every pod a controller's state file
    claims and that is in the Lium listing. Returns the number adopted."""
    reg = registry.load(cfg)
    n = 0
    for st in cfg.get("static", []):
        for name in listed:
            if not fnmatch.fnmatch(name, st["glob"]):
                continue
            rec = reg.get(name)
            if rec and not rec.get("released_at") and rec.get("source") == "static":
                continue
            registry.register(name, purpose=st.get("purpose"), owner=st.get("owner"),
                              expected_hours=float(st.get("expected_hours", 0) or 0),
                              meta={"note": st.get("note", "")}, source="static")
            n += 1
    for a in cfg.get("adopt", []):
        path = (HERE / a["file"]).resolve()
        data = common.read_json(path)
        if data is None:
            continue
        names: dict[str, dict] = {}
        kind = a["kind"]
        if kind == "validator_machines":
            for key in ("eval_machine", "bench_machine", "chat_machine"):
                m = data.get(key) or {}
                if m.get("id") and m.get("provider") in (None, "lium"):
                    names[str(m["id"])] = {"purpose": key, "owner": "pm2:affine-validator",
                                           "expected_hours": 0.0,
                                           "price": m.get("price_per_hour")}
        elif kind == "dict_pods":
            for name, mem in (data.get("pods") or {}).items():
                names[str(name)] = {"purpose": a.get("purpose"), "owner": a.get("owner"),
                                    "expected_hours": a.get("expected_hours"),
                                    "price": (mem or {}).get("price")}
        elif kind == "flat_pods":
            for name, mem in data.items():
                if isinstance(mem, dict):
                    names[str(name)] = {"purpose": a.get("purpose"), "owner": a.get("owner"),
                                        "expected_hours": a.get("expected_hours"),
                                        "price": mem.get("price")}
        elif kind == "kingpod_pods":
            for name, mem in data.items():
                if isinstance(mem, dict) and mem.get("state") != "released":
                    names[str(name)] = {"purpose": a.get("purpose"), "owner": a.get("owner"),
                                        "expected_hours": a.get("expected_hours"),
                                        "price": mem.get("price"), "base_url": mem.get("base_url")}
        for name, info in names.items():
            if name not in listed:
                continue
            rec = reg.get(name)
            if rec and not rec.get("released_at") and \
                    registry.SOURCE_RANK.get(str(rec.get("source", "auto")).split(":")[0], 1) >= 2:
                continue
            registry.register(name, purpose=info.get("purpose"), owner=info.get("owner"),
                              meta=({"base_url": info["base_url"]} if info.get("base_url") else None),
                              expected_hours=(float(info["expected_hours"])
                                              if info.get("expected_hours") is not None else None),
                              price_usd_h=info.get("price"), source=f"adopted:{a['name']}")
            n += 1
    return n


# -- the tick ---------------------------------------------------------------------------

class Reaper:
    def __init__(self, cfg: dict, *, dry_run: bool):
        self.cfg = cfg
        self.dry_run = dry_run
        r = cfg["reaper"]
        self.enforce = bool(r.get("enforce", True)) and not dry_run \
            and not (HERE / "state" / "DRY_RUN").exists()
        self.own_path = (HERE / cfg["paths"]["own_state"]).resolve()
        self.report_path = (HERE / cfg["paths"]["report"]).resolve()
        self.own = common.read_json(self.own_path, default={}) or {}
        self.own.setdefault("first_seen", {})
        self.own.setdefault("warned", {})
        self.own.setdefault("listing_failures", 0)
        self.sess = lium_api.session()

    # -- helpers
    def warn(self, key: str, text: str, *, dedupe_h: float | None = None) -> None:
        window = (dedupe_h if dedupe_h is not None else float(self.cfg["reaper"].get("warn_dedupe_h", 6))) * 3600
        last = self.own["warned"].get(key)
        if last is not None and common.now() - float(last) < window:
            return
        self.own["warned"][key] = common.now()
        self.page(text)

    def page(self, text: str) -> None:
        dc = self.cfg["discord"]
        log(f"PAGE: {text}")
        if not dc.get("enabled", True):
            return
        common.discord_post(text, channel=str(dc["channel_id"]), token_env=dc["token_env"],
                            dry_run=self.dry_run, prefix=dc.get("prefix", f"[{TAG}]"))

    def ledger(self, row: dict) -> None:
        path = registry.ledger_path(self.cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as fh:
            fh.write(json.dumps({"at": common.now_iso(), **row}, sort_keys=True, default=str) + "\n")

    def release(self, pod: dict, name: str, reason: str, rec: dict | None) -> dict:
        price = lium_api.pod_price(pod) or float((rec or {}).get("price_usd_h") or 0)
        left_h = self._hours_left(pod)
        saved = round(price * left_h, 2)
        prefix = registry.matched_prefix(name, self.cfg) or name
        action = {"pod": name, "reason": reason, "price_usd_h": price,
                  "hours_left_est": round(left_h, 1), "saved_usd_est": saved,
                  "age_h": round(self._age_h(pod), 1), "enforced": self.enforce}
        if self.enforce:
            try:
                ok = lium_api.remove(name, prefix)
            except (ValueError, subprocess.SubprocessError) as e:
                ok = False
                action["error"] = repr(e)[:200]
            action["released"] = ok
            if ok:
                registry.release(name, reason=f"reaper: {reason}", by=TAG)
        else:
            action["released"] = None
            action["note"] = "not enforced (dry-run / enforce=false / DRY_RUN file)"
        self.ledger({"action": "release", **action})
        verb = "released" if action.get("released") else ("WOULD release" if not self.enforce else "FAILED to release")
        self.page(f"{verb} `{name}` — {reason}; ${price:.2f}/h, ≈ ${saved:.0f} saved "
                  f"(≈ {left_h:.0f} h left), age {action['age_h']} h")
        return action

    @staticmethod
    def _created(pod: dict) -> float | None:
        return common.parse_iso(pod.get("created_at"))

    def _age_h(self, pod: dict) -> float:
        c = self._created(pod)
        return (common.now() - c) / 3600 if c else 0.0

    # -- lifetime rules (2026-09-22): a registered pod is released only when
    #    (a) its owner wrote the completion marker (registry `done`), or
    #    (b) its heartbeat is stale > heartbeat_stale_min AND its owner is DEAD
    #        AND its vLLM has served nothing for idle_min (no signal = no
    #        release; owner alive or unknown = page, never release — 2026-09-24), or
    #    (c) it reaches expected_hours — a hard CEILING that pages at
    #        ceiling_page_frac first. Fixed lifetimes alone killed six env
    #        serving boxes overnight while their drivers still used them.
    def feed_heartbeats(self, by_name: dict[str, dict]) -> int:
        """Owner heartbeats the reaper collects itself: the env-backfill driver
        pods' /drivers list every live driver with the serving box it talks
        to and how fresh its log is; a fresh driver = a heartbeat for that
        box. Returns the number of pods touched."""
        hb = self.cfg.get("heartbeat", {})
        ptr_path = (HERE / hb.get("driver_pointer", "../../affine/state/pods/backfill_driver.json")).resolve()
        ptr = common.read_json(ptr_path, default={}) or {}
        pods = ([ptr] if ptr.get("health_url") else []) + list(ptr.get("secondaries") or [])
        fresh_s = float(hb.get("driver_fresh_min", 10)) * 60
        # host:port -> pod name for every listed pod of ours
        addr: dict[str, str] = {}
        for name, pod in by_name.items():
            ssh = lium_api.parse_ssh(pod)
            if not ssh:
                continue
            for ext in lium_api.data_ports(pod).values():
                addr[f"{ssh[0]}:{ext}"] = name
        touched = 0
        for dp in pods:
            url = str(dp.get("health_url", "")).replace("/health", "/drivers")
            try:
                drivers = requests.get(url, timeout=10).json().get("drivers") or []
            except (requests.RequestException, ValueError):
                continue
            for d in drivers:
                if d.get("complete") or d.get("log_age_s") is None or float(d["log_age_s"]) > fresh_s:
                    continue
                base = str(d.get("base_url") or "")
                hp = base.split("://", 1)[-1].split("/", 1)[0]
                name = addr.get(hp)
                if name:
                    registry.touch(name)
                    touched += 1
        return touched

    def idle_seconds(self, name: str, pod: dict, rec: dict) -> float | None:
        """Seconds since the pod's vLLM last served a request (its /metrics
        counters moved or a request was running), or None when no metrics
        endpoint answers."""
        cands = []
        base = ((rec.get("meta") or {}).get("base_url") or "").rstrip("/")
        if base:
            cands.append(base[:-3] if base.endswith("/v1") else base)
        ssh = lium_api.parse_ssh(pod)
        if ssh:
            for ext in list(lium_api.data_ports(pod).values())[:3]:
                cands.append(f"http://{ssh[0]}:{ext}")
        text = None
        for c in cands:
            try:
                r = requests.get(f"{c}/metrics", timeout=5)
                if r.status_code == 200 and "vllm:" in r.text:
                    text = r.text
                    break
            except requests.RequestException:
                continue
        if text is None:
            return None
        running = total = 0.0
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            key, _, val = line.rpartition(" ")
            try:
                v = float(val)
            except ValueError:
                continue
            if key.startswith("vllm:num_requests_running"):
                running += v
            elif key.startswith(("vllm:request_success_total", "vllm:prompt_tokens_total")):
                total += v
        st = self.own.setdefault("activity", {}).setdefault(name, {})
        now = common.now()
        if running > 0 or st.get("total") != total or "changed_at" not in st:
            st["changed_at"] = now
        st["total"] = total
        st["running"] = running
        st["seen_at"] = now
        return now - float(st["changed_at"])

    def _hours_left(self, pod: dict) -> float:
        rem = common.parse_iso(pod.get("removal_scheduled_at"))
        if rem and rem > common.now():
            return (rem - common.now()) / 3600
        return float(self.cfg["reaper"].get("assume_hours_left", 24))

    # -- tick
    def tick(self) -> dict:
        r = self.cfg["reaper"]
        pods = lium_api.pods(self.sess)
        report: dict = {"at": common.now_iso(), "enforce": self.enforce, "ours": [],
                        "foreign": {"count": 0, "usd_h": 0.0, "names": []},
                        "actions": [], "problems": [], "listing_ok": pods is not None}
        if pods is None:
            self.own["listing_failures"] = int(self.own.get("listing_failures", 0)) + 1
            log(f"lium /pods failed ({self.own['listing_failures']}x); skipping tick")
            if self.own["listing_failures"] >= int(r.get("listing_fail_page_ticks", 3)):
                self.warn("listing_failed", f"Lium /pods listing failed "
                          f"{self.own['listing_failures']} ticks in a row — pods cannot be audited")
            self._finish(report)
            return report
        self.own["listing_failures"] = 0
        by_name = {lium_api.pod_name(p): p for p in pods}
        listed = set(by_name)
        adopted = adopt(self.cfg, listed)
        if adopted:
            log(f"adopted {adopted} pod(s) from controller state files")
        try:
            fed = self.feed_heartbeats(by_name)
            if fed:
                log(f"heartbeats from the env-backfill drivers: {fed} pod(s)")
        except Exception as e:  # noqa: BLE001 - a feeder failure must not stop the audit
            log(f"heartbeat feed failed: {e!r}")
        reg = registry.load(self.cfg)
        procs = common.pm2_jlist()
        cmdlines = _proc_cmdlines()
        now = common.now()
        ours_usd = 0.0

        for name in sorted(listed):
            pod = by_name[name]
            price = lium_api.pod_price(pod)
            if not registry.is_ours(name, self.cfg):
                report["foreign"]["count"] += 1
                report["foreign"]["usd_h"] += price
                report["foreign"]["names"].append(name)
                continue
            ours_usd += price
            age_h = self._age_h(pod)
            rec = reg.get(name)
            row = {"name": name, "status": pod.get("status"), "price_usd_h": price,
                   "age_h": round(age_h, 1), "removal_at": pod.get("removal_scheduled_at"),
                   "machine": (pod.get("executor") or {}).get("machine_name") or pod.get("gpu_name")}
            if rec is None:
                first = self.own["first_seen"].setdefault(name, now)
                unseen_min = (now - float(first)) / 60
                row.update({"registered": False, "unregistered_min": round(unseen_min)})
                if unseen_min >= float(r.get("unregistered_release_min", 90)):
                    row["decision"] = "release_unregistered"
                    report["actions"].append(self.release(pod, name, "unregistered pod of ours "
                                                          f"({unseen_min:.0f} min without an owner)", None))
                elif unseen_min >= float(r.get("unregistered_page_min", 30)):
                    row["decision"] = "page_unregistered"
                    self.warn(f"unregistered:{name}",
                              f"unregistered pod `{name}` (${price:.2f}/h, age {age_h:.1f} h) — "
                              f"register it (`python ops/pods/registry.py register {name} --purpose … "
                              f"--owner … --hours …`) or it is released at "
                              f"{r.get('unregistered_release_min', 90)} min")
                    report["problems"].append(f"unregistered {name}")
                else:
                    row["decision"] = "grace"
                report["ours"].append(row)
                continue
            self.own["first_seen"].pop(name, None)
            row.update({"registered": True, "purpose": rec.get("purpose"), "owner": rec.get("owner"),
                        "expected_hours": rec.get("expected_hours"), "source": rec.get("source")})
            if rec.get("released_at"):
                since_min = (now - float(rec["released_at"])) / 60
                row["decision"] = "zombie" if since_min >= float(r.get("zombie_min", 15)) else "releasing"
                if row["decision"] == "zombie":
                    report["actions"].append(self.release(pod, name, "marked released "
                                                          f"{since_min:.0f} min ago but still listed (lium rm failed?)", rec))
                    report["problems"].append(f"zombie {name}")
                report["ours"].append(row)
                continue
            alive = owner_alive(name, rec.get("owner", "unknown"), procs, cmdlines)
            row["owner_alive"] = alive
            exp = float(rec.get("expected_hours") or 0)
            hb_at = float(rec.get("heartbeat_at") or rec.get("registered_at") or now)
            # A live controller process (pm2:/pid:/proc: owner) IS the
            # heartbeat — kingctl, the swarm manager, the validator rotate
            # their own boxes. Indefinite pods (expected_hours 0: the fleet,
            # eval, chat, driver pods) are outside the heartbeat rule.
            owner = str(rec.get("owner") or "")
            if alive is True and owner.startswith(("pm2:", "pid:", "proc:")):
                hb_at = now
            if exp <= 0:
                hb_at = now
            hb_age_min = (now - hb_at) / 60
            stale_min = float(r.get("heartbeat_stale_min", 60))
            idle_min = float(r.get("idle_min", 30))
            frac = float(r.get("ceiling_page_frac", 0.9))
            row.update({"heartbeat_age_min": round(hb_age_min), "ceiling_h": exp or None,
                        "heartbeat_at": rec.get("heartbeat_at")})
            idle_s = None
            if hb_age_min > stale_min or (exp > 0 and age_h >= frac * exp):
                idle_s = self.idle_seconds(name, pod, rec)
                row["idle_min"] = None if idle_s is None else round(idle_s / 60)
            if rec.get("release_requested_at"):
                row["decision"] = "release_marker"
                report["actions"].append(self.release(
                    pod, name, f"owner marked it done ({rec.get('release_requested_reason') or 'done'})", rec))
            elif exp > 0 and age_h >= exp:
                row["decision"] = "release_ceiling"
                report["actions"].append(self.release(
                    pod, name, f"hard ceiling: {age_h:.1f} h >= {exp:.0f} h (paged at {frac:.0%}; "
                               f"`registry.py extend {name} --hours H` would have kept it)", rec))
            elif exp > 0 and age_h >= frac * exp:
                row["decision"] = "ceiling_soon"
                self.warn(f"ceiling:{name}", f"`{name}` ({rec.get('purpose')}, ${price:.2f}/h) is at "
                          f"{age_h / exp:.0%} of its {exp:.0f} h ceiling (heartbeat {hb_age_min:.0f} min ago"
                          f"{'' if idle_s is None else f', idle {idle_s / 60:.0f} min'}) — "
                          f"`python ops/pods/registry.py extend {name} --hours H` or it is released at {exp:.0f} h",
                          dedupe_h=max(0.5, (1 - frac) * exp / 2))
                report["problems"].append(f"ceiling soon {name} ({age_h:.0f}/{exp:.0f} h)")
            elif hb_age_min > stale_min and alive is not False and bool(r.get("require_owner_dead_for_idle_release", True)):
                # 2026-09-24: rule (b) alone released two live benchsuite pods
                # (chat cells talk to 127.0.0.1 and multi-replica boxes hide
                # their /metrics behind one replica, so "idle" lies). A stale
                # heartbeat with a live (or unknown) owner pages; only `done`,
                # the ceiling, or stale + owner DEAD + idle release.
                row["decision"] = "stale_owner_alive"
                self.warn(f"stale_alive:{name}", f"`{name}` ({rec.get('purpose')}, ${price:.2f}/h, age {age_h:.1f} h): "
                          f"no owner heartbeat for {hb_age_min:.0f} min but owner `{rec.get('owner')}` is "
                          f"{'alive' if alive else 'unknown'}"
                          f"{'' if idle_s is None else f' (vLLM idle {idle_s / 60:.0f} min)'} — NOT released; "
                          f"`registry.py touch {name}` keeps it quiet, `registry.py done {name}` releases it",
                          dedupe_h=6)
                report["problems"].append(f"stale heartbeat, owner alive {name}")
            elif hb_age_min > stale_min and alive is False and idle_s is not None and idle_s >= idle_min * 60:
                row["decision"] = "release_stale_idle"
                report["actions"].append(self.release(
                    pod, name, f"no owner heartbeat for {hb_age_min:.0f} min, owner `{rec.get('owner')}` dead, "
                               f"and no /v1 request for {idle_s / 60:.0f} min", rec))
            elif hb_age_min > stale_min and idle_s is None:
                row["decision"] = "stale_no_idle_signal"
                self.warn(f"stale:{name}", f"`{name}` ({rec.get('purpose')}, ${price:.2f}/h, age {age_h:.1f} h): "
                          f"no owner heartbeat for {hb_age_min:.0f} min and no /metrics to read idleness — "
                          f"not released; `registry.py touch {name}` if it is in use, `registry.py done {name}` if not",
                          dedupe_h=12)
                report["problems"].append(f"stale heartbeat, no idle signal {name}")
            elif hb_age_min > stale_min:
                row["decision"] = "stale_but_busy"
            else:
                row["decision"] = "ok"
            report["ours"].append(row)

        # registered pods that vanished from Lium: close the record
        for name, rec in list(reg.items()):
            if name in listed or rec.get("released_at"):
                continue
            if rec.get("registered_at") and now - float(rec["registered_at"]) > 600:
                registry.release(name, reason="gone from the Lium listing", by=TAG)
        report["ours_usd_h"] = round(ours_usd, 2)
        report["foreign"]["usd_h"] = round(report["foreign"]["usd_h"], 2)
        report["n_ours"] = len(report["ours"])
        report["over_lifetime"] = [x["name"] for x in report["ours"]
                                   if str(x.get("decision", "")).startswith(("release_ceiling", "ceiling_soon"))]
        saved = sum(a.get("saved_usd_est", 0) for a in report["actions"] if a.get("released"))
        log(f"ours={report['n_ours']} ${ours_usd:.2f}/h foreign={report['foreign']['count']} "
            f"${report['foreign']['usd_h']:.2f}/h actions={len(report['actions'])} "
            f"saved≈${saved:.0f} problems={len(report['problems'])}")
        self._finish(report)
        return report

    def _finish(self, report: dict) -> None:
        # forget first_seen entries of pods no longer listed
        listed_now = {x["name"] for x in report["ours"]}
        for name in [n for n in self.own["first_seen"] if n not in listed_now]:
            self.own["first_seen"].pop(name, None)
        for key in [k for k, v in self.own["warned"].items() if common.now() - float(v) > 3 * 86400]:
            self.own["warned"].pop(key, None)
        common.atomic_write_json(self.own_path, self.own)
        common.atomic_write_json(self.report_path, report)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(HERE / "pods.toml"))
    ap.add_argument("--interval", type=int, default=None)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="decide + log, never rm, never post")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())
    interval = args.interval or int(cfg["reaper"].get("interval_s", 900))
    reaper = Reaper(cfg, dry_run=args.dry_run)
    log(f"start interval={interval}s enforce={reaper.enforce} dry_run={args.dry_run}")
    while True:
        try:
            rep = reaper.tick()
            if args.once:
                print(json.dumps(rep, indent=1, default=str))
                return 0
        except (requests.RequestException, OSError, ValueError, KeyError) as e:
            log(f"tick failed: {e!r}")
            if args.once:
                return 1
        time.sleep(interval)


if __name__ == "__main__":
    sys.exit(main())
