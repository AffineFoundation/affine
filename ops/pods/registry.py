#!/usr/bin/env python
"""Registry of every Lium pod we rent: purpose, owner, expected lifetime.

Every rent path registers here (lium_api.rent does it automatically; kingctl,
the swarm manager and kingpod add the explicit purpose / owner); every
release marks the record. ops/pods/reaper.py reads it and releases what is
unregistered or past its expected lifetime with no live owner.

Owner kinds (how the reaper decides "alive"):
  pm2:<name>        the pm2 process is online
  pid:<n>           the pid is alive
  proc:<substring>  some process command line contains <substring>
  passlog:<dir>     a benchsuite pass log under <dir> (repo-relative) that
                    names the pod was written in the last 3 h and has no
                    .exit sibling, or a live process names the pod
  manual:<who>      always alive (only the expected lifetime can reap it;
                    expected_hours 0 = indefinite = never reaped)
  unknown           never alive

CLI:
  python registry.py list
  python registry.py register NAME --purpose P --owner O --hours H [--price X] [--note ...]
  python registry.py release NAME --reason R
  python registry.py show NAME
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
import time
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "ops" / "health"))
import common  # noqa: E402

CONFIG = HERE / "pods.toml"
SOURCE_RANK = {"static": 3, "explicit": 3, "adopted": 2, "auto": 1}


def load_config() -> dict:
    return tomllib.loads(CONFIG.read_text())


def registry_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return (HERE / cfg["paths"]["registry"]).resolve()


def ledger_path(cfg: dict | None = None) -> Path:
    cfg = cfg or load_config()
    return (HERE / cfg["paths"]["ledger"]).resolve()


def _lock_path(cfg: dict | None = None) -> Path:
    return registry_path(cfg).with_suffix(".lock")


def load(cfg: dict | None = None) -> dict:
    data = common.read_json(registry_path(cfg), default={})
    return data if isinstance(data, dict) else {}


def _save(data: dict, cfg: dict | None = None) -> None:
    common.atomic_write_json(registry_path(cfg), data)


def is_ours(name: str, cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    return any(fnmatch.fnmatch(name, g) for g in cfg["scope"]["prefixes"])


def matched_prefix(name: str, cfg: dict | None = None) -> str | None:
    """The literal prefix (glob minus its trailing `*`) the name matches —
    the guard string lium_api.remove wants."""
    cfg = cfg or load_config()
    for g in cfg["scope"]["prefixes"]:
        if fnmatch.fnmatch(name, g):
            return g.rstrip("*")
    return None


def purpose_defaults(name: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    for p in cfg.get("purposes", []):
        if fnmatch.fnmatch(name, p["glob"]):
            return {"purpose": p["purpose"], "owner": p["owner"],
                    "expected_hours": float(p.get("expected_hours", 0) or 0)}
    d = cfg.get("default_purpose") or {}
    return {"purpose": d.get("purpose", "unknown"), "owner": d.get("owner", "unknown"),
            "expected_hours": float(d.get("expected_hours", 14) or 0)}


def default_owner() -> str:
    """Best guess at who is renting: the pm2 process we run under, else our
    pid (which dies with a CLI — fine, the lifetime rule still applies)."""
    pm2_name = os.environ.get("name")
    if pm2_name and os.environ.get("pm_id") is not None:
        return f"pm2:{pm2_name}"
    return f"pid:{os.getpid()}"


def register(name: str, *, purpose: str | None = None, owner: str | None = None,
             expected_hours: float | None = None, price_usd_h: float | None = None,
             ttl_hours: float | None = None, meta: dict | None = None,
             source: str = "explicit") -> dict:
    """Upsert one pod. A weaker source (auto < adopted < explicit) never
    overwrites purpose / owner / expected_hours set by a stronger one, but
    always fills blanks and refreshes price / ttl / heartbeat."""
    cfg = load_config()
    defaults = purpose_defaults(name, cfg)
    with common.file_lock(_lock_path(cfg)):
        data = load(cfg)
        rec = data.get(name) or {"name": name, "registered_at": common.now(),
                                 "source": source}
        old_rank = SOURCE_RANK.get(str(rec.get("source", "auto")).split(":")[0], 1)
        new_rank = SOURCE_RANK.get(source.split(":")[0], 1)
        stronger = new_rank >= old_rank or rec.get("released_at")
        if rec.get("released_at"):
            # re-registering a released name = a fresh pod with a reused name
            rec = {"name": name, "registered_at": common.now(), "source": source}
            stronger = True
        for key, val, dflt in (("purpose", purpose, defaults["purpose"]),
                               ("owner", owner, defaults["owner"]),
                               ("expected_hours", expected_hours, defaults["expected_hours"])):
            if val is not None and (stronger or rec.get(key) in (None, "", "unknown")):
                rec[key] = val
            elif rec.get(key) in (None, ""):
                rec[key] = dflt
        if stronger:
            rec["source"] = source
        if price_usd_h is not None:
            rec["price_usd_h"] = float(price_usd_h)
        if ttl_hours is not None:
            rec["ttl_hours"] = float(ttl_hours)
        if meta:
            rec.setdefault("meta", {}).update(meta)
        rec["heartbeat_at"] = common.now()
        rec.setdefault("released_at", None)
        data[name] = rec
        _save(data, cfg)
        _ledger(cfg, {"at": common.now_iso(), "action": "register", "pod": name,
                      "purpose": rec["purpose"], "owner": rec["owner"],
                      "expected_hours": rec["expected_hours"], "source": source,
                      "price_usd_h": rec.get("price_usd_h")})
        return rec


def touch(name: str) -> None:
    cfg = load_config()
    with common.file_lock(_lock_path(cfg)):
        data = load(cfg)
        if name in data:
            data[name]["heartbeat_at"] = common.now()
            _save(data, cfg)


def release(name: str, reason: str, by: str = "caller") -> dict | None:
    cfg = load_config()
    with common.file_lock(_lock_path(cfg)):
        data = load(cfg)
        rec = data.get(name)
        if rec is None:
            rec = {"name": name, "registered_at": None, "source": "auto",
                   **purpose_defaults(name, cfg)}
            data[name] = rec
        rec["released_at"] = common.now()
        rec["release_reason"] = reason
        rec["released_by"] = by
        _save(data, cfg)
        _ledger(cfg, {"at": common.now_iso(), "action": "release", "pod": name,
                      "reason": reason, "by": by, "price_usd_h": rec.get("price_usd_h"),
                      "age_h": (round((common.now() - rec["registered_at"]) / 3600, 2)
                                if rec.get("registered_at") else None)})
        return rec


def forget(name: str) -> None:
    """Drop a record (a released pod that is gone from Lium for good)."""
    cfg = load_config()
    with common.file_lock(_lock_path(cfg)):
        data = load(cfg)
        if data.pop(name, None) is not None:
            _save(data, cfg)


def _ledger(cfg: dict, row: dict) -> None:
    path = ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as fh:
        fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")


def auto_register_rent(pod_name: str, *, ttl_hours: float | None, gpu_count: int | None,
                       executor_id: str | None) -> None:
    """Called by lium_api.rent after a 2xx. Never raises."""
    try:
        register(pod_name, owner=default_owner(), ttl_hours=ttl_hours,
                 meta={"gpu_count": gpu_count, "executor_id": executor_id,
                       "argv0": os.path.basename(sys.argv[0] or "")},
                 source="auto")
    except Exception as e:  # registry problems must never break a rent
        common.log("registry", f"auto-register {pod_name} failed: {e!r}")


def auto_release(pod_name: str, ok: bool) -> None:
    """Called by lium_api.remove. Never raises."""
    try:
        release(pod_name, reason="lium rm" + ("" if ok else " (reported failure)"),
                by=default_owner())
    except Exception as e:
        common.log("registry", f"auto-release {pod_name} failed: {e!r}")


# -- CLI -------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("register")
    r.add_argument("name")
    r.add_argument("--purpose", required=True)
    r.add_argument("--owner", required=True, help="pm2:<name> | pid:<n> | proc:<s> | passlog:<dir> | manual:<who>")
    r.add_argument("--hours", type=float, required=True, help="expected lifetime in hours (0 = indefinite)")
    r.add_argument("--price", type=float, default=None)
    r.add_argument("--note", default="")
    rl = sub.add_parser("release")
    rl.add_argument("name")
    rl.add_argument("--reason", required=True)
    s = sub.add_parser("show")
    s.add_argument("name")
    args = ap.parse_args()
    if args.cmd == "list":
        data = load()
        for name, rec in sorted(data.items()):
            age = (common.now() - rec["registered_at"]) / 3600 if rec.get("registered_at") else None
            print(f"{name:42} {str(rec.get('purpose')):14} {str(rec.get('owner')):34} "
                  f"exp={rec.get('expected_hours')}h age={age and round(age, 1)}h "
                  f"${rec.get('price_usd_h') or 0:.2f}/h src={rec.get('source')} "
                  f"{'RELEASED ' + str(rec.get('release_reason')) if rec.get('released_at') else ''}")
        return 0
    if args.cmd == "register":
        rec = register(args.name, purpose=args.purpose, owner=args.owner,
                       expected_hours=args.hours, price_usd_h=args.price,
                       meta={"note": args.note} if args.note else None, source="explicit")
        print(json.dumps(rec, indent=1, default=str))
        return 0
    if args.cmd == "release":
        print(json.dumps(release(args.name, args.reason, by="cli"), indent=1, default=str))
        return 0
    if args.cmd == "show":
        print(json.dumps(load().get(args.name), indent=1, default=str))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
