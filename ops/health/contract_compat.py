#!/usr/bin/env python
"""contract_compat — do the consumers of verdict numbers still understand them?

The contract (affine/affine.toml [duel].score_mode) fixes the UNITS of every
number on a verdict (per-byte nats under min_rg, teacher-sd under
sd_min_rga, …). Every downstream consumer declares in
ops/health/consumers.toml which units / score modes it handles; a consumer
that fits something (the curriculum) also stamps `input_units` /
`fitted_score_modes` into its own output, and that live declaration wins.

  python contract_compat.py                       # table + exit 0 ok / 1 mismatch / 2 blocked
  python contract_compat.py --json
  python contract_compat.py --preflight --score-mode sd_min_rga [--wvk 23]
        # flip-script step 0: evaluate against the NEW mode; freeze the
        # freezable consumers; exit 2 (refuse) only if a `block` consumer
        # is incompatible, else 0 (pages are the caller's job -> --page)
  python contract_compat.py --enforce             # what the monitor does each tick:
        # freeze incompatible freeze-consumers, auto-unfreeze once compatible
  python contract_compat.py --unfreeze curriculum --reason "re-fitted unit-free, latest.json stamped"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import common  # noqa: E402

CONFIG = HERE / "consumers.toml"
TAG = "contract-compat"

_SCORE_MODE_RE = re.compile(r'^score_mode\s*=\s*"([^"]+)"', re.M)
_WVK_RE = re.compile(r"^weight_version_key\s*=\s*(\d+)", re.M)


def load_config() -> dict:
    return tomllib.loads(CONFIG.read_text())


def units_of(mode: str | None, cfg: dict) -> str | None:
    if not mode:
        return None
    u = cfg["units"].get(mode)
    if u:
        return u
    if mode.startswith("sd_"):
        return "teacher_sd"
    return None


def live_contract(cfg: dict) -> dict:
    """score_mode + wvk from the toml; score_mode of the last stamped verdict."""
    toml_path = REPO / cfg["live"]["toml"]
    text = toml_path.read_text()
    m = _SCORE_MODE_RE.search(text)
    w = _WVK_RE.search(text)
    out = {"toml_score_mode": m.group(1) if m else None,
           "wvk": int(w.group(1)) if w else None,
           "toml_units": None, "stamped_score_mode": None, "stamped_at": None,
           "stamped_challenge": None}
    out["toml_units"] = units_of(out["toml_score_mode"], cfg)
    rows = common.jsonl_tail(REPO / cfg["live"]["history"], 3_000_000)
    for r in reversed(rows):
        if r.get("event") not in ("verdict", "crowned"):
            continue
        v = r.get("verdict") or {}
        dp = v.get("duel_params") or r.get("duel_params") or {}
        if dp.get("score_mode"):
            out["stamped_score_mode"] = dp["score_mode"]
            out["stamped_at"] = r.get("at")
            out["stamped_challenge"] = r.get("challenge_id")
            break
    out["stamped_units"] = units_of(out["stamped_score_mode"], cfg)
    return out


def declaration(consumer: dict) -> dict:
    """The consumer's effective declaration: live file stamp if present,
    else the static one. `declared` False = the live file exists but carries
    no stamp (undeclared)."""
    out = {"units": [u for u in consumer.get("expects_units", [])],
           "score_modes": list(consumer.get("expects_score_modes", [])),
           "source": "static", "declared": True, "declaration_file": None}
    df = consumer.get("declaration_file")
    if not df:
        return out
    path = REPO / df
    out["declaration_file"] = str(path)
    data = common.read_json(path)
    if data is None:
        out["source"] = "static (declaration file missing)"
        return out
    iu = data.get("input_units")
    fm = data.get("fitted_score_modes")
    if iu or fm:
        out["source"] = "live"
        out["units"] = [iu] if isinstance(iu, str) else list(iu or [])
        out["score_modes"] = list(fm or [])
        out["computed_at"] = data.get("computed_at")
    else:
        out["declared"] = False
        out["source"] = "static (live file has no input_units / fitted_score_modes stamp)"
    return out


def compatible(decl: dict, mode: str | None, units: str | None) -> bool:
    if units is None or mode is None:
        return False
    if any(u in ("any", "unit_free") for u in decl["units"]):
        return True
    if units in decl["units"]:
        return True
    if mode in decl["score_modes"]:
        return True
    return False


def freeze_state(consumer: dict) -> dict | None:
    ff = consumer.get("freeze_file")
    if not ff:
        return None
    return common.read_json(REPO / ff)


def freeze(consumer: dict, reason: str, *, by: str = TAG, dry_run: bool = False) -> Path | None:
    ff = consumer.get("freeze_file")
    if not ff:
        return None
    path = REPO / ff
    body = {"frozen_at": common.now_iso(), "by": by, "reason": reason,
            "consumer": consumer["name"],
            "unfreeze": f"python ops/health/contract_compat.py --unfreeze {consumer['name']} --reason '...'"}
    if not dry_run:
        common.atomic_write_json(path, body)
    common.log(TAG, f"FROZEN {consumer['name']} -> {path} ({reason})" + (" [dry-run]" if dry_run else ""))
    return path


def unfreeze(consumer: dict, reason: str, *, by: str = "cli") -> bool:
    ff = consumer.get("freeze_file")
    if not ff:
        return False
    path = REPO / ff
    if not path.exists():
        return False
    prev = common.read_json(path) or {}
    path.unlink()
    log_path = path.with_name("freeze_history.jsonl")
    with open(log_path, "a") as fh:
        fh.write(json.dumps({"at": common.now_iso(), "event": "unfreeze", "by": by, "reason": reason,
                             "was": prev}, default=str) + "\n")
    common.log(TAG, f"UNFROZEN {consumer['name']} ({reason})")
    return True


def evaluate(cfg: dict, *, target_mode: str | None = None, target_wvk: int | None = None) -> dict:
    live = live_contract(cfg)
    mode = target_mode or live["toml_score_mode"]
    units = units_of(mode, cfg)
    res = {"at": common.now_iso(), "live": live, "target_score_mode": mode, "target_units": units,
           "target_wvk": target_wvk or live["wvk"], "consumers": [], "ok": True, "blocked": False,
           "problems": []}
    if units is None:
        res["ok"] = False
        res["problems"].append(f"score_mode {mode!r} has no units entry in consumers.toml [units]")
    if live["stamped_score_mode"] and live["toml_score_mode"] and \
            live["stamped_score_mode"] != live["toml_score_mode"]:
        lag = common.now() - (common.parse_iso(live["stamped_at"]) or common.now())
        res["stamp_lag_h"] = round(lag / 3600, 2)
        if lag > float(cfg["live"].get("stamp_lag_h", 3)) * 3600:
            res["problems"].append(
                f"toml score_mode {live['toml_score_mode']} but the last verdict "
                f"({live['stamped_challenge']}, {common.fmt_age(lag)} ago) stamps {live['stamped_score_mode']}")
    for c in cfg.get("consumer", []):
        decl = declaration(c)
        ok = compatible(decl, mode, units)
        fs = freeze_state(c)
        row = {"name": c["name"], "path": c.get("path"), "on_mismatch": c.get("on_mismatch", "page"),
               "declared_units": decl["units"], "declared_score_modes": decl["score_modes"],
               "declaration_source": decl["source"], "declared": decl["declared"],
               "compatible": ok, "frozen": bool(fs), "frozen_reason": (fs or {}).get("reason"),
               "frozen_by": (fs or {}).get("by"), "owner": c.get("owner")}
        if not decl["declared"]:
            row["status"] = "undeclared"
            res["problems"].append(f"{c['name']}: live declaration file has no input_units stamp "
                                   f"(static says {decl['units']})")
        if not ok:
            row["status"] = "mismatch"
            res["ok"] = False
            res["problems"].append(
                f"{c['name']} expects {decl['units'] or decl['score_modes']} ({decl['source']}) but "
                f"{'target' if target_mode else 'live'} score_mode {mode} is in {units} units "
                f"-> {c.get('on_mismatch', 'page')}")
            if c.get("on_mismatch") == "block":
                res["blocked"] = True
        elif "status" not in row:
            row["status"] = "ok"
        res["consumers"].append(row)
    return res


def enforce(cfg: dict, res: dict, *, dry_run: bool = False) -> list[str]:
    """Freeze incompatible freeze-consumers; lift a freeze this tool wrote
    once the consumer declares compatible units again. Returns event lines."""
    events = []
    by_name = {c["name"]: c for c in cfg.get("consumer", [])}
    for row in res["consumers"]:
        c = by_name[row["name"]]
        if c.get("on_mismatch") != "freeze":
            continue
        if not row["compatible"] and not row["frozen"]:
            reason = (f"units mismatch: declares {row['declared_units'] or row['declared_score_modes']} "
                      f"({row['declaration_source']}), live score_mode {res['target_score_mode']} = "
                      f"{res['target_units']} units (wvk {res['target_wvk']})")
            freeze(c, reason, dry_run=dry_run)
            row["frozen"] = True
            row["frozen_reason"] = reason
            events.append(f"FROZE {c['name']}: {reason}")
        elif row["compatible"] and row["frozen"] and row.get("frozen_by") == TAG and row["declared"] \
                and row["declaration_source"] == "live":
            if not dry_run:
                unfreeze(c, "declaration compatible again (live stamp)", by=TAG)
            row["frozen"] = False
            events.append(f"UNFROZE {c['name']}: live declaration {row['declared_units']} matches "
                          f"{res['target_units']}")
    return events


def table(res: dict) -> str:
    live = res["live"]
    lines = [f"live: toml score_mode={live['toml_score_mode']} ({live['toml_units']}) wvk={live['wvk']}; "
             f"last stamped verdict {live['stamped_challenge']} score_mode={live['stamped_score_mode']}"]
    if res.get("target_score_mode") != live["toml_score_mode"]:
        lines.append(f"target: score_mode={res['target_score_mode']} ({res['target_units']}) wvk={res['target_wvk']}")
    lines.append(f"{'consumer':28} {'status':11} {'on_mismatch':11} {'frozen':6} declares")
    for r in res["consumers"]:
        decl = ", ".join(r["declared_units"] or []) + (
            f" / modes {','.join(r['declared_score_modes'])}" if r["declared_score_modes"] else "")
        lines.append(f"{r['name']:28} {r['status']:11} {r['on_mismatch']:11} {'yes' if r['frozen'] else 'no':6} "
                     f"{decl}  [{r['declaration_source']}]")
    for p in res["problems"]:
        lines.append(f"! {p}")
    lines.append("RESULT: " + ("BLOCKED" if res["blocked"] else "ok" if res["ok"] else "MISMATCH"))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--preflight", action="store_true",
                    help="evaluate against --score-mode/--wvk; freeze freezable consumers; exit 2 if blocked")
    ap.add_argument("--score-mode", default=None)
    ap.add_argument("--wvk", type=int, default=None)
    ap.add_argument("--enforce", action="store_true", help="freeze / unfreeze as the monitor does")
    ap.add_argument("--page", action="store_true", help="post problems to the private Discord channel")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--unfreeze", metavar="CONSUMER", default=None)
    ap.add_argument("--reason", default="")
    args = ap.parse_args()
    cfg = tomllib.loads(Path(args.config).read_text())

    if args.unfreeze:
        c = next((c for c in cfg["consumer"] if c["name"] == args.unfreeze), None)
        if c is None:
            print(f"unknown consumer {args.unfreeze}", file=sys.stderr)
            return 2
        if not args.reason:
            print("--reason is required", file=sys.stderr)
            return 2
        ok = unfreeze(c, args.reason, by="cli")
        print("unfrozen" if ok else "was not frozen")
        if args.page and ok:
            common.discord_post(f"{args.unfreeze} unfrozen — {args.reason}", prefix="[contract]",
                                dry_run=args.dry_run)
        return 0

    res = evaluate(cfg, target_mode=args.score_mode, target_wvk=args.wvk)
    events: list[str] = []
    if args.preflight or args.enforce:
        events = enforce(cfg, res, dry_run=args.dry_run)
        res["events"] = events
    if args.json:
        print(json.dumps(res, indent=1, default=str))
    else:
        print(table(res))
        for e in events:
            print("EVENT:", e)
    if args.page and (res["problems"] or events):
        text = "; ".join(events + res["problems"])[:1500]
        common.discord_post(("PREFLIGHT " if args.preflight else "") + text, prefix="[contract]",
                            dry_run=args.dry_run)
    if res["blocked"]:
        return 2
    if args.preflight:
        # freezable mismatches were frozen; page-only mismatches are the caller's
        # notice (printed above). Only `block` refuses the flip.
        return 0
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
