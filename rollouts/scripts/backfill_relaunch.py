"""Env-backfill driver pod: relaunch every driver that should be running but is
not — from the board, not from memory.

rollouts.backfill writes /root/rollouts/drivers/<tag>.json at start (its launch
line, tmux session, log) and stamps completed_at when the row is done. After a
container restart (`lium reboot` remounts the volume, every tmux session is
gone) this recreates the missing sessions. Run by the post-start hook and by
pipeline-health's self-heal over ssh; idempotent.

Why it re-reads the board (2026-09-22): the first version replayed the saved
launch line verbatim and three times did the wrong thing — it resurrected
reign 14's and 16's drivers against serving boxes that had died (they spun
on "preflight failed" for hours), and the 09:46 relaunch of reign 17 reused a
source list without `affine_oolong`, so that cell sat at 11 rollouts for six
hours while two drivers ran. Rules now:

  (a) a record is skipped when its row is FULL on the board (every gradable
      env cell has n >= BACKFILL_MIN_N), or when the model's serving box does
      not answer `GET <KING_BASE_URL>/models` (dead box: the coverage queue
      re-rents and repoints .king_env_<digest>; the next pass relaunches);
  (b) `--sources` is rebuilt from the board's cells with n < BACKFILL_MIN_N at
      relaunch time (minus envs with no grader, budget-tag columns, share-0
      envs, and the fixed exclusions) instead of the saved list; a record
      with a suffix (`-b`, `-e`, `-oolong`: a second driver on the same row)
      keeps only saved ∩ missing, sources an ALIVE driver of the same row is
      already running are not handed out twice, and the primary (no suffix)
      takes the rest;
  (c) drivers/retired/<name>.json is honoured: a record whose retired copy is
      at least as new as the live one is never relaunched (the benchsuite
      worker moves records there when it kills a driver on purpose).

Board unreachable → saved lists are used (with the box check still on) and
the reason is printed. Nothing is deleted; every decision prints one line.
`--dry-run` prints the decisions without launching or rewriting records.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

DRIVERS = Path(os.environ.get("BACKFILL_DRIVERS_DIR", "/root/rollouts/drivers"))
RETIRED = DRIVERS / "retired"
BOARD_URL = os.environ.get("BACKFILL_BOARD_URL", "https://kings.affine.io/api/matrix.json")
MIN_N = int(os.environ.get("BACKFILL_MIN_N", "24"))
# Envs that are never a coverage target (no grader / probe-only / excluded by
# the benchsuite's launcher) even when the board shows a low cell.
FIXED_EXCLUDE = {"affine_wiki", "affine_tau2", "affine_gdpval"}
SOURCES_RE = re.compile(r"""--sources(?:=|\s+)('[^']*'|"[^"]*"|\S+)""")
DIGEST_RE = re.compile(r"--digest12(?:=|\s+)([0-9a-f]{12})")
KING_ENV_RE = re.compile(r"ROLLOUTS_KING_ENV=(\S+)")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def tmux_alive(name: str) -> bool:
    # "=" = exact name; plain -t is a prefix match (backfill-X would match backfill-X-b).
    return subprocess.run(["tmux", "-f", "/dev/null", "has-session", "-t", "=" + name],
                          capture_output=True).returncode == 0


def saved_sources(cmd: str) -> list[str] | None:
    m = SOURCES_RE.search(cmd)
    if not m:
        return None
    raw = m.group(1).strip("'\"")
    if raw == "all":
        return None
    return [s for s in re.split(r"[,\s]+", raw) if s]


def with_sources(cmd: str, sources: list[str]) -> str:
    joined = ",".join(sources)
    if SOURCES_RE.search(cmd):
        return SOURCES_RE.sub(f"--sources {joined}", cmd, count=1)
    # No --sources in the saved line (= all live sources): insert after --digest12 / --teacher.
    m = DIGEST_RE.search(cmd)
    if m:
        return cmd[:m.end()] + f" --sources {joined}" + cmd[m.end():]
    return cmd.replace("--teacher", f"--teacher --sources {joined}", 1)


def row_key(cmd: str) -> str | None:
    m = DIGEST_RE.search(cmd)
    if m:
        return m.group(1)
    return "teacher" if "--teacher" in cmd else None


def king_env_path(cmd: str, digest12: str | None) -> Path | None:
    m = KING_ENV_RE.search(cmd)
    if m:
        return Path(m.group(1).strip("'\""))
    return Path(f"/root/rollouts/.king_env_{digest12}") if digest12 else None


def box_alive(env_path: Path | None) -> tuple[bool, str]:
    """GET <KING_BASE_URL>/models with the box's bearer; (ok, detail)."""
    if env_path is None:
        return True, "teacher endpoint (not checked here)"
    try:
        kv = dict(line.strip().split("=", 1) for line in env_path.read_text().splitlines()
                  if "=" in line and not line.startswith("#"))
    except OSError as e:
        return False, f"{env_path} unreadable ({e.__class__.__name__})"
    base = kv.get("KING_BASE_URL", "").strip().strip("'\"").rstrip("/")
    key = kv.get("KING_KEY", "").strip().strip("'\"")
    if not base:
        return False, f"{env_path.name}: no KING_BASE_URL (seat unpublished)"
    req = urllib.request.Request(base + "/models", headers={"Authorization": f"Bearer {key}"} if key else {})
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status == 200, f"{base}/models -> {r.status}"
    except Exception as e:  # noqa: BLE001 — any failure = dead box for our purpose
        return False, f"{base}/models -> {e.__class__.__name__}: {str(e)[:80]}"


def fetch_board() -> dict | None:
    try:
        # Cloudflare in front of kings.affine.io returns 403 to Python-urllib's default UA.
        req = urllib.request.Request(BOARD_URL, headers={"User-Agent": "affine-backfill-relaunch/0.1"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        print(f"board {BOARD_URL} unreachable ({e.__class__.__name__}: {str(e)[:80]}); using saved source lists")
        return None


def board_missing(board: dict, key: str) -> list[str] | None:
    """Env cells of row `key` with n < MIN_N (or no score); None = row not on the board."""
    row = next((r for r in board.get("rows", []) if r.get("key") == key or r.get("digest12") == key), None)
    if row is None:
        return None
    exclude = set(FIXED_EXCLUDE)
    for e in board.get("excluded_envs") or []:
        exclude.add(e.get("source") if isinstance(e, dict) else str(e))
    cells = row.get("cells") or {}
    missing = []
    for c in board.get("columns", []):
        if c.get("kind") != "env" or c.get("no_grader") or c.get("budget_tag"):
            continue
        env = c.get("env") or c["key"].removeprefix("env:")
        if env in exclude:
            continue
        v = cells.get(c["key"]) or {}
        if v.get("unverified") or v.get("errored_only"):
            continue  # published on purpose without a number: present, not a gap
        if v.get("score") is None or int(v.get("n") or 0) < MIN_N:
            missing.append(env)
    return missing


def registry_sources() -> set[str] | None:
    try:
        sys.path[:0] = ["/root/rollouts", "/root/affine"]
        from rollouts.registry import load_registry  # type: ignore
        return set(load_registry().sources)
    except Exception as e:  # noqa: BLE001
        print(f"registry not importable ({e.__class__.__name__}); not filtering sources by registry")
        return None


def retired(path: Path) -> bool:
    r = RETIRED / path.name
    return r.exists() and r.stat().st_mtime >= path.stat().st_mtime


def main() -> int:
    dry = "--dry-run" in sys.argv[1:]
    if not DRIVERS.is_dir():
        print("no driver records")
        return 0
    records = []
    for f in sorted(DRIVERS.glob("*.json")):
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            print(f"{f.name}: unreadable record ({e.__class__.__name__}); skipped")
            continue
        records.append((f, rec))
    board = fetch_board()
    known = registry_sources() if board is not None else None

    # Sources already running per row: alive drivers keep what they run; others must not duplicate them.
    running: dict[str, set[str]] = {}
    todo = []
    for f, rec in records:
        cmd = rec.get("cmd", "")
        key = row_key(cmd)
        if tmux_alive(rec["tmux"]):
            if key:
                running.setdefault(key, set()).update(saved_sources(cmd) or [])
            continue
        todo.append((f, rec, key))

    # Primary (tag == digest, no suffix) last, so it takes what the suffixed drivers do not.
    todo.sort(key=lambda t: t[0].name)
    todo = [t for t in todo if t[0].stem != (t[2] or "")] + [t for t in todo if t[0].stem == (t[2] or "")]

    n_launched = 0
    for f, rec, key in todo:
        name = rec["tmux"]
        cmd = rec.get("cmd", "")
        if rec.get("completed_at"):
            continue
        if retired(f):
            print(f"{name}: retired ({RETIRED.name}/{f.name}); not relaunched")
            continue
        if not os.path.exists(rec.get("wrapper", "")):
            print(f"{name}: wrapper {rec.get('wrapper')} missing; not relaunched")
            continue
        digest12 = key if key and key != "teacher" else None
        ok, detail = box_alive(king_env_path(cmd, digest12) if digest12 else None)
        if not ok:
            print(f"{name}: serving box dead ({detail}); not relaunched")
            continue
        saved = saved_sources(cmd)
        sources = saved
        if board is not None and key:
            missing = board_missing(board, key)
            if missing is None:
                print(f"{name}: row {key} not on the board; using saved list")
            else:
                if known is not None:
                    unknown = [s for s in missing if s not in known]
                    missing = [s for s in missing if s in known]
                    if unknown:
                        print(f"{name}: board cells not in this pod's registry, dropped: {','.join(unknown)}")
                taken = running.get(key, set())
                wanted = [s for s in missing if s not in taken]
                if f.stem != key and saved is not None:
                    wanted = [s for s in wanted if s in saved]
                if not wanted:
                    why = "row FULL on the board" if not missing else f"its missing cells already run elsewhere ({','.join(sorted(taken))})"
                    print(f"{name}: {why}; not relaunched")
                    continue
                sources = wanted
                cmd = with_sources(cmd, sources)
                running.setdefault(key, set()).update(sources)
        if dry:
            print(f"{name}: WOULD relaunch sources={','.join(sources) if sources else 'all'}")
            n_launched += 1
            continue
        subprocess.run(["tmux", "-f", "/dev/null", "new-session", "-d", "-s", name, cmd], check=False)
        rec["cmd"] = cmd
        rec["relaunched_at"] = now()
        rec["relaunch_sources"] = sources
        try:
            f.write_text(json.dumps(rec, indent=1))
        except OSError:
            pass
        try:
            with open(rec["log"], "a") as lf:
                lf.write(f"{now()} relaunched by backfill_relaunch.py from {f} sources={','.join(sources) if sources else 'all'}\n")
        except OSError:
            pass
        print(f"{name}: relaunched sources={','.join(sources) if sources else 'all'}")
        n_launched += 1
    print(f"checked {len(records)} driver record(s), {'would relaunch' if dry else 'relaunched'} {n_launched}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
