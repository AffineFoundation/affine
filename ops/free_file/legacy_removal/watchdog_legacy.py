#!/usr/bin/env python3
"""legacy-watchdog (RA/20261031-legacy-removal): auto-revert Step 1 (legacy_off.sh MODE=revert) if the first epochs after
legacy_off fail. Every 20 s, for epochs FROM .. FROM+N-1 (N default 3):
  (a) affine-mimo-lifecycle not active 3 checks in a row
  (b) RECOVERY-REQUIRED.json in a watched epoch, or FF-SELECT reason ff-selection-error
  (c) a watched epoch not FINALIZED within EPOCH_LIMIT s (default 1800) of the previous epoch's FINALIZED
  (d) a Traceback in the lifecycle journal since arming
  (e) mining.json status_updated_at older than 1200 s (or unreadable 10 min)
Exits after N watched epochs finalized clean. Logs watchdog.log + PROGRESS.md."""
import json, os, subprocess, time
from pathlib import Path
D = Path("/home/const/subnet120-rewrite/state/root-audits/20261031-legacy-removal")
OFF = Path("/home/const/subnet120-rewrite/state/root-audits/20261030-free-file/activation/legacy_off.sh")
E = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs")
FROM = int(os.environ["FROM"]); N = int(os.environ.get("N", 3)); LIMIT = int(os.environ.get("EPOCH_LIMIT", 1800))
ARMED = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
cr = lambda: time.strftime("%H:%M:%S CR", time.gmtime(time.time() - 21600))


def log(m, prog=True):
    open(D / "watchdog.log", "a").write(f"{cr()} {m}\n")
    if prog:
        open(D / "PROGRESS.md", "a").write(f"- {cr()[:5]} CR: legacy-watchdog: {m}\n")


def mt(p):
    try:
        return Path(p).stat().st_mtime
    except OSError:
        return None


def rj(p):
    try:
        return json.loads(Path(p).read_text())
    except Exception:
        return {}


def active():
    return subprocess.run(["systemctl", "--user", "is-active", "--quiet", "affine-mimo-lifecycle.service"]).returncode == 0


def revert(why):
    log(f"REVERT ({why})")
    r = subprocess.run(["bash", str(OFF)], env=dict(os.environ, MODE="revert"))
    log(f"revert rc={r.returncode}")
    raise SystemExit(0)


log(f"armed FROM=e{FROM} N={N} limit={LIMIT}s (journal since {ARMED} UTC)")
down, ok, fails = 0, set(), 0
while True:
    time.sleep(20)
    down = 0 if active() else down + 1
    if down >= 3:
        revert("lifecycle not active for 60 s")
    j = subprocess.run(["journalctl", "--user", "-u", "affine-mimo-lifecycle", "--since", ARMED + " UTC", "--no-pager", "-o", "cat"],
                       capture_output=True, text=True).stdout
    if "Traceback" in j:
        revert("Traceback in lifecycle journal: " + j[j.index("Traceback"):][:300].replace("\n", " | "))
    try:
        m = json.loads(subprocess.run(["curl", "-s", "-m", "15", "https://affine.io/mining.json"], capture_output=True, text=True).stdout)
        age = time.time() - float(m["status_updated_at"])
        fails = 0
        if age > 1200:
            revert(f"mining.json stale {age:.0f} s")
    except Exception:
        fails += 1
        if fails >= 30:
            revert("mining.json unreadable for 10 min")
    for e in range(FROM, FROM + N):
        d = E / str(e)
        if (d / "RECOVERY-REQUIRED.json").exists():
            revert(f"e{e} RECOVERY-REQUIRED")
        if rj(d / "FF-SELECT.json").get("reason") == "ff-selection-error":
            revert(f"e{e} ff-selection-error")
        prev = mt(E / str(e - 1) / "FINALIZED.json")
        fin = mt(d / "FINALIZED.json")
        if prev and not fin and time.time() - prev > LIMIT:
            revert(f"e{e} not FINALIZED {LIMIT} s after e{e - 1}")
        if fin and prev and e not in ok:
            ok.add(e)
            s, g = rj(d / "FF-SELECT.json"), rj(d / "PROMOTION-GATE.json")
            pop = d / "AUDIT-POPULATION.json"
            log(f"e{e} FINALIZED OK: cycle {fin - prev:.0f} s; select route={s.get('route')} reason={s.get('reason')} "
                f"uid={s.get('uid')}; gate {g.get('decision')} z={g.get('paired_z')}; "
                f"AUDIT-POPULATION {pop.stat().st_size if pop.exists() else None} B")
    if len(ok) >= N:
        log(f"{N} epochs from e{FROM} finalized clean; watchdog done (legacy_off kept)")
        break
