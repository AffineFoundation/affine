#!/bin/bash
# Env-backfill driver pod: relaunch every driver that should be running but
# is not. rollouts.backfill records its own launch line in
# /root/rollouts/drivers/<tag>.json at start (and marks completed_at when
# the row is done); after a container restart (volume remounted by `lium
# reboot`) this recreates the missing tmux sessions from those records so
# the rows resume from state without a human. Idempotent; run by the
# post-start hook and by pipeline-health's self-heal over ssh.
set -uo pipefail
DIR=/root/rollouts/drivers
[ -d "$DIR" ] || { echo "no driver records"; exit 0; }
n=0
for f in "$DIR"/*.json; do
  [ -f "$f" ] || continue
  python3 - "$f" <<'PY'
import json, sys, subprocess, os, time
rec = json.load(open(sys.argv[1]))
if rec.get("completed_at"):
    sys.exit(0)
tmux = rec["tmux"]
alive = subprocess.run(["tmux", "-f", "/dev/null", "has-session", "-t", tmux], capture_output=True).returncode == 0
if alive:
    sys.exit(0)
if not os.path.exists(rec.get("wrapper", "")):
    print(f"{tmux}: wrapper {rec.get('wrapper')} missing; not relaunched")
    sys.exit(0)
subprocess.run(["tmux", "-f", "/dev/null", "new-session", "-d", "-s", tmux, rec["cmd"]], check=False)
with open(rec["log"], "a") as f:
    f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} relaunched by backfill_relaunch.sh from {sys.argv[1]}\n")
print(f"{tmux}: relaunched")
PY
  n=$((n+1))
done
echo "checked $n driver record(s)"
