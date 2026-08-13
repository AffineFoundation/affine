#!/usr/bin/env bash
# Host-side: SCP install+serve scripts to affine-teacher and run remote_install.
set -euo pipefail

ROOT=/home/const/subnet120
EXP="$ROOT/ops/teacher-b200"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

HF_TOKEN=${HF_TOKEN:-}
if [[ -z "$HF_TOKEN" && -x "$ROOT/ralphs/secret.sh" ]]; then
  HF_TOKEN=$(bash "$ROOT/ralphs/secret.sh" HF_TOKEN)
fi
: "${HF_TOKEN:?HF_TOKEN required}"

resolve_ssh() {
  lium ps --format json | python3 -c '
import json, re, sys
name = "affine-teacher"
pods = json.load(sys.stdin)
pods = pods if isinstance(pods, list) else pods.get("pods") or []
for p in pods:
    if (p.get("name") or p.get("pod_name")) != name:
        continue
    cmd = p.get("ssh_cmd") or p.get("ssh_connect_cmd") or ""
    m = re.search(r"ssh\s+root@(\S+)\s+-p\s+(\d+)", cmd)
    if m:
        print(m.group(1), m.group(2))
        sys.exit(0)
sys.exit(1)
'
}

write_endpoint() {
  local host=$1
  lium describe affine-teacher --json | python3 -c '
import json, re, sys, time
from pathlib import Path
host = sys.argv[1]
d = json.load(sys.stdin)
ports = (d.get("ports") or {})
mapping = ports.get("mapping") if isinstance(ports, dict) else {}
if not isinstance(mapping, dict):
    mapping = {}
pub = mapping.get("40000") or mapping.get(40000) or mapping.get("8000") or mapping.get(8000)

def walk(o):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in ("ssh_cmd", "ssh_connect_cmd") and isinstance(v, str) and "root@" in v:
                return v
            r = walk(v)
            if r:
                return r
    elif isinstance(o, list):
        for i in o:
            r = walk(i)
            if r:
                return r
    return None

cmd = walk(d) or ""
m = re.search(r"root@(\S+)\s+-p\s+(\d+)", cmd)
if m:
    host = m.group(1)
base = f"http://{host}:{int(pub)}/v1" if host and pub else None
path = Path("/home/const/subnet120/ops/teacher-b200/artifacts/endpoint.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "host": host,
    "teacher_port_internal": 40000,
    "teacher_port_external": int(pub) if pub else None,
    "base_url": base,
    "repo": "zai-org/GLM-4.5-Air-FP8",
    "tp": 8,
}, indent=2) + "\n")
print("ENDPOINT", path, "base_url=", base)
' "$host"
}

log() { echo "[teacher-boot] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

if [[ -n "${SSH_HOST:-}" && -n "${SSH_PORT:-}" ]]; then
  HOST=$SSH_HOST
  PORT=$SSH_PORT
else
  read -r HOST PORT < <(resolve_ssh)
fi
[[ -n "${HOST:-}" && -n "${PORT:-}" ]] || { log "affine-teacher SSH not found"; exit 1; }
log "target root@$HOST -p $PORT"

SSH=(ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -p "$PORT" "root@$HOST")
SCP=(scp -o StrictHostKeyChecking=accept-new -P "$PORT")

"${SSH[@]}" "mkdir -p /root/teacher /root/logs"
"${SCP[@]}" "$EXP/serve_teacher.sh" "$EXP/remote_install.sh" "root@$HOST:/root/teacher/"
"${SSH[@]}" "chmod +x /root/teacher/*.sh"

log "remote install + download + serve (long)"
"${SSH[@]}" "HF_TOKEN='$HF_TOKEN' TEACHER_REPO='zai-org/GLM-4.5-Air-FP8' TEACHER_PORT=40000 TEACHER_TP=8 bash /root/teacher/remote_install.sh"

mkdir -p "$EXP/artifacts"
write_endpoint "$HOST"
python3 - <<PY
import json, time
from pathlib import Path
Path("$EXP/artifacts/bootstrapped.json").write_text(json.dumps({
    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "host": "$HOST",
    "ssh_port": int("$PORT"),
    "repo": "zai-org/GLM-4.5-Air-FP8",
    "tp": 8,
}, indent=2) + "\n")
print("BOOT_OK")
PY
