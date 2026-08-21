#!/usr/bin/env bash
# p4281: re-rent mine-r1158 after 1/8 GPU dud tear.
# NEVER blind `lium up --gpu` — that ignores executor_blacklist and re-hit
# fbb1135f@192.9.163.79 (advertised 8×B200, nvidia-smi=1). Rent by node id
# from blacklisted list_nodes; after rent verify ngpu>=8 else rm+continue.
set -euo pipefail

ROOT=/home/const/subnet120
EXP="$ROOT/mining/experiments/r1158-vera-reason-grpo"
FLEET="$ROOT/mining/experiments/fleet-rent"
LOG="$EXP/logs/wait_rent_r1158_p4281.log"
PIDF="$EXP/logs/wait_rent_r1158_p4281.pid"
BL_FILE="$FLEET/artifacts/executor_blacklist.txt"
NAME=${POD_NAME:-mine-r1158-vera-reason-grpo-1}
TTL=${TTL:-24h}
CAP=${MINE_CAP:-25}
POLL_S=${POLL_S:-20}
MAX_ITERS=${MAX_ITERS:-4320}
PASS=${PASS:-4281}
MIN_GPU=${MIN_GPU:-8}

mkdir -p "$EXP/logs" "$EXP/artifacts" "$FLEET/artifacts"
echo $$ >"$PIDF"
exec >>"$LOG" 2>&1

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

log() { echo "[r1158-rent-p4281] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

mine_names() {
  python3 - <<'PY'
import json, subprocess, sys
try:
    raw = subprocess.check_output(["lium", "ps", "--format", "json"], text=True, timeout=60)
    data = json.loads(raw)
except Exception as e:
    print(f"PS_FAIL {e}", file=sys.stderr)
    sys.exit(0)
pods = data if isinstance(data, list) else data.get("pods") or data.get("data") or []
for p in pods:
    if not isinstance(p, dict):
        continue
    name = p.get("name") or ""
    if isinstance(name, str) and name.startswith("mine-"):
        print(name)
PY
}

already_live() { mine_names | grep -qx "$NAME"; }
mine_count() { mine_names | wc -l; }

# Blacklist-aware node ids (huid/uuid), one per line — same filter as wait_fleet_b300.
list_nodes() {
  local gpu=$1
  lium ls --gpu "$gpu" --count 8 --format json 2>/dev/null \
    | LIUM_EXECUTOR_BLACKLIST_FILE="$BL_FILE" python3 -c '
import json,os,sys
from pathlib import Path
bl=set()
bf=os.environ.get("LIUM_EXECUTOR_BLACKLIST_FILE","")
if bf and Path(bf).is_file():
    for line in Path(bf).read_text().splitlines():
        line=line.strip()
        if line and not line.startswith("#"):
            bl.add(line.split()[0])
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
nodes = d if isinstance(d, list) else (d.get("nodes") or d.get("data") or [])
for n in nodes:
    if not isinstance(n, dict):
        continue
    eid = str(n.get("id") or "")
    if eid and eid in bl:
        continue
    nid = n.get("huid") or n.get("id") or n.get("name")
    if nid:
        print(nid)
'
}

resolve_ssh() {
  POD_NAME="$NAME" python3 - <<'PY'
import json, os, re, subprocess, sys
name = os.environ["POD_NAME"]
raw = subprocess.check_output(["lium", "ps", "--format", "json"], text=True, timeout=60)
pods = json.loads(raw)
if isinstance(pods, dict):
    pods = pods.get("pods") or pods.get("data") or []
for p in pods:
    if (p.get("name") or "") != name:
        continue
    ip = p.get("ip") or ""
    ports = p.get("ports") or {}
    port = ports.get("22") or ports.get(22)
    cmd = p.get("ssh_cmd") or ""
    m = re.search(r"ssh\s+root@(\S+)\s+-p\s+(\d+)", cmd)
    if m:
        ip, port = m.group(1), int(m.group(2))
    if ip and port:
        print(f"{ip} {port}")
        sys.exit(0)
sys.exit(1)
PY
}

pod_executor_and_api_gpus() {
  POD_NAME="$NAME" python3 - <<'PY'
import configparser, json, os, sys
from pathlib import Path
import requests
name = os.environ["POD_NAME"]
cfg = configparser.ConfigParser()
cfg.read(Path.home() / ".lium" / "config.ini")
key = cfg.get("api", "api_key", fallback="") or cfg.get("default", "api_key", fallback="")
r = requests.get(
    "https://lium.io/api/pods",
    headers={"X-API-KEY": key, "X-Source": "fleet-api", "X-Lium-Client-Version": "0.0.32"},
    timeout=30,
)
r.raise_for_status()
for p in r.json() if isinstance(r.json(), list) else []:
    n = p.get("pod_name") or p.get("name") or ""
    if n != name:
        continue
    eid = p.get("executor_id") or (p.get("executor") or {}).get("id") or ""
    g = p.get("gpu_count")
    try:
        g = int(g)
    except Exception:
        g = -1
    print(f"{eid} {g}")
    sys.exit(0)
sys.exit(1)
PY
}

ssh_gpu_count() {
  local host=$1 port=$2
  ssh -i "$HOME/.ssh/id_ed25519" \
    -o StrictHostKeyChecking=accept-new \
    -o UserKnownHostsFile="/tmp/${NAME}.known_hosts" \
    -o ConnectTimeout=20 -o BatchMode=yes \
    -p "$port" "root@$host" 'nvidia-smi -L 2>/dev/null | wc -l' || echo 0
}

try_rent_node() {
  local node=$1
  log "attempting lium up node=$node name=$NAME ttl=$TTL"
  set +e
  lium up "$node" --name "$NAME" --ttl "$TTL" --no-ssh -y --ports 12
  local ec=$?
  set -e
  return "$ec"
}

append_blacklist() {
  local eid=$1 note=$2
  [[ -n "$eid" ]] || return 0
  if grep -q "^${eid}" "$BL_FILE" 2>/dev/null; then
    log "blacklist already has $eid"
    return 0
  fi
  echo "$eid  # $note" >>"$BL_FILE"
  log "blacklist += $eid ($note)"
}

write_stamp() {
  local gpu=$1 host=$2 port=$3 eid=$4 ngpu=$5
  python3 - <<PY
import json
from pathlib import Path
from datetime import datetime, timezone
p = Path("$FLEET/artifacts") / "rented_${NAME}.json"
p.write_text(json.dumps({
  "name": "$NAME",
  "axis": "R1158",
  "gpu": "$gpu",
  "host": "$host",
  "port": int("$port"),
  "executor_id": "$eid",
  "ngpu": int("$ngpu"),
  "pass": $PASS,
  "note": "vera×Reason-GRPO re-rent p4281 (node-id + ngpu gate)",
  "utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
}, indent=2))
print(p)
PY
}

bootstrap() {
  local host=$1 port=$2
  log "bootstrap upload_and_launch host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$NAME" PASS="$PASS" \
    bash "$EXP/upload_and_launch.sh"
}

verify_or_reap() {
  local host=$1 port=$2
  local info eid api_g ssh_g
  info=$(pod_executor_and_api_gpus || true)
  eid=${info%% *}
  api_g=${info##* }
  ssh_g=$(ssh_gpu_count "$host" "$port" | tr -d '[:space:]')
  ssh_g=${ssh_g:-0}
  log "verify executor=$eid api_gpus=$api_g ssh_gpus=$ssh_g"
  if [[ -n "$eid" ]] && grep -q "^${eid}" "$BL_FILE" 2>/dev/null; then
    log "REAP blacklisted executor $eid"
    lium rm "$NAME" -y || true
    return 1
  fi
  if [[ "$ssh_g" -lt "$MIN_GPU" ]] || { [[ "$api_g" =~ ^[0-9]+$ ]] && [[ "$api_g" -lt "$MIN_GPU" ]]; }; then
    append_blacklist "$eid" "p4281: advertised 8×; ssh_gpus=$ssh_g api_gpus=$api_g host=$host; rm $NAME"
    log "REAP under-GPU ssh=$ssh_g api=$api_g"
    lium rm "$NAME" -y || true
    return 1
  fi
  echo "$eid $ssh_g"
  return 0
}

log "START name=$NAME cap=$CAP poll=${POLL_S}s max_iters=$MAX_ITERS (node-id rent + ngpu>=$MIN_GPU gate)"

if already_live; then
  log "already live — verify then bootstrap"
  if read -r host port < <(resolve_ssh); then
    if out=$(verify_or_reap "$host" "$port"); then
      eid=${out%% *}; ngpu=${out##* }
      write_stamp unknown "$host" "$port" "$eid" "$ngpu" || true
      bootstrap "$host" "$port" && exit 0
    fi
  fi
fi

for ((i=1; i<=MAX_ITERS; i++)); do
  if already_live; then
    log "appeared live mid-loop"
    if read -r host port < <(resolve_ssh); then
      if out=$(verify_or_reap "$host" "$port"); then
        eid=${out%% *}; ngpu=${out##* }
        write_stamp unknown "$host" "$port" "$eid" "$ngpu" || true
        bootstrap "$host" "$port" && exit 0
      fi
    fi
  fi
  n=$(mine_count || echo 99)
  if [[ "$n" -ge "$CAP" ]]; then
    log "iter=$i mine=$n>=cap sleep"
    sleep "$POLL_S"
    continue
  fi

  rented_gpu=""
  node=""
  mapfile -t b300_nodes < <(list_nodes B300 || true)
  mapfile -t b200_nodes < <(list_nodes B200 || true)
  if ((${#b300_nodes[@]})); then
    node=${b300_nodes[0]}; rented_gpu=B300
  elif ((${#b200_nodes[@]})); then
    node=${b200_nodes[0]}; rented_gpu=B200
  else
    if (( i % 15 == 0 )); then
      log "iter=$i ls-empty (post-blacklist) B300/B200×8 mine=$n/$CAP"
    fi
    sleep "$POLL_S"
    continue
  fi

  if ! try_rent_node "$node"; then
    log "rent fail node=$node"
    sleep "$POLL_S"
    continue
  fi

  log "RENT OK gpu=$rented_gpu node=$node — resolve+verify"
  sleep 10
  if ! read -r host port < <(resolve_ssh); then
    log "SSH resolve fail"
    sleep "$POLL_S"
    continue
  fi
  if ! out=$(verify_or_reap "$host" "$port"); then
    log "verify failed — continue polling"
    sleep "$POLL_S"
    continue
  fi
  eid=${out%% *}; ngpu=${out##* }
  write_stamp "$rented_gpu" "$host" "$port" "$eid" "$ngpu" || true
  if bootstrap "$host" "$port"; then
    log "DONE rented+bootstrapped gpu=$rented_gpu node=$node ngpu=$ngpu $host:$port"
    exit 0
  fi
  log "bootstrap failed — leave pod for next pass"
  exit 1
done

log "TIMEOUT after $MAX_ITERS iters"
exit 1
