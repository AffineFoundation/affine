#!/usr/bin/env bash
# p4311: R1179 (crown :8004) + R1174 (r938 :8002) REFUTE orphans
# → R1196 SoftCtx MidRank Loβ UltraLoLR TRAIN crown GPUs1,3
# → R1197 ShortCtx HiRank Loβ UltraLoLR TRAIN r938 GPUs2,3
# Never pkill -f. Do not touch teacher:8000 / king:8001 or sibling trains.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1196=r1196-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr
EXP1197=r1197-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
SSH_OPTS=(-i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)
chmod +x "$ROOT/$EXP1196"/*.sh "$ROOT/$EXP1197"/*.sh

# ---------- CROWN: R1179 → R1196 ----------
CROWN_HOST=95.133.252.28
CROWN_PORT=40298
echo "[p4311] sync R1196 → mine-crown-1"
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@${CROWN_HOST}" "mkdir -p /root/mining_src/$EXP1196 /root/r1196 /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$CROWN_PORT" -r "$ROOT/$EXP1196"/. "root@${CROWN_HOST}:/root/mining_src/$EXP1196/"
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@${CROWN_HOST}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid tok=$tok; exit 2; fi
  else echo already gone pid=$pid; fi
}
for rid in r1179; do
  for f in /root/logs/${rid}_merge_then_n80.pid /root/logs/${rid}_wait_merge.pid /root/logs/${rid}_sim_wvk7.pid /root/logs/vllm_chall_${rid}.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if echo "$cmd" | grep -Eq "$rid"; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
      fi
    fi
  done
done
for port_tok in "8004:r1179_merged"; do
  port=${port_tok%%:*}; tok=${port_tok##*:}
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq "$tok"; then
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
python3 - <<'PY'
import os, signal, subprocess, time
want={1,3}
out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    parts=[p.strip() for p in line.split(",")]
    if len(parts)>=2: idx_to_uuid[int(parts[0])]=parts[1]
uuids={idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],text=True)
kill=set()
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in uuids: continue
    try: pid=int(parts[1])
    except ValueError: continue
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd: continue
    if "r1188" in cmd or "r1189" in cmd: continue
    kill.add(pid)
    print(f"gpu1,3 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 1,3 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4311-crown] wait free GPUs1,3 used=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1196-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr/*.sh
python3 - <<'PY'
import json, time
from pathlib import Path
p = Path("/root/affine_data/r1179_decision_reign36_wvk7.json")
if p.is_file():
    d = json.loads(p.read_text())
    d["p4311_followup"] = "REFUTE v4 SoftCtx MidRank Loβ MidLR ~-0.31× → R1196 SoftCtx MidRank Loβ UltraLoLR"
    d["p4311_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(d, indent=2) + "\n")
    print("R1179 annotated", "wins", d.get("wins"), "margin", d.get("margin"))
PY
bash /root/mining_src/r1196-vera-offline-dpo-hialpha-midrank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_train_crown_gpus13_p4311.sh
echo "[p4311-crown] launched"
ps -p "$(cat /root/logs/r1196_train.pid)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1196_train_launched.json
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
REMOTE

# ---------- R938: R1174 → R1197 ----------
R938_HOST=38.255.28.21
R938_PORT=20100
echo "[p4311] sync R1197 → mine-r938"
ssh "${SSH_OPTS[@]}" -p "$R938_PORT" "root@${R938_HOST}" "mkdir -p /root/mining_src/$EXP1197 /root/r1197 /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$R938_PORT" -r "$ROOT/$EXP1197"/. "root@${R938_HOST}:/root/mining_src/$EXP1197/"
ssh "${SSH_OPTS[@]}" -p "$R938_PORT" "root@${R938_HOST}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=$cmd"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid tok=$tok; exit 2; fi
  else echo already gone pid=$pid; fi
}
for rid in r1174; do
  for f in /root/logs/${rid}_merge_then_n80.pid /root/logs/${rid}_wait_merge.pid /root/logs/${rid}_sim_wvk7.pid /root/logs/vllm_chall_${rid}.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if echo "$cmd" | grep -Eq "$rid"; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
      fi
    fi
  done
done
for port_tok in "8002:r1174_merged"; do
  port=${port_tok%%:*}; tok=${port_tok##*:}
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq "$tok"; then
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
python3 - <<'PY'
import os, signal, subprocess, time
want={2,3}
out=subprocess.check_output(["nvidia-smi","--query-gpu=index,uuid","--format=csv,noheader"],text=True)
idx_to_uuid={}
for line in out.strip().splitlines():
    parts=[p.strip() for p in line.split(",")]
    if len(parts)>=2: idx_to_uuid[int(parts[0])]=parts[1]
uuids={idx_to_uuid[i] for i in want if i in idx_to_uuid}
apps=subprocess.check_output(["nvidia-smi","--query-compute-apps=gpu_uuid,pid,process_name","--format=csv,noheader"],text=True)
kill=set()
for line in apps.strip().splitlines():
    if not line.strip(): continue
    parts=[p.strip() for p in line.split(",")]
    if len(parts)<2 or parts[0] not in uuids: continue
    try: pid=int(parts[1])
    except ValueError: continue
    try: cmd=open(f"/proc/{pid}/cmdline","rb").read().decode("utf-8","replace")
    except Exception: cmd=""
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd: continue
    kill.add(pid)
    print(f"gpu2,3 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 2,3 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
  echo "[p4311-r938] wait free GPUs2,3 used=$used iter=$i"
  [[ "$used" -lt 16384 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2,3 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 16384 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1197-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/*.sh
python3 - <<'PY'
import json, time
from pathlib import Path
p = Path("/root/affine_data/r1174_decision_reign36_wvk7.json")
if p.is_file():
    d = json.loads(p.read_text())
    d["p4311_followup"] = "REFUTE v4 ShortCtx HiRank Midβ UltraLoLR ~0.03×; Midβ LR exhausted → R1197 ShortCtx HiRank Loβ UltraLoLR"
    d["p4311_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(d, indent=2) + "\n")
    print("R1174 annotated", "wins", d.get("wins"), "margin", d.get("margin"))
PY
bash /root/mining_src/r1197-vera-offline-dpo-hialpha-hirank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r938_gpus23_p4311.sh
echo "[p4311-r938] launched"
ps -p "$(cat /root/logs/r1197_train.pid)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1197_train_launched.json
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
REMOTE

echo "[p4311] done"
