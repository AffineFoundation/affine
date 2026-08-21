#!/usr/bin/env bash
# p4312: R1169 (:8003 GPUs1,3) + R1173 (:8004 GPUs4,5) REFUTE orphans on mine-r924
# → R1198 ShortCtx MidRank Loβ MidLR TRAIN GPUs1,3
# → R1199 ShortCtx MidRank MidLoβ MidLR TRAIN GPUs4,5
# Leave R1178 n80 LIVE on :8002 GPUs6,7. Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1198=r1198-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-midlr
EXP1199=r1199-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr
KH=/home/const/subnet120/mining/.ralph/known_hosts
SSH_OPTS=(-i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)
chmod +x "$ROOT/$EXP1198"/*.sh "$ROOT/$EXP1199"/*.sh

HOST=31.22.104.113
PORT=40300
echo "[p4312] sync R1198+R1199 → mine-r924"
ssh "${SSH_OPTS[@]}" -p "$PORT" "root@${HOST}" "mkdir -p /root/mining_src/$EXP1198 /root/mining_src/$EXP1199 /root/r1198 /root/r1199 /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$PORT" -r "$ROOT/$EXP1198"/. "root@${HOST}:/root/mining_src/$EXP1198/"
scp "${SSH_OPTS[@]}" -P "$PORT" -r "$ROOT/$EXP1199"/. "root@${HOST}:/root/mining_src/$EXP1199/"

ssh "${SSH_OPTS[@]}" -p "$PORT" "root@${HOST}" 'bash -s' <<'REMOTE'
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
# Stop stale waiters for R1169/R1173 only (not R1178)
for rid in r1169 r1173; do
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
for port_tok in "8003:r1169_merged" "8004:r1173_merged"; do
  port=${port_tok%%:*}; tok=${port_tok##*:}
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq "$tok"; then
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# UUID-clear GPUs 1,3 and 4,5 only — never 0/2 (T/K) or 6,7 (R1178)
python3 - <<'PY'
import os, signal, subprocess, time
want={1,3,4,5}
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
    if "r1178" in cmd or "r1178_merged" in cmd: continue
    kill.add(pid)
    print(f"gpu1,3,4,5 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 1,3,4,5 cleared")
PY
for i in $(seq 1 90); do
  used13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
  used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4312-r924] wait free GPUs1,3 used=$used13 GPUs4,5 used=$used45 iter=$i"
  [[ "$used13" -lt 16384 && "$used45" -lt 16384 ]] && break
  sleep 2
done
used13=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3 | awk '{s+=$1} END{print s+0}')
used45=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used13" -lt 16384 && "$used45" -lt 16384 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
# Confirm R1178 still alive
ss -lptn 'sport = :8002' | head -2 || true
ps -ef | grep -E 'r1178_sim|r1178_merged' | grep -v grep | head -3 || true
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
python3 - <<'PY'
import json, time
from pathlib import Path
for rid, note in [
  ("r1169", "REFUTE v4 ShortCtx LoRank Loβ MidLR ~0.32× → R1198 ShortCtx MidRank Loβ MidLR"),
  ("r1173", "REFUTE v4 MidCtx LoRank MidLoβ MidLR ~0.09× → R1199 ShortCtx MidRank MidLoβ MidLR"),
]:
  p = Path(f"/root/affine_data/{rid}_decision_reign36_wvk7.json")
  if p.is_file():
    d = json.loads(p.read_text())
    d["p4312_followup"] = note
    d["p4312_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    p.write_text(json.dumps(d, indent=2) + "\n")
    print(rid, "annotated", "margin", d.get("margin"))
PY
chmod +x /root/mining_src/r1198-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-midlr/*.sh
chmod +x /root/mining_src/r1199-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr/*.sh
bash /root/mining_src/r1198-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-midlr/lean_train_r924_gpus13_p4312.sh
bash /root/mining_src/r1199-vera-offline-dpo-hialpha-midrank-midlobeta-shortctx-hypersuperextrasteps-ep4-midlr/lean_train_r924_gpus45_p4312.sh
echo "[p4312-r924] launched"
echo R1198_PID=$(cat /root/logs/r1198_train.pid)
echo R1199_PID=$(cat /root/logs/r1199_train.pid)
ps -p "$(cat /root/logs/r1198_train.pid)" -o pid,etime,cmd 2>&1 | head -3
ps -p "$(cat /root/logs/r1199_train.pid)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1198_train_launched.json
cat /root/affine_data/r1199_train_launched.json
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# R1178 progress check
python3 -c 'import json;d=json.load(open("/root/affine_data/r1178_sim_progress_reign36_wvk7.json"));print("R1178",d)'
REMOTE

echo "[p4312] DONE launch"
