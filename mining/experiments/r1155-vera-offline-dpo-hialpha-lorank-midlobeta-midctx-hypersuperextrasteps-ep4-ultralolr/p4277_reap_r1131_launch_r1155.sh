#!/usr/bin/env bash
# p4277: R1131 REFUTE → exact-PID reap r924 :8004 → R1155 UltraLoLR TRAIN GPUs4,5
# Never pkill -f. Do not touch teacher/king, R1154 on 1,3, or R1139 on 6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
EXP=r1155-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=31.22.104.113
PORT=40300
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1155 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid tok=$tok cmd=$cmd"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"
      kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do
        alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break; sleep 1
      done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone pid=$pid; fi
}
for pf in /root/logs/vllm_chall_r1131.pid /root/logs/r1131_sim_wvk7.pid /root/logs/r1131_merge_then_n80.pid /root/logs/p4260_r1131_lean_outer.pid; do
  [[ -f "$pf" ]] || continue
  p=$(cat "$pf" 2>/dev/null || true)
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -Eq 'r1131|8004'; then
    if echo "$cmd" | grep -q r1131_merged; then reap "$p" r1131_merged
    elif echo "$cmd" | grep -q run_sim_duel; then reap "$p" r1131
    elif echo "$cmd" | grep -q lean_chall; then reap "$p" r1131
    elif echo "$cmd" | grep -q wait_r1131; then reap "$p" r1131
    fi
  fi
done
while read -r p; do
  [[ "$p" =~ ^[0-9]+$ ]] || continue
  cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1131_merged; then reap "$p" r1131_merged; fi
done < <(ss -lptn 'sport = :8004' 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
# known APIServer pid 156139 if still r1131
if kill -0 156139 2>/dev/null; then
  cmd=$(tr "\0" " " < /proc/156139/cmdline 2>/dev/null || true)
  if echo "$cmd" | grep -q r1131_merged; then reap 156139 r1131_merged; fi
fi
# clear leftover EngineCore/Worker on GPUs 4,5 that belong to r1131 only
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5}
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
    if "train_dpo" in cmd or "train_online" in cmd: continue
    if "r1154" in cmd or "r1139" in cmd: continue
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if "r1131_merged" not in cmd: continue
    # workers often have empty cmdline beyond VLLM::Worker — kill only if parent chain is r1131
    kill.add(pid)
    print(f"gpu45 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpu45 apps cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4277] wait GPUs4,5 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi -i 4,5; exit 1; }
# protect siblings
if ! kill -0 "$(cat /root/logs/r1154_train.pid 2>/dev/null)" 2>/dev/null; then echo WARN R1154 train not alive; fi
if ! kill -0 "$(cat /root/logs/r1139_train.pid 2>/dev/null)" 2>/dev/null; then echo WARN R1139 train not alive; fi
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1155-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1155-vera-offline-dpo-hialpha-lorank-midlobeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r924_gpus45_p4277.sh \
  >/root/logs/p4277_r1155_outer.nohup 2>&1 &
echo $! >/root/logs/p4277_r1155_outer.pid
sleep 15
echo "=== verify R1155 ==="
cat /root/logs/r1155_train.pid 2>/dev/null || true
head -20 /root/logs/r1155_lean_warm.log 2>/dev/null || true
ps -p "$(cat /root/logs/r1155_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null \
  || { echo FATAL train not up; tail -60 /root/logs/r1155_lean_warm.log; exit 1; }
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4277_r1131_refute_r1155_armed.done
echo "[p4277] R1155 UltraLoLR TRAIN armed"
REMOTE
