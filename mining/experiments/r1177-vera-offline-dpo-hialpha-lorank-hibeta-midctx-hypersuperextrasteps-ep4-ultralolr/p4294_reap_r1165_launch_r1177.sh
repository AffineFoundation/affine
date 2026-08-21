#!/usr/bin/env bash
# p4294: r338 R1165 REFUTE → exact-PID reap :8002 r1165_merged → R1177 MidCtx LoRank Hiβ UltraLoLR TRAIN
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1164 TRAIN GPUs4,5.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E177=r1177-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.253.90
PORT=40099
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E177"/*.sh
echo "[p4294] sync R1177 → mine-r338"
"${SSH[@]}" "mkdir -p /root/mining_src/$E177 /root/r1177 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E177"/. "root@${HOST}:/root/mining_src/$E177/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=${cmd:0:180}"
    if echo "$cmd" | grep -q "$tok"; then
      pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do alive=0; for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done; [[ $alive -eq 0 ]] && break; sleep 1; done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else echo "skip wrong-token pid=$pid"; fi
  else echo already gone pid=$pid; fi
}
# stop leftover R1165 waiters / lean / sim / chall (token-gated)
for tag in r1165; do
  for f in /root/logs/${tag}_merge_then_n80.pid /root/logs/${tag}_wait_merge.pid /root/logs/${tag}_sim_wvk7.pid /root/logs/vllm_chall_${tag}.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        # empty cmdline OK if dying; still kill by pidfile when token matches OR empty
        if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq "$tag"; then
          kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
        fi
      fi
    fi
  done
done
# port holders on 8002 if still serving r1165 merge
for port in 8002; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1165_merged'; then
      reap "$p" "r1165_merged"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on GPUs 6,7 only (preserve R1164 on 4,5 + T/K on 0-3)
python3 - <<'PY'
import os, signal, subprocess, time
want={6,7}
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
    if "r1164" in cmd or "r1177" in cmd: continue
    kill.add(pid)
    print(f"gpu6-7 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 6-7 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4294] wait free GPUs6,7 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy 6,7; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
ps -p "$(cat /root/logs/r1164_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3 || true
chmod +x /root/mining_src/r1177-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1177-vera-offline-dpo-hialpha-lorank-hibeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r338_gpus67_p4294.sh >/root/logs/p4294_r1177_outer.nohup 2>&1 &
echo $! >/root/logs/p4294_r1177_outer.pid
sleep 35
echo "=== verify ==="
ps -p "$(cat /root/logs/r1177_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
ps -p "$(cat /root/logs/r1164_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1177_train_launched.json 2>/dev/null || true
tail -30 /root/logs/r1177_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
REMOTE
