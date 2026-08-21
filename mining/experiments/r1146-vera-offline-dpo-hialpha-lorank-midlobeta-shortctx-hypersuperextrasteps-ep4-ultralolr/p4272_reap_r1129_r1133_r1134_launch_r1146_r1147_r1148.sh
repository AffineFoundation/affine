#!/usr/bin/env bash
# p4272: crown R1129+R1133+R1134 REFUTE → exact-PID reap :8002/:8004/:8003
# → R1146/R1147/R1148 UltraLoLR TRAIN on GPUs 6,7 / 1,3 / 4,5.
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1146=r1146-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-hypersuperextrasteps-ep4-ultralolr
EXP1147=r1147-vera-offline-dpo-hialpha-lorank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr
EXP1148=r1148-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.252.28
PORT=40298
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
for e in "$EXP1146" "$EXP1147" "$EXP1148"; do chmod +x "$ROOT/$e"/*.sh; done
echo "[p4272] sync R1146/47/48 → mine-crown-1"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP1146 /root/mining_src/$EXP1147 /root/mining_src/$EXP1148 /root/r1146 /root/r1147 /root/r1148 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP1146"/. "root@${HOST}:/root/mining_src/$EXP1146/"
"${SCP[@]}" -r "$ROOT/$EXP1147"/. "root@${HOST}:/root/mining_src/$EXP1147/"
"${SCP[@]}" -r "$ROOT/$EXP1148"/. "root@${HOST}:/root/mining_src/$EXP1148/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
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
    else echo FATAL wrong pid; exit 2; fi
  else echo already gone pid=$pid; fi
}
# stop leftover sims if any
for rid in r1129 r1133 r1134; do
  if [[ -f /root/logs/${rid}_sim_wvk7.pid ]]; then
    sp=$(cat /root/logs/${rid}_sim_wvk7.pid 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -q "$rid"; then kill "$sp" 2>/dev/null || true; sleep 2; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
# exact PID from pidfiles + port listeners
reap "$(cat /root/logs/vllm_chall_r1129.pid 2>/dev/null || echo 295628)" r1129_merged
reap "$(cat /root/logs/vllm_chall_r1133.pid 2>/dev/null || echo 302463)" r1133_merged
reap "$(cat /root/logs/vllm_chall_r1134.pid 2>/dev/null || echo 299265)" r1134_merged
for port in 8002 8003 8004; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1129_merged|r1133_merged|r1134_merged'; then
      tok=$(echo "$cmd" | grep -oE 'r11(29|33|34)_merged' | head -1)
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on train GPU pairs (skip TK)
python3 - <<'PY'
import os, signal, subprocess, time
want={1,3,4,5,6,7}
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
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd:
        if not any(x in cmd for x in ("r1129_merged","r1133_merged","r1134_merged")): continue
    kill.add(pid)
    print(f"gpu train-pair app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("train-pair apps cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3,4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4272] wait free train GPUs used=$used iter=$i"
  [[ "$used" -lt 49152 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,3,4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 49152 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
# keep TK healthy
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1146-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/*.sh
chmod +x /root/mining_src/r1147-vera-offline-dpo-hialpha-lorank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr/*.sh
chmod +x /root/mining_src/r1148-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1146-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_train_crown_gpus67_p4272.sh >/root/logs/p4272_r1146_outer.nohup 2>&1 &
echo $! >/root/logs/p4272_r1146_outer.pid
nohup bash /root/mining_src/r1147-vera-offline-dpo-hialpha-lorank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_train_crown_gpus13_p4272.sh >/root/logs/p4272_r1147_outer.nohup 2>&1 &
echo $! >/root/logs/p4272_r1147_outer.pid
nohup bash /root/mining_src/r1148-vera-offline-dpo-hialpha-midrank-lobeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_train_crown_gpus45_p4272.sh >/root/logs/p4272_r1148_outer.nohup 2>&1 &
echo $! >/root/logs/p4272_r1148_outer.pid
sleep 12
echo "=== verify ==="
for rid in r1146 r1147 r1148; do
  echo "-- $rid --"
  cat /root/logs/${rid}_train.pid 2>/dev/null || true
  head -8 /root/logs/${rid}_lean_warm.log 2>/dev/null || true
  ps -p "$(cat /root/logs/${rid}_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null || { echo FATAL $rid train not up; tail -40 /root/logs/${rid}_lean_warm.log; exit 1; }
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4272_r1129_r1133_r1134_refute_r1146_47_48_armed.done
echo "[p4272] R1146+R1147+R1148 TRAIN armed"
REMOTE
