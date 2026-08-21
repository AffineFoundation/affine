#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
EXP=r1170-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr
HOST=93.120.231.186
PORT=32301
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
echo "[p4291] sync R1170 → mine-r926"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/r1170 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if ! kill -0 "$pid" 2>/dev/null; then echo already gone pid=$pid; return 0; fi
  cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
  echo "reap pid=$pid cmd=$cmd"
  if [[ -z "$cmd" ]]; then
    # process dying; try kill anyway if token in cwd/environ
    kill "$pid" 2>/dev/null || true; sleep 2; kill -9 "$pid" 2>/dev/null || true; return 0
  fi
  if echo "$cmd" | grep -q "$tok"; then
    pids="$pid"; kids=$(pgrep -P "$pid" 2>/dev/null || true)
    for k in $kids; do pids="$pids $k"; for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done; done
    for p in $pids; do kill "$p" 2>/dev/null || true; done
    sleep 3
    for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
    echo reaped
  else echo FATAL wrong pid; exit 2; fi
}
# stop leftover r1157 scripts token-gated
for f in /root/logs/r1157_merge_then_n80.pid /root/logs/r1157_wait_merge.pid /root/logs/r1157_sim_wvk7.pid /root/logs/vllm_chall_r1157.pid; do
  [[ -f "$f" ]] || continue
  sp=$(cat "$f" 2>/dev/null || true)
  if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1157' || [[ -z "$cmd" ]]; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
  fi
done
reap "$(cat /root/logs/vllm_chall_r1157.pid 2>/dev/null || echo 170683)" r1157_merged
for port in 8002; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1157_merged' || [[ -z "$cmd" ]]; then reap "$p" r1157_merged || true; fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
python3 - <<'PY'
import os, signal, subprocess, time
want={3,4}
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
    if "vera6" in cmd and "vllm serve" in cmd and "r1157" not in cmd: continue
    kill.add(pid)
    print(f"gpu3,4 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 3,4 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
  echo "[p4291] wait free GPUs3,4 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3,4 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1170-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/*.sh
nohup bash /root/mining_src/r1170-cryptodev23-offline-dpo-hialpha-midrank-midlobeta-midctx-hypersuperextrasteps-ep4-midlr/lean_train_h100_gpus34_p4291.sh >/root/logs/p4291_r1170_outer.nohup 2>&1 &
echo $! >/root/logs/p4291_r1170_outer.pid
sleep 30
echo "=== verify ==="
ps -p "$(cat /root/logs/r1170_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
cat /root/affine_data/r1170_train_launched.json 2>/dev/null || true
tail -25 /root/logs/r1170_lean_warm.log 2>/dev/null || true
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
REMOTE
echo "[p4291] R1170 launched"
