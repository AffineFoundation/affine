#!/usr/bin/env bash
# p4273: r337 R1125+R1137 REFUTE idle → R1149+R1150 UltraLoLR TRAIN
#        r338 R1135+R1136 REFUTE idle → R1151+R1152 UltraLoLR TRAIN
# Exact-PID reap chall :8002/:8003 only. Never pkill -f. Do not touch teacher/king.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP1149=r1149-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-ultralolr
EXP1150=r1150-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-hypersuperextrasteps-ep4-ultralolr
EXP1151=r1151-vera-offline-dpo-hialpha-hirank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr
EXP1152=r1152-vera-offline-dpo-hialpha-lorank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
SSH_BASE=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)
SCP_BASE=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)

for e in "$EXP1149" "$EXP1150" "$EXP1151" "$EXP1152"; do chmod +x "$ROOT/$e"/*.sh; done

launch_pod() {
  local label="$1" host="$2" port="$3"
  shift 3
  local exps=("$@")
  local SSH=( "${SSH_BASE[@]}" -p "$port" "root@$host" )
  local SCP=( "${SCP_BASE[@]}" -P "$port" )
  echo "[p4273] sync → $label $host:$port exps=${exps[*]}"
  local mkdir_args=""
  for e in "${exps[@]}"; do mkdir_args+=" /root/mining_src/$e"; done
  # shellcheck disable=SC2086
  "${SSH[@]}" "mkdir -p $mkdir_args /root/r1149 /root/r1150 /root/r1151 /root/r1152 /root/logs /root/affine_data"
  for e in "${exps[@]}"; do
    "${SCP[@]}" -r "$ROOT/$e"/. "root@${host}:/root/mining_src/$e/"
  done
}

# --- r337: R1125→R1149 GPUs6,7 :8002 ; R1137→R1150 GPUs4,5 :8003 ---
launch_pod mine-r337 150.136.46.118 20300 "$EXP1149" "$EXP1150"
"${SSH_BASE[@]}" -p 20300 root@150.136.46.118 'bash -s' <<'REMOTE337'
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
for rid in r1125 r1137; do
  if [[ -f /root/logs/${rid}_sim_wvk7.pid ]]; then
    sp=$(cat /root/logs/${rid}_sim_wvk7.pid 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -q "$rid"; then kill "$sp" 2>/dev/null || true; sleep 2; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
reap "$(cat /root/logs/vllm_chall_r1125.pid 2>/dev/null || echo 160015)" r1125_merged
reap "$(cat /root/logs/vllm_chall_r1137.pid 2>/dev/null || echo 163316)" r1137_merged
for port in 8002 8003; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1125_merged|r1137_merged'; then
      tok=$(echo "$cmd" | grep -oE 'r11(25|37)_merged' | head -1)
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5,6,7}
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
        if not any(x in cmd for x in ("r1125_merged","r1137_merged")): continue
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
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4273-r337] wait free train GPUs used=$used iter=$i"
  [[ "$used" -lt 49152 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 49152 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1149-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-ultralolr/*.sh
chmod +x /root/mining_src/r1150-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1149-vera-offline-dpo-hialpha-lorank-midbeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r337_gpus67_p4273.sh >/root/logs/p4273_r1149_outer.nohup 2>&1 &
echo $! >/root/logs/p4273_r1149_outer.pid
nohup bash /root/mining_src/r1150-vera-offline-dpo-hialpha-midrank-midbeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r337_gpus45_p4273.sh >/root/logs/p4273_r1150_outer.nohup 2>&1 &
echo $! >/root/logs/p4273_r1150_outer.pid
sleep 12
echo "=== verify r337 ==="
for rid in r1149 r1150; do
  echo "-- $rid --"
  cat /root/logs/${rid}_train.pid 2>/dev/null || true
  head -8 /root/logs/${rid}_lean_warm.log 2>/dev/null || true
  ps -p "$(cat /root/logs/${rid}_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null || { echo FATAL $rid train not up; tail -40 /root/logs/${rid}_lean_warm.log; exit 1; }
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4273_r1125_r1137_refute_r1149_r1150_armed.done
echo "[p4273] R1149+R1150 TRAIN armed on r337"
REMOTE337

# --- r338: R1136→R1152 GPUs6,7 :8002 ; R1135→R1151 GPUs4,5 :8003 ---
launch_pod mine-r338 95.133.253.90 40099 "$EXP1151" "$EXP1152"
"${SSH_BASE[@]}" -p 40099 root@95.133.253.90 'bash -s' <<'REMOTE338'
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
for rid in r1135 r1136; do
  if [[ -f /root/logs/${rid}_sim_wvk7.pid ]]; then
    sp=$(cat /root/logs/${rid}_sim_wvk7.pid 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -q "$rid"; then kill "$sp" 2>/dev/null || true; sleep 2; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
reap "$(cat /root/logs/vllm_chall_r1136.pid 2>/dev/null || echo 205265)" r1136_merged
reap "$(cat /root/logs/vllm_chall_r1135.pid 2>/dev/null || echo 208591)" r1135_merged
for port in 8002 8003; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1135_merged|r1136_merged'; then
      tok=$(echo "$cmd" | grep -oE 'r113[56]_merged' | head -1)
      reap "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
python3 - <<'PY'
import os, signal, subprocess, time
want={4,5,6,7}
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
        if not any(x in cmd for x in ("r1135_merged","r1136_merged")): continue
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
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
  echo "[p4273-r338] wait free train GPUs used=$used iter=$i"
  [[ "$used" -lt 49152 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5,6,7 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 49152 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1151-vera-offline-dpo-hialpha-hirank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr/*.sh
chmod +x /root/mining_src/r1152-vera-offline-dpo-hialpha-lorank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
nohup bash /root/mining_src/r1151-vera-offline-dpo-hialpha-hirank-lobeta-softctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r338_gpus45_p4273.sh >/root/logs/p4273_r1151_outer.nohup 2>&1 &
echo $! >/root/logs/p4273_r1151_outer.pid
nohup bash /root/mining_src/r1152-vera-offline-dpo-hialpha-lorank-lobeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r338_gpus67_p4273.sh >/root/logs/p4273_r1152_outer.nohup 2>&1 &
echo $! >/root/logs/p4273_r1152_outer.pid
sleep 12
echo "=== verify r338 ==="
for rid in r1151 r1152; do
  echo "-- $rid --"
  cat /root/logs/${rid}_train.pid 2>/dev/null || true
  head -8 /root/logs/${rid}_lean_warm.log 2>/dev/null || true
  ps -p "$(cat /root/logs/${rid}_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>/dev/null || { echo FATAL $rid train not up; tail -40 /root/logs/${rid}_lean_warm.log; exit 1; }
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4273_r1135_r1136_refute_r1151_r1152_armed.done
echo "[p4273] R1151+R1152 TRAIN armed on r338"
REMOTE338

echo "[p4273] ALL ARMED R1149+R1150@r337 R1151+R1152@r338"
