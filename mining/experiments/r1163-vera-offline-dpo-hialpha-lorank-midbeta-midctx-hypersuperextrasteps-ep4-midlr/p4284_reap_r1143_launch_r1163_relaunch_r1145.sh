#!/usr/bin/env bash
# p4284: r339 R1143 REFUTE → exact-PID reap :8002 → R1163 MidLR TRAIN on GPUs 4,5
# AND force-reseed Triton + relaunch R1145 n80 on GPUs 6,7 (prior chall died on broken .so).
# Never pkill -f. Do not touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
EXP=r1163-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-midlr
EXP1145=r1145-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=23.153.44.20
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$EXP"/*.sh
# patch R1145 lean: prefer king seed, never reuse broken chall_r1145 first
if grep -q 'chall_r1145 /root/.triton/cache/chall_r954' "$ROOT/$EXP1145/lean_chall_n80_r339_gpus67_p4271.sh"; then
  sed -i 's#for cand in /root/.triton/cache/chall_r1145 /root/.triton/cache/chall_r954#for cand in /root/.triton/cache/king /root/.triton/cache/chall_r1143 /root/.triton/cache/chall_r954#' \
    "$ROOT/$EXP1145/lean_chall_n80_r339_gpus67_p4271.sh"
fi
chmod +x "$ROOT/$EXP1145"/*.sh
echo "[p4284] sync R1163 + patched R1145 lean → mine-r339"
"${SSH[@]}" "mkdir -p /root/mining_src/$EXP /root/mining_src/$EXP1145 /root/r1163 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
"${SCP[@]}" "$ROOT/$EXP1145/lean_chall_n80_r339_gpus67_p4271.sh" "root@${HOST}:/root/mining_src/$EXP1145/lean_chall_n80_r339_gpus67_p4271.sh"
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
# stop leftover R1143 waiters
for f in /root/logs/r1143_merge_then_n80.pid /root/logs/r1143_wait_merge.pid /root/logs/r1143_sim_wvk7.pid; do
  if [[ -f "$f" ]]; then
    sp=$(cat "$f" 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -Eq 'r1143'; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
reap "$(cat /root/logs/vllm_chall_r1143.pid 2>/dev/null || echo 64315)" r1143_merged
for port in 8002; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1143_merged'; then
      reap "$p" r1143_merged
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on GPUs 4,5 only (skip TK + leave 6,7 for R1145)
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
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd: continue
    kill.add(pid)
    print(f"gpu4,5 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 4,5 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4284] wait free GPUs4,5 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL still busy; nvidia-smi; exit 1; }
curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null
curl -sf -m 5 http://127.0.0.1:8001/v1/models >/dev/null
chmod +x /root/mining_src/r1163-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-midlr/*.sh
nohup bash /root/mining_src/r1163-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-midlr/lean_train_r339_gpus45_p4284.sh >/root/logs/p4284_r1163_outer.nohup 2>&1 &
echo $! >/root/logs/p4284_r1163_outer.pid

# --- R1145 Triton fix + n80 relaunch on 6,7 ---
# wipe broken chall_r1145 cache; force seed from king
rm -rf /root/.triton/cache/chall_r1145
# stop any leftover r1145 waiters/sim
for f in /root/logs/r1145_merge_then_n80.pid /root/logs/r1145_sim_wvk7.pid /root/logs/vllm_chall_r1145.pid; do
  if [[ -f "$f" ]]; then
    sp=$(cat "$f" 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if echo "$cmd" | grep -Eq 'r1145'; then kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true; fi
    fi
  fi
done
# reap any leftover :8003
for port in 8003; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1145_merged'; then
      reap "$p" r1145_merged
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
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
    kill.add(pid)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(2)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 6,7 cleared for r1145 relaunch")
PY
# ensure merge still present
test -f /tmp/r1145_merged/config.json
n=$(ls /tmp/r1145_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
[[ "${n:-0}" -ge 16 ]] || { echo FATAL r1145 merge incomplete n=$n; exit 1; }
chmod +x /root/mining_src/r1145-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r339_gpus67_p4271.sh
nohup bash /root/mining_src/r1145-vera-offline-dpo-hialpha-lorank-midbeta-shortctx-hypersuperextrasteps-ep4-ultralolr/lean_chall_n80_r339_gpus67_p4271.sh >/root/logs/p4284_r1145_n80_relaunch.nohup 2>&1 &
echo $! >/root/logs/p4284_r1145_n80_relaunch.pid
sleep 25
echo "=== verify ==="
ps -p "$(cat /root/logs/r1163_train.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
ps -p "$(cat /root/logs/p4284_r1145_n80_relaunch.pid 2>/dev/null || echo 0)" -o pid,etime,cmd 2>&1 | head -3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
tail -20 /root/logs/r1163_lean_warm.log 2>/dev/null || true
tail -15 /root/logs/p4284_r1145_n80_relaunch.nohup 2>/dev/null || true
REMOTE
echo "[p4284] done"
