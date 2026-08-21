#!/usr/bin/env bash
# p4297: r338 R1164 REFUTE → exact-PID reap :8003 r1164_merged → R1180 MidCtx HiRank Loβ MidLR TRAIN
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1177 TRAIN GPUs6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E180=r1180-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-midlr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=95.133.253.90
PORT=40099
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E180"/*.sh
echo "[p4297] sync R1180 → mine-r338"
"${SSH[@]}" "mkdir -p /root/mining_src/$E180 /root/r1180 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E180"/. "root@${HOST}:/root/mining_src/$E180/"
"${SSH[@]}" 'bash -s' <<'REMOTE'
set -euo pipefail
reap() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid cmd=${cmd:0:180}"
    if [[ -z "$cmd" ]] || echo "$cmd" | grep -q "$tok"; then
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
# stop leftover R1164 waiters / lean / sim / chall (token-gated)
for tag in r1164; do
  for f in /root/logs/${tag}_merge_then_n80.pid /root/logs/${tag}_wait_merge.pid /root/logs/${tag}_sim_wvk7.pid /root/logs/vllm_chall_${tag}.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq "$tag"; then
          kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
        fi
      fi
    fi
  done
done
# port holders on 8003 if still serving r1164 merge
for port in 8003; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1164_merged'; then
      reap "$p" "r1164_merged"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on GPUs 4,5 only (preserve R1177 on 6,7 + T/K on 0-3)
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
    if "r1177" in cmd or "r1180" in cmd: continue
    kill.add(pid)
    print(f"gpu4-5 app pid={pid} cmd={cmd[:160]}")
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
print("gpus 4-5 cleared")
PY
for i in $(seq 1 90); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
  echo "[p4297] wait free GPUs4,5 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs4,5 still busy used=$used; exit 1; }
# seed data
mkdir -p /root/r1180
if [[ ! -s /root/r1180/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/r1180-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r1180-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl /root/r1180/dpo_duel_reason.jsonl
  elif [[ -s /root/r1164/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r1164/dpo_duel_reason.jsonl /root/r1180/dpo_duel_reason.jsonl
  fi
fi
chmod +x /root/mining_src/r1180-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-midlr/*.sh
bash /root/mining_src/r1180-vera-offline-dpo-hialpha-hirank-lobeta-midctx-hypersuperextrasteps-ep4-midlr/lean_train_r338_gpus45_p4297.sh
echo "[p4297] R1180 lean launched"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4297_r1164_refute_r1180_armed.done
cat /root/logs/r1180_lean_warm.log | tail -20
ps -eo pid,etime,cmd | awk '/[t]rain_dpo.*r1180|[r]1180_train/{print}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,6,7
REMOTE
echo "[p4297] DONE"
