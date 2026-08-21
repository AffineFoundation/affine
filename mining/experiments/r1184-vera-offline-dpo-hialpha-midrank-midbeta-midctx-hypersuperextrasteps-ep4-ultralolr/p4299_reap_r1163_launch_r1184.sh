#!/usr/bin/env bash
# p4299: r339 R1163 REFUTE → exact-PID reap :8002 r1163_merged → R1184 MidCtx MidRank Midβ UltraLoLR TRAIN
# Never pkill -f. Do not touch teacher:8000 / king:8001 / R1166 TRAIN GPUs6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
source /home/const/subnet120/.venv/bin/activate
cd /home/const/subnet120/mining
E184=r1184-vera-offline-dpo-hialpha-midrank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr
KH=/home/const/subnet120/mining/.ralph/known_hosts
HOST=23.153.44.20
PORT=40299
SSH=(ssh -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" root@"$HOST")
SCP=(scp -i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes -P "$PORT")
chmod +x "$ROOT/$E184"/*.sh
echo "[p4299] sync R1184 → mine-r339"
"${SSH[@]}" "mkdir -p /root/mining_src/$E184 /root/r1184 /root/logs /root/affine_data"
"${SCP[@]}" -r "$ROOT/$E184"/. "root@${HOST}:/root/mining_src/$E184/"
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
# stop leftover R1163 waiters / lean / sim / chall (token-gated)
for tag in r1163; do
  for f in /root/logs/${tag}_merge_then_n80.pid /root/logs/${tag}_wait_merge.pid /root/logs/${tag}_sim_wvk7.pid /root/logs/vllm_chall_${tag}.pid /root/logs/p4298_r1163_chall_n80_relaunch.pid /root/logs/p4284_r1163_outer.pid; do
    if [[ -f "$f" ]]; then
      sp=$(cat "$f" 2>/dev/null || true)
      if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
        cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
        if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq "$tag|r1163"; then
          kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
        fi
      fi
    fi
  done
done
# port holders on 8002 if still serving r1163 merge
for port in 8002; do
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq 'r1163_merged'; then
      reap "$p" "r1163_merged"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done
# clear leftover compute apps on GPUs 4,5 only (preserve R1166 on 6,7 + T/K on 0-3)
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
    if "r1166" in cmd or "r1184" in cmd: continue
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
  echo "[p4299] wait free GPUs4,5 used=$used iter=$i"
  [[ "$used" -lt 8192 ]] && break
  sleep 2
done
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 4,5 | awk '{s+=$1} END{print s+0}')
[[ "$used" -lt 8192 ]] || { echo FATAL GPUs4,5 still busy used=$used; exit 1; }
# seed data
mkdir -p /root/r1184 /root/r1163
if [[ ! -s /root/r1184/dpo_duel_reason.jsonl ]]; then
  if [[ -s /root/mining_src/r1184-vera-offline-dpo-hialpha-midrank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r1184-vera-offline-dpo-hialpha-midrank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/dpo_duel_reason.jsonl /root/r1184/dpo_duel_reason.jsonl
  elif [[ -s /root/r1163/dpo_duel_reason.jsonl ]]; then
    cp -f /root/r1163/dpo_duel_reason.jsonl /root/r1184/dpo_duel_reason.jsonl
  elif [[ -s /root/mining_src/r1163-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl ]]; then
    cp -f /root/mining_src/r1163-vera-offline-dpo-hialpha-lorank-midbeta-midctx-hypersuperextrasteps-ep4-midlr/dpo_duel_reason.jsonl /root/r1184/dpo_duel_reason.jsonl
  fi
fi
chmod +x /root/mining_src/r1184-vera-offline-dpo-hialpha-midrank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/*.sh
bash /root/mining_src/r1184-vera-offline-dpo-hialpha-midrank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr/lean_train_r339_gpus45_p4299.sh
echo "[p4299] R1184 lean launched"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p4299_r1163_refute_r1184_armed.done
cat /root/logs/r1184_lean_warm.log | tail -25
ps -eo pid,etime,cmd | awk '/[t]rain_dpo.*r1184|[r]1184_train/{print}'
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i 4,5,6,7
REMOTE
echo "[p4299] DONE"
