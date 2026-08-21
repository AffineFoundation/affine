#!/usr/bin/env bash
# p4321: reap R1184/85 (r339) + R1186/1190 (r252) orphan challs → R1205-08 Mega MidLR TRAIN
# Never pkill -f. Never touch teacher:8000 / king:8001.
set -euo pipefail
ROOT=/home/const/subnet120/mining/experiments
KH=/home/const/subnet120/mining/.ralph/known_hosts
SSH_OPTS=(-i /home/const/.ssh/id_ed25519 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KH" -o ConnectTimeout=25 -o BatchMode=yes)

sync_exp() {
  local HOST="$1" PORT="$2" EXP="$3"
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "mkdir -p /root/mining_src/$EXP /root/logs /root/affine_data"
  scp "${SSH_OPTS[@]}" -P "$PORT" -r "$ROOT/$EXP"/. "root@${HOST}:/root/mining_src/$EXP/"
}

remote_reap_launch() {
  local HOST="$1" PORT="$2"
  shift 2
  # args: TOK:GPUS:EXP:LEAN  ...
  local SPECS="$*"
  ssh "${SSH_OPTS[@]}" -p "$PORT" "root@$HOST" "SPECS='$SPECS' bash -s" <<'REMOTE'
set -euo pipefail
reap_pid() {
  local pid="$1" tok="$2"
  if kill -0 "$pid" 2>/dev/null; then
    cmd=$(tr "\0" " " < /proc/$pid/cmdline 2>/dev/null || true)
    echo "reap pid=$pid tok=$tok cmd=${cmd:0:160}"
    if [[ -z "$cmd" ]] || echo "$cmd" | grep -q "$tok"; then
      pids="$pid"
      kids=$(pgrep -P "$pid" 2>/dev/null || true)
      for k in $kids; do
        pids="$pids $k"
        for g in $(pgrep -P "$k" 2>/dev/null || true); do pids="$pids $g"; done
      done
      echo "kill set: $pids"
      for p in $pids; do kill "$p" 2>/dev/null || true; done
      for i in $(seq 1 40); do
        alive=0
        for p in $pids; do kill -0 "$p" 2>/dev/null && alive=1; done
        [[ $alive -eq 0 ]] && break
        sleep 1
      done
      for p in $pids; do kill -9 "$p" 2>/dev/null || true; done
      echo reaped
    else
      echo "skip wrong-token pid=$pid"
    fi
  else
    echo already gone pid=$pid
  fi
}

for tok in r1184 r1185 r1186 r1190; do
  for f in /root/logs/${tok}_merge_then_n80.pid /root/logs/${tok}_wait_merge.pid /root/logs/${tok}_sim_wvk7.pid /root/logs/vllm_chall_${tok}.pid; do
    [[ -f "$f" ]] || continue
    sp=$(cat "$f" 2>/dev/null || true)
    if [[ -n "${sp:-}" ]] && kill -0 "$sp" 2>/dev/null; then
      cmd=$(tr "\0" " " < /proc/$sp/cmdline 2>/dev/null || true)
      if [[ -z "$cmd" ]] || echo "$cmd" | grep -Eq "$tok"; then
        kill "$sp" 2>/dev/null || true; sleep 1; kill -9 "$sp" 2>/dev/null || true
      fi
    fi
  done
done

for port_tok in "8002:r1184_merged" "8003:r1185_merged" "8002:r1190_merged" "8003:r1186_merged"; do
  port=${port_tok%%:*}; tok=${port_tok#*:}
  while read -r p; do
    [[ "$p" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr "\0" " " < /proc/$p/cmdline 2>/dev/null || true)
    if echo "$cmd" | grep -Eq "$tok"; then
      reap_pid "$p" "$tok"
    fi
  done < <(ss -lptn "sport = :$port" 2>/dev/null | sed -n 's/.*pid=\([0-9]*\).*/\1/p')
done

python3 - <<'PY'
import subprocess, os, signal, time
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
    if "GLM-4.5-Air" in cmd or ":8000" in cmd or ":8001" in cmd: continue
    kill.add(pid)
print("gpu_clear", sorted(kill))
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGTERM)
    except ProcessLookupError: pass
time.sleep(3)
for pid in sorted(kill):
    try: os.kill(pid, signal.SIGKILL)
    except ProcessLookupError: pass
PY
sleep 2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

for spec in $SPECS; do
  IFS=':' read -r TOK GPUS EXP LEAN <<<"$spec"
  echo "[p4321] launch $EXP on GPUs $GPUS lean=$LEAN"
  chmod +x /root/mining_src/$EXP/*.sh
  nohup bash "/root/mining_src/$EXP/$LEAN" >/root/logs/${EXP%%-*}_outer_p4321.nohup 2>&1 &
  echo $! > "/root/logs/${TOK}_replaced_outer.pid"
  echo "outer_pid=$!"
done
sleep 12
echo === AFTER ===
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
for rid in 1205 1206 1207 1208; do
  if [[ -f /root/logs/r${rid}_train.pid ]]; then
    tp=$(cat /root/logs/r${rid}_train.pid)
    echo "r${rid}_train.pid=$tp alive=$(kill -0 $tp 2>/dev/null && echo Y || echo N)"
    tail -8 /root/logs/r${rid}_lean_warm.log 2>/dev/null || true
    cat /root/affine_data/r${rid}_train_launched.json 2>/dev/null || true
  fi
done
REMOTE
}

echo "[p4321] sync r339 exps"
sync_exp 23.153.44.20 40299 r1205-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-midlr
sync_exp 23.153.44.20 40299 r1206-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-megasuperextrasteps-ep4-midlr
echo "[p4321] reap+launch r339"
remote_reap_launch 23.153.44.20 40299 \
  "r1184:4,5:r1205-vera-offline-dpo-hialpha-midrank-midbeta-midctx-megasuperextrasteps-ep4-midlr:lean_train_r339_gpus45_p4321.sh" \
  "r1185:6,7:r1206-vera-offline-dpo-hialpha-lorank-midlobeta-shortctx-megasuperextrasteps-ep4-midlr:lean_train_r339_gpus67_p4321.sh"

echo "[p4321] sync r252 exps"
sync_exp 38.127.229.127 40299 r1207-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-midlr
sync_exp 38.127.229.127 40299 r1208-vera-offline-dpo-hialpha-lorank-hibeta-midctx-megasuperextrasteps-ep4-midlr
echo "[p4321] reap+launch r252"
remote_reap_launch 38.127.229.127 40299 \
  "r1190:4,5:r1208-vera-offline-dpo-hialpha-lorank-hibeta-midctx-megasuperextrasteps-ep4-midlr:lean_train_r252_gpus45_p4321.sh" \
  "r1186:6,7:r1207-vera-offline-dpo-hialpha-hirank-midlobeta-softctx-megasuperextrasteps-ep4-midlr:lean_train_r252_gpus67_p4321.sh"

echo "[p4321] DONE"
