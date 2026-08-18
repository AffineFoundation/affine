#!/usr/bin/env bash
# p3870: R798+R799 REFUTE → free crown 4,5+6,7; host-relay R784+R783 MERGE_DONE brave→crown; dual v4 n80.
# Never pkill -f. Keep teacher/king. Leave brave R800/R801 TRAIN alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r784_r783_brave_to_crown_p3870.log
: >"$LOG"
log() { echo "[p3870-relay] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
CROWN_HOST=95.133.253.90
CROWN_PORT=40099
LEAN784=$ROOT/experiments/r784-tammy-offline-dpo-hialpha-midrank-lobeta-softctx-megasuperextrasteps-ep4-ultralolr/lean_chall_n80_crown_gpus45_p3870.sh
LEAN783=$ROOT/experiments/r783-tammy-offline-dpo-hialpha-hirank-lobeta-softctx-megasuperextrasteps-ep4-lolr/lean_chall_n80_crown_gpus67_p3870.sh

log "START dual host-relay R784(:8002)+R783(:8003) brave→crown after R798/R799 REFUTE"

for hypo in 784 783; do
  n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
    "ls /tmp/r${hypo}_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
  log "brave r${hypo} shards=$n_src"
  [[ "$n_src" -ge 16 ]] || { log "FATAL r${hypo} source incomplete"; exit 1; }
done

ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
  "mkdir -p /root/mining_src/r784-chall /root/mining_src/r783-chall /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$CROWN_PORT" "$LEAN784" \
  "root@$CROWN_HOST:/root/mining_src/r784-chall/lean_chall_n80_crown_gpus45_p3870.sh"
scp "${SSH_OPTS[@]}" -P "$CROWN_PORT" "$LEAN783" \
  "root@$CROWN_HOST:/root/mining_src/r783-chall/lean_chall_n80_crown_gpus67_p3870.sh"

# Stage waiters + reap REFUTE challs + free disk on crown
ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
log(){ echo "[p3870-crown] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

stop_pid() {
  local pid=$1; local why=${2:-}
  [[ -n "${pid:-}" && "$pid" =~ ^[0-9]+$ ]] || return 0
  if kill -0 "$pid" 2>/dev/null; then
    log "kill pid=$pid ($why)"
    kill "$pid" 2>/dev/null || true
    for _ in $(seq 1 40); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    kill -9 "$pid" 2>/dev/null || true
  fi
}

# Reap R798/R799 challs by exact argv / port (never pkill -f)
for pidf in /root/logs/vllm_chall_r798.pid /root/logs/vllm_chall_r799.pid \
  /root/logs/r798_sim_wvk7.pid /root/logs/r799_sim_wvk7.pid; do
  [[ -f "$pidf" ]] || continue
  stop_pid "$(cat "$pidf" 2>/dev/null || true)" "pidf=$pidf"
  rm -f "$pidf"
done
while read -r pid; do
  [[ "$pid" =~ ^[0-9]+$ ]] || continue
  stop_pid "$pid" "r798/r799 vllm/sim"
done < <(ps -eo pid=,args= | awk '/vllm serve \/tmp\/r79[89]_merged|run_sim_duel.py.*r79[89]/ && !/awk/ {print $1}')
for port in 8002 8003; do
  while read -r pid; do
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    cmd=$(tr '\0' ' ' </proc/$pid/cmdline 2>/dev/null || true)
    echo "$cmd" | grep -qE 'GLM-4.5-Air|:8000|:8001' && continue
    stop_pid "$pid" "port $port"
  done < <(ss -lptn "sport = :$port" 2>/dev/null | grep -oP 'pid=\K[0-9]+' || true)
done
for i in $(seq 1 60); do
  used=$(nvidia-smi -i 4,5,6,7 --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END{print s+0}')
  log "wait free used457=$used iter=$i"
  [[ "${used:-999999}" -lt 16384 ]] && break
  sleep 2
done

# Free crown disk for dual 66G relays
for d in /tmp/r716_merged /tmp/r744_merged /tmp/r753_merged /tmp/r758_merged \
  /tmp/r763_merged /tmp/r775_merged /tmp/r776_merged /tmp/r786_merged \
  /tmp/r788_merged /tmp/r798_merged /tmp/r799_merged; do
  [[ -d "$d" ]] && rm -rf "$d" && log "freed $d"
done
rm -rf /tmp/r784_merged /tmp/r783_merged
rm -f /root/logs/r784_scp_ready.done /root/logs/r783_scp_ready.done
mkdir -p /tmp/r784_merged /tmp/r783_merged
df -h / | tail -1

cat > /root/mining_src/r784-chall/wait_r784_scp_then_chall_p3870.sh <<'WAIT784'
#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p3870_r784_wait_scp.log
: >"$LOG"
log(){ echo "[p3870-r784-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "waiting for /root/logs/r784_scp_ready.done"
for i in $(seq 1 900); do
  if [[ -f /root/logs/r784_scp_ready.done ]]; then
    n=$(ls /tmp/r784_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    log "SCP ready poll=$i shards=$n"
    [[ "${n:-0}" -ge 16 && -f /tmp/r784_merged/config.json ]] || { log "FATAL incomplete"; exit 2; }
    break
  fi
  (( i % 30 == 0 )) && log "still waiting iter=$i"
  sleep 10
done
[[ -f /root/logs/r784_scp_ready.done ]] || { log "FATAL timeout"; exit 3; }
chmod +x /root/mining_src/r784-chall/lean_chall_n80_crown_gpus45_p3870.sh
nohup bash /root/mining_src/r784-chall/lean_chall_n80_crown_gpus45_p3870.sh \
  >/root/logs/p3870_r784_lean.outer.log 2>&1 &
echo $! >/root/logs/p3870_r784_lean.outer.pid
log "armed lean chall pid=$(cat /root/logs/p3870_r784_lean.outer.pid)"
WAIT784

cat > /root/mining_src/r783-chall/wait_r783_scp_then_chall_p3870.sh <<'WAIT783'
#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p3870_r783_wait_scp.log
: >"$LOG"
log(){ echo "[p3870-r783-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "waiting for /root/logs/r783_scp_ready.done"
for i in $(seq 1 900); do
  if [[ -f /root/logs/r783_scp_ready.done ]]; then
    n=$(ls /tmp/r783_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    log "SCP ready poll=$i shards=$n"
    [[ "${n:-0}" -ge 16 && -f /tmp/r783_merged/config.json ]] || { log "FATAL incomplete"; exit 2; }
    break
  fi
  (( i % 30 == 0 )) && log "still waiting iter=$i"
  sleep 10
done
[[ -f /root/logs/r783_scp_ready.done ]] || { log "FATAL timeout"; exit 3; }
chmod +x /root/mining_src/r783-chall/lean_chall_n80_crown_gpus67_p3870.sh
nohup bash /root/mining_src/r783-chall/lean_chall_n80_crown_gpus67_p3870.sh \
  >/root/logs/p3870_r783_lean.outer.log 2>&1 &
echo $! >/root/logs/p3870_r783_lean.outer.pid
log "armed lean chall pid=$(cat /root/logs/p3870_r783_lean.outer.pid)"
WAIT783

chmod +x /root/mining_src/r784-chall/*.sh /root/mining_src/r783-chall/*.sh
bash -n /root/mining_src/r784-chall/lean_chall_n80_crown_gpus45_p3870.sh
bash -n /root/mining_src/r783-chall/lean_chall_n80_crown_gpus67_p3870.sh
bash -n /root/mining_src/r784-chall/wait_r784_scp_then_chall_p3870.sh
bash -n /root/mining_src/r783-chall/wait_r783_scp_then_chall_p3870.sh
echo SYNTAX_OK
nohup bash /root/mining_src/r784-chall/wait_r784_scp_then_chall_p3870.sh >/root/logs/p3870_r784_wait.nohup 2>&1 &
echo $! >/root/logs/p3870_r784_wait.pid
nohup bash /root/mining_src/r783-chall/wait_r783_scp_then_chall_p3870.sh >/root/logs/p3870_r783_wait.nohup 2>&1 &
echo $! >/root/logs/p3870_r783_wait.pid
echo "waiters r784=$(cat /root/logs/p3870_r784_wait.pid) r783=$(cat /root/logs/p3870_r783_wait.pid)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r798_r799_refute_reaped.p3870
REMOTE

log "begin parallel tar pipes brave→crown R784 + R783"
# R784 first (serial pipes avoid uplink thrash; parallel×2 still OK with care)
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r784_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r784_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r784_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r784_scp_ready.done && echo SCP_READY_R784 shards=\$n && du -sh /tmp/r784_merged"
rc784=$?
set -e
log "r784 tar rc=$rc784"
[[ "$rc784" -eq 0 ]] || { log "FATAL r784 tar failed"; exit 4; }

set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r783_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r783_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r783_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r783_scp_ready.done && echo SCP_READY_R783 shards=\$n && du -sh /tmp/r783_merged"
rc783=$?
set -e
log "r783 tar rc=$rc783"
[[ "$rc783" -eq 0 ]] || { log "FATAL r783 tar failed"; exit 5; }

log "DONE dual relay — watch p3870_r784_lean.outer.log + p3870_r783_lean.outer.log + decisions"
date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/host_relay_r784_r783_brave_to_crown_p3870.done"
