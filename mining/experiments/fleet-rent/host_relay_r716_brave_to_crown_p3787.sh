#!/usr/bin/env bash
# p3787: R715 REFUTE → free crown 4,5; host-relay R716 MERGE_DONE brave→crown; lean n80 vs reign35.
# Never pkill -f. Keep R744 TRAIN on 6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r716_brave_to_crown_p3787.log
: >"$LOG"
log() { echo "[p3787-r716] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
CROWN_HOST=95.133.253.90
CROWN_PORT=40099
LEAN_LOCAL=$ROOT/experiments/fleet-rent/lean_chall_n80_crown_r716_gpus45_p3787.sh

log "START R716 host-relay brave→crown after R715 REFUTE"

n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r716_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r716 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" "mkdir -p /root/mining_src/r716-chall /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$CROWN_PORT" "$LEAN_LOCAL" \
  "root@$CROWN_HOST:/root/mining_src/r716-chall/lean_chall_n80_crown_r716_gpus45_p3787.sh"

ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" 'cat > /root/mining_src/r716-chall/wait_r716_scp_then_chall_p3787.sh' <<'WAIT'
#!/usr/bin/env bash
set -euo pipefail
LOG=/root/logs/p3787_r716_wait_scp.log
: >"$LOG"
log(){ echo "[p3787-r716-wait] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }
log "waiting for /root/logs/r716_scp_ready.done"
for i in $(seq 1 720); do
  if [[ -f /root/logs/r716_scp_ready.done ]]; then
    n=$(ls /tmp/r716_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || true)
    log "SCP ready poll=$i shards=$n"
    [[ "${n:-0}" -ge 16 && -f /tmp/r716_merged/config.json ]] || { log "FATAL incomplete after ready"; exit 2; }
    break
  fi
  (( i % 30 == 0 )) && log "still waiting iter=$i"
  sleep 10
done
[[ -f /root/logs/r716_scp_ready.done ]] || { log "FATAL timeout"; exit 3; }
chmod +x /root/mining_src/r716-chall/lean_chall_n80_crown_r716_gpus45_p3787.sh
nohup bash /root/mining_src/r716-chall/lean_chall_n80_crown_r716_gpus45_p3787.sh \
  >/root/logs/p3787_r716_lean.outer.log 2>&1 &
echo $! >/root/logs/p3787_r716_lean.outer.pid
log "armed lean chall pid=$(cat /root/logs/p3787_r716_lean.outer.pid)"
date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/r716_chall_n80_launched.p3787
WAIT

ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
chmod +x /root/mining_src/r716-chall/*.sh
bash -n /root/mining_src/r716-chall/lean_chall_n80_crown_r716_gpus45_p3787.sh
bash -n /root/mining_src/r716-chall/wait_r716_scp_then_chall_p3787.sh
echo SYNTAX_OK
# free disk: refute merges
for d in /tmp/r715_merged /tmp/r737_merged /tmp/r726_merged /tmp/r727_merged; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "freed $d"
done
rm -rf /tmp/r716_merged
rm -f /root/logs/r716_scp_ready.done /root/logs/r716_chall_n80_launched.p3787
mkdir -p /tmp/r716_merged
df -h / | tail -1
# confirm GPUs 4,5 free and R744 alive
nvidia-smi -i 4,5 --query-gpu=memory.used --format=csv,noheader
pgrep -af 'r744' | head -3 || true
nohup bash /root/mining_src/r716-chall/wait_r716_scp_then_chall_p3787.sh \
  >/root/logs/p3787_r716_wait.nohup 2>&1 &
echo $! >/root/logs/p3787_r716_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3787_r716_wait.pid)"
REMOTE

log "begin tar pipe brave→crown ~66G R716"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r716_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$CROWN_PORT" "root@$CROWN_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r716_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r716_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r716_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r716_merged"
rc=$?
set -e
log "tar pipe rc=$rc"
[[ "$rc" -eq 0 ]] || { log "FATAL tar pipe failed"; exit 4; }
log "DONE relay — watch p3787_r716_lean.outer.log + r716_sim_*_reign35_wvk7.json"
