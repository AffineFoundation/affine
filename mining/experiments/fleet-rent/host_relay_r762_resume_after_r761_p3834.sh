#!/usr/bin/env bash
# p3834: R762 MERGE_DONE sat idle on brave → wait R761 SCP_READY (solo uplink),
# then host-relay brave→R252; waiter queues n80 on GPUs 4,5 after R761 n80.
# Never pkill -f. Do not dual-pipe with live R761 tar. Leave R782 6,7 alone.
set -euo pipefail
ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r762-r252-offline-dpo-hialpha-midrank-hibeta-softctx-megasuperextrasteps-ep4-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r762_resume_after_r761_p3834.log
: >"$LOG"
log() { echo "[p3834-r762] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299

log "armed: wait R761 SCP_READY on R252 then solo-relay R762 (timeout 6h)"
for i in $(seq 1 1440); do
  ready=$(ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    'if [[ -f /root/logs/r761_scp_ready.done ]] && [[ -f /tmp/r761_merged/config.json ]]; then
       n=$(ls /tmp/r761_merged/model-*-of-*.safetensors 2>/dev/null | wc -l); echo "ok:$n"
     else
       n=$(ls /tmp/r761_merged/model-*-of-*.safetensors 2>/dev/null | wc -l || echo 0); echo "wait:$n"
     fi' 2>/dev/null || echo "sshfail")
  if [[ "$ready" == ok:* ]]; then
    shards=${ready#ok:}
    if [[ "${shards:-0}" -ge 16 ]]; then
      log "R761 SCP_READY shards=$shards — begin R762 relay"
      break
    fi
  fi
  [[ $((i % 6)) -eq 0 ]] && log "poll i=$i r761=$ready"
  if [[ "$i" -eq 1440 ]]; then
    log "TIMEOUT waiting R761 SCP_READY"
    exit 1
  fi
  sleep 15
done

# Ensure R761 host tar pipe has exited (no dual uplink)
for i in $(seq 1 120); do
  if ! pgrep -f 'host_relay_r761_brave_to_r252_p3830' >/dev/null 2>&1 \
     && ! pgrep -f 'tar cf - r761_merged' >/dev/null 2>&1; then
    log "R761 host tar/pipe gone — uplink free"
    break
  fi
  [[ $((i % 6)) -eq 0 ]] && log "wait R761 host pipe exit poll=$i"
  sleep 10
done

n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r762_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r762 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
  "mkdir -p /root/mining_src/r762-chall /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$R252_PORT" \
  "$EXP/lean_chall_n80_r252_gpus45_p3834.sh" \
  "$EXP/wait_r762_after_r761_then_n80_p3834.sh" \
  "root@$R252_HOST:/root/mining_src/r762-chall/"

ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
chmod +x /root/mining_src/r762-chall/*.sh
bash -n /root/mining_src/r762-chall/lean_chall_n80_r252_gpus45_p3834.sh
bash -n /root/mining_src/r762-chall/wait_r762_after_r761_then_n80_p3834.sh
echo SYNTAX_OK
# free old refute merges only; KEEP /tmp/r761_merged until its n80 finishes
for d in /tmp/r759_merged /tmp/r760_merged /tmp/r751_merged /tmp/r750_merged \
         /tmp/r743_merged /tmp/r742_merged /tmp/r735_merged /tmp/r734_merged \
         /tmp/r769_merged /tmp/r770_merged; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "freed $d"
done
rm -rf /tmp/r762_merged
rm -f /root/logs/r762_scp_ready.done /root/logs/r762_chall_n80_launched.p3834
mkdir -p /tmp/r762_merged
df -h / | tail -1
nohup bash /root/mining_src/r762-chall/wait_r762_after_r761_then_n80_p3834.sh \
  >/root/logs/p3834_r762_wait.nohup 2>&1 &
echo $! >/root/logs/p3834_r762_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3834_r762_wait.pid)"
REMOTE

log "begin tar pipe brave→R252 ~66G R762"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r762_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r762_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r762_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r762_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r762_merged"
rc=$?
set -e
log "tar pipe rc=$rc"
[[ "$rc" -eq 0 ]] || { log "FATAL tar pipe failed"; exit 4; }
log "DONE relay — waiter queues R762 n80 after R761 decision"
