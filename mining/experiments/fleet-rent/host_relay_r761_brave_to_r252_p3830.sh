#!/usr/bin/env bash
# p3830: R761 MERGE_DONE on brave sat idle (no n80) → host-relay to R252;
# queue n80 on GPUs 4,5 after R769 n80 finishes. Never pkill -f. Leave R770 6,7.
set -euo pipefail
ROOT=/home/const/subnet120/mining
EXP=$ROOT/experiments/r761-r252-offline-dpo-hialpha-midrank-midbeta-softctx-megasuperextrasteps-ep4-lolr
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/host_relay_r761_brave_to_r252_p3830.log
: >"$LOG"
log() { echo "[p3830-r761] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

SSH_OPTS=(-o StrictHostKeyChecking=accept-new -o ConnectTimeout=45 -o BatchMode=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=40 -o TCPKeepAlive=yes)
BRAVE_HOST=18.118.83.97
BRAVE_PORT=40127
R252_HOST=95.133.252.28
R252_PORT=40299

log "START R761 host-relay brave→R252 (queue after R769 n80)"

n_src=$(ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" \
  "ls /tmp/r761_merged/model-*-of-*.safetensors 2>/dev/null | wc -l")
log "brave r761 shards=$n_src"
[[ "$n_src" -ge 16 ]] || { log "FATAL source incomplete"; exit 1; }

ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
  "mkdir -p /root/mining_src/r761-chall /root/logs /root/affine_data"
scp "${SSH_OPTS[@]}" -P "$R252_PORT" \
  "$EXP/lean_chall_n80_r252_gpus45_p3830.sh" \
  "$EXP/wait_r761_after_r769_then_n80_p3830.sh" \
  "root@$R252_HOST:/root/mining_src/r761-chall/"

ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" 'bash -s' <<'REMOTE'
set -euo pipefail
chmod +x /root/mining_src/r761-chall/*.sh
bash -n /root/mining_src/r761-chall/lean_chall_n80_r252_gpus45_p3830.sh
bash -n /root/mining_src/r761-chall/wait_r761_after_r769_then_n80_p3830.sh
echo SYNTAX_OK
# free old refute merges (keep r769_merged until its n80 done; keep r770 space)
for d in /tmp/r759_merged /tmp/r760_merged /tmp/r751_merged /tmp/r750_merged \
         /tmp/r743_merged /tmp/r742_merged /tmp/r735_merged /tmp/r734_merged; do
  [[ -d "$d" ]] && rm -rf "$d" && echo "freed $d"
done
rm -rf /tmp/r761_merged
rm -f /root/logs/r761_scp_ready.done /root/logs/r761_chall_n80_launched.p3830
mkdir -p /tmp/r761_merged
df -h / | tail -1
nohup bash /root/mining_src/r761-chall/wait_r761_after_r769_then_n80_p3830.sh \
  >/root/logs/p3830_r761_wait.nohup 2>&1 &
echo $! >/root/logs/p3830_r761_wait.pid
echo "waiter_started pid=$(cat /root/logs/p3830_r761_wait.pid)"
REMOTE

log "begin tar pipe brave→R252 ~66G R761"
set +e
ssh "${SSH_OPTS[@]}" -p "$BRAVE_PORT" "root@$BRAVE_HOST" "cd /tmp && tar cf - r761_merged" \
  | ssh "${SSH_OPTS[@]}" -p "$R252_PORT" "root@$R252_HOST" \
    "cd /tmp && tar xf - && n=\$(ls /tmp/r761_merged/model-*-of-*.safetensors 2>/dev/null | wc -l) && test -f /tmp/r761_merged/config.json && test \"\$n\" -ge 16 && date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r761_scp_ready.done && echo SCP_READY shards=\$n && du -sh /tmp/r761_merged"
rc=$?
set -e
log "tar pipe rc=$rc"
[[ "$rc" -eq 0 ]] || { log "FATAL tar pipe failed"; exit 4; }
log "DONE relay — waiter queues R761 n80 after R769 decision"
